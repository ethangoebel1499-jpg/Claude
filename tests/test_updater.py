import io
import json
import subprocess
import zipfile

import pytest

import updater


@pytest.fixture
def app(tmp_path, monkeypatch):
    """A ZIP-style install (no .git) in a temp folder."""
    monkeypatch.setattr(updater, "ROOT", tmp_path)
    monkeypatch.setattr(updater, "VERSION", tmp_path / ".version")
    monkeypatch.setattr(updater, "MANIFEST", tmp_path / ".manifest")
    (tmp_path / "local_settings.json").write_text('{"webull_app_key": "mine"}')
    (tmp_path / "live.toml").write_text("my runs")
    return tmp_path


def fake_github(monkeypatch, sha, zip_files=None, compare=None, raw=None):
    calls = []

    def get(url, accept="", timeout=60):
        calls.append(url)
        if url.endswith("/commits/master"):
            return sha.encode()
        if "codeload.github.com" in url:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                for name, data in zip_files.items():
                    z.writestr(f"Claude-{sha}/{name}", data)
            return buf.getvalue()
        if "/compare/" in url:
            return json.dumps(compare).encode()
        if "raw.githubusercontent.com" in url:
            return raw[url.split(f"/{sha}/", 1)[1]].encode()
        raise AssertionError(url)

    monkeypatch.setattr(updater, "_get", get)
    return calls


def test_first_update_downloads_everything_and_keeps_user_files(app, monkeypatch):
    fake_github(monkeypatch, "a" * 40, zip_files={"launcher.pyw": "v1", "mcbot/s9.py": "s9", "live.toml": "theirs",
                                                  "requirements.txt": "pandas"})
    res = updater.update(log=lambda *a: None)
    assert res["updated"] and res["to"] == "a" * 40
    assert (app / "mcbot/s9.py").read_text() == "s9" and (app / ".version").read_text() == "a" * 40
    assert (app / "live.toml").read_text() == "my runs"                       # never overwritten
    assert json.loads((app / "local_settings.json").read_text())["webull_app_key"] == "mine"
    assert updater.update(log=lambda *a: None)["message"] == "Up to date"


def test_later_updates_fetch_only_what_changed(app, monkeypatch):
    (app / ".version").write_text("a" * 40)
    (app / "old.py").write_text("gone soon")
    (app / "renamed_from.py").write_text("x")
    (app / "requirements.txt").write_text("pandas")
    compare = {"status": "ahead", "files": [
        {"filename": "mcbot/new.py", "status": "added"},
        {"filename": "requirements.txt", "status": "modified"},
        {"filename": "old.py", "status": "removed"},
        {"filename": "renamed_to.py", "status": "renamed", "previous_filename": "renamed_from.py"},
        {"filename": "../outside.txt", "status": "added"},
        {"filename": "local_settings.json", "status": "modified"},
    ]}
    raw = {"mcbot/new.py": "new", "requirements.txt": "pandas\nnumpy", "renamed_to.py": "x"}
    calls = fake_github(monkeypatch, "b" * 40, compare=compare, raw=raw)
    res = updater.update(log=lambda *a: None)
    assert res["updated"] and res["deps_changed"]
    assert (app / "mcbot/new.py").read_text() == "new" and not (app / "old.py").exists()
    assert (app / "renamed_to.py").exists() and not (app / "renamed_from.py").exists()
    assert not (app.parent / "outside.txt").exists()
    assert "mine" in (app / "local_settings.json").read_text()
    assert not any("codeload" in c for c in calls)                            # no full download
    assert (app / ".version").read_text() == "b" * 40


def test_falls_back_to_full_download_when_compare_fails(app, monkeypatch):
    (app / ".version").write_text("deadbeef" * 5)
    calls = fake_github(monkeypatch, "c" * 40, zip_files={"a.py": "1"}, compare={"status": "diverged", "files": []})
    assert updater.update(log=lambda *a: None)["updated"]
    assert any("codeload" in c for c in calls) and (app / "a.py").read_text() == "1"


def test_git_install_pulls(tmp_path, monkeypatch):
    def git(cwd, *a):
        return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *a], cwd=cwd, check=True,
                              capture_output=True, text=True).stdout.strip()
    origin, clone = tmp_path / "origin", tmp_path / "clone"
    origin.mkdir()
    git(origin, "init", "-q", "-b", "master")
    (origin / "a.py").write_text("1")
    git(origin, "add", "."), git(origin, "commit", "-qm", "one")
    git(tmp_path, "clone", "-q", str(origin), str(clone))
    (origin / "a.py").write_text("2")
    git(origin, "commit", "-qam", "two")
    new = git(origin, "rev-parse", "HEAD")
    monkeypatch.setattr(updater, "ROOT", clone)
    monkeypatch.setattr(updater, "latest", lambda: new)
    (clone / "live.toml").write_text("mine")                                   # untracked user file
    res = updater.update(log=lambda *a: None)
    assert res["updated"] and (clone / "a.py").read_text() == "2" and (clone / "live.toml").read_text() == "mine"
    # an edited tracked file blocks the update instead of being overwritten
    (origin / "a.py").write_text("3")
    git(origin, "commit", "-qam", "three")
    monkeypatch.setattr(updater, "latest", lambda: git(origin, "rev-parse", "HEAD"))
    (clone / "a.py").write_text("my edit")
    res = updater.update(log=lambda *a: None)
    assert not res["updated"] and "edited" in res["message"] and (clone / "a.py").read_text() == "my edit"
