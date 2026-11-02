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


def test_failed_update_is_rolled_back_exactly_and_not_retried(app, monkeypatch):
    (app / ".version").write_text("a" * 40)
    (app / "keep.py").write_text("old keep")
    (app / "gone.py").write_text("old gone")
    compare = {"status": "ahead", "files": [
        {"filename": "keep.py", "status": "modified"}, {"filename": "gone.py", "status": "removed"},
        {"filename": "added.py", "status": "added"}]}
    fake_github(monkeypatch, "b" * 40, compare=compare, raw={"keep.py": "new keep", "added.py": "new"})
    assert updater.update(log=lambda *a: None)["updated"]
    assert updater.pending() and (app / "keep.py").read_text() == "new keep"
    res = updater.rollback(log=lambda *a: None, reason="self-check failed")
    assert res["ok"] and res["to"] == "a" * 40
    assert (app / "keep.py").read_text() == "old keep" and (app / "gone.py").read_text() == "old gone"
    assert not (app / "added.py").exists() and (app / ".version").read_text() == "a" * 40
    assert "mine" in (app / "local_settings.json").read_text()
    again = updater.update(log=lambda *a: None)                       # same broken version: skipped
    assert not again["updated"] and "didn't work" in again["message"]
    fake_github(monkeypatch, "c" * 40, compare=compare, raw={"keep.py": "fixed", "added.py": "new"})
    assert updater.update(log=lambda *a: None)["updated"]              # a newer one is tried
    updater.confirm()
    assert updater.pending() is None and updater.state()["confirmed"]
    assert updater.undo_last(log=lambda *a: None)["ok"] and (app / "keep.py").read_text() == "old keep"


def test_download_error_changes_nothing(app, monkeypatch):
    (app / ".version").write_text("a" * 40)
    (app / "keep.py").write_text("old")

    def get(url, accept="", timeout=60):
        if url.endswith("/commits/master"):
            return b"b" * 40
        raise OSError("connection dropped")
    monkeypatch.setattr(updater, "_get", get)
    res = updater.update(log=lambda *a: None)
    assert not res["updated"] and "nothing changed" in res["message"]
    assert (app / "keep.py").read_text() == "old" and (app / ".version").read_text() == "a" * 40
    assert "b" * 40 not in updater.bad_versions()                     # retried next time


def test_bootstrap_rolls_back_a_crashing_update(app, monkeypatch):
    import importlib.machinery
    import importlib.util
    path = str(updater.Path(__file__).parent.parent / "bootstrap.pyw")
    loader = importlib.machinery.SourceFileLoader("bootstrap", path)
    bootstrap = importlib.util.module_from_spec(importlib.util.spec_from_loader("bootstrap", loader))
    loader.exec_module(bootstrap)
    monkeypatch.setattr(bootstrap, "LOG", app / "runtime" / "launcher.log")
    monkeypatch.setattr(bootstrap, "tell", lambda msg: None)
    (app / ".version").write_text("a" * 40)
    (app / "launcher.pyw").write_text("good")
    fake_github(monkeypatch, "b" * 40, compare={"status": "ahead", "files": [
        {"filename": "launcher.pyw", "status": "modified"}]}, raw={"launcher.pyw": "broken"})
    updater.update(log=lambda *a: None)
    codes = iter([1, 0])                    # new version crashes, then the old one runs and is closed
    seen = []
    def launch():
        seen.append((app / "launcher.pyw").read_text())
        return next(codes)
    assert bootstrap.run(launch=launch) == 0
    assert seen == ["broken", "good"] and "b" * 40 in updater.bad_versions()
    assert bootstrap.run(launch=lambda: 3 if not seen.append(1) and len(seen) < 4 else 0) == 0   # restart requests
