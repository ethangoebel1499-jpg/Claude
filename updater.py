"""Keep this folder up to date with GitHub - and go back if an update doesn't work.
Standard library only, so it runs before any packages are installed.

  python updater.py            check and update now
  python updater.py --check    only say whether an update is available
  python updater.py --undo     go back to the version before the last update

How an update is applied:
  git clone   -> `git pull --ff-only` (refused if you edited tracked files yourself)
  ZIP         -> asks GitHub which files changed and downloads just those (the whole repo ZIP the
                 first time, when the installed version is unknown)
Before any file is overwritten or deleted, its old copy is saved in runtime/backup/, so the update
can be undone exactly. The update stays "pending" until the app confirms it works (packages
install, the self-check passes, the trader and dashboard start). If any of that fails, rollback()
restores the saved files and the version is remembered as bad so it isn't tried again - the next
newer version on GitHub will be.

Only files that are part of the repo are ever written or deleted. Your settings, trades and keys
(live.toml, local_settings.json, *.db, .venv, runtime/, logs) are never touched.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = "ethangoebel1499-jpg/Claude"
BRANCH = "master"
ROOT = Path(__file__).resolve().parent
# PAPERTRADER_GITHUB points every request at another server (used to test broken updates end to end)
_BASE = os.environ.get("PAPERTRADER_GITHUB")
API = f"{_BASE}/api" if _BASE else f"https://api.github.com/repos/{REPO}"
CODELOAD = f"{_BASE}/zip" if _BASE else f"https://codeload.github.com/{REPO}/zip"
RAW = f"{_BASE}/raw" if _BASE else f"https://raw.githubusercontent.com/{REPO}"
DEPS = ("app-requirements.txt", "requirements.txt")
NEVER = (".git/", ".venv/", "runtime/", "live.toml", "local_settings.json", ".version", ".manifest")


def paths():
    """Where state lives (functions so tests can point ROOT somewhere else)."""
    run = ROOT / "runtime"
    return {"version": ROOT / ".version", "manifest": ROOT / ".manifest", "backup": run / "backup",
            "state": run / "backup" / "state.json", "bad": run / "bad_versions.txt"}


def _get(url: str, accept: str = "application/vnd.github+json", timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "paper-trader-updater"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def is_git() -> bool:
    return (ROOT / ".git").exists() and shutil.which("git") is not None


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def current() -> str | None:
    if is_git():
        return _git("rev-parse", "HEAD").stdout.strip() or None
    v = paths()["version"]
    return v.read_text().strip() if v.exists() else None


def latest() -> str:
    return _get(f"{API}/commits/{BRANCH}", accept="application/vnd.github.sha").decode().strip()


def check() -> tuple[str | None, str]:
    """(installed commit, newest commit on GitHub)."""
    return current(), latest()


def bad_versions() -> set[str]:
    f = paths()["bad"]
    return set(f.read_text().split()) if f.exists() else set()


def _mark_bad(sha: str | None) -> None:
    if sha:
        f = paths()["bad"]
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("\n".join(sorted(bad_versions() | {sha})))


def _deps_hash() -> str:
    h = hashlib.sha256()
    for f in DEPS:
        p = ROOT / f
        h.update(p.read_bytes() if p.exists() else b"")
    return h.hexdigest()


def _safe(path: str) -> bool:
    p = path.replace("\\", "/")
    return not (p.startswith("/") or ".." in p.split("/") or any(p == n or p.startswith(n) for n in NEVER))


# ---- the backup that makes an update undoable ----------------------------------------
class Backup:
    def __init__(self, old: str | None, new: str, git: bool):
        p = paths()
        self.dir, self.files_dir = p["backup"], p["backup"] / "files"
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.files_dir.mkdir(parents=True)
        self.state = {"from": old, "to": new, "git": git, "files": {}, "confirmed": False,
                      "version": _read(p["version"]), "manifest": _read(p["manifest"]), "deps": _deps_hash()}

    def save(self, path: str) -> None:
        """Remember a file's current content (or that it didn't exist) before it changes."""
        if path in self.state["files"]:
            return
        src = ROOT / path
        if src.is_file():
            dst = self.files_dir / path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            self.state["files"][path] = "saved"
        else:
            self.state["files"][path] = "absent"

    def commit(self) -> None:
        paths()["state"].write_text(json.dumps(self.state, indent=1))


def _read(p: Path) -> str | None:
    return p.read_text() if p.exists() else None


def _restore_meta(p: Path, value: str | None) -> None:
    if value is None:
        if p.exists():
            p.unlink()
    else:
        p.write_text(value)


def state() -> dict | None:
    f = paths()["state"]
    return json.loads(f.read_text()) if f.exists() else None


def pending() -> dict | None:
    """The last update, if it hasn't been confirmed working yet."""
    s = state()
    return s if s and not s.get("confirmed") else None


def confirm() -> None:
    """The update works: keep it (the backup stays so 'Undo last update' can still go back)."""
    s = state()
    if s and not s.get("confirmed"):
        s["confirmed"] = True
        paths()["state"].write_text(json.dumps(s, indent=1))


def rollback(log=print, reason: str = "") -> dict:
    """Put back exactly what the last update changed. Returns {"ok", "to", "deps_changed"}."""
    s = state()
    if not s:
        return {"ok": False, "to": current(), "deps_changed": False, "message": "Nothing to go back to"}
    before = _deps_hash()
    if s["git"]:
        r = _git("reset", "--keep", s["from"])
        if r.returncode != 0:
            return {"ok": False, "to": current(), "deps_changed": False,
                    "message": f"git couldn't go back: {(r.stderr or r.stdout).strip()[:200]}"}
    else:
        files_dir = paths()["backup"] / "files"
        for path, how in s["files"].items():
            dest = ROOT / path
            if how == "saved":
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(files_dir / path, dest)
            elif dest.is_file():
                dest.unlink()
        _restore_meta(paths()["version"], s.get("version"))
        _restore_meta(paths()["manifest"], s.get("manifest"))
    _mark_bad(s["to"])
    shutil.rmtree(paths()["backup"], ignore_errors=True)
    msg = f"Went back to version {(s['from'] or 'before the update')[:7]}" + (f" ({reason})" if reason else "")
    log(msg)
    return {"ok": True, "to": s["from"], "deps_changed": _deps_hash() != before, "message": msg}


# ---- applying an update ---------------------------------------------------------------
def _write(path: str, data: bytes, bk: Backup) -> bool:
    dest = ROOT / path
    if dest.is_file() and dest.read_bytes() == data:
        return False                       # unchanged: nothing to write or back up
    bk.save(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".updating")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    return True


def _delete(path: str, bk: Backup) -> None:
    if (ROOT / path).is_file():
        bk.save(path)
        (ROOT / path).unlink()


def _manifest() -> set[str]:
    m = paths()["manifest"]
    return set(m.read_text().splitlines()) if m.exists() else set()


def _full_zip(sha: str, bk: Backup, log) -> list[str]:
    log("Downloading the whole app (first update)...")
    z = zipfile.ZipFile(io.BytesIO(_get(f"{CODELOAD}/{sha}", accept="*/*", timeout=300)))
    files = []
    for info in z.infolist():
        if info.is_dir():
            continue
        path = info.filename.split("/", 1)[1] if "/" in info.filename else ""
        if path and _safe(path):
            _write(path, z.read(info), bk)
            files.append(path)
    for gone in _manifest() - set(files):          # removed upstream since the last full update
        if _safe(gone):
            _delete(gone, bk)
    return files


def _changed_files(old: str, new: str, bk: Backup, log) -> list[str] | None:
    """Apply only what changed between two commits. None = too big for the compare API."""
    cmp = json.loads(_get(f"{API}/compare/{old}...{new}"))
    files = cmp.get("files", [])
    if len(files) >= 300 or cmp.get("status") not in ("ahead", "identical"):
        return None
    downloads = {}                                  # fetch everything first: a dropped connection changes nothing
    for f in files:
        if f["status"] != "removed" and _safe(f["filename"]):
            downloads[f["filename"]] = _get(f"{RAW}/{new}/{f['filename']}",
                                            accept="*/*", timeout=120)
    man = _manifest()
    for f in files:
        name, status = f["filename"], f["status"]
        if not _safe(name):
            continue
        if status == "renamed" and f.get("previous_filename") and _safe(f["previous_filename"]):
            _delete(f["previous_filename"], bk)
            man.discard(f["previous_filename"])
        if status == "removed":
            _delete(name, bk)
            man.discard(name)
        else:
            _write(name, downloads[name], bk)
            man.add(name)
    log(f"Updated {len(files)} file(s)")
    return sorted(man)


def update(log=print) -> dict:
    """Bring the folder up to the newest commit, keeping a backup to undo it. Returns
    {"updated", "from", "to", "deps_changed", "message"}. The update is pending until confirm()."""
    old, new = check()
    res = {"updated": False, "from": old, "to": new, "deps_changed": False}
    if old == new:
        return {**res, "message": "Up to date"}
    if new in bad_versions():
        return {**res, "message": f"Skipping {new[:7]}: it didn't work here before - waiting for a newer version"}
    p = paths()
    if is_git():
        if _git("status", "--porcelain", "--untracked-files=no").stdout.strip():
            return {**res, "message": "Update available, but files in this folder were edited - not updating over them"}
        bk = Backup(old, new, git=True)
        r = _git("pull", "--ff-only", "origin", BRANCH)
        if r.returncode != 0:
            shutil.rmtree(p["backup"], ignore_errors=True)
            return {**res, "message": f"git pull failed: {(r.stderr or r.stdout).strip()[:200]}"}
        bk.commit()
    else:
        bk = Backup(old, new, git=False)
        bk.commit()
        try:
            try:
                files = _changed_files(old, new, bk, log) if old else None
            except Exception as e:                  # e.g. the old commit no longer exists on GitHub
                log(f"Couldn't fetch just the changes ({e}); downloading everything")
                files = None
            if files is None:
                files = _full_zip(new, bk, log)
        except Exception as e:
            bk.commit()
            rollback(log, reason=f"download failed: {e}")
            paths()["bad"].write_text("\n".join(sorted(bad_versions() - {new})))   # a network error isn't the version's fault
            return {**res, "message": f"Update download failed ({e}) - nothing changed"}
        for name, value in ((".manifest", "\n".join(sorted(files))), (".version", new)):
            bk.save(name)
            (ROOT / name).write_text(value)
        bk.commit()
    return {**res, "updated": True, "deps_changed": _deps_hash() != bk.state["deps"], "message": f"Updated to {new[:7]}"}


def undo_last(log=print) -> dict:
    """Manual 'go back': undo the last update even after it was confirmed."""
    return rollback(log, reason="undone by you")


# ---- checks the app runs before keeping an update ---------------------------------------
def uv() -> str | None:
    exe = ROOT / "runtime" / ("uv.exe" if os.name == "nt" else "uv")
    return str(exe) if exe.exists() else None


def install_deps(python: str, log=print) -> bool:
    """Install the packages in app-requirements.txt (after an update or a rollback changed them)."""
    log("Installing packages...")
    if uv():
        cmd = [uv(), "pip", "install", "--python", python, "-r", str(ROOT / "app-requirements.txt")]
    else:
        cmd = [python, "-m", "pip", "install", "-q", "--only-binary=:all:", "-r", str(ROOT / "app-requirements.txt")]
    env = {**os.environ, "UV_PYTHON_INSTALL_DIR": str(ROOT / "runtime" / "python"),
           "UV_CACHE_DIR": str(ROOT / "runtime" / "cache")}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        log(f"Package install failed: {(r.stderr or r.stdout).strip()[-300:]}")
    return r.returncode == 0


def healthcheck(python: str, timeout: int = 300) -> tuple[bool, str]:
    """Run healthcheck.py with the app's Python: every file compiles, the modules import, and
    each paper-trading strategy runs through the live position logic."""
    try:
        r = subprocess.run([python, str(ROOT / "healthcheck.py")], cwd=ROOT, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "self-check took too long"
    out = (r.stdout + r.stderr).strip()
    return r.returncode == 0, out[-500:]


if __name__ == "__main__":
    if "--check" in sys.argv:
        old, new = check()
        print("Up to date" if old == new else f"Update available: {(old or 'unknown')[:7]} -> {new[:7]}")
    elif "--undo" in sys.argv:
        res = undo_last()
        print(res["message"])
        if res.get("deps_changed"):
            install_deps(sys.executable)
    else:
        res = update()
        print(res["message"])
        if res["updated"]:
            ok = install_deps(sys.executable) if res["deps_changed"] else True
            ok = ok and healthcheck(sys.executable)[0]
            (confirm if ok else lambda: rollback(reason="self-check failed"))()
