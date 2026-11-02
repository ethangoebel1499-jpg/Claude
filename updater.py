"""Keep this folder up to date with GitHub (standard library only, so it works before packages install).

  python updater.py            check and update now
  python updater.py --check    only say whether an update is available

Two kinds of install:
  git clone   -> `git pull --ff-only` (refused if you edited tracked files yourself)
  ZIP         -> asks GitHub which files changed since the installed version and downloads just those
                 (the whole repo ZIP the first time, when the installed version is unknown)

Only files that are part of the repo are ever written or deleted. Your settings, trades and keys
(live.toml, local_settings.json, *.db, .venv, logs, frames) are never touched.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = "ethangoebel1499-jpg/Claude"
BRANCH = "master"
ROOT = Path(__file__).resolve().parent
VERSION = ROOT / ".version"          # commit the ZIP install is at
MANIFEST = ROOT / ".manifest"        # files the last ZIP update wrote (so removed ones can be deleted)
DEPS = ("app-requirements.txt", "requirements.txt")
NEVER = (".git/", ".venv/", "live.toml", "local_settings.json", ".version", ".manifest")
API = f"https://api.github.com/repos/{REPO}"


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
        r = _git("rev-parse", "HEAD")
        return r.stdout.strip() or None
    return VERSION.read_text().strip() if VERSION.exists() else None


def latest() -> str:
    return _get(f"{API}/commits/{BRANCH}", accept="application/vnd.github.sha").decode().strip()


def check() -> tuple[str | None, str]:
    """(installed commit, newest commit on GitHub)."""
    return current(), latest()


def _deps_hash() -> str:
    h = hashlib.sha256()
    for f in DEPS:
        p = ROOT / f
        h.update(p.read_bytes() if p.exists() else b"")
    return h.hexdigest()


def _safe(path: str) -> bool:
    p = path.replace("\\", "/")
    return not (p.startswith("/") or ".." in p.split("/") or any(p == n or p.startswith(n) for n in NEVER))


def _write(path: str, data: bytes) -> None:
    dest = ROOT / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".updating")
    tmp.write_bytes(data)
    tmp.replace(dest)


def _delete(path: str) -> None:
    p = ROOT / path
    if p.is_file():
        p.unlink()


def _manifest() -> set[str]:
    return set(MANIFEST.read_text().splitlines()) if MANIFEST.exists() else set()


def _full_zip(sha: str, log) -> list[str]:
    log("Downloading the whole app (first update)...")
    z = zipfile.ZipFile(io.BytesIO(_get(f"https://codeload.github.com/{REPO}/zip/{sha}", accept="*/*", timeout=300)))
    files = []
    for info in z.infolist():
        if info.is_dir():
            continue
        path = info.filename.split("/", 1)[1] if "/" in info.filename else ""
        if path and _safe(path):
            _write(path, z.read(info))
            files.append(path)
    for gone in _manifest() - set(files):          # removed upstream since the last full update
        if _safe(gone):
            _delete(gone)
    return files


def _changed_files(old: str, new: str, log) -> list[str] | None:
    """Apply only what changed between two commits. None = too big for the compare API."""
    cmp = json.loads(_get(f"{API}/compare/{old}...{new}"))
    files = cmp.get("files", [])
    if len(files) >= 300 or cmp.get("status") not in ("ahead", "identical"):
        return None
    man = _manifest()
    for f in files:
        name, status = f["filename"], f["status"]
        if not _safe(name):
            continue
        if status == "removed":
            _delete(name)
            man.discard(name)
            continue
        if status == "renamed" and f.get("previous_filename") and _safe(f["previous_filename"]):
            _delete(f["previous_filename"])
            man.discard(f["previous_filename"])
        _write(name, _get(f"https://raw.githubusercontent.com/{REPO}/{new}/{name}", accept="*/*", timeout=120))
        man.add(name)
    log(f"Updated {len(files)} file(s)")
    return sorted(man)


def update(log=print) -> dict:
    """Bring the folder up to the newest commit. Returns
    {"updated": bool, "from": sha, "to": sha, "deps_changed": bool, "message": str}."""
    old, new = check()
    if old == new:
        return {"updated": False, "from": old, "to": new, "deps_changed": False, "message": "Up to date"}
    before = _deps_hash()
    if is_git():
        dirty = _git("status", "--porcelain", "--untracked-files=no").stdout.strip()
        if dirty:
            return {"updated": False, "from": old, "to": new, "deps_changed": False,
                    "message": "Update available, but files in this folder were edited - not updating over them"}
        r = _git("pull", "--ff-only", "origin", BRANCH)
        if r.returncode != 0:
            return {"updated": False, "from": old, "to": new, "deps_changed": False,
                    "message": f"git pull failed: {(r.stderr or r.stdout).strip()[:200]}"}
    else:
        try:
            files = _changed_files(old, new, log) if old else None
        except Exception as e:                       # e.g. the old commit no longer exists on GitHub
            log(f"Couldn't fetch just the changes ({e}); downloading everything")
            files = None
        if files is None:
            files = _full_zip(new, log)
        MANIFEST.write_text("\n".join(sorted(files)))
        VERSION.write_text(new)
    return {"updated": True, "from": old, "to": new, "deps_changed": _deps_hash() != before,
            "message": f"Updated to {new[:7]}"}


def install_deps(python: str, log=print) -> bool:
    """Reinstall packages after an update changed the package list."""
    log("Installing updated packages...")
    r = subprocess.run([python, "-m", "pip", "install", "-q", "--only-binary=:all:", "-r",
                        str(ROOT / "app-requirements.txt")], capture_output=True, text=True)
    if r.returncode != 0:
        log(f"Package install failed: {(r.stderr or r.stdout).strip()[-300:]}")
    return r.returncode == 0


if __name__ == "__main__":
    if "--check" in sys.argv:
        old, new = check()
        print("Up to date" if old == new else f"Update available: {(old or 'unknown')[:7]} -> {new[:7]}")
    else:
        res = update()
        print(res["message"])
        if res["deps_changed"]:
            install_deps(sys.executable)
