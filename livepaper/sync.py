"""Share the paper-trading record: export CSVs and push them to the `paper-trades`
branch, so a Claude session (or anyone with the repo) can analyze them.

Uses a separate git worktree (.paper-trades/), so your working branch is never touched.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .store import Store, now_utc

BRANCH = "paper-trades"


def git(*args, cwd=None, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True)


def push_files(files: dict[str, str], folder: str) -> str:
    """Write {filename: text} into <folder>/ on the paper-trades branch, commit and push."""
    repo = Path(git("rev-parse", "--show-toplevel").stdout.strip())
    wt = repo / ".paper-trades"
    if not wt.exists():
        has_remote = git("fetch", "origin", BRANCH, cwd=repo, check=False).returncode == 0
        start = f"origin/{BRANCH}" if has_remote else "HEAD"
        git("worktree", "add", "-B", BRANCH, str(wt), start, cwd=repo)
    else:
        git("pull", "--rebase", "origin", BRANCH, cwd=wt, check=False)
    dest = wt / folder
    dest.mkdir(exist_ok=True)
    for name, text in files.items():
        (dest / name).write_text(text)
    git("add", folder, cwd=wt)
    if git("diff", "--cached", "--quiet", cwd=wt, check=False).returncode == 0:
        return "nothing new"
    git("commit", "-m", f"{folder} {now_utc()}", cwd=wt)
    push = git("push", "-u", "origin", BRANCH, cwd=wt, check=False)
    msg = "pushed" if push.returncode == 0 else f"commit saved, push failed: {push.stderr.strip()[:200]}"
    print(f"[sync] {msg}")
    return msg


def sync(store: Store) -> str:
    files = {f"{name}.csv": store.table(name).to_csv(index=False)
             for name in ("trades", "fills", "equity", "signals", "events")}
    return push_files(files, "paper_trades")
