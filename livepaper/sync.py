"""Share what the app knows with Claude, and bring Claude's reviews back.

Every hour (and on "Sync now" in ⚙️ Account) the trader exports
  paper_trades/  trades, fills, equity, signals, events, runs
  apprentice/    lessons, concepts, setups (rules in words + track record), journal, your ratings / votes
  streams/       the stream agent's setups, rules, patterns and your reviews
  SUMMARY.md     counts and the latest numbers at a glance
to the `paper-trades` branch on GitHub, then downloads reviews/claude_review.json from it: Claude's verdict
on every setup, concept and lesson (use / drop / fix, with the reason and a fix), shown in 🧠 Apprentice.

Upload: with a GitHub token (⚙️ Account; works for ZIP installs) through GitHub's API as one commit,
otherwise with git from a git clone (a separate worktree, .paper-trades/, so your branch isn't touched).
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from pathlib import Path

import requests

from .store import Store, now_utc

BRANCH = "paper-trades"
REPO = "ethangoebel1499-jpg/Claude"
ROOT = Path(__file__).resolve().parent.parent
REVIEW_PATH = "reviews/claude_review.json"
LOCAL_REVIEW = ROOT / "runtime" / "claude_review.json"
STATUS = ROOT / "runtime" / "sync_status.json"
API = "https://api.github.com"


# ---- what gets shared ---------------------------------------------------------------------------
def _csv(df, last: int | None = None) -> str:
    return (df.tail(last) if last else df).to_csv(index=False)


def export_all(lp: Store | None) -> dict[str, str]:
    """{path on the branch: text} for everything worth reviewing."""
    files: dict[str, str] = {}
    lines = [f"# Paper Trader snapshot · {now_utc()}", ""]
    if lp is not None:
        for name, last in (("trades", None), ("fills", 3000), ("equity", 3000), ("signals", 500), ("events", 1000),
                           ("runs", None)):
            try:
                df = lp.table(name)
                files[f"paper_trades/{name}.csv"] = _csv(df, last)
                if name == "trades":
                    win = (df.pnl_usd > 0).mean() * 100 if len(df) else 0
                    lines.append(f"- Paper trades closed: {len(df)} · win rate {win:.0f}% · P&L "
                                 f"{df.pnl_usd.sum():,.2f} USD" if len(df) else "- Paper trades closed: 0")
            except Exception:
                pass
    try:
        from apprentice.store import Store as AStore
        from apprentice.strategy import db_path
        from mcbot import builder
        a = AStore(db_path())
        for name in ("lessons", "concepts", "feedback", "ratings"):
            files[f"apprentice/{name}.csv"] = _csv(a.table(name))
        files["apprentice/journal.csv"] = _csv(a.table("journal"), 1000)
        setups = []
        for s in a.setups():
            setups.append({**{k: s.get(k) for k in ("id", "name", "direction", "status", "weight", "source", "stop_atr",
                                                    "rr", "timeframe", "why", "quote", "unchecked", "lesson_id")},
                           "rules": s.get("rules", []),
                           "rules_text": [builder.describe(r) for r in s.get("rules", []) if r.get("rule") in builder.RULES],
                           "stats": s.get("stats", {}).get("total", {}), "charts": s.get("stats", {}).get("charts", {})})
        files["apprentice/setups.json"] = json.dumps(setups, indent=1, default=str)
        lines.append(f"- Apprentice: {len(a.table('lessons'))} lessons, {len(a.table('concepts'))} concepts, "
                     f"{len(setups)} setups ({sum(s['status'] in ('active', 'pinned') for s in setups)} trading)")
    except Exception as e:
        lines.append(f"- Apprentice: not available ({type(e).__name__})")
    try:
        from livepaper import settings
        from streamlearn.store import Store as SStore
        db = settings.get("streamlearn_db") or "streamlearn.db"
        if Path(db).exists():
            s = SStore(db)
            for name in ("calls", "rules", "patterns", "reviews", "streams"):
                try:
                    files[f"streams/{name}.csv"] = _csv(s.table(name))
                except Exception:
                    pass
            lines.append(f"- Streams: {len(s.table('streams'))} streams, {len(s.table('calls'))} setups")
    except Exception:
        pass
    lines += ["", "Claude writes its review to reviews/claude_review.json on this branch; the app shows it in "
              "🧠 Apprentice → What it knows."]
    files["SUMMARY.md"] = "\n".join(lines) + "\n"
    return files


# ---- uploading ----------------------------------------------------------------------------------
def _blob_sha(text: str) -> str:
    data = text.encode()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class GitHub:
    def __init__(self, token: str, repo: str = REPO):
        self.repo, self.h = repo, {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                                   "User-Agent": "paper-trader"}

    def _req(self, method: str, path: str, **kw):
        r = requests.request(method, f"{API}/repos/{self.repo}{path}", headers=self.h, timeout=60, **kw)
        if r.status_code >= 400 and not (method == "GET" and r.status_code == 404):
            raise RuntimeError(f"GitHub {method} {path}: HTTP {r.status_code} {r.text[:200]}")
        return r

    def branch_head(self) -> str:
        r = self._req("GET", f"/git/ref/heads/{BRANCH}")
        if r.status_code == 404:                      # first sync: start the branch from master
            base = self._req("GET", "/git/ref/heads/master").json()["object"]["sha"]
            self._req("POST", "/git/refs", json={"ref": f"refs/heads/{BRANCH}", "sha": base})
            return base
        return r.json()["object"]["sha"]

    def push(self, files: dict[str, str], message: str) -> str:
        head = self.branch_head()
        tree = self._req("GET", f"/git/commits/{head}").json()["tree"]["sha"]
        have = {e["path"]: e["sha"] for e in self._req("GET", f"/git/trees/{tree}?recursive=1").json().get("tree", [])}
        changed = {p: t for p, t in files.items() if have.get(p) != _blob_sha(t)}
        if not changed:
            return "nothing new"
        entries = []
        for path, text in changed.items():
            blob = self._req("POST", "/git/blobs", json={"content": base64.b64encode(text.encode()).decode(),
                                                         "encoding": "base64"}).json()["sha"]
            entries.append({"path": path, "mode": "100644", "type": "blob", "sha": blob})
        new_tree = self._req("POST", "/git/trees", json={"base_tree": tree, "tree": entries}).json()["sha"]
        commit = self._req("POST", "/git/commits", json={"message": message, "tree": new_tree,
                                                         "parents": [head]}).json()["sha"]
        self._req("PATCH", f"/git/refs/heads/{BRANCH}", json={"sha": commit})
        return f"pushed {len(changed)} file(s)"

    def read(self, path: str) -> str | None:
        r = self._req("GET", f"/contents/{path}?ref={BRANCH}")
        if r.status_code == 404:
            return None
        return base64.b64decode(r.json()["content"]).decode()


def git(*args, cwd=None, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True)


def _git_push(files: dict[str, str], message: str) -> str:
    repo = Path(git("rev-parse", "--show-toplevel").stdout.strip())       # the clone we're running in
    wt = repo / ".paper-trades"
    if not wt.exists():
        has_remote = git("fetch", "origin", BRANCH, cwd=repo, check=False).returncode == 0
        git("worktree", "add", "-B", BRANCH, str(wt), f"origin/{BRANCH}" if has_remote else "HEAD", cwd=repo)
    else:
        git("pull", "--rebase", "origin", BRANCH, cwd=wt, check=False)
    for path, text in files.items():
        (wt / path).parent.mkdir(parents=True, exist_ok=True)
        (wt / path).write_text(text)
    git("add", "-A", ".", cwd=wt)
    if git("diff", "--cached", "--quiet", cwd=wt, check=False).returncode == 0:
        return "nothing new"
    git("commit", "-m", message, cwd=wt)
    push = git("push", "-u", "origin", BRANCH, cwd=wt, check=False)
    return "pushed" if push.returncode == 0 else f"commit saved, push failed: {push.stderr.strip()[:200]}"


def push_files(files: dict[str, str], folder: str = "") -> str:
    """Upload {path: text} (under folder/) to the paper-trades branch."""
    from . import settings
    files = {(f"{folder}/{p}" if folder else p): t for p, t in files.items()}
    token = settings.get("github_token")
    msg = f"{folder or 'snapshot'} {now_utc()}"
    if token:
        return GitHub(token).push(files, msg)
    if git("rev-parse", "--show-toplevel", check=False).returncode == 0:
        return _git_push(files, msg)
    raise RuntimeError("add a GitHub token in ⚙️ Account to share (this copy isn't a git clone)")


def fetch_reviews() -> dict | None:
    """Download Claude's latest review (the repo is public; the token is used when set)."""
    from . import settings
    text = None
    token = settings.get("github_token")
    try:
        if token:
            text = GitHub(token).read(REVIEW_PATH)
        else:
            r = requests.get(f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{REVIEW_PATH}", timeout=30)
            text = r.text if r.status_code == 200 else None
    except Exception:
        text = None
    if not text:
        return None
    review = json.loads(text)
    LOCAL_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_REVIEW.write_text(json.dumps(review, indent=1))
    return review


def sync_all(lp: Store | None) -> str:
    """Share everything, then bring back Claude's review. Records how it went for the Account tab."""
    try:
        msg = push_files(export_all(lp))
    except Exception as e:
        msg = f"failed: {e}"
    review = fetch_reviews()
    status = {"at": now_utc(), "result": msg, "review": (review or {}).get("reviewed_at")}
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status))
    print(f"[sync] {msg}" + (f"; Claude's review from {status['review']}" if status["review"] else ""))
    return msg


def last_status() -> dict:
    try:
        return json.loads(STATUS.read_text())
    except Exception:
        return {}


def sync(store: Store) -> str:                 # kept for older callers
    return sync_all(store)
