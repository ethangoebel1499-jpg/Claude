"""For Claude sessions: read what the app shared and publish a review the app shows in 🧠 Apprentice.

  python tools/claude_review.py show                 fetch the paper-trades branch, print a review-ready digest
  python tools/claude_review.py check review.json    validate a review file (ids exist, verdicts, fix rules)
  python tools/claude_review.py publish review.json  validate, then commit it to the paper-trades branch and push

The app downloads reviews/claude_review.json on its next sync (hourly, or "Sync with Claude now").
Format: see apprentice/review.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
BRANCH = "paper-trades"
VERDICTS = {"setups": {"use", "watch", "drop", "fix"}, "concepts": {"keep", "drop", "fix"}, "lessons": {"keep", "drop"}}


def git(*a, check=True) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, check=check, capture_output=True, text=True).stdout


def shared(path: str) -> str | None:
    r = subprocess.run(["git", "show", f"origin/{BRANCH}:{path}"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def show() -> None:
    git("fetch", "-q", "origin", BRANCH, check=False)
    summary = shared("SUMMARY.md")
    if summary is None:
        print("Nothing shared yet: the app needs 'Share with Claude' on (and a GitHub token on a ZIP install).")
        return
    print(summary)
    setups = json.loads(shared("apprentice/setups.json") or "[]")
    print(f"## Setups ({len(setups)})")
    for s in setups:
        st = s.get("stats") or {}
        print(f"- id {s['id']} · {s['direction'].upper()} · {s['name']} · {s['status']} · weight {s.get('weight', 0):+.2f}"
              f" · stop {s.get('stop_atr')} ATR, {s.get('rr')}R\n    rules: " + " AND ".join(s.get("rules_text", []))
              + (f"\n    unchecked: {s['unchecked']}" if s.get("unchecked") else "")
              + f"\n    {st.get('trades', 0)} trades, win {st.get('win_rate', 0) or 0:.0f}%, "
                f"{st.get('avg_r', 0) or 0:+.2f}R/trade, edge {st.get('edge', 0) or 0:+.2f}R"
              + (f"\n    after entry: {s['after_entry']}" if s.get("after_entry") else "")
              + "".join(f"\n    hint: {h}" for h in s.get("hints") or [])
              + (f"\n    tuning: {s['tune']['change']} (edge {s['tune']['test']['was']} -> {s['tune']['test']['edge']} "
                 f"on unseen data)" if s.get("tune") else "")
              + "".join(f"\n    exit plan {tf}: learned stop {p.get('learned_stop', p.get('stop'))}, reversal exit "
                        f"{'on' if p.get('learned_reversal', p.get('reversal')) else 'off'}, "
                        f"{'deployed' if p.get('deployed') else 'not deployed'} · {p.get('stops')}"
                        for tf, p in (s.get("exit_plan") or {}).items() if isinstance(p, dict))
              + (f"\n    why: {str(s.get('why'))[:200]}" if s.get("why") else ""))
    import io
    import pandas as pd
    con = shared("apprentice/concepts.csv")
    if con:
        df = pd.read_csv(io.StringIO(con))
        print(f"\n## Concepts ({len(df)})")
        for r in df.itertuples():
            print(f"- id {r.id} · lesson {r.lesson_id} · {r.category}: {r.text}")
    les = shared("apprentice/lessons.csv")
    if les:
        df = pd.read_csv(io.StringIO(les))
        print(f"\n## Lessons ({len(df)})")
        for r in df.itertuples():
            print(f"- id {r.id} · {r.title} · {r.status}")
    old = shared("reviews/claude_review.json")
    if old:
        print(f"\n(previous review: {json.loads(old).get('reviewed_at')})")


def check(path: str) -> dict:
    from mcbot import builder
    rv = json.loads(Path(path).read_text())
    errors = []
    setups = {s["id"]: s for s in json.loads(shared("apprentice/setups.json") or "[]")}
    for kind, ok in VERDICTS.items():
        for x in rv.get(kind, []):
            if x.get("verdict") not in ok:
                errors.append(f"{kind} id {x.get('id')}: verdict must be one of {sorted(ok)}")
            if not x.get("reason"):
                errors.append(f"{kind} id {x.get('id')}: give a reason")
    for x in rv.get("trades", []):
        tag = f"trade {x.get('id')} ({x.get('chart')} {x.get('time')})"
        if x.get("kind") not in ("decision", "missed", "signal"):
            errors.append(f"{tag}: kind must be decision, missed or signal")
        if not isinstance(x.get("stars"), int) or not 1 <= x["stars"] <= 5:
            errors.append(f"{tag}: stars must be 1-5")
        if x.get("kind") == "missed" and x.get("verdict") not in ("good", "bad"):
            errors.append(f"{tag}: a missed trade needs verdict good or bad")
        if not x.get("reason") or not x.get("chart") or not x.get("time"):
            errors.append(f"{tag}: needs chart, time and a reason")
    for x in rv.get("setups", []):
        if setups and x.get("id") not in setups:
            errors.append(f"setup id {x.get('id')} isn't in the shared setups")
        elif setups and x.get("name") and setups[x["id"]]["name"] != x["name"]:
            errors.append(f"setup id {x['id']}: name doesn't match ({setups[x['id']]['name']!r})")
        for r in (x.get("fix") or {}).get("rules", []):
            if r.get("rule") not in builder.RULES:
                errors.append(f"setup id {x.get('id')}: unknown rule {r.get('rule')!r}")
    if errors:
        raise SystemExit("Review has problems:\n- " + "\n- ".join(errors))
    rv.setdefault("reviewed_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    print(f"OK: {len(rv.get('setups', []))} setups, {len(rv.get('concepts', []))} concepts, "
          f"{len(rv.get('lessons', []))} lessons, {len(rv.get('trades', []))} chart trades")
    return rv


def publish(path: str) -> None:
    rv = check(path)
    rv["reviewed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    wt = ROOT / ".review-worktree"
    git("fetch", "-q", "origin", BRANCH)
    if not wt.exists():
        git("worktree", "add", "-f", str(wt), f"origin/{BRANCH}")
    run = lambda *a: subprocess.run(["git", *a], cwd=wt, check=True, capture_output=True, text=True)
    run("checkout", "-q", "--detach", f"origin/{BRANCH}")
    (wt / "reviews").mkdir(exist_ok=True)
    (wt / "reviews" / "claude_review.json").write_text(json.dumps(rv, indent=1))
    run("add", "reviews/claude_review.json")
    run("commit", "-q", "-m", f"Claude's review {rv['reviewed_at']}")
    run("push", "origin", f"HEAD:{BRANCH}")
    print("Published - the app picks it up on its next sync.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    {"show": lambda: show(), "check": lambda: check(sys.argv[2]), "publish": lambda: publish(sys.argv[2])}[cmd]()
