"""For Claude sessions: send a paper trade to the user's open app through GitHub.

  python tools/claude_trade.py send --action long|short|close --id <unique id> --reason "..." [--size-pct 52] [--run AUTOGRIND]
  python tools/claude_trade.py list

Appends the trade to commands/claude.json on the paper-trades branch (the last 100 are kept). The app checks that file
every ~2 minutes while it's open and paper trades it (livepaper/claude_trades.py). A command id is applied only once,
so re-sending the same id is harmless.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRANCH, PATH = "paper-trades", "commands/claude.json"


def git(*a, cwd=ROOT, check=True):
    return subprocess.run(["git", *a], cwd=cwd, check=check, capture_output=True, text=True)


def current() -> dict:
    git("fetch", "-q", "origin", BRANCH, check=False)
    r = git("show", f"origin/{BRANCH}:{PATH}", check=False)
    try:
        return json.loads(r.stdout) if r.returncode == 0 else {"commands": []}
    except Exception:
        return {"commands": []}


def send(action: str, cid: str, reason: str, run: str, symbol: str, timeframe: str,
         size_pct: float | None = None, price: float | None = None) -> str:
    data = current()
    if any(c.get("id") == cid for c in data["commands"]):
        return f"already sent ({cid})"
    cmd = {"id": cid, "run": run, "symbol": symbol, "timeframe": timeframe, "action": action, "reason": reason,
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if size_pct is not None:
        cmd["size_pct"] = size_pct
    if price is not None:
        cmd["price"] = price
    data["commands"] = (data["commands"] + [cmd])[-100:]
    wt = ROOT / ".review-worktree"
    if not wt.exists():
        git("worktree", "add", "-f", str(wt), f"origin/{BRANCH}")
    git("checkout", "-q", "--detach", f"origin/{BRANCH}", cwd=wt)
    (wt / "commands").mkdir(exist_ok=True)
    (wt / PATH).write_text(json.dumps(data, indent=1))
    git("add", PATH, cwd=wt)
    git("commit", "-q", "-m", f"Claude trade: {action} {run} ({cid})", cwd=wt)
    for attempt in range(4):
        if git("push", "origin", f"HEAD:{BRANCH}", cwd=wt, check=False).returncode == 0:
            return f"sent: {action} {run} ({cid})"
        git("fetch", "-q", "origin", BRANCH, cwd=wt)            # the app pushed a snapshot meanwhile: redo on top
        git("rebase", "-q", f"origin/{BRANCH}", cwd=wt, check=False)
    raise SystemExit("couldn't push the trade to GitHub")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("send")
    s.add_argument("--action", required=True, choices=["long", "short", "close"])
    s.add_argument("--id", required=True)
    s.add_argument("--reason", default="")
    s.add_argument("--run", default="AUTOGRIND")
    s.add_argument("--size-pct", type=float, default=None, help="percent of the account for this trade (0-100)")
    s.add_argument("--price", type=float, default=None, help="TradingView's entry/exit price to fill at")
    s.add_argument("--symbol", default="XBTUSD")
    s.add_argument("--timeframe", default="30m")
    sub.add_parser("list")
    a = ap.parse_args()
    if a.cmd == "send":
        print(send(a.action, a.id, a.reason, a.run, a.symbol, a.timeframe, a.size_pct, a.price))
    else:
        for c in current()["commands"][-20:]:
            print(f"{c['at'][:16]}  {c['action']:5s}  {c['run']}  {c['id']}  {c.get('reason', '')[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
