"""Trades Claude places for you: Claude goes over a chart on its schedule (e.g. BTC 30m with wielkieef's GRIND rules),
and when it decides to trade it pushes the trade to the paper-trades branch (commands/claude.json). This app checks
that file every couple of minutes while it's open and paper trades it in the run "Claude GRIND BTC 30m" - with the
script's exits (9% stop; +1.7% / +2.3% / +3.4%), which the paper trader manages between Claude's checks.

  commands/claude.json = {"commands": [{"id": "...", "run": "Claude GRIND BTC 30m", "symbol": "XBTUSD",
                          "timeframe": "30m", "action": "long" | "short" | "close", "reason": "...", "at": "..."}]}

Each command is applied once (its id is remembered in runtime/claude_commands_applied.json). Turn it off in
⚙️ Account (claude_trades_on).
"""

from __future__ import annotations

import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
APPLIED = ROOT / "runtime" / "claude_commands_applied.json"
PATH = "commands/claude.json"
RUNS = {"Claude GRIND BTC 30m": {"strategy": "claude_grind", "symbol": "XBTUSD", "asset": "crypto", "timeframe": "30m"}}


def run_spec(name: str, cmd: dict | None = None) -> dict:
    from mcbot.grind import STOP, TAKE_PROFITS
    base = RUNS.get(name, {"strategy": "claude_grind", "symbol": (cmd or {}).get("symbol", "XBTUSD"),
                           "asset": "crypto", "timeframe": (cmd or {}).get("timeframe", "30m")})
    return {"name": name, **base, "shorts": True, "longs": True, "capital": 10_000.0, "fraction": 1.0,
            "stop": STOP, "take_profits": [list(t) for t in TAKE_PROFITS], "active": True, "source": "claude"}


def _applied() -> set[str]:
    try:
        return set(json.loads(APPLIED.read_text()))
    except Exception:
        return set()


def fetch() -> list[dict]:
    from . import settings
    from .sync import BRANCH, REPO, GitHub
    token = settings.get("github_token")
    try:
        if token:
            text = GitHub(token).read(PATH)
        else:
            r = requests.get(f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{PATH}", timeout=20)
            text = r.text if r.status_code == 200 else None
    except Exception:
        text = None
    if not text:
        return []
    try:
        return list(json.loads(text).get("commands", []))
    except Exception:
        return []


def poll(store, commands: list[dict] | None = None) -> int:
    """Apply Claude's new trades: make sure its run exists, then queue the trade like a 📄 force-trade button."""
    done = _applied()
    n = 0
    for c in (fetch() if commands is None else commands):
        cid, action, run = str(c.get("id", "")), c.get("action"), c.get("run") or "Claude GRIND BTC 30m"
        if not cid or cid in done or action not in ("long", "short", "close"):
            continue
        have = {r["name"]: r for r in store.runs()}
        if run not in have or not have[run]["active"]:
            store.upsert_run(run_spec(run, c))
        store.command(run, action)
        store.event(run, c.get("at") or "", "claude", f"Claude: {action} - {c.get('reason', '')}"[:500])
        done.add(cid)
        n += 1
    if n:
        APPLIED.parent.mkdir(parents=True, exist_ok=True)
        APPLIED.write_text(json.dumps(sorted(done)[-500:]))
    return n


def registry_entry() -> dict:
    """The run's strategy: no signals of its own (Claude sends the trades); shows GRIND's checklist."""
    import pandas as pd

    from mcbot.grind import STOP, TAKE_PROFITS, grind_checklist

    from .core import Exits

    def signals(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"long_entry": False, "short_entry": False, "long_exit": False, "short_exit": False},
                            index=df.index)

    def checklist(df, bar=-1, label=None):
        return ("🤖 Claude trades this run by hand: it goes over the chart with wielkieef's GRIND rules every hour "
                "and sends each trade through GitHub.\n\n" + grind_checklist(df, bar, label=label or "GRIND"))

    return {"label": "🤖 Claude · GRIND (trades sent by Claude)", "short": "Claude GRIND", "signals": signals,
            "checklist": checklist, "exits": Exits(STOP, TAKE_PROFITS),
            "note": "Claude goes over the chart with wielkieef's GRIND rules on its schedule and sends the trades; the "
                    "paper trader manages the script's 9% stop and +1.7% / +2.3% / +3.4% targets in between."}
