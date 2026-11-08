"""Trades Claude places for you: Claude goes over a chart on its schedule (e.g. BTC 30m with wielkieef's GRIND rules),
and when it decides to trade it pushes the trade to the paper-trades branch (commands/claude.json). This app checks
that file every couple of minutes while it's open and paper trades it in the run "AUTOGRIND" - with the
script's exits (9% stop; +1.7% / +2.3% / +3.4%), which the paper trader manages between Claude's checks.

  commands/claude.json = {"commands": [{"id": "...", "run": "AUTOGRIND", "symbol": "XBTUSD",
                          "timeframe": "30m", "action": "long" | "short" | "close", "size_pct": 52, "reason": "...",
                          "at": "..."}]}      (size_pct optional: percent of the run's account for this trade, 0-100)

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
RUNS = {# follows TradingView's GRIND BTC strategy only: Claude sends each open / close (with a % of the account)
        "AUTOGRIND": {"strategy": "autogrind", "symbol": "XBTUSD", "asset": "crypto", "timeframe": "30m"}}
# Claude's earlier runs that traded GRIND's own signals: switched off - only AUTOGRIND (screenshot-based) trades now
RETIRED = ("Claude GRIND BTC 30m", "AUTO GRIND BTC 30m")


def run_spec(name: str, cmd: dict | None = None) -> dict:
    from mcbot.grind import STOP, TAKE_PROFITS
    base = RUNS.get(name, {"strategy": "autogrind", "symbol": (cmd or {}).get("symbol", "XBTUSD"),
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
    retire(store)
    done = _applied()
    n = 0
    for c in (fetch() if commands is None else commands):
        cid, action, run = str(c.get("id", "")), c.get("action"), c.get("run") or "AUTOGRIND"
        run = "AUTOGRIND" if run in RETIRED else run
        if not cid or cid in done or action not in ("long", "short", "close"):
            continue
        have = {r["name"]: r for r in store.runs()}
        if run not in have or not have[run]["active"]:
            store.upsert_run(run_spec(run, c))
        try:
            size = float(c["size_pct"]) if c.get("size_pct") is not None else None
        except (TypeError, ValueError):
            size = None
        if size is not None and not 0 < size <= 100:
            size = None
        store.command(run, action, size)
        store.event(run, c.get("at") or "", "claude", f"Claude: {action}"
                    + (f" ({size:g}% of the account)" if size is not None else "") + f" - {c.get('reason', '')}"[:500])
        done.add(cid)
        n += 1
    if n:
        APPLIED.parent.mkdir(parents=True, exist_ok=True)
        APPLIED.write_text(json.dumps(sorted(done)[-500:]))
    return n


def autogrind_entry() -> dict:
    """AUTOGRIND: no signals of its own. It only opens / closes what Claude sends from TradingView's GRIND BTC
    strategy (with the % of the account to use), and manages the script's exits (9% stop; +1.7 / +2.3 / +3.4%)."""
    import pandas as pd

    from mcbot.grind import STOP, TAKE_PROFITS

    from .core import Exits

    def signals(df: pd.DataFrame) -> pd.DataFrame:
        off = pd.Series(False, index=df.index)
        return pd.DataFrame({"long_entry": off, "long_exit": off, "short_entry": off, "short_exit": off})

    def checklist(df, bar=-1, label=None):
        return ("🤖 AUTOGRIND follows TradingView's GRIND BTC [45MIN] strategy: Claude checks it every 30 minutes and "
                "sends each open / close, with the % of the account to use. It has no signals of its own.")

    return {"label": "🤖 AUTOGRIND (follows TradingView's GRIND BTC)", "short": "AUTOGRIND", "signals": signals,
            "checklist": checklist, "exits": Exits(STOP, TAKE_PROFITS),
            "note": "opens / closes only what Claude sends from TradingView's GRIND BTC strategy, sized as a % of the "
                    "account, with the script's 9% stop and +1.7% / +2.3% / +3.4% targets."}


def retire(store) -> None:
    """Claude's earlier GRIND runs traded the script's own signals; they stop (their trades stay in the log)."""
    for r in store.runs():
        if r["name"] in RETIRED and r["active"]:
            store.set_active(r["name"], False)


def registry_entry() -> dict:
    """The retired 'claude_grind' strategy, kept so old runs still load: it no longer trades anything itself."""
    e = autogrind_entry()
    return {**e, "label": "Claude GRIND (retired - use AUTOGRIND)", "short": "Claude GRIND (retired)"}
