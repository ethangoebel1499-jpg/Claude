"""The apprentice as a paper-trading strategy ("🧠 Apprentice"): it goes long / short whenever one of
its active setups (tested, beating random, weight above zero) fires on a closed candle. As it learns,
the strategy changes by itself - the trader re-reads its setups on every price check.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from mcbot import builder

ROOT = Path(__file__).resolve().parent.parent
KEY = "apprentice"


def db_path() -> str:
    env = os.environ.get("APPRENTICE_DB")
    if env:
        return env
    try:
        from livepaper import settings
        p = settings.get("apprentice_db") or "apprentice.db"
    except Exception:
        p = "apprentice.db"
    return str(p if Path(p).is_absolute() else ROOT / p)


def live_setups() -> list[dict]:
    if not Path(db_path()).exists():
        return []
    from .store import Store
    st = Store(db_path())
    try:
        return [s for s in st.setups(("active", "pinned")) if s["weight"] > 0 or s["status"] == "pinned"]
    finally:
        st.db.close()


def signals(df: pd.DataFrame, setups: list[dict] | None = None) -> pd.DataFrame:
    setups = live_setups() if setups is None else setups
    long_ = pd.Series(False, index=df.index)
    short = pd.Series(False, index=df.index)
    for s in setups:
        m = builder._all(df, s["rules"])
        if s["direction"] == "long":
            long_ |= m
        else:
            short |= m
    return pd.DataFrame({"long_entry": long_ & ~short, "short_entry": short & ~long_,
                         "long_exit": pd.Series(False, index=df.index), "short_exit": pd.Series(False, index=df.index)},
                        index=df.index)


def checklist(df: pd.DataFrame, bar: int = -1, label: str | None = None) -> str:
    setups = live_setups()
    i = len(df) + bar if bar < 0 else bar
    lines = [f"[{label or 'Apprentice'}] candle {df.index[i]:%Y-%m-%d %H:%M} UTC"]
    if not setups:
        return lines[0] + "\nNo active setups yet - teach it in 🧠 Apprentice and press Learn."
    for s in setups:
        ok = [bool(builder.rule_series(df, r).iloc[i]) for r in s["rules"]]
        lines += ["", f"{s['direction'].upper()} · {s['name']}: {'🟢 TRADE' if all(ok) else '🔴 NO TRADE'} "
                      f"(weight {s['weight']:+.2f})"]
        lines += [f"  {'🟢 YES' if o else '🔴 NO '}  {builder.describe(r)}" for o, r in zip(ok, s["rules"])]
    return "\n".join(lines)


def entry() -> dict:
    from livepaper.core import Exits
    return {"label": "🧠 Apprentice (learns from your lessons)", "short": "Apprentice", "signals": signals,
            "checklist": checklist, "exits": Exits(0.04, [(0.06, 0.5), (0.12, 0.5)]),
            "note": "Trades every setup the apprentice has learned and tested (🧠 Apprentice tab). It changes as it "
                    "learns. Default exits: 4% stop, 50% at +6%, 50% at +12%."}


def refresh(registry: dict) -> dict:
    registry[KEY] = entry()
    return registry
