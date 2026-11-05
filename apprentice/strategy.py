"""The apprentice as a paper-trading strategy ("🧠 Apprentice"): it goes long / short whenever one of
its active setups (tested, beating random, weight above zero) fires on a closed candle. As it learns,
the strategy changes by itself - the trader re-reads its setups on every price check.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from mcbot import builder

ROOT = Path(__file__).resolve().parent.parent
KEY = "apprentice"
# moderate exits: 3% stop, half off at +3% (1R), the rest at +6% (2R) - close to how its setups are tested
# (1.5 ATR stop, 2R target)
STOP = 0.03
TAKE_PROFITS = [(0.03, 0.5), (0.06, 0.5)]
BREAKEVEN = 0.5          # stop to the entry once a trade is half way to its first target (+1.5% with the 3% stop)
# trade-safety rules (you asked for both; 🎓 Learn checks each per timeframe and turns it off if it hurt)
SAFETY = {"no_opposite": "Never enter against an opposite dot or divergence on the entry candle",
          "breakeven": "Stop to the entry once the trade is +1.5% (half way to the first target)"}


def safety_rules(tf: str, store=None) -> dict[str, bool]:
    """Which safety rules are on for this timeframe: on until 🎓 Learn has checked them, then what held up."""
    learned = {}
    try:
        if store is None and Path(db_path()).exists():
            from .store import Store
            s = Store(db_path())
            try:
                learned = (s.meta_get("safety") or {}).get(tf, {})
            finally:
                s.db.close()
        elif store is not None:
            learned = (store.meta_get("safety") or {}).get(tf, {})
    except Exception:
        learned = {}
    return {k: bool(learned.get(k, {}).get("on", True)) for k in SAFETY}


def timeframe_status(tf: str, store=None) -> dict:
    """{"trade": bool, ...} for a timeframe: 🎓 Learn pauses one where all the live setups together lost money on
    prices they never saw. A timeframe it hasn't checked yet trades as before."""
    try:
        if store is None:
            if not Path(db_path()).exists():
                return {"trade": True}
            from .store import Store
            s = Store(db_path())
            try:
                st = (s.meta_get("timeframes") or {}).get(tf)
            finally:
                s.db.close()
        else:
            st = (store.meta_get("timeframes") or {}).get(tf)
    except Exception:
        st = None
    return st or {"trade": True}


def opposite_signals(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(blocks a long, blocks a short) on each closed candle: a red dot or a bearish divergence (RSI, wave or
    AO, in the last 10 candles) blocks longs; a green dot or a bullish divergence blocks shorts."""
    r = lambda rule, **kw: builder.rule_series(df, {"rule": rule, **kw}).to_numpy(bool)
    bear = r("rsi_bear_div", a=10) | r("wt_bear_div", a=10) | r("ao_bear_div", a=10) | r("wt_red_dot", a=-1000)
    bull = r("rsi_bull_div", a=10) | r("wt_bull_div", a=10) | r("ao_bull_div", a=10) | r("wt_green_dot", a=1000)
    return bear, bull


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


def signals(df: pd.DataFrame, setups: list[dict] | None = None, mode: str | None = None,
            safety: dict[str, bool] | None = None, paused: bool | None = None) -> pd.DataFrame:
    """Entries from every live setup, plus what it learned about getting out (🎓 Learn decides, per setup):
    stop_pct - the trade's own stop (fixed 3% / ATR / beyond the last swing), only where that beat the 3%
               stop on prices it never saw; NaN = the run's stop
    use_reversal / *_reversal - close a losing trade on a confirmed reversal, only where that helped."""
    from .brain import reversal_signals, stop_pct, timeframe_of
    setups = live_setups() if setups is None else setups
    tf = timeframe_of(df)                              # plans are learned (and checked) per timeframe
    n = len(df)
    long_ = np.zeros(n, bool)
    short = np.zeros(n, bool)
    stop = np.full(n, np.nan)
    rev = np.zeros(n, bool)
    best = np.full(n, -np.inf)                        # the most trusted setup that fired sets the exits
    mode = mode or reversal_mode()
    for s in sorted(setups, key=lambda s: s.get("weight") or 0):
        m = builder._all(df, s["rules"]).to_numpy(bool)
        if not m.any():
            continue
        (long_ if s["direction"] == "long" else short)[m] = True
        plan = ((s.get("stats") or {}).get("exit_plan") or {}).get(tf) or {}
        w = float(s.get("weight") or 0)
        take = m & (w >= best)
        best[take] = w
        sp = stop_pct(df, s["direction"], plan["stop"], float(s.get("stop_atr") or 1.5)) \
            if plan.get("stop") in ("atr", "swing") else np.full(n, np.nan)
        stop[take] = sp[take]
        rev[take] = mode == "always" or (mode == "learned" and bool(plan.get("reversal")))
    safety = safety_rules(tf) if safety is None else safety
    if safety.get("no_opposite"):
        no_long, no_short = opposite_signals(df)
        long_ &= ~no_long
        short &= ~no_short
    be = BREAKEVEN * np.where(np.isfinite(stop), stop, STOP) if safety.get("breakeven") else np.full(n, np.nan)
    out = pd.DataFrame({"long_entry": long_ & ~short, "short_entry": short & ~long_,
                        "long_exit": False, "short_exit": False, "stop_pct": stop, "be_pct": be}, index=df.index)
    out["pause"] = (not timeframe_status(tf)["trade"]) if paused is None else bool(paused)
    if mode != "off":
        out["long_reversal"], out["short_reversal"] = reversal_signals(df)
        out["use_reversal"] = rev
    return out


def reversal_mode() -> str:
    """Closing a LOSING trade early on a confirmed reversal (🧠 Apprentice → After the entry):
    "learned" (default) - only for setups where 🎓 Learn showed it helped, on past AND unseen prices
    "always" / "off"."""
    try:
        from livepaper import settings
        v = settings.get("apprentice_reversal_exit", "learned")
    except Exception:
        return "learned"
    if v is True:
        return "always"
    if v is False or v not in ("learned", "always", "off"):
        return "off" if v is False else "learned"
    return v


def checklist(df: pd.DataFrame, bar: int = -1, label: str | None = None) -> str:
    setups = live_setups()
    i = len(df) + bar if bar < 0 else bar
    lines = [f"[{label or 'Apprentice'}] candle {df.index[i]:%Y-%m-%d %H:%M} UTC"]
    if not setups:
        return lines[0] + "\nNo active setups yet - teach it in 🧠 Apprentice and press Learn."
    from .brain import timeframe_of
    tf = timeframe_of(df)
    tfs = timeframe_status(tf)
    if not tfs["trade"]:
        lines.append(f"⏸️ {tf} is paused: in 🎓 Learn all its setups together lost {tfs.get('unseen_pct', 0):+.1f}% on prices "
                     "they never saw, so it opens no new trades here (open ones keep their stop and targets). It "
                     "resumes by itself if a later Learn shows it making money.")
    rules = safety_rules(tf)
    if rules["no_opposite"]:
        no_long, no_short = opposite_signals(df)
        if no_long[i]:
            lines.append("⛔ No new LONG on this candle: a red dot or bearish divergence is printing (safety rule)")
        if no_short[i]:
            lines.append("⛔ No new SHORT on this candle: a green dot or bullish divergence is printing (safety rule)")
    if rules["breakeven"]:
        lines.append("🛡️ Breakeven: the stop moves to the entry once a trade is +1.5% (half way to the first target)")
    for s in setups:
        ok = [bool(builder.rule_series(df, r).iloc[i]) for r in s["rules"]]
        lines += ["", f"{s['direction'].upper()} · {s['name']}: {'🟢 TRADE' if all(ok) else '🔴 NO TRADE'} "
                      f"(weight {s['weight']:+.2f})"]
        lines += [f"  {'🟢 YES' if o else '🔴 NO '}  {builder.describe(r)}" for o, r in zip(ok, s["rules"])]
    return "\n".join(lines)


def entry() -> dict:
    from livepaper.core import Exits
    return {"label": "🧠 Apprentice (learns from your lessons)", "short": "Apprentice", "signals": signals,
            "checklist": checklist, "exits": Exits(STOP, TAKE_PROFITS),
            "note": "Trades every setup the apprentice has learned and tested (🧠 Apprentice tab), long and short. It "
                    "changes as it learns. Exits: 3% stop, 50% at +1R (3%), 50% at +2R (6%), stop to the entry once "
                    "it's +1.5% up - unless 🎓 Learn found a better stop for the setup that fired (ATR or beyond the "
                    "last swing low/high); the targets then move with that stop. Never enters against an opposite "
                    "dot or divergence."}


def refresh(registry: dict) -> dict:
    registry[KEY] = entry()
    return registry


def run_for(coin: str, pair: str, tf: str, capital: float = 10_000.0) -> dict:
    """The paper-trading run for one market: long and short, moderate exits."""
    return {"name": f"Apprentice {coin} {tf}", "strategy": KEY, "symbol": pair, "asset": "crypto", "timeframe": tf,
            "shorts": True, "capital": capital, "fraction": 1.0, "stop": STOP,
            "take_profits": [list(t) for t in TAKE_PROFITS]}
