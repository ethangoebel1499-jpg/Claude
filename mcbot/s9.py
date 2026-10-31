"""S9 - Falling wedge breakout after an uptrend (long only).

From the pattern study (reports/patterns.md): falling wedges that formed after an uptrend broke
UP 85% of the time - the best combination found, about 10 points better than the same shape on
random prices. Breaks before 75% of the way to the apex reached their target far more often
than late ones.

  setup    a falling wedge (both lines falling, converging) found from confirmed swing points
  filter   uptrend before it: price higher than 50 candles before the wedge started
           early break: less than `max_progress` (75%) of the way to where the lines meet
  entry    a candle CLOSES above the wedge's top line -> buy at the next open
  stop     the wedge's bottom line at the break candle ("back inside past the opposite line")
  exits    "measured": all at the measured target (break + wedge height)
           "half":     50% at half the height, stop to entry, 50% at the full height

Signals use closed candles only; the detector never looks ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import Result, run_level_backtest
from .fmt import price
from .patterns import Detector

MAX_PROGRESS = 0.75


def events(df: pd.DataFrame, kinds: tuple[str, ...] = ("falling_wedge",)) -> pd.DataFrame:
    """Every candle that closed above the top line of a wedge found on the candle before."""
    det = Detector(df)
    rows = []
    for i in range(31, len(df)):
        p = det.find(i - 1)
        if p is None or p.kind not in kinds or det.broke(p, i) != "up":
            continue
        rows.append({"i": i, "trend": p.trend, "progress": p.progress(i - 1), "stop": p.lo(i), "top": p.up(i),
                     "height": p.height})
    return pd.DataFrame(rows, columns=["i", "trend", "progress", "stop", "top", "height"])


def frame(df: pd.DataFrame, ev: pd.DataFrame, *, uptrend: bool = True, max_progress: float | None = MAX_PROGRESS,
          exits: str = "measured") -> pd.DataFrame:
    n = len(df)
    entry = np.zeros(n, bool)
    stop, tp1, tp2 = (np.full(n, np.nan) for _ in range(3))
    for e in ev.itertuples(index=False):
        if uptrend and e.trend != "up":
            continue
        if max_progress is not None and (e.progress is None or pd.isna(e.progress) or e.progress >= max_progress):
            continue
        entry[e.i], stop[e.i] = True, e.stop
        if exits == "half":
            tp1[e.i], tp2[e.i] = e.top + e.height / 2, e.top + e.height
        else:
            tp1[e.i] = e.top + e.height
    return pd.DataFrame({"long_entry": entry, "short_entry": False, "long_exit": False, "short_exit": False,
                         "long_stop": stop, "long_tp1": tp1, "long_tp2": tp2, "long_trail": -np.inf,
                         "short_stop": np.nan, "short_tp1": np.nan, "short_tp2": np.nan, "short_trail": np.inf},
                        index=df.index)


def signals(df: pd.DataFrame, *, uptrend: bool = True, max_progress: float | None = MAX_PROGRESS,
            kinds: tuple[str, ...] = ("falling_wedge",), exits: str = "measured") -> pd.DataFrame:
    return frame(df, events(df, kinds), uptrend=uptrend, max_progress=max_progress, exits=exits)


def backtest(df: pd.DataFrame, fr: pd.DataFrame, *, allow_short: bool = False, fee: float = 0.001,
             exits: str = "measured") -> Result:
    lv = fr[["long_stop", "long_tp1", "long_tp2", "long_trail", "short_stop", "short_tp1", "short_tp2",
             "short_trail"]]
    sizes = (0.5, 0.5) if exits == "half" else (1.0,)
    return run_level_backtest(df, fr[["long_entry", "short_entry", "long_exit", "short_exit"]], lv,
                              allow_short=False, fee=fee, sizes=sizes, breakeven_after_tp1=exits == "half")


def checklist(df: pd.DataFrame, bar: int = -1, label: str = "S9 - Falling wedge after uptrend") -> str:
    det = Detector(df)
    i = len(df) + bar if bar < 0 else bar
    p = det.find(i - 1)
    wedge = p is not None and p.kind == "falling_wedge"
    up = wedge and p.trend == "up"
    prog = p.progress(i - 1) if wedge else None
    early = prog is not None and prog < MAX_PROGRESS
    broke = wedge and det.broke(p, i) == "up"
    go = wedge and up and early and broke
    yes = lambda b: "🟢 YES" if b else "🔴 NO "
    lines = [f"[{label}] Checklist for candle {df.index[i]:%Y-%m-%d %H:%M} UTC (close {price(df['close'].iloc[i])})", "",
             f"LONG: {'🟢 TRADE' if go else '🔴 NO TRADE'}",
             f"  {yes(wedge)}  Falling wedge on the chart" + (f" (top line {price(p.up(i))}, bottom line {price(p.lo(i))})" if wedge else ""),
             f"  {yes(up)}  Uptrend before the wedge",
             f"  {yes(early)}  Early: less than 75% of the way to the apex" + (f" (now {prog:.0%})" if prog is not None else ""),
             f"  {yes(broke)}  Candle closed above the top line"]
    if go:
        lines.append(f"  Exit plan: stop {price(p.lo(i))} (bottom line); target {price(p.up(i) + p.height)} "
                     "(wedge height added to the break)")
    return "\n".join(lines)
