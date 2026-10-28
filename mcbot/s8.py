"""S8 - Basic long-only reversal (any timeframe).

  entry  reversal up: Market Cipher B WaveTrend crosses up below zero (a green
         dot in the lower half) and the Heikin Ashi candle is green
  exit   reversal down: WaveTrend crosses down above zero and the Heikin Ashi
         candle is red. It closes whatever is left of the position
  take profits  50% at +2%, 25% at +5%, the last 25% rides until the reversal down
  no shorts, no stop-loss (stop=... adds an optional safety stop)

Signals use closed candles; trades fill at the next open.
"""

from __future__ import annotations

import pandas as pd

from .indicators import heikin_ashi, wavetrend

TAKE_PROFITS = [(0.02, 0.5), (0.05, 0.25), (None, 0.25)]   # None = rides until the reversal down


def conditions(df: pd.DataFrame) -> pd.DataFrame:
    wt1, wt2 = wavetrend(df)
    ha = heikin_ashi(df)
    green = ha["close"] > ha["open"]
    cross_up = (wt1 > wt2) & (wt1.shift() <= wt2.shift())
    cross_dn = (wt1 < wt2) & (wt1.shift() >= wt2.shift())
    return pd.DataFrame({
        "wt_up_below_zero": cross_up & (wt2 < 0), "ha_green": green,
        "wt_down_above_zero": cross_dn & (wt2 > 0), "ha_red": ~green,
        "wt2": wt2,
    }, index=df.index)


def signals(df: pd.DataFrame) -> pd.DataFrame:
    c = conditions(df)
    up = c.wt_up_below_zero & c.ha_green
    down = c.wt_down_above_zero & c.ha_red
    # short_entry is the "reversal down" that closes the long; shorts are never opened.
    return pd.DataFrame({"long_entry": up, "long_exit": False, "short_entry": down, "short_exit": False},
                        index=df.index)


def checklist(df: pd.DataFrame, bar: int = -1, label: str = "S8 - Long-only reversal") -> str:
    c = conditions(df).iloc[bar]
    real = df.attrs.get("real_dates")
    ts = real[bar] if real is not None else df.index[bar]
    up = bool(c.wt_up_below_zero and c.ha_green)
    down = bool(c.wt_down_above_zero and c.ha_red)
    px = df["close"].iloc[bar]
    lines = [f"[{label}] Checklist for candle {ts:%Y-%m-%d %H:%M} UTC (close {px:,.2f})", "",
             f"LONG: {'🟢 TRADE' if up else '🔴 NO TRADE'}",
             f"  {'🟢 YES' if c.wt_up_below_zero else '🔴 NO '}  WaveTrend crossed up below zero (green dot, wt2 {c.wt2:.0f})",
             f"  {'🟢 YES' if c.ha_green else '🔴 NO '}  Heikin Ashi candle green"]
    if up:
        lines.append(f"  Exit plan: 50% at {px * 1.02:,.2f} (+2%), 25% at {px * 1.05:,.2f} (+5%), "
                     "rest rides until the reversal down")
    lines += ["", f"EXIT (reversal down): {'🔴 GET OUT' if down else '🟢 STAY IN'}",
              f"  {'🔴 YES' if c.wt_down_above_zero else '🟢 NO '}  WaveTrend crossed down above zero (red dot)",
              f"  {'🔴 YES' if c.ha_red else '🟢 NO '}  Heikin Ashi candle red"]
    return "\n".join(lines)
