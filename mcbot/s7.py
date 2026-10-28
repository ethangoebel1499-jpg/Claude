"""S7 - Range levels + 24m / 12m Market Cipher confirmations (Jason Casper's "bread and butter").

  levels  4h range highs / lows and support-resistance flips (clustered 4h swing points)
  long    price comes to a range low / support, and on BOTH the 24m and 12m:
            two big green dots (both wave lows below -53) with a bullish
            divergence (price lower low, Market Cipher B higher low)
  short   price comes to a range high / resistance, and on BOTH the 24m and 12m:
            a bearish divergence (price higher high, MCB lower high) with the
            money flow getting lower as price gets higher
  stop    below the last low (above the last high), with some room
  exits   "partial" (his way): 50% at the first level, stop to entry, rest at
          the level after; "single": one target at the next range boundary
  rr      optional: only take trades whose final target is >= rr x the risk

Regular candles. Signals on closed candles; trades run on the 12m chart and
fill at the next 12m open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import Result, run_level_backtest
from .data import resample
from .divergence import recent, waves
from .indicators import OS_LEVEL, money_flow, sr_levels, wavetrend

NEAR = 0.005          # "areas, not exact levels": within 0.5%
LEVEL_BARS = 10       # price touched the level in the last 10 x 24m (4 hours)
SIGNAL_BARS = 5       # the dot / divergence printed in the last 5 bars of its chart
MAX_GAP = 40          # the two waves of a divergence are at most 40 bars apart
STOP_ROOM = 0.003


def _mcb_divs(df: pd.DataFrame) -> pd.DataFrame:
    wt1, wt2 = wavetrend(df)
    up = (wt1 > wt2) & (wt1.shift() <= wt2.shift())
    dn = (wt1 < wt2) & (wt1.shift() >= wt2.shift())
    mf = money_flow(df)
    w = waves(wt2, df["low"], df["high"], up, dn, extra={"mf": mf})
    tgap = (w.t1_i - w.t0_i) <= MAX_GAP
    pgap = (w.p1_i - w.p0_i) <= MAX_GAP
    bull = up & tgap & (w.t1_px < w.t0_px) & (w.t1_osc > w.t0_osc) & \
        (w.t1_osc <= OS_LEVEL) & (w.t0_osc <= OS_LEVEL)
    bear = dn & pgap & (w.p1_px > w.p0_px) & (w.p1_osc < w.p0_osc) & (w.p1_mf < w.p0_mf)
    return pd.DataFrame({"bull": recent(bull.fillna(False), SIGNAL_BARS),
                         "bear": recent(bear.fillna(False), SIGNAL_BARS),
                         "low_px": w.t1_px, "high_px": w.p1_px}, index=df.index)


def _on(fast_index: pd.DatetimeIndex, slow: pd.DataFrame, slow_step: pd.Timedelta,
        fast_step: pd.Timedelta) -> pd.DataFrame:
    s = slow.copy()
    s.index = slow.index + slow_step - fast_step     # known when the slow candle closes
    return s.reindex(fast_index, method="ffill")


def signals(m12: pd.DataFrame, rr: float = 0.0, exits: str = "partial") -> pd.DataFrame:
    """m12: 12-minute candles. Returns entries plus the exit levels per 12m bar."""
    step = pd.Timedelta("12min")
    m24 = resample(m12, "24min")
    d24 = _on(m12.index, _mcb_divs(m24), pd.Timedelta("24min"), step)
    d12 = _mcb_divs(m12)
    lv = sr_levels(m12, factor=20)          # 20 x 12m = 4h levels
    lo_k = m12["low"].rolling(LEVEL_BARS * 2).min()
    hi_k = m12["high"].rolling(LEVEL_BARS * 2).max()
    at_support = lo_k <= lv.sr_next_below * (1 + NEAR)
    at_resist = hi_k >= lv.sr_next_above * (1 - NEAR)
    long_e = at_support & d24.bull & d12.bull
    short_e = at_resist & d24.bear & d12.bear

    c = m12["close"]
    long_stop = np.minimum(d24.low_px, d12.low_px) * (1 - STOP_ROOM)
    short_stop = np.maximum(d24.high_px, d12.high_px) * (1 + STOP_ROOM)
    a1, a2 = lv.sr_next_above, lv.sr_next_above2.fillna(lv.sr_next_above)
    b1, b2 = lv.sr_next_below, lv.sr_next_below2.fillna(lv.sr_next_below)
    if exits == "single":
        long_tp1, long_tp2, short_tp1, short_tp2 = a2, a2 * np.nan, b2, b2 * np.nan
        long_final, short_final = a2, b2
    else:
        long_tp1, long_tp2, short_tp1, short_tp2 = a1, a2, b1, b2
        long_final, short_final = a2, b2
    if rr > 0:
        long_e &= (long_final - c) >= rr * (c - long_stop)
        short_e &= (c - short_final) >= rr * (short_stop - c)
    return pd.DataFrame({
        "long_entry": long_e.fillna(False), "short_entry": short_e.fillna(False),
        "long_exit": False, "short_exit": False,
        "at_support": at_support, "at_resist": at_resist,
        "bull24": d24.bull, "bull12": d12.bull, "bear24": d24.bear, "bear12": d12.bear,
        "long_stop": long_stop, "short_stop": short_stop,
        "long_tp1": long_tp1, "long_tp2": long_tp2, "short_tp1": short_tp1, "short_tp2": short_tp2,
        "long_trail": -np.inf, "short_trail": np.inf,
    }, index=m12.index)


def backtest(m12: pd.DataFrame, fr: pd.DataFrame, *, allow_short: bool = True, fee: float = 0.001,
             exits: str = "partial") -> Result:
    lv = fr[["long_stop", "long_tp1", "long_tp2", "long_trail", "short_stop", "short_tp1", "short_tp2",
             "short_trail"]]
    sizes = (1.0,) if exits == "single" else (0.5, 0.5)
    return run_level_backtest(m12, fr[["long_entry", "short_entry", "long_exit", "short_exit"]], lv,
                              allow_short=allow_short, fee=fee, sizes=sizes,
                              breakeven_after_tp1=exits != "single")


def checklist(m12: pd.DataFrame, bar: int = -1, label: str = "S7 - Range levels + 24m/12m") -> str:
    fr = signals(m12)
    r = fr.iloc[bar]
    ts = m12.index[bar] + pd.Timedelta("12min")
    lines = [f"[{label}] Checklist for candle closed {ts:%Y-%m-%d %H:%M} UTC (close {m12['close'].iloc[bar]:,.2f})"]
    for side, conds, go in (
            ("LONG", [(r.at_support, "Price came to a range low / support (4h levels)"),
                      (r.bull24, "24m: two big green dots with a bullish divergence"),
                      (r.bull12, "12m: two big green dots with a bullish divergence")], r.long_entry),
            ("SHORT", [(r.at_resist, "Price came to a range high / resistance (4h levels)"),
                       (r.bear24, "24m: bearish divergence with money flow getting lower"),
                       (r.bear12, "12m: bearish divergence with money flow getting lower")], r.short_entry)):
        lines += ["", f"{side}: {'🟢 TRADE' if go else '🔴 NO TRADE'}  ({sum(bool(c) for c, _ in conds)}/3 yes)"]
        lines += [f"  {'🟢 YES' if c else '🔴 NO '}  {lab}" for c, lab in conds]
        if go:
            k = "long" if side == "LONG" else "short"
            lines.append(f"  Exit plan: stop {r[f'{k}_stop']:,.2f}; 50% at {r[f'{k}_tp1']:,.2f}, stop to entry; "
                         f"rest at {r[f'{k}_tp2']:,.2f}")
    return "\n".join(lines)
