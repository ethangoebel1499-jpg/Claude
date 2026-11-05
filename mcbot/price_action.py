"""S10 Price action: support / resistance, chart shapes, swing failures and market structure - no indicators.

Everything here is read from the candles themselves (highs, lows, opens, closes); no RSI, MACD, moving averages,
Bollinger bands, Market Cipher or volume. Only confirmed swing points are used (a swing is known a few candles
after it happens), so nothing looks ahead. Works on any timeframe.

Entries (long; shorts are the mirror image):
  1. Support bounce   price dips into a support zone (two or more earlier turning points lined up) and closes back
                      above it with a bullish candle shape (hammer, bullish engulfing, or a close in the top third)
  2. Break & retest   a swing high is broken, then price comes back to it, holds it and closes above
                      ("resistance broken becomes support")
  3. Shape breakout   a close out of a chart shape upward: triangle, wedge, flag, channel, range, double bottom,
                      inverse head & shoulders
  4. Swing failure    price wicks below the last swing low (stops taken) and closes back above it

Filters: no breakout / retest long while the structure is lower highs and lower lows (reversal entries - bounce
and swing failure - are allowed, they are how a trend turns); a swing failure only counts at a support /
resistance zone (a liquidity sweep of a real level); and room to the next resistance must be at least 2.5x the
risk, or the trade is skipped.

Stop: beyond the structure - below the lowest of the candle and the last swing low, plus 2.5 average candles of
breathing room. Targets: half at 1R, half at 2R of that stop (the paper trader's exits, scaled).

Honest note: tested on BTC / ETH / SOL 1h-1D with fees, these rules as a whole did NOT make money (a wider stop,
level-only swing failures and more room roughly halved the losses). The 🔬 Proving ground tests it per market and
timeframe, longs and shorts apart, and only trials it where it made money on both halves of history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .builder import _engulf, _hammer, pattern_frame
from .indicators import sr_levels
from .patterns import swings

K = 3                      # candles each side of a swing point (same as the pattern detector)
RETEST_BARS = 12           # a retest must come within this many candles of the break
ROOM = 2.5                 # room to the next level, in R
STOP_BUFFER = 2.5          # stop this many average candles beyond the structure (tighter stops got wicked out)
STOP = 0.03
TAKE_PROFITS = [(0.03, 0.5), (0.06, 0.5)]  # 1R / 2R of the run's stop - scaled to each trade's own stop


def candle_size(df: pd.DataFrame, n: int = 20) -> np.ndarray:
    """The average high-to-low size of the last n candles (raw price, no indicator)."""
    return (df["high"] - df["low"]).rolling(n, min_periods=1).median().to_numpy(float)


def frame(df: pd.DataFrame) -> pd.DataFrame:
    """Every price-action fact per closed candle (only what was known at its close)."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(df)
    size = candle_size(df)
    hi_idx, lo_idx = swings(df, K)
    # last confirmed swing high / low (and the one before) as of each candle
    last_hi, prev_hi, last_lo, prev_lo = (np.full(n, np.nan) for _ in range(4))
    hk, lk = [], []
    hset, lset = {int(i) + K: int(i) for i in hi_idx}, {int(i) + K: int(i) for i in lo_idx}   # known K bars later
    for i in range(n):
        if i in hset:
            hk.append(h[hset[i]])
        if i in lset:
            lk.append(l[lset[i]])
        if hk:
            last_hi[i] = hk[-1]
            prev_hi[i] = hk[-2] if len(hk) > 1 else np.nan
        if lk:
            last_lo[i] = lk[-1]
            prev_lo[i] = lk[-2] if len(lk) > 1 else np.nan
    up_struct = (last_hi > prev_hi) & (last_lo > prev_lo)
    down_struct = (last_hi < prev_hi) & (last_lo < prev_lo)

    sr = sr_levels(df)
    bull_candle = (_hammer(df, True) | _engulf(df, True)).to_numpy(bool) | (c >= l + (h - l) * 2 / 3)
    bear_candle = (_hammer(df, False) | _engulf(df, False)).to_numpy(bool) | (c <= h - (h - l) * 2 / 3)
    bounce = sr["sr_reclaim_up"].to_numpy(bool) & bull_candle
    reject = sr["sr_reject_down"].to_numpy(bool) & bear_candle

    # break & retest of the last swing high / low
    retest_up, retest_dn = np.zeros(n, bool), np.zeros(n, bool)
    broken_hi = broken_lo = None              # (level, bar it broke)
    for i in range(1, n):
        if np.isfinite(last_hi[i - 1]) and c[i] > last_hi[i - 1] and c[i - 1] <= last_hi[i - 1]:
            broken_hi = (last_hi[i - 1], i)
        if np.isfinite(last_lo[i - 1]) and c[i] < last_lo[i - 1] and c[i - 1] >= last_lo[i - 1]:
            broken_lo = (last_lo[i - 1], i)
        if broken_hi and 1 <= i - broken_hi[1] <= RETEST_BARS:
            lvl = broken_hi[0]
            if l[i] <= lvl + 0.25 * size[i] and c[i] > lvl and c[i] > o[i]:
                retest_up[i] = True
                broken_hi = None
        elif broken_hi and i - broken_hi[1] > RETEST_BARS:
            broken_hi = None
        if broken_lo and 1 <= i - broken_lo[1] <= RETEST_BARS:
            lvl = broken_lo[0]
            if h[i] >= lvl - 0.25 * size[i] and c[i] < lvl and c[i] < o[i]:
                retest_dn[i] = True
                broken_lo = None
        elif broken_lo and i - broken_lo[1] > RETEST_BARS:
            broken_lo = None

    pf = pattern_frame(df)
    shape_up = pf.broke.eq("up").to_numpy(bool)
    shape_dn = pf.broke.eq("down").to_numpy(bool)
    lvl_below = sr["sr_next_below"].to_numpy(float)
    lvl_above = sr["sr_next_above"].to_numpy(float)
    # swing failure = a sweep of a swing low / high that sits on a support / resistance zone
    sfp_up = np.isfinite(last_lo) & (l < last_lo) & (c > last_lo) & np.isfinite(lvl_below) \
        & (np.abs(last_lo - lvl_below) <= 0.5 * size)
    sfp_dn = np.isfinite(last_hi) & (h > last_hi) & (c < last_hi) & np.isfinite(lvl_above) \
        & (np.abs(last_hi - lvl_above) <= 0.5 * size)

    # stops beyond the structure, and the room to the next level
    stop_long = np.fmin(l, np.where(np.isfinite(last_lo) & (last_lo < c), last_lo, l)) - STOP_BUFFER * size
    stop_short = np.fmax(h, np.where(np.isfinite(last_hi) & (last_hi > c), last_hi, h)) + STOP_BUFFER * size
    risk_long = (c - stop_long) / c
    risk_short = (stop_short - c) / c
    room_long = np.where(np.isfinite(lvl_above), (lvl_above - c) / np.maximum(c - stop_long, 1e-12), np.inf)
    room_short = np.where(np.isfinite(lvl_below), (c - lvl_below) / np.maximum(stop_short - c, 1e-12), np.inf)
    return pd.DataFrame({
        "bounce": bounce, "reject": reject, "retest_up": retest_up, "retest_down": retest_dn,
        "shape_up": shape_up, "shape_down": shape_dn, "sfp_up": sfp_up, "sfp_down": sfp_dn,
        "pattern": pf.kind.to_numpy(object), "up_structure": up_struct, "down_structure": down_struct,
        "risk_long": risk_long, "risk_short": risk_short, "room_long": room_long, "room_short": room_short,
        "support": lvl_below, "resistance": lvl_above, "last_swing_low": last_lo, "last_swing_high": last_hi,
    }, index=df.index)


def _entries(f: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ok_risk_l = f.risk_long.between(0.002, 0.10).to_numpy()
    ok_risk_s = f.risk_short.between(0.002, 0.10).to_numpy()
    long_ = ((f.bounce | f.sfp_up).to_numpy() |
             ((f.retest_up | f.shape_up).to_numpy() & ~f.down_structure.to_numpy())) \
        & ok_risk_l & (f.room_long.to_numpy() >= ROOM)
    short = ((f.reject | f.sfp_down).to_numpy() |
             ((f.retest_down | f.shape_down).to_numpy() & ~f.up_structure.to_numpy())) \
        & ok_risk_s & (f.room_short.to_numpy() >= ROOM)
    return long_ & ~short, short & ~long_


def signals(df: pd.DataFrame, direction: str = "both") -> pd.DataFrame:
    f = frame(df)
    long_, short = _entries(f)
    if direction == "long":
        short = np.zeros_like(short)
    elif direction == "short":
        long_ = np.zeros_like(long_)
    stop = np.where(long_, f.risk_long.to_numpy(), np.where(short, f.risk_short.to_numpy(), np.nan))
    return pd.DataFrame({"long_entry": long_, "short_entry": short, "long_exit": False, "short_exit": False,
                         "stop_pct": stop}, index=df.index)


def checklist(df: pd.DataFrame, bar: int = -1, label: str | None = None, direction: str = "both") -> str:
    from .fmt import price
    f = frame(df)
    i = len(df) + bar if bar < 0 else bar
    r = f.iloc[i]
    long_, short = _entries(f)
    yn = lambda x: "🟢 YES" if bool(x) else "🔴 NO "
    struct = "higher highs & higher lows" if r.up_structure else "lower highs & lower lows" if r.down_structure \
        else "mixed / sideways"
    lines = [f"[{label or 'S10 Price action'}] candle {df.index[i]:%Y-%m-%d %H:%M} UTC (close {price(df.close.iloc[i])})",
             f"Structure: {struct} · support {price(r.support) if np.isfinite(r.support) else '-'} · "
             f"resistance {price(r.resistance) if np.isfinite(r.resistance) else '-'}"
             + (f" · shape: {str(r.pattern).replace('_', ' ')}" if r.pattern else "")]
    if direction in ("both", "long"):
        lines += ["", f"LONG: {'🟢 TRADE' if long_[i] else '🔴 NO TRADE'}",
                  f"  {yn(r.bounce)}  Bounced off support with a bullish candle",
                  f"  {yn(r.retest_up)}  Broke a swing high and held it on the retest",
                  f"  {yn(r.shape_up)}  Broke out of a chart shape upward",
                  f"  {yn(r.sfp_up)}  Swing failure: swept the last swing low at support, closed back above",
                  f"  {yn(not r.down_structure)}  Not in a downtrend (needed for breakouts / retests)",
                  f"  {yn(r.room_long >= ROOM)}  Room to the next resistance at least {ROOM:g}x the risk",
                  f"  {yn(0.002 <= r.risk_long <= 0.10)}  Stop below the structure: {r.risk_long * 100:.2f}% away"]
    if direction in ("both", "short"):
        lines += ["", f"SHORT: {'🟢 TRADE' if short[i] else '🔴 NO TRADE'}",
                  f"  {yn(r.reject)}  Rejected at resistance with a bearish candle",
                  f"  {yn(r.retest_down)}  Broke a swing low and failed on the retest",
                  f"  {yn(r.shape_down)}  Broke out of a chart shape downward",
                  f"  {yn(r.sfp_down)}  Swing failure: swept the last swing high at resistance, closed back below",
                  f"  {yn(not r.up_structure)}  Not in an uptrend (needed for breakdowns / retests)",
                  f"  {yn(r.room_short >= ROOM)}  Room to the next support at least {ROOM:g}x the risk",
                  f"  {yn(0.002 <= r.risk_short <= 0.10)}  Stop above the structure: {r.risk_short * 100:.2f}% away"]
    return "\n".join(lines)
