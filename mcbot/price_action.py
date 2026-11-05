"""S10 Price action: support / resistance, chart shapes, swing failures and market structure - no indicators.

Everything is read from the candles themselves (highs, lows, opens, closes) - no RSI, MACD, moving averages,
Bollinger bands, Market Cipher or volume. Only confirmed swing points are used, so nothing looks ahead. Any
timeframe.

ENTRIES (long; shorts are the mirror image)
  1. Support bounce   price dips into a support zone (2+ earlier turning points lined up) and closes back above it
                      with a bullish candle shape (hammer, bullish engulfing, or a close in the top third)
  2. Break & retest   a swing high breaks, then price comes back to it, holds it and closes above
  3. Shape breakout   a triangle / wedge / channel / flag / range / double top-bottom / head & shoulders breaks -
                      and S10 follows the BREAKOUT STATISTICS measured on crypto since 2017 (mcbot/pattern_stats.json):
                        - only in the direction that shape broke at least 60% of the time (in the current trend) -
                          e.g. falling wedges and descending channels up, rising wedges and ascending channels down,
                          ascending triangles DOWN (they broke up only 36% on crypto)
                        - entered on the RETEST of the broken line (67-83% of breaks came back to it), not the break
                        - the measured move is used as the far target only for shapes that reached it often enough
  4. Swing failure    price sweeps the last swing low where it sits on a support zone (the stops below it are taken)
                      and closes back above it

STOPS, TARGETS, MANAGEMENT - as the lessons teach
  - "stop below the last swing low / above the swing high", "above the wick of the high"
  - breakout: "stop just behind the broken line" (and below the retest wick)
  - bounce: "stop below the zone" / "below the low of the range"; swing failure: beyond the sweep wick
  - "stops below zones of liquidity" with "playing room" - whales hunt the obvious stops - so a buffer of
    STOP_BUFFER average candles goes beyond the structure
  - "take profit just before the next level of resistance": TP1 just before the next level, TP2 the level after
    it (or the measured move), half each; a trade needs TP1 at least MIN_RR x its risk
  - "after the first take profit, move the stop to breakeven"

Structure filter: no breakout / retest long while the swings make lower highs and lower lows (bounces and swing
failures are allowed - they're how a trend turns).

WHAT IT TRADES: a statistically-backed shape-breakout retest on its own, or any CONFLUENCE of two reasons on the same
candle (e.g. a support bounce that is also a break & retest). Single bounces, rejections and swing failures show in
the checklist but aren't traded: measured on BTC / ETH / SOL 1h-1D (fees in, half at TP1, breakeven after), they
lost or were flat on their own, while shape-retest longs made +0.32R a trade on the earlier half and +0.21R on the
later half. (That rule was chosen after seeing both halves, so the 🔬 Proving ground's 100-trade trial is the real
test.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .builder import _engulf, _hammer, pattern_frame
from .indicators import sr_levels
from .patterns import swings

K = 3                      # candles each side of a swing point (same as the pattern detector)
RETEST_BARS = 12           # a retest must come within this many candles of the break
STOP_BUFFER = 1.0          # average candles of playing room beyond the structure (stop hunts)
MIN_RR = 1.0               # TP1 must be at least this many R away
MIN_ODDS = 60.0            # a shape break is traded only if that direction won this % of the time on crypto
MIN_TARGET = 45.0          # ... and its measured move is used as TP2 only if reached this % of the time
CONFLUENCE = 2             # other entries need this many price-action reasons on the same candle
STOP = 0.03
TAKE_PROFITS = [(0.03, 0.5), (0.06, 0.5)]  # fallback 1R / 2R; S10 sets each trade's own targets
WITH = {"ascending_triangle": "up", "descending_triangle": "down", "rising_wedge": "down", "falling_wedge": "up",
        "ascending_channel": "up", "descending_channel": "down", "bull_flag": "up", "bear_flag": "down",
        "double_top": "down", "double_bottom": "up", "head_shoulders": "down", "inverse_head_shoulders": "up"}


def candle_size(df: pd.DataFrame, n: int = 20) -> np.ndarray:
    """The average high-to-low size of the last n candles (raw price, no indicator)."""
    return (df["high"] - df["low"]).rolling(n, min_periods=1).median().to_numpy(float)


def odds(kind: str, direction: str, trend: str | None) -> tuple[float | None, float | None]:
    """(% of breaks in this direction, % of such breaks that reached the measured target) - crypto since 2017."""
    from .patterns import stats
    st = stats().get(kind) or {}
    o = st.get("odds") or {}
    t = (st.get("odds_trend") or {}).get(trend) if trend else None
    up = t["up"] if t and t.get("n", 0) >= 20 else o.get("up")
    if up is None or not st.get("n"):
        return None, None
    p = up if direction == "up" else 100 - up
    tgt = st.get("target_with") if WITH.get(kind) == direction else st.get("target_against", st.get("target_all"))
    return float(p), (float(tgt) if tgt is not None else None)


def frame(df: pd.DataFrame) -> pd.DataFrame:
    """Every price-action fact and the planned trade per closed candle (only what was known at its close)."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(df)
    size = candle_size(df)
    hi_idx, lo_idx = swings(df, K)
    last_hi, prev_hi, last_lo, prev_lo = (np.full(n, np.nan) for _ in range(4))
    hk, lk = [], []
    hset, lset = {int(i) + K: int(i) for i in hi_idx}, {int(i) + K: int(i) for i in lo_idx}   # known K bars later
    for i in range(n):
        if i in hset:
            hk.append(h[hset[i]])
        if i in lset:
            lk.append(l[lset[i]])
        if hk:
            last_hi[i], prev_hi[i] = hk[-1], (hk[-2] if len(hk) > 1 else np.nan)
        if lk:
            last_lo[i], prev_lo[i] = lk[-1], (lk[-2] if len(lk) > 1 else np.nan)
    up_struct = (last_hi > prev_hi) & (last_lo > prev_lo)
    down_struct = (last_hi < prev_hi) & (last_lo < prev_lo)
    trend = np.where(up_struct, "up", np.where(down_struct, "down", ""))

    sr = sr_levels(df)
    below, above = sr["sr_next_below"].to_numpy(float), sr["sr_next_above"].to_numpy(float)
    below2, above2 = sr["sr_next_below2"].to_numpy(float), sr["sr_next_above2"].to_numpy(float)
    bull_candle = (_hammer(df, True) | _engulf(df, True)).to_numpy(bool) | (c >= l + (h - l) * 2 / 3)
    bear_candle = (_hammer(df, False) | _engulf(df, False)).to_numpy(bool) | (c <= h - (h - l) * 2 / 3)
    reclaim, reject_ = sr["sr_reclaim_up"].to_numpy(bool), sr["sr_reject_down"].to_numpy(bool)

    why = np.array([""] * n, dtype=object)
    note = np.array([""] * n, dtype=object)
    long_, short = np.zeros(n, bool), np.zeros(n, bool)
    stop_l, stop_s = np.full(n, np.nan), np.full(n, np.nan)
    t1l, t2l, t1s, t2s = (np.full(n, np.nan) for _ in range(4))

    def targets(i, entry, risk_px, sgn, extra=np.nan):
        """TP1 just before the next level, TP2 just before the one after (or the measured move)."""
        lv = (above[i], above2[i], extra) if sgn > 0 else (below[i], below2[i], extra)
        lv = sorted((x for x in lv if np.isfinite(x) and sgn * (x - entry) > 0.25 * size[i]),
                    key=lambda x: sgn * (x - entry))
        if not lv:
            return entry + sgn * risk_px, entry + sgn * 2 * risk_px
        t1 = lv[0] - sgn * 0.25 * size[i]
        t2 = lv[1] - sgn * 0.25 * size[i] if len(lv) > 1 else entry + sgn * max(2 * risk_px, sgn * (t1 - entry))
        return t1, (t2 if sgn * (t2 - t1) >= 0 else t1)

    def plan(i, sgn, stop_px, reason, measured=np.nan):
        risk_px = sgn * (c[i] - stop_px)
        if not (0.002 <= risk_px / c[i] <= 0.10):
            return
        t1, t2 = targets(i, c[i], risk_px, sgn, measured)
        if sgn * (t1 - c[i]) < MIN_RR * risk_px:
            return
        if sgn > 0:
            long_[i], stop_l[i], t1l[i], t2l[i] = True, stop_px, t1, t2
        else:
            short[i], stop_s[i], t1s[i], t2s[i] = True, stop_px, t1, t2
        why[i] = (why[i] + " + " if why[i] else "") + reason

    buf = STOP_BUFFER * size
    # 1. support bounce / resistance rejection: stop beyond the zone and the candle's wick
    for i in np.flatnonzero(reclaim & bull_candle):
        plan(i, 1, min(l[i], below[i] if np.isfinite(below[i]) else l[i]) - buf[i], "support bounce")
    for i in np.flatnonzero(reject_ & bear_candle):
        plan(i, -1, max(h[i], above[i] if np.isfinite(above[i]) else h[i]) + buf[i], "resistance rejection")

    # 2. break & retest of the last swing high / low: stop behind the level and beyond the retest wick
    broken_hi = broken_lo = None
    for i in range(1, n):
        if np.isfinite(last_hi[i - 1]) and c[i] > last_hi[i - 1] >= c[i - 1]:
            broken_hi = (last_hi[i - 1], i)
        if np.isfinite(last_lo[i - 1]) and c[i] < last_lo[i - 1] <= c[i - 1]:
            broken_lo = (last_lo[i - 1], i)
        if broken_hi and i - broken_hi[1] > RETEST_BARS:
            broken_hi = None
        if broken_lo and i - broken_lo[1] > RETEST_BARS:
            broken_lo = None
        if broken_hi and i > broken_hi[1] and not down_struct[i]:
            lvl = broken_hi[0]
            if l[i] <= lvl + 0.25 * size[i] and c[i] > lvl and c[i] > o[i]:
                plan(i, 1, min(l[i], lvl - 0.5 * size[i]) - buf[i], "break & retest")
                broken_hi = None
        if broken_lo and i > broken_lo[1] and not up_struct[i]:
            lvl = broken_lo[0]
            if h[i] >= lvl - 0.25 * size[i] and c[i] < lvl and c[i] < o[i]:
                plan(i, -1, max(h[i], lvl + 0.5 * size[i]) + buf[i], "break & retest")
                broken_lo = None

    # 3. shape breakouts the statistics back, entered on the retest of the broken line
    pf = pattern_frame(df)
    pend = None                                       # (direction, line, target, kind, bar, odds, target%)
    for i in range(n):
        b = pf.broke.iat[i]
        if b:
            k = pf.kind.iat[i]
            p, tg = odds(k, b, trend[i] or None)
            ok = p is not None and p >= MIN_ODDS and np.isfinite(pf.line.iat[i])
            pend = (b, pf.line.iat[i], pf.target.iat[i], k, i, p, tg) if ok else None
            if p is not None and not ok:
                note[i] = (f"{k.replace('_', ' ')} broke {b} - not traded: that direction won only {p:.0f}% of the "
                           "time on crypto")
            elif ok:
                note[i] = (f"{k.replace('_', ' ')} broke {b} ({p:.0f}% of these break {b} on crypto) - waiting for "
                           "the retest of the broken line")
            continue
        if not pend or i - pend[4] > RETEST_BARS:
            pend = None
            continue
        d, line, target, k, _, p, tg = pend
        measured = target if tg is not None and tg >= MIN_TARGET else np.nan
        reason = f"{k.replace('_', ' ')} breakout retest ({p:.0f}% break {d})"
        if d == "up" and not down_struct[i] and l[i] <= line + 0.5 * size[i] and c[i] > line and c[i] > o[i]:
            plan(i, 1, min(l[i], line - 0.5 * size[i]) - buf[i], reason, measured)
            pend = None
        elif d == "down" and not up_struct[i] and h[i] >= line - 0.5 * size[i] and c[i] < line and c[i] < o[i]:
            plan(i, -1, max(h[i], line + 0.5 * size[i]) + buf[i], reason, measured)
            pend = None

    # 4. swing failure at a level: stop beyond the sweep wick
    sfp_up = np.isfinite(last_lo) & (l < last_lo) & (c > last_lo) & np.isfinite(below) & \
        (np.abs(last_lo - below) <= 0.5 * size)
    sfp_dn = np.isfinite(last_hi) & (h > last_hi) & (c < last_hi) & np.isfinite(above) & \
        (np.abs(last_hi - above) <= 0.5 * size)
    for i in np.flatnonzero(sfp_up):
        plan(i, 1, l[i] - buf[i], "swing failure at support")
    for i in np.flatnonzero(sfp_dn):
        plan(i, -1, h[i] + buf[i], "swing failure at resistance")

    # trade a shape retest on its own, anything else only with confluence
    for i in np.flatnonzero(long_ | short):
        reasons = why[i].split(" + ")
        if not any("breakout retest" in r for r in reasons) and len(reasons) < CONFLUENCE:
            note[i] = (note[i] + " · " if note[i] else "") + f"{why[i]} alone - not traded (needs a second reason)"
            long_[i] = short[i] = False
    both = long_ & short
    long_ &= ~both
    short &= ~both
    return pd.DataFrame({
        "long": long_, "short": short, "why": why, "note": note,
        "stop_long": stop_l, "stop_short": stop_s, "tp1_long": t1l, "tp2_long": t2l, "tp1_short": t1s,
        "tp2_short": t2s, "pattern": pf.kind.to_numpy(object), "up_structure": up_struct,
        "down_structure": down_struct, "support": below, "resistance": above,
        "last_swing_low": last_lo, "last_swing_high": last_hi,
    }, index=df.index)


def signals(df: pd.DataFrame, direction: str = "both") -> pd.DataFrame:
    f = frame(df)
    c = df["close"].to_numpy(float)
    L, S = f["long"].to_numpy(bool), f["short"].to_numpy(bool)
    if direction == "long":
        S = np.zeros_like(S)
    elif direction == "short":
        L = np.zeros_like(L)
    stop = np.where(L, (c - f.stop_long) / c, np.where(S, (f.stop_short - c) / c, np.nan))
    tp1 = np.where(L, (f.tp1_long - c) / c, np.where(S, (c - f.tp1_short) / c, np.nan))
    tp2 = np.where(L, (f.tp2_long - c) / c, np.where(S, (c - f.tp2_short) / c, np.nan))
    return pd.DataFrame({"long_entry": L, "short_entry": S, "long_exit": False, "short_exit": False,
                         "stop_pct": stop, "tp1_pct": tp1, "tp2_pct": np.fmax(tp2, tp1),
                         "be_pct": tp1},                 # after TP1: stop to breakeven
                        index=df.index)


def checklist(df: pd.DataFrame, bar: int = -1, label: str | None = None, direction: str = "both") -> str:
    from .fmt import price
    f = frame(df)
    i = len(df) + bar if bar < 0 else bar
    r = f.iloc[i]
    c = float(df.close.iloc[i])
    struct = "higher highs & higher lows" if r.up_structure else "lower highs & lower lows" if r.down_structure \
        else "mixed / sideways"
    px = lambda x: price(x) if np.isfinite(x) else "-"
    lines = [f"[{label or 'S10 Price action'}] candle {df.index[i]:%Y-%m-%d %H:%M} UTC (close {price(c)})",
             f"Structure: {struct} · support {px(r.support)} · resistance {px(r.resistance)}"
             + (f" · shape: {str(r.pattern).replace('_', ' ')}" if r.pattern else "")]
    if r.note:
        lines.append(f"📐 {r.note}")
    for side, flag, stop, t1, t2 in (("LONG", r.long, r.stop_long, r.tp1_long, r.tp2_long),
                                     ("SHORT", r.short, r.stop_short, r.tp1_short, r.tp2_short)):
        if direction not in ("both", side.lower()):
            continue
        if flag:
            lines += ["", f"{side}: 🟢 TRADE - {r.why}",
                      f"  stop {price(stop)} ({abs(c - stop) / c * 100:.2f}% - beyond the structure, with playing room)",
                      f"  TP1 {price(t1)} (just before the next level, half) · TP2 {price(t2)} (half) · "
                      "stop to breakeven after TP1"]
        else:
            lines += ["", f"{side}: 🔴 NO TRADE"]
    lines += ["", "Looks for: support bounce / resistance rejection with a candle shape · break & retest of a swing "
                  "high/low · statistically-backed shape breakouts, entered on the retest · swing failure at a level"]
    return "\n".join(lines)
