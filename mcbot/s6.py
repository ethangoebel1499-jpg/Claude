"""S6 - 1-minute Bollinger Bands + Awesome Oscillator divergence scalp (Jason Casper).

  long:  price was inside the bands, then touches / pierces the lower band
         while the Awesome Oscillator shows a bullish divergence (price lower
         low than at the last AO trough, AO higher low)
  short: the mirror at the upper band with a bearish divergence
  stop:  just beyond the last low (high), a little breathing room
  take profit: the opposite band, re-placed every bar as the bands move
         (he drags the limit order to meet price)

He enters the moment price pierces the band; here signals use the closed 1m
candle and fill at the next open. AO = SMA(5) - SMA(34) of hl2; AO troughs and
peaks are 3-bar swing points. Bollinger Bands 20, 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import Result, Trade, _stats
from .divergence import waves
from .fmt import price

BB_LEN, BB_MULT = 20, 2.0
SWING = 3
MAX_GAP = 60          # the earlier AO trough must be within the last hour
STOP_ROOM = 0.0005    # 0.05% beyond the last low / high


def awesome(df: pd.DataFrame) -> pd.Series:
    hl2 = (df["high"] + df["low"]) / 2
    return hl2.rolling(5).mean() - hl2.rolling(34).mean()


def signals(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    mid = c.rolling(BB_LEN).mean()
    sd = c.rolling(BB_LEN).std(ddof=0)
    upper, lower = mid + BB_MULT * sd, mid - BB_MULT * sd
    ao = awesome(df)
    win = 2 * SWING + 1
    # Swing low / high of the AO, confirmed SWING bars later.
    trough = (ao.shift(SWING) == ao.rolling(win).min()).fillna(False)
    peak = (ao.shift(SWING) == ao.rolling(win).max()).fillna(False)
    w = waves(ao, df["low"], df["high"], trough, peak)
    n = pd.Series(np.arange(len(df)), index=df.index)
    touch_low = (df["low"] <= lower) & (c.shift() > lower.shift())
    touch_high = (df["high"] >= upper) & (c.shift() < upper.shift())
    bull = (w.run_min_px < w.t1_px) & (w.run_min_osc > w.t1_osc) & (w.t1_osc < 0) & (n - w.t1_i <= MAX_GAP)
    bear = (w.run_max_px > w.p1_px) & (w.run_max_osc < w.p1_osc) & (w.p1_osc > 0) & (n - w.p1_i <= MAX_GAP)
    return pd.DataFrame({
        "long_entry": (touch_low & bull).fillna(False), "short_entry": (touch_high & bear).fillna(False),
        "touch_low": touch_low, "touch_high": touch_high, "bull_div": bull.fillna(False),
        "bear_div": bear.fillna(False), "upper": upper, "lower": lower, "ao": ao,
        "long_stop": w.run_min_px * (1 - STOP_ROOM), "short_stop": w.run_max_px * (1 + STOP_ROOM),
    }, index=df.index)


def backtest(df: pd.DataFrame, fr: pd.DataFrame, *, allow_short: bool = True, fee: float = 0.0,
             start_equity: float = 10_000.0) -> Result:
    """Full position per trade (no leverage). Stop checked before target in a bar."""
    o, h, lo, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    le, se = fr.long_entry.to_numpy(), fr.short_entry.to_numpy()
    up, dn = fr.upper.to_numpy(float), fr.lower.to_numpy(float)
    ls, ss = fr.long_stop.to_numpy(float), fr.short_stop.to_numpy(float)
    idx = df.index
    equity, trades = start_equity, []
    curve = np.empty(len(df))
    side, entry, stop, t = 0, 0.0, 0.0, None
    for i in range(1, len(df)):
        p = i - 1
        if side == 0:
            if le[p] and ls[p] < o[i]:
                side, entry, stop = 1, o[i], ls[p]
            elif allow_short and se[p] and ss[p] > o[i]:
                side, entry, stop = -1, o[i], ss[p]
            if side:
                t = Trade("long" if side == 1 else "short", idx[i], entry)
        if side:
            target = up[p] if side == 1 else dn[p]
            exit_px = reason = None
            if (lo[i] <= stop) if side == 1 else (h[i] >= stop):
                exit_px, reason = (min(stop, o[i]) if side == 1 else max(stop, o[i])), "stop"
            elif (h[i] >= target) if side == 1 else (lo[i] <= target):
                exit_px, reason = (max(target, o[i]) if side == 1 else min(target, o[i])), "target1"
            if exit_px is not None:
                t.pnl_pct = side * (exit_px - entry) / entry - 2 * fee
                t.exit_time, t.exit_price, t.reason, t.remaining = idx[i], exit_px, reason, 0.0
                t.targets_hit = int(reason == "target1")
                equity *= 1 + t.pnl_pct
                trades.append(t)
                side = 0
        curve[i] = equity * (1 + side * (c[i] - entry) / entry) if side else equity
    curve[0] = start_equity
    eq = pd.Series(curve, index=idx)
    return Result(trades, eq, _stats(df, trades, eq, start_equity))


def checklist(df: pd.DataFrame, bar: int = -1, label: str = "S6 - 1m Bollinger + AO divergence") -> str:
    fr = signals(df)
    r = fr.iloc[bar]
    lines = [f"[{label}] Checklist for candle {df.index[bar]:%Y-%m-%d %H:%M} UTC (close {price(df['close'].iloc[bar])})"]
    for side, touch, div, go, lab in (("LONG", r.touch_low, r.bull_div, r.long_entry, "lower"),
                                      ("SHORT", r.touch_high, r.bear_div, r.short_entry, "upper")):
        lines += ["", f"{side}: {'🟢 TRADE' if go else '🔴 NO TRADE'}",
                  f"  {'🟢 YES' if touch else '🔴 NO '}  Price came from inside and touched the {lab} band",
                  f"  {'🟢 YES' if div else '🔴 NO '}  Awesome Oscillator {'bullish' if side == 'LONG' else 'bearish'} divergence"]
        if go:
            stop = r.long_stop if side == "LONG" else r.short_stop
            tgt = r.upper if side == "LONG" else r.lower
            lines.append(f"  Exit plan: stop {price(stop)}, take profit at the {'upper' if side == 'LONG' else 'lower'} "
                         f"band (now {price(tgt)}, moves with the band)")
    return "\n".join(lines)
