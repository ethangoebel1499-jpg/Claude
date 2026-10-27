"""Measure entry quality independently of exits.

For each entry signal: starting at the next bar's open, does price reach the
target before the stop within `max_bars`? If both are touched in the same bar
the stop is assumed first (conservative). Distances are either fractions of
price ("pct") or multiples of ATR at entry ("atr", comparable across markets).

The baseline is the same test from every bar, i.e. entering at random. An
entry rule only has an edge if its hit rate is clearly above the baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(),
                    (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def outcomes(df: pd.DataFrame, entries: pd.Series, side: str, target: float, stop: float,
             mode: str = "pct", max_bars: int = 48) -> pd.Series:
    """1.0 = target first, 0.0 = stop first, NaN = neither within max_bars.
    Indexed by the signal bar."""
    o, h, lo = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low"))
    a = atr(df).to_numpy()
    sig_bars = np.flatnonzero(entries.to_numpy()[:-1])
    res = np.full(len(sig_bars), np.nan)
    for j, i in enumerate(sig_bars):
        e = o[i + 1]
        t_dist, s_dist = (target * e, stop * e) if mode == "pct" else (target * a[i], stop * a[i])
        if side == "long":
            tp, sl = e + t_dist, e - s_dist
        else:
            tp, sl = e - t_dist, e + s_dist
        for k in range(i + 1, min(i + 1 + max_bars, len(df))):
            hit_sl = lo[k] <= sl if side == "long" else h[k] >= sl
            hit_tp = h[k] >= tp if side == "long" else lo[k] <= tp
            if hit_sl:
                res[j] = 0.0
                break
            if hit_tp:
                res[j] = 1.0
                break
    return pd.Series(res, index=entries.index[sig_bars])


def summarize(df: pd.DataFrame, sig: pd.DataFrame, target: float, stop: float,
              mode: str = "pct", max_bars: int = 48, warmup: int = 200) -> dict:
    """Hit rate of long and short entries vs. entering on every bar."""
    sig = sig.iloc[warmup:]
    d = df.iloc[warmup:]
    longs = outcomes(d, sig["long_entry"], "long", target, stop, mode, max_bars)
    shorts = outcomes(d, sig["short_entry"], "short", target, stop, mode, max_bars)
    every = pd.Series(True, index=d.index)
    base_l = outcomes(d, every, "long", target, stop, mode, max_bars)
    base_s = outcomes(d, every, "short", target, stop, mode, max_bars)
    both = pd.concat([longs, shorts]).dropna()
    base = pd.concat([base_l, base_s]).dropna()
    return {
        "signals": int(len(both)),
        "hit_rate_pct": round(both.mean() * 100, 1) if len(both) else float("nan"),
        "baseline_pct": round(base.mean() * 100, 1),
        "edge_pts": round((both.mean() - base.mean()) * 100, 1) if len(both) else float("nan"),
        # Rough standard error of the hit rate, to judge whether the edge is noise.
        "std_err_pts": round(np.sqrt(base.mean() * (1 - base.mean()) / max(len(both), 1)) * 100, 1),
    }
