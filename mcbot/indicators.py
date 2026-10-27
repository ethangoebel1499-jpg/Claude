"""Indicators for the top-down confluence strategy.

* Market Cipher B pieces (open-source VuManChu approximation, since Market
  Cipher itself is closed source): WaveTrend wt1/wt2, green/red dots, and the
  RSI+MFI money flow.
* Naked-chart pieces: support/resistance zones from the 4x higher timeframe
  (top-down analysis) and the 21 EMA (Market Cipher A ribbon 5).
* Heikin Ashi candles (optional) for the Market Cipher B calculations.

Values on bar i never depend on bars after i.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OB_LEVEL = 53
OS_LEVEL = -53


def ema(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False).mean()


def sma(s: pd.Series, length: int) -> pd.Series:
    return s.rolling(length).mean()


def wavetrend(df: pd.DataFrame, channel_len: int = 9, average_len: int = 12,
              ma_len: int = 3) -> tuple[pd.Series, pd.Series]:
    src = (df["high"] + df["low"] + df["close"]) / 3
    esa = ema(src, channel_len)
    de = ema((src - esa).abs(), channel_len)
    ci = (src - esa) / (0.015 * de.replace(0, np.nan))
    wt1 = ema(ci.fillna(0), average_len)
    wt2 = sma(wt1, ma_len)
    return wt1, wt2


def money_flow(df: pd.DataFrame, period: int = 60, multiplier: float = 150,
               pos_y: float = 2.5) -> pd.Series:
    """RSI+MFI area: positive = buying pressure, negative = selling pressure."""
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    raw = ((df["close"] - df["open"]) / rng * multiplier).fillna(0)
    return sma(raw, period) - pos_y


def sr_levels(df: pd.DataFrame, factor: int = 4, zone_pct: float = 0.004, min_touches: int = 2,
              max_levels: int = 30) -> pd.DataFrame:
    """Support/resistance zones from the higher timeframe (top-down analysis).

    Turning points (fractal highs/lows) on `factor`x candles are clustered: a
    turning point becomes a level once at least `min_touches` turning points sit
    within `zone_pct` of each other (price reacted there repeatedly). Each bar
    only uses turning points confirmed by a *closed* higher-timeframe candle.

    Returns per bar:
      sr_reclaim_up   low dipped into / below a zone and the candle closed above it
                      (bounce off support, or confirmed break above resistance)
      sr_reject_down  high pushed into / above a zone and the candle closed below it
      sr_next_above / sr_next_below   nearest level above / below the close
    """
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
        ts = df.index.asi8
        keys = ts // (int(np.median(np.diff(ts))) * factor)
    else:
        keys = np.arange(len(df)) // factor
    group = pd.factorize(keys)[0]
    htf = df.groupby(group).agg(high=("high", "max"), low=("low", "min"))
    hh, ll = htf["high"].to_numpy(), htf["low"].to_numpy()

    # Pivot at HTF candle j is confirmed when candle j+2 closes, usable from group j+3.
    pivots_by_group: dict[int, list[float]] = {}
    for j in range(2, len(htf) - 2):
        found = []
        if hh[j] > max(hh[j - 2], hh[j - 1], hh[j + 1], hh[j + 2]):
            found.append(hh[j])
        if ll[j] < min(ll[j - 2], ll[j - 1], ll[j + 1], ll[j + 2]):
            found.append(ll[j])
        if found:
            pivots_by_group.setdefault(j + 3, []).extend(found)

    o_low, o_high, close = (df[k].to_numpy(dtype=float) for k in ("low", "high", "close"))
    n = len(df)
    reclaim = np.zeros(n, dtype=bool)
    reject = np.zeros(n, dtype=bool)
    above = np.full(n, np.nan)
    below = np.full(n, np.nan)
    pivots: list[float] = []
    levels = np.array([])
    current = -1
    for i in range(n):
        g = group[i]
        if g != current:
            current = g
            new = pivots_by_group.get(g)
            if new:
                pivots.extend(new)
                pivots = pivots[-200:]
                p = np.array(pivots)
                # A level = pivot with >= min_touches pivots (itself included) in its zone.
                counts = np.array([(np.abs(p - x) <= x * zone_pct).sum() for x in p])
                cand = np.sort(p[counts >= min_touches])
                merged: list[float] = []
                for x in cand:  # merge overlapping zones into one level
                    if merged and x - merged[-1] <= merged[-1] * zone_pct:
                        merged[-1] = (merged[-1] + x) / 2
                    else:
                        merged.append(x)
                levels = np.array(merged[-max_levels:])
        if len(levels):
            top = levels * (1 + zone_pct / 2)
            bot = levels * (1 - zone_pct / 2)
            reclaim[i] = bool(((o_low[i] <= top) & (close[i] > top)).any())
            reject[i] = bool(((o_high[i] >= bot) & (close[i] < bot)).any())
            up = levels[levels > close[i]]
            dn = levels[levels < close[i]]
            above[i] = up.min() if len(up) else np.nan
            below[i] = dn.max() if len(dn) else np.nan
    return pd.DataFrame({"sr_reclaim_up": reclaim, "sr_reject_down": reject,
                         "sr_next_above": above, "sr_next_below": below}, index=df.index)


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Heikin Ashi candles: smoothed averages that filter out noise."""
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    ha_close = (o + h + l + c) / 4
    ha_open = np.empty(len(df))
    if len(df):
        ha_open[0] = (o[0] + c[0]) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
    out = df.copy()
    out["open"], out["close"] = ha_open, ha_close
    out["high"] = np.maximum.reduce([h, ha_open, ha_close])
    out["low"] = np.minimum.reduce([l, ha_open, ha_close])
    return out


def market_cipher(df: pd.DataFrame, ob_level: float = OB_LEVEL,
                  os_level: float = OS_LEVEL, use_heikin_ashi: bool = False,
                  htf_factor: int = 4) -> pd.DataFrame:
    """Compute every input the strategy uses and return them alongside df.

    Support/resistance and the 21 EMA always use the real candles (the "naked
    chart"). With use_heikin_ashi, the Market Cipher B parts are calculated on
    Heikin Ashi candles and the returned open/high/low/close are the Heikin Ashi
    values. Trades always fill at real prices.
    """
    real_dates = df.attrs.get("real_dates")
    naked = sr_levels(df, htf_factor)
    naked["real_close"] = df["close"]
    naked["real_ema21"] = ema(df["close"], 21)
    if use_heikin_ashi:
        df = heikin_ashi(df)
    out = df.copy()
    wt1, wt2 = wavetrend(df)
    out["wt1"], out["wt2"] = wt1, wt2
    cross_up = (wt1 > wt2) & (wt1.shift() <= wt2.shift())
    cross_down = (wt1 < wt2) & (wt1.shift() >= wt2.shift())
    out["wt_cross_up"] = cross_up
    out["wt_cross_down"] = cross_down
    out["green_dot"] = cross_up & (wt2 <= os_level)
    out["red_dot"] = cross_down & (wt2 >= ob_level)
    out["money_flow"] = money_flow(df)
    out = out.join(naked)
    out.attrs["htf_factor"] = htf_factor
    if real_dates is not None:
        out.attrs["real_dates"] = real_dates
    return out
