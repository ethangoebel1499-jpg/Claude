"""Market Cipher style indicators.

Market Cipher A/B are closed-source TradingView scripts. These are the
widely used open-source approximations (VuManChu Cipher A/B), which follow
the same components and default settings:

* WaveTrend oscillator (wt1 / wt2) with overbought / oversold levels
* Green dots (bullish WT cross while oversold) and red dots (bearish cross
  while overbought)
* RSI + MFI "money flow" area
* RSI and Stochastic RSI
* WaveTrend divergences (fractal based)
* Cipher A EMA ribbon (5/11/15/18/21/24/28/34) for trend direction

All functions take an OHLCV DataFrame with columns
open, high, low, close, volume and only use data up to each bar, so values
on bar i never depend on bars after i.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OB_LEVEL = 53
OS_LEVEL = -53
RIBBON_LENGTHS = (5, 11, 15, 18, 21, 24, 28, 34)


def ema(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False).mean()


def sma(s: pd.Series, length: int) -> pd.Series:
    return s.rolling(length).mean()


def rsi(s: pd.Series, length: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(loss.notna())


def stoch_rsi(s: pd.Series, rsi_len: int = 14, stoch_len: int = 14,
              k_smooth: int = 3, d_smooth: int = 3) -> tuple[pd.Series, pd.Series]:
    r = rsi(s, rsi_len)
    lo = r.rolling(stoch_len).min()
    hi = r.rolling(stoch_len).max()
    stoch = 100 * (r - lo) / (hi - lo).replace(0, np.nan)
    k = sma(stoch, k_smooth)
    d = sma(k, d_smooth)
    return k, d


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


def _fractal_top(s: pd.Series) -> pd.Series:
    # Peak at bar i-2, confirmed on bar i.
    return (s.shift(4) < s.shift(2)) & (s.shift(3) < s.shift(2)) & \
           (s.shift(2) > s.shift(1)) & (s.shift(2) > s)


def _fractal_bot(s: pd.Series) -> pd.Series:
    return (s.shift(4) > s.shift(2)) & (s.shift(3) > s.shift(2)) & \
           (s.shift(2) < s.shift(1)) & (s.shift(2) < s)


def divergences(df: pd.DataFrame, osc: pd.Series, top_limit: float = 45,
                bot_limit: float = -65) -> tuple[pd.Series, pd.Series]:
    """Regular WaveTrend divergences, flagged on the bar they are confirmed.

    Bearish: oscillator makes a lower high (above top_limit) while price makes
    a higher high. Bullish: oscillator makes a higher low (below bot_limit)
    while price makes a lower low.
    """
    tops = _fractal_top(osc).to_numpy()
    bots = _fractal_bot(osc).to_numpy()
    o = osc.to_numpy()
    hi = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    n = len(df)
    bear = np.zeros(n, dtype=bool)
    bull = np.zeros(n, dtype=bool)
    last_top = last_bot = None  # (osc value, price) at previous fractal
    for i in range(4, n):
        p = i - 2  # pivot bar
        if tops[i] and o[p] >= top_limit:
            if last_top is not None and o[p] < last_top[0] and hi[p] > last_top[1]:
                bear[i] = True
            last_top = (o[p], hi[p])
        if bots[i] and o[p] <= bot_limit:
            if last_bot is not None and o[p] > last_bot[0] and lo[p] < last_bot[1]:
                bull[i] = True
            last_bot = (o[p], lo[p])
    return pd.Series(bull, index=df.index), pd.Series(bear, index=df.index)


def mf_convergence(df: pd.DataFrame, mf: pd.Series, window: int = 20) -> pd.DataFrame:
    """Compare price with money flow over the last `window` bars vs the window before.

    Divergence (they disagree, a reversal warning):
      bull_mf_div: price made a lower low, money flow made a higher low
      bear_mf_div: price made a higher high, money flow made a lower high
    Convergence (they agree, the move is confirmed):
      bull_mf_conv: price and money flow both made a higher high
      bear_mf_conv: price and money flow both made a lower low
    """
    def now(s, fn):
        return getattr(s.rolling(window), fn)()

    def prev(s, fn):
        return now(s, fn).shift(window)

    price_lower_low = now(df["low"], "min") < prev(df["low"], "min")
    price_higher_high = now(df["high"], "max") > prev(df["high"], "max")
    mf_lower_low = now(mf, "min") < prev(mf, "min")
    mf_higher_high = now(mf, "max") > prev(mf, "max")

    return pd.DataFrame({
        "bull_mf_div": price_lower_low & ~mf_lower_low,
        "bear_mf_div": price_higher_high & ~mf_higher_high,
        "bull_mf_conv": price_higher_high & mf_higher_high,
        "bear_mf_conv": price_lower_low & mf_lower_low,
    }, index=df.index)


def ema_ribbon(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({f"ema{n}": ema(df["close"], n) for n in RIBBON_LENGTHS},
                        index=df.index)


def market_cipher(df: pd.DataFrame, ob_level: float = OB_LEVEL,
                  os_level: float = OS_LEVEL) -> pd.DataFrame:
    """Compute all Cipher A/B components and return them alongside df."""
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
    out = out.join(mf_convergence(df, out["money_flow"]))
    out["rsi"] = rsi(df["close"])
    out["stoch_k"], out["stoch_d"] = stoch_rsi(df["close"])
    out["bull_div"], out["bear_div"] = divergences(df, wt2)

    ribbon = ema_ribbon(df)
    out = out.join(ribbon)
    out["ribbon_bull"] = ribbon["ema5"] > ribbon["ema34"]
    out["ema200"] = ema(df["close"], 200)
    return out
