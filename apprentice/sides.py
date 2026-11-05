"""Side filters: when may the apprentice open longs, when shorts? 🎓 Learn picks one per timeframe and side.

A filter = a DAILY TREND switch x a CONFIRMATION:
  trend    none            - no switch
           50-day          - longs only while yesterday's daily close was above its 50-day average (shorts: below)
           200-day         - the same with the 200-day average (the classic bull / bear line)
           structure       - daily swings make higher highs & higher lows (shorts: lower highs & lower lows)
  confirm  none
           macd            - MACD above its signal line for longs, below for shorts
           mf_macd         - that, plus Market Cipher money flow rising (longs) / falling (shorts) over 3 candles

Only yesterday's closed daily candle is used (no peeking). The daily candles come from the coin's own daily history
(the live trader keeps only ~30 days of hourly candles - not enough for a 50 / 200-day average), cached for an hour;
without a symbol they're built from the chart itself.

Measured on your setups (BTC / ETH / SOL, fees in, avg R a trade, earlier / later half): daily shorts with no filter
-0.03R / +0.04R, with MACD below its signal +0.11R / +0.09R; 1h longs with MACD +0.08R / +0.30R; 1h and 4h shorts
didn't pass with any filter.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

TRENDS = ("none", "50-day", "200-day", "structure")
CONFIRMS = ("none", "macd", "mf_macd")
NAMES = {"none": "no daily switch", "50-day": "daily 50-day average", "200-day": "daily 200-day average",
         "structure": "daily structure"}
CONFIRM_NAMES = {"none": "no confirmation", "macd": "MACD vs its signal", "mf_macd": "money flow + MACD"}
_DAILY: dict = {}


def tag(df: pd.DataFrame, symbol: str, asset: str = "crypto") -> pd.DataFrame:
    """Tell the strategy which market a chart is (so it can fetch that market's daily candles)."""
    df.attrs["symbol"], df.attrs["asset"] = symbol, asset
    return df


def daily_candles(df: pd.DataFrame) -> pd.DataFrame:
    sym, asset = df.attrs.get("symbol"), df.attrs.get("asset", "crypto")
    if sym:
        key = (sym, asset, int(time.time() // 3600))
        if key not in _DAILY:
            try:
                from livepaper.history import history
                if len(_DAILY) > 64:
                    _DAILY.clear()
                _DAILY[key] = history(sym, asset, "1D")
            except Exception:
                _DAILY[key] = None
        if _DAILY[key] is not None and len(_DAILY[key]) > 60:
            return _DAILY[key]
    d = df.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return d


def _structure(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    from mcbot.patterns import swings
    hi, lo = swings(d, 3)
    h, l = d["high"].to_numpy(float), d["low"].to_numpy(float)
    n = len(d)
    up, dn = np.zeros(n, bool), np.zeros(n, bool)
    hk, lk = [], []
    hs, ls = {int(i) + 3: int(i) for i in hi}, {int(i) + 3: int(i) for i in lo}
    for i in range(n):
        if i in hs:
            hk.append(h[hs[i]])
        if i in ls:
            lk.append(l[ls[i]])
        if len(hk) > 1 and len(lk) > 1:
            up[i] = hk[-1] > hk[-2] and lk[-1] > lk[-2]
            dn[i] = hk[-1] < hk[-2] and lk[-1] < lk[-2]
    return up, dn


def trend_masks(df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{trend: (longs allowed, shorts allowed)} on each candle of df, from yesterday's closed daily candle."""
    d = daily_candles(df)
    c = d["close"]
    raw = {"50-day": (c > c.rolling(50).mean(), c < c.rolling(50).mean()),
           "200-day": (c > c.rolling(200).mean(), c < c.rolling(200).mean())}
    su, sd = _structure(d)
    raw["structure"] = (pd.Series(su, index=d.index), pd.Series(sd, index=d.index))
    out = {"none": (np.ones(len(df), bool), np.ones(len(df), bool))}
    for k, (u, w) in raw.items():
        masks = []
        for s in (u, w):
            s = s.astype(float)
            s.index = s.index + pd.Timedelta(days=1)          # known once the day has closed
            masks.append(s.reindex(df.index, method="ffill").fillna(0).astype(bool).to_numpy())
        out[k] = tuple(masks)
    return out


def confirm_masks(df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from mcbot.indicators import money_flow
    e = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
    m, s = e.to_numpy(float), e.ewm(span=9, adjust=False).mean().to_numpy(float)
    mf = np.asarray(money_flow(df), dtype=float)
    mf3 = np.r_[np.full(3, np.nan), mf[:-3]]
    one = np.ones(len(df), bool)
    return {"none": (one, one), "macd": (m > s, m < s), "mf_macd": ((mf > mf3) & (m > s), (mf < mf3) & (m < s))}


def masks(df: pd.DataFrame, filt: dict | None, side: str) -> np.ndarray:
    """Candles where this side may open a trade under a learned filter (None = not allowed)."""
    if not filt:
        return np.zeros(len(df), bool)
    k = 0 if side == "long" else 1
    return trend_masks(df)[filt.get("trend", "none")][k] & confirm_masks(df)[filt.get("confirm", "none")][k]


def describe(filt: dict | None) -> str:
    if not filt:
        return "off"
    bits = [NAMES[filt["trend"]] if filt["trend"] != "none" else "", CONFIRM_NAMES[filt["confirm"]]
            if filt["confirm"] != "none" else ""]
    return " + ".join(b for b in bits if b) or "any time (no filter needed)"
