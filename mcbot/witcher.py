# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.
#
# Python port of "The Witcher [30MIN]" TradingView script, (c) wielkieef, MPL-2.0.
# Ported for backtesting in this project. Indicator settings are the script's
# defaults. Differences from the original are noted inline.
"""S4 - The Witcher (30m) by wielkieef.

Long when all nine "basic" conditions agree, or when the Heikin Ashi scalping
signal fires while ADX agrees. Shorts mirror it. Entries alternate direction
(a long only after a short signal, or after a take-profit), take-profit +1.1%,
stop 8.2% (long) / 7.8% (short) from the script's alert messages.

Signals are evaluated on a closed candle and act on the next candle's open,
like everything else in this project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import heikin_ashi
from .fmt import price

TAKE_PROFIT = 0.011
STOP_LONG = 0.082
STOP_SHORT = 0.078

LONG_LABELS = {
    "adx": "ADX above 17 and +DI over -DI (trend strength)",
    "rsi": "RSI(35, lows) below 70",
    "mfi": "Money flow index above +10",
    "jma": "Price above its 22-bar midpoint average",
    "sar": "Parabolic SAR below price",
    "trend": "Donchian(13) midline rising",
    "momentum": "Smoothed RSI above 50 and OBV averages rising",
    "ma_speed": "72-bar volume-weighted MA rising",
    "range": "Range filter pointing up",
}
SHORT_LABELS = {
    "adx": "ADX above 17 and -DI over +DI (trend strength)",
    "rsi": "RSI(35, lows) above 30",
    "mfi": "Money flow index below -10",
    "jma": "Price below its 22-bar midpoint average",
    "sar": "Parabolic SAR above price",
    "trend": "Donchian(13) midline falling",
    "momentum": "Smoothed RSI below 50 and OBV averages falling",
    "ma_speed": "72-bar volume-weighted MA falling",
    "range": "Range filter pointing down",
}


# --- Pine helpers -------------------------------------------------------------

def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False).mean()


def _rsi(s: pd.Series, n: int) -> pd.Series:
    d = s.diff()
    up, down = _rma(d.clip(lower=0), n), _rma(-d.clip(upper=0), n)
    return pd.Series(np.where(down == 0, 100.0, np.where(up == 0, 0.0, 100 - 100 / (1 + up / down))),
                     index=s.index)


def _wma(s: pd.Series, n: int) -> pd.Series:
    """Linearly weighted moving average (Pine wma), via convolution for speed."""
    w = np.arange(1, n + 1, dtype=float)
    x = s.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1:] = np.convolve(x, w[::-1], mode="valid") / w.sum()
    return pd.Series(out, index=s.index)


def _hma(s: pd.Series, n: int) -> pd.Series:
    return _wma(2 * _wma(s, n // 2) - _wma(s, n), int(round(np.sqrt(n))))


def _dema(s: pd.Series, n: int) -> pd.Series:
    e = s.ewm(span=n, adjust=False).mean()
    return 2 * e - e.ewm(span=n, adjust=False).mean()


def _psar(h: np.ndarray, lo: np.ndarray, start: float, inc: float, maximum: float) -> np.ndarray:
    """Wilder's parabolic SAR (Pine's sar())."""
    n = len(h)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    up = h[1] >= h[0]
    sar, ep, af = (lo[0], h[0], start) if up else (h[0], lo[0], start)
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, lo[i - 1], lo[i - 2] if i >= 2 else lo[i - 1])
            if lo[i] < sar:
                up, sar, ep, af = False, ep, lo[i], start
            elif h[i] > ep:
                ep, af = h[i], min(af + inc, maximum)
        else:
            sar = max(sar, h[i - 1], h[i - 2] if i >= 2 else h[i - 1])
            if h[i] > sar:
                up, sar, ep, af = True, ep, h[i], start
            elif lo[i] < ep:
                ep, af = lo[i], min(af + inc, maximum)
        out[i] = sar
    return out


def _bars_since(cond: np.ndarray) -> np.ndarray:
    out = np.full(len(cond), np.inf)
    last = None
    for i, c in enumerate(cond):
        if c:
            last = i
        if last is not None:
            out[i] = i - last
    return out


# --- Conditions ---------------------------------------------------------------

def witcher_conditions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """The nine basic conditions (long and short) plus the scalp signals."""
    o, h, lo, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
    hl2, hlc3 = (h + lo) / 2, (h + lo + c) / 3

    # ADX (classic), length 18, threshold 17
    up, down = h.diff(), -lo.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    atr = _rma(tr, 18)
    di_p = (100 * _rma(pd.Series(plus_dm, index=df.index), 18) / atr).ffill()
    di_m = (100 * _rma(pd.Series(minus_dm, index=df.index), 18) / atr).ffill()
    s = (di_p + di_m).replace(0, 1)
    adx = 100 * _rma((di_p - di_m).abs() / s, 18)
    l_adx, s_adx = (di_p > di_m) & (adx > 17), (di_p < di_m) & (adx > 17)

    # RSI(35) on lows
    rsi3 = _rsi(lo, 35)

    # Money flow index (58), scaled
    ch = hlc3.diff()
    mf_up = (v * hlc3.where(ch > 0, 0.0)).rolling(58).sum()
    mf_dn = (v * hlc3.where(ch < 0, 0.0)).rolling(58).sum()
    mf = 100 - 100 / (1 + mf_up / mf_dn.replace(0, np.nan))
    mfi = (mf.fillna(100) - 50) * 3

    # "JMA": previous bar's hlc3 vs the midpoint of it and its value 22 bars earlier
    src0 = hlc3.shift(1)
    jsa = (src0 + src0.shift(22)) / 2

    # Parabolic SAR
    psar = pd.Series(_psar(h.to_numpy(float), lo.to_numpy(float), 0.007, 0.018, 0.05), index=df.index)

    # Donchian(13) midline: current (offset 1) vs previous (offset 12)
    basis = (h.rolling(13).max() + lo.rolling(13).min()) / 2
    purple, orange = basis.shift(1), basis.shift(12)

    # Momentum: HMA-smoothed RSI(56) via DEMA(9), and OBV EMAs
    dema1 = _dema(_hma(_rsi(c, 56), 40) - 50, 9)
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    e1, e2 = obv.ewm(span=30, adjust=False).mean(), obv.ewm(span=12, adjust=False).mean()

    # 72-bar volume-weighted MA speed on hl2
    vwma = (hl2 * v).rolling(72).sum() / v.rolling(72).sum()
    ma_speed = vwma / vwma.shift() - 1

    # Range filter (22, mult 3) on hl2
    src = hl2.to_numpy(float)
    var = hl2.rolling(22).var(ddof=0).to_numpy() * 3
    sma = hl2.rolling(22).mean().to_numpy()
    cma = np.full(len(df), np.nan)
    cts = np.full(len(df), np.nan)
    for i in range(len(df)):
        pc = cma[i - 1] if i else np.nan
        pt = cts[i - 1] if i else np.nan
        secma = (sma[i] - pc) ** 2 if np.isfinite(sma[i]) and np.isfinite(pc) else 0.0
        sects = (src[i] - pt) ** 2 if np.isfinite(pt) else 0.0
        ka = 1 - var[i] / secma if np.isfinite(var[i]) and var[i] < secma else 0.0
        kb = 1 - var[i] / sects if np.isfinite(var[i]) and var[i] < sects else 0.0
        base_c = pc if np.isfinite(pc) else src[i]   # Pine: nz(cma[1], src)
        base_t = pt if np.isfinite(pt) else src[i]
        cma[i] = ka * sma[i] + (1 - ka) * base_c if ka else base_c
        cts[i] = kb * src[i] + (1 - kb) * base_t
    cma_s, cts_s = pd.Series(cma, index=df.index), pd.Series(cts, index=df.index)

    long_c = pd.DataFrame({
        "adx": l_adx, "rsi": rsi3 < 70, "mfi": mfi > 10, "jma": src0 > jsa, "sar": psar < c,
        "trend": purple > orange, "momentum": ((e1 > e1.shift()) | (e2 > e2.shift())) & (dema1 > 0),
        "ma_speed": ma_speed > 0, "range": cts_s > cma_s,
    }, index=df.index).fillna(False).astype(bool)
    short_c = pd.DataFrame({
        "adx": s_adx, "rsi": rsi3 > 30, "mfi": mfi < -10, "jma": src0 < jsa, "sar": psar > c,
        "trend": purple < orange, "momentum": ((e1 < e1.shift()) | (e2 < e2.shift())) & (dema1 < 0),
        "ma_speed": ma_speed < 0, "range": cts_s < cma_s,
    }, index=df.index).fillna(False).astype(bool)

    # Scalping signal on Heikin Ashi candles
    ha = heikin_ashi(df)
    hc, ho, hh, hl = ha["close"], ha["open"], ha["high"], ha["low"]
    fast, med = hc.ewm(span=10, adjust=False).mean(), hc.ewm(span=120, adjust=False).mean()
    pac_c = hc.ewm(span=6, adjust=False).mean()
    pac_l = hl.ewm(span=6, adjust=False).mean()
    pac_u = hh.ewm(span=6, adjust=False).mean()
    trend_dir = np.where((fast > med) & (pac_l > med), 1, np.where((fast < med) & (pac_u < med), -1, 0))
    exit_u = ((ho < pac_u) & (hc > pac_u)).to_numpy() & (_bars_since((hc < pac_c).to_numpy()) <= 5)
    exit_l = ((ho > pac_l) & (hc < pac_l)).to_numpy() & (_bars_since((hc > pac_c).to_numpy()) <= 5)
    buy, sell = (trend_dir == 1) & exit_u, (trend_dir == -1) & exit_l
    below, above = (hc < pac_c).to_numpy(), (hc > pac_c).to_numpy()
    td = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        p = td[i - 1] if i else 0
        td[i] = 0 if (p == 1 and below[i]) or (p == -1 and above[i]) else \
            1 if p == 0 and buy[i] else -1 if p == 0 and sell[i] else p
    prev = np.r_[0, td[:-1]]
    l_scalp = pd.Series((prev == 0) & (td == 1), index=df.index)
    s_scalp = pd.Series((prev == 0) & (td == -1), index=df.index)
    return long_c, short_c, l_scalp, s_scalp


def witcher_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Entry signals on the bar where the script's conditions are known.

    The script fires its alert one bar later (it reads longCond[1]); acting on
    the next open after this signal bar is the same moment. Direction
    alternates: a long only after the last signal was a short. (The original
    also allows a repeat entry right after a take-profit; that re-entry is not
    modelled here.)
    """
    long_c, short_c, l_scalp, s_scalp = witcher_conditions(df)
    long_cond = (long_c.all(axis=1) | (l_scalp & long_c["adx"])).to_numpy()
    short_cond = (short_c.all(axis=1) | (s_scalp & short_c["adx"])).to_numpy()
    cond_ini = 0
    le = np.zeros(len(df), dtype=bool)
    se = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        le[i] = long_cond[i] and cond_ini == -1
        se[i] = short_cond[i] and cond_ini == 1
        cond_ini = 1 if long_cond[i] else -1 if short_cond[i] else cond_ini
    le_s, se_s = pd.Series(le, index=df.index), pd.Series(se, index=df.index)
    # An opposite signal closes the trade (the script's alerts flip the position).
    return pd.DataFrame({"long_entry": le_s, "long_exit": se_s, "short_entry": se_s,
                         "short_exit": le_s}, index=df.index)


def witcher_checklist(df: pd.DataFrame, bar: int = -1, label: str = "S4 - The Witcher (30m)") -> str:
    long_c, short_c, l_scalp, s_scalp = witcher_conditions(df)
    sig = witcher_signals(df)
    real = df.attrs.get("real_dates")
    ts = real[bar] if real is not None else df.index[bar]
    lines = [f"[{label}] Checklist for candle closed {ts:%Y-%m-%d %H:%M} UTC (close {price(df['close'].iloc[bar])})"]
    for side, cond, scalp, labels, key in (("LONG", long_c, l_scalp, LONG_LABELS, "long_entry"),
                                           ("SHORT", short_c, s_scalp, SHORT_LABELS, "short_entry")):
        row = cond.iloc[bar]
        go = bool(sig[key].iloc[bar])
        lines += ["", f"{side}: {'🟢 TRADE' if go else '🔴 NO TRADE'}  ({int(row.sum())}/{len(row)} basic yes)"]
        for k, lab in labels.items():
            lines.append(f"  {'🟢 YES' if row[k] else '🔴 NO '}  {lab}")
        lines.append(f"  {'🟢 YES' if scalp.iloc[bar] else '🔴 NO '}  Heikin Ashi scalp signal (counts on its own with ADX)")
        if go:
            stop = STOP_LONG if side == "LONG" else STOP_SHORT
            d = 1 if side == "LONG" else -1
            px = df["close"].iloc[bar]
            lines += ["  Exit plan (from the script):",
                      f"    Take profit {price(px * (1 + d * TAKE_PROFIT))}  (+1.1%)",
                      f"    Stop        {price(px * (1 - d * stop))}  ({stop * 100:.1f}%)",
                      "    An opposite signal flips the position"]
    return "\n".join(lines)
