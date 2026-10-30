# Python port of the "GRIND BTC [45MIN]" TradingView strategy by wielkieef, published
# as an open-source TradingView script (TradingView's default licence for open-source
# scripts is the Mozilla Public License 2.0, https://mozilla.org/MPL/2.0/).
# Ported for backtesting in this project with the script's default settings.
"""S3 - GRIND BTC (45m) by wielkieef.

Long when all nine conditions agree (Masanakamura ADX, RSI(100) on opens,
MFI, volume > 1.3x its 51-bar SMA, 50-bar midpoint average, fast MACD 3/4/7,
rising 55-bar SMA, Parabolic SAR, cumulative volume delta above its 5-bar
SMA), or when its Heikin Ashi scalp signal fires while ADX agrees. Shorts
mirror it. Like the script's backtest, it enters whenever the conditions hold
and an opposite signal reverses the position.

Exits (script defaults): stop 9%; take profits +1.7% / +2.3% / +3.4% sized
33% / 66% / 100% of the *original* position, which TradingView fills as 33%,
66% and the remaining 1%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import heikin_ashi
from .witcher import _bars_since, _psar, _rma
from .fmt import price

STOP = 0.09
TAKE_PROFITS = [(0.017, 0.33), (0.023, 0.66), (0.034, 0.01)]
SCRIPT_FEE = 0.0004   # commission_value = 0.04 (%)

LONG_LABELS = {
    "adx": "ADX (Masanakamura, 9) above 12 and +DI over -DI",
    "rsi": "RSI(100, opens) below 70",
    "mfi": "Money flow index above +8",
    "volume": "Volume above 1.3x its 51-bar average",
    "jma": "Close above its 50-bar midpoint average",
    "macd": "MACD (3/4/7) above its signal",
    "ma_speed": "55-bar average rising",
    "sar": "Parabolic SAR below price",
    "delta": "Cumulative volume delta above its 5-bar average",
}
SHORT_LABELS = {
    "adx": "ADX (Masanakamura, 9) above 12 and -DI over +DI",
    "rsi": "RSI(100, opens) above 30",
    "mfi": "Money flow index below -8",
    "volume": "Volume above 1.3x its 51-bar average",
    "jma": "Close below its 50-bar midpoint average",
    "macd": "MACD (3/4/7) below its signal",
    "ma_speed": "55-bar average falling",
    "sar": "Parabolic SAR above price",
    "delta": "Cumulative volume delta below its 5-bar average",
}


def _adx_masanakamura(h, lo, c, n: int):
    h, lo, c = (np.asarray(x, dtype=float) for x in (h, lo, c))
    size = len(h)
    str_ = dmp = dmm = 0.0
    dip = np.full(size, np.nan)
    dim = np.full(size, np.nan)
    for i in range(size):
        ph = h[i - 1] if i else 0.0      # Pine nz(high[1]) etc.
        pl = lo[i - 1] if i else 0.0
        pc = c[i - 1] if i else 0.0
        tr = max(h[i] - lo[i], abs(h[i] - pc), abs(lo[i] - pc))
        up, down = h[i] - ph, pl - lo[i]
        plus = max(up, 0.0) if up > down else 0.0
        minus = max(down, 0.0) if down > up else 0.0
        str_ = str_ - str_ / n + tr
        dmp = dmp - dmp / n + plus
        dmm = dmm - dmm / n + minus
        if str_:
            dip[i], dim[i] = dmp / str_ * 100, dmm / str_ * 100
    with np.errstate(invalid="ignore", divide="ignore"):
        dx = np.abs(dip - dim) / (dip + dim) * 100
    adx = pd.Series(dx).rolling(n).mean().to_numpy()
    return dip, dim, adx


def _bull_bear_power(o, h, lo, c):
    """Bull and Bear Balance (Vadim Gimelfarb, S&C Oct 2003), as in the script."""
    o, h, lo, c = (np.asarray(x, dtype=float) for x in (o, h, lo, c))
    c1 = np.r_[np.nan, c[:-1]]
    bull = np.empty(len(c))
    bear = np.empty(len(c))
    for i in range(len(c)):
        oi, hi, li, ci, p = o[i], h[i], lo[i], c[i], c1[i]
        if ci < oi:
            bull[i] = max(hi - p, ci - li) if p < oi else max(hi - oi, ci - li)
            bear[i] = max(p - oi, hi - li) if p > oi else hi - li
        elif ci > oi:
            bull[i] = hi - li if p > oi else max(oi - p, hi - li)
            bear[i] = max(p - li, hi - ci) if p > oi else max(oi - li, hi - ci)
        elif hi - ci > ci - li:
            bull[i] = max(hi - p, ci - li) if p < oi else hi - oi
            bear[i] = max(p - oi, hi - li) if p > oi else hi - li
        elif hi - ci < ci - li:
            bull[i] = hi - li if p > oi else max(oi - p, hi - li)
            bear[i] = max(p - li, hi - ci) if p > oi else oi - li
        else:
            bull[i] = max(hi - oi, ci - li) if p > oi else max(oi - p, hi - li) if p < oi else hi - li
            bear[i] = max(p - oi, hi - li) if p > oi else max(oi - li, hi - ci) if p < oi else hi - li
    return bull, bear


def grind_conditions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    o, h, lo, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
    hl2, hlc3 = (h + lo) / 2, (h + lo + c) / 3
    idx = df.index

    dip, dim, adx = _adx_masanakamura(h, lo, c, 9)
    dip, dim, adx = (pd.Series(x, index=idx) for x in (dip, dim, adx))
    l_adx, s_adx = (dip > dim) & (adx > 12), (dip < dim) & (adx > 12)

    d = o.diff()
    up, down = _rma(d.clip(lower=0), 100), _rma(-d.clip(upper=0), 100)
    rsi3 = pd.Series(np.where(down == 0, 100.0, np.where(up == 0, 0.0, 100 - 100 / (1 + up / down))), index=idx)

    ch = hlc3.diff()
    mf_up = (v * hlc3.where(ch > 0, 0.0)).rolling(58).sum()
    mf_dn = (v * hlc3.where(ch < 0, 0.0)).rolling(58).sum()
    mfi = (100 - 100 / (1 + mf_up / mf_dn.replace(0, np.nan))).fillna(100).sub(50).mul(3)

    volume_good = v > 1.3 * v.rolling(51).mean()

    src0 = c.shift(1)                       # non-repainting: previous bar's close
    jsa = (src0 + src0.shift(50)) / 2

    macd = hl2.ewm(span=3, adjust=False).mean() - hl2.ewm(span=4, adjust=False).mean()
    signal = macd.rolling(7).mean()

    sma55 = hl2.rolling(55).mean()
    ma_speed = sma55 / sma55.shift() - 1

    psar = pd.Series(_psar(h.to_numpy(float), lo.to_numpy(float), 0.011, 0.006, 0.08), index=idx)

    bull, bear = _bull_bear_power(o, h, lo, c)
    tot = bull + bear
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = np.where(tot > 0, (bull - bear) / tot, 0.0) * v.to_numpy(float)
    cvd = pd.Series(np.nancumsum(delta), index=idx)
    cvd_ma = cvd.rolling(5).mean()

    long_c = pd.DataFrame({
        "adx": l_adx, "rsi": rsi3 < 70, "mfi": mfi > 8, "volume": volume_good, "jma": src0 > jsa,
        "macd": macd > signal, "ma_speed": ma_speed > 0, "sar": psar < c, "delta": cvd > cvd_ma,
    }, index=idx).fillna(False).astype(bool)
    short_c = pd.DataFrame({
        "adx": s_adx, "rsi": rsi3 > 30, "mfi": mfi < -8, "volume": volume_good, "jma": src0 < jsa,
        "macd": macd < signal, "ma_speed": ma_speed < 0, "sar": psar > c, "delta": cvd < cvd_ma,
    }, index=idx).fillna(False).astype(bool)

    # Heikin Ashi scalp signal (lengths 24 / 10 / 100, lookback 6)
    ha = heikin_ashi(df)
    hc, ho, hh, hl = ha["close"], ha["open"], ha["high"], ha["low"]
    fast, med = hc.ewm(span=10, adjust=False).mean(), hc.ewm(span=100, adjust=False).mean()
    pac_c = hc.ewm(span=24, adjust=False).mean()
    pac_l = hl.ewm(span=24, adjust=False).mean()
    pac_u = hh.ewm(span=24, adjust=False).mean()
    trend_dir = np.where((fast > med) & (pac_l > med), 1, np.where((fast < med) & (pac_u < med), -1, 0))
    exit_u = ((ho < pac_u) & (hc > pac_u)).to_numpy() & (_bars_since((hc < pac_c).to_numpy()) <= 6)
    exit_l = ((ho > pac_l) & (hc < pac_l)).to_numpy() & (_bars_since((hc > pac_c).to_numpy()) <= 6)
    buy, sell = (trend_dir == 1) & exit_u, (trend_dir == -1) & exit_l
    below, above = (hc < pac_c).to_numpy(), (hc > pac_c).to_numpy()
    td = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        p = td[i - 1] if i else 0
        td[i] = 0 if (p == 1 and below[i]) or (p == -1 and above[i]) else \
            1 if p == 0 and buy[i] else -1 if p == 0 and sell[i] else p
    prev = np.r_[0, td[:-1]]
    l_scalp = pd.Series((prev == 0) & (td == 1), index=idx)
    s_scalp = pd.Series((prev == 0) & (td == -1), index=idx)
    return long_c, short_c, l_scalp, s_scalp


def grind_signals(df: pd.DataFrame) -> pd.DataFrame:
    """As the script's backtest: enter whenever the conditions hold (one position at
    a time); an opposite signal reverses."""
    long_c, short_c, l_scalp, s_scalp = grind_conditions(df)
    long_cond = long_c.all(axis=1) | (l_scalp & long_c["adx"])
    short_cond = short_c.all(axis=1) | (s_scalp & short_c["adx"])
    return pd.DataFrame({"long_entry": long_cond, "long_exit": short_cond,
                         "short_entry": short_cond, "short_exit": long_cond}, index=df.index)


def grind_checklist(df: pd.DataFrame, bar: int = -1, label: str = "S3 - GRIND BTC (45m)") -> str:
    long_c, short_c, l_scalp, s_scalp = grind_conditions(df)
    real = df.attrs.get("real_dates")
    ts = real[bar] if real is not None else df.index[bar]
    lines = [f"[{label}] Checklist for candle closed {ts:%Y-%m-%d %H:%M} UTC (close {price(df['close'].iloc[bar])})"]
    for side, cond, scalp, labels in (("LONG", long_c, l_scalp, LONG_LABELS),
                                      ("SHORT", short_c, s_scalp, SHORT_LABELS)):
        row = cond.iloc[bar]
        go = bool(row.all() or (scalp.iloc[bar] and row["adx"]))
        lines += ["", f"{side}: {'🟢 TRADE' if go else '🔴 NO TRADE'}  ({int(row.sum())}/{len(row)} basic yes)"]
        for k, lab in labels.items():
            lines.append(f"  {'🟢 YES' if row[k] else '🔴 NO '}  {lab}")
        lines.append(f"  {'🟢 YES' if scalp.iloc[bar] else '🔴 NO '}  Heikin Ashi scalp signal (counts on its own with ADX)")
        if go:
            d = 1 if side == "LONG" else -1
            px = df["close"].iloc[bar]
            lines += ["  Exit plan (from the script):",
                      f"    TP1 {price(px * (1 + d * 0.017))} (+1.7%, 33%)   TP2 {price(px * (1 + d * 0.023))} "
                      f"(+2.3%, 66%)   TP3 {price(px * (1 + d * 0.034))} (+3.4%, rest)",
                      f"    Stop {price(px * (1 - d * STOP))} (9%)   An opposite signal reverses the position"]
    return "\n".join(lines)
