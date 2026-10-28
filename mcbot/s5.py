"""S5 - Levels + multi-timeframe divergence scalp (Jason Casper).

  1  price comes to a significant level (4h support / resistance)
  2  higher timeframes (4h, 2h, 1h): Market Cipher B momentum direction, averaged.
     He scalps against them too, so it is only required with htf=True
  3  lower timeframes (24m, 12m, 6m): a bullish divergence (regular or hidden)
     on at least 2 of the 3 (bearish for shorts)
  4  the reaction: his conservative entry, the 1m money flow crossing into the
     green (red for shorts) while the setup is live
  stop: beyond the last swing low (high) on the 6m
  exits: 30% at the next level and the stop moves to the entry; the rest at the
         level after (his take profits sit at the next support / resistance levels)

Regular candles. Every timeframe is built from 1m candles and only used once its
candle has closed. Trades run on the 1m chart and fill at the next 1m open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import Result, run_level_backtest
from .data import resample
from .divergence import recent, waves
from .indicators import money_flow, sr_levels, wavetrend

NEAR = 0.005
LEVEL_BARS = 60        # price was at the level within the last hour
SETUP_BARS = 60        # the money flow cross must come within an hour of the setup
DIV_BARS = 5           # divergence printed in the last 5 bars of its chart
MAX_GAP = 40
STOP_ROOM = 0.001
LTF = ("24min", "12min", "6min")
HTF = ("4h", "2h", "1h")


def _on(fast_index, slow: pd.DataFrame, slow_step: str) -> pd.DataFrame:
    s = slow.copy()
    s.index = slow.index + pd.Timedelta(slow_step) - pd.Timedelta("1min")
    return s.reindex(fast_index, method="ffill")


def _tf_frame(df: pd.DataFrame) -> pd.DataFrame:
    wt1, wt2 = wavetrend(df)
    up = (wt1 > wt2) & (wt1.shift() <= wt2.shift())
    dn = (wt1 < wt2) & (wt1.shift() >= wt2.shift())
    w = waves(wt2, df["low"], df["high"], up, dn)
    tgap = (w.t1_i - w.t0_i) <= MAX_GAP
    pgap = (w.p1_i - w.p0_i) <= MAX_GAP
    bull = up & tgap & (w.t1_osc != w.t0_osc) & (
        ((w.t1_px < w.t0_px) & (w.t1_osc > w.t0_osc)) | ((w.t1_px > w.t0_px) & (w.t1_osc < w.t0_osc)))
    bear = dn & pgap & (
        ((w.p1_px > w.p0_px) & (w.p1_osc < w.p0_osc)) | ((w.p1_px < w.p0_px) & (w.p1_osc > w.p0_osc)))
    return pd.DataFrame({"bull": recent(bull.fillna(False), DIV_BARS), "bear": recent(bear.fillna(False), DIV_BARS),
                         "mom": np.sign(wt1 - wt2), "low_px": w.t1_px, "high_px": w.p1_px}, index=df.index)


def signals(m1: pd.DataFrame, htf: bool = False) -> pd.DataFrame:
    idx = m1.index
    tf = {t: _on(idx, _tf_frame(resample(m1, t)), t) for t in LTF + HTF}
    m6 = resample(m1, "6min")
    lv = _on(idx, sr_levels(m6, factor=40), "6min")          # 40 x 6m = 4h levels
    at_support = (m1["low"].rolling(LEVEL_BARS).min() <= lv.sr_next_below * (1 + NEAR)).to_numpy()
    at_resist = (m1["high"].rolling(LEVEL_BARS).max() >= lv.sr_next_above * (1 - NEAR)).to_numpy()
    n_bull = sum(tf[t].bull.fillna(False).astype(int) for t in LTF).to_numpy()
    n_bear = sum(tf[t].bear.fillna(False).astype(int) for t in LTF).to_numpy()
    htf_mom = sum(tf[t].mom.fillna(0) for t in HTF).to_numpy()
    mf = money_flow(m1)
    mf_up = ((mf > 0) & (mf.shift() <= 0)).to_numpy()
    mf_dn = ((mf < 0) & (mf.shift() >= 0)).to_numpy()

    setup_l = at_support & (n_bull >= 2) & ((htf_mom > 0) if htf else True)
    setup_s = at_resist & (n_bear >= 2) & ((htf_mom < 0) if htf else True)
    n = len(m1)
    pos = np.arange(n)
    last_l = pd.Series(np.where(setup_l, pos, np.nan)).ffill().to_numpy()
    last_s = pd.Series(np.where(setup_s, pos, np.nan)).ffill().to_numpy()
    long_e = mf_up & (pos - last_l <= SETUP_BARS)
    short_e = mf_dn & (pos - last_s <= SETUP_BARS)

    lo6, hi6 = tf["6min"].low_px, tf["6min"].high_px
    a1, a2 = lv.sr_next_above, lv.sr_next_above2.fillna(lv.sr_next_above)
    b1, b2 = lv.sr_next_below, lv.sr_next_below2.fillna(lv.sr_next_below)
    return pd.DataFrame({
        "long_entry": long_e, "short_entry": short_e, "long_exit": False, "short_exit": False,
        "at_support": at_support, "at_resist": at_resist, "ltf_bull": n_bull, "ltf_bear": n_bear,
        "htf_mom": htf_mom, "mf": mf,
        "long_stop": np.minimum(lo6, m1["low"].rolling(SETUP_BARS).min()) * (1 - STOP_ROOM),
        "short_stop": np.maximum(hi6, m1["high"].rolling(SETUP_BARS).max()) * (1 + STOP_ROOM),
        "long_tp1": a1, "long_tp2": a2, "short_tp1": b1, "short_tp2": b2,
        "long_trail": -np.inf, "short_trail": np.inf,
    }, index=idx)


def backtest(m1: pd.DataFrame, fr: pd.DataFrame, *, allow_short: bool = True, fee: float = 0.001) -> Result:
    lv = fr[["long_stop", "long_tp1", "long_tp2", "long_trail", "short_stop", "short_tp1", "short_tp2",
             "short_trail"]]
    return run_level_backtest(m1, fr[["long_entry", "short_entry", "long_exit", "short_exit"]], lv,
                              allow_short=allow_short, fee=fee, sizes=(0.3, 0.7), breakeven_after_tp1=True)


def checklist(m1: pd.DataFrame, bar: int = -1, label: str = "S5 - Levels + MTF divergence scalp") -> str:
    fr = signals(m1)
    r = fr.iloc[bar]
    lines = [f"[{label}] Checklist for candle {m1.index[bar]:%Y-%m-%d %H:%M} UTC (close {m1['close'].iloc[bar]:,.2f})"]
    for side, sgn, at, ltf, go in (("LONG", 1, r.at_support, r.ltf_bull, r.long_entry),
                                   ("SHORT", -1, r.at_resist, r.ltf_bear, r.short_entry)):
        word = "bullish" if sgn > 0 else "bearish"
        lines += ["", f"{side}: {'🟢 TRADE' if go else '🔴 NO TRADE'}",
                  f"  {'🟢 YES' if at else '🔴 NO '}  1. Price at a 4h {'support' if sgn > 0 else 'resistance'} level",
                  f"  {'🟢 YES' if sgn * r.htf_mom > 0 else '🔴 NO '}  2. Higher timeframes (4h/2h/1h) lean {word} "
                  f"({int(sgn * r.htf_mom):+d} of 3; scalps can go against them)",
                  f"  {'🟢 YES' if ltf >= 2 else '🔴 NO '}  3. {word.capitalize()} divergence on {int(ltf)}/3 of 24m/12m/6m",
                  f"  {'🟢 YES' if go else '🔴 NO '}  4. 1m money flow crossed {'green' if sgn > 0 else 'red'} (the reaction)"]
    return "\n".join(lines)
