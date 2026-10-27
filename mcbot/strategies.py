"""Top-down confluence strategy (from a Market Cipher top-down analysis video).

4h support/resistance + 21 EMA on the naked chart, confirmed by Market Cipher B
(anchor wave with green dot, money flow crossing, trigger wave). Entries on the
chart timeframe (1h); levels from the 4x higher timeframe.

The strategy returns boolean columns long_entry, long_exit, short_entry,
short_exit. Signals are evaluated on a closed bar; the backtester and the bot
act on them at the next bar's open.
"""

from __future__ import annotations

import pandas as pd


def _frame(mc: pd.DataFrame, long_entry, long_exit, short_entry, short_exit) -> pd.DataFrame:
    return pd.DataFrame({
        "long_entry": long_entry.fillna(False).astype(bool),
        "long_exit": long_exit.fillna(False).astype(bool),
        "short_entry": short_entry.fillna(False).astype(bool),
        "short_exit": short_exit.fillna(False).astype(bool),
    }, index=mc.index)


def _recent(s: pd.Series, n: int) -> pd.Series:
    return s.astype(int).rolling(n, min_periods=1).max() > 0


def topdown_confluence(mc: pd.DataFrame, window: int = 3, anchor_lookback: int = 20,
                       mf_lookback: int = 10, mf_near: float = 5, trigger_max: float = 20) -> pd.DataFrame:
    """Long when, within the last `window` bars:
      * a candle closed back above a 4h support/resistance zone,
      * price is closed above the 21 EMA and the EMA is rising ("white"),
      * Market Cipher B: a green-dot anchor wave in the last `anchor_lookback`
        bars, money flow rising and either crossed above 0 in the last
        `mf_lookback` bars or within `mf_near` of crossing ("crossed, or right
        about to cross"), and a trigger wave (WT cross up at a higher low than the
        anchor, still below +`trigger_max`, i.e. not about to print a red dot).
    Exit when a candle closes below the 21 EMA. Shorts mirror everything.
    """
    close, e21 = mc["real_close"], mc["real_ema21"]
    mf, wt2 = mc["money_flow"], mc["wt2"]
    ema_up = e21 > e21.shift()
    mf_cross_up = (mf > 0) & (mf.shift() <= 0)
    mf_cross_down = (mf < 0) & (mf.shift() >= 0)
    anchor_low = wt2.rolling(anchor_lookback).min()
    anchor_high = wt2.rolling(anchor_lookback).max()

    trigger_up = mc["wt_cross_up"] & (wt2 > anchor_low + 5) & (wt2 < trigger_max)
    trigger_down = mc["wt_cross_down"] & (wt2 < anchor_high - 5) & (wt2 > -trigger_max)

    mf_long_ok = (mf > mf.shift(3)) & ((_recent(mf_cross_up, mf_lookback) & (mf > 0))
                                       | mf.between(-mf_near, 0))
    mf_short_ok = (mf < mf.shift(3)) & ((_recent(mf_cross_down, mf_lookback) & (mf < 0))
                                        | mf.between(0, mf_near))

    long_entry = (_recent(mc["sr_reclaim_up"], window) & (close > e21) & ema_up
                  & _recent(mc["green_dot"], anchor_lookback) & mf_long_ok
                  & _recent(trigger_up, window))
    short_entry = (_recent(mc["sr_reject_down"], window) & (close < e21) & ~ema_up
                   & _recent(mc["red_dot"], anchor_lookback) & mf_short_ok
                   & _recent(trigger_down, window))
    long_exit = close < e21
    short_exit = close > e21
    return _frame(mc, long_entry, long_exit, short_entry, short_exit)


STRATEGIES = {"topdown_confluence": topdown_confluence}


def build_signals(mc: pd.DataFrame, strategy: str = "topdown_confluence") -> pd.DataFrame:
    return STRATEGIES[strategy](mc)
