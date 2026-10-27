"""Trading strategies built on Market Cipher signals.

Each strategy takes the frame from indicators.market_cipher() and returns a
DataFrame with boolean columns:

    long_entry, long_exit, short_entry, short_exit

Signals are evaluated on the close of a bar; the backtester and the bot act
on them at the next bar's open.
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


def dots(mc: pd.DataFrame) -> pd.DataFrame:
    """Classic Cipher B: buy green dots, sell red dots."""
    return _frame(mc, mc["green_dot"], mc["red_dot"], mc["red_dot"], mc["green_dot"])


def dots_confirmed(mc: pd.DataFrame) -> pd.DataFrame:
    """Dots filtered by money flow and trend.

    Long: green dot, money flow rising, price above EMA200, Cipher A ribbon bullish.
    Short: the mirror image. Exit on the opposite dot or when money flow flips.
    """
    mf = mc["money_flow"]
    mf_rising = mf > mf.shift(3)
    mf_falling = mf < mf.shift(3)
    uptrend = (mc["close"] > mc["ema200"]) & mc["ribbon_bull"]
    downtrend = (mc["close"] < mc["ema200"]) & ~mc["ribbon_bull"]

    long_entry = mc["green_dot"] & mf_rising & uptrend
    short_entry = mc["red_dot"] & mf_falling & downtrend
    long_exit = mc["red_dot"] | ((mf < 0) & (mf.shift() >= 0))
    short_exit = mc["green_dot"] | ((mf > 0) & (mf.shift() <= 0))
    return _frame(mc, long_entry, long_exit, short_entry, short_exit)


def divergence(mc: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Divergence + dot: enter on a dot that comes within `window` bars of a
    matching WaveTrend divergence (the "gold dot"-style setup)."""
    recent_bull = mc["bull_div"].astype(int).rolling(window, min_periods=1).max() > 0
    recent_bear = mc["bear_div"].astype(int).rolling(window, min_periods=1).max() > 0
    long_entry = mc["green_dot"] & recent_bull
    short_entry = mc["red_dot"] & recent_bear
    return _frame(mc, long_entry, mc["red_dot"], short_entry, mc["green_dot"])


STRATEGIES = {
    "dots": dots,
    "dots_confirmed": dots_confirmed,
    "divergence": divergence,
}
