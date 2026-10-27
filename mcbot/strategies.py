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


def money_flow_div(mc: pd.DataFrame) -> pd.DataFrame:
    """Money flow divergence / convergence.

    Long: bullish money flow divergence (price lower low, money flow higher low)
    and money flow turning up. Exit when price and money flow converge to the
    downside, or on a bearish divergence. Shorts are the mirror image.
    """
    mf = mc["money_flow"]
    long_entry = mc["bull_mf_div"] & (mf > mf.shift())
    short_entry = mc["bear_mf_div"] & (mf < mf.shift())
    long_exit = mc["bear_mf_div"] | mc["bear_mf_conv"]
    short_exit = mc["bull_mf_div"] | mc["bull_mf_conv"]
    return _frame(mc, long_entry, long_exit, short_entry, short_exit)


# --- CryptoFace (Market Cipher creator) style -------------------------------
# Built from public Market Cipher B guides: higher-timeframe bias first,
# anchor/trigger waves for swing entries, dots + money flow + VWAP wave for scalps.

def cryptoface_swing(mc: pd.DataFrame) -> pd.DataFrame:
    """Long-term: trade the anchor/trigger wave in the direction of the
    higher-timeframe money flow. Hold until the higher timeframe turns."""
    htf_bull = (mc["htf_money_flow"] > 0) & mc["htf_ribbon_bull"]
    htf_bear = (mc["htf_money_flow"] < 0) & ~mc["htf_ribbon_bull"]
    long_entry = mc["anchor_trigger_bull"] & htf_bull
    short_entry = mc["anchor_trigger_bear"] & htf_bear
    long_exit = mc["anchor_trigger_bear"] | (mc["red_dot"] & ~mc["htf_wt_rising"]) | \
        (mc["htf_money_flow"] < 0)
    short_exit = mc["anchor_trigger_bull"] | (mc["green_dot"] & mc["htf_wt_rising"]) | \
        (mc["htf_money_flow"] > 0)
    return _frame(mc, long_entry, long_exit, short_entry, short_exit)


def cryptoface_scalp(mc: pd.DataFrame) -> pd.DataFrame:
    """Scalp: green dot while money flow is rising and the VWAP wave crosses
    above zero, only with the higher-timeframe wave. Quick exit on the
    opposite WaveTrend cross."""
    mf = mc["money_flow"]
    v = mc["mcb_vwap"]
    vwap_up = (v > 0) & (v.shift() <= 0)
    vwap_down = (v < 0) & (v.shift() >= 0)
    recent_green = mc["green_dot"].astype(int).rolling(3, min_periods=1).max() > 0
    recent_red = mc["red_dot"].astype(int).rolling(3, min_periods=1).max() > 0
    long_entry = recent_green & vwap_up & (mf > mf.shift()) & mc["htf_wt_rising"]
    short_entry = recent_red & vwap_down & (mf < mf.shift()) & ~mc["htf_wt_rising"]
    return _frame(mc, long_entry, mc["wt_cross_down"], short_entry, mc["wt_cross_up"])


# --- Jayson Casper style -----------------------------------------------------
# Built from his course topics: swing failure patterns, VWAP, and swing
# trading from support to the next resistance.

def casper_scalp(mc: pd.DataFrame) -> pd.DataFrame:
    """Scalp: bullish SFP (stop hunt below a swing low) while price is under
    the session VWAP; take profit when price gets back to VWAP."""
    long_entry = mc["sfp_bull"] & (mc["close"] < mc["vwap"])
    short_entry = mc["sfp_bear"] & (mc["close"] > mc["vwap"])
    long_exit = (mc["close"] >= mc["vwap"]) | mc["sfp_bear"]
    short_exit = (mc["close"] <= mc["vwap"]) | mc["sfp_bull"]
    return _frame(mc, long_entry, long_exit, short_entry, short_exit)


def casper_swing(mc: pd.DataFrame) -> pd.DataFrame:
    """Long-term: buy an SFP at support (near the 50-bar low) once money flow
    agrees, and hold to resistance (the 50-bar high)."""
    near_support = mc["low"] <= mc["swing_low50"].shift() * 1.01
    near_resistance = mc["high"] >= mc["swing_high50"].shift() * 0.99
    mf_up = mc["money_flow"] > mc["money_flow"].shift(3)
    mf_down = mc["money_flow"] < mc["money_flow"].shift(3)
    long_entry = mc["sfp_bull"] & near_support & mf_up
    short_entry = mc["sfp_bear"] & near_resistance & mf_down
    long_exit = near_resistance | mc["sfp_bear"]
    short_exit = near_support | mc["sfp_bull"]
    return _frame(mc, long_entry, long_exit, short_entry, short_exit)


# --- Double dot ("uneven butt cheeks") ----------------------------------------

def _previous_dot(dot: pd.Series, wt2: pd.Series, confirmed: pd.Series):
    """For every bar: bars since the previous dot, its wt2 level, and whether
    momentum lined up at it. Only looks at dots strictly before the bar."""
    pos = pd.Series(range(len(dot)), index=dot.index, dtype=float)
    last_pos = pos.where(dot).shift().ffill()
    last_wt2 = wt2.where(dot).shift().ffill()
    last_ok = confirmed.astype(float).where(dot).shift().ffill()
    return pos - last_pos, last_wt2, last_ok == 1


def double_dot(mc: pd.DataFrame, max_gap: int = 30, min_gap: int = 3,
               second_only: bool = False) -> pd.DataFrame:
    """Take a green dot when momentum (money flow) lines up. If it doesn't,
    wait for a second, *higher* green dot (a higher low on the wave, the
    "uneven butt cheeks") within `max_gap` bars where momentum does line up.
    Shorts mirror it with a second, lower red dot. With second_only, single
    dots are never traded."""
    mf = mc["money_flow"]
    wt2 = mc["wt2"]

    def entries(dot, mom, higher):
        gap, prev_wt2, prev_ok = _previous_dot(dot, wt2, mom)
        uneven = (wt2 > prev_wt2) if higher else (wt2 < prev_wt2)
        second = dot & mom & ~prev_ok & uneven & gap.between(min_gap, max_gap)
        return second if second_only else second | (dot & mom)

    long_entry = entries(mc["green_dot"], mf > mf.shift(3), higher=True)
    short_entry = entries(mc["red_dot"], mf < mf.shift(3), higher=False)
    long_exit = mc["red_dot"] | ((mf < 0) & (mf.shift() >= 0))
    short_exit = mc["green_dot"] | ((mf > 0) & (mf.shift() <= 0))
    return _frame(mc, long_entry, long_exit, short_entry, short_exit)


def double_dot_only(mc: pd.DataFrame) -> pd.DataFrame:
    """Only the second, higher green dot (or lower red dot); skip single dots."""
    return double_dot(mc, second_only=True)


# --- Heikin Ashi --------------------------------------------------------------

def heikin_ashi_trend(mc: pd.DataFrame) -> pd.DataFrame:
    """Green dot in the last 5 bars, confirmed by a strong green Heikin Ashi
    candle (no lower wick) with money flow rising. Ride the trend until the
    first red candle. Use with --heikin-ashi (the default)."""
    mf = mc["money_flow"]
    recent_green = mc["green_dot"].astype(int).rolling(5, min_periods=1).max() > 0
    recent_red = mc["red_dot"].astype(int).rolling(5, min_periods=1).max() > 0
    long_entry = recent_green & mc["candle_strong_bull"] & (mf > mf.shift())
    short_entry = recent_red & mc["candle_strong_bear"] & (mf < mf.shift())
    return _frame(mc, long_entry, ~mc["candle_bull"], short_entry, mc["candle_bull"])


STRATEGIES = {
    "double_dot": double_dot,
    "double_dot_only": double_dot_only,
    "heikin_ashi_trend": heikin_ashi_trend,
    "cryptoface_swing": cryptoface_swing,
    "cryptoface_scalp": cryptoface_scalp,
    "casper_swing": casper_swing,
    "casper_scalp": casper_scalp,
    "dots": dots,
    "dots_confirmed": dots_confirmed,
    "divergence": divergence,
    "money_flow_div": money_flow_div,
}
