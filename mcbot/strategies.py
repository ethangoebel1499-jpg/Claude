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
    mf_rising = mc["mf_turn_up"]
    mf_falling = mc["mf_turn_down"]
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
    long_entry = mc["bull_mf_div"] & mc["mf_turn_up"]
    short_entry = mc["bear_mf_div"] & mc["mf_turn_down"]
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
    v = mc["mcb_vwap"]
    vwap_up = (v > 0) & (v.shift() <= 0)
    vwap_down = (v < 0) & (v.shift() >= 0)
    recent_green = mc["green_dot"].astype(int).rolling(3, min_periods=1).max() > 0
    recent_red = mc["red_dot"].astype(int).rolling(3, min_periods=1).max() > 0
    long_entry = recent_green & vwap_up & mc["mf_turn_up"] & mc["htf_wt_rising"]
    short_entry = recent_red & vwap_down & mc["mf_turn_down"] & ~mc["htf_wt_rising"]
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
    mf_up = mc["mf_turn_up"]
    mf_down = mc["mf_turn_down"]
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

    long_entry = entries(mc["green_dot"], mc["mf_turn_up"], higher=True)
    short_entry = entries(mc["red_dot"], mc["mf_turn_down"], higher=False)
    long_exit = mc["red_dot"] | ((mf < 0) & (mf.shift() >= 0))
    short_exit = mc["green_dot"] | ((mf > 0) & (mf.shift() <= 0))
    return _frame(mc, long_entry, long_exit, short_entry, short_exit)


def double_dot_only(mc: pd.DataFrame) -> pd.DataFrame:
    """Only the second, higher green dot (or lower red dot); skip single dots."""
    return double_dot(mc, second_only=True)


# --- Refined entries ------------------------------------------------------------
# Chosen by entry quality (hitting a target before a stop twice as far away) on
# real EUR/USD 1h data: plain dots ~72%, random ~66%, these two ~82-84%.

def _confirm_next_candle(long_setup: pd.Series, short_setup: pd.Series, mc: pd.DataFrame):
    """Enter one bar after the setup, only if that candle closes in our direction
    (green / red Heikin Ashi candle when running on Heikin Ashi)."""
    return (long_setup.shift(1, fill_value=False) & mc["candle_bull"],
            short_setup.shift(1, fill_value=False) & ~mc["candle_bull"])


def second_dot_confirmed(mc: pd.DataFrame) -> pd.DataFrame:
    """Second green dot within 3-30 bars that is HIGHER on the wave than the
    first (uneven butt cheeks), then a green candle to confirm. Shorts mirror it."""
    yes = pd.Series(True, index=mc.index)
    gap, prev_wt2, _ = _previous_dot(mc["green_dot"], mc["wt2"], yes)
    long_setup = mc["green_dot"] & gap.between(3, 30) & (mc["wt2"] > prev_wt2)
    gap, prev_wt2, _ = _previous_dot(mc["red_dot"], mc["wt2"], yes)
    short_setup = mc["red_dot"] & gap.between(3, 30) & (mc["wt2"] < prev_wt2)
    long_entry, short_entry = _confirm_next_candle(long_setup, short_setup, mc)
    return _frame(mc, long_entry, mc["red_dot"], short_entry, mc["green_dot"])


def dot_htf_confirmed(mc: pd.DataFrame, htf_level: float = -40) -> pd.DataFrame:
    """Green dot while the higher-timeframe wave is oversold (below -40), then a
    green candle to confirm. Shorts mirror it above +40."""
    long_setup = mc["green_dot"] & (mc["htf_wt2"] < htf_level)
    short_setup = mc["red_dot"] & (mc["htf_wt2"] > -htf_level)
    long_entry, short_entry = _confirm_next_candle(long_setup, short_setup, mc)
    return _frame(mc, long_entry, mc["red_dot"], short_entry, mc["green_dot"])


# --- Top-down confluence (from a Market Cipher top-down analysis video) ----------
# 4h support/resistance + 21 EMA on the naked chart, confirmed by Market Cipher B
# (anchor wave with green dot, money flow crossing, trigger wave). Entries on
# the chart timeframe (1h); levels from the 4x higher timeframe.

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


# --- Multi-timeframe dots ------------------------------------------------------

def _armed(start: pd.Series, cancel: pd.Series, window: int) -> pd.Series:
    """True for up to `window` bars after `start` (inclusive), until `cancel`."""
    pos = pd.Series(range(len(start)), index=start.index, dtype=float)
    since_start = pos - pos.where(start).ffill()
    since_cancel = pos - pos.where(cancel & ~start).ffill()
    not_cancelled = since_cancel.isna() | (since_cancel > since_start)
    return (since_start < window) & not_cancelled


def mtf_dots(mc: pd.DataFrame, wait_htf_candles: int = 3) -> pd.DataFrame:
    """A higher-timeframe green dot (on a closed HTF candle) arms the setup;
    enter on the next lower-timeframe green dot within `wait_htf_candles` HTF
    candles. An HTF red dot cancels it. Shorts mirror it.
    Exit on a lower-timeframe red dot (long) / green dot (short)."""
    window = mc.attrs.get("htf_factor", 4) * wait_htf_candles
    htf_green = mc["htf_green_dot"] & mc["htf_new"]
    htf_red = mc["htf_red_dot"] & mc["htf_new"]
    long_entry = mc["green_dot"] & _armed(htf_green, htf_red, window)
    short_entry = mc["red_dot"] & _armed(htf_red, htf_green, window)
    return _frame(mc, long_entry, mc["red_dot"], short_entry, mc["green_dot"])


# --- Heikin Ashi --------------------------------------------------------------

def heikin_ashi_trend(mc: pd.DataFrame) -> pd.DataFrame:
    """Green dot in the last 5 bars, confirmed by a strong green Heikin Ashi
    candle (no lower wick) with money flow rising. Ride the trend until the
    first red candle. Use with --heikin-ashi (the default)."""
    recent_green = mc["green_dot"].astype(int).rolling(5, min_periods=1).max() > 0
    recent_red = mc["red_dot"].astype(int).rolling(5, min_periods=1).max() > 0
    long_entry = recent_green & mc["candle_strong_bull"] & mc["mf_turn_up"]
    short_entry = recent_red & mc["candle_strong_bear"] & mc["mf_turn_down"]
    return _frame(mc, long_entry, ~mc["candle_bull"], short_entry, mc["candle_bull"])


STRATEGIES = {
    "topdown_confluence": topdown_confluence,
    "second_dot_confirmed": second_dot_confirmed,
    "dot_htf_confirmed": dot_htf_confirmed,
    "mtf_dots": mtf_dots,
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


def _recent(s: pd.Series, n: int) -> pd.Series:
    return s.astype(int).rolling(n, min_periods=1).max() > 0


def apply_filters(sig: pd.DataFrame, mc: pd.DataFrame, *, mf_turn: bool = False,
                  convergence: bool = False, mf_window: int = 3,
                  conv_window: int = 20) -> pd.DataFrame:
    """Extra entry filters that can be added to any strategy.

    mf_turn: money flow changed direction in our favour within `mf_window` bars.
    convergence: a bullish money flow divergence or convergence within
    `conv_window` bars and no bearish convergence now (mirror for shorts).
    """
    sig = sig.copy()
    if mf_turn:
        sig["long_entry"] &= _recent(mc["mf_turn_up"], mf_window)
        sig["short_entry"] &= _recent(mc["mf_turn_down"], mf_window)
    if convergence:
        sig["long_entry"] &= (_recent(mc["bull_mf_div"], conv_window) |
                              _recent(mc["bull_mf_conv"], conv_window)) & ~mc["bear_mf_conv"]
        sig["short_entry"] &= (_recent(mc["bear_mf_div"], conv_window) |
                               _recent(mc["bear_mf_conv"], conv_window)) & ~mc["bull_mf_conv"]
    return sig


def build_signals(mc: pd.DataFrame, strategy: str, *, mf_turn: bool = False,
                  convergence: bool = False) -> pd.DataFrame:
    return apply_filters(STRATEGIES[strategy](mc), mc, mf_turn=mf_turn, convergence=convergence)
