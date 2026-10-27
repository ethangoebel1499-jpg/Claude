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


# Plain-English names for each condition, in checklist order.
LONG_LABELS = {
    "level": "Closed back above a 4h support/resistance zone (last 3 bars)",
    "above_ema": "Price closed above the 21 EMA",
    "ema_trend": "21 EMA rising (white)",
    "anchor_dot": "Green-dot anchor wave (last 20 bars)",
    "money_flow": "Money flow rising and crossed / about to cross above 0",
    "trigger": "Trigger wave: higher low, below +20 (last 3 bars)",
}
SHORT_LABELS = {
    "level": "Rejected below a 4h support/resistance zone (last 3 bars)",
    "above_ema": "Price closed below the 21 EMA",
    "ema_trend": "21 EMA falling (grey)",
    "anchor_dot": "Red-dot anchor wave (last 20 bars)",
    "money_flow": "Money flow falling and crossed / about to cross below 0",
    "trigger": "Trigger wave: lower high, above -20 (last 3 bars)",
}


def topdown_conditions(mc: pd.DataFrame, window: int = 3, anchor_lookback: int = 20,
                       mf_lookback: int = 10, mf_near: float = 5,
                       trigger_max: float = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every entry condition as its own yes/no column, for longs and shorts.

    An entry fires only when every column is True, so the checklist and the
    trades can never disagree.
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

    long_c = pd.DataFrame({
        "level": _recent(mc["sr_reclaim_up"], window),
        "above_ema": close > e21,
        "ema_trend": ema_up,
        "anchor_dot": _recent(mc["green_dot"], anchor_lookback),
        "money_flow": (mf > mf.shift(3)) & ((_recent(mf_cross_up, mf_lookback) & (mf > 0))
                                            | mf.between(-mf_near, 0)),
        "trigger": _recent(trigger_up, window),
    }, index=mc.index)
    short_c = pd.DataFrame({
        "level": _recent(mc["sr_reject_down"], window),
        "above_ema": close < e21,
        "ema_trend": ~ema_up,
        "anchor_dot": _recent(mc["red_dot"], anchor_lookback),
        "money_flow": (mf < mf.shift(3)) & ((_recent(mf_cross_down, mf_lookback) & (mf < 0))
                                            | mf.between(0, mf_near)),
        "trigger": _recent(trigger_down, window),
    }, index=mc.index)
    return long_c.fillna(False).astype(bool), short_c.fillna(False).astype(bool)


def topdown_confluence(mc: pd.DataFrame, **kw) -> pd.DataFrame:
    """Long when every long condition is a yes (see topdown_conditions):
      * a candle closed back above a 4h support/resistance zone,
      * price is closed above the 21 EMA and the EMA is rising ("white"),
      * Market Cipher B: a green-dot anchor wave, money flow rising and crossed
        or about to cross above 0, and a trigger wave (WT cross up at a higher
        low than the anchor, not about to print a red dot).
    Exit when a candle closes below the 21 EMA. Shorts mirror everything.
    """
    long_c, short_c = topdown_conditions(mc, **kw)
    close, e21 = mc["real_close"], mc["real_ema21"]
    return _frame(mc, long_c.all(axis=1), close < e21, short_c.all(axis=1), close > e21)


def checklist(mc: pd.DataFrame, bar: int = -1) -> str:
    """Green/red light checklist for one bar (default: the last closed bar)."""
    long_c, short_c = topdown_conditions(mc)
    real = mc.attrs.get("real_dates")
    ts = real[bar] if real is not None else mc.index[bar]
    lines = [f"Checklist for candle closed {ts:%Y-%m-%d %H:%M} UTC "
             f"(close {mc['real_close'].iloc[bar]:,.2f})"]
    for side, cond, labels in (("LONG", long_c, LONG_LABELS), ("SHORT", short_c, SHORT_LABELS)):
        row = cond.iloc[bar]
        go = bool(row.all())
        lines.append("")
        lines.append(f"{side}: {'🟢 TRADE' if go else '🔴 NO TRADE'}  ({int(row.sum())}/{len(row)} yes)")
        for key, label in labels.items():
            lines.append(f"  {'🟢 YES' if row[key] else '🔴 NO '}  {label}")
    return "\n".join(lines)


STRATEGIES = {"topdown_confluence": topdown_confluence}


def build_signals(mc: pd.DataFrame, strategy: str = "topdown_confluence") -> pd.DataFrame:
    return STRATEGIES[strategy](mc)
