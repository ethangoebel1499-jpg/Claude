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
    "level": "This candle closed back above a 4h support/resistance zone",
    "above_ema": "Price closed above the 21 EMA",
    "ema_trend": "21 EMA rising (white)",
    "anchor_dot": "Green-dot anchor wave (last 20 bars)",
    "came_down": "MCB came down first: red dots printing mostly below zero (last 30 bars)",
    "money_flow": "Money flow blue (above 0), crossed up in the last 5 bars",
    "trigger": "Trigger wave: higher low, below +20 (last 3 bars)",
}
SHORT_LABELS = {
    "level": "This candle rejected below a 4h support/resistance zone",
    "above_ema": "Price closed below the 21 EMA",
    "ema_trend": "21 EMA falling (grey)",
    "anchor_dot": "Red-dot anchor wave (last 20 bars)",
    "came_down": "MCB went up first: green dots printing mostly above zero (last 30 bars)",
    "money_flow": "Money flow red (below 0), crossed down in the last 5 bars",
    "trigger": "Trigger wave: lower high, above -20 (last 3 bars)",
}


def _mostly(crosses: pd.Series, side_ok: pd.Series, lookback: int, min_count: int = 2,
            share: float = 2 / 3) -> pd.Series:
    """At least `min_count` crosses in the last `lookback` bars, and at least
    `share` of them on the required side of the zero line."""
    total = crosses.astype(int).rolling(lookback, min_periods=1).sum()
    good = (crosses & side_ok).astype(int).rolling(lookback, min_periods=1).sum()
    return (total >= min_count) & (good >= share * total)


def topdown_conditions(mc: pd.DataFrame, window: int = 3, anchor_lookback: int = 20,
                       mf_lookback: int = 5, trigger_max: float = 20,
                       came_down_lookback: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        "level": mc["sr_reclaim_up"],
        "above_ema": close > e21,
        "ema_trend": ema_up,
        "anchor_dot": _recent(mc["green_dot"], anchor_lookback),
        "came_down": _mostly(mc["wt_cross_down"], wt2 < 0, came_down_lookback),
        "money_flow": (mf > 0) & _recent(mf_cross_up, mf_lookback),
        "trigger": _recent(trigger_up, window),
    }, index=mc.index)
    short_c = pd.DataFrame({
        "level": mc["sr_reject_down"],
        "above_ema": close < e21,
        "ema_trend": ~ema_up,
        "anchor_dot": _recent(mc["red_dot"], anchor_lookback),
        "came_down": _mostly(mc["wt_cross_up"], wt2 > 0, came_down_lookback),
        "money_flow": (mf < 0) & _recent(mf_cross_down, mf_lookback),
        "trigger": _recent(trigger_down, window),
    }, index=mc.index)
    return long_c.fillna(False).astype(bool), short_c.fillna(False).astype(bool)


def topdown_confluence(mc: pd.DataFrame, **kw) -> pd.DataFrame:
    """Long when every long condition is a yes (see topdown_conditions):
      * this candle closed back above a 4h support/resistance zone,
      * price is closed above the 21 EMA and the EMA is rising ("white"),
      * Market Cipher B: a green-dot anchor wave, momentum came down first (red
        dots printing mostly below zero), money flow blue and freshly crossed
        up, and a trigger wave (WT cross up at a higher low than the anchor,
        not about to print a red dot).
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
    lines = [f"[{strategy_label('S1')}] Checklist for candle closed {ts:%Y-%m-%d %H:%M} UTC "
             f"(close {mc['real_close'].iloc[bar]:,.2f})"]
    for side, cond, labels in (("LONG", long_c, LONG_LABELS), ("SHORT", short_c, SHORT_LABELS)):
        row = cond.iloc[bar]
        go = bool(row.all())
        lines.append("")
        lines.append(f"{side}: {'🟢 TRADE' if go else '🔴 NO TRADE'}  ({int(row.sum())}/{len(row)} yes)")
        for key, label in labels.items():
            lines.append(f"  {'🟢 YES' if row[key] else '🔴 NO '}  {label}")
        if go:
            lines.extend(_exit_plan(mc, bar, side.lower()))
    return "\n".join(lines)


def _exit_plan(mc: pd.DataFrame, bar: int, side: str) -> list[str]:
    lv = topdown_exit_levels(mc).iloc[bar]
    fmt = (lambda x: f"{x:,.2f}" if pd.notna(x) else "none (no level)")
    trail = "under" if side == "long" else "above"
    return ["  Exit plan (from the video):",
            f"    Stop      {fmt(lv[f'{side}_stop'])}",
            f"    Target 1  {fmt(lv[f'{side}_tp1'])}  -> close 50%",
            f"    Target 2  {fmt(lv[f'{side}_tp2'])}  -> close 25%",
            f"    Rest      stop trails {trail} the 21 EMA"]


# Strategy catalog: each strategy from a video/source gets a number and a label so
# they stay separate. Rules marked pending are waiting for their transcript.
CATALOG = {
    "S1": {"name": "Top-down confluence",
           "source": "Market Cipher top-down analysis video (4h levels, 21 EMA, MCB anchor/trigger)",
           "timeframes": "4h levels, 1h entries", "func": topdown_confluence},
    "S2": {"name": "4-hour / 24-minute",
           "source": "'Definitive guide to my 4 hour 24 minute strategy' video: 4h trigger wave + WolfPack ID, "
                     "24m anchor/trigger waves, money flow and WolfPack crosses; swing-low trailing stop",
           "timeframes": "4h environment, 24m entries (datasets/XBTUSD_24m.csv)", "func": "custom",
           "module": "s2"},
    "S3": {"name": "GRIND BTC (wielkieef)",
           "source": "wielkieef's 'GRIND BTC [45MIN]' TradingView strategy, its own indicators only: "
                     "Masanakamura ADX, RSI, MFI, volume weight, volume delta, MACD, SMA, SAR, HA scalp",
           "timeframes": "45m (built from Kraken 15m candles)", "func": "raw",
           "raw": "grind"},
    "S4": {"name": "The Witcher (wielkieef)",
           "source": "wielkieef's 'The Witcher [30MIN]' TradingView script (MPL-2.0), its own "
                     "indicators only: ADX, RSI, MFI, SAR, Donchian, momentum, VWMA, range filter, HA scalp",
           "timeframes": "30m (built from Kraken 15m candles)", "func": "raw",
           "raw": "witcher"},
    "S5": {"name": "Levels + MTF divergence scalp",
           "source": "Jason Casper's scalping video: price at a level, higher timeframes 4h/2h/1h, divergences "
                     "on 24m/12m/6m, entry on the 1m money flow cross; take profits at the next levels",
           "timeframes": "1m execution (1m candles via tools/kraken_extract.py)", "func": "custom",
           "module": "s5", "fill_gaps": "1min"},
    "S6": {"name": "1m Bollinger + AO divergence",
           "source": "Jason Casper's 1-minute scalping video: Bollinger Band touch + Awesome Oscillator "
                     "divergence, target the opposite band",
           "timeframes": "1m (1m candles via tools/kraken_extract.py)", "func": "custom",
           "module": "s6", "fill_gaps": "1min"},
    "S7": {"name": "Range levels + 24m/12m",
           "source": "Jason Casper's 'bread and butter' video: range highs/lows, two big green dots + bullish "
                     "divergence (longs) or bearish divergence + falling money flow (shorts) on 24m and 12m",
           "timeframes": "4h levels, 24m + 12m confirmations (datasets/XBTUSD_12m.csv)", "func": "custom",
           "module": "s7"},
    "S8": {"name": "Long-only reversal",
           "source": "Basic rules: WaveTrend green dot below zero + green Heikin Ashi = reversal up (long); "
                     "red dot above zero + red Heikin Ashi = reversal down (exit). 50% at +2%, 25% at +5%, "
                     "the rest rides until the reversal down. No shorts",
           "timeframes": "any (best on 8h-2D)", "func": "raw", "raw": "s8", "long_only": True},
}


def custom_module(sid: str):
    """Strategies with their own signals / backtest / checklist (S2, S5, S6, S7)."""
    import importlib
    return importlib.import_module(f"mcbot.{CATALOG[sid]['module']}")


def raw_strategy(sid: str):
    """(signals, checklist, exits) for strategies that use only their own indicators."""
    if CATALOG[sid].get("raw") == "s8":
        from . import s8
        return s8.signals, s8.checklist, {"take_profits": s8.TAKE_PROFITS, "stop_loss": None}
    if CATALOG[sid].get("raw") == "grind":
        from . import grind
        return grind.grind_signals, grind.grind_checklist, {
            "take_profits": grind.TAKE_PROFITS, "stop_loss": grind.STOP}
    from . import witcher
    return witcher.witcher_signals, witcher.witcher_checklist, {
        "take_profit": witcher.TAKE_PROFIT, "stop_loss": witcher.STOP_LONG,
        "stop_loss_short": witcher.STOP_SHORT}


STRATEGIES = {sid: v["func"] for sid, v in CATALOG.items() if callable(v["func"])}


def strategy_label(sid: str) -> str:
    return f"{sid} - {CATALOG[sid]['name']}"


def build_signals(mc: pd.DataFrame, strategy: str = "S1") -> pd.DataFrame:
    if CATALOG.get(strategy, {}).get("func") is None:
        raise SystemExit(f"{strategy} has no rules yet: {CATALOG.get(strategy, {}).get('source', 'unknown')}")
    return STRATEGIES[strategy](mc)


def topdown_exit_levels(mc: pd.DataFrame, buffer: float = 0.002, zone: float = 0.004) -> pd.DataFrame:
    """The video's exits as prices, per bar (used at the entry bar, and every bar
    for the trailing stop):

      long:  stop just below the 21 EMA; 50% at the next 4h level above, 25% at
             the level after; the stop then keeps trailing under the 21 EMA.
      short: stop just above the rejected resistance level; 50% / 25% at the next
             levels below; the stop trails above the 21 EMA.

    Targets sit at the near edge of each zone (zone = zone width as a fraction).
    """
    e21 = mc["real_ema21"]
    return pd.DataFrame({
        "long_stop": e21 * (1 - buffer),
        "long_trail": e21 * (1 - buffer),
        "long_tp1": mc["sr_next_above"] * (1 - zone / 2),
        "long_tp2": mc["sr_next_above2"] * (1 - zone / 2),
        "short_stop": mc["sr_next_above"] * (1 + zone / 2 + buffer),
        "short_trail": e21 * (1 + buffer),
        "short_tp1": mc["sr_next_below"] * (1 + zone / 2),
        "short_tp2": mc["sr_next_below2"] * (1 + zone / 2),
    }, index=mc.index)
