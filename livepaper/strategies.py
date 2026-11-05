"""Strategies the paper trader (and the Strategy Lab) can run, with their default exits.

Keys are what live.toml and the runs table use for `strategy`.
"""

from __future__ import annotations

from mcbot import price_action as s10, s8
from mcbot.grind import STOP as G_STOP, TAKE_PROFITS as G_TP, grind_checklist, grind_signals
from mcbot.indicators import market_cipher
from mcbot.strategies import build_signals, checklist as topdown_checklist
from mcbot.witcher import STOP_LONG as W_STOP, TAKE_PROFIT as W_TP, witcher_checklist, witcher_signals

from .core import Exits


def _topdown_signals(df):
    return build_signals(market_cipher(df, use_heikin_ashi=True), "S1")


def _topdown_checklist(df, bar=-1, label=None):
    return topdown_checklist(market_cipher(df, use_heikin_ashi=True), bar)


_S10_NOTE = ("Price action confirmed by RSI and money flow: support/resistance bounces, break-and-retest, chart-shape "
             "breakouts that the crypto breakout statistics back (entered on the retest) and swing failures; money flow "
             "must move the trade's way and RSI must not be stretched. Stop beyond the structure, TP1 just before the "
             "next level, TP2 the level after (half each), breakeven after TP1. Any timeframe.")

REGISTRY = {
    "witcher": {"label": "S4b Witcher (wide exits)", "short": "S4b Witcher", "signals": witcher_signals,
                "checklist": witcher_checklist, "exits": Exits(0.04, [(0.12, 0.5), (0.24, 0.5)]),
                "note": "Held up best across coins: 4% stop, 50% at +12%, 50% at +24%."},
    "witcher_script": {"label": "S4 Witcher (script exits)", "short": "S4 Witcher", "signals": witcher_signals,
                       "checklist": witcher_checklist, "exits": Exits(W_STOP, [(W_TP, 1.0)]),
                       "note": "The script's own exits: +1.1% target, ~8% stop."},
    "grind": {"label": "S3 GRIND (targets x3)", "short": "S3 GRIND", "signals": grind_signals,
              "checklist": grind_checklist, "exits": Exits(0.06, [(a * 3, s) for a, s in G_TP]),
              "note": "Tuned: 6% stop, 33% / 66% / 1% at +5.1% / +6.9% / +10.2%. Best on BTC 4h."},
    "grind_script": {"label": "S3 GRIND (script exits)", "short": "S3 GRIND orig", "signals": grind_signals,
                     "checklist": grind_checklist, "exits": Exits(G_STOP, G_TP),
                     "note": "The script's own exits: 9% stop, +1.7% / +2.3% / +3.4%."},
    "s8": {"label": "S8 Long-only reversal", "short": "S8", "signals": s8.signals, "checklist": s8.checklist,
           "exits": Exits(0.05, s8.TAKE_PROFITS), "long_only": True,
           "note": "5% stop, 50% at +2%, 25% at +5%, the rest rides until the reversal down. Best on 12h-2D."},
    "topdown": {"label": "S1 Top-down confluence (fixed exits)", "short": "S1 Top-down",
                "signals": _topdown_signals, "checklist": _topdown_checklist, "exits": Exits(0.03, [(0.06, 1.0)]),
                "note": "Fixed 3% stop / 6% target here; the video's level-based exits are backtest-only. "
                        "Failed testing: few trades and no edge."},
    "s10": {"label": "S10 Price action (S/R + shapes, RSI & money flow)", "short": "S10 Price action",
            "signals": s10.signals, "checklist": s10.checklist, "exits": Exits(s10.STOP, s10.TAKE_PROFITS),
            "note": _S10_NOTE},
    "s10_long": {"label": "S10 Price action - longs only", "short": "S10 long",
                 "signals": lambda df: s10.signals(df, "long"),
                 "checklist": lambda df, bar=-1, label=None: s10.checklist(df, bar, label, "long"),
                 "exits": Exits(s10.STOP, s10.TAKE_PROFITS), "long_only": True, "note": _S10_NOTE},
    "s10_short": {"label": "S10 Price action - shorts only", "short": "S10 short",
                  "signals": lambda df: s10.signals(df, "short"),
                  "checklist": lambda df, bar=-1, label=None: s10.checklist(df, bar, label, "short"),
                  "exits": Exits(s10.STOP, s10.TAKE_PROFITS), "note": _S10_NOTE},
}

# S10: 1-minute Bollinger bands + Awesome Oscillator divergence, built from the Builder's rules (so it can be
# opened in 🛠️ Builder, copied and changed). Tested on real Kraken 1m BTC/ETH, Apr-Sep 2026: before fees it made
# +1.6% to +6.7% in each of 4 test periods; at a 0.1% fee per side it lost 4-15% in each.
S10_SPEC = {
    "name": "S10 1m Bollinger + AO divergence",
    "long": [{"rule": "bb_touch_lower"}, {"rule": "ao_bull_div"}, {"rule": "ao_turns_green"},
             {"rule": "close_above_ema", "a": 60}],
    "short": [{"rule": "bb_touch_upper"}, {"rule": "ao_bear_div"}, {"rule": "ao_turns_red"},
              {"rule": "close_below_ema", "a": 60}],
    "exit_long": [], "exit_short": [], "stop": 0.01, "take_profits": [[0.02, 1.0]],
}


def _s10():
    from functools import partial
    from mcbot import builder
    spec = S10_SPEC
    return {"label": "S10 1m Bollinger + AO divergence", "short": "S10 BB+AO", "spec": spec,
            "signals": partial(builder.signals, spec=spec),
            "checklist": lambda df, bar=-1, label=None: builder.checklist(df, bar, label, spec),
            "exits": Exits(0.01, [(0.02, 1.0)]),
            "note": "Use on 1m charts. Long: low touched the lower Bollinger band (last 3 candles) + AO bullish "
                    "divergence + AO changed to green + close above EMA 60 (the 1h trend). Short: the mirror. 1% stop, "
                    "2% target. Tested on real 1m BTC/ETH (Apr-Sep 2026): small profit before fees, LOST at a 0.1% "
                    "fee in every period - only worth it with fees near 0.02% (maker/futures)."}


REGISTRY["s10"] = _s10()

NOT_LIVE = {
    "S2 4-hour / 24-minute": "needs 24-minute candles built from 1-minute data; live feeds only give ~12 hours of it",
    "S5 Levels + MTF divergence scalp": "1-minute scalp; lost at every real fee in testing",
    "S6 1m Bollinger + AO": "1-minute scalp; only profitable at 0% fees",
    "S7 Range levels + 24m/12m": "needs 12/24-minute candles; failed testing (0 of 66 settings worked)",
}


def refresh() -> dict:
    """Add your own strategies and names (livepaper/custom.py) - called at start and on every price check."""
    from .custom import refresh as _refresh
    try:
        from apprentice.strategy import refresh as _apprentice
        _apprentice(REGISTRY)
    except Exception:
        pass
    return _refresh(REGISTRY)


refresh()
