"""Strategies the paper trader (and the Strategy Lab) can run, with their default exits.

Keys are what live.toml and the runs table use for `strategy`.
"""

from __future__ import annotations

from mcbot import s8
from mcbot.grind import STOP as G_STOP, TAKE_PROFITS as G_TP, grind_checklist, grind_signals
from mcbot.indicators import market_cipher
from mcbot.strategies import build_signals, checklist as topdown_checklist
from mcbot.witcher import STOP_LONG as W_STOP, TAKE_PROFIT as W_TP, witcher_checklist, witcher_signals

from .core import Exits


def _topdown_signals(df):
    return build_signals(market_cipher(df, use_heikin_ashi=True), "S1")


def _topdown_checklist(df, bar=-1, label=None):
    return topdown_checklist(market_cipher(df, use_heikin_ashi=True), bar)


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
}

NOT_LIVE = {
    "S2 4-hour / 24-minute": "needs 24-minute candles built from 1-minute data; live feeds only give ~12 hours of it",
    "S5 Levels + MTF divergence scalp": "1-minute scalp; lost at every real fee in testing",
    "S6 1m Bollinger + AO": "1-minute scalp; only profitable at 0% fees",
    "S7 Range levels + 24m/12m": "needs 12/24-minute candles; failed testing (0 of 66 settings worked)",
}
