"""Strategies you build yourself from rules - no code.

A strategy is a name, entry rules for longs (all must be true), optional entry rules for shorts,
optional early-exit rules, and its stop / take profits. Rules use closed candles only.

  spec = {"name": "My RSI dip", "long": [{"rule": "rsi_below", "a": 30}, {"rule": "close_above_ema", "a": 200}],
          "short": [], "exit_long": [{"rule": "rsi_above", "a": 70}], "exit_short": [],
          "stop": 0.04, "take_profits": [[0.06, 0.5], [0.12, 0.5]]}
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fmt import price
from .indicators import ema, heikin_ashi, money_flow, sma, wavetrend


def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _macd(c: pd.Series):
    line = ema(c, 12) - ema(c, 26)
    return line, ema(line, 9)


def _cross_up(a: pd.Series, b) -> pd.Series:
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return (a > b) & (a.shift() <= b.shift())


def _run(mask: pd.Series, n: int) -> pd.Series:
    return mask.astype(int).rolling(max(int(n), 1)).sum() >= max(int(n), 1)


def _n(x, default):
    return default if x is None or (isinstance(x, float) and np.isnan(x)) else x


# rule id -> (label with A / B for its numbers, default A, default B, function(df, a, b) -> bool Series)
RULES = {
    "close_above_ema": ("Close above EMA (A)", 50, None, lambda d, a, b: d.close > ema(d.close, int(a))),
    "close_below_ema": ("Close below EMA (A)", 50, None, lambda d, a, b: d.close < ema(d.close, int(a))),
    "close_above_sma": ("Close above SMA (A)", 200, None, lambda d, a, b: d.close > sma(d.close, int(a))),
    "close_below_sma": ("Close below SMA (A)", 200, None, lambda d, a, b: d.close < sma(d.close, int(a))),
    "ema_cross_up": ("EMA (A) crosses above EMA (B)", 9, 21,
                     lambda d, a, b: _cross_up(ema(d.close, int(a)), ema(d.close, int(b)))),
    "ema_cross_down": ("EMA (A) crosses below EMA (B)", 9, 21,
                       lambda d, a, b: _cross_up(ema(d.close, int(b)), ema(d.close, int(a)))),
    "ema_above_ema": ("EMA (A) is above EMA (B) (uptrend)", 50, 200,
                      lambda d, a, b: ema(d.close, int(a)) > ema(d.close, int(b))),
    "ema_below_ema": ("EMA (A) is below EMA (B) (downtrend)", 50, 200,
                      lambda d, a, b: ema(d.close, int(a)) < ema(d.close, int(b))),
    "rsi_below": ("RSI (14) below A", 30, None, lambda d, a, b: _rsi(d.close) < a),
    "rsi_above": ("RSI (14) above A", 70, None, lambda d, a, b: _rsi(d.close) > a),
    "rsi_cross_up": ("RSI (14) crosses up through A", 30, None, lambda d, a, b: _cross_up(_rsi(d.close), a)),
    "rsi_cross_down": ("RSI (14) crosses down through A", 70, None,
                       lambda d, a, b: _cross_up(-_rsi(d.close), -a)),
    "macd_cross_up": ("MACD crosses above its signal line", None, None, lambda d, a, b: _cross_up(*_macd(d.close))),
    "macd_cross_down": ("MACD crosses below its signal line", None, None,
                        lambda d, a, b: _cross_up(*reversed(_macd(d.close)))),
    "macd_positive": ("MACD above zero", None, None, lambda d, a, b: _macd(d.close)[0] > 0),
    "macd_negative": ("MACD below zero", None, None, lambda d, a, b: _macd(d.close)[0] < 0),
    "wt_green_dot": ("Market Cipher green dot (WaveTrend crosses up) below A", 0, None,
                     lambda d, a, b: (lambda w1, w2: _cross_up(w1, w2) & (w2 < a))(*wavetrend(d))),
    "wt_red_dot": ("Market Cipher red dot (WaveTrend crosses down) above A", 0, None,
                   lambda d, a, b: (lambda w1, w2: _cross_up(w2, w1) & (w2 > a))(*wavetrend(d))),
    "wt_below": ("Market Cipher wave below A (oversold)", -53, None, lambda d, a, b: wavetrend(d)[1] < a),
    "wt_above": ("Market Cipher wave above A (overbought)", 53, None, lambda d, a, b: wavetrend(d)[1] > a),
    "mf_positive": ("Market Cipher money flow above zero (green)", None, None, lambda d, a, b: money_flow(d) > 0),
    "mf_negative": ("Market Cipher money flow below zero (red)", None, None, lambda d, a, b: money_flow(d) < 0),
    "bb_below_lower": ("Close below the lower Bollinger band (A candles, B deviations)", 20, 2,
                       lambda d, a, b: d.close < sma(d.close, int(a)) - b * d.close.rolling(int(a)).std()),
    "bb_above_upper": ("Close above the upper Bollinger band (A candles, B deviations)", 20, 2,
                       lambda d, a, b: d.close > sma(d.close, int(a)) + b * d.close.rolling(int(a)).std()),
    "ha_green": ("Heikin Ashi candle is green", None, None,
                 lambda d, a, b: (lambda h: h.close > h.open)(heikin_ashi(d))),
    "ha_red": ("Heikin Ashi candle is red", None, None, lambda d, a, b: (lambda h: h.close < h.open)(heikin_ashi(d))),
    "green_candles": ("A green candles in a row", 3, None, lambda d, a, b: _run(d.close > d.open, a)),
    "red_candles": ("A red candles in a row", 3, None, lambda d, a, b: _run(d.close < d.open, a)),
    "breakout_high": ("Close above the highest high of the last A candles", 20, None,
                      lambda d, a, b: d.close > d.high.shift().rolling(int(a)).max()),
    "breakdown_low": ("Close below the lowest low of the last A candles", 20, None,
                      lambda d, a, b: d.close < d.low.shift().rolling(int(a)).min()),
    "volume_spike": ("Volume above A × its B-candle average", 1.5, 20,
                     lambda d, a, b: d.volume > a * d.volume.rolling(int(b)).mean()),
    "up_pct": ("Price up more than A% over the last B candles", 5, 10,
               lambda d, a, b: d.close / d.close.shift(int(b)) - 1 > a / 100),
    "down_pct": ("Price down more than A% over the last B candles", 5, 10,
                 lambda d, a, b: d.close / d.close.shift(int(b)) - 1 < -a / 100),
}
LABELS = {k: v[0] for k, v in RULES.items()}
BY_LABEL = {v[0]: k for k, v in RULES.items()}


def rule_series(df: pd.DataFrame, r: dict) -> pd.Series:
    label, da, db, fn = RULES[r["rule"]]
    a, b = _n(r.get("a"), da), _n(r.get("b"), db)
    return fn(df, a, b).fillna(False).astype(bool)


def describe(r: dict) -> str:
    """'RSI (14) below A' + a=30 -> 'RSI (14) below 30'."""
    import re
    label, da, db, _ = RULES[r["rule"]]
    a, b = _n(r.get("a"), da), _n(r.get("b"), db)
    fmt = lambda v: f"{v:g}" if isinstance(v, (int, float)) else str(v)
    if a is not None:
        label = re.sub(r"\bA\b", fmt(a), label)
    if b is not None:
        label = re.sub(r"\bB\b", fmt(b), label)
    return label


def _all(df: pd.DataFrame, rules: list[dict]) -> pd.Series:
    if not rules:
        return pd.Series(False, index=df.index)
    out = pd.Series(True, index=df.index)
    for r in rules:
        out &= rule_series(df, r)
    return out


def _any(df: pd.DataFrame, rules: list[dict]) -> pd.Series:
    out = pd.Series(False, index=df.index)
    for r in rules:
        out |= rule_series(df, r)
    return out


def signals(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    return pd.DataFrame({"long_entry": _all(df, spec.get("long", [])), "short_entry": _all(df, spec.get("short", [])),
                         "long_exit": _any(df, spec.get("exit_long", [])),
                         "short_exit": _any(df, spec.get("exit_short", []))}, index=df.index)


def checklist(df: pd.DataFrame, bar: int = -1, label: str | None = None, spec: dict | None = None) -> str:
    spec = spec or {}
    i = len(df) + bar if bar < 0 else bar
    lines = [f"[{label or spec.get('name', 'Custom')}] Checklist for candle {df.index[i]:%Y-%m-%d %H:%M} UTC "
             f"(close {price(df['close'].iloc[i])})"]
    for side, key in (("LONG", "long"), ("SHORT", "short")):
        rules = spec.get(key, [])
        if not rules:
            continue
        ok = [bool(rule_series(df, r).iloc[i]) for r in rules]
        lines += ["", f"{side}: {'🟢 TRADE' if all(ok) else '🔴 NO TRADE'}  ({sum(ok)}/{len(ok)} yes)"]
        lines += [f"  {'🟢 YES' if o else '🔴 NO '}  {describe(r)}" for o, r in zip(ok, rules)]
    for side, key in (("EXIT LONG", "exit_long"), ("EXIT SHORT", "exit_short")):
        rules = spec.get(key, [])
        if rules:
            ok = [bool(rule_series(df, r).iloc[i]) for r in rules]
            lines += ["", f"{side} (any): {'🔴 GET OUT' if any(ok) else '🟢 STAY IN'}"]
            lines += [f"  {'🔴 YES' if o else '🟢 NO '}  {describe(r)}" for o, r in zip(ok, rules)]
    return "\n".join(lines)
