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
from .indicators import awesome_oscillator, ema, heikin_ashi, money_flow, sma, wavetrend


def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _macd(c: pd.Series):
    line = ema(c, 12) - ema(c, 26)
    return line, ema(line, 9)


def _ao_diff(d: pd.DataFrame) -> pd.Series:
    """AO minus AO one candle ago (the Pine script's `diff`)."""
    return awesome_oscillator(d).diff()


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
    # Awesome Oscillator (TradingView: AO = SMA(hl2, 5) - SMA(hl2, 34); green column = AO rose, red = it didn't)
    "ao_green": ("Awesome Oscillator column is green (AO rising)", None, None, lambda d, a, b: _ao_diff(d) > 0),
    "ao_red": ("Awesome Oscillator column is red (AO falling)", None, None, lambda d, a, b: _ao_diff(d) <= 0),
    "ao_turns_green": ("Awesome Oscillator changed to green (TradingView's 'AO color changed to green' alert)",
                       None, None, lambda d, a, b: _cross_up(_ao_diff(d), 0)),
    "ao_turns_red": ("Awesome Oscillator changed to red (TradingView's 'AO color changed to red' alert)",
                     None, None, lambda d, a, b: _cross_up(-_ao_diff(d), 0)),
    "ao_above_zero": ("Awesome Oscillator above zero", None, None, lambda d, a, b: awesome_oscillator(d) > 0),
    "ao_below_zero": ("Awesome Oscillator below zero", None, None, lambda d, a, b: awesome_oscillator(d) < 0),
    "ao_cross_up_zero": ("Awesome Oscillator crosses above zero", None, None,
                         lambda d, a, b: _cross_up(awesome_oscillator(d), 0)),
    "ao_cross_down_zero": ("Awesome Oscillator crosses below zero", None, None,
                           lambda d, a, b: _cross_up(-awesome_oscillator(d), 0)),
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


# ---- reading the chart: candles, support / resistance, chart patterns -----------------------------
def _body(d):
    return (d.close - d.open).abs()


def _range(d):
    return (d.high - d.low).replace(0, np.nan)


def _engulf(d, up: bool) -> pd.Series:
    po, pc = d.open.shift(), d.close.shift()
    if up:
        return (pc < po) & (d.close > d.open) & (d.close >= po) & (d.open <= pc)
    return (pc > po) & (d.close < d.open) & (d.close <= po) & (d.open >= pc)


def _hammer(d, up: bool) -> pd.Series:
    lower = d[["open", "close"]].min(axis=1) - d.low
    upper = d.high - d[["open", "close"]].max(axis=1)
    small = _body(d) <= 0.35 * _range(d)
    return small & ((lower >= 2 * upper) & (lower >= 0.55 * _range(d)) if up
                    else (upper >= 2 * lower) & (upper >= 0.55 * _range(d)))


_SR_CACHE: dict = {}
_PAT_CACHE: dict = {}


def _key(d) -> tuple:
    return (id(d), len(d), d.index[0] if len(d) else None, d.index[-1] if len(d) else None)


def _sr(d) -> pd.DataFrame:
    from .indicators import sr_levels
    k = _key(d)
    if k not in _SR_CACHE:
        _SR_CACHE.clear()
        _SR_CACHE[k] = sr_levels(d)
    return _SR_CACHE[k]


def pattern_frame(d: pd.DataFrame) -> pd.DataFrame:
    """Per closed candle: the chart pattern price is in (if any), whether it broke up / down on this
    candle, and how close it is to the apex. Uses mcbot.patterns (confirmed swing points only)."""
    k = _key(d)
    if k in _PAT_CACHE:
        return _PAT_CACHE[k]
    from .patterns import Detector
    det = Detector(d)
    n = len(d)
    kind = np.array([""] * n, dtype=object)
    bias = np.array([""] * n, dtype=object)
    brk = np.array([""] * n, dtype=object)
    prog = np.full(n, np.nan)
    prev = None
    for i in range(n):
        p = det.find(i) if i >= 30 else None
        if prev is not None and p is None:          # the pattern of the last candle broke on this one?
            brk[i] = det.broke(prev, i) or ""
            if brk[i]:                              # the break candle still belongs to the pattern that broke
                kind[i], bias[i] = prev.kind, prev.bias or ""
        if p is not None:
            kind[i], bias[i] = p.kind, p.bias or ""
            pr = p.progress(i)
            prog[i] = np.nan if pr is None else pr
        prev = p
    out = pd.DataFrame({"kind": kind, "bias": bias, "broke": brk, "progress": prog}, index=d.index)
    if len(_PAT_CACHE) > 8:
        _PAT_CACHE.clear()
    _PAT_CACHE[k] = out
    return out


RULES.update({
    "bull_engulf": ("Bullish engulfing candle", None, None, lambda d, a, b: _engulf(d, True)),
    "bear_engulf": ("Bearish engulfing candle", None, None, lambda d, a, b: _engulf(d, False)),
    "hammer": ("Hammer / bullish pin bar (long lower wick)", None, None, lambda d, a, b: _hammer(d, True)),
    "shooting_star": ("Shooting star / bearish pin bar (long upper wick)", None, None,
                      lambda d, a, b: _hammer(d, False)),
    "sr_bounce_up": ("Bounced off support / reclaimed a level (closed back above it)", None, None,
                     lambda d, a, b: _sr(d)["sr_reclaim_up"].astype(bool)),
    "sr_reject_down": ("Rejected at resistance / lost a level (closed back below it)", None, None,
                       lambda d, a, b: _sr(d)["sr_reject_down"].astype(bool)),
    "pattern_break_up": ("Chart pattern broke out upward on this candle", None, None,
                         lambda d, a, b: pattern_frame(d).broke.eq("up")),
    "pattern_break_down": ("Chart pattern broke down on this candle", None, None,
                           lambda d, a, b: pattern_frame(d).broke.eq("down")),
    "in_bullish_pattern": ("Inside a pattern that usually breaks up (falling wedge, ascending triangle, bull flag...)",
                           None, None, lambda d, a, b: pattern_frame(d).bias.eq("up")),
    "in_bearish_pattern": ("Inside a pattern that usually breaks down (rising wedge, descending triangle, "
                           "bear flag...)", None, None, lambda d, a, b: pattern_frame(d).bias.eq("down")),
    "near_apex": ("Squeezed near a triangle / wedge apex (A% of the way there or more)", 75, None,
                  lambda d, a, b: pattern_frame(d).progress >= a / 100),
    "squeeze": ("Volatility squeeze: ATR (14) below A × its 50-candle average", 0.7, None,
                lambda d, a, b: (lambda t: t < a * t.rolling(50).mean())(
                    pd.concat([d.high - d.low, (d.high - d.close.shift()).abs(),
                               (d.low - d.close.shift()).abs()], axis=1).max(axis=1).rolling(14).mean())),
    "higher_low": ("Higher low: the last A candles' low is above the A candles before", 5, None,
                   lambda d, a, b: d.low.rolling(int(a)).min() > d.low.shift(int(a)).rolling(int(a)).min()),
    "lower_high": ("Lower high: the last A candles' high is below the A candles before", 5, None,
                   lambda d, a, b: d.high.rolling(int(a)).max() < d.high.shift(int(a)).rolling(int(a)).max()),
})
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
