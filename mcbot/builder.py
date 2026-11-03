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


def _bb(d: pd.DataFrame, n: int = 20, k: float = 2.0):
    """(lower, middle, upper) Bollinger bands, as TradingView draws them."""
    mid = sma(d.close, n)
    sd = d.close.rolling(n).std(ddof=0)
    return mid - k * sd, mid, mid + k * sd


def recent_flag(flag: pd.Series, bars) -> pd.Series:
    """True if `flag` was true on any of the last `bars` candles (this one included)."""
    return flag.astype(int).rolling(max(int(bars), 1), min_periods=1).max().astype(bool)


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
    "bb_touch_lower": ("Low touched the lower Bollinger band (20, 2) in the last A candles", 3, None,
                       lambda d, a, b: recent_flag(d.low <= _bb(d)[0], a)),
    "bb_touch_upper": ("High touched the upper Bollinger band (20, 2) in the last A candles", 3, None,
                       lambda d, a, b: recent_flag(d.high >= _bb(d)[2], a)),
    "bb_above_mid": ("Close above the Bollinger middle band (20-candle average)", None, None,
                     lambda d, a, b: d.close > _bb(d)[1]),
    "bb_below_mid": ("Close below the Bollinger middle band (20-candle average)", None, None,
                     lambda d, a, b: d.close < _bb(d)[1]),
    "bb_width_above": ("Bollinger bands wider than A% of price (enough room to trade)", 0.4, None,
                       lambda d, a, b: (lambda lo, mid, up: (up - lo) / mid * 100 > a)(*_bb(d))),
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


# ---- divergences: price makes a higher high / lower low but the oscillator doesn't ----------------
# An oscillator wave is a trough (from a turn down to the next turn up) or a peak. Turns: RSI crossing its
# 9-candle average, AO changing colour, MACD crossing its signal line, Market Cipher's green / red dots.
# The two latest completed troughs (peaks) are compared on the candle the second one completes, so nothing
# looks ahead. mcbot/divergence.py does the work.
_DIV_CACHE: dict = {}
OSCILLATORS = {"rsi": "RSI", "ao": "Awesome Oscillator", "macd": "MACD", "wt": "Market Cipher wave"}


# ---- per-rule settings (the ⚙️ panel in the Builder) ------------------------------------------------
# A rule may carry {"set": {...}}; anything left out uses the default. kinds: int / float / choice / bool
SOURCES = ["close", "open", "high", "low", "hl2", "hlc3", "ohlc4"]
BB_SETTINGS = [("length", "Length", 20, "int", 2, 500), ("mult", "StdDev", 2.0, "float", 0.1, 10.0),
               ("source", "Source", "close", "choice", SOURCES), ("basis", "Basis MA type", "SMA", "choice",
                                                                    ["SMA", "EMA"])]
AO_SETTINGS = [("fast", "Fast length", 5, "int", 1, 200), ("slow", "Slow length", 34, "int", 2, 500)]
OSC_SETTINGS = {
    "rsi": [("rsi_length", "RSI length", 14, "int", 2, 200), ("rsi_source", "RSI source", "close", "choice", SOURCES),
            ("rsi_signal", "Turn = RSI crossing its average of (candles)", 9, "int", 2, 100)],
    "ao": AO_SETTINGS,
    "macd": [("macd_fast", "MACD fast length", 12, "int", 1, 200), ("macd_slow", "MACD slow length", 26, "int", 2, 500),
             ("macd_signal", "MACD signal length", 9, "int", 1, 100)],
    "wt": [("wt_channel", "Channel length", 9, "int", 1, 100), ("wt_average", "Average length", 12, "int", 1, 100)],
}
DIV_SETTINGS = [
    ("method", "Find the lows / highs by", "turns", "choice", ["turns", "pivots"]),
    ("left", "Pivot lookback left (pivots)", 5, "int", 1, 50),
    ("right", "Pivot lookback right (pivots - candles to confirm)", 5, "int", 1, 50),
    ("min_gap", "Min of lookback range (candles between the two lows / highs; TradingView uses 5)", 1, "int", 1, 500),
    ("zone", "Bullish only when the oscillator is below its midline, bearish only above", False, "bool"),
    ("min_move", "Price must make the new low / high by at least (%)", 0.0, "float", 0.0, 20.0),
]
SETTINGS: dict[str, list] = {}
_SET_FN: dict = {}


def options(r: dict) -> dict:
    """The rule's settings with defaults filled in."""
    return {k: (r.get("set") or {}).get(k, spec[2]) for spec in SETTINGS.get(r["rule"], []) for k in [spec[0]]}


def _src(d: pd.DataFrame, name: str) -> pd.Series:
    return {"hl2": (d.high + d.low) / 2, "hlc3": (d.high + d.low + d.close) / 3,
            "ohlc4": (d.open + d.high + d.low + d.close) / 4}.get(name, d[name] if name in d else d.close)


def _bbx(d: pd.DataFrame, o: dict):
    """(lower, middle, upper) with TradingView's Bollinger settings."""
    src = _src(d, o.get("source", "close"))
    n = int(o.get("length", 20))
    mid = ema(src, n) if o.get("basis") == "EMA" else sma(src, n)
    sd = src.rolling(n).std(ddof=0)
    k = float(o.get("mult", 2.0))
    return mid - k * sd, mid, mid + k * sd


def _aox(d: pd.DataFrame, o: dict) -> pd.Series:
    return awesome_oscillator(d, int(o.get("fast", 5)), int(o.get("slow", 34)))


def _turns(d: pd.DataFrame, osc: str, o: dict | None = None):
    """(oscillator, turned up, turned down, midline)."""
    o = o or {}
    if osc == "rsi":
        r = _rsi(_src(d, o.get("rsi_source", "close")), int(o.get("rsi_length", 14)))
        avg = sma(r, int(o.get("rsi_signal", 9)))
        return r, _cross_up(r, avg), _cross_up(avg, r), 50.0
    if osc == "ao":
        ao = _aox(d, o)
        diff = ao.diff()
        return ao, _cross_up(diff, 0), _cross_up(-diff, 0), 0.0
    if osc == "macd":
        c = d.close
        line = ema(c, int(o.get("macd_fast", 12))) - ema(c, int(o.get("macd_slow", 26)))
        sig = ema(line, int(o.get("macd_signal", 9)))
        return line, _cross_up(line, sig), _cross_up(sig, line), 0.0
    w1, w2 = wavetrend(d, int(o.get("wt_channel", 9)), int(o.get("wt_average", 12)))
    return w2, _cross_up(w1, w2), _cross_up(w2, w1), 0.0


def _pivot_divergences(osc: pd.Series, d: pd.DataFrame, left: int, right: int, lo_gap: int, hi_gap: int,
                       mid: float, zone: bool, move: float) -> pd.DataFrame:
    """TradingView's divergence method: oscillator pivot lows / highs (`left` candles before, `right` after);
    compare each with the previous one between lo_gap and hi_gap candles earlier, using price's low / high
    on those same candles. Flagged `right` candles later, when the pivot is confirmed."""
    o = osc.to_numpy(float)
    lo, hi = d.low.to_numpy(float), d.high.to_numpy(float)
    n, w = len(o), left + right + 1
    s = pd.Series(o)
    is_lo = (s == s.rolling(w).min()).shift(-right).fillna(False).to_numpy() if n else np.array([], bool)
    is_hi = (s == s.rolling(w).max()).shift(-right).fillna(False).to_numpy() if n else np.array([], bool)
    out = {k: np.zeros(n, bool) for k in ("bull", "hidden_bull", "bear", "hidden_bear")}
    for piv, name, hname, px, sgn in ((is_lo, "bull", "hidden_bull", lo, 1), (is_hi, "bear", "hidden_bear", hi, -1)):
        idx = [i for i in np.flatnonzero(piv) if i + right < n]
        for a, j in enumerate(idx):
            prev = [p for p in idx[:a] if lo_gap <= j - p <= hi_gap]
            if not prev:
                continue
            p = prev[-1]
            if zone and (sgn * (o[j] - mid) > 0 or sgn * (o[p] - mid) > 0):
                continue
            new_extreme = sgn * (px[p] - px[j]) > px[p] * move / 100        # lower low (higher high), by move%
            weaker_osc = sgn * (o[j] - o[p]) > 0                            # oscillator higher low (lower high)
            if new_extreme and weaker_osc:
                out[name][j + right] = True
            elif sgn * (px[j] - px[p]) > px[p] * move / 100 and sgn * (o[p] - o[j]) > 0:
                out[hname][j + right] = True
    return pd.DataFrame(out, index=d.index)


def divergences(d: pd.DataFrame, osc: str, max_gap: int = 60, o: dict | None = None) -> pd.DataFrame:
    """Columns bull / hidden_bull / bear / hidden_bear, True on the candle the divergence is confirmed."""
    from .divergence import pair_divergences, waves
    o = o or {}
    k = (*_key(d), osc, int(max_gap), tuple(sorted(o.items())))
    if k not in _DIV_CACHE:
        if len(_DIV_CACHE) > 32:
            _DIV_CACHE.clear()
        x, up, down, mid = _turns(d, osc, o)
        up, down = up.fillna(False), down.fillna(False)
        if o.get("method") == "pivots":
            res = _pivot_divergences(x, d, int(o.get("left", 5)), int(o.get("right", 5)), int(o.get("min_gap", 1)),
                                     int(max_gap), mid, bool(o.get("zone")), float(o.get("min_move", 0)))
        else:
            w = waves(x, d.low, d.high, up, down)
            res = pair_divergences(w, up, down, int(max_gap))
            gap_t, gap_p = w.t1_i - w.t0_i, w.p1_i - w.p0_i
            min_gap = int(o.get("min_gap", 1))
            res["bull"] &= gap_t >= min_gap
            res["hidden_bull"] &= gap_t >= min_gap
            res["bear"] &= gap_p >= min_gap
            res["hidden_bear"] &= gap_p >= min_gap
            mv = float(o.get("min_move", 0)) / 100
            if mv:
                res["bull"] &= w.t1_px < w.t0_px * (1 - mv)
                res["bear"] &= w.p1_px > w.p0_px * (1 + mv)
                res["hidden_bull"] &= w.t1_px > w.t0_px * (1 + mv)
                res["hidden_bear"] &= w.p1_px < w.p0_px * (1 - mv)
            if o.get("zone"):
                res["bull"] &= (w.t1_osc < mid) & (w.t0_osc < mid)
                res["hidden_bull"] &= (w.t1_osc < mid) & (w.t0_osc < mid)
                res["bear"] &= (w.p1_osc > mid) & (w.p0_osc > mid)
                res["hidden_bear"] &= (w.p1_osc > mid) & (w.p0_osc > mid)
            res = res.fillna(False).astype(bool)
        _DIV_CACHE[k] = res
    return _DIV_CACHE[k]


def _div_rule(osc: str, kind: str):
    def fn(d, a, b, o=None):
        return recent_flag(divergences(d, osc, b, o)[kind], a)
    return fn


_DIV_TEXT = {"bull": "Bullish divergence: price lower low, {o} higher low",
             "bear": "Bearish divergence: price higher high, {o} lower high",
             "hidden_bull": "Hidden bullish divergence: price higher low, {o} lower low (trend continuation)",
             "hidden_bear": "Hidden bearish divergence: price lower high, {o} higher high (trend continuation)"}
for _o, _name in OSCILLATORS.items():
    for _kind, _text in _DIV_TEXT.items():
        RULES[f"{_o}_{_kind}_div"] = (_text.format(o=_name) + " - in the last A candles, lows/highs at most B "
                                      "candles apart", 3, 60, _div_rule(_o, _kind))

# rules that take settings: their function gets the settings dict as a 4th argument
def _bb_rule(fn):
    return lambda d, a, b, o=None: fn(d, a, b, *_bbx(d, o or {}))


_BB_FNS = {
    "bb_touch_lower": _bb_rule(lambda d, a, b, lo, mid, up: recent_flag(d.low <= lo, a)),
    "bb_touch_upper": _bb_rule(lambda d, a, b, lo, mid, up: recent_flag(d.high >= up, a)),
    "bb_above_mid": _bb_rule(lambda d, a, b, lo, mid, up: d.close > mid),
    "bb_below_mid": _bb_rule(lambda d, a, b, lo, mid, up: d.close < mid),
    "bb_width_above": _bb_rule(lambda d, a, b, lo, mid, up: (up - lo) / mid * 100 > a),
}
for _k, _f in _BB_FNS.items():
    RULES[_k] = (RULES[_k][0].replace(" (20, 2)", "").replace(" (20-candle average)", ""), RULES[_k][1], RULES[_k][2], _f)
    SETTINGS[_k] = BB_SETTINGS
_AO_FNS = {
    "ao_green": lambda d, a, b, o=None: _aox(d, o or {}).diff() > 0,
    "ao_red": lambda d, a, b, o=None: _aox(d, o or {}).diff() <= 0,
    "ao_turns_green": lambda d, a, b, o=None: _cross_up(_aox(d, o or {}).diff(), 0),
    "ao_turns_red": lambda d, a, b, o=None: _cross_up(-_aox(d, o or {}).diff(), 0),
    "ao_above_zero": lambda d, a, b, o=None: _aox(d, o or {}) > 0,
    "ao_below_zero": lambda d, a, b, o=None: _aox(d, o or {}) < 0,
    "ao_cross_up_zero": lambda d, a, b, o=None: _cross_up(_aox(d, o or {}), 0),
    "ao_cross_down_zero": lambda d, a, b, o=None: _cross_up(-_aox(d, o or {}), 0),
}
for _k, _f in _AO_FNS.items():
    RULES[_k] = (*RULES[_k][:3], _f)
    SETTINGS[_k] = AO_SETTINGS
for _o in OSCILLATORS:
    for _kind in _DIV_TEXT:
        SETTINGS[f"{_o}_{_kind}_div"] = OSC_SETTINGS[_o] + DIV_SETTINGS

LABELS = {k: v[0] for k, v in RULES.items()}
BY_LABEL = {v[0]: k for k, v in RULES.items()}

# what A and B mean, in plain words, for the Builder (rules not listed: their label says it)
_DIV_AB = ("how many candles it stays true after it's confirmed", "most candles between the two lows / highs")
AB_HELP = {
    "close_above_ema": ("EMA length", None), "close_below_ema": ("EMA length", None),
    "close_above_sma": ("SMA length", None), "close_below_sma": ("SMA length", None),
    "ema_cross_up": ("fast EMA length", "slow EMA length"), "ema_cross_down": ("fast EMA length", "slow EMA length"),
    "ema_above_ema": ("fast EMA length", "slow EMA length"), "ema_below_ema": ("fast EMA length", "slow EMA length"),
    "rsi_below": ("RSI level", None), "rsi_above": ("RSI level", None),
    "rsi_cross_up": ("RSI level", None), "rsi_cross_down": ("RSI level", None),
    "wt_green_dot": ("only dots below this wave level", None), "wt_red_dot": ("only dots above this wave level", None),
    "wt_below": ("wave level", None), "wt_above": ("wave level", None),
    "bb_below_lower": ("Bollinger length", "StdDev"), "bb_above_upper": ("Bollinger length", "StdDev"),
    "bb_touch_lower": ("touch still counts for this many candles", None),
    "bb_touch_upper": ("touch still counts for this many candles", None),
    "bb_width_above": ("band width, % of price", None),
    "green_candles": ("candles in a row", None), "red_candles": ("candles in a row", None),
    "breakout_high": ("candles to look back", None), "breakdown_low": ("candles to look back", None),
    "volume_spike": ("times the average", "average over this many candles"),
    "up_pct": ("% move", "over this many candles"), "down_pct": ("% move", "over this many candles"),
    "near_apex": ("% of the way to the apex", None), "squeeze": ("times its 50-candle average", None),
    "higher_low": ("candles per window", None), "lower_high": ("candles per window", None),
    **{f"{o}_{k}_div": _DIV_AB for o in OSCILLATORS for k in _DIV_TEXT},
}


def ab_help(rid: str) -> str:
    """'A = EMA length (default 50)' - for the Builder's rule list."""
    _, da, db, _ = RULES[rid]
    names = AB_HELP.get(rid)
    parts = []
    for letter, d, name in (("A", da, names[0] if names else None), ("B", db, names[1] if names else None)):
        if d is None and name is None:
            continue
        parts.append(f"{letter} = {name or 'the ' + letter + ' in the rule'}" + (f" (default {d:g})" if d is not None else ""))
    return " · ".join(parts) if parts else "no numbers to set"


def rule_series(df: pd.DataFrame, r: dict) -> pd.Series:
    label, da, db, fn = RULES[r["rule"]]
    a, b = _n(r.get("a"), da), _n(r.get("b"), db)
    out = fn(df, a, b, options(r)) if r["rule"] in SETTINGS else fn(df, a, b)
    return out.fillna(False).astype(bool)


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
    changed = [f"{spec[1].split(' (')[0].lower()} {fmt(r['set'][spec[0]])}" for spec in SETTINGS.get(r["rule"], [])
               if spec[0] in (r.get("set") or {}) and r["set"][spec[0]] != spec[2]]
    return label + (f" [{', '.join(changed)}]" if changed else "")


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
