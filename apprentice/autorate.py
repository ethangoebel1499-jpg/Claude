"""Rate every missed trade and every setup arrow on a chart - for any coin, stock and timeframe.

Judged the way the lessons teach, on what was visible at the time (never on how the trade turned out):
confirmed by a divergence / dot / reversal candle, at a band or support/resistance zone, with the
long-term trend, not chasing the move, a sensible stop. Missed trades also get 👍 (worth taking, 3★+) or
👎 (hindsight only), which is what the apprentice learns from. Ratings are saved once per trade.

  python -m apprentice rate                       # every coin and stock, every timeframe
  python -m apprentice rate --markets BTC:crypto SPY:stock --timeframes 4h 1D
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import brain

BULLC = ["wt_green_dot", "rsi_bull_div", "wt_bull_div", "macd_bull_div", "ao_bull_div", "bull_engulf", "hammer",
         "rsi_cross_up", "macd_cross_up", "sr_bounce_up"]
BEARC = ["wt_red_dot", "rsi_bear_div", "wt_bear_div", "macd_bear_div", "ao_bear_div", "bear_engulf", "shooting_star",
         "rsi_cross_down", "macd_cross_down", "sr_reject_down"]
NAME = {"wt_green_dot": "Market Cipher green dot", "wt_red_dot": "Market Cipher red dot",
        "rsi_bull_div": "RSI bullish divergence", "rsi_bear_div": "RSI bearish divergence",
        "wt_bull_div": "WaveTrend bullish divergence", "wt_bear_div": "WaveTrend bearish divergence",
        "macd_bull_div": "MACD bullish divergence", "macd_bear_div": "MACD bearish divergence",
        "ao_bull_div": "AO bullish divergence", "ao_bear_div": "AO bearish divergence",
        "bull_engulf": "bullish engulfing", "bear_engulf": "bearish engulfing", "hammer": "hammer",
        "shooting_star": "shooting star", "rsi_cross_up": "RSI cross up", "rsi_cross_down": "RSI cross down",
        "macd_cross_up": "MACD cross up", "macd_cross_down": "MACD cross down", "sr_bounce_up": "bounce off support",
        "sr_reject_down": "rejection at resistance"}


class Ctx:
    """The numbers one chart's judgments need, computed once."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.fr = brain.feature_frame(df)
        self.atr = brain.atr_arr(df)
        self.close = df.close.to_numpy(float)
        self.ema = df.close.ewm(span=21).mean().to_numpy()
        self.sma = df.close.rolling(200, min_periods=50).mean().bfill().to_numpy()


def judge(cx: Ctx, i: int, direction: str, stop_atr: float | None = None) -> tuple[int, str]:
    long = direction == "long"
    row = cx.fr.iloc[i]
    conf = [NAME[k] for k in (BULLC if long else BEARC) if k in cx.fr and row[k]]
    zone = bool(row.get("bb_touch_lower" if long else "bb_touch_upper", False)
                or row.get("sr_bounce_up" if long else "sr_reject_down", False))
    c, a = cx.close[i], cx.atr[i]
    trend = c > cx.sma[i] if long else c < cx.sma[i]
    ext = abs(c - cx.ema[i]) / a if a > 0 else 9.0
    chasing = ext > 2.5 and ((c > cx.ema[i]) if long else (c < cx.ema[i]))
    sane = stop_atr is None or 0.7 <= stop_atr <= 3.0
    vol = bool(row.get("volume_spike", False))
    base = (2 if conf else 0) + (1 if zone else 0) + (1 if trend else 0) + (0 if chasing else .5) + (.5 if sane else 0) \
        + (.25 if vol else 0)
    stars = int(min(5, max(1, round(base))))
    if not zone:
        stars = min(stars, 4)               # the lessons: the best trades sit at an earlier support/resistance zone
    if not conf:
        stars = min(stars, 3)
    good, weak = [], []
    (good if conf else weak).append("confirmed by " + ", ".join(conf[:2]) if conf
                                    else "no divergence / dot / reversal candle to confirm it")
    (good if zone else weak).append("at a band or support/resistance zone" if zone else "not at a clear zone")
    (good if trend else weak).append("with the long-term trend" if trend else "against the long-term trend")
    (weak if chasing else good).append(f"chasing: {ext:.1f} ATR from the average" if chasing else "not chasing the move")
    if not sane:
        weak.append(f"a {stop_atr:g} ATR stop is not sensible")
    if vol:
        good.append("volume spike")
    return stars, ("Judged only on what was visible then. " + ("Good: " + "; ".join(good) + ". " if good else "")
                   + ("Weak: " + "; ".join(weak) + "." if weak else "")).strip()


def _done(store, kind: str, ref: str) -> bool:
    return bool(store.ratings(kind, ref))


def rate_chart(store, chart: str, df: pd.DataFrame, setups: list[dict] | None = None) -> dict[str, int]:
    """Rate the missed trades and setup arrows on one chart that haven't been rated yet."""
    setups = setups if setups is not None else store.setups()
    cx = Ctx(df)
    out = {"missed": 0, "arrows": 0}
    rejected = set()
    for m in brain.missed_trades(df.iloc[-1000:], setups, rejected=rejected):
        ref = f"{chart}|{m['time']}|{m['direction']}"
        if _done(store, "missed", ref):
            continue
        stars, why = judge(cx, df.index.get_loc(m["time"]), m["direction"])
        verdict = "good" if stars >= 3 else "bad"
        note = f"🤖 Claude (rule-based): {'👍 Worth taking' if verdict == 'good' else '👎 Hindsight only - do not take'} " \
               f"({stars}★): {why}"
        ctx = {"chart": chart, "time": str(m["time"]), "direction": m["direction"], "by": "Claude"}
        store.feedback(None, verdict, note, ctx)
        store.rate("missed", ref, stars, note, ctx)
        out["missed"] += 1
    start = max(0, len(df) - 400)
    for s in setups:
        if s["status"] not in ("active", "pinned"):
            continue
        mask = brain.dedupe(brain.setup_mask(df, s))
        for i in np.flatnonzero(mask[start:]) + start:
            ref = f"{chart}|{df.index[i]}|{s['id']}"
            if _done(store, "signal", ref):
                continue
            stars, why = judge(cx, int(i), s["direction"], s.get("stop_atr"))
            store.rate("signal", ref, stars, f"🤖 Claude (rule-based): {why}",
                       {"chart": chart, "time": str(df.index[i]), "direction": s["direction"], "setups": [s["id"]],
                        "by": "Claude"})
            out["arrows"] += 1
    return out


def rate_all(store, markets: list[tuple[str, str]], timeframes: list[str], log=print) -> dict[str, int]:
    from livepaper.history import history
    total = {"missed": 0, "arrows": 0, "charts": 0, "skipped": 0}
    setups = store.setups()
    for symbol, asset in markets:
        for tf in timeframes:
            try:
                df = history(symbol, asset, tf).iloc[-3000:]
                if len(df) < 300:
                    raise RuntimeError("too few candles")
                n = rate_chart(store, f"{symbol} {tf}", df, setups)
            except Exception as e:
                total["skipped"] += 1
                log(f"  {symbol} {tf}: skipped ({e})")
                continue
            total["charts"] += 1
            total["missed"] += n["missed"]
            total["arrows"] += n["arrows"]
            log(f"  {symbol} {tf}: {n['missed']} missed trades, {n['arrows']} arrows rated")
    return total
