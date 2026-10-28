"""S2 - 4-hour / 24-minute (Market Cipher B + WolfPack ID).

From the "definitive guide to my 4 hour 24 minute strategy" video. The 4h chart
(regular candles) sets the environment, the 24m chart (Heikin Ashi) times the
entry. Longs need, in order:

  environment  4h money flow green
  1  4h trigger wave: a green dot printing a higher low than the last one, money flow green
  2  4h WolfPack ID crossed above zero, lined up with that dot
  3  24m anchor wave (green dot below -60 with money flow red), then a trigger
     wave (a later green dot with a higher low)
  4  24m money flow crossed into the green after the anchor
  5  24m WolfPack ID crossed above zero after the anchor
  (optional 6th: the 4h 21 EMA "ribbon 5" is white)

Exits (his): stop under the last 4h swing low; every new higher swing low
moves it up, but never above the 4h 21 EMA. Take ~20% of the position at each
Market Cipher A red diamond and always leave some. Get out altogether when the
21 EMA turns gray. Shorts are the mirror image.

Approximations: WolfPack ID = EMA(3) - EMA(8) of the close (the free indicator's
default); a red diamond = a 4h WaveTrend cross down above zero; the ribbon is
white when the 4h close is above a rising 21 EMA and gray when it is below a
falling one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import Result, Trade, _stats
from .indicators import ema, heikin_ashi, money_flow, wavetrend

ANCHOR_LEVEL = 60
LINE_UP_BARS = 6        # 4h WolfPack cross may come up to 6 bars (1 day) before the 4h dot
ARMED_BARS = 12         # the 4h confirmation stays fresh for 12 bars (2 days)
ANCHOR_BARS = 120       # 24m: the trigger, money flow and WolfPack must follow within 2 days
PIVOT = 2               # swing low = lowest low of 2 bars either side
STOP_BUFFER = 0.002
PARTIAL = 0.2           # 20% at each red diamond ...
MIN_LEFT = 0.2          # ... always leaving at least 20% in


def wolfpack(close: pd.Series) -> pd.Series:
    return ema(close, 3) - ema(close, 8)


def _waves(df: pd.DataFrame) -> pd.DataFrame:
    wt1, wt2 = wavetrend(df)
    wp, mf = wolfpack(df["close"]), money_flow(df)
    return pd.DataFrame({
        "wt2": wt2, "up": (wt1 > wt2) & (wt1.shift() <= wt2.shift()),
        "down": (wt1 < wt2) & (wt1.shift() >= wt2.shift()),
        "mf": mf, "wp": wp,
        "mf_up": (mf > 0) & (mf.shift() <= 0), "mf_down": (mf < 0) & (mf.shift() >= 0),
        "wp_up": (wp > 0) & (wp.shift() <= 0), "wp_down": (wp < 0) & (wp.shift() >= 0),
    }, index=df.index)


def _bars_since(flag: pd.Series) -> pd.Series:
    n = np.arange(len(flag))
    last = pd.Series(np.where(flag.to_numpy(), n, np.nan), index=flag.index).ffill()
    return pd.Series(n, index=flag.index) - last   # NaN until the first event


def htf_frame(h4: pd.DataFrame) -> pd.DataFrame:
    """4h conditions and exit inputs, one row per 4h bar (known at its close)."""
    w = _waves(h4)
    prev_up = w.wt2.where(w.up).ffill().shift()
    prev_dn = w.wt2.where(w.down).ffill().shift()
    out = pd.DataFrame(index=h4.index)
    out["mf"], out["wp"] = w.mf, w.wp
    out["trig_long"] = w.up & (w.wt2 > prev_up) & (w.mf > 0)
    out["trig_short"] = w.down & (w.wt2 < prev_dn) & (w.mf < 0)
    for side, trig, cross, ok in (("long", "trig_long", "wp_up", (w.wp > 0) & (w.mf > 0)),
                                  ("short", "trig_short", "wp_down", (w.wp < 0) & (w.mf < 0))):
        bt, bw = _bars_since(out[trig]), _bars_since(w[cross])
        out[f"armed_{side}"] = ok & (np.minimum(bt, bw) <= ARMED_BARS) & (bw <= bt + LINE_UP_BARS)
        out[f"bars_trig_{side}"], out[f"bars_wp_{side}"] = bt, bw
    e21 = ema(h4["close"], 21)
    out["ema21"] = e21
    white = (h4["close"] > e21) & (e21 > e21.shift())
    gray = (h4["close"] < e21) & (e21 < e21.shift())
    out["white"], out["gray"] = white, gray
    out["turn_gray"] = gray & ~gray.shift(fill_value=False)
    out["turn_white"] = white & ~white.shift(fill_value=False)
    out["red_diamond"] = w.down & (w.wt2 > 0)
    out["green_diamond"] = w.up & (w.wt2 < 0)
    lo, hi = h4["low"], h4["high"]
    win = 2 * PIVOT + 1
    piv_lo = lo.where(lo == lo.rolling(win, center=True).min()).shift(PIVOT)   # confirmed PIVOT bars later
    piv_hi = hi.where(hi == hi.rolling(win, center=True).max()).shift(PIVOT)
    out["pivot_low"], out["pivot_high"] = piv_lo, piv_hi
    out["trail_long"] = np.minimum(piv_lo.ffill(), e21) * (1 - STOP_BUFFER)
    out["trail_short"] = np.maximum(piv_hi.ffill(), e21) * (1 + STOP_BUFFER)
    return out


def _to_ltf(h: pd.DataFrame, ltf_index: pd.DatetimeIndex, step: pd.Timedelta) -> pd.DataFrame:
    """Put each 4h row on the last lower-timeframe bar inside that 4h bar, so it is
    only used once the 4h candle has closed."""
    shifted = h.copy()
    shifted.index = h.index + pd.Timedelta("4h") - step
    events = [c for c in shifted.columns if shifted[c].dtype == bool and c not in ("armed_long", "armed_short",
                                                                                     "white", "gray")]
    out = shifted.reindex(ltf_index, method="ffill")
    exact = shifted.reindex(ltf_index)
    for c in events + ["pivot_low", "pivot_high"]:
        out[c] = exact[c]
    for c in events:
        out[c] = out[c].fillna(False).astype(bool)
    for c in ("armed_long", "armed_short", "white", "gray"):
        out[c] = out[c].fillna(False).astype(bool)
    return out


def s2_frame(ltf: pd.DataFrame, need_ribbon: bool = False, ha_ltf: bool = True) -> pd.DataFrame:
    """Everything S2 needs on the execution timeframe (24m by default).

    Signal columns long_entry/short_entry are on the bar whose close confirms the
    setup; trades fill at the next bar's open."""
    step = ltf.index[1] - ltf.index[0]
    h4 = ltf.resample("4h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    h = _to_ltf(htf_frame(h4), ltf.index, step)
    w = _waves(heikin_ashi(ltf) if ha_ltf else ltf)

    n = len(ltf)
    out = {f"{s}_{k}": np.zeros(n, dtype=bool) for s in ("long", "short")
           for k in ("entry", "anchor", "trigger", "mf", "wp")}
    up, dn, wt2 = w.up.to_numpy(), w.down.to_numpy(), w.wt2.to_numpy()
    mf, wp = w.mf.to_numpy(), w.wp.to_numpy()
    mf_up, mf_dn, wp_up, wp_dn = (w[k].to_numpy() for k in ("mf_up", "mf_down", "wp_up", "wp_down"))
    armed = {"long": h.armed_long.to_numpy(), "short": h.armed_short.to_numpy()}
    if need_ribbon:
        armed = {"long": armed["long"] & h.white.to_numpy(), "short": armed["short"] & h.gray.to_numpy()}
    spec = {
        "long": (up & (wt2 <= -ANCHOR_LEVEL) & (mf < 0), up, lambda a, b: a > b, mf_up, wp_up, mf > 0, wp > 0),
        "short": (dn & (wt2 >= ANCHOR_LEVEL) & (mf > 0), dn, lambda a, b: a < b, mf_dn, wp_dn, mf < 0, wp < 0),
    }
    for side, (anchor, dot, better, mf_x, wp_x, mf_ok, wp_ok) in spec.items():
        a_i, a_val, trig, mfd, wpd, used = -10**9, 0.0, False, False, False, True
        for i in range(n):
            if anchor[i]:
                a_i, a_val, trig, mfd, wpd, used = i, wt2[i], False, False, False, False
            elif i - a_i <= ANCHOR_BARS:
                trig |= bool(dot[i] and better(wt2[i], a_val))
                mfd |= bool(mf_x[i])
                wpd |= bool(wp_x[i])
            live = i - a_i <= ANCHOR_BARS
            out[f"{side}_anchor"][i] = live
            out[f"{side}_trigger"][i] = live and trig
            out[f"{side}_mf"][i] = live and mfd and mf_ok[i]
            out[f"{side}_wp"][i] = live and wpd and wp_ok[i]
            if live and not used and trig and mfd and wpd and mf_ok[i] and wp_ok[i] and armed[side][i]:
                out[f"{side}_entry"][i] = True
                used = True
    frame = pd.DataFrame(out, index=ltf.index)
    return pd.concat([frame, h.add_prefix("h4_")], axis=1)


@dataclass
class _Pos:
    trade: Trade
    stop: float
    partials: int = 0
    seen_trend: bool = False   # EMA white (gray for shorts) at some point in the trade


def s2_backtest(ltf: pd.DataFrame, frame: pd.DataFrame, *, allow_short: bool = True, fee: float = 0.001,
                start_equity: float = 10_000.0, partial: float = PARTIAL, min_left: float = MIN_LEFT,
                gray_exit: str = "after_white") -> Result:
    """His exits: swing-low stop trailed under new higher lows (capped at the 4h 21
    EMA), 20% off at each red diamond, all out when the 21 EMA turns gray. Stops
    are checked intrabar; everything else fills at the next bar's open.

    gray_exit: "after_white" (default) only exits on a turn to gray once the EMA
    has been white during the trade ("turns gray from white"); "always" exits on
    any turn; "never" leaves exits to the stop and diamonds."""
    o, hi, lo, c = (ltf[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    idx = ltf.index
    f = {k: frame[k].to_numpy() for k in ("long_entry", "short_entry", "h4_trail_long", "h4_trail_short",
                                         "h4_red_diamond", "h4_green_diamond", "h4_turn_gray",
                                         "h4_turn_white", "h4_pivot_low", "h4_pivot_high",
                                         "h4_white", "h4_gray")}
    pivots = {"long": [], "short": []}   # confirmed swing lows / highs seen so far
    equity, curve, trades = start_equity, np.empty(len(ltf)), []
    pos: _Pos | None = None

    def mv(t: Trade, price: float) -> float:
        return (price - t.entry_price) / t.entry_price * (1 if t.side == "long" else -1)

    def exit_part(i, price, frac, reason):
        nonlocal pos, equity
        t = pos.trade
        frac = min(frac, t.remaining)
        t.pnl_pct += frac * (mv(t, price) - 2 * fee)
        t.remaining -= frac
        t.fills.append((idx[i], price, frac, reason))
        if t.remaining <= 1e-9:
            equity *= 1 + t.pnl_pct
            t.exit_time, t.exit_price, t.reason = idx[i], price, reason
            trades.append(t)
            pos = None

    for i in range(len(ltf)):
        if i > 0:
            p = i - 1
            if np.isfinite(f["h4_pivot_low"][p]):
                pivots["long"].append(f["h4_pivot_low"][p])
            if np.isfinite(f["h4_pivot_high"][p]):
                pivots["short"].append(f["h4_pivot_high"][p])
            if pos is not None:
                t = pos.trade
                long_ = t.side == "long"
                trail = f["h4_trail_long" if long_ else "h4_trail_short"][p]
                if np.isfinite(trail) and ((trail < c[p]) if long_ else (trail > c[p])):
                    pos.stop = max(pos.stop, trail) if long_ else min(pos.stop, trail)
                pos.seen_trend |= bool(f["h4_white" if long_ else "h4_gray"][p])
                turned = f["h4_turn_gray" if long_ else "h4_turn_white"][p]
                if turned and (gray_exit == "always" or (gray_exit == "after_white" and pos.seen_trend)):
                    exit_part(i, o[i], t.remaining, "ema turned")
                elif f["h4_red_diamond" if long_ else "h4_green_diamond"][p] and t.remaining - partial >= min_left - 1e-9:
                    pos.partials += 1
                    t.targets_hit += 1
                    exit_part(i, o[i], partial, "diamond")
            if pos is None:
                for side in ("long", "short") if allow_short else ("long",):
                    if not f[f"{side}_entry"][p]:
                        continue
                    e = o[i]
                    below = [x for x in reversed(pivots[side]) if (x < e if side == "long" else x > e)]
                    if not below:
                        continue
                    s0 = below[0] * (1 - STOP_BUFFER if side == "long" else 1 + STOP_BUFFER)
                    pos = _Pos(Trade(side, idx[i], e), s0)
                    break
        if pos is not None:
            t = pos.trade
            hit = lo[i] <= pos.stop if t.side == "long" else hi[i] >= pos.stop
            if hit:
                fill = min(pos.stop, o[i]) if t.side == "long" else max(pos.stop, o[i])
                exit_part(i, fill, t.remaining, "stop" if mv(t, pos.stop) < 0 else "trail stop")
        if pos is not None:
            t = pos.trade
            curve[i] = equity * (1 + t.pnl_pct + t.remaining * (mv(t, c[i]) - fee))
        else:
            curve[i] = equity
    if pos is not None:
        exit_part(len(ltf) - 1, c[-1], pos.trade.remaining, "end")
        curve[-1] = equity
    eq = pd.Series(curve, index=idx)
    return Result(trades, eq, _stats(ltf, trades, eq, start_equity))


LABELS = {
    "long": ["4h money flow green (environment)",
             "4h trigger wave: green dot with a higher low, money flow green",
             "4h WolfPack ID crossed above zero, lined up with the dot",
             "24m anchor wave (green dot below -60, money flow red) then a trigger wave",
             "24m money flow crossed into the green",
             "24m WolfPack ID crossed above zero",
             "(optional) 4h 21 EMA white"],
    "short": ["4h money flow red (environment)",
              "4h trigger wave: red dot with a lower high, money flow red",
              "4h WolfPack ID crossed below zero, lined up with the dot",
              "24m anchor wave (red dot above +60, money flow green) then a trigger wave",
              "24m money flow crossed into the red",
              "24m WolfPack ID crossed below zero",
              "(optional) 4h 21 EMA gray"],
}


def s2_checklist(ltf: pd.DataFrame, bar: int = -1, label: str = "S2 - 4-hour / 24-minute") -> str:
    fr = s2_frame(ltf)
    r = fr.iloc[bar]
    ts = ltf.index[bar] + (ltf.index[1] - ltf.index[0])
    lines = [f"[{label}] Checklist for candle closed {ts:%Y-%m-%d %H:%M} UTC (close {ltf['close'].iloc[bar]:,.2f})"]
    for side, sign in (("long", 1), ("short", -1)):
        h4_wp = r.h4_armed_long if side == "long" else r.h4_armed_short
        h4_dot = h4_wp or r[f"h4_bars_trig_{side}"] <= ARMED_BARS   # the dot is part of a live confirmation
        conds = [sign * r.h4_mf > 0, bool(h4_dot), bool(h4_wp), bool(r[f"{side}_trigger"]),
                 bool(r[f"{side}_mf"]), bool(r[f"{side}_wp"]), bool(r.h4_white if side == "long" else r.h4_gray)]
        go = all(conds[:6])
        lines += ["", f"{side.upper()}: {'🟢 TRADE' if go else '🔴 NO TRADE'}  ({sum(conds[:6])}/6 yes)"]
        for ok, lab in zip(conds, LABELS[side]):
            lines.append(f"  {'🟢 YES' if ok else '🔴 NO '}  {lab}")
    return "\n".join(lines)


# Common strategy interface (see strategies.CATALOG)
signals, backtest, checklist = s2_frame, s2_backtest, s2_checklist
