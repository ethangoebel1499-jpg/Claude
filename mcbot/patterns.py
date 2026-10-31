"""Chart pattern recognition: find the pattern, say which situation price is in, and what to do.

Patterns are drawn from confirmed swing points (a swing high needs K candles on each side, so
it is only known K candles later - nothing here looks ahead). Two trendlines are fitted, one
through the swing highs and one through the swing lows, and their slopes name the pattern:

  top line   bottom line   converging   ->  pattern
  flat       rising        yes              ascending triangle      (usually breaks up)
  falling    flat          yes              descending triangle     (usually breaks down)
  falling    rising        yes              symmetrical triangle    (usually breaks with the prior trend)
  rising     rising        yes              rising wedge            (usually breaks down)
  falling    falling       yes              falling wedge           (usually breaks up)
  rising     rising        parallel         ascending channel
  falling    falling       parallel         descending channel
  flat       flat          parallel         range
  a short channel / triangle after a sharp move ("pole")  ->  bull / bear flag, pennant
  plus double top / bottom and head & shoulders (and inverse) from the last swing points.

`situation()` then says where price is in the pattern - early, in the breakout zone, near the
apex (where the lines meet), at a line, broken (with or against the usual direction), a fake
breakout, a retest, target reached - with the levels to act on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .fmt import price

K = 3                 # candles each side of a swing point
MAX_WIDTH = 160       # longest pattern, in candles
MIN_WIDTH = 12
BREAK = 0.15          # a close this many ATRs beyond a line counts as a break

NAMES = {
    "ascending_triangle": "Ascending triangle", "descending_triangle": "Descending triangle",
    "symmetrical_triangle": "Symmetrical triangle", "rising_wedge": "Rising wedge", "falling_wedge": "Falling wedge",
    "ascending_channel": "Ascending channel", "descending_channel": "Descending channel", "range": "Range",
    "bull_flag": "Bull flag", "bear_flag": "Bear flag", "pennant": "Pennant", "broadening": "Broadening pattern",
    "double_top": "Double top", "double_bottom": "Double bottom", "head_shoulders": "Head and shoulders",
    "inverse_head_shoulders": "Inverse head and shoulders",
}
CONVERGING = {"ascending_triangle", "descending_triangle", "symmetrical_triangle", "rising_wedge", "falling_wedge",
              "pennant"}
REVERSALS = {"double_top", "double_bottom", "head_shoulders", "inverse_head_shoulders"}
FIXED_BIAS = {"ascending_triangle": "up", "descending_triangle": "down", "rising_wedge": "down",
              "falling_wedge": "up", "ascending_channel": "up", "descending_channel": "down", "bull_flag": "up",
              "bear_flag": "down", "double_top": "down", "double_bottom": "up", "head_shoulders": "down",
              "inverse_head_shoulders": "up", "range": None, "broadening": None}


@dataclass
class Pattern:
    kind: str
    start: int                       # first swing point (bar number)
    found: int                       # bar it was recognised on
    upper: tuple[float, float]       # line = slope * bar + intercept
    lower: tuple[float, float]
    atr: float
    bias: str | None                 # "up" / "down" / None
    highs: list[int] = field(default_factory=list)
    lows: list[int] = field(default_factory=list)
    head: float | None = None        # head & shoulders / double top: the extreme, for the measured move
    trend: str | None = None         # price higher / lower than 50 candles before the pattern started

    def up(self, i: float) -> float:
        return self.upper[0] * i + self.upper[1]

    def lo(self, i: float) -> float:
        return self.lower[0] * i + self.lower[1]

    @property
    def name(self) -> str:
        return NAMES[self.kind]

    @property
    def height(self) -> float:
        """Size used for the measured-move target: widest part of the pattern."""
        if self.head is not None:
            return abs(self.head - (self.lo(self.start) if self.bias == "down" else self.up(self.start)))
        return max(self.up(self.start) - self.lo(self.start), self.up(self.found) - self.lo(self.found))

    @property
    def apex(self) -> float | None:
        """Bar where the two lines meet (converging patterns only)."""
        ds = self.upper[0] - self.lower[0]
        if self.kind not in CONVERGING or ds >= 0:
            return None
        return (self.lower[1] - self.upper[1]) / ds

    def progress(self, i: int) -> float | None:
        """0 at the first swing point, 1 at the apex."""
        a = self.apex
        return None if a is None or a <= self.start else (i - self.start) / (a - self.start)

    def target(self, direction: str, at: int) -> float:
        base = self.up(at) if direction == "up" else self.lo(at)
        return base + self.height if direction == "up" else base - self.height


# ---- building blocks --------------------------------------------------------------
def atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, lo, c = df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float)
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - lo, np.maximum(abs(h - pc), abs(lo - pc)))
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def swings(df: pd.DataFrame, k: int = K) -> tuple[np.ndarray, np.ndarray]:
    """Bar numbers of swing highs and lows (each known k candles after it happens)."""
    h, lo = df["high"], df["low"]
    w = 2 * k + 1
    hi = np.flatnonzero((h == h.rolling(w, center=True).max()).to_numpy())
    lw = np.flatnonzero((lo == lo.rolling(w, center=True).min()).to_numpy())
    # equal highs / lows within one window are one swing point (keep the first)
    keep = lambda idx: idx[np.r_[True, np.diff(idx) > k]] if len(idx) else idx
    return keep(hi), keep(lw)


class Detector:
    """Finds patterns on one chart; precomputes what every bar needs."""

    def __init__(self, df: pd.DataFrame, k: int = K):
        self.df, self.k = df, k
        self.h, self.l = df["high"].to_numpy(float), df["low"].to_numpy(float)
        self.c, self.o = df["close"].to_numpy(float), df["open"].to_numpy(float)
        self.atr = atr(df)
        self.hi, self.lw = swings(df, k)

    def _known(self, idx: np.ndarray, end: int) -> np.ndarray:
        return idx[(idx + self.k <= end) & (idx >= end - MAX_WIDTH)]

    @staticmethod
    def _fit(x, y):
        if len(x) < 2:
            return None
        s, b = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)
        return float(s), float(b)

    def find(self, end: int) -> Pattern | None:
        """The pattern price is inside of at bar `end` (closed candle), if any."""
        if end < 30:
            return None
        a = self.atr[end]
        hs, ls = self._known(self.hi, end), self._known(self.lw, end)
        best = None
        for nh in (4, 3, 2):
            for nl in (4, 3, 2):
                if len(hs) < nh or len(ls) < nl:
                    continue
                p = self._lines(hs[-nh:], ls[-nl:], end, a)
                if p and (best is None or len(p.highs) + len(p.lows) > len(best.highs) + len(best.lows)):
                    best = p
            if best:
                break
        if best is None or len(best.highs) + len(best.lows) < 5:   # well-touched lines beat a 2-point double top
            best = self._reversal(hs, ls, end, a) or best
        if best is not None:
            best.trend = self._trend(best.start)
        return best

    def _lines(self, hs, ls, end: int, a: float) -> Pattern | None:
        start = int(min(hs[0], ls[0]))
        width = end - start
        if width < MIN_WIDTH or hs[-1] < ls[0] or ls[-1] < hs[0]:        # highs and lows must interleave
            return None
        up, lo = self._fit(hs, self.h[hs]), self._fit(ls, self.l[ls])
        line_u = lambda i: up[0] * i + up[1]
        line_l = lambda i: lo[0] * i + lo[1]
        # every swing point close to its line
        if max(abs(self.h[hs] - line_u(hs))) > 0.6 * a or max(abs(self.l[ls] - line_l(ls))) > 0.6 * a:
            return None
        h0, h1 = line_u(start) - line_l(start), line_u(end) - line_l(end)
        if h0 < 1.5 * a or h1 <= 0:
            return None
        # price stayed inside the lines while the pattern formed, and is inside now
        seg = np.arange(start, end + 1)
        tol = BREAK * a
        if (self.c[seg] > line_u(seg) + tol).any() or (self.c[seg] < line_l(seg) - tol).any():
            return None
        du, dl = up[0] * width, lo[0] * width
        flat = 1.0 * a
        u = "flat" if abs(du) < flat else "up" if du > 0 else "down"
        d = "flat" if abs(dl) < flat else "up" if dl > 0 else "down"
        ratio = h1 / h0
        shape = "converging" if ratio < 0.75 else "diverging" if ratio > 1.4 else "parallel"
        kind = {("flat", "up", "converging"): "ascending_triangle",
                ("down", "flat", "converging"): "descending_triangle",
                ("down", "up", "converging"): "symmetrical_triangle",
                ("up", "up", "converging"): "rising_wedge",
                ("down", "down", "converging"): "falling_wedge",
                ("up", "up", "parallel"): "ascending_channel",
                ("down", "down", "parallel"): "descending_channel",
                ("flat", "flat", "parallel"): "range"}.get((u, d, shape))
        if shape == "diverging":
            if len(hs) + len(ls) < 6:        # widening swings are only a pattern with plenty of touches
                return None
            kind = "broadening"
        if kind is None:
            return None
        # a short pattern right after a sharp move is a flag / pennant
        pole = self._pole(start, h0)
        if pole and width <= 40:
            if kind in ("descending_channel", "falling_wedge", "range") and pole == "up":
                kind = "bull_flag"
            elif kind in ("ascending_channel", "rising_wedge", "range") and pole == "down":
                kind = "bear_flag"
            elif kind == "symmetrical_triangle":
                kind = "pennant"
        bias = FIXED_BIAS.get(kind)
        if kind in ("symmetrical_triangle", "pennant"):
            bias = pole or self._trend(start)
        return Pattern(kind, start, end, up, lo, a, bias, list(map(int, hs)), list(map(int, ls)))

    def _pole(self, start: int, h0: float) -> str | None:
        i0 = max(start - 15, 0)
        move = self.c[start] - self.c[i0]
        return ("up" if move > 0 else "down") if abs(move) >= 2.5 * h0 else None

    def _trend(self, start: int) -> str:
        i0 = max(start - 50, 0)
        return "up" if self.c[start] >= self.c[i0] else "down"

    def _reversal(self, hs, ls, end: int, a: float) -> Pattern | None:
        """Double top / bottom and head & shoulders from the last swing points."""
        c = self.c[end]
        if len(hs) >= 3 and len(ls) >= 2:            # head and shoulders
            l_, hd, r_ = hs[-3:]
            t1, t2 = [x for x in ls if l_ < x < hd][-1:], [x for x in ls if hd < x < r_][-1:]
            if t1 and t2 and self.h[hd] > max(self.h[l_], self.h[r_]) + 0.8 * a \
                    and abs(self.h[l_] - self.h[r_]) < 1.2 * a:
                neck = self._fit([t1[0], t2[0]], self.l[[t1[0], t2[0]]])
                if c > neck[0] * end + neck[1] and end - r_ < 25:
                    top = (0.0, max(self.h[l_], self.h[r_]))
                    return Pattern("head_shoulders", int(l_), end, top, neck, a, "down", [int(l_), int(hd), int(r_)],
                                   [int(t1[0]), int(t2[0])], head=float(self.h[hd]))
        if len(ls) >= 3 and len(hs) >= 2:            # inverse head and shoulders
            l_, hd, r_ = ls[-3:]
            p1, p2 = [x for x in hs if l_ < x < hd][-1:], [x for x in hs if hd < x < r_][-1:]
            if p1 and p2 and self.l[hd] < min(self.l[l_], self.l[r_]) - 0.8 * a \
                    and abs(self.l[l_] - self.l[r_]) < 1.2 * a:
                neck = self._fit([p1[0], p2[0]], self.h[[p1[0], p2[0]]])
                if c < neck[0] * end + neck[1] and end - r_ < 25:
                    bot = (0.0, min(self.l[l_], self.l[r_]))
                    return Pattern("inverse_head_shoulders", int(l_), end, neck, bot, a, "up",
                                   [int(p1[0]), int(p2[0])], [int(l_), int(hd), int(r_)], head=float(self.l[hd]))
        if len(hs) >= 2 and len(ls) >= 1:            # double top
            p1, p2 = hs[-2:]
            mid = [x for x in ls if p1 < x < p2]
            if mid and p2 - p1 >= 8 and abs(self.h[p1] - self.h[p2]) < 0.6 * a and end - p2 < 20:
                neck = float(self.l[mid].min())
                top = max(self.h[p1], self.h[p2])
                if top - neck > 2.5 * a and neck < c < top:
                    return Pattern("double_top", int(p1), end, (0.0, float(top)), (0.0, neck), a, "down",
                                   [int(p1), int(p2)], [int(mid[int(np.argmin(self.l[mid]))])], head=float(top))
        if len(ls) >= 2 and len(hs) >= 1:            # double bottom
            p1, p2 = ls[-2:]
            mid = [x for x in hs if p1 < x < p2]
            if mid and p2 - p1 >= 8 and abs(self.l[p1] - self.l[p2]) < 0.6 * a and end - p2 < 20:
                neck = float(self.h[mid].max())
                bot = min(self.l[p1], self.l[p2])
                if neck - bot > 2.5 * a and bot < c < neck:
                    return Pattern("double_bottom", int(p1), end, (0.0, neck), (0.0, float(bot)), a, "up",
                                   [int(mid[int(np.argmax(self.h[mid]))])], [int(p1), int(p2)], head=float(bot))
        return None

    # ---- breaks ---------------------------------------------------------------------
    def broke(self, p: Pattern, i: int) -> str | None:
        tol = BREAK * p.atr
        if self.c[i] > p.up(i) + tol:
            return "up"
        if self.c[i] < p.lo(i) - tol:
            return "down"
        return None

    def last_break(self, end: int, lookback: int = 40):
        """(pattern, break bar, direction) for the most recent break within `lookback` candles."""
        for b in range(end, max(end - lookback, 31), -1):
            p = self.find(b - 1)
            if p is not None:
                d = self.broke(p, b)
                if d:
                    return p, b, d
        return None


# ---- published figures (Bulkowski, US stocks) - only ones confirmed from his site's text ----
PUBLISHED = {
    "ascending_triangle": {"up": 63, "text": "breaks up 63%"},
    "descending_triangle": {"up": 53, "text": "breaks up 53%"},
    "rising_wedge": {"up": 40, "text": "breaks down 60% (\"close to random\")"},
}


# ---- what actually happened (tools/pattern_stats.py) -------------------------------------
_STATS = None


def stats() -> dict:
    global _STATS
    if _STATS is None:
        import json
        from pathlib import Path
        f = Path(__file__).with_name("pattern_stats.json")
        _STATS = json.loads(f.read_text()) if f.exists() else {}
    return _STATS


def _p(x) -> str:
    return "?" if x is None else f"{x:.0f}%"


def odds_note(kind: str, trend: str | None = None) -> str:
    """'Breaks up 63% of the time (58-68%), 71% after an uptrend' style sentence."""
    st = stats().get(kind, {})
    o = st.get("odds")
    if not o or not o.get("n"):
        return ""
    out = f"Chance of breaking UP on crypto: {o['up']:.0f}% (likely {o['lo']:.0f}-{o['hi']:.0f}%, {o['n']} breaks)"
    t = (st.get("odds_trend") or {}).get(trend) if trend else None
    if t and t.get("n", 0) >= 20:
        out += f"; {t['up']:.0f}% when it formed after {'an uptrend' if trend == 'up' else 'a downtrend'} like now"
    rnd = st.get("random")
    if rnd:
        out += (f". On random prices the same shape breaks up {rnd['up']:.0f}% just from how the lines are drawn, "
                f"so the real edge is {o['up'] - rnd['up']:+.0f} points")
    pub = PUBLISHED.get(kind)
    if pub:
        out += f". Published stock studies (Bulkowski): {pub['text']}"
    return out + "."


def history_note(kind: str, key: str | None = None, with_bias: bool | None = None, trend: str | None = None) -> str:
    """What this pattern did on real crypto candles since 2017."""
    st = stats().get(kind)
    if not st:
        return ""
    bias = FIXED_BIAS.get(kind) if kind not in ("symmetrical_triangle", "pennant") else "trend"
    out = []
    if st.get("with_pct") is not None and bias:
        way = "the textbook way" if bias == "trend" else f"{bias.upper()} (textbook)"
        out.append(f"On crypto since 2017 ({st['n']} breaks) it broke {way} {_p(st['with_pct'])} of the time")
    else:
        out.append(f"On crypto since 2017: {st['n']} breaks")
    if key in ("near_apex", "past_apex") and "squeezed" in st:
        out.append(f"when squeezed near the apex it fizzled without breaking {_p(st.get('squeeze_no_break'))} of the "
                   f"time, and late breaks reached their target only {_p(st.get('squeeze_target'))}")
    elif key in ("broke_with", "broke_against", "retest"):
        t = st.get("target_with") if with_bias else st.get("target_against")
        out.append(f"breaks like this one reached the measured target {_p(t)} of the time, fake breaks {_p(st['fake'])}")
    elif key == "fake_break":
        out.append(f"{_p(st['fake'])} of its breaks were fakes")
    else:
        out.append(f"breaks reached their measured target {_p(st['target_all'])} of the time "
                   f"(fakes {_p(st['fake'])}, retests of the broken line {_p(st['retest'])})")
    odds = odds_note(kind, trend)
    return "History: " + ", ".join(out) + "." + (f" {odds}" if odds and key not in ("target_hit",) else "")


# ---- what to do -----------------------------------------------------------------------
def _after(det: Detector, p: Pattern, b: int, d: str, end: int) -> dict:
    """Follow a break from bar b to `end`: fake breakout, retest, target, stop."""
    sgn = 1 if d == "up" else -1
    line = p.up if d == "up" else p.lo
    other = p.lo if d == "up" else p.up
    tgt = p.target(d, b)
    stop = other(b) if p.kind not in REVERSALS else (p.head if d == p.bias else other(b))
    out = {"target": tgt, "stop": stop, "fake": False, "retest": False, "target_hit": False, "stopped": False,
           "bars": end - b}
    for i in range(b + 1, end + 1):
        if sgn * (det.c[i] - line(i)) < 0 and i - b <= 3:
            out["fake"] = True
        lo_, hi_ = det.l[i], det.h[i]
        if (d == "up" and lo_ <= line(i) + 0.3 * p.atr) or (d == "down" and hi_ >= line(i) - 0.3 * p.atr):
            out["retest"] = True
        if (d == "up" and lo_ <= stop) or (d == "down" and hi_ >= stop):
            out["stopped"] = True
            break
        if (d == "up" and hi_ >= tgt) or (d == "down" and lo_ <= tgt):
            out["target_hit"] = True
            break
    return out


def situation(df: pd.DataFrame, end: int | None = None, det: Detector | None = None) -> dict | None:
    """Where price is right now relative to the latest pattern, and what to do about it."""
    det = det or Detector(df)
    end = len(df) - 1 if end is None else end
    c = det.c[end]
    p = det.find(end)
    if p is not None:
        prog = p.progress(end)
        u, lo = p.up(end), p.lo(end)
        room = (c - lo) / (u - lo) if u > lo else 0.5
        if prog is not None and prog >= 1:
            key = "past_apex"
        elif prog is not None and prog >= 0.75:
            key = "near_apex"
        elif room >= 0.85:
            key = "at_top"
        elif room <= 0.15:
            key = "at_bottom"
        elif prog is not None and prog >= 0.5:
            key = "breakout_zone"
        else:
            key = "forming"
        return {"pattern": p, "key": key, "price": c, "upper": u, "lower": lo, "progress": prog, "broke": None,
                "text": advice(p, key, end, c)}
    hit = det.last_break(end)
    if hit is None:
        return None
    p, b, d = hit
    after = _after(det, p, b, d, end)
    if after["target_hit"]:
        key = "target_hit"
    elif after["stopped"] or after["fake"]:
        key = "fake_break"
    elif after["retest"]:
        key = "retest"
    else:
        key = "broke_with" if d == p.bias else "broke_against" if p.bias else "broke"
    return {"pattern": p, "key": key, "price": c, "upper": p.up(end), "lower": p.lo(end),
            "progress": p.progress(b), "broke": d, "break_bar": b, **after,
            "text": advice(p, key, end, c, d, b, after)}


def advice(p: Pattern, key: str, end: int, c: float, d: str | None = None, b: int | None = None,
           after: dict | None = None) -> str:
    """Plain-English plan for this situation, with full prices, plus what history says."""
    text = _advice(p, key, end, c, d, b, after)
    note = history_note(p.kind, key, d == p.bias if d else None, p.trend)
    return f"{text} {note}".strip()


def _advice(p: Pattern, key: str, end: int, c: float, d: str | None = None, b: int | None = None,
            after: dict | None = None) -> str:
    u, lo, bias = p.up(end), p.lo(end), p.bias
    tu, td = p.target("up", end), p.target("down", end)
    lean = {"up": "textbook says it breaks UP", "down": "textbook says it breaks DOWN",
            None: "has no textbook direction"}[bias]
    side = "top line" if p.kind not in REVERSALS else "top"
    neck = "neckline" if p.kind in REVERSALS else "bottom line"
    if p.kind in ("double_bottom", "inverse_head_shoulders"):
        side, neck = "neckline", "bottom"
    head = f"{p.name} - {lean}."
    plan_up = f"Long on a candle CLOSE above {price(u)}; stop back inside at {price(lo if p.kind not in REVERSALS else (p.head or lo))}; target {price(tu)} (the pattern's height added to the break)."
    plan_dn = f"Short / exit longs on a candle CLOSE below {price(lo)}; stop back inside at {price(u if p.kind not in REVERSALS else (p.head or u))}; target {price(td)} (the pattern's height taken from the break)."
    first, second = (plan_dn, plan_up) if bias == "down" else (plan_up, plan_dn)
    if key == "forming":
        return (f"{head} Still early (price {price(c)} between {price(lo)} and {price(u)}). No trade inside the "
                f"pattern - wait for a break. If it breaks: {first} If it breaks the other way: {second}")
    if key == "breakout_zone":
        return (f"{head} In the breakout zone (half to three quarters of the way to where the lines meet) - "
                f"the most reliable breaks happen here. {first} Other way: {second}")
    if key == "near_apex":
        return (f"{head} Price is squeezed close to where the lines meet ({price(u)} / {price(lo)}). Breaks this "
                f"late are weaker and fake out more often - the pattern is losing its meaning. Don't anticipate: "
                f"only act on a decisive close (a full candle) outside, and use the smaller target. {first} "
                f"Other way: {second} If price drifts through the apex without a real break, treat it as no "
                f"pattern and wait for a new range.")
    if key == "past_apex":
        return (f"{head} Price went past the point where the lines meet without breaking - the pattern is no "
                f"longer valid. Expect sideways chop; the next real move usually starts from a new range. Wait.")
    if key == "at_top":
        if bias == "down" or p.kind in ("descending_channel", "range", "rising_wedge"):
            return (f"{head} Price is at the {side} ({price(u)}): the usual rejection point. Shorts can sell here with a "
                    f"stop just above {price(u)}; longs should not chase. A close above {price(u)} would break the "
                    f"pattern upward - then: {plan_up}")
        return (f"{head} Price is pressing the {side} ({price(u)}). Breakout watch: {plan_up} Until it closes "
                f"above, a rejection back to {price(lo)} is possible.")
    if key == "at_bottom":
        if bias == "up" or p.kind in ("ascending_channel", "range", "falling_wedge"):
            return (f"{head} Price is at the {neck} ({price(lo)}): the usual bounce point. Longs can buy here with a "
                    f"stop just below {price(lo)}. A close below {price(lo)} breaks the pattern down - then: {plan_dn}")
        return (f"{head} Price is pressing the {neck} ({price(lo)}). Breakdown watch: {plan_dn} Until it closes "
                f"below, a bounce back to {price(u)} is possible.")
    tgt, stop = after["target"], after["stop"]
    dirw = "UP" if d == "up" else "DOWN"
    if key == "broke_with":
        return (f"{p.name} broke {dirw}, the textbook direction, {after['bars']} candles ago. Expected: continuation "
                f"to about {price(tgt)}. In the trade: stop {price(stop)}. Not in: wait for a retest of the broken "
                f"line rather than chasing.")
    if key == "broke_against":
        return (f"{p.name} broke {dirw} - against the textbook direction, {after['bars']} candles ago. On crypto that "
                f"happens often and these breaks reach their target about as often as textbook ones. Target about {price(tgt)}, stop "
                f"{price(stop)}. If price closes back inside, it's a trap: exit and look for the move the other way.")
    if key == "broke":
        return f"{p.name} broke {dirw} {after['bars']} candles ago. Target about {price(tgt)}, stop {price(stop)}."
    if key == "retest":
        return (f"{p.name} broke {dirw} and price came back to test the broken line. Holding the line = the classic "
                f"second-chance entry toward {price(tgt)}, stop {price(stop)}. A close back inside the pattern "
                f"cancels the break.")
    if key == "fake_break":
        return (f"{p.name} broke {dirw} but price fell back inside - a fake breakout. Exit any breakout trade. "
                f"Fake breaks often run hard the other way: watch for a break of the opposite line "
                f"({price(lo if d == 'up' else u)}).")
    if key == "target_hit":
        return (f"{p.name} broke {dirw} and reached its measured target ({price(tgt)}). Take profit / tighten "
                f"the stop; the pattern's job is done.")
    return head


# ---- the playbook: every situation, for every pattern ------------------------------------
PLAYBOOK = {
    "descending_triangle": {
        "what": "Flat support with lower highs pressing down on it. Sellers keep getting more aggressive; buyers defend one price.",
        "usual": "Breaks DOWN through the flat support more often than not.",
    },
    "ascending_triangle": {
        "what": "Flat resistance with higher lows pushing up into it. Buyers keep stepping in higher; sellers defend one price.",
        "usual": "Breaks UP through the flat resistance more often than not.",
    },
    "symmetrical_triangle": {
        "what": "Lower highs and higher lows squeezing to a point. Indecision; volatility shrinks.",
        "usual": "Breaks in the direction of the trend before it formed (a pause, then continuation).",
    },
    "rising_wedge": {
        "what": "Both lines rising, but the bottom line steeper: each push higher gains less.",
        "usual": "Breaks DOWN - a weakening uptrend.",
    },
    "falling_wedge": {
        "what": "Both lines falling, the top line steeper: each push lower gains less.",
        "usual": "Breaks UP - a weakening downtrend.",
    },
    "ascending_channel": {"what": "Parallel lines rising: an orderly uptrend.",
                          "usual": "Buy the bottom line, take profit at the top line; a break down ends the trend."},
    "descending_channel": {"what": "Parallel lines falling: an orderly downtrend.",
                           "usual": "Short the top line, cover at the bottom line; a break up ends the trend."},
    "range": {"what": "Flat top and flat bottom: neither side in control.",
              "usual": "Fade the edges (buy support, sell resistance) until one breaks; the break is then traded."},
    "bull_flag": {"what": "A sharp move up (the pole), then a small, gentle pullback channel.",
                  "usual": "Breaks UP; target is often the length of the pole added to the break."},
    "bear_flag": {"what": "A sharp move down (the pole), then a small, gentle bounce channel.",
                  "usual": "Breaks DOWN; target is often the length of the pole taken from the break."},
    "pennant": {"what": "A sharp move, then a small symmetrical triangle.",
                "usual": "Breaks in the direction of the pole."},
    "broadening": {"what": "Lines spreading apart: swings getting bigger. Volatile and unreliable.",
                   "usual": "No reliable direction - smaller size or stay out."},
    "double_top": {"what": "Two highs at about the same price with a dip between (the neckline).",
                   "usual": "Confirms DOWN on a close below the neckline; target = top-to-neckline height below it."},
    "double_bottom": {"what": "Two lows at about the same price with a bounce between (the neckline).",
                      "usual": "Confirms UP on a close above the neckline; target = neckline-to-bottom height above it."},
    "head_shoulders": {"what": "Three highs, the middle one highest, with a neckline under the two dips.",
                       "usual": "Confirms DOWN on a close below the neckline; target = head-to-neckline height below it."},
    "inverse_head_shoulders": {"what": "Three lows, the middle one lowest, with a neckline over the two bounces.",
                               "usual": "Confirms UP on a close above the neckline; target = neckline-to-head height above it."},
}

SITUATIONS = [
    ("forming", "Early in the pattern", "Don't trade inside it. Mark both lines and wait for a candle to close outside."),
    ("breakout_zone", "Half to three quarters of the way to the apex",
     "Where the most reliable breaks happen. Have orders planned for both directions."),
    ("near_apex", "Squeezed near where the lines meet",
     "Breaks get weaker and fake out more. Only act on a decisive close outside; use the smaller target. "
     "If price drifts through the apex without breaking, the pattern is dead."),
    ("past_apex", "Through the apex without a break", "Pattern invalid. Expect chop; wait for a new structure."),
    ("at_top", "Price at the top line", "Resistance: shorts can sell with a stop just above; longs wait for a close above."),
    ("at_bottom", "Price at the bottom line", "Support: longs can buy with a stop just below; shorts wait for a close below."),
    ("broke_with", "Broke the textbook way", "Trade the break or the retest. Stop back inside the pattern; target = pattern height."),
    ("broke_against", "Broke against the textbook way",
     "Often a strong move when it holds. Same plan (stop back inside, height as target); exit fast if it closes back inside."),
    ("retest", "Back to test the broken line", "The second-chance entry: holding the line confirms the break."),
    ("fake_break", "Broke, then closed back inside", "Get out. Fake breaks often run hard toward the opposite line."),
    ("target_hit", "Measured target reached", "Take profit or tighten the stop."),
]
