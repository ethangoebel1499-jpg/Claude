"""What price actually did after each chart pattern, on real Kraken candles.

  python tools/pattern_stats.py            -> reports/patterns.md + mcbot/pattern_stats.json

For every pattern the detector finds (walking forward candle by candle, no look-ahead) it records:
  the break: which way, how far toward the apex, whether it was the pattern's usual direction,
  then over the following candles: fake breakout (closed back inside within 3 candles),
  retest of the broken line, measured target reached before the stop (back inside past the
  opposite line), and the move 10 candles later.
And for converging patterns that got squeezed near the apex (85%+ of the way) without breaking:
  did they still break, which way, and did that late break reach its target.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from livepaper.history import history  # noqa: E402
from mcbot.patterns import CONVERGING, NAMES, Detector, _after  # noqa: E402

COINS = ["XBTUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD", "XDGUSD", "LTCUSD", "LINKUSD"]
TIMEFRAMES = ["4h", "1D"]


def scan(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    det = Detector(df)
    breaks, squeezes, seen, squeezed = [], [], set(), {}
    n = len(df)
    for end in range(31, n):
        p = det.find(end - 1)
        if p is None:
            continue
        key = (p.kind, p.start)
        prog = p.progress(end - 1)
        if p.kind in CONVERGING and prog is not None and prog >= 0.85 and key not in squeezed:
            squeezed[key] = end - 1
        d = det.broke(p, end)
        if not d or key in seen:
            continue
        seen.add(key)
        width = end - p.start
        horizon = min(n - 1, end + max(2 * width, 30))
        a = _after(det, p, end, d, horizon)
        c0 = det.c[end]
        later = det.c[min(n - 1, end + 10)]
        move10 = (later / c0 - 1) * 100 * (1 if d == "up" else -1)
        row = {"kind": p.kind, "dir": d, "trend": det._trend(p.start), "with": d == p.bias if p.bias else None, "progress": prog, "late": key in squeezed,
               "fake": a["fake"], "retest": a["retest"], "target": a["target_hit"], "stopped": a["stopped"],
               "move10": move10, "time": str(df.index[end])}
        breaks.append(row)
        if key in squeezed:
            squeezes.append(row)
    # squeezed patterns that never broke at all before falling apart
    for key, i in squeezed.items():
        if key not in seen:
            squeezes.append({"kind": key[0], "dir": None, "with": None, "target": False, "fake": False, "move10": 0.0})
    return breaks, squeezes


def wilson(k: int, n: int) -> tuple[float, float, float]:
    """Share and its 95% range (Wilson interval), in %."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    z, p = 1.96, k / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return p * 100, (mid - half) * 100, (mid + half) * 100


def up_odds(g: pd.DataFrame) -> dict:
    k, n = int((g["dir"] == "up").sum()), int(len(g))
    p, lo, hi = wilson(k, n)
    return {"n": n, "up": p, "lo": lo, "hi": hi}


def random_breaks(series: int = 6, n: int = 15000) -> pd.DataFrame:
    """The same detector on random prices (fat-tailed, no memory, so no real patterns). Whatever
    'direction' shows up here comes from how the lines are drawn - the baseline real odds must beat."""
    import numpy as np
    rows = []
    for seed in range(series):
        rng = np.random.default_rng(seed)
        c = 100 * np.exp(np.cumsum(rng.standard_t(4, n) * 0.01))
        o = np.r_[c[0], c[:-1]]
        sp = np.abs(rng.normal(0, 0.006, n)) * c
        df = pd.DataFrame({"open": o, "close": c, "high": np.maximum(o, c) + sp, "low": np.minimum(o, c) - sp,
                           "volume": 1.0}, index=pd.date_range("2018-01-01", periods=n, freq="4h", tz="UTC"))
        rows += scan(df)[0]
    return pd.DataFrame(rows)


def pct(x) -> str:
    return "-" if x is None or pd.isna(x) else f"{x:.0f}%"


def main() -> None:
    all_b, all_s = [], []
    for coin in COINS:
        for tf in TIMEFRAMES:
            try:
                df = history(coin, "crypto", tf)
            except Exception as e:
                print(f"skip {coin} {tf}: {e}")
                continue
            df = df[df.index >= "2017-01-01"]
            b, s = scan(df)
            for r in b + s:
                r.update(coin=coin, tf=tf)
            all_b += b
            all_s += s
            print(f"{coin} {tf}: {len(df)} candles, {len(b)} breaks, {len(s)} squeezed near the apex")
    B, S = pd.DataFrame(all_b), pd.DataFrame(all_s)
    print("random-price baseline...")
    R = random_breaks()
    stats = {}
    lines = ["# What price did after each pattern", "",
             f"Real Kraken candles, 2017 to now, {len(COINS)} coins × {', '.join(TIMEFRAMES)}. The detector walks forward "
             "candle by candle (no look-ahead). A break = a candle closing beyond a line. **Target** = the pattern's "
             "height measured from the break, reached before the stop (price back inside past the opposite line). "
             "**Fake** = closed back inside within 3 candles. **Move after 10** = average % in the break's direction "
             "10 candles later.", "",
             "| Pattern | Breaks | Broke the usual way | Usual-way breaks: target hit | Against-usual breaks: target hit | "
             "Fake breaks | Retested the line | Move after 10 candles |",
             "|---|---|---|---|---|---|---|---|"]
    for kind in NAMES:
        g = B[B.kind == kind] if len(B) else B
        if g.empty:
            continue
        w = g[g["with"] == True]  # noqa: E712
        ag = g[g["with"] == False]  # noqa: E712
        st = {"n": int(len(g)),
              "with_pct": float(w.shape[0] / g["with"].notna().sum() * 100) if g["with"].notna().any() else None,
              "target_with": float(w.target.mean() * 100) if len(w) else None,
              "target_against": float(ag.target.mean() * 100) if len(ag) else None,
              "target_all": float(g.target.mean() * 100),
              "fake": float(g.fake.mean() * 100), "retest": float(g.retest.mean() * 100),
              "move10": float(g.move10.mean())}
        st["odds"] = up_odds(g)
        rg = R[R.kind == kind]
        if len(rg) >= 30:
            st["random"] = {"n": int(len(rg)), "up": float((rg["dir"] == "up").mean() * 100),
                            "target": float(rg.target.mean() * 100)}
        st["odds_trend"] = {t: up_odds(g[g.trend == t]) for t in ("up", "down") if (g.trend == t).any()}
        st["odds_tf"] = {t: up_odds(g[g.tf == t]) for t in TIMEFRAMES if (g.tf == t).any()}
        sq = S[S.kind == kind] if len(S) else S
        if len(sq):
            st["squeezed"] = int(len(sq))
            st["squeeze_no_break"] = float(sq["dir"].isna().mean() * 100)
            broke = sq[sq["dir"].notna()]
            st["squeeze_with"] = float((broke["with"] == True).mean() * 100) if len(broke) and broke["with"].notna().any() else None  # noqa: E712
            st["squeeze_target"] = float(broke.target.mean() * 100) if len(broke) else None
            st["squeeze_fake"] = float(broke.fake.mean() * 100) if len(broke) else None
        stats[kind] = st
        lines.append(f"| {NAMES[kind]} | {st['n']} | {pct(st['with_pct'])} | {pct(st['target_with'])} | "
                     f"{pct(st['target_against'])} | {pct(st['fake'])} | {pct(st['retest'])} | {st['move10']:+.2f}% |")
    lines += ["", "## Squeezed near the apex (85%+ of the way to where the lines meet)", "",
              "| Pattern | Times | Never broke (fizzled) | Broke the usual way | Late break hit target | Late break faked |",
              "|---|---|---|---|---|---|"]
    for kind, st in stats.items():
        if "squeezed" in st:
            lines.append(f"| {NAMES[kind]} | {st['squeezed']} | {pct(st['squeeze_no_break'])} | {pct(st['squeeze_with'])} | "
                         f"{pct(st['squeeze_target'])} | {pct(st['squeeze_fake'])} |")
    def odds(o):
        return "-" if not o or not o["n"] else f"{o['up']:.0f}% ({o['lo']:.0f}-{o['hi']:.0f}%, n={o['n']})"
    lines += ["", "## Chance of breaking UP (95% range, number of breaks)", "",
              "**On random data** = the same detector on random prices with no real patterns: the share of breaks "
              "UP that comes purely from how the lines are drawn (the sloped line is closer to price, so it gets "
              "broken first). **Real edge** = real minus random - the part that is actually market behaviour. ",
              "",
              "Break = first candle close beyond either line. The range is where the true chance most likely sits "
              "given how many breaks were seen. *Trend before* = price higher or lower than 50 candles before "
              "the pattern started. Published = Bulkowski, *Encyclopedia of Chart Patterns* / thepatternsite.com "
              "(US stocks).", "",
              "| Pattern | All | On random data | Real edge | Trend before UP | Trend before DOWN | 4h | Daily | "
              "Published (stocks) |",
              "|---|---|---|---|---|---|---|---|---|"]
    from mcbot.patterns import PUBLISHED
    for kind, st in stats.items():
        if kind.startswith("_") or "odds" not in st:
            continue
        pub = PUBLISHED.get(kind, {}).get("text", "-")
        rnd = st.get("random")
        edge = f"{st['odds']['up'] - rnd['up']:+.0f} pts" if rnd else "-"
        lines.append(f"| {NAMES[kind]} | {odds(st['odds'])} | {pct(rnd['up']) if rnd else '-'} | {edge} | "
                     f"{odds(st['odds_trend'].get('up'))} | "
                     f"{odds(st['odds_trend'].get('down'))} | {odds(st['odds_tf'].get('4h'))} | "
                     f"{odds(st['odds_tf'].get('1D'))} | {pub} |")
    early = B[(B.kind.isin(CONVERGING)) & (B.progress < 0.75)] if len(B) else B
    late = B[(B.kind.isin(CONVERGING)) & (B.progress >= 0.75)] if len(B) else B
    if len(early) and len(late):
        lines += ["", "## Early vs late breaks (triangles, wedges, pennants)", "",
                  "| When it broke | Breaks | Target hit | Fake | Move after 10 candles |", "|---|---|---|---|---|",
                  f"| Before 75% of the way to the apex | {len(early)} | {pct(early.target.mean() * 100)} | "
                  f"{pct(early.fake.mean() * 100)} | {early.move10.mean():+.2f}% |",
                  f"| 75% or later | {len(late)} | {pct(late.target.mean() * 100)} | {pct(late.fake.mean() * 100)} | "
                  f"{late.move10.mean():+.2f}% |"]
        stats["_timing"] = {"early_target": float(early.target.mean() * 100), "late_target": float(late.target.mean() * 100),
                            "early_fake": float(early.fake.mean() * 100), "late_fake": float(late.fake.mean() * 100),
                            "early_n": int(len(early)), "late_n": int(len(late))}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/patterns.md").write_text("\n".join(lines) + "\n")
    (ROOT / "mcbot/pattern_stats.json").write_text(json.dumps(stats, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
