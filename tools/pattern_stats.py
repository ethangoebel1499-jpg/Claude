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
        row = {"kind": p.kind, "dir": d, "with": d == p.bias if p.bias else None, "progress": prog, "late": key in squeezed,
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
