"""Backtest S9 (falling wedge breakout after an uptrend) across coins, timeframes and variants.

  python tools/s9_test.py   -> reports/s9.md

Periods: 2017-2021 (where the pattern stats were found) and 2022-now (out of sample).
Variants switch the two filters off one at a time, so you can see whether they add anything.
Fee 0.1% per side, full account per trade, one trade at a time per coin, no leverage.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from livepaper.history import CRYPTO, history  # noqa: E402
from mcbot import s9  # noqa: E402

TIMEFRAMES = ["2h", "4h", "12h", "1D"]
PERIODS = {"2017-21": ("2017-01-01", "2022-01-01"), "2022-now": ("2022-01-01", "2100-01-01")}
VARIANTS = {
    "S9 (uptrend + early, measured target)": dict(uptrend=True, max_progress=0.75, exits="measured"),
    "S9 half exits (50% at half height, stop to entry)": dict(uptrend=True, max_progress=0.75, exits="half"),
    "no uptrend filter": dict(uptrend=False, max_progress=0.75, exits="measured"),
    "no early filter": dict(uptrend=True, max_progress=None, exits="measured"),
    "any falling wedge break": dict(uptrend=False, max_progress=None, exits="measured"),
}


def md(df: pd.DataFrame, index: bool = True) -> str:
    """Markdown table with full numbers (no extra libraries)."""
    def cell(v):
        return f"{v:,.2f}" if isinstance(v, float) else str(v)
    cols = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for i, r in zip(df.index, df.itertuples(index=False)):
        lead = [" / ".join(map(str, i)) if isinstance(i, tuple) else str(i)] if index else []
        lines.append("| " + " | ".join(lead + [cell(v) for v in r]) + " |")
    return "\n".join(lines)


def main() -> None:
    rows = []
    for coin, pair in CRYPTO.items():
        for tf in TIMEFRAMES:
            try:
                df = history(pair, "crypto", tf)
            except Exception as e:
                print(f"skip {coin} {tf}: {e}")
                continue
            df = df[df.index >= "2016-09-01"]          # warm-up before 2017
            if len(df) < 300:
                continue
            ev = s9.events(df)
            for vname, kw in VARIANTS.items():
                fr = s9.frame(df, ev, **kw)
                for per, (a, b) in PERIODS.items():
                    m = (df.index >= a) & (df.index < b)
                    if m.sum() < 200:
                        continue
                    r = s9.backtest(df[m], fr[m], exits=kw["exits"])
                    st = r.stats
                    rows.append({"variant": vname, "coin": coin, "tf": tf, "period": per, "trades": st["trades"],
                                 "ret": float(st["total_return_pct"]), "bh": float(st["buy_hold_return_pct"]),
                                 "win": st["win_rate_pct"], "avg": float(st["avg_trade_pct"] or 0),
                                 "dd": float(st["max_drawdown_pct"]), "pf": float(st["profit_factor"] or 0)})
            print(f"{coin} {tf}: {len(ev)} wedge breaks")
    R = pd.DataFrame(rows)
    R.to_csv(ROOT / "reports/s9_results.csv", index=False)

    def agg(g):
        t = g[g.trades > 0]
        return pd.Series({"markets": len(g), "trades": int(g.trades.sum()),
                          "win rate": (t.win * t.trades).sum() / max(t.trades.sum(), 1),
                          "avg trade %": (t.avg * t.trades).sum() / max(t.trades.sum(), 1),
                          "profitable markets": f"{(g.ret > 0).sum()}/{len(g)}",
                          "median return %": g.ret.median(), "median buy&hold %": g.bh.median(),
                          "median max DD %": g.dd.median()})

    out = ["# S9 - Falling wedge breakout after an uptrend", "",
           "Long only, no leverage, 0.1% fee per side, full account per trade. 20 coins × 2h / 4h / 12h / daily. "
           "Entry = next open after a candle closes above the wedge's top line; stop = the bottom line at the break; "
           "target = wedge height added to the break. *Markets* = coin × timeframe combinations with data.", ""]
    for per in PERIODS:
        g = R[R.period == per]
        t = g.groupby("variant", sort=False).apply(agg, include_groups=False)
        out += [f"## {per}{' (out of sample)' if per == '2022-now' else ''}", "", md(t), ""]
    base = R[(R.variant == list(VARIANTS)[0])]
    by_tf = base.groupby(["period", "tf"], sort=False).apply(agg, include_groups=False)
    out += ["## S9 by timeframe", "", md(by_tf), ""]
    best = base[base.period == "2022-now"].sort_values("ret", ascending=False)
    out += ["## S9 out of sample, every market (2022-now)", "",
            md(best[["coin", "tf", "trades", "win", "avg", "ret", "bh", "dd", "pf"]].rename(columns={
                "win": "win %", "avg": "avg trade %", "ret": "return %", "bh": "buy&hold %", "dd": "max DD %",
                "pf": "profit factor"}), index=False), ""]
    (ROOT / "reports/s9.md").write_text("\n".join(out))
    print("\n".join(out[:40]))


if __name__ == "__main__":
    main()
