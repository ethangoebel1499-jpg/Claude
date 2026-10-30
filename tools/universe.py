"""The surviving strategies on the top 20 coins and every common timeframe.

  python tools/universe.py          # writes reports/universe.md and reports/universe.csv

Breadth is the main measure: with this many combinations some will look great by
luck, so a setup only counts if it makes money on many coins, in both 2017-2021
and 2022-2026, at a 0.1% fee. Real Kraken candles, no leverage.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mcbot import s8  # noqa: E402
from mcbot.backtest import run_backtest  # noqa: E402
from mcbot.data import load_csv, resample  # noqa: E402
from mcbot.grind import STOP as G_SL, TAKE_PROFITS as G_TP, grind_signals  # noqa: E402
from mcbot.witcher import witcher_signals  # noqa: E402

COINS = {"BTC": "XBTUSD", "ETH": "ETHUSD", "SOL": "SOLUSD", "XRP": "XRPUSD",
         "BNB": "crypto/BNBUSD", "DOGE": "crypto/XDGUSD", "TRX": "crypto/TRXUSD", "ADA": "crypto/ADAUSD",
         "LINK": "crypto/LINKUSD", "AVAX": "crypto/AVAXUSD", "XLM": "crypto/XLMUSD", "SUI": "crypto/SUIUSD",
         "BCH": "crypto/BCHUSD", "LTC": "crypto/LTCUSD", "DOT": "crypto/DOTUSD", "HBAR": "crypto/HBARUSD",
         "SHIB": "crypto/SHIBUSD", "TON": "crypto/TONUSD", "UNI": "crypto/UNIUSD", "XMR": "crypto/XMRUSD"}
TIMEFRAMES = ["1h", "2h", "4h", "6h", "8h", "12h", "1D", "2D"]
FEES = (0.0004, 0.001)
START, MID = pd.Timestamp("2017-01-01", tz="UTC"), pd.Timestamp("2022-01-01", tz="UTC")
x3 = [(a * 3, s) for a, s in G_TP]
SETUPS = {   # name: (signal function, allow_short, exits)
    "S3 GRIND (script exits)": ("grind", True, {"take_profits": G_TP, "stop_loss": G_SL}),
    "S3 GRIND targets x3, 6% stop": ("grind", True, {"take_profits": x3, "stop_loss": 0.06}),
    "S4b Witcher 3% / 5%+10%": ("witcher", True, {"stop_loss": 0.03, "take_profits": [(0.05, 0.5), (0.10, 0.5)]}),
    "S4b Witcher 4% / 12%+24%": ("witcher", True, {"stop_loss": 0.04, "take_profits": [(0.12, 0.5), (0.24, 0.5)]}),
    "S8 long-only reversal": ("s8", False, {"take_profits": s8.TAKE_PROFITS, "stop_loss": None}),
    "S8 long-only reversal, 5% stop": ("s8", False, {"take_profits": s8.TAKE_PROFITS, "stop_loss": 0.05}),
}
SIGNALS = {"grind": grind_signals, "witcher": witcher_signals, "s8": s8.signals}


def one(job):
    coin, tf = job
    h1 = load_csv(ROOT / f"datasets/{COINS[coin]}_60.csv")
    full = h1 if tf == "1h" else resample(h1, tf)
    if len(full) < 300:
        return []
    start = max(START, full.index[200])
    df = full[full.index >= start]
    early, late = df[df.index < MID], df[df.index >= MID]
    sigs = {k: f(full).loc[df.index] for k, f in SIGNALS.items()}
    rows = []
    for name, (sk, shorts, exits) in SETUPS.items():
        sig = sigs[sk]
        r = {"setup": name, "coin": coin, "tf": tf, "from": f"{df.index[0]:%Y-%m-%d}",
             "bh_total": (df.close.iloc[-1] / df.close.iloc[0] - 1) * 100,
             "bh_late": (late.close.iloc[-1] / late.close.iloc[0] - 1) * 100 if len(late) > 50 else np.nan}
        for fee in FEES:
            s = run_backtest(df, sig, allow_short=shorts, fee=fee, **exits).stats
            r[f"total_{fee}"], r[f"dd_{fee}"], r["trades"] = s["total_return_pct"], s["max_drawdown_pct"], s["trades"]
            r[f"early_{fee}"] = (run_backtest(early, sig.loc[early.index], allow_short=shorts, fee=fee, **exits)
                                 .stats["total_return_pct"] if len(early) > 300 else np.nan)
            r[f"late_{fee}"] = (run_backtest(late, sig.loc[late.index], allow_short=shorts, fee=fee, **exits)
                                .stats["total_return_pct"] if len(late) > 50 else np.nan)
        # both halves positive (a coin listed after 2021 only needs 2022-26), at least 20 trades
        e, l_ = r["early_0.001"], r["late_0.001"]
        r["both"] = bool((np.isnan(e) or e > 0) and l_ > 0 and r["trades"] >= 20)
        r["beats_bh_late"] = bool(l_ > r["bh_late"]) if not np.isnan(r["bh_late"]) else False
        rows.append(r)
    return rows


def main():
    jobs = [(c, tf) for c in COINS for tf in TIMEFRAMES]
    rows = []
    with ProcessPoolExecutor(4) as ex:
        for i, res in enumerate(ex.map(one, jobs), 1):
            rows += res
            if res:
                print(f"[{i}/{len(jobs)}] {res[0]['coin']} {res[0]['tf']} done", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "universe.csv", index=False)

    lines = ["# Top 20 coins x every timeframe", "",
             "Real Kraken candles, no leverage, from 2017 or the coin's Kraken listing. **Breadth** = coins where "
             "the setup made money in both 2017-21 and 2022-26 at a 0.1% fee (coins listed after 2021: 2022-26 "
             "only), with 20+ trades. With ~1,000 combinations some single results are luck; breadth is not.", "",
             "## Breadth: coins (of 20) profitable in both periods @0.1%", "",
             "| Setup | " + " | ".join(TIMEFRAMES) + " |", "|---|" + "---|" * len(TIMEFRAMES)]
    for name in SETUPS:
        cells = []
        for tf in TIMEFRAMES:
            g = df[(df.setup == name) & (df.tf == tf)]
            n = int(g.both.sum())
            cells.append(f"{'🟢' if n >= 10 else '🟡' if n >= 5 else '🔴'} {n}/{len(g)}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## Median across coins: 2022-26 return @0.1% vs buy & hold", "",
              "| Setup | " + " | ".join(TIMEFRAMES) + " |", "|---|" + "---|" * len(TIMEFRAMES)]
    for name in SETUPS:
        cells = []
        for tf in TIMEFRAMES:
            g = df[(df.setup == name) & (df.tf == tf)]
            cells.append(f"{g['late_0.001'].median():+.0f}% (B&H {g.bh_late.median():+.0f}%)")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## The 25 best single results that passed both periods (sorted by 2022-26)", "",
              "| Setup | Coin | TF | Trades | Total @0.1% | 2017-21 | 2022-26 | B&H 2022-26 | Max DD |",
              "|---|---|---|---|---|---|---|---|---|"]
    best = df[df.both].sort_values("late_0.001", ascending=False).head(25)
    for _, r in best.iterrows():
        e = "n/a" if np.isnan(r["early_0.001"]) else f"{r['early_0.001']:+,.0f}%"
        lines.append(f"| {r.setup} | {r.coin} | {r.tf} | {r.trades} | {r['total_0.001']:+,.0f}% | {e} | "
                     f"{r['late_0.001']:+,.0f}% | {r.bh_late:+,.0f}% | {r['dd_0.001']:.0f}% |")
    (out / "universe.md").write_text("\n".join(lines) + "\n")
    print("DONE -> reports/universe.md", flush=True)


if __name__ == "__main__":
    main()
