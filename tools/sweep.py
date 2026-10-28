"""Strategy sweep: every strategy x coin x timeframe, with robustness checks.

  python tools/sweep.py                 # writes reports/sweep.md and reports/sweep.csv

For each combination: trades, win rate, total return at two fee levels, the
2017-2021 and 2022-2026 halves, worst drawdown, entry quality vs random entries
(hit 1 ATR before a 2 ATR stop), and buy & hold. Verdict:
  GREEN   positive in both halves at the higher fee, and entries beat random by > 2 std errors
  YELLOW  positive overall at the lower fee
  RED     everything else
All tests: real Kraken candles, no leverage, signals on closed candles, fills next open.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from mcbot.backtest import run_backtest, run_level_backtest  # noqa: E402
from mcbot.data import load_csv, resample  # noqa: E402
from mcbot.entry_quality import summarize  # noqa: E402
from mcbot.grind import STOP as G_SL, TAKE_PROFITS as G_TP, grind_signals  # noqa: E402
from mcbot.indicators import market_cipher  # noqa: E402
from mcbot.strategies import build_signals, topdown_exit_levels  # noqa: E402
from mcbot.witcher import STOP_LONG as W_SL, STOP_SHORT as W_SS, TAKE_PROFIT as W_TP, witcher_signals  # noqa: E402

COINS = {"BTC": "XBTUSD", "ETH": "ETHUSD", "SOL": "SOLUSD", "XRP": "XRPUSD"}
TIMEFRAMES = ["1h", "2h", "4h", "6h", "8h", "12h", "1D"]
FEES = (0.0004, 0.001)
START = pd.Timestamp("2017-01-01", tz="UTC")
MID = pd.Timestamp("2022-01-01", tz="UTC")


def s1(full):
    mc = market_cipher(full, use_heikin_ashi=True)
    lv = topdown_exit_levels(mc)
    sig = build_signals(mc, "S1")
    return sig, lambda d, s, fee: run_level_backtest(d, s, lv.loc[d.index], allow_short=True, fee=fee)


def fixed(signal_fn, **exits):
    def make(full):
        sig = signal_fn(full)
        return sig, lambda d, s, fee: run_backtest(d, s, allow_short=True, fee=fee, **exits)
    return make


STRATEGIES = {
    "S1 Top-down confluence": s1,
    "S3 GRIND BTC": fixed(grind_signals, take_profits=G_TP, stop_loss=G_SL),
    "S4 Witcher (script exits)": fixed(witcher_signals, take_profit=W_TP, stop_loss=W_SL, stop_loss_short=W_SS),
    "S4b Witcher (3% SL, 5%/10% TP)": fixed(witcher_signals, stop_loss=0.03, take_profits=[(0.05, 0.5), (0.10, 0.5)]),
}


def verdict(r: dict) -> str:
    hi = FEES[-1]
    if (r[f"early_{hi}"] > 0 and r[f"late_{hi}"] > 0 and r["trades"] >= 30
            and r["edge"] > 2 * r["edge_se"]):
        return "GREEN"
    if r[f"total_{FEES[0]}"] > 0:
        return "YELLOW"
    return "RED"


def run() -> pd.DataFrame:
    rows = []
    hourly = {c: load_csv(ROOT / "datasets" / f"{p}_60.csv") for c, p in COINS.items()}
    for coin, h1 in hourly.items():
        for tf in TIMEFRAMES:
            full = h1 if tf == "1h" else resample(h1, tf)
            start = max(START, full.index[min(200, len(full) - 1)])
            df = full[full.index >= start]
            bh = (df.close.iloc[-1] / df.close.iloc[0] - 1) * 100
            for name, make in STRATEGIES.items():
                t0 = time.time()
                sig_full, bt = make(full)
                sig = sig_full.loc[df.index]
                r = {"strategy": name, "coin": coin, "timeframe": tf, "from": f"{df.index[0]:%Y-%m-%d}",
                     "buy_hold_pct": round(bh, 1)}
                for fee in FEES:
                    res = bt(df, sig, fee)
                    r["trades"] = res.stats["trades"]
                    r["win_pct"] = res.stats["win_rate_pct"]
                    r[f"total_{fee}"] = res.stats["total_return_pct"]
                    r[f"maxdd_{fee}"] = res.stats["max_drawdown_pct"]
                    early, late = df[df.index < MID], df[df.index >= MID]
                    r[f"early_{fee}"] = bt(early, sig.loc[early.index], fee).stats["total_return_pct"] if len(early) > 50 else 0.0
                    r[f"late_{fee}"] = bt(late, sig.loc[late.index], fee).stats["total_return_pct"]
                eq = summarize(df, sig, 1.0, 2.0, "atr", 48, warmup=0)
                r["edge"], r["edge_se"] = eq["edge_pts"], eq["std_err_pts"]
                r["verdict"] = verdict(r)
                rows.append(r)
                print(f"{name:32} {coin} {tf:>3}  trades {r['trades']:>5}  total@0.04% {r['total_0.0004']:>+9.1f}%  "
                      f"total@0.1% {r['total_0.001']:>+9.1f}%  {r['verdict']}  ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def write_report(df: pd.DataFrame) -> None:
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "sweep.csv", index=False)
    light = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}
    lines = ["# Strategy sweep", "",
             "Real Kraken candles, no leverage, from 2017 (or the coin's Kraken listing). "
             "Verdict: 🟢 positive in both halves at the 0.1% fee **and** entries beat random by > 2 std errors; "
             "🟡 positive overall at 0.04%; 🔴 otherwise.", ""]
    greens = df[df.verdict == "GREEN"]
    lines += ["## 🟢 Setups that passed every check", ""]
    if greens.empty:
        lines += ["None.", ""]
    else:
        lines += ["| Strategy | Coin | TF | Trades | Total @0.04% | Total @0.1% | 2017-21 @0.1% | 2022-26 @0.1% | Max DD @0.1% | Entry edge | Buy & hold |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in greens.iterrows():
            lines.append(f"| {r.strategy} | {r.coin} | {r.timeframe} | {r.trades} | {r['total_0.0004']:+,.0f}% | "
                         f"{r['total_0.001']:+,.0f}% | {r['early_0.001']:+,.0f}% | {r['late_0.001']:+,.0f}% | "
                         f"{r['maxdd_0.001']:.0f}% | {r.edge:+} ±{r.edge_se} | {r.buy_hold_pct:+,.0f}% |")
        lines.append("")
    for name in df.strategy.unique():
        sub = df[df.strategy == name]
        lines += [f"## {name}", "", "Total return @0.04% fee [@0.1%] and verdict", "",
                  "| Coin | " + " | ".join(TIMEFRAMES) + " |", "|---|" + "---|" * len(TIMEFRAMES)]
        for coin in COINS:
            cells = []
            for tf in TIMEFRAMES:
                r = sub[(sub.coin == coin) & (sub.timeframe == tf)].iloc[0]
                cells.append(f"{light[r.verdict]} {r['total_0.0004']:+,.0f}% [{r['total_0.001']:+,.0f}%]")
            lines.append(f"| {coin} | " + " | ".join(cells) + " |")
        lines.append("")
    (out / "sweep.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    t = time.time()
    result = run()
    write_report(result)
    print(f"DONE in {time.time() - t:.0f}s -> reports/sweep.md, reports/sweep.csv", flush=True)
