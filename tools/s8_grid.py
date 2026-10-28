"""S8 (long-only reversal) on every common timeframe and coin.

  python tools/s8_grid.py        # writes reports/s8.md

Real Kraken candles, no leverage. 15m-1h BTC timeframes come from the 15m file,
24m/12m from their own files, everything else from 1h candles.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from mcbot import s8  # noqa: E402
from mcbot.backtest import run_backtest  # noqa: E402
from mcbot.data import load_csv, resample  # noqa: E402

COINS = {"BTC": "XBTUSD", "ETH": "ETHUSD", "SOL": "SOLUSD", "XRP": "XRPUSD"}
FAST = ["15m", "30m", "45m"]
SLOW = ["1h", "2h", "4h", "6h", "8h", "12h", "1D", "2D"]
FEES = (0.0004, 0.001)
MID = pd.Timestamp("2022-01-01", tz="UTC")


def frames():
    m15 = load_csv(ROOT / "datasets/XBTUSD_15.csv")
    for tf in FAST:
        yield "BTC", tf, m15 if tf == "15m" else resample(m15, tf.replace("m", "min"))
    yield "BTC", "12m", load_csv(ROOT / "datasets/XBTUSD_12m.csv")
    yield "BTC", "24m", load_csv(ROOT / "datasets/XBTUSD_24m.csv")
    for coin, p in COINS.items():
        h1 = load_csv(ROOT / f"datasets/{p}_60.csv")
        for tf in SLOW:
            yield coin, tf, h1 if tf == "1h" else resample(h1, tf)


def run(df, sig, fee, stop):
    return run_backtest(df, sig, allow_short=False, take_profits=s8.TAKE_PROFITS, stop_loss=stop, fee=fee).stats


def main():
    rows = []
    for coin, tf, full in frames():
        sig_full = s8.signals(full)
        df = full[full.index >= "2017-01-01"]
        df = df.iloc[50:] if df.index[0] == full.index[0] else df
        sig = sig_full.loc[df.index]
        early, late = df[df.index < MID], df[df.index >= MID]
        for stop in (None, 0.05):
            r = {"coin": coin, "tf": tf, "stop": "none" if stop is None else f"{stop:.0%}",
                 "buy_hold": round((df.close.iloc[-1] / df.close.iloc[0] - 1) * 100),
                 "bh_late": round((late.close.iloc[-1] / late.close.iloc[0] - 1) * 100)}
            for fee in FEES:
                s = run(df, sig, fee, stop)
                r[f"total_{fee}"], r[f"dd_{fee}"] = s["total_return_pct"], s["max_drawdown_pct"]
                r["trades"], r["win"] = s["trades"], s["win_rate_pct"]
                r[f"early_{fee}"] = run(early, sig.loc[early.index], fee, stop)["total_return_pct"] if len(early) > 100 else float("nan")
                r[f"late_{fee}"] = run(late, sig.loc[late.index], fee, stop)["total_return_pct"]
            ok = r["early_0.001"] > 0 if r["early_0.001"] == r["early_0.001"] else True
            r["verdict"] = "GREEN" if ok and r["late_0.001"] > 0 and r["trades"] >= 30 else \
                "YELLOW" if r["total_0.0004"] > 0 else "RED"
            rows.append(r)
            print(f"{coin} {tf:>3} stop {r['stop']:>4}: trades {r['trades']:5} win {r['win']:5}%  "
                  f"@0.04% {r['total_0.0004']:+9.0f}%  @0.1% {r['total_0.001']:+9.0f}% (dd {r['dd_0.001']:.0f}%)  "
                  f"17-21 {r['early_0.001']:+7.0f}%  22-26 {r['late_0.001']:+7.0f}% (B&H {r['bh_late']:+.0f}%)  {r['verdict']}",
                  flush=True)
    out = pd.DataFrame(rows)
    (ROOT / "reports").mkdir(exist_ok=True)
    out.to_csv(ROOT / "reports/s8.csv", index=False)
    light = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}
    lines = ["# S8 - Long-only reversal on every timeframe", "",
             "Entry: WaveTrend green dot below zero + green Heikin Ashi candle. Exits: 50% at +2%, 25% at +5%, "
             "the rest rides until the reversal down (red dot above zero + red Heikin Ashi candle), which also "
             "closes anything left. No shorts. Real Kraken candles from 2017, no leverage.", "",
             "🟢 = profitable at a 0.1% fee in both 2017-21 and 2022-26 (30+ trades); 🟡 = profitable overall "
             "at 0.04%; 🔴 = otherwise.", ""]
    for stop in ("none", "5%"):
        sub = out[out.stop == stop]
        lines += [f"## Stop-loss: {stop}", "",
                  "| Coin | TF | Trades | Win | Total @0.04% | Total @0.1% | Max DD @0.1% | 2017-21 @0.1% | "
                  "2022-26 @0.1% | Buy & hold 2022-26 | |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            lines.append(f"| {r.coin} | {r.tf} | {r.trades} | {r.win}% | {r['total_0.0004']:+,.0f}% | "
                         f"{r['total_0.001']:+,.0f}% | {r['dd_0.001']:.0f}% | {r['early_0.001']:+,.0f}% | "
                         f"{r['late_0.001']:+,.0f}% | {r.bh_late:+,}% | {light[r.verdict]} |")
        lines.append("")
    (ROOT / "reports/s8.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
