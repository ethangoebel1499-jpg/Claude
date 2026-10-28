"""Audit a TradingView-style GRIND trade list against real candles and our port.

  python tools/tv_audit.py datasets/tv_grind_trades.csv

Expects columns: trade, side, entry_date, entry_price, exit_date, exit_price,
notional_usd, net_pnl_usd, return_pct (one row per partial exit). Checks:
  1. the list's own numbers: positions, win rate, compounding, fee sensitivity
  2. do the prices exist? each entry vs Kraken 15m opens on the listed day / the day before
  3. does our GRIND port take the same trades (same side, price within 0.5%, within a day)?
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from mcbot.backtest import run_backtest  # noqa: E402
from mcbot.data import load_csv, resample  # noqa: E402
from mcbot.grind import STOP, TAKE_PROFITS, grind_signals  # noqa: E402


def positions(tv: pd.DataFrame) -> pd.DataFrame:
    p = tv.groupby(["entry_date", "entry_price", "side"], sort=False).agg(
        pnl=("net_pnl_usd", "sum"), notional=("notional_usd", "sum"), worst=("return_pct", "min")).reset_index()
    p["entry_date"] = pd.to_datetime(p.entry_date).dt.tz_localize("UTC")
    p["ret"] = p.pnl / p.notional
    return p


def main(path: str) -> None:
    tv = pd.read_csv(path)
    pos = positions(tv)
    print(f"1. THE LIST ITSELF: {len(pos)} positions ({(pos.side == 'Long').sum()} long / "
          f"{(pos.side == 'Short').sum()} short), {pos.entry_date.min():%Y-%m-%d} -> {pos.entry_date.max():%Y-%m-%d}")
    print(f"   winning positions {(pos.pnl > 0).mean():.0%}, positions with a stop-out {int((pos.worst < -5).sum())}")
    for fee in (0.0004, 0.001, 0.002):
        eq = (1 + pos.ret - 2 * (fee - 0.0004)).cumprod()
        print(f"   compounded at {fee:.2%} fee per side: {eq.iloc[-1] * 100 - 100:+,.0f}%  "
              f"(max drawdown {(eq / eq.cummax() - 1).min():.0%})")

    m15 = load_csv(ROOT / "datasets/XBTUSD_15.csv")
    near, prev_day = 0, 0
    for _, r in pos.iterrows():
        win = m15[(m15.index >= r.entry_date - pd.Timedelta("1D")) & (m15.index < r.entry_date + pd.Timedelta("1D"))]
        if win.empty:
            continue
        k = (win.open - r.entry_price).abs().idxmin()
        if abs(win.open.loc[k] / r.entry_price - 1) < 0.003:
            near += 1
            prev_day += k < r.entry_date
    print(f"\n2. PRICES: {near}/{len(pos)} entries match a real Kraken 15m open within 0.3%; "
          f"{prev_day} of those are on the day BEFORE the listed date")

    print("\n3. OUR PORT (Kraken data, script settings, 0.04% fee): trades taken by both")
    h1 = load_csv(ROOT / "datasets/XBTUSD_60.csv")
    start = pos.entry_date.min() - pd.Timedelta("3D")
    for tf, base in (("45min", m15), ("1h", h1), ("2h", h1), ("4h", h1), ("1D", h1)):
        df = resample(base, tf)
        sig = grind_signals(df)
        d = df[df.index >= start]
        res = run_backtest(d, sig.loc[d.index], allow_short=True, take_profits=TAKE_PROFITS, stop_loss=STOP,
                           fee=0.0004)
        ours = pd.DataFrame([(t.entry_time, t.side, t.entry_price) for t in res.trades], columns=["t", "side", "px"])
        both = sum(((ours.t >= p.entry_date - pd.Timedelta("1D")) & (ours.t < p.entry_date + pd.Timedelta("1D")) &
                    (ours.side == p.side.lower()) & ((ours.px / p.entry_price - 1).abs() < 0.005)).any()
                   for _, p in pos.iterrows())
        print(f"   {tf:>5}: {both:3}/{len(pos)} of the list's positions | our run: {res.stats['trades']} trades, "
              f"{res.stats['total_return_pct']:+,.0f}%, max DD {res.stats['max_drawdown_pct']:.0f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "datasets/tv_grind_trades.csv"))
