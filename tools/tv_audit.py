"""Audit a TradingView-style GRIND trade list against real candles and our port.

  python tools/tv_audit.py datasets/tv_grind_trades.csv

Expects columns: trade, side, entry_date, entry_price, exit_date, exit_price,
notional_usd, net_pnl_usd, return_pct (one row per partial exit). Checks:
  1. the list's own numbers: positions, win rate, compounding, fee sensitivity
  2. do the prices exist? each entry vs Kraken 15m opens on the listed day / the day before,
     and vs the daily real open and Heikin Ashi open (a backtest run on a Heikin Ashi chart
     fills at Heikin Ashi prices, which are not real prices)
  3. does our GRIND port take the same trades (same side, price within 0.5%, within a day)?
  4. the list's own trades replayed with GRIND's exits on Kraken 1h candles, at the
     listed prices vs at a price a daily chart can actually fill (the listed day's open)
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
from mcbot.indicators import heikin_ashi  # noqa: E402


def positions(tv: pd.DataFrame) -> pd.DataFrame:
    p = tv.groupby(["entry_date", "entry_price", "side"], sort=False).agg(
        pnl=("net_pnl_usd", "sum"), notional=("notional_usd", "sum"), worst=("return_pct", "min")).reset_index()
    p["entry_date"] = pd.to_datetime(p.entry_date).dt.tz_localize("UTC")
    p["ret"] = p.pnl / p.notional
    return p


def replay(h1: pd.DataFrame, t, px: float, sgn: int, until, fee: float = 0.0004) -> float:
    """One position with GRIND's exits (stop checked first in a bar), closed at `until`."""
    left, pnl, k, last = 1.0, 0.0, 0, px
    for _, b in h1[(h1.index >= t) & (h1.index < until)].iterrows():
        worst, best = (b.low, b.high) if sgn > 0 else (b.high, b.low)
        if sgn * (worst - px) / px <= -STOP:
            return pnl + left * (-STOP - 2 * fee)
        while k < len(TAKE_PROFITS) and sgn * (best - px) / px >= TAKE_PROFITS[k][0]:
            size = left if k == len(TAKE_PROFITS) - 1 else TAKE_PROFITS[k][1]
            pnl, left, k = pnl + size * (TAKE_PROFITS[k][0] - 2 * fee), left - size, k + 1
        if left <= 1e-9:
            return pnl
        last = b.close
    return pnl + left * (sgn * (last - px) / px - 2 * fee)


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
    d1 = resample(load_csv(ROOT / "datasets/XBTUSD_60.csv"), "1D")
    ha = heikin_ashi(d1)
    for name, opens in (("real daily open", d1.open), ("Heikin Ashi daily open", ha.open)):
        diff = pd.Series([abs(opens.loc[p.entry_date] / p.entry_price - 1) for _, p in pos.iterrows()
                          if p.entry_date in opens.index])
        print(f"   entries within 0.5% of the {name}: {(diff < 0.005).mean():.0%} (median gap {diff.median():.2%})")

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

    print("\n4. THE LIST'S OWN TRADES REPLAYED (GRIND exits, 0.04% fee, Kraken 1h)")
    for label, shift in (("at the listed prices", None), ("at the listed day's open", 0)):
        rets = []
        for i, p in pos.iterrows():
            sgn = 1 if p.side == "Long" else -1
            until = pos.entry_date.iloc[i + 1] if i + 1 < len(pos) else h1.index[-1]
            if shift is None:
                t, px = p.entry_date - pd.Timedelta("1D"), p.entry_price
            else:
                t = p.entry_date
                if t not in h1.index:
                    continue
                px = h1.open.loc[t]
            rets.append(replay(h1, t, px, sgn, max(until, t + pd.Timedelta("1h"))))
        eq = (1 + pd.Series(rets)).cumprod()
        print(f"   {label:26}: win {(pd.Series(rets) > 0).mean():.0%}, total {eq.iloc[-1] * 100 - 100:+,.0f}%, "
              f"max DD {(eq / eq.cummax() - 1).min():.0%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "datasets/tv_grind_trades.csv"))
