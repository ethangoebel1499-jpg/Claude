"""Go over BTC's 30-minute chart with wielkieef's GRIND rules and say what the script is doing right now.

  python tools/grind_watch.py [--symbol BTC] [--tf 30m]

Runs the exact GRIND BTC conditions (mcbot/grind.py, a port of wielkieef's open-source TradingView script) over the
recent candles, replays the script's trades with its own exits (9% stop, +1.7% / +2.3% / +3.4%), and prints:
  - NEW: what happened on the candle that just closed (a trade opened, reversed, hit a target or the stop) - or "none"
  - the open position now (side, entry, stop, targets, P&L)
  - the nine conditions on the last closed candle, the last few trades
Prices are Kraken's BTC/USD; TradingView's chart may use another exchange's feed, so levels can differ slightly.
The first line is machine-readable: "GRIND_EVENT: <none|entry|exit|...>".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livepaper.core import Exits, PositionManager, SimBroker               # noqa: E402
from livepaper.history import history                               # noqa: E402
from mcbot.fmt import price                                         # noqa: E402
from mcbot.grind import STOP, TAKE_PROFITS, SCRIPT_FEE, grind_checklist, grind_signals   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--tf", default="30m")
    a = ap.parse_args()
    df = history(a.symbol, "crypto", a.tf).iloc[-1500:]
    sig = grind_signals(df)
    pm = PositionManager("GRIND", Exits(STOP, TAKE_PROFITS), shorts=True, broker=SimBroker(fee=SCRIPT_FEE))
    events = []

    class Log:                                        # collect what the replay did, candle by candle
        def event(self, run, time, kind, text):
            events.append((str(time), kind, text))
        def __getattr__(self, name):
            return lambda *x, **k: None
    pm.store = Log()
    # replay the script's trades (no forced close at the end, so an open position stays open)
    o, lo, hi = (df[k].to_numpy(float) for k in ("open", "low", "high"))
    for i, t in enumerate(df.index):
        if i > 0:
            pm.on_bar_close(sig.iloc[i - 1], o[i], t, bar_time=df.index[i - 1])
        pm.on_tick(o[i], lo[i], hi[i], t)
    last_bar = df.index[-1]                            # the candle that just closed (the forming one isn't loaded)
    new = [e for e in events if e[0] >= str(last_bar) and e[1] in ("entry", "exit")]
    p = pm.state.position
    s = sig.iloc[-1]
    pending = None                                     # a signal on the closed candle -> acts at the next open
    if s["long_entry"] and (p is None or p.side == "short"):
        pending = "LONG" if p is None else "REVERSE to LONG (close the short, open a long)"
    elif s["short_entry"] and (p is None or p.side == "long"):
        pending = "SHORT" if p is None else "REVERSE to SHORT (close the long, open a short)"
    kinds = sorted({e[1] for e in new} | ({"signal"} if pending else set()))
    print(f"GRIND_EVENT: {','.join(kinds) if kinds else 'none'}")
    print(f"{a.symbol} {a.tf} · candle closed {last_bar:%Y-%m-%d %H:%M} UTC · close {price(df.close.iloc[-1])}")
    if pending:
        print(f"SIGNAL: GRIND says {pending} - the script enters at the next candle's open (~{price(df.close.iloc[-1])})")
    if new:
        print("ON THAT CANDLE:")
        for t, k, txt in new:
            print(f"  - {k.upper()}: {txt}")
    if not pending and not new:
        print("NEW: nothing - no signal, entry, exit, target or stop on the candle that just closed")
    if p:
        sgn = 1 if p.side == "long" else -1
        stop = p.entry_price * (1 - sgn * STOP)
        tps = [p.entry_price * (1 + sgn * lv) for lv, _ in TAKE_PROFITS]
        pnl = sgn * (df.close.iloc[-1] / p.entry_price - 1) * 100
        print(f"OPEN: {p.side.upper()} since {p.entry_time[:16]} at {price(p.entry_price)} · now {pnl:+.2f}% · stop "
              f"{price(stop)} · targets {', '.join(price(x) for x in tps)} ({p.targets_hit} hit)")
    else:
        print("OPEN: no position - waiting for all GRIND conditions to line up")
    print()
    print(grind_checklist(df, -1, label=f"GRIND {a.symbol} {a.tf}"))
    if pm.trades:
        print("\nLast trades:")
        for t in pm.trades[-5:]:
            print(f"  {t['side']:5s} {t['entry_time'][:16]} @ {price(t['entry_price'])} -> {t['exit_time'][:16]} @ "
                  f"{price(t['exit_price'])} ({t['reason']}) {t['pnl_pct']:+.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
