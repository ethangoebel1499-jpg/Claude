"""Market Cipher trading bot CLI.

Examples:
  python main.py backtest --synthetic
  python main.py backtest --exchange binance --symbol BTC/USDT --timeframe 4h --strategy dots_confirmed
  python main.py backtest --csv btc_1h.csv --compare
  python main.py signals --exchange binance --symbol ETH/USDT
  python main.py paper --exchange binance --symbol BTC/USDT --timeframe 1h
  python main.py live  --exchange binance --symbol BTC/USDT --timeframe 1h --live
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from mcbot import bot, data
from mcbot.backtest import run_backtest
from mcbot.indicators import market_cipher
from mcbot.strategies import STRATEGIES


def load(args) -> pd.DataFrame:
    if args.synthetic:
        return data.synthetic(n=args.limit, timeframe=args.timeframe)
    if args.csv:
        return data.load_csv(args.csv)
    return data.fetch_ohlcv(data.make_exchange(args.exchange), args.symbol, args.timeframe, args.limit)


def cmd_backtest(args) -> None:
    df = load(args)
    mc = market_cipher(df)
    names = list(STRATEGIES) if args.compare else [args.strategy]
    print(f"{len(df)} bars  {df.index[0]} -> {df.index[-1]}\n")
    rows = {}
    for name in names:
        res = run_backtest(df, STRATEGIES[name](mc), allow_short=args.short,
                           stop_loss=args.stop or None, take_profit=args.target or None,
                           fee=args.fee)
        rows[name] = res.stats
        if not args.compare:
            for t in res.trades[-10:]:
                print(f"  {t.side:5} {t.entry_time:%Y-%m-%d %H:%M} {t.entry_price:>10.2f} -> "
                      f"{t.exit_price:>10.2f} {t.pnl_pct * 100:+6.2f}% ({t.reason})")
            print()
    print(pd.DataFrame(rows).to_string())


def cmd_signals(args) -> None:
    mc = market_cipher(load(args))
    cols = ["close", "wt1", "wt2", "money_flow", "rsi", "stoch_k", "ribbon_bull",
            "green_dot", "red_dot", "bull_div", "bear_div",
            "bull_mf_div", "bear_mf_div", "bull_mf_conv", "bear_mf_conv"]
    print(mc[cols].tail(args.rows).round(2).to_string())
    sig = STRATEGIES[args.strategy](mc).iloc[-1]
    print(f"\n{args.strategy} on last closed bar: { {k: bool(v) for k, v in sig.items()} }")


def cmd_trade(args, live: bool) -> None:
    if live:
        if not args.live:
            raise SystemExit("Refusing to trade real money without the --live flag.")
        key, secret = bot.live_credentials()
        ex = data.make_exchange(args.exchange, key, secret, sandbox=args.sandbox)
        ex.load_markets()
        broker = bot.LiveBroker(ex, args.symbol)
        allow_short = False
    else:
        ex = data.make_exchange(args.exchange)
        broker = bot.PaperBroker(args.state, fee=args.fee)
        allow_short = args.short
    bot.run(ex, args.symbol, args.timeframe, args.strategy, broker, allow_short=allow_short,
            fraction=args.fraction, stop_loss=args.stop or None, take_profit=args.target or None,
            poll_seconds=args.poll)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--exchange", default="binance", help="any ccxt exchange id")
    common.add_argument("--symbol", default="BTC/USDT")
    common.add_argument("--timeframe", default="1h")
    common.add_argument("--strategy", default="dots_confirmed", choices=list(STRATEGIES))
    common.add_argument("--short", action="store_true", help="allow short trades (backtest/paper)")
    common.add_argument("--stop", type=float, default=0.03, help="stop-loss fraction, 0 to disable")
    common.add_argument("--target", type=float, default=0.06, help="take-profit fraction, 0 to disable")
    common.add_argument("--fee", type=float, default=0.001, help="fee per side")

    source = argparse.ArgumentParser(add_help=False)
    source.add_argument("--limit", type=int, default=3000, help="number of candles")
    source.add_argument("--csv", help="load candles from a CSV instead of the exchange")
    source.add_argument("--synthetic", action="store_true", help="use random demo data")

    b = sub.add_parser("backtest", parents=[common, source])
    b.add_argument("--compare", action="store_true", help="run every strategy")
    b.set_defaults(func=cmd_backtest)

    s = sub.add_parser("signals", parents=[common, source])
    s.add_argument("--rows", type=int, default=15)
    s.set_defaults(func=cmd_signals)

    for name, live in (("paper", False), ("live", True)):
        t = sub.add_parser(name, parents=[common])
        t.add_argument("--fraction", type=float, default=0.95, help="fraction of balance per trade")
        t.add_argument("--poll", type=int, default=30, help="seconds between checks")
        t.add_argument("--state", default="paper_state.json")
        t.add_argument("--live", action="store_true", help="confirm real-money trading")
        t.add_argument("--sandbox", action="store_true", help="use the exchange testnet")
        t.set_defaults(func=lambda a, live=live: cmd_trade(a, live))

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
