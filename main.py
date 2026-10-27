"""Top-down confluence trading bot CLI.

Examples:
  python main.py backtest --exchange binance --symbol BTC/USDT --short --leverage 5 \\
      --stop 0.03 --tp "1.5:50,2.5:25,ride:25" --trail --fee 0.0005
  python main.py entries --csv btc_1h.csv
  python main.py signals --exchange binance --symbol BTC/USDT
  python main.py paper --exchange binance --symbol BTC/USDT
  python main.py live  --exchange binance --symbol BTC/USDT --live
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from mcbot import bot, data
from mcbot.backtest import parse_take_profits, run_backtest, run_level_backtest
from mcbot.entry_quality import summarize
from mcbot.indicators import market_cipher
from mcbot.witcher import witcher_checklist, witcher_signals
from mcbot.strategies import CATALOG, build_signals, checklist, strategy_label, topdown_exit_levels


def load(args) -> pd.DataFrame:
    if args.synthetic:
        df = data.synthetic(n=args.limit, timeframe=args.timeframe)
    elif args.csv:
        df = data.load_csv(args.csv)
    else:
        df = data.fetch_ohlcv(data.make_exchange(args.exchange), args.symbol, args.timeframe, args.limit)
    if args.resample:
        df = data.resample(df, args.resample)
    return data.relabel(df, args.relabel) if args.relabel else df


def signals_for(df: pd.DataFrame, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    if CATALOG[args.strategy]["func"] == "raw":
        # Strategy works on the real candles with its own indicators.
        return df, witcher_signals(df)
    mc = market_cipher(df, use_heikin_ashi=args.heikin_ashi, htf_factor=args.htf,
                       mf_period=args.mf_period)
    return mc, build_signals(mc, args.strategy)


def cmd_backtest(args) -> None:
    df = load(args)
    if args.start:
        df = df[df.index >= pd.Timestamp(args.start, tz="UTC")]
    mc, sig = signals_for(df, args)
    print(f"[{strategy_label(args.strategy)}]  {len(df)} bars  {df.index[0]} -> {df.index[-1]}  "
          f"({'Heikin Ashi' if args.heikin_ashi else 'regular'} candles, {args.exits} exits)\n")
    own = CATALOG[args.strategy].get("exits")
    if args.exits == "strategy" and own:
        # The strategy's own exits (S4: +1.1% take profit, 8.2% / 7.8% stop).
        res = run_backtest(df, sig, allow_short=args.short, fee=args.fee, leverage=args.leverage,
                           take_profit=own["take_profit"], stop_loss=own["stop_loss"],
                           stop_loss_short=own["stop_loss_short"], stops_on_close=args.stops_on_close)
    elif args.exits == "strategy":
        # Stop under the 21 EMA (above resistance for shorts), 50% at the next 4h
        # level, 25% at the level after, stop trails the 21 EMA.
        res = run_level_backtest(df, sig, topdown_exit_levels(mc, buffer=args.stop_buffer),
                                 allow_short=args.short, leverage=args.leverage, fee=args.fee)
    else:
        res = run_backtest(df, sig, allow_short=args.short,
                           stop_loss=args.stop or None, take_profit=args.target or None,
                           fee=args.fee, stops_on_close=args.stops_on_close,
                           leverage=args.leverage, breakeven_after_tp1=args.breakeven,
                           trail_targets=args.trail,
                           take_profits=parse_take_profits(args.tp) if args.tp else None)
    for t in res.trades[-10:]:
        print(f"  {t.side:5} {t.entry_time:%Y-%m-%d %H:%M} {t.entry_price:>10.2f} -> "
              f"{t.exit_price:>10.2f} {t.pnl_pct * 100:+7.2f}% ({t.reason}, "
              f"{t.targets_hit} targets)")
    print()
    print(pd.Series(res.stats).to_string())


def cmd_strategies(args) -> None:
    for sid, v in CATALOG.items():
        status = "ready" if v["func"] else "rules pending"
        print(f"{'🟢' if v['func'] else '🔴'} {strategy_label(sid):30} [{status}]  {v['timeframes']}\n"
              f"     source: {v['source']}")


def cmd_signals(args) -> None:
    mc, _ = signals_for(load(args), args)
    if args.date:
        dates = mc.attrs.get("real_dates", mc.index)
        bar = int(dates.get_indexer([pd.Timestamp(args.date, tz="UTC")], method="pad")[0])
    else:
        bar = -1
    if CATALOG[args.strategy]["func"] == "raw":
        print(witcher_checklist(mc, bar))
    else:
        print(checklist(mc, bar))


def cmd_entries(args) -> None:
    """Score entries only: how often price hits the target before the stop."""
    df = load(args)
    _, sig = signals_for(df, args)
    unit = "x ATR" if args.mode == "atr" else "price fraction"
    print(f"[{strategy_label(args.strategy)}]  {len(df)} bars. Target {args.target_dist} / stop {args.stop_dist} ({unit}), "
          f"within {args.max_bars} bars.\n")
    st = summarize(df, sig, args.target_dist, args.stop_dist, args.mode, args.max_bars)
    print(pd.Series(st).to_string())


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
    bot.run(ex, args.symbol, args.timeframe, broker, allow_short=allow_short,
            fraction=args.fraction, stop_loss=args.stop or None, take_profit=args.target or None,
            poll_seconds=args.poll, heikin_ashi=args.heikin_ashi,
            stops_on_close=args.stops_on_close, htf_factor=args.htf,
            mf_period=args.mf_period)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--exchange", default="binance", help="any ccxt exchange id")
    common.add_argument("--symbol", default="BTC/USDT")
    common.add_argument("--timeframe", default="1h", help="chart timeframe (default 1h)")
    common.add_argument("--strategy", default="S1", choices=list(CATALOG),
                        help="which strategy to run (see: python main.py strategies)")
    common.add_argument("--short", action="store_true", help="allow short trades (backtest/paper)")
    common.add_argument("--stop", type=float, default=0.03, help="stop-loss fraction, 0 to disable")
    common.add_argument("--target", type=float, default=0.06, help="take-profit fraction, 0 to disable")
    common.add_argument("--fee", type=float, default=0.001, help="fee per side")
    common.add_argument("--htf", type=int, default=4,
                        help="support/resistance timeframe = this many chart candles (4: 1h -> 4h)")
    common.add_argument("--mf-period", type=int, default=60,
                        help="money flow length in candles; smaller = looser / faster (default 60)")
    common.add_argument("--stops-on-close", action="store_true",
                        help="only trigger stop-loss/take-profit on a closed candle (default: intrabar)")
    common.add_argument("--heikin-ashi", default=True, action=argparse.BooleanOptionalAction,
                        help="Market Cipher B on Heikin Ashi candles (trades still fill at real prices)")

    source = argparse.ArgumentParser(add_help=False)
    source.add_argument("--limit", type=int, default=3000, help="number of candles")
    source.add_argument("--csv", help="load candles from a CSV instead of the exchange")
    source.add_argument("--synthetic", action="store_true", help="use random demo data")
    source.add_argument("--resample", metavar="TIMEFRAME",
                        help="combine smaller candles into this timeframe, e.g. 1-minute data into 1h")
    source.add_argument("--relabel", metavar="TIMEFRAME",
                        help="treat candles as this timeframe, e.g. daily stock data as 1h")

    b = sub.add_parser("backtest", parents=[common, source])
    b.add_argument("--exits", choices=["strategy", "fixed"], default="strategy",
                   help="strategy: each strategy's own exits (S1: stop under the 21 EMA, 50%%/25%% at "
                        "the next 4h levels, trail the 21 EMA; S4: +1.1%% TP, 8.2%%/7.8%% stop). "
                        "fixed: --stop/--tp/--trail percentages")
    b.add_argument("--from", dest="start", help="only use candles from this date on, e.g. 2017-01-01")
    b.add_argument("--stop-buffer", type=float, default=0.002,
                   help="video exits: distance beyond the 21 EMA / level for the stop (0.002 = 0.2%%)")
    b.add_argument("--breakeven", action="store_true", help="move stop to entry after the first target")
    b.add_argument("--trail", action="store_true",
                   help="step the stop up: breakeven after target 1, target 1 after target 2, ...")
    b.add_argument("--leverage", type=float, default=1.0, help="e.g. 5 for 5x")
    b.add_argument("--tp", help='scaled take-profits, price%%:size%%, e.g. "1:50,2:25,3:25"; '
                        '"ride" = hold the rest until the opposite setup, e.g. "1.5:50,2.5:25,ride:25"')
    b.set_defaults(func=cmd_backtest)

    e = sub.add_parser("entries", parents=[common, source],
                       help="score entry quality: target-before-stop hit rate vs random")
    e.add_argument("--mode", choices=["pct", "atr"], default="pct")
    e.add_argument("--target-dist", type=float, default=0.015, help="target distance (0.015 = 1.5%%)")
    e.add_argument("--stop-dist", type=float, default=0.03, help="stop distance (0.03 = 3%%)")
    e.add_argument("--max-bars", type=int, default=48)
    e.set_defaults(func=cmd_entries)

    s = sub.add_parser("signals", parents=[common, source],
                       help="green/red light checklist for the last closed candle")
    s.add_argument("--date", help="show the checklist for the candle at or before this date instead")
    s.set_defaults(func=cmd_signals)

    sub.add_parser("strategies", help="list the labelled strategies").set_defaults(func=cmd_strategies)

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
