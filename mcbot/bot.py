"""Live loop: wait for each candle to close, compute signals, trade.

Paper trading is the default. Live trading needs --live plus API keys in the
environment, and only trades spot long positions (no shorting with real money).
Stops and targets are checked by this process on every poll, so they only
work while the bot is running.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .data import fetch_ohlcv
from .indicators import market_cipher
from .strategies import build_signals, checklist

log = logging.getLogger("mcbot")


@dataclass
class Position:
    side: str
    qty: float
    entry_price: float


class PaperBroker:
    """Simulated account persisted to a JSON file."""

    def __init__(self, state_file: str = "paper_state.json", start_cash: float = 10_000.0,
                 fee: float = 0.001):
        self.path = Path(state_file)
        self.fee = fee
        if self.path.exists():
            s = json.loads(self.path.read_text())
            self.cash = s["cash"]
            self.position = Position(**s["position"]) if s["position"] else None
        else:
            self.cash, self.position = start_cash, None

    def _save(self) -> None:
        self.path.write_text(json.dumps({
            "cash": self.cash,
            "position": asdict(self.position) if self.position else None,
        }, indent=2))

    def open(self, side: str, price: float, fraction: float) -> None:
        spend = self.cash * fraction
        qty = spend * (1 - self.fee) / price
        self.cash -= spend
        self.position = Position(side, qty, price)
        self._save()
        log.info("PAPER %s %.6f @ %.2f", side.upper(), qty, price)

    def close(self, price: float, reason: str) -> None:
        p = self.position
        direction = 1 if p.side == "long" else -1
        value = p.qty * p.entry_price + direction * p.qty * (price - p.entry_price)
        self.cash += value * (1 - self.fee)
        log.info("PAPER CLOSE %s @ %.2f (%s) cash=%.2f", p.side, price, reason, self.cash)
        self.position = None
        self._save()


class LiveBroker:
    """Spot market orders through ccxt. Long-only."""

    def __init__(self, exchange, symbol: str):
        self.ex, self.symbol = exchange, symbol
        self.base, self.quote = symbol.split("/")[0], symbol.split("/")[1].split(":")[0]
        self.position: Position | None = None
        self._entry_price: float | None = None
        self._sync()

    def _sync(self) -> None:
        bal = self.ex.fetch_balance()
        qty = float(bal.get("free", {}).get(self.base) or 0)
        min_amt = (self.ex.market(self.symbol).get("limits", {}).get("amount", {}) or {}).get("min") or 0
        if qty > max(min_amt, 0):
            self.position = Position("long", qty, self._entry_price or 0.0)
        else:
            self.position = None

    def open(self, side: str, price: float, fraction: float) -> None:
        if side != "long":
            log.info("Skipping short signal: live mode is spot long-only")
            return
        quote_free = float(self.ex.fetch_balance()["free"].get(self.quote) or 0)
        qty = float(self.ex.amount_to_precision(self.symbol, quote_free * fraction / price))
        order = self.ex.create_market_buy_order(self.symbol, qty)
        self._entry_price = float(order.get("average") or price)
        log.info("LIVE BUY %s %s -> order %s", qty, self.symbol, order.get("id"))
        self._sync()

    def close(self, price: float, reason: str) -> None:
        qty = float(self.ex.amount_to_precision(self.symbol, self.position.qty))
        order = self.ex.create_market_sell_order(self.symbol, qty)
        log.info("LIVE SELL %s %s (%s) -> order %s", qty, self.symbol, reason, order.get("id"))
        self._entry_price = None
        self._sync()


def check_stops(broker, price: float, stop_loss: float | None, take_profit: float | None) -> None:
    p = broker.position
    if p is None or not p.entry_price:
        return
    move = (price - p.entry_price) / p.entry_price * (1 if p.side == "long" else -1)
    if stop_loss and move <= -stop_loss:
        broker.close(price, "stop")
    elif take_profit and move >= take_profit:
        broker.close(price, "target")


def closed_candles(exchange, symbol: str, timeframe: str, history: int = 500) -> pd.DataFrame:
    """Fetch candles and drop the last one if it is still forming."""
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    df = fetch_ohlcv(exchange, symbol, timeframe, limit=history)
    if df.index[-1].value // 10**6 + tf_ms > exchange.milliseconds():
        df = df.iloc[:-1]
    return df


def run(exchange, symbol: str, timeframe: str, broker, *,
        allow_short: bool = False, fraction: float = 0.95, stop_loss: float | None = 0.03,
        take_profit: float | None = 0.06, poll_seconds: int = 30, history: int = 500,
        heikin_ashi: bool = True, stops_on_close: bool = False, htf_factor: int = 4,
        mf_period: int = 60, mf_multiplier: float = 150) -> None:
    """Trade only on confirmed (closed) candles.

    Entries and signal exits happen once per closed candle. Stops and targets are
    checked on every poll unless stops_on_close is set, in which case they are
    also only checked against each closed candle's close.
    """
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    last_bar = None
    log.info("Starting top-down confluence on %s %s with %s (%s candles)", symbol, timeframe,
             type(broker).__name__, "Heikin Ashi" if heikin_ashi else "regular")

    while True:
        try:
            price = float(exchange.fetch_ticker(symbol)["last"])
            if not stops_on_close:
                check_stops(broker, price, stop_loss, take_profit)

            df = closed_candles(exchange, symbol, timeframe, history)
            bar = df.index[-1]
            if bar != last_bar:
                first = last_bar is None
                last_bar = bar
                closed_ago_ms = exchange.milliseconds() - (bar.value // 10**6 + tf_ms)
                if first and closed_ago_ms > 2 * poll_seconds * 1000:
                    log.info("Bar %s closed %ds ago; waiting for the next candle to close",
                             bar, closed_ago_ms // 1000)
                    time.sleep(poll_seconds)
                    continue
                if stops_on_close:
                    check_stops(broker, float(df["close"].iloc[-1]), stop_loss, take_profit)
                mc = market_cipher(df, use_heikin_ashi=heikin_ashi, htf_factor=htf_factor,
                                   mf_period=mf_period, mf_multiplier=mf_multiplier)
                sig = build_signals(mc).iloc[-1]
                log.info("\n%s", checklist(mc))
                pos = broker.position
                log.info("Bar %s close=%.2f signals=%s pos=%s", bar, df["close"].iloc[-1],
                         {k: bool(v) for k, v in sig.items() if v}, pos.side if pos else None)

                if pos and pos.side == "long" and sig["long_exit"]:
                    broker.close(price, "signal")
                elif pos and pos.side == "short" and sig["short_exit"]:
                    broker.close(price, "signal")
                if broker.position is None:
                    if sig["long_entry"]:
                        broker.open("long", price, fraction)
                    elif allow_short and sig["short_entry"]:
                        broker.open("short", price, fraction)
        except KeyboardInterrupt:
            raise
        except Exception:
            log.exception("Loop error; retrying")
        time.sleep(poll_seconds)


def live_credentials() -> tuple[str, str]:
    key, secret = os.environ.get("MCBOT_API_KEY"), os.environ.get("MCBOT_API_SECRET")
    if not key or not secret:
        raise SystemExit("Set MCBOT_API_KEY and MCBOT_API_SECRET to trade live.")
    return key, secret
