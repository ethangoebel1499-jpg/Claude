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

from .data import fetch_ohlcv
from .indicators import market_cipher
from .strategies import STRATEGIES

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


def run(exchange, symbol: str, timeframe: str, strategy: str, broker, *,
        allow_short: bool = False, fraction: float = 0.95, stop_loss: float | None = 0.03,
        take_profit: float | None = 0.06, poll_seconds: int = 30, history: int = 500) -> None:
    strat = STRATEGIES[strategy]
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    last_bar = None
    log.info("Starting %s on %s %s with %s", strategy, symbol, timeframe, type(broker).__name__)

    while True:
        try:
            price = float(exchange.fetch_ticker(symbol)["last"])
            check_stops(broker, price, stop_loss, take_profit)

            df = fetch_ohlcv(exchange, symbol, timeframe, limit=history)
            # Drop the candle that is still forming.
            if df.index[-1].value // 10**6 + tf_ms > exchange.milliseconds():
                df = df.iloc[:-1]
            bar = df.index[-1]
            if bar != last_bar:
                last_bar = bar
                sig = strat(market_cipher(df)).iloc[-1]
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
