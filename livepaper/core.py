"""Position management shared by live paper trading and historical replay.

It follows mcbot.backtest.run_backtest exactly: at each closed candle the
opposite setup closes the trade (a runner ignores normal exit signals), then a
new entry may open; between candles the stop is checked first, then each take
profit in order. Fills go through a broker; P&L uses the broker's actual fill
prices and quantities, so live slippage shows up in the results.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd

from mcbot.fmt import price as fmt_price, qty as fmt_qty


@dataclass
class Exits:
    stop: float | None                              # 0.04 = 4% stop, None = no stop
    take_profits: list[tuple[float | None, float]]  # (move, fraction of the original position); None = runner


@dataclass
class Position:
    side: str
    entry_time: str
    entry_price: float
    qty: float                 # original quantity
    remaining_qty: float
    targets_hit: int = 0
    realized_usd: float = 0.0
    fees_usd: float = 0.0


@dataclass
class State:
    equity: float
    position: Position | None = None
    last_bar: str | None = None
    pending_entry: str | None = None   # side waiting for a fill (e.g. stock market closed)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "State":
        pos = Position(**d["position"]) if d.get("position") else None
        return cls(equity=d["equity"], position=pos, last_bar=d.get("last_bar"), pending_entry=d.get("pending_entry"))


@dataclass
class Fill:
    price: float
    qty: float
    fee_usd: float
    order_id: str = ""


class PositionManager:
    def __init__(self, name: str, exits: Exits, *, shorts: bool, broker, store=None,
                 fraction: float = 1.0, qty_step: float = 0.0, state: State | None = None,
                 start_equity: float = 10_000.0):
        self.name, self.exits, self.shorts = name, exits, shorts
        self.broker, self.store = broker, store
        self.fraction, self.qty_step = fraction, qty_step
        self.state = state or State(equity=start_equity)
        self.fixed = [t for t in exits.take_profits if t[0] is not None]
        self.has_runner = any(t[0] is None for t in exits.take_profits)
        self.trades: list[dict] = []

    # ---- helpers -------------------------------------------------------------
    def _dir(self) -> int:
        return 1 if self.state.position.side == "long" else -1

    def _move(self, price: float) -> float:
        p = self.state.position
        return self._dir() * (price - p.entry_price) / p.entry_price

    def _price_at(self, move: float) -> float:
        p = self.state.position
        return p.entry_price * (1 + self._dir() * move)

    def _round_qty(self, qty: float) -> float:
        if self.qty_step:
            return math.floor(qty / self.qty_step + 1e-9) * self.qty_step
        return qty

    def _save(self):
        if self.store:
            self.store.save_state(self.name, self.state)

    def _event(self, time, kind: str, text: str):
        if self.store:
            self.store.event(self.name, time, kind, text)

    # ---- actions -------------------------------------------------------------
    def open(self, side: str, price: float, time) -> bool:
        qty = self._round_qty(self.state.equity * self.fraction / price)
        if qty <= 0:
            self._event(time, "skip", f"{side} entry skipped: position size rounds to 0")
            return True
        fill = self.broker.fill(self.name, "buy" if side == "long" else "sell", qty, price, "entry")
        if fill is None:
            self.state.pending_entry = side
            self._event(time, "pending", f"{side} entry not filled yet (will retry)")
            self._save()
            return False
        self.state.pending_entry = None
        self.state.position = Position(side, str(time), fill.price, fill.qty, fill.qty, fees_usd=fill.fee_usd)
        if self.store:
            self.store.fill(self.name, time, side, "entry", fill)
        self._event(time, "entry", f"{side.upper()} {fmt_qty(fill.qty)} @ {fmt_price(fill.price)}")
        self._save()
        return True

    def close_part(self, qty: float, price: float, time, reason: str) -> bool:
        p = self.state.position
        qty = min(qty, p.remaining_qty)
        fill = self.broker.fill(self.name, "sell" if p.side == "long" else "buy", qty, price, reason)
        if fill is None:
            self._event(time, "pending", f"{reason} exit not filled yet (will retry)")
            return False
        pnl = self._dir() * (fill.price - p.entry_price) * fill.qty
        p.realized_usd += pnl
        p.fees_usd += fill.fee_usd
        p.remaining_qty -= fill.qty
        if self.store:
            self.store.fill(self.name, time, p.side, reason, fill)
        if p.remaining_qty <= max(1e-12, p.qty * 1e-9):
            net = p.realized_usd - p.fees_usd
            trade = {"run": self.name, "side": p.side, "entry_time": p.entry_time, "entry_price": p.entry_price,
                     "qty": p.qty, "exit_time": str(time), "exit_price": fill.price, "reason": reason,
                     "targets_hit": p.targets_hit, "pnl_usd": net, "pnl_pct": net / self.state.equity * 100}
            self.state.equity += net
            self.state.position = None
            self.trades.append(trade)
            if self.store:
                self.store.trade(trade)
            self._event(time, "exit", f"closed ({reason}) {net:+,.2f} USD")
        self._save()
        return True

    def close_all(self, price: float, time, reason: str) -> bool:
        return self.close_part(self.state.position.remaining_qty, price, time, reason)

    # ---- the two hooks ---------------------------------------------------------
    def on_bar_close(self, sig: pd.Series, price: float, time, bar_time=None) -> None:
        """Act on the candle that just closed (bar_time = its start), at the current
        price (the next open). time is when it happens."""
        p = self.state.position
        if p is not None:
            riding = self.has_runner and p.targets_hit == len(self.fixed)
            exit_now, opposite = ("long_exit", "short_entry") if p.side == "long" else ("short_exit", "long_entry")
            if sig[opposite]:
                self.close_all(price, time, "opposite setup")
            elif sig[exit_now] and not riding:
                self.close_all(price, time, "signal")
        self.state.pending_entry = None
        if self.state.position is None and self.state.equity > 0:
            if sig["long_entry"]:
                self.open("long", price, time)
            elif self.shorts and sig["short_entry"]:
                self.open("short", price, time)
        self.state.last_bar = str(bar_time if bar_time is not None else time)
        self._save()

    def retry_pending(self, price: float, time) -> None:
        if self.state.pending_entry and self.state.position is None:
            self.open(self.state.pending_entry, price, time)

    def on_tick(self, open_: float, low: float, high: float, time) -> None:
        """Price check between candles: stop first, then targets in order.
        open_ is where the price started (gaps through a level fill there)."""
        p = self.state.position
        if p is None:
            return
        adverse, favourable = (low, high) if p.side == "long" else (high, low)
        stop = self._stop_move()
        if stop is not None and self._move(adverse) <= stop:
            reason = "stop" if stop < 0 else "breakeven" if stop == 0 else "trail stop"
            self.close_all(self._price_at(min(stop, self._move(open_))), time, reason)
            return
        while self.state.position is not None and p.targets_hit < len(self.fixed):
            level, size = self.fixed[p.targets_hit]
            if self._move(favourable) < level:
                break
            px = self._price_at(max(level, self._move(open_)))
            last = p.targets_hit + 1 == len(self.fixed) and not self.has_runner
            qty = p.remaining_qty if last else self._round_qty(p.qty * size)
            p.targets_hit += 1
            if not self.close_part(qty, px, time, f"target{p.targets_hit}"):
                p.targets_hit -= 1
                break

    def _stop_move(self) -> float | None:
        return -self.exits.stop if self.exits.stop else None

    def unrealized(self, price: float) -> float:
        p = self.state.position
        if p is None:
            return 0.0
        return p.realized_usd - p.fees_usd + self._dir() * (price - p.entry_price) * p.remaining_qty


class SimBroker:
    """Paper broker. fill='intended' fills at the price the logic asks for (replay,
    matches the backtester); fill='market' fills at the latest market price (live)."""

    def __init__(self, fee: float = 0.001, fill: str = "intended", market_price=None):
        self.fee, self.mode, self.market_price = fee, fill, market_price
        self.n = 0

    def fill(self, run: str, side: str, qty: float, price: float, reason: str) -> Fill:
        px = self.market_price(run) if self.mode == "market" and self.market_price else price
        self.n += 1
        return Fill(px, qty, abs(px * qty) * self.fee, f"sim-{self.n}")


def replay(df: pd.DataFrame, sig: pd.DataFrame, manager: PositionManager) -> PositionManager:
    """Run the live logic over historical candles: act on candle i-1 at candle i's
    open, then check the stop / targets against candle i's range."""
    o, lo, hi, c = (df[k].to_numpy(float) for k in ("open", "low", "high", "close"))
    for i, t in enumerate(df.index):
        if i > 0:
            manager.on_bar_close(sig.iloc[i - 1], o[i], t)
        manager.on_tick(o[i], lo[i], hi[i], t)
    if manager.state.position is not None:
        manager.close_all(c[-1], df.index[-1], "end")
    return manager
