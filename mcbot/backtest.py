"""Bar-by-bar backtester.

Signals computed on bar i's close (a confirmed candle) are filled at bar i+1's
open. By default stop-loss and take-profit are checked intrabar against each
bar's high/low; if both are hit in the same bar the stop is assumed to fill
first (conservative). With stops_on_close they are only checked against each
bar's close and filled at the next open, like every other exit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Trade:
    side: str          # "long" or "short"
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    reason: str = ""
    pnl_pct: float = 0.0


@dataclass
class Result:
    trades: list[Trade]
    equity: pd.Series
    stats: dict = field(default_factory=dict)


def run_backtest(df: pd.DataFrame, signals: pd.DataFrame, *, allow_short: bool = False,
                 stop_loss: float | None = 0.03, take_profit: float | None = 0.06,
                 fee: float = 0.001, start_equity: float = 10_000.0,
                 stops_on_close: bool = False) -> Result:
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    c = df["close"].to_numpy()
    idx = df.index
    sig = {k: signals[k].to_numpy() for k in signals.columns}

    equity = start_equity
    curve = np.empty(len(df))
    trades: list[Trade] = []
    pos: Trade | None = None

    def close(i: int, price: float, reason: str) -> None:
        nonlocal equity, pos
        direction = 1 if pos.side == "long" else -1
        gross = direction * (price - pos.entry_price) / pos.entry_price
        pos.pnl_pct = gross - 2 * fee
        equity *= 1 + pos.pnl_pct
        pos.exit_time, pos.exit_price, pos.reason = idx[i], price, reason
        trades.append(pos)
        pos = None

    for i in range(len(df)):
        # 1) act on the previous bar's signals at this bar's open
        if i > 0:
            p = i - 1
            if pos is not None and stops_on_close:
                direction = 1 if pos.side == "long" else -1
                move = direction * (c[p] - pos.entry_price) / pos.entry_price
                if stop_loss and move <= -stop_loss:
                    close(i, o[i], "stop")
                elif take_profit and move >= take_profit:
                    close(i, o[i], "target")
            if pos is not None:
                if pos.side == "long" and sig["long_exit"][p]:
                    close(i, o[i], "signal")
                elif pos.side == "short" and sig["short_exit"][p]:
                    close(i, o[i], "signal")
            if pos is None:
                if sig["long_entry"][p]:
                    pos = Trade("long", idx[i], o[i])
                elif allow_short and sig["short_entry"][p]:
                    pos = Trade("short", idx[i], o[i])

        # 2) intrabar stop-loss / take-profit
        if pos is not None and not stops_on_close:
            ep = pos.entry_price
            if pos.side == "long":
                sl = ep * (1 - stop_loss) if stop_loss else None
                tp = ep * (1 + take_profit) if take_profit else None
                if sl is not None and lo[i] <= sl:
                    close(i, min(sl, o[i]), "stop")
                elif tp is not None and h[i] >= tp:
                    close(i, max(tp, o[i]), "target")
            else:
                sl = ep * (1 + stop_loss) if stop_loss else None
                tp = ep * (1 - take_profit) if take_profit else None
                if sl is not None and h[i] >= sl:
                    close(i, max(sl, o[i]), "stop")
                elif tp is not None and lo[i] <= tp:
                    close(i, min(tp, o[i]), "target")

        # 3) mark to market
        if pos is not None:
            direction = 1 if pos.side == "long" else -1
            curve[i] = equity * (1 + direction * (c[i] - pos.entry_price) / pos.entry_price - fee)
        else:
            curve[i] = equity

    if pos is not None:
        close(len(df) - 1, c[-1], "end")
        curve[-1] = equity

    eq = pd.Series(curve, index=idx)
    return Result(trades, eq, _stats(df, trades, eq, start_equity))


def _stats(df: pd.DataFrame, trades: list[Trade], eq: pd.Series, start: float) -> dict:
    pnls = np.array([t.pnl_pct for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    dd = (eq / eq.cummax() - 1).min()
    rets = eq.pct_change().dropna()
    bars_per_year = _bars_per_year(df.index)
    sharpe = (rets.mean() / rets.std() * np.sqrt(bars_per_year)) if rets.std() > 0 else 0.0
    return {
        "total_return_pct": round((eq.iloc[-1] / start - 1) * 100, 2),
        "buy_hold_return_pct": round((df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100, 2),
        "trades": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else 0.0,
        "avg_trade_pct": round(pnls.mean() * 100, 2) if trades else 0.0,
        "profit_factor": round(wins.sum() / -losses.sum(), 2) if losses.sum() < 0 else float("inf") if len(wins) else 0.0,
        "max_drawdown_pct": round(dd * 100, 2),
        "sharpe": round(float(sharpe), 2),
        "final_equity": round(float(eq.iloc[-1]), 2),
    }


def _bars_per_year(index: pd.Index) -> float:
    if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
        step = pd.Series(index).diff().median().total_seconds()
        if step > 0:
            return 365 * 24 * 3600 / step
    return 252.0
