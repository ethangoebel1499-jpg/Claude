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
    exit_price: float | None = None   # price of the final exit
    reason: str = ""                  # reason for the final exit
    pnl_pct: float = 0.0              # return on account equity, after leverage and fees
    remaining: float = 1.0            # fraction of the position still open
    targets_hit: int = 0
    stop_move: float | None = None    # current stop as a price move (-0.03, or 0 at breakeven)
    fills: list = field(default_factory=list)  # (time, price, fraction, reason)


@dataclass
class Result:
    trades: list[Trade]
    equity: pd.Series
    stats: dict = field(default_factory=dict)


def parse_take_profits(spec: str) -> list[tuple[float | None, float]]:
    """"1:50,2:25,3:25" -> [(0.01, 0.5), (0.02, 0.25), (0.03, 0.25)].

    Each item is price-move percent : percent of the original position to close.
    Use "ride" instead of a percent for a runner that stays open until the
    opposite setup appears: "1.5:50,2.5:25,ride:25".
    """
    levels = []
    for part in spec.split(","):
        pct, size = part.split(":")
        level = None if pct.strip().lower() == "ride" else float(pct) / 100
        levels.append((level, float(size) / 100))
    if abs(sum(size for _, size in levels) - 1) > 1e-9:
        raise ValueError("take-profit sizes must add up to 100%")
    if sum(level is None for level, _ in levels) > 1:
        raise ValueError("only one 'ride' portion is allowed")
    fixed = sorted(lv for lv in levels if lv[0] is not None)
    return fixed + [lv for lv in levels if lv[0] is None]


def run_backtest(df: pd.DataFrame, signals: pd.DataFrame, *, allow_short: bool = False,
                 stop_loss: float | None = 0.03, take_profit: float | None = 0.06,
                 take_profits: list[tuple[float, float]] | None = None,
                 leverage: float = 1.0, maintenance_margin: float = 0.005,
                 fee: float = 0.001, start_equity: float = 10_000.0,
                 stops_on_close: bool = False, breakeven_after_tp1: bool = False,
                 trail_targets: bool = False) -> Result:
    """stop_loss / take_profit levels are price moves (0.03 = 3%). With leverage L,
    each 1% price move is L% on the account, and fees are charged on the
    leveraged notional. take_profits overrides take_profit with scaled exits.
    A take_profits level of None is a runner: once every fixed target has
    filled, the rest ignores normal exit signals and stays open until the
    opposite entry setup appears (or the stop / breakeven stop is hit).
    The whole account is used as margin; a position is liquidated (-100%) if
    price moves 1/L - maintenance_margin against it before the stop fills.
    With breakeven_after_tp1, the stop moves to the entry price once the first
    target fills, starting from the next bar (within one bar the order of the
    high and low is unknown). trail_targets steps the stop up behind each target:
    breakeven after the first, then the previous target's level after each
    later one (after target 2 the stop sits at target 1).
    """
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    c = df["close"].to_numpy()
    idx = df.index
    sig = {k: signals[k].to_numpy() for k in signals.columns}
    targets = take_profits if take_profits else ([(take_profit, 1.0)] if take_profit else [])
    has_runner = any(level is None for level, _ in targets)
    targets = [t for t in targets if t[0] is not None]
    liq_move = 1 / leverage - maintenance_margin if leverage > 1 else None

    equity = start_equity
    curve = np.empty(len(df))
    trades: list[Trade] = []
    pos: Trade | None = None

    def move_of(price: float) -> float:
        d = 1 if pos.side == "long" else -1
        return d * (price - pos.entry_price) / pos.entry_price

    def price_at(move: float) -> float:
        d = 1 if pos.side == "long" else -1
        return pos.entry_price * (1 + d * move)

    def exit_part(i: int, price: float, frac: float, reason: str) -> None:
        nonlocal equity, pos
        frac = min(frac, pos.remaining)
        pos.pnl_pct += frac * (leverage * move_of(price) - 2 * fee * leverage)
        pos.remaining -= frac
        pos.fills.append((idx[i], price, frac, reason))
        if pos.remaining <= 1e-9:
            pos.pnl_pct = max(pos.pnl_pct, -1.0)
            equity *= 1 + pos.pnl_pct
            pos.exit_time, pos.exit_price, pos.reason = idx[i], price, reason
            trades.append(pos)
            pos = None

    def liquidate(i: int) -> None:
        nonlocal equity, pos
        pos.pnl_pct = -1.0
        equity = 0.0
        pos.remaining = 0.0
        pos.exit_time, pos.exit_price, pos.reason = idx[i], price_at(-liq_move), "liquidated"
        trades.append(pos)
        pos = None

    def check_levels(i: int, worst: float, best: float, fill_open: float | None) -> None:
        """worst/best: the most adverse / favourable price move seen. Stop first
        (conservative), then each target in order."""
        stop = pos.stop_move
        if liq_move is not None and worst <= -liq_move and (stop is None or -stop >= liq_move
                                                           or move_of(o[i]) <= -liq_move):
            liquidate(i)
            return
        if stop is not None and worst <= stop:
            px = fill_open if fill_open is not None else price_at(min(stop, move_of(o[i])))
            reason = "stop" if stop < 0 else "breakeven" if stop == 0 else "trail stop"
            exit_part(i, px, pos.remaining, reason)
            return
        while pos is not None and pos.targets_hit < len(targets):
            level, size = targets[pos.targets_hit]
            if best < level:
                break
            px = fill_open if fill_open is not None else price_at(max(level, move_of(o[i])))
            pos.targets_hit += 1
            if trail_targets:
                k = pos.targets_hit
                pos.stop_move = 0.0 if k == 1 else targets[k - 2][0]
            elif breakeven_after_tp1:
                pos.stop_move = 0.0
            last = pos.targets_hit == len(targets) and not has_runner
            exit_part(i, px, pos.remaining if last else size, f"target{pos.targets_hit}")

    for i in range(len(df)):
        # 1) act on the previous (confirmed) bar at this bar's open
        if i > 0:
            p = i - 1
            if pos is not None and stops_on_close:
                m = move_of(c[p])
                check_levels(i, m, m, o[i])
            if pos is not None:
                # The opposite setup always closes the trade. A runner (all fixed
                # targets hit) ignores the normal exit signals and rides until then.
                riding = has_runner and pos.targets_hit == len(targets)
                exit_now, opposite = (("long_exit", "short_entry") if pos.side == "long"
                                      else ("short_exit", "long_entry"))
                if sig[opposite][p]:
                    exit_part(i, o[i], pos.remaining, "opposite setup")
                elif sig[exit_now][p] and not riding:
                    exit_part(i, o[i], pos.remaining, "signal")
            if pos is None and equity > 0:
                stop0 = -stop_loss if stop_loss else None
                if sig["long_entry"][p]:
                    pos = Trade("long", idx[i], o[i], stop_move=stop0)
                elif allow_short and sig["short_entry"][p]:
                    pos = Trade("short", idx[i], o[i], stop_move=stop0)

        # 2) intrabar stop-loss / take-profits / liquidation
        if pos is not None:
            adverse, favourable = (lo[i], h[i]) if pos.side == "long" else (h[i], lo[i])
            if stops_on_close:
                if liq_move is not None and move_of(adverse) <= -liq_move:
                    liquidate(i)
            else:
                check_levels(i, move_of(adverse), move_of(favourable), None)

        # 3) mark to market
        if pos is not None:
            unreal = pos.remaining * (leverage * move_of(c[i]) - fee * leverage)
            curve[i] = equity * (1 + max(pos.pnl_pct + unreal, -1.0))
        else:
            curve[i] = equity

    if pos is not None:
        exit_part(len(df) - 1, c[-1], pos.remaining, "end")
        curve[-1] = equity

    eq = pd.Series(curve, index=idx)
    return Result(trades, eq, _stats(df, trades, eq, start_equity))


def run_level_backtest(df: pd.DataFrame, signals: pd.DataFrame, levels: pd.DataFrame, *,
                       allow_short: bool = False, leverage: float = 1.0,
                       maintenance_margin: float = 0.005, fee: float = 0.001,
                       sizes: tuple[float, float] = (0.5, 0.25),
                       min_rr: float | None = None,
                       start_equity: float = 10_000.0) -> Result:
    """Backtest with price-level exits (the top-down video's exit plan).

    At entry (next bar's open after the signal bar) the trade takes its stop and
    two targets from `levels` on the signal bar: long_stop/long_tp1/long_tp2 (or
    the short_ columns). `sizes` of the position close at target 1 and target 2;
    the rest has no target. After every closed bar the stop trails to
    long_trail (only ever upward; downward for shorts). The opposite entry setup
    closes the trade. Stop is assumed to fill first if a bar touches both.

    min_rr: target 1 must be at least min_rr times as far from the entry as the
    stop. If the first level is too close, the second level becomes target 1
    (and there is no target 2); if neither qualifies the trade is skipped.
    """
    o, h, lo, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    idx = df.index
    sig = {k: signals[k].to_numpy() for k in signals.columns}
    lv = {k: levels[k].to_numpy(dtype=float) for k in levels.columns}
    liq_move = 1 / leverage - maintenance_margin if leverage > 1 else None

    equity = start_equity
    curve = np.empty(len(df))
    trades: list[Trade] = []
    pos: Trade | None = None
    stop = 0.0
    targets: list[tuple[float, float]] = []

    def move(price: float) -> float:
        d = 1 if pos.side == "long" else -1
        return d * (price - pos.entry_price) / pos.entry_price

    def exit_part(i: int, price: float, frac: float, reason: str) -> None:
        nonlocal equity, pos
        frac = min(frac, pos.remaining)
        pos.pnl_pct += frac * (leverage * move(price) - 2 * fee * leverage)
        pos.remaining -= frac
        pos.fills.append((idx[i], price, frac, reason))
        if pos.remaining <= 1e-9:
            pos.pnl_pct = max(pos.pnl_pct, -1.0)
            equity *= 1 + pos.pnl_pct
            pos.exit_time, pos.exit_price, pos.reason = idx[i], price, reason
            trades.append(pos)
            pos = None

    def open_trade(i: int, p: int, side: str) -> None:
        nonlocal pos, stop, targets
        e = o[i]
        s0 = lv[f"{side}_stop"][p]
        t1, t2 = lv[f"{side}_tp1"][p], lv[f"{side}_tp2"][p]
        ok = (lambda x: x < e) if side == "long" else (lambda x: x > e)
        beyond = (lambda x, y: x > y) if side == "long" else (lambda x, y: x < y)
        if not np.isfinite(s0) or not ok(s0):
            return  # no valid stop on the right side of the entry: skip the trade
        levels = [t for t in (t1, t2) if np.isfinite(t) and beyond(t, e)]
        if len(levels) == 2 and not beyond(levels[1], levels[0]):
            levels = levels[:1]
        if min_rr:
            risk = abs(e - s0)
            levels = [t for t in levels if abs(t - e) >= min_rr * risk]
            if not levels:
                return  # no target far enough away: skip the trade
        tg = [(t, size) for t, size in zip(levels, sizes)]
        pos = Trade(side, idx[i], e)
        stop, targets = s0, tg

    for i in range(len(df)):
        if i > 0:
            p = i - 1
            if pos is not None:
                # Trail the stop with the previous closed bar's level.
                if pos.side == "long":
                    stop = max(stop, lv["long_trail"][p])
                else:
                    stop = min(stop, lv["short_trail"][p])
                opposite = "short_entry" if pos.side == "long" else "long_entry"
                if sig[opposite][p]:
                    exit_part(i, o[i], pos.remaining, "opposite setup")
            if pos is None and equity > 0:
                if sig["long_entry"][p]:
                    open_trade(i, p, "long")
                elif allow_short and sig["short_entry"][p]:
                    open_trade(i, p, "short")

        if pos is not None:
            long_ = pos.side == "long"
            worst = lo[i] if long_ else h[i]
            best = h[i] if long_ else lo[i]
            if liq_move is not None and move(worst) <= -liq_move and \
                    move(stop) < -liq_move:
                pos.pnl_pct, pos.remaining = -1.0, 0.0
                equity = 0.0
                pos.exit_time, pos.exit_price, pos.reason = idx[i], worst, "liquidated"
                trades.append(pos)
                pos = None
            elif (worst <= stop) if long_ else (worst >= stop):
                fill = min(stop, o[i]) if long_ else max(stop, o[i])
                reason = "stop" if move(stop) < 0 else "trail stop"
                exit_part(i, fill, pos.remaining, reason)
            else:
                while pos is not None and pos.targets_hit < len(targets):
                    price, size = targets[pos.targets_hit]
                    if (best < price) if long_ else (best > price):
                        break
                    fill = max(price, o[i]) if long_ else min(price, o[i])
                    pos.targets_hit += 1
                    exit_part(i, fill, size, f"target{pos.targets_hit}")

        if pos is not None:
            unreal = pos.remaining * (leverage * move(c[i]) - fee * leverage)
            curve[i] = equity * (1 + max(pos.pnl_pct + unreal, -1.0))
        else:
            curve[i] = equity

    if pos is not None:
        exit_part(len(df) - 1, c[-1], pos.remaining, "end")
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
        "stopped_out": sum(t.reason == "stop" for t in trades),
        "breakeven_exits": sum(t.reason == "breakeven" for t in trades),
        "trail_stop_exits": sum(t.reason == "trail stop" for t in trades),
        "liquidated": sum(t.reason == "liquidated" for t in trades),
        "hit_target1_pct": round(sum(t.targets_hit >= 1 for t in trades) / len(trades) * 100, 1) if trades else 0.0,
    }


def _bars_per_year(index: pd.Index) -> float:
    if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
        step = pd.Series(index).diff().median().total_seconds()
        if step > 0:
            return 365 * 24 * 3600 / step
    return 252.0
