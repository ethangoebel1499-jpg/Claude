"""Take-profit plans for the Strategy Lab: presets, % or risk:reward targets, and checking a plan."""

from __future__ import annotations

import pandas as pd

OTHER = "Other ticker..."
STOCK_NAMES = {"SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100 ETF", "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia",
               "AMZN": "Amazon", "GOOGL": "Alphabet (Google)", "META": "Meta", "TSLA": "Tesla", "AVGO": "Broadcom",
               "BRK-B": "Berkshire Hathaway", "JPM": "JPMorgan", "LLY": "Eli Lilly", "V": "Visa", "WMT": "Walmart",
               "XOM": "Exxon Mobil", "UNH": "UnitedHealth", "MA": "Mastercard", "COST": "Costco", "NFLX": "Netflix",
               "AMD": "AMD", "PLTR": "Palantir"}
# take-profit plans: (% moves, R multiples, sizes); None = the rest rides until the exit signal
TP_PLANS = {
    "Strategy's tested setting": None,
    "1 target (all out)": ([10], [2], [100]),
    "2 targets: 50 / 50": ([5, 10], [1, 2], [50, 50]),
    "3 targets: 50 / 25 / 25": ([3, 6, 10], [1, 2, 3], [50, 25, 25]),
    "3 targets: 33 / 33 / 34": ([3, 6, 10], [1, 2, 3], [33, 33, 34]),
    "2 targets + runner: 50 / 25 / rest rides": ([2, 5, None], [1, 2, None], [50, 25, 25]),
    "1 target + runner: 50 / rest rides": ([5, None], [1.5, None], [50, 50]),
}


def tp_rows(plan: str, r_mode: bool, base, stop: float) -> list[dict]:
    if TP_PLANS[plan] is None:                  # the strategy's own exits
        pairs = [(None if lv is None else (lv / base.stop if r_mode and base.stop else lv * 100), sz * 100)
                 for lv, sz in base.take_profits]
    else:
        moves, rs, sizes = TP_PLANS[plan]
        pairs = list(zip(rs if r_mode else moves, sizes))
    return [{"Target": "Runner" if at is None else f"TP{i + 1}", "At": None if at is None else round(at, 2),
             "Close %": round(sz)} for i, (at, sz) in enumerate(pairs)]


def tps_from_table(df: pd.DataFrame, r_mode: bool, stop: float) -> list[tuple[float | None, float]]:
    rows = [(None if pd.isna(r["At"]) else float(r["At"]), float(r["Close %"])) for _, r in df.iterrows()
            if not pd.isna(r.get("Close %"))]
    if not rows:
        raise ValueError("Add at least one take profit.")
    if abs(sum(sz for _, sz in rows) - 100) > 0.01:
        raise ValueError(f"The close sizes add up to {sum(sz for _, sz in rows):g}% - they need to add up to 100%.")
    if any(at is None for at, _ in rows[:-1]):
        raise ValueError("Only the last row can be left empty (the runner).")
    if r_mode and not stop:
        raise ValueError("Risk:reward targets need a stop - set Stop % above 0.")
    out = [(None if at is None else (at * stop if r_mode else at / 100), sz / 100) for at, sz in rows]
    fixed = [lv for lv, _ in out if lv is not None]
    if fixed != sorted(fixed):
        raise ValueError("Targets must go from nearest to farthest.")
    return out


def plan_text(tps, stop: float) -> str:
    parts = []
    for i, (lv, sz) in enumerate(tps):
        if lv is None:
            parts.append(f"rest ({sz * 100:g}%) rides until the exit signal")
        else:
            r = f" = {lv / stop:.2g}R" if stop else ""
            parts.append(f"TP{i + 1} at +{lv * 100:.2f}%{r}: close {sz * 100:g}%")
    return " · ".join(parts) + (f" · stop -{stop * 100:g}%" if stop else " · no stop")
