"""The stock list the Strategy Lab searches: ~1,100 US stocks (S&P 500, Nasdaq 100, S&P 600 small caps)
plus common ETFs, with names and sectors. Built from Wikipedia index lists (via the pytickersymbols
package's data); any other Yahoo Finance ticker can still be typed in."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

FILE = Path(__file__).with_name("data") / "stocks.csv"


@lru_cache(maxsize=1)
def universe() -> list[dict]:
    if not FILE.exists():                      # never crash the dashboard over the list: fall back to the short one
        from .history import STOCKS
        return [{"ticker": t, "name": "", "sector": "", "index": ""} for t in STOCKS]
    with open(FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tickers() -> list[str]:
    return [r["ticker"] for r in universe()]


@lru_cache(maxsize=1)
def labels() -> dict[str, str]:
    """'AAPL' -> 'AAPL · Apple Inc. · Information Technology' (what the search box matches against)."""
    return {r["ticker"]: " · ".join(x for x in (r["ticker"], r["name"], r["sector"]) if x) for r in universe()}
