"""Number formatting for every screen: full prices, never scientific notation."""

from __future__ import annotations

import math


def _bad(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def price(v) -> str:
    """77,168.70 · 2,226.72 · 95.80 · 0.5432 · 0.00001234 (always written out in full)."""
    if isinstance(v, str):
        return v
    if _bad(v):
        return "-"
    v = float(v)
    if abs(v) >= 1:
        return f"{v:,.2f}"
    if v == 0:
        return "0.00"
    digits = max(4, -int(math.floor(math.log10(abs(v)))) + 3)   # 4 significant figures
    return f"{v:.{digits}f}".rstrip("0").ljust(4, "0")


def money(v) -> str:
    if _bad(v):
        return "-"
    v = float(v)
    s = f"${abs(v):,.2f}" if abs(v) >= 0.01 or v == 0 else f"${price(abs(v))}"
    return f"-{s}" if v < 0 else s


def pct(v) -> str:
    return "-" if _bad(v) else f"{float(v):+.2f}%"


def qty(v) -> str:
    if _bad(v):
        return "-"
    return f"{float(v):,.8f}".rstrip("0").rstrip(".")
