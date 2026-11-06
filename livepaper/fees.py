"""Trading fees: one setting (⚙️ Account → Fees) used everywhere - the paper broker, every backtest, 🎓 Learn's
scoring and checks, and the 🔬 Proving ground - so what the apprentice learns is after the costs you'd pay.

Fees are charged on the full position (so leverage multiplies them too); a round trip pays twice.
"""

from __future__ import annotations

PRESETS = {
    "kraken_spot_taker": ("Kraken spot - market orders (taker) 0.40%", 0.004),
    "kraken_spot_maker": ("Kraken spot - limit orders (maker) 0.25%", 0.0025),
    "kraken_futures": ("Kraken futures - taker 0.05% (plus funding while you hold)", 0.0005),
    "custom": ("Custom", None),
}
DEFAULT = "kraken_spot_taker"


def preset() -> str:
    try:
        from . import settings
        p = settings.saved().get("fee_preset") or DEFAULT
    except Exception:
        p = DEFAULT
    return p if p in PRESETS else DEFAULT


def current() -> float:
    """The fee per side as a fraction (0.004 = 0.40%)."""
    p = preset()
    if p == "custom":
        try:
            from . import settings
            return float(settings.saved().get("fee", 0.001))
        except Exception:
            return 0.001
    return PRESETS[p][1]


def label() -> str:
    p = preset()
    return PRESETS[p][0] if p != "custom" else f"Custom {current() * 100:.2f}%"


def round_trip_pct() -> float:
    return current() * 2 * 100
