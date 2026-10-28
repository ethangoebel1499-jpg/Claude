"""Divergences between price and an oscillator, without lookahead.

The oscillator is split into waves: a trough runs from a down-turn to the next
up-turn (for Market Cipher B: a red-dot cross to a green-dot cross), a peak from
an up-turn to the next down-turn. A wave is known once it completes (at the
turn bar). Per bar we keep the last two completed troughs and peaks: the lowest
oscillator value and lowest price low in each trough, the highest values in each
peak (plus any extra series, e.g. money flow, tracked over the same waves).

  regular bullish  price lower low,  oscillator higher low   (the uneven butt cheeks)
  hidden bullish   price higher low, oscillator lower low
  regular bearish  price higher high, oscillator lower high
  hidden bearish   price lower high,  oscillator higher high
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def waves(osc: pd.Series, low: pd.Series, high: pd.Series, up: pd.Series, down: pd.Series,
          extra: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Last two completed troughs (t1 = latest, t0 = the one before) and peaks
    (p1, p0), as known at each bar's close. Columns: t1_osc, t1_px, t1_i, t0_osc,
    t0_px, t0_i, p1_osc, p1_px, p1_i, p0_..., plus t1_<extra>/t0_<extra> (min over
    the trough) and p1_<extra>/p0_<extra> (max over the peak). Also run_min_osc /
    run_min_px (the trough forming now, since the last down-turn) and run_max_osc /
    run_max_px."""
    o, lo, hi = (np.asarray(x, dtype=float) for x in (osc, low, high))
    u, d = np.asarray(up, dtype=bool), np.asarray(down, dtype=bool)
    ex = {k: np.asarray(v, dtype=float) for k, v in (extra or {}).items()}
    n = len(o)
    names = ["osc", "px", "i"] + list(ex)
    out = {f"{w}{k}_{c}": np.full(n, np.nan) for w in "tp" for k in "01" for c in names}
    for c in ("run_min_osc", "run_min_px", "run_max_osc", "run_max_px"):
        out[c] = np.full(n, np.nan)
    inf = np.inf
    tmin = {"osc": inf, "px": inf, **{k: inf for k in ex}}
    pmax = {"osc": -inf, "px": -inf, **{k: -inf for k in ex}}
    t1 = t0 = p1 = p0 = None
    for i in range(n):
        if d[i]:
            tmin = {"osc": inf, "px": inf, **{k: inf for k in ex}}
        if u[i]:
            pmax = {"osc": -inf, "px": -inf, **{k: -inf for k in ex}}
        tmin["osc"] = min(tmin["osc"], o[i]); tmin["px"] = min(tmin["px"], lo[i])
        pmax["osc"] = max(pmax["osc"], o[i]); pmax["px"] = max(pmax["px"], hi[i])
        for k, v in ex.items():
            tmin[k] = min(tmin[k], v[i]); pmax[k] = max(pmax[k], v[i])
        if u[i] and np.isfinite(tmin["osc"]):
            t0, t1 = t1, {**tmin, "i": i}
        if d[i] and np.isfinite(pmax["osc"]):
            p0, p1 = p1, {**pmax, "i": i}
        for tag, wv in (("t1", t1), ("t0", t0), ("p1", p1), ("p0", p0)):
            if wv is not None:
                for c in names:
                    out[f"{tag}_{c}"][i] = wv[c]
        out["run_min_osc"][i], out["run_min_px"][i] = tmin["osc"], tmin["px"]
        out["run_max_osc"][i], out["run_max_px"][i] = pmax["osc"], pmax["px"]
    return pd.DataFrame(out, index=osc.index)


def pair_divergences(w: pd.DataFrame, up: pd.Series, down: pd.Series, max_gap: int) -> pd.DataFrame:
    """Divergence between the last two completed waves, flagged on the bar the
    second wave completes (the green / red dot)."""
    tgap = (w.t1_i - w.t0_i) <= max_gap
    pgap = (w.p1_i - w.p0_i) <= max_gap
    return pd.DataFrame({
        "bull": up & tgap & (w.t1_px < w.t0_px) & (w.t1_osc > w.t0_osc),
        "hidden_bull": up & tgap & (w.t1_px > w.t0_px) & (w.t1_osc < w.t0_osc),
        "bear": down & pgap & (w.p1_px > w.p0_px) & (w.p1_osc < w.p0_osc),
        "hidden_bear": down & pgap & (w.p1_px < w.p0_px) & (w.p1_osc > w.p0_osc),
    }, index=w.index).fillna(False).astype(bool)


def recent(flag: pd.Series, bars: int) -> pd.Series:
    """True if flag fired in the last `bars` bars (including this one)."""
    return flag.astype(int).rolling(bars, min_periods=1).max().astype(bool)
