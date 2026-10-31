"""Did each call play out? Checked against real candles from the moment it was said.

  calls with levels:  wait for the entry (or take the price when it was said), then walk
                      forward: target first = win, stop first = loss (stop wins a tie,
                      conservative), neither within the horizon = open / expired
  every call:         return in the called direction after 24h / 72h / 7d, and the best
                      (MFE) and worst (MAE) move while it was live
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
KRAKEN = {"BTC": "XBTUSD", "XBT": "XBTUSD", "DOGE": "XDGUSD"}
LOCAL = {"XBTUSD": "datasets/XBTUSD_60.csv", "ETHUSD": "datasets/ETHUSD_60.csv",
         "SOLUSD": "datasets/SOLUSD_60.csv", "XRPUSD": "datasets/XRPUSD_60.csv"}
ENTRY_WINDOW = pd.Timedelta("3D")
HORIZON = pd.Timedelta("14D")


def pair(asset: str) -> str:
    a = asset.upper().replace("/", "").replace("USDT", "").replace("USD", "").strip()
    return KRAKEN.get(a, f"{a}USD")


class Candles:
    """Hourly candles: the repo's Kraken history first, Kraken's public API for anything newer."""

    def __init__(self):
        self.cache: dict[str, pd.DataFrame] = {}
        self.http = requests.Session()

    def _api(self, p: str, since: pd.Timestamp) -> pd.DataFrame:
        age = pd.Timestamp(datetime.now(timezone.utc)) - since
        interval = 60 if age < pd.Timedelta("29D") else 240 if age < pd.Timedelta("118D") else 1440
        r = self.http.get("https://api.kraken.com/0/public/OHLC", params={"pair": p, "interval": interval},
                          timeout=30)
        js = r.json()
        if js.get("error"):
            raise RuntimeError(f"Kraken {p}: {js['error']}")
        rows = next(v for k, v in js["result"].items() if k != "last")
        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
        df.index = pd.to_datetime(df.pop("time").astype(int), unit="s", utc=True)
        return df[["open", "high", "low", "close"]].astype(float)

    def get(self, asset: str, since: pd.Timestamp) -> pd.DataFrame:
        p = pair(asset)
        key = f"{p}@{since.floor('D')}"
        if key not in self.cache:
            frames = []
            if p in LOCAL and (ROOT / LOCAL[p]).exists():
                from mcbot.data import load_csv
                frames.append(load_csv(ROOT / LOCAL[p])[["open", "high", "low", "close"]])
            if not frames or frames[0].index[-1] < since + HORIZON:
                try:
                    frames.append(self._api(p, since))
                except Exception as e:  # no data -> the call stays unchecked
                    if not frames:
                        raise RuntimeError(f"no candles for {asset} ({p}): {e}")
            df = pd.concat(frames).sort_index()
            self.cache[key] = df[~df.index.duplicated(keep="last")]
        return self.cache[key]


def check(call: dict, candles: pd.DataFrame, now: pd.Timestamp | None = None) -> dict:
    now = now or pd.Timestamp(datetime.now(timezone.utc))
    said = pd.Timestamp(call["said_at"])
    d = 1 if call["direction"] == "long" else -1
    after = candles[candles.index >= said.floor("h")]
    out = {"call_id": call["id"], "checked": str(now)}
    if after.empty:
        return {**out, "status": "no data"}
    entry = call.get("entry")
    targets = sorted(json.loads(call["targets"]) if isinstance(call["targets"], str) else call["targets"],
                     reverse=d < 0)
    stop = call.get("stop")

    # 1) entry: a limit at the stated price within ENTRY_WINDOW, else the price when it was said
    if entry and entry > 0:
        window = after[after.index < said + ENTRY_WINDOW]
        touched = window[(window.low <= entry) & (window.high >= entry)]
        if touched.empty:
            # price may already be beyond the entry in the called direction: treat as filled at once
            first = after.iloc[0]
            if (d > 0 and first.open <= entry) or (d < 0 and first.open >= entry):
                t0, px = after.index[0], float(first.open)
            else:
                return {**out, "status": "entry never hit" if said + ENTRY_WINDOW < now else "waiting for entry"}
        else:
            t0, px = touched.index[0], float(entry)
    else:
        t0, px = after.index[0], float(after.iloc[0].open)
    live = after[after.index >= t0]
    out.update(entry_time=str(t0), entry_price=px)

    def ret_at(delta):
        seg = live[live.index <= t0 + delta]
        return round(d * (seg.close.iloc[-1] / px - 1) * 100, 2) if t0 + delta <= live.index[-1] else None

    out.update(ret_24h=ret_at(pd.Timedelta("24h")), ret_72h=ret_at(pd.Timedelta("72h")),
               ret_7d=ret_at(pd.Timedelta("7D")))

    # 2) target vs stop
    horizon = live[live.index < t0 + HORIZON]
    result, exit_t, exit_px = None, None, None
    tgt = targets[0] if targets else None
    for t, bar in horizon.iterrows():
        adverse, favourable = (bar.low, bar.high) if d > 0 else (bar.high, bar.low)
        if stop and ((d > 0 and adverse <= stop) or (d < 0 and adverse >= stop)):
            result, exit_t, exit_px = "loss", t, float(stop)
            break
        if tgt and ((d > 0 and favourable >= tgt) or (d < 0 and favourable <= tgt)):
            result, exit_t, exit_px = "win", t, float(tgt)
            break
    seg = horizon if exit_t is None else horizon[horizon.index <= exit_t]
    mfe = d * ((seg.high.max() if d > 0 else seg.low.min()) / px - 1) * 100
    mae = d * ((seg.low.min() if d > 0 else seg.high.max()) / px - 1) * 100
    out.update(mfe_pct=round(mfe, 2), mae_pct=round(mae, 2))
    if result:
        risk = abs(px - stop) / px if stop else None
        ret = d * (exit_px / px - 1)
        out.update(status="closed", result=result, exit_time=str(exit_t), exit_price=exit_px,
                   ret_pct=round(ret * 100, 2), r_multiple=round(ret / risk, 2) if risk else None)
    elif not (stop or tgt):
        r7 = out["ret_7d"]
        out.update(status="directional" if r7 is not None else "too recent",
                   result=None if r7 is None else ("right" if r7 > 0 else "wrong"))
    else:
        done = t0 + HORIZON <= now
        last = float(horizon.close.iloc[-1])
        out.update(status="expired" if done else "open", ret_pct=round(d * (last / px - 1) * 100, 2))
    return out


def check_pattern(pt: dict, candles: pd.DataFrame, now: pd.Timestamp | None = None) -> dict:
    """Which of his lines broke first (candle close beyond it), and did the move reach his target
    (or the pattern's height) before price closed back past the other line - within 14 days."""
    from mcbot.fmt import price
    top, bottom = pt.get("top"), pt.get("bottom")
    if top is None and bottom is None:
        return {"outcome": "⚪ No levels given - can't check", "broke": None}
    t0 = pd.Timestamp(pt["said_at"])
    c = candles[(candles.index > t0) & (candles.index <= t0 + HORIZON)]
    if c.empty:
        return {"outcome": "⏳ Too recent to check", "broke": None}
    up = c.index[c.close > top] if top is not None else []
    dn = c.index[c.close < bottom] if bottom is not None else []
    first_up, first_dn = (up[0] if len(up) else None), (dn[0] if len(dn) else None)
    if first_up is None and first_dn is None:
        done = (now or pd.Timestamp.now(tz="UTC")) > t0 + HORIZON
        return {"outcome": "⚪ Didn't break either line within 14 days" if done else "⏳ No break yet", "broke": None}
    d = "up" if first_dn is None or (first_up is not None and first_up < first_dn) else "down"
    tb = first_up if d == "up" else first_dn
    level = top if d == "up" else bottom
    height = (top - bottom) if top is not None and bottom is not None else level * 0.05
    tgts = [x for x in json.loads(pt.get("targets") or "[]") if (x > level if d == "up" else x < level)]
    tgt = (min(tgts) if d == "up" else max(tgts)) if tgts else (level + height if d == "up" else level - height)
    other = bottom if d == "up" else top
    after = c[c.index > tb]
    hit = after.index[after.high >= tgt] if d == "up" else after.index[after.low <= tgt]
    back = after.index[after.close < other] if (d == "up" and other is not None) else \
        after.index[after.close > other] if other is not None else []
    called = pt.get("bias") in ("up", "down")
    right = "" if not called else (" - the way he expected" if pt["bias"] == d else " - against what he expected")
    head = f"{'🟢' if d == 'up' else '🔴'} Broke {d.upper()} on {tb:%Y-%m-%d %H:%M} (close {price(c.close.loc[tb])}){right}"
    if len(hit) and (not len(back) or hit[0] < back[0]):
        return {"outcome": f"{head}; reached {price(tgt)} on {hit[0]:%Y-%m-%d}", "broke": d}
    if len(back):
        return {"outcome": f"{head}; failed - closed back past {price(other)} on {back[0]:%Y-%m-%d}", "broke": d}
    return {"outcome": f"{head}; target {price(tgt)} not reached within 14 days", "broke": d}


def verify_patterns(store) -> int:
    pats = store.patterns_with_streams()
    src = Candles()
    for _, pt in pats.iterrows():
        try:
            o = check_pattern(pt.to_dict(), src.get(pt.asset, pd.Timestamp(pt.said_at)))
        except Exception as e:
            o = {"outcome": f"error: {e}"[:200], "broke": None}
        store.update_pattern(int(pt.id), outcome=o["outcome"], broke=o["broke"])
    return len(pats)


def verify_all(store, only_unchecked: bool = False) -> pd.DataFrame:
    calls = store.calls_with_streams()
    if only_unchecked:
        calls = calls[calls.status.isna() | calls.status.isin(["open", "waiting for entry", "too recent"])]
    src = Candles()
    rows = []
    for _, c in calls.iterrows():
        try:
            o = check(c.to_dict(), src.get(c.asset, pd.Timestamp(c.said_at)))
        except Exception as e:
            o = {"call_id": int(c.id), "status": f"error: {e}"[:200], "checked": str(pd.Timestamp.now(tz="UTC"))}
        store.save_outcome(o)
        rows.append(o)
    verify_patterns(store)
    return pd.DataFrame(rows)
