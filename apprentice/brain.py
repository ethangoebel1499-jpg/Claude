"""The apprentice's brain: test what it was taught, read a chart, decide, and learn from missed trades.

How it scores a trade idea (the same everywhere, so numbers compare): enter at the next candle's open,
stop `stop_atr` ATRs away, target `rr` times the risk. +rr R if the target is hit first, -1 R if the
stop is hit first (or both in one candle), otherwise the move after `horizon` candles in R.
"Edge" = a setup's average R minus the average R of entering on *every* candle (what random entries
would have made), so a setup only counts as good if it beats random on that chart.

Missed trades are found in hindsight: swing lows (highs) where a long (short) would have hit a 3R
target before its stop, and none of the active setups fired. `mine()` then looks at what was true on
the chart at those moments more often than usual, builds candidate setups from those conditions,
and tests them on the later part of history they weren't found on (out of sample).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from mcbot import builder
from mcbot.patterns import Detector, atr as atr_arr, situation

HORIZON = 48
MIN_TRADES = 10
COOLDOWN = 3          # signals closer than this many candles to the previous one are the same trade


# ---- scoring -----------------------------------------------------------------------------------
_OUT: dict = {}


def outcomes(df: pd.DataFrame, direction: str, stop_atr: float = 1.5, rr: float = 2.0,
             horizon: int = HORIZON) -> np.ndarray:
    """R result of entering on every candle (NaN where there isn't enough future)."""
    key = (id(df), len(df), df.index[-1], direction, float(stop_atr), float(rr), horizon)
    if key in _OUT:
        return _OUT[key]
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    a = atr_arr(df)
    n = len(df)
    sgn = 1 if direction == "long" else -1
    entry = np.full(n, np.nan)
    entry[:-1] = o[1:]
    risk = stop_atr * a
    stop = entry - sgn * risk
    tgt = entry + sgn * rr * risk
    res = np.full(n, np.nan)
    open_ = np.isfinite(entry) & (risk > 0)
    for k in range(1, horizon + 1):
        idx = np.arange(n) + k
        ok = idx < n
        hk = np.where(ok, h[np.minimum(idx, n - 1)], np.nan)
        lk = np.where(ok, l[np.minimum(idx, n - 1)], np.nan)
        hit_stop = open_ & ok & ((lk <= stop) if sgn > 0 else (hk >= stop))
        hit_tgt = open_ & ok & ~hit_stop & ((hk >= tgt) if sgn > 0 else (lk <= tgt))
        res[hit_stop] = -1.0
        res[hit_tgt] = rr
        open_ &= ~(hit_stop | hit_tgt)
        if k == horizon:
            last = np.where(ok, c[np.minimum(idx, n - 1)], np.nan)
            res[open_ & ok] = (sgn * (last - entry) / risk)[open_ & ok]
    res = np.clip(res, -1.0, rr)
    if len(_OUT) > 64:
        _OUT.clear()
    _OUT[key] = res
    return res


def dedupe(mask: np.ndarray, gap: int = COOLDOWN) -> np.ndarray:
    """Keep the first signal of each cluster."""
    out = np.zeros_like(mask, dtype=bool)
    last = -10 ** 9
    for i in np.flatnonzero(mask):
        if i - last > gap:
            out[i] = True
        last = i
    return out


def setup_mask(df: pd.DataFrame, setup: dict) -> np.ndarray:
    return builder._all(df, setup["rules"]).to_numpy(bool)


def stats_of(res: np.ndarray, mask: np.ndarray, base: float) -> dict:
    r = res[mask & np.isfinite(res)]
    n = len(r)
    if not n:
        return {"trades": 0, "win_rate": None, "avg_r": None, "edge": None, "base_r": round(base, 3), "sum_r": 0.0}
    avg = float(r.mean())
    sd = float(r.std(ddof=1)) if n > 1 else 1.0
    return {"trades": n, "win_rate": round(float((r > 0).mean()) * 100, 1), "avg_r": round(avg, 3),
            "edge": round(avg - base, 3), "base_r": round(base, 3), "sum_r": round(float(r.sum()), 2),
            "t": round((avg - base) / (sd / math.sqrt(n)), 2) if sd > 0 and n > 1 else 0.0}


def evaluate(df: pd.DataFrame, setup: dict, start: int = 0, end: int | None = None) -> dict:
    res = outcomes(df, setup["direction"], setup.get("stop_atr") or 1.5, setup.get("rr") or 2.0)
    end = len(df) if end is None else end
    window = np.zeros(len(df), dtype=bool)
    window[start:end] = True
    mask = dedupe(setup_mask(df, setup)) & window
    base = float(np.nanmean(res[window])) if np.isfinite(res[window]).any() else 0.0
    return stats_of(res, mask, base)


def combine(per_chart: dict[str, dict]) -> dict:
    """Stats over several charts: trade-weighted average R and edge."""
    rows = [s for s in per_chart.values() if s.get("trades")]
    n = sum(s["trades"] for s in rows)
    if not n:
        return {"trades": 0, "avg_r": None, "edge": None, "win_rate": None, "charts": len(per_chart)}
    w = lambda k: sum(s[k] * s["trades"] for s in rows) / n
    return {"trades": n, "avg_r": round(w("avg_r"), 3), "edge": round(w("edge"), 3),
            "win_rate": round(w("win_rate"), 1), "charts": len(per_chart),
            "charts_positive": sum(1 for s in rows if s["edge"] > 0)}


def weight(total: dict, votes: tuple[int, int] = (0, 0)) -> float:
    """How much the apprentice trusts a setup: its edge, shrunk toward 0 when it has few trades,
    nudged by your 👍 / 👎."""
    n, edge = total.get("trades") or 0, total.get("edge") or 0.0
    good, bad = votes
    nudge = max(-0.6, min(0.6, 0.15 * good - 0.2 * bad))
    return round(edge * n / (n + 20) + nudge, 3)


def status_for(setup: dict, total: dict, w: float) -> str:
    if setup["status"] in ("pinned", "off"):          # your choice wins
        return setup["status"]
    if not setup.get("rules"):
        return "needs rules"                           # nothing it can check yet - you pick the rules
    n = total.get("trades") or 0
    if n < MIN_TRADES:
        return "candidate"
    return "active" if w > 0.02 and (total.get("edge") or 0) > 0 else "retired"


def learn(store, charts: dict[str, pd.DataFrame], log=print) -> list[dict]:
    """Test every setup on every chart, update its track record, weight and status."""
    out = []
    for s in store.setups():
        if s["status"] == "off":
            continue
        per = {}
        for name, df in charts.items():
            try:
                per[name] = evaluate(df, s)
            except Exception as e:                       # a rule that can't run on this chart
                log(f"  {s['name']} on {name}: {e}")
        total = combine(per)
        w = weight(total, store.votes(s["id"]))
        st = status_for(s, total, w)
        store.update_setup(s["id"], stats={"total": total, "charts": per}, weight=w, status=st)
        out.append({"id": s["id"], "name": s["name"], "status": st, "weight": w, **total})
        log(f"  {s['name']} ({s['direction']}): {total['trades']} trades, edge "
            f"{total['edge'] if total['edge'] is not None else '-'}R -> {st}")
    return out


# ---- reading a chart --------------------------------------------------------------------------
def levels(df: pd.DataFrame, lookback: int = 300, max_levels: int = 6) -> list[dict]:
    """Support / resistance zones: confirmed swing points that line up (within half an ATR)."""
    det = Detector(df)
    end = len(df) - 1
    a = float(det.atr[end]) or float(df.close.iloc[-1]) * 0.01
    pts = [(det.h[i], i) for i in det.hi if end - lookback <= i <= end - det.k] + \
          [(det.l[i], i) for i in det.lw if end - lookback <= i <= end - det.k]
    pts.sort()
    zones: list[list] = []
    for p, i in pts:
        if zones and p - zones[-1][-1][0] <= 0.5 * a:
            zones[-1].append((p, i))
        else:
            zones.append([(p, i)])
    c = float(df.close.iloc[-1])
    out = [{"price": float(np.mean([p for p, _ in z])), "touches": len(z), "last": max(i for _, i in z)}
           for z in zones if len(z) >= 2]
    for z in out:
        z["kind"] = "support" if z["price"] < c else "resistance"
    out.sort(key=lambda z: (-z["touches"], abs(z["price"] - c)))
    return sorted(out[:max_levels], key=lambda z: z["price"])


def true_features(df: pd.DataFrame, i: int, frame: pd.DataFrame | None = None) -> list[str]:
    frame = feature_frame(df) if frame is None else frame
    row = frame.iloc[i]
    return [k for k, v in row.items() if v]


_FEAT: dict = {}


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Every rule at its default numbers, on every candle (True/False)."""
    key = (id(df), len(df), df.index[-1])
    if key not in _FEAT:
        _FEAT.clear()
        cols = {}
        for k in builder.RULES:
            try:
                cols[k] = builder.rule_series(df, {"rule": k}).to_numpy(bool)
            except Exception:
                pass
        _FEAT[key] = pd.DataFrame(cols, index=df.index)
    return _FEAT[key]


def swing_points(df: pd.DataFrame, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Hindsight swing lows / highs (uses the future: only for finding missed trades, never for trading)."""
    w = 2 * k + 1
    lo_min = df.low.rolling(w, center=True).min().to_numpy(float)
    hi_max = df.high.rolling(w, center=True).max().to_numpy(float)
    is_lo = df.low.to_numpy(float) == lo_min
    is_hi = df.high.to_numpy(float) == hi_max
    return is_lo, is_hi


def good_entries(df: pd.DataFrame, direction: str, rr: float = 3.0, stop_atr: float = 1.5) -> np.ndarray:
    """Candles a trader would have loved to enter on, in hindsight. The signal candle is the swing
    candle itself (entry at the next open), so what was true *on* it is what to look for."""
    res = outcomes(df, direction, stop_atr, rr)
    lo, hi = swing_points(df)
    swing = lo if direction == "long" else hi
    return swing & (res >= rr - 1e-9)


def missed_trades(df: pd.DataFrame, setups: list[dict], top: int = 12, rr: float = 3.0,
                  rejected: set | None = None) -> list[dict]:
    """The best hindsight trades none of the active setups caught (within 3 candles)."""
    frame = feature_frame(df)
    a = atr_arr(df)
    out = []
    for direction in ("long", "short"):
        good = good_entries(df, direction, rr)
        caught = np.zeros(len(df), bool)
        for s in setups:
            if s["direction"] == direction and s["status"] in ("active", "pinned"):
                m = setup_mask(df, s)
                for sh in range(-3, 4):
                    caught |= pd.Series(m).shift(sh, fill_value=False).to_numpy(bool)
        c = df.close.to_numpy(float)
        h, l = df.high.to_numpy(float), df.low.to_numpy(float)
        for i in np.flatnonzero(good & ~caught):
            t = df.index[i]
            if rejected and (str(t), direction) in rejected:
                continue
            j = min(i + HORIZON, len(df) - 1)
            move = (h[i + 1:j + 1].max() - c[i]) if direction == "long" else (c[i] - l[i + 1:j + 1].min())
            out.append({"i": int(i), "time": t, "direction": direction, "price": float(c[i]),
                        "move_atr": round(float(move / a[i]), 1) if a[i] > 0 else 0.0,
                        "move_pct": round(float(move / c[i] * 100), 1),
                        "features": [k for k, v in frame.iloc[i].items() if v]})
    out.sort(key=lambda m: -m["move_atr"])
    picked: list[dict] = []
    for m in out:
        if all(abs(m["i"] - p["i"]) > 5 for p in picked):
            picked.append(m)
        if len(picked) >= top:
            break
    return sorted(picked, key=lambda m: m["i"])


def mine(charts: dict[str, pd.DataFrame], direction: str, confirmed: dict | None = None,
         rejected: dict | None = None, train: float = 0.7, n_best: int = 3, max_rules: int = 3) -> list[dict]:
    """Learn from good trades: which conditions were true at good entries much more often than on an
    average candle? Combine the strongest into setups, keep those that worked in the first `train` of
    history *and* on the rest (which they never saw). confirmed / rejected: {chart: set of times}
    from your 👍 / 👎 on missed trades."""
    lifts: dict[str, list] = {}
    for name, df in charts.items():
        frame = feature_frame(df)
        cut = int(len(df) * train)
        good = good_entries(df, direction)
        idx = df.index
        if confirmed and confirmed.get(name):
            good |= idx.astype(str).isin(list(confirmed[name]))
        if rejected and rejected.get(name):
            good &= ~idx.astype(str).isin(list(rejected[name]))
        good[cut:] = False
        if good.sum() < 5:
            continue
        f = frame.iloc[:cut]
        g = frame[good]
        for k in frame.columns:
            base, hit = f[k].mean(), g[k].mean()
            if base > 0 and 0.03 <= base <= 0.6:
                lifts.setdefault(k, []).append((hit / base, hit, len(g)))
    scored = []
    for k, v in lifts.items():
        n = sum(x[2] for x in v)
        lift = sum(x[0] * x[2] for x in v) / n
        support = sum(x[1] * x[2] for x in v) / n
        if support >= 0.12 and lift > 1.15:
            scored.append((lift, support, k))
    scored.sort(reverse=True)
    pool = [k for _, _, k in scored[:8]]
    tried = []
    for r in range(1, max_rules + 1):
        for combo in itertools.combinations(pool, r):
            setup = {"direction": direction, "rules": [{"rule": k} for k in combo], "stop_atr": 1.5, "rr": 2.0}
            per_tr, per_te = {}, {}
            for name, df in charts.items():
                cut = int(len(df) * train)
                per_tr[name] = evaluate(df, setup, 0, cut)
                per_te[name] = evaluate(df, setup, cut)
            tr, te = combine(per_tr), combine(per_te)
            if (tr["trades"] or 0) >= MIN_TRADES and (tr["edge"] or 0) > 0:
                tried.append((tr["edge"] * tr["trades"] / (tr["trades"] + 20), setup, tr, te))
    tried.sort(key=lambda x: -x[0])
    out, used = [], []
    for _, setup, tr, te in tried:
        keys = {r["rule"] for r in setup["rules"]}
        if any(keys <= u or u <= keys for u in used):     # skip near-duplicates of a better one
            continue
        used.append(keys)
        name = " + ".join(builder.describe(r).split(" (")[0] for r in setup["rules"])
        out.append({**setup, "name": f"{'Long' if direction == 'long' else 'Short'}: {name}"[:80],
                    "train": tr, "test": te, "held_up": bool(te["trades"] and (te["edge"] or 0) > 0)})
        if len(out) >= n_best:
            break
    return out


def decide(df: pd.DataFrame, setups: list[dict], concepts: list[dict] | None = None) -> dict:
    """The apprentice's call on the last closed candle: TRADE LONG / TRADE SHORT / WAIT, and why."""
    i = len(df) - 1
    c = float(df.close.iloc[i])
    a = float(atr_arr(df)[i])
    live = [s for s in setups if s["status"] in ("active", "pinned")]
    checks = []
    for s in live:
        ok = [bool(builder.rule_series(df, r).iloc[i]) for r in s["rules"]]
        checks.append({"setup": s, "ok": ok, "met": sum(ok), "of": len(ok), "fired": all(ok)})
    score = {"long": 0.0, "short": 0.0}
    for ch in checks:
        if ch["fired"]:
            score[ch["setup"]["direction"]] += max(ch["setup"]["weight"], 0.01)
    side = max(score, key=score.get)
    net = score[side] - score["short" if side == "long" else "long"]
    sit = None
    try:
        sit = situation(df)
    except Exception:
        pass
    reasons = []
    if net > 0:
        best = max((ch for ch in checks if ch["fired"] and ch["setup"]["direction"] == side),
                   key=lambda ch: ch["setup"]["weight"])
        s = best["setup"]
        sgn = 1 if side == "long" else -1
        risk = (s.get("stop_atr") or 1.5) * a
        action, conf = f"TRADE {side.upper()}", round(1 / (1 + math.exp(-8 * net)) * 100)
        entry, stop, target = c, c - sgn * risk, c + sgn * (s.get("rr") or 2.0) * risk
        for ch in checks:
            if ch["fired"]:
                st = ch["setup"]["stats"].get("total", {})
                reasons.append(f"✅ {ch['setup']['name']} ({ch['setup']['direction']}) fired - it has made "
                               f"{st.get('avg_r', 0) or 0:+.2f}R per trade over {st.get('trades', 0)} past trades "
                               f"(edge {st.get('edge', 0) or 0:+.2f}R vs random)")
    else:
        action, conf, entry, stop, target = "WAIT", 0, None, None, None
        near = sorted(checks, key=lambda ch: -ch["met"] / max(ch["of"], 1))[:3]
        for ch in near:
            missing = [builder.describe(r) for r, o in zip(ch["setup"]["rules"], ch["ok"]) if not o]
            if ch["met"]:
                reasons.append(f"👀 {ch['setup']['name']}: {ch['met']}/{ch['of']} - still needs "
                               + "; ".join(missing))
        if not live:
            reasons.append("I don't have any tested setups yet - teach me some lessons, then press Learn.")
    if sit:
        reasons.append(f"📐 {sit['pattern'].name}: {sit['text']}")
    if concepts:
        words = set(re_words(" ".join(reasons)))
        rel = [cp for cp in concepts if len(words & set(re_words(cp["text"]))) >= 2][:4]
        for cp in rel:
            reasons.append(f"📚 I was taught: {cp['text']}")
    return {"action": action, "confidence": conf, "entry": entry, "stop": stop, "target": target,
            "reasons": reasons, "checks": checks, "situation": sit, "bar_time": df.index[i], "close": c}


STOP_WORDS = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "is", "it", "for", "with", "this", "that",
              "be", "when", "if", "you", "i", "my", "price", "candle", "candles", "has", "have", "still", "needs",
              "per", "over", "past", "trades", "trade", "vs", "random", "edge", "made"}


def re_words(text: str) -> list[str]:
    import re
    return [w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOP_WORDS]


def history_signals(df: pd.DataFrame, setups: list[dict], last: int = 400) -> list[dict]:
    """Where each active setup fired in the last `last` candles, and how it went."""
    out = []
    start = max(0, len(df) - last)
    for s in setups:
        if s["status"] not in ("active", "pinned"):
            continue
        res = outcomes(df, s["direction"], s.get("stop_atr") or 1.5, s.get("rr") or 2.0)
        m = dedupe(setup_mask(df, s))
        for i in np.flatnonzero(m[start:]) + start:
            out.append({"i": int(i), "time": df.index[i], "direction": s["direction"], "setup": s["name"],
                        "r": None if not np.isfinite(res[i]) else round(float(res[i]), 2)})
    return sorted(out, key=lambda x: x["i"])


def grade_journal(store, df: pd.DataFrame, symbol: str, timeframe: str) -> int:
    """Fill in how earlier TRADE decisions on this chart turned out."""
    rows = store.db.execute("SELECT * FROM journal WHERE symbol = ? AND timeframe = ? AND outcome IS NULL "
                            "AND action LIKE 'TRADE%'", (symbol, timeframe)).fetchall()
    n = 0
    for r in rows:
        t = pd.Timestamp(r["bar_time"])
        if t not in df.index:
            continue
        i = df.index.get_loc(t)
        after = df.iloc[i + 1:i + 1 + HORIZON]
        if after.empty:
            continue
        long = r["action"].endswith("LONG")
        risk = abs(r["entry"] - r["stop"]) or 1e-12
        result = None
        for _, b in after.iterrows():
            if (long and b.low <= r["stop"]) or (not long and b.high >= r["stop"]):
                result = ("stopped", -1.0)
                break
            if (long and b.high >= r["target"]) or (not long and b.low <= r["target"]):
                result = ("target", abs(r["target"] - r["entry"]) / risk)
                break
        if result is None and len(after) >= HORIZON:
            last = float(after.close.iloc[-1])
            result = ("timed out", ((last - r["entry"]) if long else (r["entry"] - last)) / risk)
        if result:
            store.db.execute("UPDATE journal SET outcome = ?, r = ? WHERE id = ?",
                             (result[0], round(result[1], 2), r["id"]))
            n += 1
    store.commit()
    return n
