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


def _fingerprint(df: pd.DataFrame) -> tuple:
    """Cache key for a chart: its size, time span and prices (not its memory address, which Python reuses)."""
    c = df["close"].to_numpy(float)
    return (len(df), df.index[0], df.index[-1], float(np.nansum(c)), float(c[-1]) if len(c) else 0.0,
            float(np.nansum(df["high"].to_numpy(float))))


def outcomes(df: pd.DataFrame, direction: str, stop_atr: float = 1.5, rr: float = 2.0,
             horizon: int = HORIZON) -> np.ndarray:
    """R result of entering on every candle (NaN where there isn't enough future)."""
    key = (_fingerprint(df), direction, float(stop_atr), float(rr), horizon)
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
        stats = {"total": total, "charts": per}
        if (total.get("trades") or 0) >= MIN_TRADES:
            try:
                stats["after"] = combine_after({name: after_entry(df, s) for name, df in charts.items()})
                stats["hints"] = hints(s, stats["after"])
                stats["exit_plan"] = {tf: exit_plan(group, s) for tf, group in by_timeframe(charts).items()}
                if st in ("active", "pinned"):
                    stats["tune"] = tune(charts, s)
            except Exception as e:
                log(f"  {s['name']}: after-entry stats failed: {e}")
        store.update_setup(s["id"], stats=stats, weight=w, status=st)
        out.append({"id": s["id"], "name": s["name"], "status": st, "weight": w, **total})
        log(f"  {s['name']} ({s['direction']}): {total['trades']} trades, edge "
            f"{total['edge'] if total['edge'] is not None else '-'}R -> {st}")
    try:
        deploy_check(store, charts, log)
    except Exception as e:
        log(f"  exit-plan check failed ({e}) - keeping the 3% stop everywhere")
        for s in store.setups():
            plans = (s.get("stats") or {}).get("exit_plan") or {}
            for p in plans.values():
                p.update(stop="fixed", reversal=False, deployed=False)
            if plans:
                store.update_setup(s["id"], stats=s["stats"])
    return out


def timeframe_of(df: pd.DataFrame) -> str:
    """'1h', '4h', '1D', ... from the candle spacing."""
    if len(df) < 3:
        return "?"
    mins = int(round(pd.Series(df.index).diff().median().total_seconds() / 60))
    return f"{mins // 1440}D" if mins >= 1440 and mins % 1440 == 0 else f"{mins // 60}h" if mins % 60 == 0 \
        else f"{mins}m"


def by_timeframe(charts: dict[str, pd.DataFrame]) -> dict[str, dict[str, pd.DataFrame]]:
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for name, df in charts.items():
        out.setdefault(timeframe_of(df), {})[name] = df
    return out


def deploy_check(store, charts: dict[str, pd.DataFrame], log=print, split: float = 0.7) -> dict[str, bool]:
    """The last gate before the trader uses what it learned about exits: trade ALL the live setups together,
    with their learned stops / early exits and with the plain 3% stop, on the last 30% of every chart of a
    timeframe. The learned plans for that timeframe are deployed only if they made more there; otherwise
    every setup goes back to the 3% stop and no early exit on that timeframe (what it learned stays visible)."""
    from mcbot.backtest import run_backtest
    from . import strategy
    setups = [s for s in store.setups(("active", "pinned")) if s["weight"] > 0 or s["status"] == "pinned"]
    verdict = {}
    for tf, group in by_timeframe(charts).items():
        learned = [{**s, "stats": {"exit_plan": {tf: {**p, "stop": p.get("learned_stop", p["stop"]),
                                                      "reversal": p.get("learned_reversal", p["reversal"])}
                                                 for p in [(s["stats"].get("exit_plan") or {}).get(tf, {})] if p}}}
                   for s in setups]
        plain = [{**s, "stats": {}} for s in setups]
        if not any((s["stats"]["exit_plan"].get(tf) or {}).get("stop") not in (None, "fixed")
                   or (s["stats"]["exit_plan"].get(tf) or {}).get("reversal") for s in learned):
            verdict[tf] = False
            continue
        ret = {"learned": 0.0, "plain": 0.0}
        for name, df in group.items():
            cut = int(len(df) * split)
            for k, ss in (("learned", learned), ("plain", plain)):
                sig = strategy.signals(df, ss, mode="learned", avoid={}, paused=False)
                st = run_backtest(df.iloc[cut:], sig.iloc[cut:], allow_short=True, stop_loss=strategy.STOP,
                                  take_profits=strategy.TAKE_PROFITS).stats
                ret[k] += st["total_return_pct"]
        verdict[tf] = bool(ret["learned"] > ret["plain"])
        log(f"  exit plans on {tf}: all setups together on unseen prices {ret['learned']:+.1f}% with them vs "
            f"{ret['plain']:+.1f}% with the 3% stop -> {'deployed' if verdict[tf] else 'NOT deployed'}")
    for s in store.setups():
        plans = (s.get("stats") or {}).get("exit_plan") or {}
        for tf, p in plans.items():
            p.setdefault("learned_stop", p["stop"])
            p.setdefault("learned_reversal", p["reversal"])
            ok = verdict.get(tf, False)
            p["deployed"] = ok
            p["stop"], p["reversal"] = (p["learned_stop"], p["learned_reversal"]) if ok else ("fixed", False)
        if plans:
            store.update_setup(s["id"], stats=s["stats"])
    safety_check(store, charts, log, split)
    return verdict


def safety_check(store, charts: dict[str, pd.DataFrame], log=print, split: float = 0.7) -> dict:
    """The two trade-safety rules (no entry against an opposite dot / divergence; stop to the entry at +1.5%)
    are on by default. Per timeframe, trade all the live setups together on the last 30% of every chart with
    both rules, and with each one switched off: a rule stays on unless the strategy did better without it
    on those unseen prices."""
    from mcbot.backtest import run_backtest
    from . import strategy
    setups = [s for s in store.setups(("active", "pinned")) if s["weight"] > 0 or s["status"] == "pinned"]
    out = {}
    for tf, group in by_timeframe(charts).items():
        def ret(safety):
            total = 0.0
            for df in group.values():
                cut = int(len(df) * split)
                sig = strategy.signals(df, setups, mode="learned", safety=safety, avoid={}, paused=False)
                total += run_backtest(df.iloc[cut:], sig.iloc[cut:], allow_short=True, stop_loss=strategy.STOP,
                                      take_profits=strategy.TAKE_PROFITS).stats["total_return_pct"]
            return round(total, 2)
        both = ret({"no_opposite": True, "breakeven": True})
        res = {}
        for rule in strategy.SAFETY:
            without = ret({k: k != rule for k in strategy.SAFETY})
            res[rule] = {"on": bool(both >= without), "with": both, "without": without}
            log(f"  safety rule '{rule}' on {tf}: {both:+.1f}% with it vs {without:+.1f}% without on unseen prices "
                f"-> {'on' if res[rule]['on'] else 'OFF'}")
        out[tf] = res
    if out:
        store.meta_set("safety", {**(store.meta_get("safety") or {}), **out})
    avoid_check(store, charts, log, split)
    timeframe_check(store, charts, log, split)
    return out


def avoid_check(store, charts: dict[str, pd.DataFrame], log=print, split: float = 0.7, max_rules: int = 3,
                min_share: float = 0.1) -> dict:
    """Learning from its losing trades: take every entry the live setups made (per timeframe and direction),
    and look for a condition that was true much more often at the losers - e.g. 'after 3 green candles in a row'
    for longs. Found on the first 70% of every chart; kept as an AVOID rule (no entry while it's true) only if
    skipping those trades also made more on the last 30% it never saw. Losing trades are noisy, so most
    candidates don't survive that test - which is the point."""
    from . import strategy
    setups = [s for s in store.setups(("active", "pinned")) if s["weight"] > 0 or s["status"] == "pinned"]
    out: dict = {}
    for tf, group in by_timeframe(charts).items():
        out[tf] = {}
        for d in ("long", "short"):
            rows = {"train": [], "test": []}
            for df in group.values():
                sig = strategy.signals(df, setups, mode="learned", safety=strategy.safety_rules(tf, store),
                                       paused=False, avoid={})
                entries = np.flatnonzero(dedupe(sig[f"{d}_entry"].to_numpy(bool)))
                res = outcomes(df, d)
                ff = feature_frame(df)
                cut = int(len(df) * split)
                for i in entries:
                    if np.isfinite(res[i]):
                        rows["train" if i < cut else "test"].append((ff.iloc[i], float(res[i])))
            tr, te = rows["train"], rows["test"]
            picked = []
            if len(tr) >= 30 and len(te) >= 10:
                r_tr = np.array([r for _, r in tr]); r_te = np.array([r for _, r in te])
                cands = []
                for f in tr[0][0].index:
                    m = np.array([bool(row[f]) for row, _ in tr])
                    n = int(m.sum())
                    if n < max(10, min_share * len(tr)) or n > 0.6 * len(tr):
                        continue
                    if r_tr[m].mean() < r_tr.mean() - 0.15 and (r_tr[m] < 0).mean() > (r_tr < 0).mean() + 0.08:
                        cands.append((-float(r_tr[m].sum()), f))           # R saved on the train part by skipping
                cands.sort(reverse=True)
                keep_tr = np.ones(len(tr), bool); keep_te = np.ones(len(te), bool)
                for saved, f in cands:
                    if len(picked) >= max_rules or saved <= 0:
                        break
                    m_te = np.array([bool(row[f]) for row, _ in te]) & keep_te
                    m_tr = np.array([bool(row[f]) for row, _ in tr]) & keep_tr
                    if m_te.sum() < 3 or r_te[m_te].sum() >= 0:
                        continue                                          # didn't help on unseen prices: not real
                    keep_tr &= ~m_tr; keep_te &= ~m_te
                    picked.append({"rule": f, "text": builder.describe({"rule": f}),
                                   "train": {"skipped": int(m_tr.sum()), "saved_r": round(-float(r_tr[m_tr].sum()), 1)},
                                   "test": {"skipped": int(m_te.sum()), "saved_r": round(-float(r_te[m_te].sum()), 1)}})
            out[tf][d] = picked
        # each rule helped on its own; together they must also help the whole strategy on unseen prices
        if any(out[tf].values()):
            from mcbot.backtest import run_backtest

            def unseen(av):
                total = 0.0
                for df in group.values():
                    cut = int(len(df) * split)
                    sig = strategy.signals(df, setups, mode="learned", safety=strategy.safety_rules(tf, store),
                                           paused=False, avoid=av)
                    total += run_backtest(df.iloc[cut:], sig.iloc[cut:], allow_short=True, stop_loss=strategy.STOP,
                                          take_profits=strategy.TAKE_PROFITS).stats["total_return_pct"]
                return round(total, 2)
            with_, without = unseen(out[tf]), unseen({})
            if with_ <= without:
                log(f"  avoid rules on {tf}: whole strategy {with_:+.1f}% with them vs {without:+.1f}% without on unseen "
                    "prices -> not used")
                out[tf] = {"long": [], "short": [], "rejected": {"with": with_, "without": without}}
            else:
                out[tf]["checked"] = {"with": with_, "without": without}
        for d in ("long", "short"):
            for p in out[tf].get(d, []):
                log(f"  avoid on {tf} {d}s: {p['text']} - skipping those saved {p['train']['saved_r']:+.1f}R before and "
                    f"{p['test']['saved_r']:+.1f}R on unseen prices")
    if out:
        store.meta_set("avoid", {**(store.meta_get("avoid") or {}), **out})
    return out


def timeframe_check(store, charts: dict[str, pd.DataFrame], log=print, split: float = 0.7) -> dict:
    """Last gate: with everything it just learned (setups, stops, safety and avoid rules), trade all the live setups
    together on every chart of a timeframe. Unless that made money both on the first 70% and on the last 30% it never
    saw, the timeframe is paused - no new trades there until a later Learn shows it making money on both again."""
    from mcbot.backtest import run_backtest
    from . import strategy
    setups = [s for s in store.setups(("active", "pinned")) if s["weight"] > 0 or s["status"] == "pinned"]
    out = {}
    for tf, group in by_timeframe(charts).items():
        total, before, trades = 0.0, 0.0, 0
        for df in group.values():
            cut = int(len(df) * split)
            sig = strategy.signals(df, setups, mode="learned", safety=strategy.safety_rules(tf, store), paused=False,
                                   avoid=strategy.avoid_rules(tf, store))
            kw = dict(allow_short=True, stop_loss=strategy.STOP, take_profits=strategy.TAKE_PROFITS)
            st = run_backtest(df.iloc[cut:], sig.iloc[cut:], **kw).stats
            before += run_backtest(df.iloc[:cut], sig.iloc[:cut], **kw).stats["total_return_pct"]
            total += st["total_return_pct"]
            trades += st["trades"]
        ok = bool(total > 0 and before > 0)            # made money before AND on prices it never saw
        out[tf] = {"trade": ok, "unseen_pct": round(total, 2), "before_pct": round(before, 2), "trades": trades,
                   "charts": len(group)}
        log(f"  {tf}: all setups together {before:+.1f}% on the first 70%, {total:+.1f}% on unseen prices "
            f"({trades} trades on {len(group)} charts) -> " + ("trading" if ok else "PAUSED - no new trades on this timeframe"))
    if out:
        store.meta_set("timeframes", {**(store.meta_get("timeframes") or {}), **out})
    return out


# ---- after the entry: what trades did, and can the trigger be tuned? ----------------------------
def reversal_signals(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """A *confirmed* reversal, as the lessons describe it: a divergence confirmed by the dot printing on a
    closed candle (bearish divergence + red dot / bullish + green dot), or the structure breaking with money
    flow behind it (close below the 20-candle low with money flow red / above the high with it green).
    Returns (against a long, against a short)."""
    r = lambda rules: builder._all(df, rules).to_numpy(bool)
    bear = (r([{"rule": "rsi_bear_div", "a": 10, "b": 60}]) | r([{"rule": "wt_bear_div", "a": 10, "b": 60}])) \
        & r([{"rule": "wt_red_dot", "a": -1000}])
    bull = (r([{"rule": "rsi_bull_div", "a": 10, "b": 60}]) | r([{"rule": "wt_bull_div", "a": 10, "b": 60}])) \
        & r([{"rule": "wt_green_dot", "a": 1000}])
    down = r([{"rule": "breakdown_low", "a": 20}]) & r([{"rule": "mf_negative"}])
    up = r([{"rule": "breakout_high", "a": 20}]) & r([{"rule": "mf_positive"}])
    return bear | down, bull | up


def after_entry(df: pd.DataFrame, setup: dict, horizon: int = HORIZON) -> dict:
    """Follow every past trade of a setup candle by candle (same entry / stop / target as its test):
    how far it went for you (MFE) and against you (MAE) in R, how long it took, how often it got to 1R,
    and what closing a losing trade on a confirmed reversal would have changed."""
    d = setup["direction"]
    sgn = 1 if d == "long" else -1
    stop_atr, rr = float(setup.get("stop_atr") or 1.5), float(setup.get("rr") or 2.0)
    res = outcomes(df, d, stop_atr, rr, horizon)
    mask = dedupe(setup_mask(df, setup)) & np.isfinite(res)
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    a = atr_arr(df)
    against = reversal_signals(df)[0 if d == "long" else 1]
    n = len(df)
    rows = []
    for i in np.flatnonzero(mask):
        if i + 1 >= n or not a[i] > 0:
            continue
        entry, risk = o[i + 1], stop_atr * a[i]
        mfe = mae = 0.0
        bars, rev_r, win_bars = horizon, None, None
        for k in range(1, horizon + 1):
            j = i + k
            if j >= n:
                break
            fav = sgn * ((h[j] if sgn > 0 else l[j]) - entry) / risk
            adv = sgn * ((l[j] if sgn > 0 else h[j]) - entry) / risk
            mae = min(mae, adv)
            if adv <= -1:                                   # stopped
                bars = k
                break
            mfe = max(mfe, fav)
            if fav >= rr:
                bars = win_bars = k
                break
            # confirmed reversal on this closed candle while the trade is losing: out at the next open
            if rev_r is None and against[j] and j + 1 < n and sgn * (c[j] - entry) < 0:
                rev_r = max(-1.0, sgn * (o[j + 1] - entry) / risk)
        rows.append((mfe, mae, bars, win_bars, res[i], rev_r))
    if not rows:
        return {"trades": 0}
    mfe, mae, bars = (np.array([r[k] for r in rows], float) for k in range(3))
    wins = [r for r in rows if r[4] > 0]
    rev = [(r[4], r[5]) for r in rows if r[5] is not None]
    return {
        "trades": len(rows),
        "mfe_r": round(float(np.median(mfe)), 2), "mae_r": round(float(np.median(mae)), 2),
        "reached_1r_pct": round(float((mfe >= 1).mean()) * 100, 1),
        "reached_target_pct": round(float((mfe >= rr).mean()) * 100, 1),
        "bars_held": round(float(np.median(bars)), 1),
        "bars_to_win": round(float(np.median([r[3] for r in wins if r[3]])), 1) if any(r[3] for r in wins) else None,
        "winners_heat_r": round(float(np.median([r[1] for r in wins])), 2) if wins else None,
        "losers_best_r": round(float(np.median([r[0] for r in rows if r[4] <= 0])), 2)
        if any(r[4] <= 0 for r in rows) else None,
        "reversal_exits": len(rev),
        "reversal_saved_r": round(float(sum(x - y for y, x in rev)), 2),   # + = closing early was better
    }


def combine_after(per: dict[str, dict]) -> dict:
    rows = [s for s in per.values() if s.get("trades")]
    n = sum(s["trades"] for s in rows)
    if not n:
        return {"trades": 0}
    out = {"trades": n}
    for k in ("mfe_r", "mae_r", "reached_1r_pct", "reached_target_pct", "bars_held", "bars_to_win",
              "winners_heat_r", "losers_best_r"):
        vals = [(s[k], s["trades"]) for s in rows if s.get(k) is not None]
        if vals:
            out[k] = round(sum(v * w for v, w in vals) / sum(w for _, w in vals), 2)
    out["reversal_exits"] = sum(s.get("reversal_exits", 0) for s in rows)
    out["reversal_saved_r"] = round(sum(s.get("reversal_saved_r", 0) for s in rows), 2)
    return out


def hints(setup: dict, after: dict) -> list[str]:
    """Plain-words reading of the after-entry numbers."""
    out = []
    if not after.get("trades") or after["trades"] < MIN_TRADES:
        return out
    rr = float(setup.get("rr") or 2.0)
    if after.get("reached_1r_pct", 0) >= 55 and after.get("reached_target_pct", 0) < 35:
        out.append(f"{after['reached_1r_pct']:.0f}% of trades got to +1R but only {after['reached_target_pct']:.0f}% "
                   f"reached the {rr:g}R target - take partial profit at 1R or use a closer target.")
    if after.get("winners_heat_r") is not None and after["winners_heat_r"] <= -0.6:
        out.append(f"Even winners usually went {abs(after['winners_heat_r']):.1f}R against you first - the entry is "
                   "early: wait for one more confirmation candle or a better price.")
    if after.get("losers_best_r") is not None and after["losers_best_r"] >= 0.8:
        out.append(f"Losing trades were typically up {after['losers_best_r']:.1f}R before turning - moving the stop "
                   "to breakeven after +1R would have saved many of them.")
    if after.get("reversal_exits"):
        s = after["reversal_saved_r"]
        out.append(f"Closing losing trades on a confirmed reversal: {after['reversal_exits']} times, "
                   + (f"saved {s:+.1f}R in total." if s > 0 else f"cost {s:+.1f}R in total (they often recovered)."))
    return out


def tune(charts: dict[str, pd.DataFrame], setup: dict, split: float = 0.7) -> dict | None:
    """Try nearby trigger settings (each rule's numbers, the stop and the target) on the first 70% of every
    chart; keep a change only if it also beats the current setting on the last 30% it never saw.
    Returns the best change, or None."""
    def score(s, part):
        per = {}
        for name, df in charts.items():
            cut = int(len(df) * split)
            per[name] = evaluate(df, s, *((0, cut) if part == "train" else (cut, len(df))))
        return combine(per)

    base_tr, base_te = score(setup, "train"), score(setup, "test")
    if (base_tr.get("trades") or 0) < MIN_TRADES:
        return None
    variants = []
    for i, r in enumerate(setup.get("rules", [])):
        if r.get("rule") not in builder.RULES:
            continue
        _, da, db, _ = builder.RULES[r["rule"]]
        for key, default in (("a", da), ("b", db)):
            cur = r.get(key, default)
            if not isinstance(cur, (int, float)) or isinstance(cur, bool) or cur == 0:
                continue
            for f in (0.5, 0.75, 1.25, 1.5):
                v = cur * f
                v = int(round(v)) if isinstance(cur, int) or float(cur).is_integer() else round(v, 2)
                if v == cur or (isinstance(cur, int) and v < 1 and cur >= 1):
                    continue
                rules = [dict(x) for x in setup["rules"]]
                rules[i][key] = v
                variants.append((f"{builder.describe(r)} -> {builder.describe(rules[i])}", {"rules": rules}))
    for st in (1.0, 1.5, 2.0, 2.5):
        if st != float(setup.get("stop_atr") or 1.5):
            variants.append((f"stop {setup.get('stop_atr') or 1.5:g} ATR -> {st:g} ATR", {"stop_atr": st}))
    for rr in (1.5, 2.0, 3.0):
        if rr != float(setup.get("rr") or 2.0):
            variants.append((f"target {setup.get('rr') or 2:g}R -> {rr:g}R", {"rr": rr}))
    best = None
    for label, change in variants:
        s = {**setup, **change}
        tr = score(s, "train")
        if (tr.get("trades") or 0) < MIN_TRADES or (tr["edge"] or 0) <= (base_tr["edge"] or 0) + 0.03:
            continue
        te = score(s, "test")
        if (te.get("trades") or 0) < 8 or (te["edge"] or 0) <= (base_te.get("edge") or 0):
            continue                                    # better in hindsight only: over-fitted
        gain = (tr["edge"] - (base_tr["edge"] or 0)) + (te["edge"] - (base_te.get("edge") or 0))
        if best is None or gain > best["gain"]:
            best = {"change": label, "set": change, "gain": round(gain, 3),
                    "train": {"edge": tr["edge"], "trades": tr["trades"], "was": base_tr["edge"]},
                    "test": {"edge": te["edge"], "trades": te["trades"], "was": base_te.get("edge")}}
    return best


# ---- the exit plan it deploys: its own stop and the early exit, both learned before they're used -------
FIXED_STOP = 0.03          # the app's default (apprentice.strategy.STOP)
FEE = 0.001
STOP_METHODS = ("fixed", "atr", "swing")
SWING_LOOKBACK = 10


def stop_pct(df: pd.DataFrame, direction: str, method: str, stop_atr: float = 1.5) -> np.ndarray:
    """Each candle's stop distance (fraction of price) for an entry at the next open:
      fixed - the app's 3%
      atr   - stop_atr x ATR
      swing - beyond the last swing low (long) / high (short) of the last 10 candles plus a quarter ATR,
              as the lessons teach ("place the stop below the last low"); kept between 0.5 and 4 ATR."""
    c = df["close"].to_numpy(float)
    a = atr_arr(df)
    if method == "fixed":
        return np.full(len(df), FIXED_STOP)
    if method == "atr":
        out = stop_atr * a / c
    else:
        if direction == "long":
            dist = c - df["low"].rolling(SWING_LOOKBACK, min_periods=1).min().to_numpy(float)
        else:
            dist = df["high"].rolling(SWING_LOOKBACK, min_periods=1).max().to_numpy(float) - c
        out = np.clip(dist + 0.25 * a, 0.5 * a, 4 * a) / c
    return np.clip(out, 0.002, 0.10)


def _plan_trade(o, h, l, c, i, sgn, risk_pct, against, horizon, use_rev) -> float | None:
    """Result (fraction of the position, after fees) of the trader's exits: half off at 1R, half at 2R,
    stop at -1R; optionally out at the next open if a confirmed reversal prints while losing."""
    n = len(o)
    if i + 1 >= n:
        return None
    entry = o[i + 1]
    risk = risk_pct * entry
    left, got = [1.0, 2.0], 0.0                        # open halves, by their target in R
    for k in range(1, horizon + 1):
        j = i + k
        if j >= n:
            return None
        adv = sgn * ((l[j] if sgn > 0 else h[j]) - entry) / risk
        fav = sgn * ((h[j] if sgn > 0 else l[j]) - entry) / risk
        if adv <= -1:
            got -= 0.5 * len(left)
            left = []
        for tg in list(left):
            if fav >= tg:
                got += 0.5 * tg
                left.remove(tg)
        if not left:
            break
        if use_rev and against[j] and j + 1 < n and sgn * (c[j] - entry) < 0:
            got += 0.5 * len(left) * max(-1.0, sgn * (o[j + 1] - entry) / risk)
            left = []
            break
        if k == horizon:
            got += 0.5 * len(left) * max(-1.0, sgn * (c[j] - entry) / risk)
    return got * risk_pct - 2 * FEE


def exit_plan(charts: dict[str, pd.DataFrame], setup: dict, split: float = 0.7, horizon: int = HORIZON) -> dict:
    """Learn which stop this setup should use and whether to close losers on a confirmed reversal.
    A change from the defaults (fixed 3% stop, no early exit) is only deployed if it made more per trade on
    the first 70% of every chart AND on the last 30% it never saw."""
    d = setup["direction"]
    sgn = 1 if d == "long" else -1
    res = {m: {"train": [], "test": []} for m in STOP_METHODS}
    rev = {"train": [], "test": []}
    cache = {}
    for name, df in charts.items():
        o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        mask = dedupe(setup_mask(df, setup))
        cut = int(len(df) * split)
        against = reversal_signals(df)[0 if d == "long" else 1]
        cache[name] = (o, h, l, c, mask, cut, against)
        for m in STOP_METHODS:
            sp = stop_pct(df, d, m, float(setup.get("stop_atr") or 1.5))
            for i in np.flatnonzero(mask):
                r = _plan_trade(o, h, l, c, i, sgn, sp[i], against, horizon, False)
                if r is not None:
                    res[m]["train" if i < cut else "test"].append(r)
    mean = lambda xs: float(np.mean(xs)) if xs else None
    per = {m: {"train_pct": None if mean(v["train"]) is None else round(mean(v["train"]) * 100, 3),
               "test_pct": None if mean(v["test"]) is None else round(mean(v["test"]) * 100, 3),
               "trades": len(v["train"]) + len(v["test"])} for m, v in res.items()}
    stop = "fixed"
    ok = [m for m in STOP_METHODS if len(res[m]["train"]) >= MIN_TRADES and len(res[m]["test"]) >= 5]
    if ok:
        best = max(ok, key=lambda m: mean(res[m]["train"]))
        if best != "fixed" and "fixed" in ok and mean(res[best]["test"]) > mean(res["fixed"]["test"]):
            stop = best
    # the early exit, with the stop it will actually use
    for name, df in charts.items():
        o, h, l, c, mask, cut, against = cache[name]
        sp = stop_pct(df, d, stop, float(setup.get("stop_atr") or 1.5))
        for i in np.flatnonzero(mask):
            a = _plan_trade(o, h, l, c, i, sgn, sp[i], against, horizon, False)
            b = _plan_trade(o, h, l, c, i, sgn, sp[i], against, horizon, True)
            if a is not None and b is not None and a != b:
                rev["train" if i < cut else "test"].append(b - a)
    gain = {k: round(float(sum(v)) * 100, 2) for k, v in rev.items()}
    use_rev = bool(len(rev["train"]) >= 5 and len(rev["test"]) >= 3 and gain["train"] > 0 and gain["test"] > 0)
    return {"stop": stop, "reversal": use_rev, "stops": per,
            "reversal_effect": {"times": len(rev["train"]) + len(rev["test"]), "train_pct": gain["train"],
                                "test_pct": gain["test"]}}


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
    key = _fingerprint(df)
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
