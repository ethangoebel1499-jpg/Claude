"""🔬 Proving ground: which setup works on which timeframe and market - then prove it with 100 paper trades.

1. Scan  - every live setup x every timeframe from 1 minute to 1 day x the markets you pick (any crypto, any
           stock): its past trades with its own ATR stop and target, FEES INCLUDED (0.1% each way - what kills
           most 1-minute ideas), on the first 70% of history and on the last 30% it never saw.
2. Trial - the best combinations (made money on both parts, enough trades) start a paper trial: a paper run that
           trades just that setup on that market and timeframe.
3. Decide- a trial stays until it has 100 closed trades, then it's decided: kept (made money after fees, the run
           keeps trading) or dropped (the run stops). Its slot goes to the next best candidate.

The trader checks the trials every few minutes (livepaper runner), so this works with the tab closed.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import brain

TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1D"]
TRIAL_TRADES = 100
FEE = 0.001                      # per side, like the paper broker
PREFIX = "apprentice_setup:"     # strategy key of a trial run: one setup on its own

# built-in strategies tested next to the setups (negative ids): pure price action, longs and shorts apart
PSEUDO = {-10: {"id": -10, "key": "s10_long", "name": "S10 Price action (+RSI & money flow) - longs", "direction": "long"},
          -11: {"id": -11, "key": "s10_short", "name": "S10 Price action (+RSI & money flow) - shorts", "direction": "short"}}

SCHEMA = """
CREATE TABLE IF NOT EXISTS proving_scan (setup_id INTEGER, symbol TEXT, asset TEXT, timeframe TEXT, trades INTEGER,
    train_trades INTEGER, test_trades INTEGER, avg_r REAL, train_r REAL, test_r REAL, edge REAL, avg_pct REAL,
    candles INTEGER, scanned TEXT, PRIMARY KEY (setup_id, symbol, timeframe));
CREATE TABLE IF NOT EXISTS proving_trials (run TEXT PRIMARY KEY, setup_id INTEGER, symbol TEXT, asset TEXT,
    timeframe TEXT, started TEXT, status TEXT, decided TEXT, result TEXT);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure(store) -> None:
    store.db.executescript(SCHEMA)
    store.db.commit()


def pair_for(symbol: str, asset: str) -> str:
    """What the paper trader needs: a Kraken pair for crypto (any coin: XYZ -> XYZUSD), the ticker for stocks."""
    if asset != "crypto":
        return symbol.upper()
    from livepaper.history import CRYPTO
    s = symbol.upper()
    return CRYPTO.get(s, s if s.endswith("USD") else f"{s}USD")


# ---- 1. scan -----------------------------------------------------------------------------------------------
def net_outcomes(df: pd.DataFrame, setup: dict) -> tuple[np.ndarray, np.ndarray]:
    """(R after fees, stop distance as a fraction of price) of entering on every candle with the setup's exits."""
    stop_atr, rr = float(setup.get("stop_atr") or 1.5), float(setup.get("rr") or 2.0)
    res = brain.outcomes(df, setup["direction"], stop_atr, rr)
    risk = stop_atr * brain.atr_arr(df) / df["close"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        fee_r = np.where(risk > 0, 2 * FEE / risk, np.nan)
    return res - fee_r, risk


def outcomes_pct(df: pd.DataFrame, direction: str, risk: np.ndarray, rr: float, horizon: int = brain.HORIZON) -> np.ndarray:
    """R of entering on every candle with a stop `risk` (fraction of price) away and a target rr x that."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(df)
    sgn = 1 if direction == "long" else -1
    entry = np.full(n, np.nan)
    entry[:-1] = o[1:]
    dist = risk * entry
    stop, tgt = entry - sgn * dist, entry + sgn * rr * dist
    res = np.full(n, np.nan)
    open_ = np.isfinite(entry) & (dist > 0)
    for k in range(1, horizon + 1):
        idx = np.arange(n) + k
        ok = idx < n
        hk = np.where(ok, h[np.minimum(idx, n - 1)], np.nan)
        lk = np.where(ok, l[np.minimum(idx, n - 1)], np.nan)
        hs = open_ & ok & ((lk <= stop) if sgn > 0 else (hk >= stop))
        ht = open_ & ok & ~hs & ((hk >= tgt) if sgn > 0 else (lk <= tgt))
        res[hs], res[ht] = -1.0, rr
        open_ &= ~(hs | ht)
        if k == horizon:
            last = np.where(ok, c[np.minimum(idx, n - 1)], np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):      # candles with no trade have no stop
                res[open_ & ok] = (sgn * (last - entry) / dist)[open_ & ok]
    return np.clip(res, -1.0, rr)


def simulate(df: pd.DataFrame, i: int, sgn: int, stop: float, tp1: float, tp2: float, max_bars: int = 300) -> float | None:
    """One trade signalled on candle i (entry at the next open): half at TP1, then the stop moves to breakeven, the
    rest at TP2. Moves are fractions of the entry. Returns R after fees, or None if there isn't enough future."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(df)
    if i + 1 >= n or not (stop > 0 and tp1 > 0):
        return None
    e = o[i + 1]
    stop_px, t1, t2 = e * (1 - sgn * stop), e * (1 + sgn * tp1), e * (1 + sgn * max(tp2, tp1))
    left, got, be = 1.0, 0.0, False
    for j in range(i + 1, min(n, i + 1 + max_bars)):
        adv, fav = (l[j], h[j]) if sgn > 0 else (h[j], l[j])
        s_px = e if be else stop_px
        if sgn * (adv - s_px) <= 0:                         # stop (or breakeven) first - the cautious order
            got += left * sgn * (s_px - e) / e
            left = 0.0
            break
        if left == 1.0 and sgn * (fav - t1) >= 0:
            got += 0.5 * tp1
            left, be = 0.5, True
        if left == 0.5 and sgn * (fav - t2) >= 0:
            got += 0.5 * max(tp2, tp1)
            left = 0.0
            break
    if left > 0:
        if j >= n - 1:
            return None
        got += left * sgn * (c[j] - e) / e
    return (got - 2 * FEE) / stop


def scan_s10(df: pd.DataFrame, direction: str, split: float = 0.7) -> dict | None:
    """S10 as it trades: its own stop and chart-level targets on every trade, breakeven after TP1, fees included."""
    from mcbot import price_action as pa
    if len(df) < 300:
        return None
    sig = pa.signals(df, direction)
    sgn = 1 if direction == "long" else -1
    idx = np.flatnonzero(brain.dedupe(sig[f"{direction}_entry"].to_numpy(bool)))
    st, t1, t2 = (sig[k].to_numpy(float) for k in ("stop_pct", "tp1_pct", "tp2_pct"))
    rows = [(i, simulate(df, i, sgn, st[i], t1[i], t2[i])) for i in idx]
    rows = [(i, r) for i, r in rows if r is not None and np.isfinite(r)]
    cut = int(len(df) * split)
    if not rows:
        return {"trades": 0, "train_trades": 0, "test_trades": 0, "avg_r": None, "train_r": None, "test_r": None,
                "edge": None, "avg_pct": None, "candles": len(df)}
    r_all = np.array([r for _, r in rows])
    tr = np.array([r for i, r in rows if i < cut])
    te = np.array([r for i, r in rows if i >= cut])
    mean = lambda a: round(float(a.mean()), 3) if len(a) else None
    pct = np.array([r * st[i] for i, r in rows])
    return {"trades": len(rows), "train_trades": len(tr), "test_trades": len(te), "avg_r": mean(r_all),
            "train_r": mean(tr), "test_r": mean(te), "edge": mean(r_all),
            "avg_pct": round(float(pct.mean()) * 100, 3), "candles": len(df)}


def scan_one(df: pd.DataFrame, setup: dict, split: float = 0.7) -> dict | None:
    if len(df) < 300:
        return None
    net, risk = net_outcomes(df, setup)
    mask = brain.dedupe(brain.setup_mask(df, setup)) & np.isfinite(net)
    cut = int(len(df) * split)
    idx = np.flatnonzero(mask)
    if not len(idx):
        return {"trades": 0, "train_trades": 0, "test_trades": 0, "avg_r": None, "train_r": None, "test_r": None,
                "edge": None, "avg_pct": None, "candles": len(df)}
    tr, te = idx[idx < cut], idx[idx >= cut]
    base = float(np.nanmean(net))
    mean = lambda a: round(float(np.mean(a)), 3) if len(a) else None
    return {"trades": int(len(idx)), "train_trades": int(len(tr)), "test_trades": int(len(te)),
            "avg_r": mean(net[idx]), "train_r": mean(net[tr]), "test_r": mean(net[te]),
            "edge": round(float(np.mean(net[idx])) - base, 3),
            "avg_pct": round(float(np.mean(net[idx] * risk[idx])) * 100, 3), "candles": len(df)}


def scan(store, markets: list[tuple[str, str]], timeframes: list[str] | None = None, setups: list[dict] | None = None,
         log=print) -> int:
    """Test every live setup on every market x timeframe; results go to the proving_scan table."""
    from livepaper.history import history
    ensure(store)
    timeframes = timeframes or TIMEFRAMES
    setups = setups if setups is not None else [s for s in store.setups(("active", "pinned")) if s.get("rules")]
    n = 0
    for symbol, asset in markets:
        for tf in timeframes:
            try:
                df = history(pair_for(symbol, asset) if asset == "crypto" else symbol, asset, tf).iloc[-5000:]
            except Exception as e:
                log(f"  {symbol} {tf}: couldn't load prices ({e})")
                continue
            if len(df) < 300:
                log(f"  {symbol} {tf}: only {len(df)} candles - skipped")
                continue
            good = 0
            for s in setups + list(PSEUDO.values()):
                try:
                    r = scan_s10(df, s["direction"]) if s["id"] in PSEUDO else scan_one(df, s)
                except Exception as e:
                    log(f"  #{s['id']} on {symbol} {tf}: {e}")
                    continue
                if r is None:
                    continue
                store.db.execute("INSERT OR REPLACE INTO proving_scan VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                 (s["id"], symbol.upper(), asset, tf, r["trades"], r["train_trades"], r["test_trades"],
                                  r["avg_r"], r["train_r"], r["test_r"], r["edge"], r["avg_pct"], r["candles"], now()))
                n += 1
                good += bool(candidate_ok(r))
            store.db.commit()
            log(f"  {symbol} {tf}: {len(df):,} candles, {good} of {len(setups) + len(PSEUDO)} setups/strategies made "
                "money after fees on both the first 70% and the unseen 30%")
    log(f"RESULT scanned={n}")
    return n


def candidate_ok(r: dict) -> bool:
    return bool(r.get("trades", 0) >= 20 and r.get("test_trades", 0) >= 5 and (r.get("train_r") or 0) > 0
                and (r.get("test_r") or 0) > 0 and (r.get("edge") or 0) > 0)


def results(store) -> pd.DataFrame:
    ensure(store)
    return pd.read_sql_query("SELECT * FROM proving_scan", store.db)


def candidates(store) -> pd.DataFrame:
    """Combinations worth a trial, best first: made money after fees on both parts of history."""
    df = results(store)
    if df.empty:
        return df
    ok = df.apply(lambda r: candidate_ok(r.to_dict()), axis=1)
    df = df[ok].copy()
    live = {s["id"] for s in store.setups(("active", "pinned"))} | set(PSEUDO)
    df = df[df.setup_id.isin(live)]
    df["score"] = df.avg_r * df.trades.apply(math.sqrt)
    return df.sort_values("score", ascending=False)


def best_timeframes(store) -> pd.DataFrame:
    """Per setup: its best timeframe (and market) by R after fees, among those that held up on unseen prices."""
    c = candidates(store)
    if c.empty:
        return c
    return c.sort_values("score", ascending=False).groupby("setup_id").head(1)


# ---- 2. trials -------------------------------------------------------------------------------------------
def trials(store) -> pd.DataFrame:
    ensure(store)
    return pd.read_sql_query("SELECT * FROM proving_trials ORDER BY started", store.db)


def run_name(setup_id: int, symbol: str, tf: str) -> str:
    if setup_id in PSEUDO:
        return f"Trial S10 {PSEUDO[setup_id]['direction']}s {symbol.upper()} {tf}"
    return f"Trial #{setup_id} {symbol.upper()} {tf}"


def start_trial(store, lp_store, setup: dict, symbol: str, asset: str, tf: str, capital: float = 10_000.0) -> str:
    from .strategy import STOP
    ensure(store)
    name = run_name(setup["id"], symbol, tf)
    rr = float(setup.get("rr") or 2.0)
    if setup["id"] in PSEUDO:                       # a built-in strategy (S10): its own exits
        from mcbot import price_action as pa
        strat, tps = PSEUDO[setup["id"]]["key"], [list(x) for x in pa.TAKE_PROFITS]
    else:
        strat, tps = f"{PREFIX}{setup['id']}", [[STOP * rr, 1.0]]
    lp_store.upsert_run({"name": name, "strategy": strat, "symbol": pair_for(symbol, asset),
                         "asset": asset, "timeframe": tf, "shorts": setup["direction"] == "short", "capital": capital,
                         "fraction": 1.0, "stop": STOP, "take_profits": tps, "active": True,
                         "source": "proving"})
    store.db.execute("INSERT OR REPLACE INTO proving_trials VALUES (?,?,?,?,?,?,?,?,?)",
                     (name, setup["id"], symbol.upper(), asset, tf, now(), "running", None, None))
    store.db.commit()
    return name


def closed_trades(lp_store, run: str) -> pd.DataFrame:
    try:
        t = lp_store.table("trades", run)
    except Exception:
        return pd.DataFrame()
    if t.empty:
        return t
    return t[~t.reason.astype(str).str.contains("switched|removed")]


def verdict(t: pd.DataFrame) -> tuple[bool, dict]:
    """Keep a trial after 100 trades if it made money after fees and its wins outweigh its losses."""
    wins, losses = t.pnl_usd[t.pnl_usd > 0].sum(), -t.pnl_usd[t.pnl_usd < 0].sum()
    pf = float(wins / losses) if losses > 0 else float("inf")
    res = {"trades": int(len(t)), "pnl_usd": round(float(t.pnl_usd.sum()), 2),
           "avg_pct": round(float(t.pnl_pct.mean()), 3), "win_rate": round(float((t.pnl_usd > 0).mean() * 100), 1),
           "profit_factor": round(pf, 2) if math.isfinite(pf) else None}
    return bool(res["pnl_usd"] > 0 and pf >= 1.1), res


def tick(store, lp_store, max_trials: int = 10, auto_start: bool = True, log=print) -> dict:
    """Decide trials that reached 100 trades, then fill free slots with the best untried candidates."""
    ensure(store)
    out = {"kept": 0, "dropped": 0, "started": 0}
    setups = {s["id"]: s for s in store.setups()} | dict(PSEUDO)
    for r in trials(store).itertuples():
        if r.status != "running":
            continue
        t = closed_trades(lp_store, r.run)
        gone = r.setup_id not in setups or setups[r.setup_id]["status"] == "off"
        if len(t) < TRIAL_TRADES and not gone:
            continue
        keep, res = verdict(t) if len(t) else (False, {"trades": 0})
        if gone:
            keep, res["why"] = False, "its setup was turned off"
        status = "kept" if keep else "dropped"
        store.db.execute("UPDATE proving_trials SET status = ?, decided = ?, result = ? WHERE run = ?",
                         (status, now(), json.dumps(res), r.run))
        if not keep:
            lp_store.set_active(r.run, False)
        out[status] += 1
        log(f"[proving] {r.run}: {status} after {res.get('trades')} trades ({res})")
    store.db.commit()
    if auto_start:
        tr = trials(store)
        running = int((tr.status == "running").sum()) if len(tr) else 0
        tried = set(zip(tr.setup_id, tr.symbol, tr.timeframe)) if len(tr) else set()
        for c in candidates(store).itertuples():
            if running >= max_trials:
                break
            if (c.setup_id, c.symbol, c.timeframe) in tried or c.setup_id not in setups:
                continue
            start_trial(store, lp_store, setups[c.setup_id], c.symbol, c.asset, c.timeframe)
            tried.add((c.setup_id, c.symbol, c.timeframe))
            running += 1
            out["started"] += 1
            log(f"[proving] started a trial: setup #{c.setup_id} on {c.symbol} {c.timeframe}")
    return out


def tick_from_runner(lp_store, log=print) -> None:
    """Called by the paper trader every few minutes."""
    from pathlib import Path

    from livepaper import settings

    from .store import Store
    from .strategy import db_path
    if not Path(db_path()).exists() or not settings.get("proving_on", True):
        return
    st = Store(db_path())
    try:
        tick(st, lp_store, int(settings.get("proving_max_trials", 10) or 10),
             bool(settings.get("proving_auto_start", True)), log)
    finally:
        st.db.close()


def scan_if_due() -> bool:
    """Re-scan in the background when the last scan is older than proving_scan_hours (default a day)."""
    from pathlib import Path

    from livepaper import settings

    from .job import running, spawn
    from .store import Store
    from .strategy import db_path
    hours = float(settings.get("proving_scan_hours", 24) or 0)
    if not hours or not settings.get("proving_on", True) or not Path(db_path()).exists() or running("scan"):
        return False
    st = Store(db_path())
    try:
        ensure(st)
        last = st.db.execute("SELECT MAX(scanned) FROM proving_scan").fetchone()[0]
    finally:
        st.db.close()
    if last and (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds() < hours * 3600:
        return False
    spawn(["scan"], "🔬 Proving ground scan (automatic)")
    return True


# ---- the strategy a trial run trades ------------------------------------------------------------------------
def trial_entry(setup: dict) -> dict:
    """Registry entry for one setup on its own: its ATR stop on every trade, its target, nothing else
    (no safety / avoid rules, no pausing) - a clean test of the setup itself."""
    from livepaper.core import Exits

    from . import strategy as S
    stop_atr, rr = float(setup.get("stop_atr") or 1.5), float(setup.get("rr") or 2.0)

    def signals(df: pd.DataFrame) -> pd.DataFrame:
        sig = S.signals(df, [setup], mode="off", safety={"no_opposite": False, "breakeven": False},
                        paused=False, avoid={})
        stop = np.clip(stop_atr * brain.atr_arr(df) / df["close"].to_numpy(float), 0.0005, 0.10)
        sig["stop_pct"] = stop
        sig["be_pct"] = np.nan
        return sig

    def checklist(df: pd.DataFrame, bar: int = -1, label: str | None = None) -> str:
        from mcbot import builder
        i = len(df) + bar if bar < 0 else bar
        ok = [bool(builder.rule_series(df, r).iloc[i]) for r in setup["rules"]]
        lines = [f"[{label or 'Trial'}] candle {df.index[i]:%Y-%m-%d %H:%M} UTC · setup #{setup['id']} "
                 f"{setup['direction'].upper()}: {'🟢 TRADE' if all(ok) else '🔴 NO TRADE'}"]
        lines += [f"  {'🟢 YES' if o else '🔴 NO '}  {builder.describe(r)}" for o, r in zip(ok, setup["rules"])]
        return "\n".join(lines)

    return {"label": f"🔬 Trial: setup #{setup['id']} {setup['name'][:40]}", "short": f"Trial #{setup['id']}",
            "signals": signals, "checklist": checklist, "exits": Exits(S.STOP, [(S.STOP * rr, 1.0)]),
            "long_only": setup["direction"] == "long",
            "note": f"Trades only setup #{setup['id']} ({setup['direction']}): stop {stop_atr:g} ATR, target {rr:g}R. "
                    f"Decided after {TRIAL_TRADES} trades in 🔬 Proving ground."}


def refresh(registry: dict) -> dict:
    """Registry entries for every setup that has (or had) a trial."""
    from pathlib import Path

    from .store import Store
    from .strategy import db_path
    for k in [k for k in registry if k.startswith(PREFIX)]:
        del registry[k]
    if not Path(db_path()).exists():
        return registry
    st = Store(db_path())
    try:
        ensure(st)
        ids = {int(r[0]) for r in st.db.execute("SELECT DISTINCT setup_id FROM proving_trials").fetchall()}
        for s in st.setups():
            if s["id"] in ids and s.get("rules"):
                registry[f"{PREFIX}{s['id']}"] = trial_entry(s)
    finally:
        st.db.close()
    return registry
