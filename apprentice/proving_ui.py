"""The main page's 🔬 Proving ground tab."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from apprentice import proving
from apprentice.job import DONE, JOBS, running, spawn
from apprentice.store import Store
from apprentice.strategy import db_path


def _last_log(kind: str) -> tuple[str, str] | None:
    if not JOBS.exists():
        return None
    for meta in sorted(JOBS.glob("*.json"), reverse=True):
        try:
            info = json.loads(meta.read_text())
        except Exception:
            continue
        if (info.get("args") or [None])[0] == kind:
            log = meta.with_suffix(".log")
            return info.get("title", kind), log.read_text(errors="ignore") if log.exists() else ""
    return None


def render(lp_store) -> None:
    from livepaper import settings
    from livepaper.history import CRYPTO
    store = Store(db_path())
    proving.ensure(store)
    st.markdown("##### 🔬 Proving ground")
    st.caption("The apprentice tests **every setup on every timeframe from 1 minute to 1 day**, on any crypto and any "
               "stock, with fees included. The best combinations - the ones that made money on both the first 70% "
               f"of history and the last 30% they never saw - get a **paper trial**. A trial stays until it has "
               f"**{proving.TRIAL_TRADES} trades**, then it's decided: kept if it made money after fees (it keeps "
               "trading), dropped if not (its slot goes to the next best). Runs by itself, also with this tab closed.")

    # ---- what to scan
    with st.expander("⚙️ Markets and timeframes to scan", expanded=not len(proving.results(store))):
        saved = settings.get("proving_markets") or []
        coins = [m.split(":")[0] for m in saved if m.endswith(":crypto")]
        stocks_ = [m.split(":")[0] for m in saved if m.endswith(":stock")]
        c1, c2 = st.columns(2)
        pick_c = c1.multiselect("Crypto", sorted(set(CRYPTO) | set(coins)), [c for c in coins if c in CRYPTO] or ["BTC"])
        extra = c1.text_input("Any other coin (comma separated, e.g. PEPE, ARB)",
                              ", ".join(c for c in coins if c not in CRYPTO))
        try:
            from livepaper.stocks import labels
            lab = labels()
        except Exception:
            lab = {}
        pick_s = c2.multiselect("Stocks (type to search)", sorted(set(lab) | set(stocks_)), stocks_,
                                format_func=lambda t: lab.get(t, t))
        tfs = st.multiselect("Timeframes", proving.TIMEFRAMES,
                             settings.get("proving_timeframes") or proving.TIMEFRAMES)
        markets = [f"{c}:crypto" for c in pick_c] + [f"{c.strip().upper()}:crypto" for c in extra.split(",") if c.strip()] \
            + [f"{s}:stock" for s in pick_s]
        b1, b2, b3 = st.columns([1, 1, 2])
        if b1.button("💾 Save", key="pg_save"):
            settings.save({"proving_markets": markets, "proving_timeframes": tfs})
            st.toast("Saved - the daily re-scan uses these")
        busy = running("scan")
        if b2.button("🔬 Scan now", type="primary", disabled=busy or not markets or not tfs, key="pg_scan"):
            settings.save({"proving_markets": markets, "proving_timeframes": tfs})
            spawn(["scan", "--markets", *markets, "--timeframes", *tfs], "🔬 Proving ground scan")
            st.success("Scanning in the background - minute charts take longest (a few minutes per market).")
        b3.caption(f"{len(markets)} markets x {len(tfs)} timeframes x every live setup. Re-scans by itself every "
                   f"{settings.get('proving_scan_hours', 24)} h. 1-minute history: about a week for crypto (Coinbase) "
                   "and 7 days for stocks (Yahoo); hourly and up go back years.")
    log = _last_log("scan")
    if log:
        title, text = log
        done = DONE in text
        with st.expander(f"{'✅' if done else '⏳'} {title}", expanded=not done):
            st.code("\n".join(text.splitlines()[-25:]) or "starting...")

    res = proving.results(store)
    setups = {s["id"]: s for s in store.setups()}
    name = lambda i: f"#{i} {setups[i]['direction'][0].upper()} · {setups[i]['name'][:38]}" if i in setups else f"#{i}"
    if res.empty:
        st.info("No scan yet - pick markets and timeframes above and press 🔬 Scan now.")
        return

    # ---- best timeframe per setup
    st.markdown("##### 🏆 Best timeframe for each setup")
    best = proving.best_timeframes(store)
    if best.empty:
        st.warning("Nothing made money after fees on both the first 70% and the unseen 30% yet. That's an honest "
                   "result: most setups don't survive fees on low timeframes. Scan more markets, or teach it more.")
    else:
        st.dataframe(pd.DataFrame([{"setup": name(r.setup_id), "best timeframe": r.timeframe, "market": r.symbol,
                                    "R per trade after fees": r.avg_r, "unseen 30%": r.test_r, "trades": r.trades,
                                    "% per trade": r.avg_pct} for r in best.itertuples()]),
                     hide_index=True, width="stretch")
    with st.expander("Every setup x timeframe (best market, R per trade after fees; blank = didn't hold up)"):
        c = proving.candidates(store)
        if len(c):
            grid = c.sort_values("score", ascending=False).groupby(["setup_id", "timeframe"]).head(1)
            piv = grid.pivot_table(index="setup_id", columns="timeframe", values="avg_r", aggfunc="max")
            piv = piv.reindex(columns=[t for t in proving.TIMEFRAMES if t in piv.columns])
            piv.index = [name(i) for i in piv.index]
            st.dataframe(piv.round(2), width="stretch")
        tot = res.groupby("timeframe").apply(lambda g: (g.apply(lambda r: proving.candidate_ok(r.to_dict()), axis=1)).sum())
        st.caption("Combinations that held up, per timeframe: " + " · ".join(
            f"{tf} {int(tot.get(tf, 0))}" for tf in proving.TIMEFRAMES if tf in tot.index))

    # ---- trials
    st.markdown(f"##### 🧪 Trials ({proving.TRIAL_TRADES} trades each, then keep or drop)")
    c1, c2 = st.columns(2)
    mx = c1.number_input("Trials at the same time", 1, 50, int(settings.get("proving_max_trials", 10) or 10))
    auto = c2.toggle("Start the best untried candidates by themselves", bool(settings.get("proving_auto_start", True)))
    if mx != int(settings.get("proving_max_trials", 10) or 10) or auto != bool(settings.get("proving_auto_start", True)):
        settings.save({"proving_max_trials": int(mx), "proving_auto_start": auto})
    if st.button("▶️ Fill the trial slots now", key="pg_tick"):
        out = proving.tick(store, lp_store, int(mx), True, log=lambda m: None)
        st.success(f"Started {out['started']}, kept {out['kept']}, dropped {out['dropped']}. The paper trader picks "
                   "new trials up on its next price check.")
    tr = proving.trials(store)
    run_rows = tr[tr.status == "running"] if len(tr) else tr
    for r in run_rows.itertuples() if len(run_rows) else []:
        t = proving.closed_trades(lp_store, r.run)
        n, pnl = len(t), float(t.pnl_usd.sum()) if len(t) else 0.0
        a, b, c = st.columns([3, 2, 1])
        a.markdown(f"**{r.run}** · {name(r.setup_id)}")
        b.progress(min(n / proving.TRIAL_TRADES, 1.0), text=f"{n}/{proving.TRIAL_TRADES} trades · {pnl:+,.2f} USD")
        if c.button("Stop", key=f"pg_stop_{r.run}"):
            lp_store.set_active(r.run, False)
            store.db.execute("UPDATE proving_trials SET status = 'stopped', decided = ? WHERE run = ?",
                             (proving.now(), r.run))
            store.db.commit()
            st.rerun()
    if not len(run_rows):
        st.caption("No trials running.")
    with st.expander("Candidates waiting for a trial"):
        cand = proving.candidates(store)
        tried = set(zip(tr.setup_id, tr.symbol, tr.timeframe)) if len(tr) else set()
        for i, r in enumerate(cand.head(25).itertuples()):
            if (r.setup_id, r.symbol, r.timeframe) in tried:
                continue
            a, b = st.columns([5, 1])
            a.markdown(f"{name(r.setup_id)} · **{r.symbol} {r.timeframe}** · {r.avg_r:+.2f}R per trade after fees "
                       f"({r.trades} trades, unseen {r.test_r:+.2f}R)")
            if b.button("Start", key=f"pg_start_{i}") and r.setup_id in setups:
                proving.start_trial(store, lp_store, setups[r.setup_id], r.symbol, r.asset, r.timeframe)
                st.rerun()
    done = tr[tr.status.isin(["kept", "dropped", "stopped"])] if len(tr) else tr
    if len(done):
        st.markdown("##### ✅ Decided")
        rows = []
        for r in done.itertuples():
            res_ = json.loads(r.result) if r.result else {}
            rows.append({"": {"kept": "✅ kept", "dropped": "❌ dropped", "stopped": "⏹️ stopped"}[r.status],
                         "trial": r.run, "setup": name(r.setup_id), "trades": res_.get("trades"),
                         "P&L USD": res_.get("pnl_usd"), "win %": res_.get("win_rate"),
                         "profit factor": res_.get("profit_factor"), "decided": (r.decided or "")[:16]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
