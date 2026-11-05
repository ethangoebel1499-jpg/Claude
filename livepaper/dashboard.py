"""livepaper dashboard. Start it with:  python -m livepaper dashboard"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from livepaper import settings  # noqa: E402
from livepaper.fmt import money, pct, price  # noqa: E402  (full prices, never scientific notation)
from livepaper.history import CRYPTO, STOCKS, history  # noqa: E402
from livepaper.runner import exits_for, load_config, max_stop, normalize_run, seed_runs  # noqa: E402
from livepaper import stocks, tvchart  # noqa: E402
from livepaper.tp_plans import OTHER, STOCK_NAMES, TP_PLANS, plan_text, tp_rows, tps_from_table  # noqa: E402
from livepaper.store import Store  # noqa: E402
from livepaper.strategies import NOT_LIVE, REGISTRY, refresh as refresh_strategies  # noqa: E402
from livepaper import custom  # noqa: E402
from mcbot import builder  # noqa: E402
from mcbot.backtest import run_backtest  # noqa: E402

DB = os.environ.get("LIVEPAPER_DB", "livepaper.db")
CONFIG = os.environ.get("LIVEPAPER_CONFIG", "live.toml")
TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1D", "2D"]
PERIODS = {"Last year": 365, "Last 3 years": 3 * 365, "All history": None}
INTRADAY_PERIODS = {"Last day": 1, "Last 3 days": 3, "Last week": 7, "Last 2 weeks": 14, "Last month": 30,
                    "All available": None}

st.set_page_config(page_title="Paper Trading", page_icon="📈", layout="wide")


# ---- helpers ------------------------------------------------------------------
@st.cache_resource
def get_store(path: str) -> Store:
    return Store(path)


COIN = {pair: coin for coin, pair in CRYPTO.items()}


def color(v):
    if isinstance(v, (int, float)) and not pd.isna(v):
        return "color: #16a34a; font-weight: 600" if v > 0 else "color: #dc2626; font-weight: 600" if v < 0 else ""
    return ""


def exit_text(ex) -> str:
    tps = ", ".join(f"{s:.0%} at +{lv:.1%}" if lv is not None else f"{s:.0%} rides" for lv, s in ex.take_profits)
    return f"stop {ex.stop:.0%} · {tps}" if ex.stop else f"no stop · {tps}"


def parse_tps(text: str) -> list[tuple[float | None, float]]:
    """'12:50, 24:50' or '2:50, 5:25, ride:25'  ->  [(0.12, 0.5), (0.24, 0.5)]"""
    out = []
    for part in text.split(","):
        move, size = (x.strip() for x in part.split(":"))
        out.append((None if move.lower() == "ride" else float(move) / 100, float(size) / 100))
    if abs(sum(s for _, s in out) - 1) > 1e-6:
        raise ValueError("take-profit sizes must add up to 100")
    return out


def tps_text(tps) -> str:
    return ", ".join(f"{'ride' if lv is None else f'{lv * 100:g}'}:{s * 100:g}" for lv, s in tps)


def trade_view(r: dict, state, quote) -> dict | None:
    """Live numbers for one open position."""
    p = state.position if state else None
    if p is None or quote is None:
        return None
    price, d = quote["price"], 1 if p.side == "long" else -1
    unreal = d * (price - p.entry_price) * p.remaining_qty
    pnl_usd = p.realized_usd + unreal - p.fees_usd
    ex = exits_for(r)
    fixed = [lv for lv, _ in ex.take_profits if lv is not None]
    nxt = fixed[p.targets_hit] if p.targets_hit < len(fixed) else None
    return {"Run": r["name"], "Symbol": COIN.get(r["symbol"], r["symbol"]), "Side": "🟢 LONG" if d > 0 else "🔴 SHORT",
            "Entry": p.entry_price, "Price": price, "24h": quote["change_24h"],
            "Since entry": (price / p.entry_price - 1) * 100, "My trade %": pnl_usd / (p.entry_price * p.qty) * 100,
            "My trade $": pnl_usd,
            "Stop": p.entry_price * (1 - d * ex.stop) if ex.stop else None,
            "Stop held by": "🛡️ broker" if p.stop_order else "💻 this computer",
            "Next target": p.entry_price * (1 + d * nxt) if nxt else ("rides" if p.targets_hit else None),
            "Left": f"{p.remaining_qty / p.qty:.0%}", "unreal": unreal}


# ---- data ---------------------------------------------------------------------
store = get_store(DB)
refresh_strategies()                                   # your own strategies and names
cfg = load_config(CONFIG) if Path(CONFIG).exists() else {"run": [], "app": {}}
seed_runs(cfg, store)

st.title("📈 Paper Trading")
tab_dash, tab_lab, tab_build, tab_streams, tab_app, tab_prove, tab_log, tab_acct = st.tabs(
    ["📊 Dashboard", "🧪 Strategy Lab", "🛠️ Builder", "🎙️ Streams", "🧠 Apprentice", "🔬 Proving ground", "📜 Activity",
     "⚙️ Account"])


# ---- Dashboard ----------------------------------------------------------------
@st.fragment(run_every=30)
def dashboard():
    runs = [normalize_run(r) for r in store.runs(active_only=True)]
    quotes = store.table("quotes").set_index("run").to_dict("index")
    trades, eq = store.table("trades"), store.table("equity")
    states = {r["name"]: store.load_state(r["name"]) for r in runs}
    broker = cfg.get("app", {}).get("broker", "sim")

    newest = max((q["time"] for q in quotes.values()), default=None)
    age = (datetime.now(timezone.utc) - pd.Timestamp(newest)).total_seconds() if newest else None
    status = ("🟢 trader running" if age is not None and age < 180 else
              "⚠️ trader not running - start it with `python -m livepaper run`")
    st.caption(f"{status} · broker: **{'Webull paper account' if broker == 'webull' else 'built-in paper broker'}**"
               f" · {len(runs)} active runs · refreshes every 30 s")
    try:
        from apprentice.ui import trade_bar
        trade_bar(store)
    except Exception as e:
        st.caption(f"🧠 Apprentice bar unavailable: {type(e).__name__}: {e}")
    if not runs:
        st.info("No runs are paper trading yet. Open the **🧪 Strategy Lab** tab, test a strategy, and tick "
                "**Paper trade this setup**.")
        return

    opens = [v for r in runs if (v := trade_view(r, states[r["name"]], quotes.get(r["name"])))]
    rows, total_cap, total_eq = [], 0.0, 0.0
    for r in runs:
        s, q, t = states[r["name"]], quotes.get(r["name"]), trades[trades.run == r["name"]]
        unreal = next((o["unreal"] for o in opens if o["Run"] == r["name"]), 0.0)
        e = (s.equity if s else r["capital"]) + unreal
        total_cap, total_eq = total_cap + r["capital"], total_eq + e
        rows.append({"Run": r["name"], "Strategy": REGISTRY[r["strategy"]]["short"], "Market": COIN.get(r["symbol"], r["symbol"]),
                     "TF": r["timeframe"], "Price": q["price"] if q else float("nan"), "Equity": e,
                     "Return": (e / r["capital"] - 1) * 100, "Trades": len(t),
                     "Win rate": (t.pnl_usd > 0).mean() * 100 if len(t) else float("nan"),
                     "Position": "🟢 long" if s and s.position and s.position.side == "long" else
                                 "🔴 short" if s and s.position else "⚪ flat"})

    closed = trades[trades.run.isin([r["name"] for r in runs])]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total equity", money(total_eq), pct((total_eq / total_cap - 1) * 100))
    open_pnl = sum(o["My trade $"] for o in opens)
    c2.metric("Open P&L", money(open_pnl), delta_color="normal",
              delta=pct(open_pnl / max(sum(o["Entry"] for o in opens), 1e-9) * 100) if opens else None)
    c3.metric("Closed P&L", money(closed.pnl_usd.sum()) if len(closed) else "$0")
    c4.metric("Open positions", len(opens))
    c5.metric("Win rate", f"{(closed.pnl_usd > 0).mean() * 100:.0f}%" if len(closed) else "-")

    st.subheader("Open trades")
    if opens:
        df = pd.DataFrame(opens).drop(columns="unreal")
        st.dataframe(df.style.format({"Entry": price, "Price": price, "24h": pct, "Since entry": pct,
                                      "My trade %": pct, "My trade $": money, "Stop": price,
                                      "Next target": price}, na_rep="-")
                     .map(color, subset=["24h", "Since entry", "My trade %", "My trade $"]),
                     hide_index=True, width="stretch")
    else:
        st.caption("No open trades right now - every run is waiting for its next signal.")

    with st.expander("All runs", expanded=True):
        table = pd.DataFrame(rows)
        table["Win rate"] = ["-" if pd.isna(v) else f"{v:.0f}%" for v in table["Win rate"]]   # grid shows NaN as None
        st.dataframe(table.style.format({"Price": price, "Equity": money, "Return": pct})
                     .map(color, subset=["Return"]), hide_index=True, width="stretch")

    st.subheader("Run details")
    for r, row in zip(runs, rows):
        with st.expander(f"{r['name']} · {pct(row['Return'])} · {row['Position']}"):
            st.caption(f"{REGISTRY[r['strategy']]['label']} · {r['symbol']} {r['timeframe']} · "
                       f"{exit_text(exits_for(r))} · shorts {'on' if r['shorts'] else 'off'} · "
                       f"capital {money(r['capital'])}")
            run_chart(r, store.load_state(r["name"]))
            manual_controls(r)
            left, right = st.columns([3, 2])
            e = eq[eq.run == r["name"]]
            with left:
                if len(e) > 1:
                    ch = alt.Chart(e.assign(time=pd.to_datetime(e.time, format="ISO8601", utc=True))).mark_line().encode(
                        x=alt.X("time:T", title=None),
                        y=alt.Y("equity:Q", title="Equity ($)", scale=alt.Scale(zero=False)),
                        tooltip=["time:T", alt.Tooltip("equity:Q", format="$,.0f")])
                    st.altair_chart(ch.properties(height=220), width="stretch")
                else:
                    st.caption("Equity curve appears after the trader has run a few minutes.")
                t = trades[trades.run == r["name"]]
                if len(t):
                    st.dataframe(t[["side", "entry_time", "entry_price", "exit_time", "exit_price", "reason",
                                    "pnl_usd", "pnl_pct"]].iloc[::-1].style.format(
                        {"entry_price": price, "exit_price": price, "pnl_usd": money, "pnl_pct": pct})
                        .map(color, subset=["pnl_usd", "pnl_pct"]), hide_index=True, width="stretch")
            with right:
                sig = store.table("signals", r["name"])
                st.code(sig.checklist.iloc[-1] if len(sig) else "waiting for the first closed candle", language=None)


def manual_controls(r: dict) -> None:
    """Force a long / short or close the position now; the trader does it on its next price check
    with this run's stop and take profits."""
    ex = exits_for(r)
    b1, b2, b3, info = st.columns([1, 1, 1, 3])
    for col, action, label in ((b1, "long", "🟢 Force LONG now"), (b2, "short", "🔴 Force SHORT now"),
                               (b3, "close", "✖ Close position now")):
        if col.button(label, key=f"cmd_{action}_{r['name']}"):
            store.command(r["name"], action)
            st.toast(f"{r['name']}: {action} sent - the trader does it on its next price check")
    note = f"Forced trades get this run's exits: {exit_text(ex)}."
    if r["asset"] == "crypto":
        on_webull = settings.get("broker") == "webull" and settings.get("webull_crypto", "auto") == "webull"
        note += (" ⚠️ Webull can't short spot crypto - Force SHORT won't work here." if on_webull else
                 " Crypto trades here go to the built-in paper broker (shorts work).")
    info.caption(note)
    q = store.db.execute("SELECT time FROM quotes WHERE run = ?", (r["name"],)).fetchone()
    poll = int(settings.get("poll_seconds") or 30)
    age = (datetime.now(timezone.utc) - pd.Timestamp(q[0]).tz_convert("UTC")).total_seconds() if q else None
    cmds = pd.read_sql_query("SELECT action, created, done, result FROM commands WHERE run = ? ORDER BY id DESC LIMIT 3",
                             store.db, params=(r["name"],))
    if age is None or age > max(3 * poll, 180):
        info.warning("⏸️ The trader isn't checking prices right now"
                     + (f" (last check {int(age // 60)} min ago)" if age else "") + " - forced trades wait until it runs. "
                     "Open Paper Trader and press **Start trading**.")
    for c in cmds.itertuples():
        done = isinstance(c.result, str) and c.result       # a waiting order's result is empty (NaN from the db)
        res = c.result if done else "⏳ waiting for the trader's next price check..."
        bad = bool(done) and any(w in c.result for w in ("refused", "not sent", "couldn't", "not filled"))
        (info.error if bad else info.caption)(f"{c.action} ({str(c.created)[11:16]} UTC): {res}")


@st.cache_data(ttl=120, show_spinner=False)
def live_candles(symbol: str, asset: str, tf: str) -> pd.DataFrame:
    return history(symbol, asset, tf).iloc[-600:]


def run_chart(r: dict, state) -> None:
    """Candles with this run's entries, take profits and stops, like TradingView's strategy tester."""
    try:
        candles = live_candles(r["symbol"], r["asset"], r["timeframe"])
    except Exception as e:
        st.caption(f"No chart: couldn't load {r['symbol']} candles ({e})")
        return
    fills = store.table("fills", r["name"])
    marks = tvchart.markers_from_fills(candles, fills.to_dict("records")) if len(fills) else []
    lines = []
    p = state.position if state else None
    if p is not None:
        ex, d = exits_for(r), 1 if p.side == "long" else -1
        lines.append({"price": p.entry_price, "title": f"{p.side.capitalize()} entry", "color": tvchart.BLUE})
        if ex.stop:
            lines.append({"price": p.entry_price * (1 - d * ex.stop), "title": "SL", "color": tvchart.RED})
        fixed = [lv for lv, _ in ex.take_profits if lv is not None]
        for k, lv in enumerate(fixed[p.targets_hit:], start=p.targets_hit + 1):
            lines.append({"price": p.entry_price * (1 + d * lv), "title": f"TP{k}", "color": tvchart.GREEN})
    tvchart.show(candles, marks, lines, height=440)


with tab_dash:
    dashboard()


# ---- Strategy Lab ---------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_history(symbol: str, asset: str, tf: str) -> pd.DataFrame:
    return history(symbol, asset, tf)


@st.cache_data(ttl=3600, show_spinner=False)
def backtest(strategy, symbol, asset, tf, days, stop, tps, shorts, fee, version=""):
    """version: a custom strategy's rules, so editing it doesn't reuse a stale cached result."""
    full = load_history(symbol, asset, tf)
    sig = REGISTRY[strategy]["signals"](full)
    start = full.index[-1] - pd.Timedelta(days=days) if days else full.index[min(200, len(full) - 1)]
    df, s = full[full.index >= start], sig[sig.index >= start]
    res = run_backtest(df, s, allow_short=shorts, stop_loss=stop, take_profits=list(tps), fee=fee)
    trades = pd.DataFrame([{"side": t.side, "entry": t.entry_time, "entry price": t.entry_price, "exit": t.exit_time,
                            "exit price": t.exit_price, "reason": t.reason, "return": t.pnl_pct * 100}
                           for t in res.trades])
    bh = df.close / df.close.iloc[0] * res.equity.iloc[0]
    candles = df[["open", "high", "low", "close"]]
    return res.stats, res.equity, bh, trades, candles, tvchart.markers_from_trades(candles, res.trades)


def toggle_run(run: dict, key: str):
    if st.session_state[key]:
        store.upsert_run({**run, "active": 1, "source": "dashboard"})
    else:
        store.set_active(run["name"], False)


with tab_lab:
    st.caption("Backtest any strategy on any crypto or stock, then tick **Paper trade this setup** to trade it "
               "live in the paper account. Real Kraken / Yahoo candles, signals on closed candles, fills at the "
               "next open.")
    keys = list(REGISTRY)
    c1, c2 = st.columns([3, 2])
    strategy = c1.selectbox("Strategy", keys, format_func=lambda k: REGISTRY[k]["label"])
    market = c2.radio("Market", ["Crypto", "Stocks"], horizontal=True)
    asset = "crypto" if market == "Crypto" else "stock"
    c3, c4, c5 = st.columns([2, 1, 1.4])
    if asset == "crypto":
        symbol = c3.selectbox("Coin", list(CRYPTO))
    else:
        names = stocks.labels()
        pick = c3.selectbox(f"Stock ({len(names):,} - type to search by ticker or name)", list(names) + [OTHER],
                            format_func=lambda t: names.get(t, t))
        symbol = c3.text_input("Ticker", "", placeholder="any Yahoo Finance ticker, e.g. SHOP") if pick == OTHER else pick
    tf = c4.selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index("4h"))
    minute_chart = tf.endswith("m")
    periods = INTRADAY_PERIODS if minute_chart else PERIODS
    period = c5.selectbox("Period", list(periods), index=2 if minute_chart else 1, key=f"period_{minute_chart}")
    if minute_chart:
        st.caption("⏱️ Minute charts: crypto reaches back weeks (Coinbase), stocks 7 days at 1m and 60 days at 5m-30m "
                   "(Yahoo). Fees weigh much more on small moves, and these strategies were tested on 2h to daily "
                   "charts - expect different results.")
    base = REGISTRY[strategy]["exits"]
    long_only = REGISTRY[strategy].get("long_only", False)
    with st.expander("Exits & options", expanded=True):
        e1, e2, e3 = st.columns(3)
        stop_pct = e1.number_input("Stop %", 0.0, 50.0, (base.stop or 0) * 100, 0.5, key=f"stop_{strategy}",
                                   help="0 = no stop (paper trading still caps it - see Account). "
                                        "Defaults to the strategy's tested setting.")
        fee_pct = e2.number_input("Fee % per side", 0.0, 1.0, 0.10, 0.01)
        capital = e3.number_input("Paper capital $", 100.0, 1e7, 10_000.0, 1000.0)
        t1, t2 = st.columns([2, 3])
        tp_mode = t1.radio("Targets set as", ["% move", "Risk:reward (× the stop)"], horizontal=True,
                           help="Risk:reward: 2R = twice the stop distance, e.g. a 4% stop makes 2R = +8%.")
        r_mode = tp_mode.startswith("Risk")
        preset = t2.selectbox("Take-profit plan", list(TP_PLANS))
        rows = tp_rows(preset, r_mode, base, stop_pct / 100)
        edited = st.data_editor(
            pd.DataFrame(rows), hide_index=True, width="stretch", num_rows="dynamic",
            key=f"tp_{strategy}_{preset}_{r_mode}",
            column_config={
                "Target": st.column_config.TextColumn(disabled=True),
                "At": st.column_config.NumberColumn("Take profit at (×R)" if r_mode else "Take profit at (+%)",
                                                    min_value=0.01, step=0.1, format="%.2f",
                                                    help="Leave empty on the last row to let the rest ride until "
                                                         "the exit signal"),
                "Close %": st.column_config.NumberColumn("Close (% of position)", min_value=1, max_value=100, step=1,
                                                         format="%d%%")})
        try:
            tps = tps_from_table(edited, r_mode, stop_pct / 100)
            st.caption("Plan: " + plan_text(tps, stop_pct / 100))
            tp_error = None
        except ValueError as e:
            tps, tp_error = None, str(e)
            st.error(tp_error)
        shorts = st.checkbox("Allow shorts", value=False, disabled=long_only,
                             help="Spot crypto can't be shorted in a real account; stocks can.")
    st.caption(REGISTRY[strategy]["note"])
    go = st.button("Run backtest", type="primary", disabled=tps is None or not symbol)

    if go:
        try:
            tps = tuple(tps)
            with st.spinner("Loading candles and backtesting..."):
                out = backtest(strategy, symbol.upper(), asset, tf, periods[period], stop_pct / 100 or None, tps,
                               shorts and not long_only, fee_pct / 100,
                               json.dumps(REGISTRY[strategy].get("spec", ""), sort_keys=True))
            st.session_state["lab_result"] = {"strategy": strategy, "symbol": symbol.upper(), "asset": asset, "tf": tf,
                                       "period": period, "stop": stop_pct / 100, "tps": tps,
                                       "shorts": shorts and not long_only, "capital": capital, "out": out}
        except Exception as e:
            st.error(f"Backtest failed: {e}")

    lab = st.session_state.get("lab_result")
    if lab:
        stats, equity, bh, trades, candles, marks = lab["out"]
        spec = REGISTRY[lab["strategy"]]
        st.subheader(f"{spec['label']} · {lab['symbol']} {lab['tf']} · {lab['period'].lower()}")
        m = st.columns(6)
        m[0].metric("Strategy return", pct(stats["total_return_pct"]))
        m[1].metric("Buy & hold", pct(stats["buy_hold_return_pct"]))
        m[2].metric("Trades", stats["trades"])
        m[3].metric("Win rate", f"{stats['win_rate_pct']:.0f}%")
        m[4].metric("Max drawdown", f"{stats['max_drawdown_pct']:.1f}%")
        m[5].metric("Profit factor", f"{stats['profit_factor']:.2f}")
        verdict = "🟢 beat buy & hold" if stats["total_return_pct"] > stats["buy_hold_return_pct"] else \
            "🟡 made money but trailed buy & hold" if stats["total_return_pct"] > 0 else "🔴 lost money"
        st.caption(f"{verdict} over this period. A good backtest is a reason to paper trade, not proof.")
        tvchart.show(candles, marks, height=480)
        curve = pd.concat([pd.DataFrame({"time": equity.index, "value": equity.values, "line": "Strategy"}),
                           pd.DataFrame({"time": bh.index, "value": bh.values, "line": "Buy & hold"})])
        ch = alt.Chart(curve).mark_line().encode(
            x=alt.X("time:T", title=None), y=alt.Y("value:Q", title="Account ($)", scale=alt.Scale(zero=False)),
            color=alt.Color("line:N", title=None, scale=alt.Scale(domain=["Strategy", "Buy & hold"],
                                                                range=["#2563eb", "#9ca3af"])),
            tooltip=["line:N", "time:T", alt.Tooltip("value:Q", format="$,.0f")])
        st.altair_chart(ch.properties(height=300), width="stretch")
        with st.expander("Year by year"):
            yearly = pd.DataFrame({"Strategy": equity.resample("YE").last().pct_change().fillna(
                equity.resample("YE").last().iloc[0] / equity.iloc[0] - 1) * 100,
                                   "Buy & hold": bh.resample("YE").last().pct_change().fillna(
                                       bh.resample("YE").last().iloc[0] / bh.iloc[0] - 1) * 100})
            yearly.index = pd.Index(yearly.index.year, name="Year")
            st.dataframe(yearly.style.format(pct).map(color), width="stretch")
        with st.expander(f"Trades ({len(trades)})"):
            if len(trades):
                st.dataframe(trades.iloc[::-1].style.format({"entry price": price, "exit price": price,
                                                             "return": pct}).map(color, subset=["return"]),
                             hide_index=True, width="stretch")

        run = {"name": f"{spec['short']} {lab['symbol']} {lab['tf']}", "strategy": lab["strategy"],
               "symbol": CRYPTO.get(lab["symbol"], lab["symbol"]), "asset": lab["asset"], "timeframe": lab["tf"],
               "shorts": lab["shorts"], "capital": lab["capital"], "fraction": 1.0,
               "stop": lab["stop"], "take_profits": [list(t) for t in lab["tps"]]}
        active = {r["name"] for r in store.runs(active_only=True)}
        key = f"pt::{run['name']}"
        st.checkbox(f"📄 Paper trade this setup as **{run['name']}** ({money(lab['capital'])} paper capital)",
                    value=run["name"] in active, key=key, on_change=toggle_run, args=(run, key))
        st.caption("Takes effect on the trader's next price check. Unticking closes any open paper position.")
        cap = max_stop()
        if not lab["stop"] or lab["stop"] > cap:
            st.warning(f"🛡️ Paper trading always uses a stop: this setup trades with a {cap:.0%} stop (the safety cap "
                       f"in ⚙️ Account), so live results will differ from the backtest above.")

    with st.expander("Manage paper-trading runs"):
        all_runs = store.runs()
        if not all_runs:
            st.caption("None yet.")
        for r in all_runs:
            key = f"mg::{r['name']}"
            st.checkbox(f"{r['name']} · {REGISTRY.get(r['strategy'], {}).get('label', r['strategy'])} · "
                        f"{r['symbol']} {r['timeframe']} · from {r['source']}", value=r["active"], key=key,
                        on_change=lambda name=r["name"], k=key: store.set_active(name, st.session_state[k]))

    with st.expander("Strategies that can't be paper traded"):
        for name, why in NOT_LIVE.items():
            st.markdown(f"- **{name}** - {why}")


# ---- Activity -------------------------------------------------------------------
with tab_log:
    ev, fills = store.table("events"), store.table("fills")
    with st.expander(f"Events ({len(ev)})", expanded=True):
        ev["time"] = pd.to_datetime(ev.time, format="ISO8601", utc=True).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(ev.iloc[::-1].head(300), hide_index=True, width="stretch")
    with st.expander(f"Fills ({len(fills)})"):
        if len(fills):
            fills["time"] = pd.to_datetime(fills.time, format="ISO8601", utc=True).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(fills.iloc[::-1].style.format({"price": price, "fee_usd": money}),
                     hide_index=True, width="stretch")


# ---- Streams (the stream study agent) -------------------------------------------
# ---- Builder: your own strategies from rules ---------------------------------------------------
NEW = "➕ New strategy"
DEFAULT_SPEC = {"name": "My strategy", "long": [{"rule": "ema_cross_up", "a": 9, "b": 21}], "short": [],
                "exit_long": [{"rule": "ema_cross_down", "a": 9, "b": 21}], "exit_short": [],
                "stop": 0.04, "take_profits": [[0.08, 0.5], [0.16, 0.5]]}


def rules_editor(title: str, rules: list[dict], key: str) -> list[dict]:
    st.markdown(title)
    df = pd.DataFrame([{"Rule": builder.LABELS[r["rule"]], "A": r.get("a"), "B": r.get("b")} for r in rules] or
                      [{"Rule": None, "A": None, "B": None}][:0], columns=["Rule", "A", "B"])
    ed = st.data_editor(df, num_rows="dynamic", hide_index=True, width="stretch", key=key, column_config={
        "Rule": st.column_config.SelectboxColumn("Rule", options=list(builder.LABELS.values()), width="large",
                                                 help="A and B are the numbers in the rule's name"),
        "A": st.column_config.NumberColumn("A", step=0.5, help="Empty = the rule's usual value"),
        "B": st.column_config.NumberColumn("B", step=0.5, help="Empty = the rule's usual value")})
    out = []
    for _, row in ed.iterrows():
        if row["Rule"] in builder.BY_LABEL:
            rid = builder.BY_LABEL[row["Rule"]]
            out.append({"rule": rid, "a": None if pd.isna(row["A"]) else float(row["A"]),
                        "b": None if pd.isna(row["B"]) else float(row["B"])})
    for i, r in enumerate(out):                     # what A / B mean, and the rule's ⚙️ settings
        old = rules[i] if i < len(rules) and rules[i].get("rule") == r["rule"] else {}
        specs = builder.SETTINGS.get(r["rule"], [])
        if not specs:
            st.caption(f"{i + 1}. {builder.LABELS[r['rule']].split(' - ')[0]}: {builder.ab_help(r['rule'])}")
            continue
        with st.expander(f"⚙️ {i + 1}. {builder.LABELS[r['rule']].split(' - ')[0]} - settings · "
                         f"{builder.ab_help(r['rule'])}"):
            chosen = {}
            cols = st.columns(3)
            for j, spec in enumerate(specs):
                name, label, default, kind = spec[:4]
                cur = (old.get("set") or {}).get(name, default)
                wkey = f"{key}_set_{i}_{r['rule']}_{name}"
                c = cols[j % 3]
                if kind == "choice":
                    opts = spec[4]
                    val = c.selectbox(label, opts, index=opts.index(cur) if cur in opts else 0, key=wkey,
                                      format_func=lambda v: {"turns": "Oscillator turns (AO colour change, MACD / "
                                                                      "RSI crossing their signal)",
                                                             "pivots": "Pivots (TradingView's divergence indicator)"
                                                             }.get(v, v))
                elif kind == "bool":
                    val = c.checkbox(label, bool(cur), key=wkey)
                elif kind == "int":
                    val = int(c.number_input(label, int(spec[4]), int(spec[5]), int(cur), 1, key=wkey))
                else:
                    val = float(c.number_input(label, float(spec[4]), float(spec[5]), float(cur), 0.1, key=wkey))
                if val != default:
                    chosen[name] = val
            if chosen:
                r["set"] = chosen
            if r["rule"].endswith("_div"):
                st.caption("Pivots: a low is the oscillator's lowest point with *left* candles before and *right* "
                           "after (so it's confirmed *right* candles later); B = Max of lookback range. Turns: the "
                           "low between the oscillator turning down and turning back up, confirmed when it turns up.")
    if out:
        st.caption(" AND ".join(builder.describe(r) for r in out) if "ANY" not in title else
                   " OR ".join(builder.describe(r) for r in out))
    return out


with tab_build:
    st.caption("Make your own strategy from rules - no code. Saved strategies show up in 🧪 Strategy Lab (backtest, "
               "chart, paper trade) with a checklist of each rule. Rules use closed candles only.")
    data = custom.load()
    keys = list(data["strategies"])
    from livepaper.strategies import S10_SPEC
    COPY_S10 = "📋 Copy of S10 1m Bollinger + AO divergence"
    choice = st.selectbox("Strategy to edit", [NEW, COPY_S10] + keys,
                          format_func=lambda k: k if k in (NEW, COPY_S10) else data["strategies"][k].get("name", k))
    if choice == COPY_S10:
        import copy as _copy
        spec = {**_copy.deepcopy(S10_SPEC), "name": "S10 (my settings)"}
    else:
        spec = dict(DEFAULT_SPEC if choice == NEW else data["strategies"][choice])
    k = choice.replace(" ", "_")
    name = st.text_input("Name", spec.get("name", ""), key=f"b_name_{k}")
    long_rules = rules_editor("**Go LONG when ALL of these are true**", spec.get("long", []), f"b_long_{k}")
    short_rules = rules_editor("**Go SHORT when ALL of these are true** (optional - leave empty for long only)",
                               spec.get("short", []), f"b_short_{k}")
    with st.expander("Exit early (optional) - otherwise trades end at the stop, the take profits, or the opposite "
                     "entry signal"):
        exit_long = rules_editor("Close a long when ANY of these is true", spec.get("exit_long", []), f"b_xl_{k}")
        exit_short = rules_editor("Close a short when ANY of these is true", spec.get("exit_short", []), f"b_xs_{k}")
    x1, x2 = st.columns([1, 3])
    b_stop = x1.number_input("Stop %", 0.0, 50.0, float(spec.get("stop") or 0) * 100, 0.5, key=f"b_stop_{k}")
    plan = x2.selectbox("Take-profit plan", list(TP_PLANS), key=f"b_plan_{k}",
                        index=list(TP_PLANS).index("2 targets: 50 / 50") if choice == NEW else 0)
    from livepaper.core import Exits as _Exits
    own = _Exits(spec.get("stop"), [tuple(t) for t in spec.get("take_profits") or [[0.08, 0.5], [0.16, 0.5]]])
    tp_df = st.data_editor(pd.DataFrame(tp_rows(plan, False, own, b_stop / 100)), hide_index=True, width="stretch",
                           num_rows="dynamic", key=f"b_tp_{k}_{plan}", column_config={
                               "Target": st.column_config.TextColumn(disabled=True),
                               "At": st.column_config.NumberColumn("Take profit at (+%)", min_value=0.01, step=0.5),
                               "Close %": st.column_config.NumberColumn("Close (% of position)", min_value=1,
                                                                        max_value=100, step=1)})
    try:
        b_tps = tps_from_table(tp_df, False, b_stop / 100)
        st.caption("Plan: " + plan_text(b_tps, b_stop / 100))
    except ValueError as e:
        b_tps = None
        st.error(str(e))
    new_spec = {"name": name.strip() or "My strategy", "long": long_rules, "short": short_rules,
                "exit_long": exit_long, "exit_short": exit_short, "stop": b_stop / 100 or None,
                "take_profits": [list(t) for t in b_tps or []]}
    ok = bool(long_rules or short_rules) and b_tps is not None
    if not (long_rules or short_rules):
        st.warning("Add at least one rule for longs or shorts.")
    q1, q2, q3, q4 = st.columns([1.2, 1.2, 1.4, 3])
    q_coin = q1.selectbox("Quick test on", list(CRYPTO), key="b_coin")
    q_tf = q2.selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index("4h"), key="b_tf")
    if q3.button("🧪 Quick test", disabled=not ok):
        with st.spinner("Backtesting..."):
            full = load_history(CRYPTO[q_coin], "crypto", q_tf)
            minutes = q_tf.endswith("m")
            start = full.index[-1] - pd.Timedelta(days=30 if minutes else 3 * 365)
            df_ = full[full.index >= start]
            sig_ = builder.signals(full, new_spec)[full.index >= start]
            res = run_backtest(df_, sig_, allow_short=bool(short_rules), stop_loss=new_spec["stop"],
                               take_profits=[tuple(t) for t in new_spec["take_profits"]], fee=0.001)
        stt = res.stats
        m = st.columns(5)
        m[0].metric("Return", pct(stt["total_return_pct"]))
        m[1].metric("Buy & hold", pct(stt["buy_hold_return_pct"]))
        m[2].metric("Trades", stt["trades"])
        m[3].metric("Win rate", f"{stt['win_rate_pct']:.0f}%" if stt["trades"] else "-")
        m[4].metric("Max drawdown", f"{stt['max_drawdown_pct']:.1f}%")
        cndl = df_[["open", "high", "low", "close"]]
        tvchart.show(cndl, tvchart.markers_from_trades(cndl, res.trades), height=420)
        st.caption(f"{q_coin} {q_tf}, last {'30 days' if minutes else '3 years'}, 0.1% fee per side. Save it to "
                   "backtest other markets and paper trade it in 🧪 Strategy Lab.")
    s1, s2, _ = st.columns([1, 1, 3])
    if s1.button("💾 Save strategy", type="primary", disabled=not ok):
        key = custom.save_strategy(new_spec, None if choice in (NEW, COPY_S10) else choice)
        refresh_strategies()
        st.success(f"Saved **{new_spec['name']}** - pick it in 🧪 Strategy Lab.")
    if choice not in (NEW, COPY_S10) and s2.button("🗑️ Delete"):
        custom.delete_strategy(choice)
        refresh_strategies()
        st.rerun()
    with st.expander("✏️ Rename any strategy (built-in ones too)"):
        r1, r2, r3 = st.columns([2, 2, 1])
        target = r1.selectbox("Strategy", list(REGISTRY), format_func=lambda x: REGISTRY[x]["label"], key="rn_t")
        new_name = r2.text_input("New name (empty = original name)", key=f"rn_n_{target}")
        r3.write("")
        if r3.button("Rename", key="rn_go"):
            custom.rename(target, new_name)
            refresh_strategies()
            st.rerun()


with tab_streams:
    from streamlearn.dashboard import render as render_streams
    render_streams(settings.get("streamlearn_db", "streamlearn.db"))

with tab_app:
    try:
        from apprentice.ui import render as render_apprentice
        render_apprentice(store)
    except Exception as e:
        st.error(f"The apprentice couldn't start: {type(e).__name__}: {e}")

with tab_prove:
    try:
        from apprentice.proving_ui import render as render_proving
        render_proving(store)
    except Exception as e:
        st.error(f"The proving ground couldn't start: {type(e).__name__}: {e}")


# ---- Account ----------------------------------------------------------------------
def test_webull_stops() -> list[str]:
    """Place and cancel a tiny buy-stop far above the price, for a stock and for crypto: shows whether
    Webull accepts the stop orders that protect trades while the computer is off. Nothing fills."""
    from livepaper.brokers import WebullBroker, api_client
    b = WebullBroker(api_client(), {"stock": ("SPY", "stock"), "crypto": ("XBTUSD", "crypto")}, fee=0)
    out = []
    for kind, label in (("stock", "Stocks (SPY)"), ("crypto", "Crypto (BTC)")):
        try:
            px = float(history(*(("SPY", "stock") if kind == "stock" else ("XBTUSD", "crypto")), "1D").close.iloc[-1])
            qty = 1 if kind == "stock" else round(20 / px, 6)
            coid = b.place_stop(kind, "buy", qty, px * 1.5)
            if coid is None:
                out.append(f"🔴 {label}: Webull refused stop orders - those trades are protected only while "
                           "the trader runs (details in the trader log)")
                continue
            b.cancel_stop(kind, coid, qty, px * 1.5)
            out.append(f"🟢 {label}: stop order accepted (test order at {price(px * 1.5)}, cancelled) - the stop works "
                       "with this computer off")
        except Exception as e:
            out.append(f"🔴 {label}: {type(e).__name__}: {e}")
    return out


def test_webull() -> str:
    try:
        from livepaper.brokers import _find, api_client
        from webull.trade.trade_client import TradeClient
    except ImportError:
        return "❌ Webull's SDK isn't installed: pip install webull-openapi-python-sdk"
    try:
        res = TradeClient(api_client()).account_v2.get_account_list()
    except Exception as e:
        return f"❌ {e}"
    if res.status_code != 200:
        return f"❌ Webull answered HTTP {res.status_code}: {res.text[:300]}"
    acc = _find(res.json(), "account_id", "accountId")
    return f"✅ Connected. Paper account found: {acc}" if acc else f"⚠️ Connected, but no account id in: {res.text[:300]}"


with tab_acct:
    cur = settings.load()
    st.caption(f"Saved on this computer in `{settings.PATH.name}` (plain text, paper trading only; never pushed to "
               f"GitHub). Restart the trader (`python -m livepaper run`) after changing the broker or keys.")
    with st.form("account"):
        with st.expander("Paper trading", expanded=True):
            a1, a2, a3 = st.columns(3)
            broker = a1.radio("Broker", ["sim", "webull"], index=0 if cur["broker"] == "sim" else 1,
                              format_func=lambda b: "Built-in paper broker" if b == "sim" else "Webull paper account")
            fee = a2.number_input("Fee % per side (for P&L)", 0.0, 1.0, float(cur["fee"]) * 100, 0.01)
            poll = a3.number_input("Check prices every (seconds)", 10, 600, int(cur["poll_seconds"]), 5)
            b1, b2 = st.columns(2)
            sync_git = b1.checkbox("Share with Claude: trades, the apprentice's setups / concepts / lessons, stream "
                                   "results (and bring Claude's reviews back)", bool(cur["sync_git"]))
            sync_min = b2.number_input("Share every (minutes)", 5, 1440, int(cur["sync_minutes"]), 5)
            gh_token = st.text_input(
                "GitHub token (needed to share from this ZIP install)", cur.get("github_token", ""), type="password",
                help="github.com → Settings → Developer settings → Personal access tokens → Fine-grained tokens → "
                     "Generate new token. Repository access: Only select repositories → Claude. Permissions → "
                     "Contents: Read and write. It can't touch anything else in your account.")
            cap_pct = st.number_input("🛡️ Maximum stop loss % (safety cap)", 1.0, 50.0, float(cur["max_stop"]) * 100, 0.5,
                                      help="Every paper trade gets a stop, never wider than this - even if a run has "
                                           "its stop turned off or set wider. With Webull, the stop is placed as an "
                                           "order at Webull so it works with this computer off.")
        with st.expander("Webull paper account", expanded=cur["broker"] == "webull"):
            st.caption("Webull Developer Center → *Using OpenAPI service in Paper Trading* → API Management → "
                       "My Application. Use the **sandbox (paper)** keys.")
            w1, w2 = st.columns(2)
            key = w1.text_input("App key", cur["webull_app_key"])
            secret = w2.text_input("App secret", cur["webull_app_secret"], type="password")
            w3, w4, w5 = st.columns([2, 1, 2])
            account = w3.text_input("Account ID (optional - found automatically)", cur["webull_account_id"])
            region = w4.text_input("Region", cur["webull_region"])
            endpoint = w5.text_input("API host", cur["webull_endpoint"],
                                     help="api.sandbox.webull.com = paper trading")
            crypto_modes = {"auto": "Automatic - Webull, or the built-in paper broker if Webull refuses crypto",
                            "sim": "Built-in paper broker (live Kraken prices)", "webull": "Webull"}
            webull_crypto = st.radio("Crypto runs trade in", list(crypto_modes),
                                     index=list(crypto_modes).index(cur.get("webull_crypto", "auto"))
                                     if cur.get("webull_crypto", "auto") in crypto_modes else 0,
                                     format_func=crypto_modes.get,
                                     help="Many Webull paper accounts can't trade crypto ('The current account does "
                                          "not support crypto trading'). Stock runs always go to Webull.")
        with st.expander("Stream study agent"):
            s1, s2 = st.columns(2)
            backend = s1.radio("Extraction model", ["ollama", "claude"], horizontal=True,
                               index=0 if cur["stream_backend"] == "ollama" else 1,
                               format_func=lambda b: "Local (Ollama)" if b == "ollama" else "Claude (API)")
            tz = s2.text_input("Your timezone (for stream start times)", cur["timezone"])
            s3, s4 = st.columns(2)
            ollama_model = s3.text_input("Ollama model", cur["ollama_model"],
                                         help="auto = sized for this computer's memory and set up by itself "
                                              "(qwen2.5:3b / 7b / 14b)")
            ollama_url = s4.text_input("Ollama address", cur["ollama_url"])
            anthropic_key = st.text_input("Anthropic API key (only for Claude)", cur["anthropic_api_key"],
                                          type="password")
            v1, v2 = st.columns(2)
            screen_on = v1.checkbox("Read his screen too (trades he draws on the chart but doesn't say)",
                                    bool(cur.get("stream_screen", True)),
                                    help="Looks at the stream video for TradingView's long/short position drawing "
                                         "(green profit box, red stop box) and reads the entry, stop and target "
                                         "prices off the chart. Needs the stream's YouTube link or video file.")
            vision_model = v2.text_input("Screen-reading model (local)", cur.get("ollama_vision_model", "auto"),
                                         help="auto = qwen2.5vl:3b / 7b sized for this computer; Claude reads "
                                              "charts much better if you use the Claude option")
            sl_db = st.text_input("Stream study database", cur["streamlearn_db"])
            ap_db = st.text_input("Apprentice database (what the 🧠 Apprentice has learned)",
                                  cur.get("apprentice_db", "apprentice.db"))
        if st.form_submit_button("Save settings", type="primary"):
            settings.save({"github_token": gh_token.strip(), "broker": broker, "fee": fee / 100, "max_stop": cap_pct / 100, "poll_seconds": int(poll), "sync_git": sync_git,
                           "sync_minutes": int(sync_min), "webull_app_key": key.strip(),
                           "webull_app_secret": secret.strip(), "webull_account_id": account.strip(),
                           "webull_region": region.strip() or "us", "webull_endpoint": endpoint.strip(),
                           "webull_crypto": webull_crypto,
                           "stream_backend": backend, "timezone": tz.strip(), "ollama_model": ollama_model.strip(),
                           "ollama_url": ollama_url.strip(), "anthropic_api_key": anthropic_key.strip(),
                           "stream_screen": screen_on, "ollama_vision_model": vision_model.strip() or "auto",
                           "streamlearn_db": sl_db.strip() or "streamlearn.db",
                           "apprentice_db": ap_db.strip() or "apprentice.db"})
            st.success("Saved. The trader switches to these settings on its next price check - no restart needed.")
    c1, c2 = st.columns([1, 3])
    if c1.button("Test Webull connection"):
        c2.write(test_webull())
    from livepaper import sync as _sync
    s1, s2 = st.columns([1, 3])
    if s1.button("🔄 Sync with Claude now", help="Share everything now and download Claude's latest review"):
        with st.spinner("Sharing..."):
            s2.write(_sync.sync_all(store))
    last = _sync.last_status()
    if last:
        s2.caption(f"Last share {last.get('at', '')[:16]} UTC: {last.get('result', '')}"
                   + (f" · Claude's review from {last['review'][:16]}" if last.get("review") else " · no review yet"))
    c3, c4 = st.columns([1, 3])
    if c3.button("Test stop orders", help="Places and cancels a test stop far from the price - nothing fills"):
        for line in test_webull_stops():
            c4.write(line)
    with st.expander("How to switch to Webull paper trading"):
        st.markdown(
            "1. **Get paper (sandbox) API keys** in Webull's Developer Center (developer.webull.com): sign in with your "
            "Webull account, open API Management → My Application, and create/apply for an application for "
            "**paper trading**. Approval can take a little while.\n"
            "2. Paste the **App key** and **App secret** above (Webull paper account section). Leave the API host as "
            "`api.sandbox.webull.com` - that's the paper (sandbox) server, not real money.\n"
            "3. Set **Broker** to **Webull paper account** and press **Save settings**.\n"
            "4. Press **Test Webull connection** (finds your paper account) and **Test stop orders** (checks the "
            "safety stops work).\n"
            "5. That's it: from the trader's next price check, every order goes to your Webull paper account. "
            "Positions from the built-in paper broker are closed on the books (they don't exist at Webull); "
            "their history stays.")
    st.caption("Webull keys: " + (f"set ({settings.mask(settings.get('webull_app_key'))})"
                                  if settings.get("webull_app_key") else "not set") +
               " · Anthropic key: " + (f"set ({settings.mask(settings.get('anthropic_api_key'))})"
                                       if settings.get("anthropic_api_key") else "not set"))
