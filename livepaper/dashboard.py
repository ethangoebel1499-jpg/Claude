"""livepaper dashboard. Start it with:  python -m livepaper dashboard"""

from __future__ import annotations

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
from livepaper.store import Store  # noqa: E402
from livepaper.strategies import NOT_LIVE, REGISTRY  # noqa: E402
from mcbot.backtest import run_backtest  # noqa: E402

DB = os.environ.get("LIVEPAPER_DB", "livepaper.db")
CONFIG = os.environ.get("LIVEPAPER_CONFIG", "live.toml")
TIMEFRAMES = ["1h", "2h", "4h", "6h", "8h", "12h", "1D", "2D"]
PERIODS = {"Last year": 365, "Last 3 years": 3 * 365, "All history": None}

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
cfg = load_config(CONFIG) if Path(CONFIG).exists() else {"run": [], "app": {}}
seed_runs(cfg, store)

st.title("📈 Paper Trading")
tab_dash, tab_lab, tab_streams, tab_log, tab_acct = st.tabs(
    ["📊 Dashboard", "🧪 Strategy Lab", "🎙️ Streams", "📜 Activity", "⚙️ Account"])


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


with tab_dash:
    dashboard()


# ---- Strategy Lab ---------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_history(symbol: str, asset: str, tf: str) -> pd.DataFrame:
    return history(symbol, asset, tf)


@st.cache_data(ttl=3600, show_spinner=False)
def backtest(strategy, symbol, asset, tf, days, stop, tps, shorts, fee):
    full = load_history(symbol, asset, tf)
    sig = REGISTRY[strategy]["signals"](full)
    start = full.index[-1] - pd.Timedelta(days=days) if days else full.index[min(200, len(full) - 1)]
    df, s = full[full.index >= start], sig[sig.index >= start]
    res = run_backtest(df, s, allow_short=shorts, stop_loss=stop, take_profits=list(tps), fee=fee)
    trades = pd.DataFrame([{"side": t.side, "entry": t.entry_time, "entry price": t.entry_price, "exit": t.exit_time,
                            "exit price": t.exit_price, "reason": t.reason, "return": t.pnl_pct * 100}
                           for t in res.trades])
    bh = df.close / df.close.iloc[0] * res.equity.iloc[0]
    return res.stats, res.equity, bh, trades


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
    with st.form("lab_form"):
        c3, c4, c5 = st.columns([2, 1, 1.4])
        symbol = c3.selectbox("Coin", list(CRYPTO)) if asset == "crypto" else \
            c3.text_input("Ticker", "SPY", help="Any Yahoo Finance ticker, e.g. " + ", ".join(STOCKS[:8]))
        tf = c4.selectbox("Timeframe", TIMEFRAMES, index=2)
        period = c5.selectbox("Period", list(PERIODS), index=1)
        base = REGISTRY[strategy]["exits"]
        with st.expander("Exits & options"):
            e1, e2, e3, e4 = st.columns([1, 2, 1, 1])
            stop_pct = e1.number_input("Stop %", 0.0, 50.0, (base.stop or 0) * 100, 0.5,
                                       help="0 = no stop. Defaults to the strategy's tested setting.")
            tps_in = e2.text_input("Take profits (move % : size %)", tps_text(base.take_profits),
                                   help="e.g. 12:50, 24:50  or  2:50, 5:25, ride:25 (ride = runner)")
            fee_pct = e3.number_input("Fee % per side", 0.0, 1.0, 0.10, 0.01)
            capital = e4.number_input("Paper capital $", 100.0, 1e7, 10_000.0, 1000.0)
            long_only = REGISTRY[strategy].get("long_only", False)
            shorts = st.checkbox("Allow shorts", value=False, disabled=long_only,
                                 help="Spot crypto can't be shorted in a real account; stocks can.")
        st.caption(REGISTRY[strategy]["note"])
        go = st.form_submit_button("Run backtest", type="primary")

    if go:
        try:
            tps = tuple(parse_tps(tps_in))
            with st.spinner("Loading candles and backtesting..."):
                out = backtest(strategy, symbol.upper(), asset, tf, PERIODS[period], stop_pct / 100 or None, tps,
                               shorts and not long_only, fee_pct / 100)
            st.session_state["lab_result"] = {"strategy": strategy, "symbol": symbol.upper(), "asset": asset, "tf": tf,
                                       "period": period, "stop": stop_pct / 100, "tps": tps,
                                       "shorts": shorts and not long_only, "capital": capital, "out": out}
        except Exception as e:
            st.error(f"Backtest failed: {e}")

    lab = st.session_state.get("lab_result")
    if lab:
        stats, equity, bh, trades = lab["out"]
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
with tab_streams:
    from streamlearn.dashboard import render as render_streams
    render_streams(settings.get("streamlearn_db", "streamlearn.db"))


# ---- Account ----------------------------------------------------------------------
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
            sync_git = b1.checkbox("Share trades with Claude (push to the paper-trades branch)", bool(cur["sync_git"]))
            sync_min = b2.number_input("Share every (minutes)", 5, 1440, int(cur["sync_minutes"]), 5)
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
        with st.expander("Stream study agent"):
            s1, s2 = st.columns(2)
            backend = s1.radio("Extraction model", ["ollama", "claude"], horizontal=True,
                               index=0 if cur["stream_backend"] == "ollama" else 1,
                               format_func=lambda b: "Local (Ollama)" if b == "ollama" else "Claude (API)")
            tz = s2.text_input("Your timezone (for stream start times)", cur["timezone"])
            s3, s4 = st.columns(2)
            ollama_model = s3.text_input("Ollama model", cur["ollama_model"])
            ollama_url = s4.text_input("Ollama address", cur["ollama_url"])
            anthropic_key = st.text_input("Anthropic API key (only for Claude)", cur["anthropic_api_key"],
                                          type="password")
            sl_db = st.text_input("Stream study database", cur["streamlearn_db"])
        if st.form_submit_button("Save settings", type="primary"):
            settings.save({"broker": broker, "fee": fee / 100, "max_stop": cap_pct / 100, "poll_seconds": int(poll), "sync_git": sync_git,
                           "sync_minutes": int(sync_min), "webull_app_key": key.strip(),
                           "webull_app_secret": secret.strip(), "webull_account_id": account.strip(),
                           "webull_region": region.strip() or "us", "webull_endpoint": endpoint.strip(),
                           "stream_backend": backend, "timezone": tz.strip(), "ollama_model": ollama_model.strip(),
                           "ollama_url": ollama_url.strip(), "anthropic_api_key": anthropic_key.strip(),
                           "streamlearn_db": sl_db.strip() or "streamlearn.db"})
            st.success("Saved.")
    c1, c2 = st.columns([1, 3])
    if c1.button("Test Webull connection"):
        c2.write(test_webull())
    st.caption("Webull keys: " + (f"set ({settings.mask(settings.get('webull_app_key'))})"
                                  if settings.get("webull_app_key") else "not set") +
               " · Anthropic key: " + (f"set ({settings.mask(settings.get('anthropic_api_key'))})"
                                       if settings.get("anthropic_api_key") else "not set"))
