"""livepaper dashboard. Start it with:  python -m livepaper dashboard"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from livepaper.runner import exits_for, load_config  # noqa: E402
from livepaper.store import Store  # noqa: E402

DB = os.environ.get("LIVEPAPER_DB", "livepaper.db")
CONFIG = os.environ.get("LIVEPAPER_CONFIG", "live.toml")

st.set_page_config(page_title="Paper Trading", page_icon="📈", layout="wide")


def exit_text(r: dict) -> str:
    ex = exits_for(r)
    tps = ", ".join(f"{s:.0%} at +{lv:.1%}" if lv is not None else f"{s:.0%} rides" for lv, s in ex.take_profits)
    return f"stop {ex.stop:.0%}; {tps}" if ex.stop else f"no stop; {tps}"


@st.fragment(run_every=30)
def page():
    store = Store(DB)
    cfg = load_config(CONFIG) if Path(CONFIG).exists() else {"run": []}
    runs = {r["name"]: r for r in cfg.get("run", [])}
    trades, eq, sig, ev = (store.table(t) for t in ("trades", "equity", "signals", "events"))
    states = {name: store.load_state(name) for name in runs}

    st.title("📈 Paper Trading")
    broker = cfg.get("app", {}).get("broker", "sim")
    st.caption(f"Broker: **{'Webull paper account' if broker == 'webull' else 'built-in paper broker'}** · "
               f"database `{DB}` · refreshes every 30 s")

    # ---- summary across runs -------------------------------------------------
    rows = []
    for name, r in runs.items():
        s = states.get(name)
        e_last = eq[eq.run == name].equity.iloc[-1] if (eq.run == name).any() else (s.equity if s else r["capital"])
        t = trades[trades.run == name]
        pos = s.position if s else None
        rows.append({"Run": name, "Equity": e_last, "Return": (e_last / r["capital"] - 1) * 100,
                     "Trades": len(t), "Win rate": (t.pnl_usd > 0).mean() * 100 if len(t) else float("nan"),
                     "Position": f"🟢 {pos.side.upper()} @ {pos.entry_price:,.2f}" if pos else "⚪ flat",
                     "Last candle": s.last_bar if s else "-"})
    if rows:
        total_cap = sum(r["capital"] for r in runs.values())
        total_eq = sum(x["Equity"] for x in rows)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total equity", f"${total_eq:,.0f}", f"{(total_eq / total_cap - 1) * 100:+.1f}%")
        c2.metric("Closed trades", len(trades))
        c3.metric("Win rate", f"{(trades.pnl_usd > 0).mean() * 100:.0f}%" if len(trades) else "-")
        c4.metric("Open positions", sum(1 for s in states.values() if s and s.position))
        st.dataframe(pd.DataFrame(rows).style.format({"Equity": "${:,.0f}", "Return": "{:+.1f}%",
                                                      "Win rate": "{:.0f}%"}, na_rep="-"),
                     hide_index=True, width="stretch")
    else:
        st.info(f"No runs in {CONFIG} yet.")

    # ---- per run --------------------------------------------------------------
    if runs:
        name = st.selectbox("Run", list(runs))
        r, s = runs[name], states.get(name)
        st.caption(f"{r['strategy']} · {r['symbol']} {r['timeframe']} · {exit_text(r)} · "
                   f"shorts {'on' if r['shorts'] else 'off'} · feed {r['feed']}")
        left, right = st.columns([3, 2])
        with left:
            e = eq[eq.run == name]
            if len(e):
                ch = alt.Chart(e.assign(time=pd.to_datetime(e.time))).mark_line().encode(
                    x=alt.X("time:T", title=None), y=alt.Y("equity:Q", title="Equity ($)", scale=alt.Scale(zero=False)),
                    tooltip=["time:T", alt.Tooltip("equity:Q", format="$,.0f"), alt.Tooltip("price:Q", format=",.2f")])
                st.altair_chart(ch.properties(height=260), width="stretch")
            else:
                st.caption("Equity curve appears after the first few minutes of running.")
            t = trades[trades.run == name]
            st.subheader("Closed trades")
            if len(t):
                st.dataframe(t.drop(columns=["run"]).iloc[::-1].style.format(
                    {"entry_price": "{:,.2f}", "exit_price": "{:,.2f}", "qty": "{:g}", "pnl_usd": "{:+,.2f}",
                     "pnl_pct": "{:+.2f}%"}), hide_index=True, width="stretch")
            else:
                st.caption("No closed trades yet.")
        with right:
            st.subheader("Latest candle checklist")
            g = sig[sig.run == name]
            st.code(g.checklist.iloc[-1] if len(g) else "waiting for the first closed candle", language=None)
            st.subheader("Activity")
            st.dataframe(ev[ev.run == name].drop(columns=["run"]).iloc[::-1].head(50),
                         hide_index=True, width="stretch")


page()
