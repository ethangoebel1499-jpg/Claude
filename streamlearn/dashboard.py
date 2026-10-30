"""Stream study dashboard. Start it with:  python -m streamlearn dashboard"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from streamlearn.playbook import group_rules, scorecard  # noqa: E402
from streamlearn.store import Store  # noqa: E402

DB = os.environ.get("STREAMLEARN_DB", "streamlearn.db")
st.set_page_config(page_title="Stream Study", page_icon="🎙️", layout="wide")

OUTCOME = {"win": "🟢 win", "loss": "🔴 loss", "right": "🟢 right", "wrong": "🔴 wrong"}


def outcome_label(r) -> str:
    if r.result in OUTCOME:
        return OUTCOME[r.result]
    return {"open": "🟡 open", "expired": "⚪ expired", "entry never hit": "⚪ no entry",
            "waiting for entry": "🟡 waiting", "too recent": "🟡 too recent"}.get(r.status, r.status or "⚪ unchecked")


store = Store(DB)
streams, rules, levels = store.table("streams"), store.table("rules"), store.table("levels")
calls = store.calls_with_streams()

st.title("🎙️ Stream Study")
st.caption(f"{len(streams)} streams · {len(calls)} calls · {len(rules)} rules · database `{DB}`")
if streams.empty:
    st.info("No streams yet. Add one: python -m streamlearn add <transcript.txt> --start \"2026-10-28 20:00\"")
    st.stop()

tab_score, tab_calls, tab_book, tab_streams = st.tabs(["Scorecard", "Calls", "Playbook", "Streams"])

with tab_score:
    sc = scorecard(calls)
    if sc:
        o = sc["overall"].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Calls", int(o["calls"]))
        c2.metric("Win rate (stop/target)", "-" if pd.isna(o["win rate"]) else f"{o['win rate']:.0f}%",
                  help=f"{int(o['with levels, closed'])} calls with a stop and target that finished")
        c3.metric("Right after 7 days", "-" if pd.isna(o["right after 7d"]) else f"{o['right after 7d']:.0f}%")
        c4.metric("Avg R", "-" if pd.isna(o["avg R"]) else f"{o['avg R']:+.2f}")
        for name in ("by kind", "by asset", "by direction"):
            st.subheader(name.capitalize())
            st.dataframe(sc[name].round(2), width="stretch")
    else:
        st.caption("No calls extracted yet.")

with tab_calls:
    if len(calls):
        f1, f2, f3 = st.columns(3)
        asset = f1.multiselect("Asset", sorted(calls.asset.unique()))
        kind = f2.multiselect("Kind", sorted(calls.kind.unique()))
        direction = f3.multiselect("Direction", ["long", "short"])
        v = calls
        if asset:
            v = v[v.asset.isin(asset)]
        if kind:
            v = v[v.kind.isin(kind)]
        if direction:
            v = v[v.direction.isin(direction)]
        v = v.assign(outcome=v.apply(outcome_label, axis=1),
                     targets=v.targets.map(lambda x: ", ".join(f"{t:,.0f}" for t in json.loads(x))),
                     when=v.t.map(lambda t: f"{t // 60}:{t % 60:02d}"))
        st.dataframe(v.sort_values("said_at", ascending=False)[
            ["said_at", "title", "when", "asset", "direction", "kind", "timeframe", "entry", "stop", "targets",
             "outcome", "ret_7d", "mfe_pct", "mae_pct", "quote"]], hide_index=True, width="stretch")

with tab_book:
    groups = group_rules(rules, streams)
    cats = sorted({g["category"] for g in groups})
    pick = st.multiselect("Category", cats)
    for g in groups:
        if pick and g["category"] not in pick:
            continue
        st.markdown(f"**{g['rule']}**  \n`{g['category']}` · {g['streams']} streams · *\"{g['quote']}\"* "
                    f"({g['source']})")

with tab_streams:
    sel = st.selectbox("Stream", streams.sort_values("start", ascending=False).apply(
        lambda r: f"{r.start[:16]} · {r.title}", axis=1).tolist())
    sid = int(streams.sort_values("start", ascending=False).iloc[
        streams.sort_values("start", ascending=False).apply(lambda r: f"{r.start[:16]} · {r.title}", axis=1)
        .tolist().index(sel)].id)
    q = st.text_input("Search the transcript")
    seg = pd.read_sql_query("SELECT t, text FROM segments WHERE stream_id = ? ORDER BY t", store.db, params=(sid,))
    if q:
        seg = seg[seg.text.str.contains(q, case=False, regex=False)]
    seg = seg.assign(time=seg.t.map(lambda t: f"{t // 60}:{t % 60:02d}"))[["time", "text"]]
    left, right = st.columns([3, 2])
    left.dataframe(seg, hide_index=True, width="stretch", height=500)
    lv = levels[levels.stream_id == sid]
    right.subheader("Levels he marked")
    right.dataframe(lv[["asset", "price", "kind"]], hide_index=True, width="stretch")
