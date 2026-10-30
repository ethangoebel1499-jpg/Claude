"""Stream study dashboard. Standalone:  python -m streamlearn dashboard
It is also the 🎙️ Streams tab of the paper-trading dashboard (render())."""

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

from livepaper.fmt import pct, price  # noqa: E402
from streamlearn.learning import accuracy  # noqa: E402
from streamlearn.playbook import group_rules, scorecard  # noqa: E402
from streamlearn.store import Store  # noqa: E402
from streamlearn.transcripts import MEDIA  # noqa: E402

AUDIO = {".mp3", ".m4a", ".wav", ".aac", ".ogg", ".flac"}


def mmss(t) -> str:
    t = int(t or 0)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def targets_of(c) -> list[float]:
    return json.loads(c["targets"]) if isinstance(c["targets"], str) else list(c["targets"] or [])


def restate(c) -> str:
    """The agent's own restatement, or one built from the fields."""
    if c.get("summary"):
        return c["summary"]
    conds = json.loads(c["conditions"]) if isinstance(c.get("conditions"), str) else (c.get("conditions") or [])
    side = "Buy" if c["direction"] == "long" else "Short"
    parts = [f"{side} {c['asset']}" + (f" on the {c['timeframe']}" if c.get("timeframe") else ""),
             f"entry {price(c['entry'])}" if c.get("entry") else "entry at the market",
             f"only if {'; '.join(conds)}" if conds else None,
             f"stop {price(c['stop'])}" if c.get("stop") else "no stop given",
             "targets " + ", ".join(price(t) for t in targets_of(c)) if targets_of(c) else "no target given"]
    return ", ".join(p for p in parts if p) + "."


def outcome_text(c) -> str:
    r, s = c.get("result"), c.get("status")
    if r == "win":
        return f"🟢 Hit the target {price(c['exit_price'])} ({pct(c['ret_pct'])}, {c['r_multiple']:+.2f}R) on {str(c['exit_time'])[:16]}"
    if r == "loss":
        return f"🔴 Stopped out at {price(c['exit_price'])} ({pct(c['ret_pct'])}) on {str(c['exit_time'])[:16]}"
    if r in ("right", "wrong"):
        return f"{'🟢' if r == 'right' else '🔴'} No levels given; price moved {pct(c['ret_7d'])} his way after 7 days"
    return {"open": f"🟡 Still open ({pct(c.get('ret_pct'))} so far)", "expired": "⚪ Neither target nor stop in 14 days",
            "entry never hit": "⚪ Price never reached his entry", "waiting for entry": "🟡 Waiting for the entry price",
            "too recent": "🟡 Too recent to judge"}.get(s, "⚪ Not checked yet - run `python -m streamlearn verify`")


@st.cache_data(ttl=3600, show_spinner=False)
def candles_around(asset: str, said_at: str) -> pd.DataFrame:
    from streamlearn.verify import Candles
    t = pd.Timestamp(said_at)
    df = Candles().get(asset, t)
    win = df[(df.index >= t - pd.Timedelta("3D")) & (df.index <= t + pd.Timedelta("7D"))]
    step = pd.Series(win.index).diff().median()
    if pd.notna(step) and step <= pd.Timedelta("1h"):
        win = win.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return win


def price_chart(c) -> alt.Chart | None:
    try:
        df = candles_around(c["asset"], c["said_at"]).reset_index(names="time")
    except Exception as e:
        st.caption(f"No price data for {c['asset']}: {e}")
        return None
    if df.empty:
        return None
    df["up"] = df.close >= df.open
    base = alt.Chart(df).encode(x=alt.X("time:T", title=None))
    y = alt.Y("low:Q", title=None, scale=alt.Scale(zero=False), axis=alt.Axis(format=",.2f"))
    wick = base.mark_rule().encode(y=y, y2="high:Q", color=alt.condition("datum.up", alt.value("#16a34a"),
                                                                           alt.value("#dc2626")))
    body = base.mark_bar(size=5).encode(y="open:Q", y2="close:Q",
                                        color=alt.condition("datum.up", alt.value("#16a34a"), alt.value("#dc2626")),
                                        tooltip=[alt.Tooltip("time:T"), alt.Tooltip("open:Q", format=",.2f"),
                                                 alt.Tooltip("high:Q", format=",.2f"),
                                                 alt.Tooltip("low:Q", format=",.2f"),
                                                 alt.Tooltip("close:Q", format=",.2f")])
    levels = [("entry", c.get("entry"), "#2563eb"), ("stop", c.get("stop"), "#dc2626")] + \
        [(f"target {i + 1}", t, "#16a34a") for i, t in enumerate(targets_of(c))]
    lv = pd.DataFrame([{"level": n, "price": p, "color": col} for n, p, col in levels if p])
    layers = [wick, body]
    if len(lv):
        layers.append(alt.Chart(lv).mark_rule(strokeDash=[6, 4]).encode(
            y="price:Q", color=alt.Color("color:N", scale=None),
            tooltip=["level:N", alt.Tooltip("price:Q", format=",.2f")]))
    said = pd.DataFrame({"time": [pd.Timestamp(c["said_at"])]})
    layers.append(alt.Chart(said).mark_rule(color="#7c3aed", strokeWidth=2).encode(x="time:T"))
    return alt.layer(*layers).properties(height=260)


def clip(c, store) -> None:
    """Play the part of the stream where he lays out this setup, start to finish."""
    start, end = int(c.get("t_start") or c["t"]), int(c.get("t_end") or c["t"] + 120)
    st.caption(f"Stream {mmss(start)} → {mmss(end)} ({(end - start) // 60} min {(end - start) % 60} s)")
    src = Path(str(c.get("source") or ""))
    if c.get("url"):
        st.video(c["url"], start_time=start, end_time=end)
    elif src.suffix.lower() in MEDIA and src.exists():
        (st.audio if src.suffix.lower() in AUDIO else st.video)(str(src), start_time=start, end_time=end)
    else:
        st.info("Add this stream's YouTube link (or keep the video file) in the **Streams** tab to watch "
                "this part. The transcript of it is below.")
    seg = pd.read_sql_query("SELECT t, text FROM segments WHERE stream_id = ? AND t BETWEEN ? AND ? ORDER BY t",
                            store.db, params=(int(c["stream_id"]), start, end))
    with st.expander("Transcript of this part"):
        st.markdown("\n\n".join(f"`{mmss(t)}` {txt}" for t, txt in seg.itertuples(index=False)) or "-")


def review_form(c, store) -> None:
    cid = int(c["id"])
    with st.form(f"review_{cid}"):
        st.markdown("**Is this setup extracted correctly?** Fix anything that's off, then award or punish.")
        f1, f2, f3, f4 = st.columns([1, 1.2, 1.2, 2])
        direction = f1.selectbox("Direction", ["long", "short"], index=0 if c["direction"] == "long" else 1,
                                 key=f"d{cid}")
        entry = f2.text_input("Entry", price(c["entry"]) if c.get("entry") else "", key=f"e{cid}")
        stop = f3.text_input("Stop", price(c["stop"]) if c.get("stop") else "", key=f"s{cid}")
        tgts = f4.text_input("Targets", ", ".join(price(t) for t in targets_of(c)), key=f"t{cid}")
        summary = st.text_area("Setup in plain words", restate(c), key=f"m{cid}", height=80)
        note = st.text_input("Note for the agent (what it got right or wrong)", key=f"n{cid}")
        b1, b2, _ = st.columns([1, 1, 3])
        award = b1.form_submit_button("👍 Correct - award", type="primary")
        punish = b2.form_submit_button("👎 Wrong - punish")
    if award or punish:
        num = (lambda s: float(s.replace(",", "")) if s.strip() else None)
        try:
            new = {"direction": direction, "entry": num(entry), "stop": num(stop),
                   "targets": [float(x.replace(",", "")) for x in tgts.split(",") if x.strip()], "summary": summary}
        except ValueError:
            st.error("Entry, stop and targets must be numbers (e.g. 64,200 or 64200).")
            return
        old = {"direction": c["direction"], "entry": c.get("entry"), "stop": c.get("stop"),
               "targets": targets_of(c), "summary": restate(c)}
        changes = {k: v for k, v in new.items() if v != old[k]} if award else {}
        store.review(cid, "correct" if award else "wrong", note, changes,
                     datetime.now(timezone.utc).isoformat(timespec="seconds"))
        if changes:
            from streamlearn.verify import Candles, check
            fresh = store.calls_with_streams().set_index("id").loc[cid].to_dict()
            store.save_outcome(check({**fresh, "id": cid}, Candles().get(fresh["asset"], pd.Timestamp(fresh["said_at"]))))
        st.rerun()


def setup_card(c, store, expanded: bool) -> None:
    arrow = "🟢 LONG" if c["direction"] == "long" else "🔴 SHORT"
    badge = {"correct": " · 👍 awarded", "wrong": " · 👎 punished"}.get(c.get("verdict"), "")
    lv = " · ".join(x for x in [f"entry {price(c['entry'])}" if c.get("entry") else "",
                                f"stop {price(c['stop'])}" if c.get("stop") else "",
                                ("targets " + ", ".join(price(t) for t in targets_of(c))) if targets_of(c) else ""] if x)
    with st.expander(f"{arrow} {c['asset']} · {lv or 'no levels'} · {str(c['said_at'])[:16]} UTC{badge}",
                     expanded=expanded):
        st.markdown(f"**The setup:** {restate(c)}")
        st.caption(f"{c['kind']} · {c['timeframe'] or 'timeframe not stated'} · said at {mmss(c['t'])} in "
                   f"*{c['title']}* · \"{c['quote']}\"")
        st.markdown(f"**What happened:** {outcome_text(c)}")
        left, right = st.columns([1, 1])
        with left:
            clip(c, store)
        with right:
            ch = price_chart(c)
            if ch is not None:
                st.altair_chart(ch, width="stretch")
                st.caption("Price around the call (purple line = when he said it; blue entry, red stop, "
                           "green targets).")
            if c.get("frame") and Path(str(c["frame"])).exists():
                st.image(str(c["frame"]), caption="His screen at that moment")
        if c.get("verdict"):
            st.caption(f"Reviewed: {c['verdict']}" + (f" - {c['review_note']}" if c.get("review_note") else ""))
            if st.button("Undo review", key=f"undo{int(c['id'])}"):
                store.unreview(int(c["id"]))
                st.rerun()
        else:
            review_form(c, store)


def score_table(df: pd.DataFrame) -> pd.DataFrame:
    """Text cells so blanks show as '-' (the grid shows NaN as 'None')."""
    fmt = {"calls": "{:,.0f}", "with levels, closed": "{:,.0f}", "win rate": "{:.0f}%", "avg R": "{:+.2f}",
           "right after 7d": "{:.0f}%"}
    out = df.astype(float).copy().astype(object)
    for col in df.columns:
        f = fmt.get(col)
        out[col] = [("-" if pd.isna(v) else f.format(v) if f else pct(v)) for v in df[col].astype(float)]
    return out


def render(db: str) -> None:
    store = Store(db)
    streams, rules, levels = store.table("streams"), store.table("rules"), store.table("levels")
    calls = store.calls_with_streams()
    good, bad = accuracy(store)
    acc = f"agent accuracy {good}/{good + bad} ({good / (good + bad):.0%})" if good + bad else "no reviews yet"
    st.caption(f"{len(streams)} streams · {len(calls)} setups · {len(rules)} rules · {acc} · `{Path(db).name}`")
    if streams.empty:
        st.info('No streams yet. Add one: python -m streamlearn add <transcript.txt> --start "2026-10-28 20:00" '
                '--url <YouTube link>')
        return
    t_rev, t_score, t_calls, t_book, t_streams = st.tabs(["📝 Review", "Scorecard", "All setups", "Playbook",
                                                          "Streams"])
    with t_rev:
        show = st.radio("Show", ["Needs review", "Reviewed", "All"], horizontal=True, key="rv_show")
        v = calls if show == "All" else calls[calls.verdict.isna()] if show == "Needs review" else \
            calls[calls.verdict.notna()]
        v = v.sort_values("said_at", ascending=False)
        if v.empty:
            st.caption("Nothing here. 🎉" if show == "Needs review" else "No reviews yet.")
        for i, c in enumerate(v.to_dict("records")[:50]):
            setup_card(c, store, expanded=(i == 0 and show == "Needs review"))
        if len(v) > 50:
            st.caption(f"Showing 50 of {len(v)}.")

    with t_score:
        judged = calls[calls.verdict.ne("wrong")]                 # punished setups were extraction mistakes
        sc = scorecard(judged)
        if sc:
            o = sc["overall"].iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Setups", int(o["calls"]))
            c2.metric("Win rate (stop/target)", "-" if pd.isna(o["win rate"]) else f"{o['win rate']:.0f}%")
            c3.metric("Right after 7 days", "-" if pd.isna(o["right after 7d"]) else f"{o['right after 7d']:.0f}%")
            c4.metric("Avg R", "-" if pd.isna(o["avg R"]) else f"{o['avg R']:+.2f}")
            st.caption("Setups you punished as extraction mistakes are left out.")
            for name in ("by kind", "by asset", "by direction"):
                with st.expander(name.capitalize(), expanded=name == "by asset"):
                    st.dataframe(score_table(sc[name]), width="stretch")

    with t_calls:
        if len(calls):
            show = calls.sort_values("said_at", ascending=False).assign(
                entry=lambda d: d.entry.map(price), stop=lambda d: d.stop.map(price),
                targets=lambda d: d.targets.map(lambda x: ", ".join(price(t) for t in json.loads(x))),
                outcome=lambda d: [outcome_text(r) for r in d.to_dict("records")],
                review=lambda d: d.verdict.map({"correct": "👍", "wrong": "👎"}).fillna(""))
            st.dataframe(show[["said_at", "title", "asset", "direction", "kind", "entry", "stop", "targets",
                               "outcome", "review", "quote"]], hide_index=True, width="stretch")

    with t_book:
        groups = group_rules(rules, streams)
        for cat in sorted({g["category"] for g in groups}):
            sub = [g for g in groups if g["category"] == cat]
            with st.expander(f"{cat.capitalize()} ({len(sub)})", expanded=cat == "entry"):
                for g in sub:
                    st.markdown(f"**{g['rule']}** · {g['streams']} streams  \n*\"{g['quote']}\"* ({g['source']})")

    with t_streams:
        for s in streams.sort_values("start", ascending=False).to_dict("records"):
            n = int((calls.stream_id == s["id"]).sum())
            with st.expander(f"{s['start'][:16]} UTC · {s['title']} · {n} setups"):
                url = st.text_input("Stream link (YouTube etc.) - lets each setup play its part of the stream",
                                    s.get("url") or "", key=f"url{s['id']}")
                if st.button("Save link", key=f"saveurl{s['id']}"):
                    store.set_url(int(s["id"]), url.strip())
                    st.rerun()
                q = st.text_input("Search the transcript", key=f"q{s['id']}")
                seg = pd.read_sql_query("SELECT t, text FROM segments WHERE stream_id = ? ORDER BY t", store.db,
                                        params=(int(s["id"]),))
                if q:
                    seg = seg[seg.text.str.contains(q, case=False, regex=False)]
                st.dataframe(seg.assign(time=seg.t.map(mmss))[["time", "text"]], hide_index=True, width="stretch",
                             height=300)
                lv = levels[levels.stream_id == s["id"]]
                if len(lv):
                    st.dataframe(lv.assign(price=lv.price.map(price))[["asset", "price", "kind"]], hide_index=True)


if __name__ == "__main__" or os.environ.get("STREAMLEARN_STANDALONE"):
    st.set_page_config(page_title="Stream Study", page_icon="🎙️", layout="wide")
    st.title("🎙️ Stream Study")
    render(os.environ.get("STREAMLEARN_DB", "streamlearn.db"))
