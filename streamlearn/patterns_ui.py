"""📐 Patterns tab: a live pattern check on any coin, every pattern Jason points out (with the clip
and what happened), and the playbook - what to do in every situation, with how each pattern
actually behaved on crypto since 2017."""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from livepaper.fmt import price
from mcbot.patterns import CONVERGING, NAMES, PLAYBOOK, PUBLISHED, SITUATIONS, Detector, situation, stats

KEY_NAMES = dict((k, n) for k, n, _ in SITUATIONS)
KEY_NAMES.update({"broke": "Broke out"})
AXIS = ",.8~f"            # full prices, never exponents


def pattern_chart(df: pd.DataFrame, s: dict | None, bars: int = 160) -> alt.Chart:
    view = df.iloc[-bars:].reset_index(names="time")
    view["up"] = view.close >= view.open
    base = alt.Chart(view).encode(x=alt.X("time:T", title=None))
    color = alt.condition("datum.up", alt.value("#16a34a"), alt.value("#dc2626"))
    y = alt.Y("low:Q", title=None, scale=alt.Scale(zero=False), axis=alt.Axis(format=AXIS))
    layers = [base.mark_rule().encode(y=y, y2="high:Q", color=color),
              base.mark_bar(size=4).encode(y="open:Q", y2="close:Q", color=color,
                                           tooltip=[alt.Tooltip("time:T"), alt.Tooltip("close:Q", format=AXIS)])]
    if s:
        p, n = s["pattern"], len(df)
        first = max(p.start, n - bars)
        last = n - 1
        if p.apex is not None:
            last = int(min(max(p.apex, last), last + 40))
        step = df.index[-1] - df.index[-2]
        xs = list(range(first, last + 1))
        t = [df.index[i] if i < n else df.index[-1] + step * (i - n + 1) for i in xs]
        lines = pd.DataFrame({"time": t * 2, "price": [p.up(i) for i in xs] + [p.lo(i) for i in xs],
                              "line": ["top line"] * len(xs) + ["bottom line"] * len(xs)})
        layers.append(alt.Chart(lines).mark_line(strokeWidth=2).encode(
            x="time:T", y=alt.Y("price:Q", axis=alt.Axis(format=AXIS)), detail="line:N",
            color=alt.value("#7c3aed"), tooltip=["line:N", alt.Tooltip("price:Q", format=AXIS)]))
        marks = []
        if p.apex is not None:
            i = int(round(p.apex))
            ta = df.index[i] if i < n else df.index[-1] + step * (i - n + 1)
            marks.append({"time": ta, "label": "apex (lines meet)", "color": "#f59e0b"})
        if s.get("break_bar") is not None:
            marks.append({"time": df.index[s["break_bar"]], "label": f"break {s['broke']}", "color": "#2563eb"})
        if marks:
            m = pd.DataFrame(marks)
            layers.append(alt.Chart(m).mark_rule(strokeDash=[4, 3], strokeWidth=2).encode(
                x="time:T", color=alt.Color("color:N", scale=None), tooltip=["label:N"]))
        lv = [("target", s.get("target")), ("stop", s.get("stop"))] if s.get("broke") else []
        lv = pd.DataFrame([{"level": a, "price": b} for a, b in lv if b])
        if len(lv):
            layers.append(alt.Chart(lv).mark_rule(strokeDash=[6, 4]).encode(
                y="price:Q", color=alt.Color("level:N", scale=alt.Scale(domain=["target", "stop"],
                                                                       range=["#16a34a", "#dc2626"])),
                tooltip=["level:N", alt.Tooltip("price:Q", format=AXIS)]))
    return alt.layer(*layers).properties(height=320)


def stats_table(kind: str) -> pd.DataFrame | None:
    st_ = stats().get(kind)
    if not st_:
        return None
    f = lambda x: "-" if x is None else f"{x:.0f}%"
    o = lambda d: "-" if not d or not d.get("n") else f"{d['up']:.0f}% (likely {d['lo']:.0f}-{d['hi']:.0f}%, {d['n']:,} breaks)"
    rows = [("Breaks found (8 coins, 4h + daily, 2017 to now)", f"{st_['n']:,}"),
            ("Chance of breaking UP", o(st_.get("odds"))),
            ("...when it formed after an uptrend", o((st_.get("odds_trend") or {}).get("up"))),
            ("...when it formed after a downtrend", o((st_.get("odds_trend") or {}).get("down"))),
            ("Same shape on random prices breaks UP (from line geometry alone)",
             f"{st_['random']['up']:.0f}%" if st_.get("random") else "-"),
            ("Real edge (crypto minus random)",
             f"{st_['odds']['up'] - st_['random']['up']:+.0f} points" if st_.get("random") and st_.get("odds") else "-"),
            ("Published stock studies (Bulkowski)", PUBLISHED.get(kind, {}).get("text", "-")),
            ("Broke the textbook direction", f(st_.get("with_pct"))),
            ("Textbook-direction breaks that reached the target", f(st_.get("target_with"))),
            ("Opposite-direction breaks that reached the target", f(st_.get("target_against"))),
            ("Fake breaks (closed back inside within 3 candles)", f(st_["fake"])),
            ("Came back to retest the broken line", f(st_["retest"]))]
    if "squeezed" in st_:
        rows += [("Squeezed near the apex (85%+)", f"{st_['squeezed']:,}"),
                 ("...then fizzled without breaking", f(st_.get("squeeze_no_break"))),
                 ("...late break went the textbook way", f(st_.get("squeeze_with"))),
                 ("...late break reached its target", f(st_.get("squeeze_target")))]
    return pd.DataFrame(rows, columns=["", "History"])


def live_check() -> None:
    from livepaper.history import CRYPTO, history
    c1, c2, c3 = st.columns([2, 1, 1])
    coin = c1.selectbox("Coin", list(CRYPTO), key="pat_coin")
    tf = c2.selectbox("Timeframe", ["1h", "2h", "4h", "12h", "1D"], index=2, key="pat_tf")
    c3.write("")
    go = c3.button("Check chart", type="primary", key="pat_go")
    if go:
        with st.spinner("Loading candles..."):
            try:
                df = history(CRYPTO[coin], "crypto", tf).iloc[-600:]
            except Exception as e:
                st.error(f"No candles for {coin}: {e}")
                return
        st.session_state["pat_result"] = (coin, tf, df, situation(df))
    res = st.session_state.get("pat_result")
    if not res:
        st.caption("Pick a coin and timeframe: the agent finds the latest pattern, draws its lines and the apex, "
                   "says where price is in it, and what to do.")
        return
    coin, tf, df, s = res
    if s is None:
        st.info(f"No clear pattern on {coin} {tf} right now (last close {price(df.close.iloc[-1])}).")
        st.altair_chart(pattern_chart(df, None), use_container_width=True)
        return
    p = s["pattern"]
    prog = f" · {s['progress']:.0%} of the way to the apex" if s.get("progress") is not None else ""
    st.subheader(f"{coin} {tf}: {p.name} — {KEY_NAMES.get(s['key'], s['key'])}")
    st.caption(f"Last close {price(s['price'])} · top line {price(s['upper'])} · bottom line {price(s['lower'])}{prog} "
               f"· candle closed {df.index[-1]:%Y-%m-%d %H:%M} UTC")
    st.altair_chart(pattern_chart(df, s), use_container_width=True)
    st.markdown(f"**What to do:** {s['text']}")
    situations_for(p.kind, current=s["key"])


def situations_for(kind: str, current: str | None = None) -> None:
    with st.expander(f"Every situation for a {NAMES[kind].lower()}", expanded=current is None):
        rows = [(("👉 " if k == current else "") + n, a) for k, n, a in SITUATIONS
                if kind in CONVERGING or k not in ("breakout_zone", "near_apex", "past_apex")]
        st.dataframe(pd.DataFrame(rows, columns=["Situation", "What to do"]), hide_index=True, width="stretch")
        t = stats_table(kind)
        if t is not None:
            st.dataframe(t, hide_index=True, width="stretch")


def jason_patterns(store, clip) -> None:
    pats = store.patterns_with_streams()
    if pats.empty:
        st.info("No patterns from streams yet. New streams are scanned for patterns automatically; for streams "
                "added before, rerun: python -m streamlearn add <file> --start ... --reextract")
        return
    for _, r in pats.sort_values("said_at", ascending=False).iterrows():
        name = NAMES.get(r.pattern, r.pattern_name or r.pattern)
        icon = {"correct": "👍 ", "wrong": "👎 "}.get(r.verdict or "", "")
        with st.expander(f"{icon}{r.asset} {r.timeframe or ''} · {name} · said {str(r.said_at)[:16]} · "
                         f"{r.outcome or 'not checked yet'}"):
            st.markdown(f"**What he said:** {r.summary or r.quote}")
            levels = " · ".join(x for x in [f"top line {price(r.top)}" if pd.notna(r.top) else "",
                                            f"bottom line {price(r.bottom)}" if pd.notna(r.bottom) else ""] if x)
            exp = {"up": "expects a break UP", "down": "expects a break DOWN"}.get(r.bias, "no direction given")
            st.caption(f"He {exp}" + (f" · {levels}" if levels else ""))
            plan = [("If it breaks up", r.if_up), ("If it breaks down", r.if_down), ("At the apex", r.at_apex)]
            plan = [(a, b) for a, b in plan if b]
            tg = json.loads(r.targets or "[]")
            if tg:
                plan.append(("His targets", ", ".join(price(x) for x in tg)))
            if plan:
                st.table(pd.DataFrame(plan, columns=["", "His plan"]).set_index(""))
            if r.outcome:
                st.markdown(f"**What happened:** {r.outcome}")
            clip({**r.to_dict(), "t": r.t}, store)
            if r.frame and Path(r.frame).exists():
                st.image(r.frame, caption="His screen at that moment")
            if r.pattern in PLAYBOOK:
                st.caption(f"Textbook: {PLAYBOOK[r.pattern]['usual']}")
            b1, b2, _ = st.columns([1, 1, 3])
            if b1.button("👍 Agent got it right", key=f"pg_{r.id}"):
                store.update_pattern(int(r.id), verdict="correct")
                st.rerun()
            if b2.button("👎 Agent got it wrong", key=f"pb_{r.id}"):
                store.update_pattern(int(r.id), verdict="wrong")
                st.rerun()


def playbook(store) -> None:
    kind = st.selectbox("Pattern", list(PLAYBOOK), format_func=lambda k: NAMES[k], key="pb_kind")
    pb = PLAYBOOK[kind]
    st.markdown(f"**What it is:** {pb['what']}  \n**Textbook:** {pb['usual']}")
    situations_for(kind)
    pats = store.patterns_with_streams()
    mine = pats[pats.pattern == kind] if len(pats) else pats
    if len(mine):
        st.markdown("**What Jason says about it**")
        for _, r in mine.iterrows():
            said = " · ".join(x for x in [r.if_up and f"break up: {r.if_up}", r.if_down and f"break down: {r.if_down}",
                                          r.at_apex and f"at the apex: {r.at_apex}"] if x)
            st.markdown(f"- *\"{r.quote}\"* ({r.title}, {r.asset}) {('— ' + said) if said else ''}")


def render(store, clip) -> None:
    t1, t2, t3 = st.tabs(["🔎 Check a chart now", "🎙️ Patterns from streams", "📖 Pattern playbook"])
    with t1:
        live_check()
    with t2:
        jason_patterns(store, clip)
    with t3:
        playbook(store)
