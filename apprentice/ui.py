"""🧠 Apprentice tab: teach it (YouTube videos, whole playlists, pasted transcripts, the stream agent's
streams), see what it knows, rate its accuracy, watch it read a chart and decide, review the trades it
missed, and let it paper trade."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from apprentice import brain, youtube
from apprentice.job import DONE, JOBS, spawn
from apprentice.store import Store
from apprentice.strategy import KEY as STRATEGY_KEY, db_path
from livepaper import tvchart
from livepaper.fmt import price
from livepaper.history import CRYPTO, history
from mcbot import builder

ROOT = Path(__file__).resolve().parent.parent
LESSONS = ROOT / "runtime" / "apprentice_lessons"
TIMEFRAMES = ["15m", "30m", "1h", "2h", "4h", "6h", "12h", "1D"]
STARS = ["1 - way off", "2 - mostly wrong", "3 - partly right", "4 - mostly right", "5 - spot on"]
KIND_NAMES = {"setup": "Setups it learned", "concept": "Concepts it learned", "decision": "Its chart calls",
              "missed": "Missed trades it found"}


@st.cache_resource
def get_store(path: str) -> Store:
    return Store(path)


@st.cache_data(ttl=1800, show_spinner=False)
def load_chart(symbol: str, asset: str, tf: str) -> pd.DataFrame:
    return history(symbol, asset, tf).iloc[-3000:]


# ---- rating widget (used on every AI output) ----------------------------------------------------
def rate(store: Store, kind: str, ref, snapshot: dict, key: str, label: str = "Rate how accurate this is") -> None:
    """1-5 stars + a comment. Every rating is kept with its comment; past ones show underneath."""
    past = store.ratings(kind, ref)
    with st.popover(f"⭐ {label}" + (f" · {past[0]['stars']}/5" if past else "")):
        stars = st.radio("Accuracy", STARS, index=None, horizontal=False, key=f"rs_{key}")
        comment = st.text_area("Comment (what it got right or wrong - it reads this)", key=f"rc_{key}",
                               placeholder="e.g. He only takes this on the 4h, and the stop goes under the wick")
        if st.button("Save rating", key=f"rb_{key}", disabled=stars is None, type="primary"):
            store.rate(kind, ref, STARS.index(stars) + 1, comment, snapshot)
            st.success("Saved - thanks. It uses this the next time it learns.")
        for r in past[:10]:
            st.markdown(f"**{r['stars']}/5** · {r['at'][:16].replace('T', ' ')}" +
                        (f" - {r['comment']}" if r["comment"] else ""))
            if st.button("Delete", key=f"rd_{key}_{r['id']}"):
                store.delete_rating(r["id"])
                st.rerun()


def accuracy_panel(store: Store) -> None:
    acc = store.accuracy()
    cols = st.columns(5)
    a = acc["all"]
    cols[0].metric("Your accuracy rating", f"{a['avg']:.1f} / 5" if a["avg"] else "-",
                   f"{a['accurate_pct']}% rated accurate (4-5★)" if a["n"] else "rate its work below",
                   delta_color="off")
    for c, k in zip(cols[1:], ("setup", "concept", "decision", "missed")):
        v = acc[k]
        c.metric(KIND_NAMES[k], f"{v['avg']:.1f} / 5" if v["avg"] else "-", f"{v['n']} ratings", delta_color="off")
    rows = store.ratings()
    if len(rows) >= 3:
        df = pd.DataFrame(rows)[["at", "kind", "stars"]].iloc[::-1]
        df["n"] = range(1, len(df) + 1)
        df["running average"] = df.stars.expanding().mean()
        st.line_chart(df.set_index("n")[["running average"]], height=140)
    with st.expander(f"All your ratings and comments ({len(rows)})"):
        if rows:
            t = pd.DataFrame([{"when": r["at"][:16].replace("T", " "), "what": KIND_NAMES.get(r["kind"], r["kind"]),
                               "item": r["snapshot"].get("name") or r["snapshot"].get("text") or
                                       r["snapshot"].get("action") or r["ref"],
                               "stars": "★" * r["stars"] + "☆" * (5 - r["stars"]), "comment": r["comment"]}
                              for r in rows])
            st.dataframe(t, hide_index=True, width="stretch")
        else:
            st.caption("No ratings yet. Every setup, concept, chart call and missed trade has a ⭐ button.")


# ---- background jobs --------------------------------------------------------------------------
def job_state(text: str) -> str:
    if DONE not in text:
        m = re.findall(r"\[(\d+)/(\d+)\]", text)
        if m:
            return f"⏳ Learning... video {m[-1][0]} of {m[-1][1]}"
        if "Testing every setup" in text:
            return "⏳ Testing setups on real prices..."
        return "⏳ Working..."
    if f"{DONE} 0]" not in text:
        return "❌ Failed - see the log"
    parts = []
    m = re.search(r"RESULT lessons=(\d+) concepts=(\d+) setups=(\d+) skipped=(\d+) failed=(\d+)", text)
    if m:
        les, con, sets, skip, fail = map(int, m.groups())
        parts.append(f"{les} lesson(s), {con} concepts, {sets} new setups"
                     + (f", {skip} already learned" if skip else "") + (f", {fail} couldn't be read" if fail else ""))
    m = re.search(r"RESULT tested=(\d+) active=(\d+)", text)
    if m:
        parts.append(f"tested {m.group(1)} setups, {m.group(2)} beat random and are active")
    return "✅ Done - " + "; ".join(parts) if parts else "✅ Done"


def job_lessons(store: Store, log: Path, meta: dict) -> list[int]:
    """Lessons a job added: tagged with the job, or (jobs from before tagging) the videos its log names."""
    ids = [r[0] for r in store.db.execute("SELECT id FROM lessons WHERE job = ?", (log.stem,)).fetchall()]
    if ids:
        return ids
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    names = set(re.findall(r"^\[\d+/\d+\] (.+?): \d+ caption lines", text, re.M))
    if meta.get("args") and len(meta["args"]) > 1:
        names.add(meta["args"][1])
    started = meta.get("started", "")[:10]
    return [r[0] for r in store.db.execute("SELECT id, title, source, added FROM lessons").fetchall()
            if (r[1] in names or r[2] in names) and (r[3] or "")[:10] >= started]


def stop_job(store: Store, log: Path, meta: dict) -> None:
    from streamlearn import jobctl
    jobctl.stop(log)
    store.job_stopped(log.stem)
    for lid in job_lessons(store, log, meta):
        store.db.execute("UPDATE lessons SET status = 'stopped', note = 'stopped part-way' "
                         "WHERE id = ? AND status = 'learning'", (lid,))
    store.commit()


def delete_job(store: Store, log: Path, meta: dict) -> int:
    from streamlearn import jobctl
    stop_job(store, log, meta)
    n = 0
    for lid in job_lessons(store, log, meta):
        store.forget_lesson(lid)
        n += 1
    jobctl.delete_log(log)
    return n


@st.fragment(run_every=5)
def jobs() -> None:
    from streamlearn import jobctl
    store = get_store(db_path())
    logs = sorted(JOBS.glob("*.log"), reverse=True)[:8] if JOBS.exists() else []
    if not logs:
        return
    texts = {log: log.read_text(encoding="utf-8", errors="replace") if log.exists() else "" for log in logs}
    table = jobctl.processes() if any(DONE not in t for t in texts.values()) else []
    live = {log for log in logs if DONE not in texts[log] and jobctl.job_pids(log, table)}
    h1, h2 = st.columns([3, 1.3])
    h1.markdown(f"**Recent lessons and learning runs** · {len(live)} running")
    if live and h2.button(f"⏹ Stop all {len(live)} running", key="ap_stopall"):
        for log in live:
            meta = json.loads(log.with_suffix(".json").read_text()) if log.with_suffix(".json").exists() else {}
            stop_job(store, log, meta)
        st.rerun()
    for log in logs:
        meta = json.loads(log.with_suffix(".json").read_text()) if log.with_suffix(".json").exists() else {}
        text = texts[log]
        state = job_state(text) if (DONE in text or log in live) else "⚪ Not running (stopped or the app was closed)"
        if log in live and "Waiting for the job before it" in text and "[1/" not in text and "part 1/" not in text:
            state = "⏸️ Queued - waits for the job before it (one at a time on the AI)"
        with st.expander(f"{meta.get('title', log.stem)} · {state}", expanded=log in live):
            lines = [x for x in text.splitlines() if x.strip()]
            st.code("\n".join(lines[-14:]) or "starting...", language=None)
            b1, b2, _ = st.columns([1, 1.3, 3])
            if log in live and b1.button("⏹ Stop", key=f"ap_stop_{log.stem}",
                                         help="Ends it now; videos it already finished are kept"):
                stop_job(store, log, meta)
                st.rerun()
            with b2.popover("🗑️ Delete"):
                n = len(job_lessons(store, log, meta))
                st.warning(f"Stops it if it's running and deletes the {n} lesson(s)/video(s) it added and the setups "
                           "learned from them (pinned ones stay). This can't be undone.")
                if st.button("Yes, delete", key=f"ap_deljob_{log.stem}", type="primary"):
                    delete_job(store, log, meta)
                    st.rerun()


# ---- tabs ---------------------------------------------------------------------------------------
def teach_tab(store: Store) -> None:
    st.caption("Feed it anything that teaches trading: a YouTube video, a **whole YouTube playlist**, a pasted "
               "transcript or notes. It pulls out the concepts and the setups, turns each setup into rules it can "
               "check on a chart, then tests them on real prices - only setups that beat random get used.")
    no_ai = st.checkbox("Read lessons without the AI (keyword matching - rough, but needs no model)", False,
                        help="Normally it uses the same AI as the stream agent (⚙️ Account: local or Claude).")
    then_learn = st.checkbox("Test what it learned on real prices right after", True)
    extra = (["--no-ai"] if no_ai else []) + (["--then-learn"] if then_learn else [])

    st.markdown("##### 🔗 YouTube link - a video or a whole playlist")
    url = st.text_input("Link", placeholder="https://www.youtube.com/playlist?list=...   or   "
                                            "https://www.youtube.com/watch?v=...", key="ap_url").strip()
    k = youtube.kind(url) if url else None
    if url and not k:
        st.warning("That doesn't look like a YouTube video or playlist link.")
    elif k == "playlist":
        st.caption("📃 Playlist: it learns every video in it, one after another, using each video's captions. Videos "
                   "it has already learned are skipped, so you can paste the same playlist again when new videos "
                   "are added.")
    elif k == "video":
        st.caption("🎬 One video: it fetches the captions by itself - no need to copy the transcript.")
    if st.button("Learn from this link", type="primary", disabled=not k, key="ap_go_url"):
        spawn(["teach", url, *extra], f"{'Playlist' if k == 'playlist' else 'Video'}: {url[-40:]}")
        st.success("Learning in the background - progress shows below.")

    with st.expander("📝 Or paste a transcript / notes"):
        title = st.text_input("Lesson title", key="ap_title")
        text = st.text_area("Transcript or notes", height=200, key="ap_text",
                            placeholder="YouTube's Show transcript text, subtitles, or plain notes - timestamps are "
                                        "optional")
        if st.button("Learn this lesson", disabled=not (title.strip() and text.strip()), key="ap_go_text"):
            LESSONS.mkdir(parents=True, exist_ok=True)
            f = LESSONS / f"{int(time.time())} {re.sub(r'[^\w\- ]+', '', title)[:50]}.txt"
            f.write_text(text, encoding="utf-8")
            spawn(["teach", str(f), "--title", title.strip(), *extra], title.strip())
            st.success("Learning in the background - progress shows below.")

    with st.expander("🎙️ Learn from the streams the stream agent already studied"):
        st.caption("Every rule and chart pattern the stream study agent pulled out of its streams becomes a lesson "
                   "here too. Streams already imported are skipped.")
        if st.button("Import the stream agent's streams", key="ap_import"):
            spawn(["import-streams", *extra], "Stream agent's streams")
            st.success("Importing in the background.")
    jobs()

    les = store.table("lessons")
    with st.expander(f"Lessons it has learned ({len(les)})"):
        for r in les.iloc[::-1].itertuples():
            c1, c2 = st.columns([6, 1])
            c1.markdown(f"**{r.title}** · {r.status} · {r.note or ''}" + (f" · [video]({r.url})" if r.url else "")
                        + (f" · playlist *{r.playlist}*" if r.playlist else ""))
            with c2.popover("🗑️ Delete"):
                st.caption("Deletes this video/lesson, its concepts and the setups learned from it (pinned ones stay).")
                if st.button("Yes, delete", key=f"ap_forget_{r.id}", type="primary"):
                    store.forget_lesson(r.id)
                    st.rerun()


def stats_line(s: dict) -> str:
    t = s["stats"].get("total", {})
    if not t.get("trades"):
        return "not tested yet" if not t else "never fired on the test charts"
    return (f"{t['trades']} past trades · win {t['win_rate']:.0f}% · {t['avg_r']:+.2f}R per trade · edge vs random "
            f"{t['edge']:+.2f}R · positive on {t.get('charts_positive', 0)}/{t.get('charts', 0)} charts")


def knowledge_tab(store: Store) -> None:
    setups = store.setups()
    by = {k: [s for s in setups if s["status"] == k] for k in ("pinned", "active", "candidate", "retired", "off")}
    m = st.columns(5)
    m[0].metric("Lessons", len(store.table("lessons")))
    m[1].metric("Concepts", len(store.table("concepts")))
    m[2].metric("Active setups", len(by["active"]) + len(by["pinned"]))
    m[3].metric("Candidates", len(by["candidate"]))
    m[4].metric("Retired (didn't work)", len(by["retired"]))
    c1, c2 = st.columns([3, 1])
    picks = c1.multiselect("Test on these charts", ["BTC:4h", "ETH:4h", "SOL:4h", "XRP:4h", "BTC:1D", "ETH:1D",
                                                    "BTC:1h", "SPY:1D", "QQQ:1D"],
                           ["BTC:4h", "ETH:4h", "SOL:4h", "BTC:1D", "ETH:1D"])
    c2.write("")
    if c2.button("🎓 Learn now", type="primary", disabled=not picks,
                 help="Tests every setup on real prices, keeps what beats random, and learns new setups from the "
                      "good trades it missed"):
        spawn(["learn", "--charts", *picks], "Learning run")
        st.success("Learning in the background - see 📚 Teach for progress.")
    st.markdown("##### Accuracy (your ratings)")
    accuracy_panel(store)

    st.markdown("##### Setups")
    st.caption("**Active** = tested and beat random (the apprentice trades these). **Candidate** = not enough "
               "trades yet. **Retired** = lost to random. 📌 Pin forces one on, 🚫 Off turns it off. Your ratings "
               "change how much it trusts each one.")
    labels = {"auto": "🤖 Let it decide", "pinned": "📌 Always use", "off": "🚫 Off"}
    for status in ("pinned", "active", "candidate", "retired", "off"):
        for s in by[status]:
            icon = {"pinned": "📌", "active": "🟢", "candidate": "🟡", "retired": "⚪", "off": "🚫"}[status]
            src = {"lesson": "from a lesson", "missed": "learned from missed trades", "you": "yours"}.get(s["source"], "")
            with st.expander(f"{icon} {s['direction'].upper()} · {s['name']} · weight {s['weight']:+.2f} · {src}"):
                st.markdown("**Enters when all of these are true on a closed candle:**\n" +
                            "\n".join(f"- {builder.describe(r)}" for r in s["rules"]))
                if s["unchecked"]:
                    st.markdown("**What the teacher also said, that it can't check on a chart yet:**\n" +
                                "\n".join(f"- {u}" for u in s["unchecked"]))
                st.caption(f"Stop {s['stop_atr']:g} ATR · target {s['rr']:g}R · {stats_line(s)}")
                if s["why"]:
                    st.markdown(f"*Why:* {s['why']}")
                if s["quote"]:
                    st.markdown(f"> {s['quote']}")
                per = s["stats"].get("charts", {})
                if per:
                    st.dataframe(pd.DataFrame(per).T[["trades", "win_rate", "avg_r", "edge"]], width="stretch")
                a1, a2, a3 = st.columns([2, 2, 1])
                cur = s["status"] if s["status"] in ("pinned", "off") else "auto"
                new = a1.selectbox("Use it?", list(labels), index=list(labels).index(cur), format_func=labels.get,
                                   key=f"ap_st_{s['id']}")
                if new != cur:
                    store.update_setup(s["id"], status="candidate" if new == "auto" else new)
                    st.rerun()
                with a2:
                    rate(store, "setup", s["id"], {"name": s["name"], "direction": s["direction"],
                                                    "rules_text": [builder.describe(r) for r in s["rules"]]},
                         f"setup_{s['id']}", "Did it understand this setup right?")
                if a3.button("Delete", key=f"ap_del_{s['id']}"):
                    store.delete_setup(s["id"])
                    st.rerun()

    st.markdown("##### Concepts it learned")
    con = store.table("concepts")
    if con.empty:
        st.caption("Nothing yet - teach it a lesson.")
        return
    les = store.table("lessons").set_index("id")["title"].to_dict()
    q = st.text_input("Search", key="ap_csearch")
    cats = sorted(con.category.unique())
    pick = st.multiselect("Categories", cats, cats, key="ap_ccat")
    view = con[con.category.isin(pick)]
    if q:
        view = view[view.text.str.contains(q, case=False) | view.quote.str.contains(q, case=False)]
    for r in view.iloc[::-1].head(60).itertuples():
        c1, c2 = st.columns([5, 1.4])
        c1.markdown(f"**{r.category}** · {r.text}  \n<small>{les.get(r.lesson_id, '')} @ {r.t // 60}:{r.t % 60:02d}"
                    f"</small>", unsafe_allow_html=True)
        with c2:
            rate(store, "concept", r.id, {"text": r.text, "category": r.category}, f"concept_{r.id}", "Rate")


def chart_tab(store: Store) -> None:
    c1, c2, c3 = st.columns([1.2, 2, 1])
    market = c1.radio("Market", ["Crypto", "Stocks"], horizontal=True, key="ap_mkt")
    if market == "Crypto":
        symbol, asset = c2.selectbox("Coin", list(CRYPTO), key="ap_coin"), "crypto"
    else:
        symbol, asset = c2.text_input("Ticker", "SPY", key="ap_tick").upper().strip(), "stock"
    tf = c3.selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index("4h"), key="ap_tf")
    if not symbol:
        return
    try:
        with st.spinner("Loading candles and reading the chart..."):
            df = load_chart(symbol, asset, tf)
    except Exception as e:
        st.error(f"Couldn't load {symbol} {tf}: {e}")
        return
    chart_name = f"{symbol} {tf}"
    setups = store.setups()
    concepts = [dict(r) for r in store.db.execute("SELECT * FROM concepts").fetchall()]
    d = brain.decide(df, setups, concepts)
    brain.grade_journal(store, df, symbol, tf)
    if d["action"] != "WAIT":
        exists = store.db.execute("SELECT 1 FROM journal WHERE symbol = ? AND timeframe = ? AND bar_time = ?",
                                  (symbol, tf, str(d["bar_time"]))).fetchone()
        if not exists:
            store.log_decision({**d, "symbol": symbol, "timeframe": tf})
    rejected = {(ctx.get("time"), ctx.get("direction")) for ctx in
                (json.loads(r["context"] or "{}") for r in
                 store.db.execute("SELECT context FROM feedback WHERE setup_id IS NULL AND verdict = 'bad'"))
                if ctx.get("chart") == chart_name}
    missed = brain.missed_trades(df.iloc[-1000:], setups, rejected=rejected)
    past = brain.history_signals(df, setups)
    lv = brain.levels(df)

    # --- the call ---
    color = {"WAIT": "🟡"}.get(d["action"], "🟢" if d["action"].endswith("LONG") else "🔴")
    st.subheader(f"{color} {d['action']}" + (f" · {d['confidence']}% confident" if d["entry"] else "")
                 + f" · {symbol} {tf} · candle {d['bar_time']:%Y-%m-%d %H:%M} UTC")
    if d["entry"]:
        st.markdown(f"Entry **{price(d['entry'])}** · stop **{price(d['stop'])}** · target **{price(d['target'])}**")
    for r in d["reasons"]:
        st.markdown(f"- {r}")
    fired = [ch["setup"]["id"] for ch in d["checks"] if ch["fired"]]
    rate(store, "decision", f"{chart_name}|{d['bar_time']}",
         {"action": d["action"], "chart": chart_name, "time": str(d["bar_time"]), "setups": fired,
          "reasons": d["reasons"]}, f"dec_{chart_name}_{d['bar_time']}", "Rate this call")

    # --- the chart ---
    markers, lines, segs = [], [], []
    for p in past:
        win = p["r"] is not None and p["r"] > 0
        markers.append({"time": int(p["time"].timestamp()), "position": "belowBar" if p["direction"] == "long"
                        else "aboveBar", "color": "#22c55e" if win else "#ef4444" if p["r"] is not None else "#94a3b8",
                        "shape": "arrowUp" if p["direction"] == "long" else "arrowDown",
                        "text": p["setup"][:18] + ("" if p["r"] is None else f" {p['r']:+.1f}R")})
    for m in missed:
        markers.append({"time": int(m["time"].timestamp()), "position": "belowBar" if m["direction"] == "long"
                        else "aboveBar", "color": "#facc15", "shape": "circle",
                        "text": f"Missed {m['direction']} +{m['move_pct']:.0f}%"})
    markers.sort(key=lambda x: x["time"])
    for z in lv:
        lines.append({"price": z["price"], "title": f"{z['kind']} ({z['touches']})",
                      "color": "#64748b" if z["kind"] == "support" else "#a855f7"})
    if d["entry"]:
        lines += [{"price": d["entry"], "title": "entry", "color": "#38bdf8"},
                  {"price": d["stop"], "title": "stop", "color": "#ef4444"},
                  {"price": d["target"], "title": "target", "color": "#22c55e"}]
    sit = d["situation"]
    if sit:
        p = sit["pattern"]
        end = sit.get("break_bar") or len(df) - 1
        a, b = df.index[p.start], df.index[min(end, len(df) - 1)]
        segs += [{"points": [(a, p.up(p.start)), (b, p.up(end))], "color": "#facc15"},
                 {"points": [(a, p.lo(p.start)), (b, p.lo(end))], "color": "#facc15"}]
    tvchart.show(df, markers, lines, height=520, show_bars=250, segments=segs)
    st.caption("Arrows: where its active setups fired (green = won, red = lost, grey = too recent). 🟡 dots: good "
               "trades it missed (hindsight). Yellow lines: the chart pattern it sees. Dashed: support / resistance "
               "and its entry / stop / target.")

    # --- missed trades ---
    st.markdown(f"##### 🟡 Good trades it missed ({len(missed)}, last 1,000 candles)")
    st.caption("Spots where a trade would have hit 3× its risk before the stop, and none of its active setups fired. "
               "Tell it whether you'd have taken each one - 👍 ones are learned from, 👎 ones are ignored.")
    for m in missed[::-1]:
        feats = [builder.describe({"rule": f}) for f in m["features"]]
        with st.expander(f"{m['time']:%Y-%m-%d %H:%M} · missed {m['direction']} at {price(m['price'])} · ran "
                         f"{m['move_pct']:+.1f}% ({m['move_atr']:.0f} ATR)"):
            st.markdown("**What was on the chart:** " + (", ".join(feats) if feats else "nothing it knows yet"))
            ctx = {"chart": chart_name, "time": str(m["time"]), "direction": m["direction"]}
            b1, b2, b3 = st.columns([1, 1, 2])
            if b1.button("👍 I'd take that", key=f"ap_mg_{chart_name}_{m['i']}"):
                store.feedback(None, "good", "", ctx)
                st.toast("Noted - it'll learn from this one.")
            if b2.button("👎 Not a real trade", key=f"ap_mb_{chart_name}_{m['i']}"):
                store.feedback(None, "bad", "", ctx)
                st.rerun()
            with b3:
                rate(store, "missed", f"{chart_name}|{m['time']}|{m['direction']}",
                     {"name": f"missed {m['direction']} {chart_name} {m['time']}", **ctx, "features": m["features"]},
                     f"missed_{chart_name}_{m['i']}", "Rate this find")
    if st.button("🧠 Learn from the trades it missed on this chart", key="ap_mine"):
        with st.spinner("Looking for what those trades had in common..."):
            from apprentice.__main__ import feedback_marks
            conf, rej = feedback_marks(store)
            st.session_state["ap_cands"] = brain.mine({chart_name: df}, "long", conf, rej) + \
                brain.mine({chart_name: df}, "short", conf, rej)
    for i, c in enumerate(st.session_state.get("ap_cands", [])):
        tr, te = c["train"], c["test"]
        ok = "✅ held up on unseen data" if c["held_up"] else "⚠️ didn't hold up on unseen data"
        st.markdown(f"**{c['name']}** - first 70% of history: {tr['trades']} trades, edge {tr['edge']:+.2f}R · "
                    f"last 30% (never seen): {te['trades']} trades, edge "
                    f"{(te['edge'] or 0):+.2f}R · {ok}")
        if st.button("Adopt this setup", key=f"ap_adopt_{i}"):
            sid = store.add_setup({"name": c["name"], "direction": c["direction"], "rules": c["rules"],
                                   "source": "missed", "why": f"Learned from missed trades on {chart_name}."})
            s = next(x for x in store.setups() if x["id"] == sid)
            total = brain.combine({chart_name: brain.evaluate(df, s)})
            w = brain.weight(total)
            store.update_setup(sid, stats={"total": total, "charts": {chart_name: total}}, weight=w,
                               status=brain.status_for(s, total, w))
            st.success("Adopted - press 🎓 Learn now to test it on more charts.")


def paper_tab(store: Store, lp_store=None) -> None:
    st.caption("The apprentice trades on its own in the paper account: whenever one of its active setups fires on "
               "a closed candle. As it learns, what it trades changes by itself.")
    if lp_store is not None:
        c1, c2, c3 = st.columns([2, 1, 1])
        coin = c1.selectbox("Coin", list(CRYPTO), key="ap_pt_coin")
        tf = c2.selectbox("Timeframe", ["1h", "4h", "1D"], index=1, key="ap_pt_tf")
        cap = c3.number_input("Paper capital $", 100.0, 1e7, 10_000.0, 1000.0, key="ap_pt_cap")
        name = f"Apprentice {coin} {tf}"
        active = {r["name"] for r in lp_store.runs(active_only=True)}
        run = {"name": name, "strategy": STRATEGY_KEY, "symbol": CRYPTO[coin], "asset": "crypto", "timeframe": tf,
               "shorts": False, "capital": cap, "fraction": 1.0, "stop": 0.04,
               "take_profits": [[0.06, 0.5], [0.12, 0.5]]}
        key = f"ap_pt_{name}"

        def toggle():
            if st.session_state[key]:
                lp_store.upsert_run({**run, "active": 1, "source": "apprentice"})
            else:
                lp_store.set_active(name, False)
        st.checkbox(f"📄 Let the apprentice paper trade {coin} {tf}", value=name in active, key=key, on_change=toggle)
        st.caption("Spot crypto is long-only. Results show in 📊 Dashboard like any other run.")
    st.markdown("##### Its journal (every trade call it made on a chart)")
    j = store.table("journal")
    if j.empty:
        st.caption("No calls yet - open 📈 Chart.")
        return
    graded = j.dropna(subset=["r"])
    if len(graded):
        m = st.columns(3)
        m[0].metric("Calls graded", len(graded))
        m[1].metric("Right", f"{(graded.r > 0).mean() * 100:.0f}%")
        m[2].metric("Average", f"{graded.r.mean():+.2f}R")
    st.dataframe(j.iloc[::-1][["at", "symbol", "timeframe", "bar_time", "action", "confidence", "entry", "stop",
                               "target", "outcome", "r"]], hide_index=True, width="stretch")


def render(lp_store=None) -> None:
    store = get_store(db_path())
    st.caption("🧠 A trading apprentice you teach. Feed it streams and videos (or whole playlists); it learns the "
               "setups, tests them on real prices, reads charts, decides when to trade, and learns from the good "
               "trades it missed. Rate its accuracy anywhere you see ⭐ - it learns from your comments too.")
    t1, t2, t3, t4 = st.tabs(["📚 Teach", "🧠 What it knows", "📈 Chart", "📄 Paper trade & journal"])
    with t1:
        teach_tab(store)
    with t2:
        knowledge_tab(store)
    with t3:
        chart_tab(store)
    with t4:
        paper_tab(store, lp_store)
