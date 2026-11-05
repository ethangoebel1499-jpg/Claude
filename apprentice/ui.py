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

from apprentice import autoreview, review, brain, youtube
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
              "missed": "Missed trades it found", "signal": "Past setup signals (Claude)"}


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
        if les == 0 and skip and not fail:
            return "ℹ️ Already learned - nothing new. Use 🔁 Re-learn on the video (Lessons below) to read it again"
        if les == 0 and fail:
            why = re.search(r"skipped .*?: (.+)", text)
            reason = why.group(1).split(" - paste")[0] if why else ""
            return f"⚠️ Couldn't read the video{': ' + reason if reason else ''} - paste its transcript instead"
        parts.append(f"{les} lesson(s), {con} concepts, {sets} new setups"
                     + (f", {skip} already learned" if skip else "") + (f", {fail} couldn't be read" if fail else ""))
        if not sets:
            parts.append("no new setups - see 🧠 What it knows (it may already know them, or they need rules)")
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
    relearn = st.checkbox("Learn videos again even if it already learned them", False,
                          help="Use after an update that taught it new rules, or after changing the AI model")
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
        spawn(["teach", url, *extra, *(["--relearn"] if relearn else [])],
              f"{'Playlist' if k == 'playlist' else 'Video'}: {url[-40:]}")
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
            c1, c3, c2 = st.columns([6, 1.2, 1])
            n_set = store.db.execute("SELECT count(*) FROM setups WHERE lesson_id = ?", (r.id,)).fetchone()[0]
            n_con = store.db.execute("SELECT count(*) FROM concepts WHERE lesson_id = ?", (r.id,)).fetchone()[0]
            c1.markdown(f"**{r.title}** · {r.status} · {n_con} concepts, {n_set} setups"
                        + (f" · [video]({r.url})" if r.url else "")
                        + (f" · playlist *{r.playlist}*" if r.playlist else ""))
            src = r.url or (r.source if r.source and Path(str(r.source)).exists() else None)
            if src and c3.button("🔁 Re-learn", key=f"ap_relearn_{r.id}",
                                 help="Read this lesson again with the current AI and rules (replaces what it learned "
                                      "from it; pinned setups stay)"):
                store.forget_lesson(r.id)
                spawn(["teach", src, "--relearn", "--title", r.title, *extra], f"Re-learn: {r.title}")
                st.rerun()
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


def after_entry_line(a: dict) -> str:
    return (f"After entry (median over {a['trades']} past trades): went up to **{a.get('mfe_r', 0):+.2f}R** for you, "
            f"**{a.get('mae_r', 0):+.2f}R** against you · {a.get('reached_1r_pct', 0):.0f}% reached +1R, "
            f"{a.get('reached_target_pct', 0):.0f}% the target · held {a.get('bars_held', 0):g} candles"
            + (f" (winners took {a['bars_to_win']:g})" if a.get("bars_to_win") else ""))


def after_entry_lines(store: Store, s: dict) -> None:
    """What this setup's trades did after the entry, what that says about the trigger, and a tested tweak."""
    a = s["stats"].get("after") or {}
    if a.get("trades"):
        st.caption(after_entry_line(a))
    names = {"fixed": "the 3% stop", "atr": f"{s['stop_atr']:g} ATR stop", "swing": "beyond the last swing low/high"}
    for tf, plan in (s["stats"].get("exit_plan") or {}).items():
        if "stop" not in plan:
            continue
        per = plan.get("stops", {})
        line = " · ".join(f"{names[m]}: {v['train_pct']:+.2f}% / {v['test_pct']:+.2f}%" for m, v in per.items()
                          if v.get("train_pct") is not None and v.get("test_pct") is not None)
        ls, lr = plan.get("learned_stop", plan["stop"]), plan.get("learned_reversal", plan["reversal"])
        status = ("✅ in use" if plan.get("deployed") else
                  "⏸️ not in use - all setups together did worse with it on unseen prices, so it trades with the 3% stop"
                  if (ls != "fixed" or lr) else "")
        st.caption(f"🛡️ {tf}: learned stop **{names.get(ls, ls)}**, early exit on a reversal **{'on' if lr else 'off'}**"
                   + (f" · {status}" if status else "") + f" · per trade, past / unseen prices: {line}")
    for h in s["stats"].get("hints") or []:
        st.caption(f"💡 {h}")
    tn = s["stats"].get("tune")
    if tn:
        t1, t2 = st.columns([5, 1.3])
        t1.markdown(f"🎯 **Tuning that held up on unseen prices:** {tn['change']} - edge "
                    f"{tn['test']['was']:+.2f}R → {tn['test']['edge']:+.2f}R on the last 30% "
                    f"({tn['test']['trades']} trades), {tn['train']['was']:+.2f}R → {tn['train']['edge']:+.2f}R before.")
        if t2.button("Apply tuning", key=f"ap_tune_{s['id']}"):
            store.update_setup(s["id"], **tn["set"], stats={**s["stats"], "tune": None})
            st.toast("Tuned - re-tested on the next 🎓 Learn")
            st.rerun()


def after_entry_panel(store: Store, setups: list[dict]) -> None:
    """Across the setups it trades: what happens after entries, and the early-exit switch."""
    from livepaper import settings
    live = [s for s in setups if s["status"] in ("active", "pinned") and (s["stats"].get("after") or {}).get("trades")]
    st.markdown("##### 📈 After the entry")
    if not live:
        st.caption("Press 🎓 Learn now - it then follows every past trade of each setup after the entry (how far it "
                   "went for and against you, how long it took) and tries nearby trigger settings.")
    else:
        n = sum(s["stats"]["after"]["trades"] for s in live)
        w = lambda k: sum((s["stats"]["after"].get(k) or 0) * s["stats"]["after"]["trades"] for s in live) / n
        c = st.columns(4)
        c[0].metric("Best move for you (median)", f"{w('mfe_r'):+.2f}R")
        c[1].metric("Worst move against (median)", f"{w('mae_r'):+.2f}R")
        c[2].metric("Reached +1R", f"{w('reached_1r_pct'):.0f}%")
        tuned = sum(1 for s in live if s["stats"].get("tune"))
        c[3].metric("Setups with a tested tweak", tuned)
        saved = sum(s["stats"]["after"].get("reversal_saved_r", 0) for s in live)
        times = sum(s["stats"]["after"].get("reversal_exits", 0) for s in live)
    tfs = store.meta_get("timeframes") or {}
    if tfs:
        st.markdown("**⏯️ Timeframes** (🎓 Learn pauses one where all its setups together lost money on unseen prices):")
        st.caption(" · ".join(f"{'✅' if v['trade'] else '⏸️'} {tf} {'trading' if v['trade'] else 'PAUSED'} "
                              f"({v['unseen_pct']:+.1f}%, {v['trades']} trades)" for tf, v in sorted(tfs.items())))
    from apprentice.strategy import SAFETY
    learned = store.meta_get("safety") or {}
    st.markdown("**🛡️ Safety rules** (on unless 🎓 Learn found the strategy did better without them):")
    for rule, text in SAFETY.items():
        per = {tf: v[rule] for tf, v in learned.items() if rule in v}
        if not per:
            st.caption(f"✅ {text} - on everywhere (not checked by 🎓 Learn yet)")
            continue
        st.caption(f"{'✅' if all(v['on'] for v in per.values()) else '⚠️'} {text} - " + " · ".join(
            f"{tf} {'on' if v['on'] else 'OFF'} ({v['with']:+.1f}% with, {v['without']:+.1f}% without)"
            for tf, v in sorted(per.items())))
    from apprentice.strategy import reversal_mode
    modes = {"learned": "Only for setups where 🎓 Learn showed it helps (past and unseen prices)",
             "off": "Never - always wait for the stop or the targets", "always": "Always"}
    cur = reversal_mode()
    new = st.radio("Close a losing trade early when a confirmed reversal prints", list(modes), list(modes).index(cur),
                   format_func=modes.get, horizontal=False,
                   help="Confirmed reversal = a divergence confirmed by the dot on a closed candle (bearish divergence "
                        "+ red dot against a long, bullish + green dot against a short), or the structure breaking "
                        "with money flow behind it. Only ever closes a trade that is losing; winners keep their "
                        "targets. The lessons say to stick to the planned stop, so it's only used where the data "
                        "clearly says otherwise.")
    if live:
        plans = lambda s: [p for p in (s["stats"].get("exit_plan") or {}).values() if isinstance(p, dict)]
        used = [s for s in live if any(p.get("reversal") for p in plans(s))]
        own = [s for s in live if any(p.get("stop") in ("atr", "swing") for p in plans(s))]
        st.caption(f"Learned so far: **{len(own)}** of {len(live)} trading setups use their own stop "
                   f"(ATR or beyond the last swing), the rest the 3% stop · the early exit helped for **{len(used)}**"
                   + (f" ({', '.join('#' + str(s['id']) for s in used)})" if used else "")
                   + (f" · across all of them it would have closed {times} losers early, {saved:+.1f}R in total"
                      if times else ""))
    if new != cur:
        settings.save({"apprentice_reversal_exit": new})
        st.toast("Saved - the trader uses it from its next price check")


def auto_panel(store: Store) -> None:
    """The apprentice's own review (runs after every lesson and every Learn) and the automatic switches."""
    from livepaper import settings
    st.markdown("##### 🤖 Automatic review")
    c1, c2 = st.columns(2)
    a_new = c1.toggle("Review everything it learns by itself", bool(settings.get("apprentice_auto_review", True)),
                      help="After every lesson and every 🎓 Learn: deletes junk concepts (ads, leverage talk, transcript "
                           "fragments, dated commentary, duplicates), turns off setups that can't work or have no edge, "
                           "deletes lessons that aren't trading material - and rates and comments on everything. Never "
                           "touches what you pinned, made or rated 4-5★, or what Claude's review covers.")
    c_new = c2.toggle("Apply Claude's review automatically", bool(settings.get("apprentice_auto_apply_claude", True)),
                      help="Apply Claude's verdicts as soon as a sync brings them (then re-test with 🎓 Learn), "
                           "instead of waiting for the Apply all button.")
    if a_new != bool(settings.get("apprentice_auto_review", True)) or \
            c_new != bool(settings.get("apprentice_auto_apply_claude", True)):
        settings.save({"apprentice_auto_review": a_new, "apprentice_auto_apply_claude": c_new})
        st.toast("Saved")
    ar = store.meta_get("auto_review") or {}
    if ar:
        n = ar.get("counts", {})
        tally = lambda k: " · ".join(f"{autoreview.STARS.get(v, 0)}★ {v} {c}" for v, c in
                                     sorted({v: sum(1 for x in ar.get(k, {}).values() if x["verdict"] == v)
                                             for v in ("use", "keep", "watch", "drop")}.items()) if c)
        st.caption(f"Last run {ar.get('at', '')[:16].replace('T', ' ')} UTC: deleted {n.get('concepts_deleted', 0)} "
                   f"concepts and {n.get('lessons_deleted', 0)} lessons, turned off {n.get('setups_off', 0)} setups. "
                   f"Setups: {tally('setups')} · concepts: {tally('concepts')}. Each verdict and comment shows under "
                   "its item below.")
        dropped = [x for x in ar.get("lessons", {}).values() if x["verdict"] == "drop"]
        if dropped:
            st.caption("Deleted lessons: " + "; ".join(f"{x['title']} ({x['comment']})" for x in dropped))
    if st.button("🤖 Review everything now", key="ap_auto_now"):
        c = autoreview.run(store, log=lambda m: None)
        st.success(f"Deleted {c['concepts_deleted']} concepts and {c['lessons_deleted']} lessons, turned off "
                   f"{c['setups_off']} setups.")
        st.rerun()


def claude_panel(store: Store, charts: list[str] | None = None) -> None:
    """Claude's latest review of the setups, concepts and lessons (comes back with each sync)."""
    from livepaper import sync
    rv = review.load()
    st.markdown("##### 🤖 Claude's review")
    status = sync.last_status()
    if not rv:
        st.caption("No review yet. Turn on **Share with Claude** in ⚙️ Account (with a GitHub token), then ask Claude "
                   "to review the apprentice - its verdicts on every setup and concept show up here."
                   + (f" Last share: {status.get('at', '')[:16]} - {status.get('result', '')}" if status else ""))
        return
    counts = {k: {} for k in ("setups", "concepts", "lessons")}
    for k in counts:
        for x in rv.get(k, []):
            counts[k][x.get("verdict", "?")] = counts[k].get(x.get("verdict", "?"), 0) + 1
    st.caption(f"Reviewed {rv.get('reviewed_at', '')[:16]} UTC · setups "
               + ", ".join(f"{review.ICON.get(v, '')} {n} {v}" for v, n in counts["setups"].items())
               + " · concepts " + ", ".join(f"{review.ICON.get(v, '')} {n} {v}" for v, n in counts["concepts"].items()))
    stt = review.status(store)
    s_, c_ = stt["setups"], stt["concepts"]
    concepts = (f"✅ all {c_['total']} rated, applied" if not c_["waiting"] else
                f"⏳ {len(c_['waiting'])} changes waiting - press **Apply all** below")
    setups_ = ("✅ all applied" if not s_["waiting"] else
               f"⏳ {len(s_['waiting'])} of {s_['total']} verdicts waiting - press **Apply all** below")
    tr = stt.get("trades") or {}
    trades_ = ("" if not tr.get("total") else f" · **Chart trades rated:** " +
               (f"✅ all {tr['total']} saved" if not tr["waiting"] else f"⏳ {tr['waiting']} of {tr['total']} waiting"))
    st.markdown(f"**Concepts:** {concepts} ({stt['concepts_now']} left after the drops) · **Setups:** {setups_}"
                + trades_)
    if rv.get("summary"):
        st.markdown(rv["summary"])
    lessons = {x["id"]: x for x in rv.get("lessons", []) if x.get("verdict") == "drop"}
    if lessons:
        titles = store.table("lessons").set_index("id")["title"].to_dict() if len(store.table("lessons")) else {}
        st.markdown("**Lessons Claude says not to learn from** (delete them in 📚 Teach if you agree):\n" +
                    "\n".join(f"- {titles.get(i, f'lesson {i}')}: {x.get('reason', '')}" for i, x in lessons.items()))
    c1, c2 = st.columns([1.3, 4])
    if c1.button("Apply all of Claude's verdicts", key="ap_rv_all"):
        n = review.apply_all(store)
        from apprentice.__main__ import DEFAULT_CHARTS
        from apprentice.job import running
        if running("learn"):
            st.success(f"Applied: {n['setups']} setups, {n['concepts']} concepts, {n.get('trades', 0)} chart-trade ratings. A 🎓 Learn run is already going - "
                       "press 🎓 Learn now when it finishes so it re-tests these changes.")
        else:
            spawn(["learn", "--charts", *(charts or DEFAULT_CHARTS)], "Learning run (after Claude's review)")
            st.success(f"Applied: {n['setups']} setups, {n['concepts']} concepts, {n.get('trades', 0)} chart-trade ratings - and started 🎓 Learn to re-test "
                       "them and learn their stops (progress in 📚 Teach).")
    c2.caption("Nothing Claude suggests changes anything until you press this: it applies every verdict (setups: use = "
               "📌 always use, drop = 🚫 off, fix = corrected rules; concepts: delete or reword), then starts 🎓 Learn "
               "on the charts above. Each verdict also has its own Apply button below.")


def knowledge_tab(store: Store) -> None:
    setups = store.setups()
    by = {k: [s for s in setups if s["status"] == k]
          for k in ("needs rules", "pinned", "active", "candidate", "retired", "off")}
    m = st.columns(5)
    m[0].metric("Lessons", len(store.table("lessons")))
    m[1].metric("Concepts", len(store.table("concepts")))
    m[2].metric("Active setups", len(by["active"]) + len(by["pinned"]))
    m[3].metric("Candidates", len(by["candidate"]))
    m[4].metric("Retired (didn't work)", len(by["retired"]))
    c1, c2 = st.columns([3, 1])
    picks = c1.multiselect("Test on these charts", ["BTC:4h", "ETH:4h", "SOL:4h", "XRP:4h", "BTC:1D", "ETH:1D",
                                                    "BTC:1h", "ETH:1h", "SOL:1h", "SPY:1D", "QQQ:1D"],
                           ["BTC:4h", "ETH:4h", "SOL:4h", "BTC:1h", "ETH:1h", "SOL:1h", "BTC:1D", "ETH:1D"],
                           help="Include the timeframes it paper trades: stops and early exits are learned and checked "
                                "per timeframe, and a timeframe it never learned on trades with the plain 3% stop.")
    c2.write("")
    if c2.button("🎓 Learn now", type="primary", disabled=not picks,
                 help="Tests every setup on real prices, keeps what beats random, and learns new setups from the "
                      "good trades it missed"):
        spawn(["learn", "--charts", *picks], "Learning run")
        st.success("Learning in the background - see 📚 Teach for progress.")
    st.markdown("##### Accuracy (your ratings)")
    accuracy_panel(store)
    claude_panel(store, picks)
    auto_panel(store)
    after_entry_panel(store, setups)

    st.markdown("##### Setups")
    st.caption("**Active** = tested and beat random (the apprentice trades these). **Candidate** = not enough "
               "trades yet. **Retired** = lost to random. 📌 Pin forces one on, 🚫 Off turns it off. Your ratings "
               "change how much it trusts each one.")
    labels = {"auto": "🤖 Let it decide", "pinned": "📌 Always use", "off": "🚫 Off"}
    if by["needs rules"]:
        st.markdown(f"##### ✏️ Setups it understood but couldn't put into rules ({len(by['needs rules'])})")
        st.caption("Pick the rules that match what the teacher said; it then tests the setup like any other.")
        ids = list(builder.LABELS)
        for s in by["needs rules"]:
            with st.expander(f"✏️ {s['direction'].upper()} · {s['name']}"):
                st.markdown("**What the teacher said:**\n" + "\n".join(f"- {u}" for u in s["unchecked"]))
                if s["quote"]:
                    st.markdown(f"> {s['quote']}")
                pick = st.multiselect("Rules that must all be true (default numbers; change them in 🛠️ Builder)", ids,
                                      format_func=lambda k: builder.describe({"rule": k}), key=f"ap_nr_{s['id']}")
                c1, c2 = st.columns([1, 1])
                if c1.button("Save rules", key=f"ap_nrs_{s['id']}", disabled=not pick, type="primary"):
                    store.update_setup(s["id"], rules=[{"rule": k} for k in pick], status="candidate")
                    st.rerun()
                if c2.button("Delete", key=f"ap_nrd_{s['id']}"):
                    store.delete_setup(s["id"])
                    st.rerun()
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
                after_entry_lines(store, s)
                av = autoreview.verdict(store, "setups", s["id"])
                if av:
                    st.caption(f"🤖 Auto-review: {'⭐' * av['stars']} {av['verdict']} - {av['comment']}")
                rv = review.for_setup(s)
                if rv:
                    r1, r2 = st.columns([5, 1.3])
                    r1.markdown(review.describe(rv))
                    if rv.get("verdict") == "fix" and (rv.get("fix") or {}).get("rules"):
                        r1.caption("Suggested rules: " + " AND ".join(
                            builder.describe(r) for r in rv["fix"]["rules"] if r.get("rule") in builder.RULES))
                    if review.is_applied("setups", rv):
                        r2.caption("✔ applied")
                    elif r2.button("Apply", key=f"ap_rv_{s['id']}"):
                        st.toast(review.apply_setup(store, s, rv))
                        st.rerun()
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
        rv = review.for_concept(r.id)
        av = autoreview.verdict(store, "concepts", r.id)
        if av and not rv:
            st.caption(f"🤖 Auto-review: {'⭐' * av['stars']} {av['verdict']} - {av['comment']}")
        if rv and rv.get("verdict") != "keep":
            d1, d2 = st.columns([5, 1.4])
            d1.markdown(review.describe(rv) + (f"  \n*Better:* {rv['fix_text']}" if rv.get("fix_text") else ""))
            if review.is_applied("concepts", rv):
                d2.caption("✔ applied")
            elif d2.button("Apply", key=f"ap_rvc_{r.id}"):
                st.toast(review.apply_concept(store, r.id, rv))
                st.rerun()


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
    cr = review.for_trade("decision", chart_name, d["bar_time"])
    if cr:
        st.info(f"🤖 Claude: {'⭐' * cr['stars']} {cr['reason']}"
                + ("" if review.is_applied("trades", cr) else "  \n*(saved as a rating when you press Apply all)*"))
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
    tvchart.show(df, markers, lines, height=520, show_bars=min(len(df), 600), segments=segs)
    span = df.index[-1] - df.index[0]
    st.caption(f"Loaded **{len(df):,} candles**: {df.index[0]:%Y-%m-%d %H:%M} → {df.index[-1]:%Y-%m-%d %H:%M} UTC "
               f"({span.days} days{f' {span.seconds // 3600} h' if span.days < 3 else ''}). It opens on the latest "
               "600 - press **All** / **1M** above the chart or scroll to zoom out. Arrows: where its active setups "
               "fired (green = won, red = lost, grey = too recent). 🟡 dots: good trades it missed (hindsight). Yellow "
               "lines: the chart pattern it sees. Dashed: support / resistance and its entry / stop / target.")
    if tf.endswith("m") and span < pd.Timedelta(days=5):
        st.warning("Only a few days of minute candles came back - the longer history source (Coinbase) didn't answer, "
                   "so this is Kraken's latest 720 candles. Try again in a minute, or use a 1h+ timeframe for more "
                   "history.")

    # --- Claude's rating of every arrow on this chart ---
    rated = [x for x in review.load().get("trades", []) if x.get("kind") == "signal" and x.get("chart") == chart_name]
    if rated:
        avg = sum(x["stars"] for x in rated) / len(rated)
        with st.expander(f"🤖 Claude's rating of every arrow on this chart ({len(rated)} trades, average {avg:.1f}★ · "
                         f"{sum(x['stars'] >= 4 for x in rated)} good, {sum(x['stars'] <= 2 for x in rated)} weak)"):
            st.caption("Judged on what the lessons teach - at an earlier support/resistance zone, confirmed by a "
                       "divergence / dot / reversal candle, with the trend, not chasing, a sane stop - never on how "
                       "it turned out. Saved as ratings when you press Apply all.")
            per: dict = {}
            for x in rated:
                per.setdefault(x.get("setup_name") or "?", []).append(x["stars"])
            st.dataframe(pd.DataFrame([{"setup": k, "arrows": len(v), "avg ★": round(sum(v) / len(v), 1),
                                        "good (4-5★)": sum(s >= 4 for s in v), "weak (1-2★)": sum(s <= 2 for s in v)}
                                       for k, v in per.items()]).sort_values("avg ★", ascending=False),
                         hide_index=True, use_container_width=True)
            for x in sorted(rated, key=lambda x: x["time"], reverse=True):
                st.markdown(f"- **{x['time'][:16]}** · {x['direction']} · {x.get('setup_name', '')[:40]} · "
                            f"{'⭐' * x['stars']} - {x['reason']}")

    # --- missed trades ---
    st.markdown(f"##### 🟡 Good trades it missed ({len(missed)}, last 1,000 candles)")
    st.caption("Spots where a trade would have hit 3× its risk before the stop, and none of its active setups fired. "
               "Tell it whether you'd have taken each one - 👍 ones are learned from, 👎 ones are ignored.")
    for m in missed[::-1]:
        feats = [builder.describe({"rule": f}) for f in m["features"]]
        with st.expander(f"{m['time']:%Y-%m-%d %H:%M} · missed {m['direction']} at {price(m['price'])} · ran "
                         f"{m['move_pct']:+.1f}% ({m['move_atr']:.0f} ATR)"):
            st.markdown("**What was on the chart:** " + (", ".join(feats) if feats else "nothing it knows yet"))
            cr = review.for_trade("missed", chart_name, m["time"], m["direction"])
            if cr:
                st.markdown(f"🤖 **Claude:** {'⭐' * cr['stars']} {cr['reason']}")
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


def trade_bar(lp_store) -> None:
    """The main page's 🧠 Apprentice bar: pick the markets, and it paper trades them long and short whenever one
    of its tested setups fires. Shows what it knows, its open trades and how it has done."""
    from apprentice.strategy import STOP, TAKE_PROFITS, live_setups, run_for
    from livepaper import settings
    store = get_store(db_path())
    live = live_setups()
    n_long = sum(1 for s in live if s["direction"] == "long")
    n_short = len(live) - n_long
    runs = [r for r in lp_store.runs() if r["strategy"] == STRATEGY_KEY]
    active = [r for r in runs if r["active"]]
    with st.container(border=True):
        h1, h2 = st.columns([3, 2])
        h1.markdown(f"#### 🧠 Apprentice · trades what it learned")
        h2.caption(f"{len(store.table('lessons'))} lessons · {len(live)} tested setups it trades "
                   f"({n_long} long, {n_short} short) · stop {STOP:.0%}, half off at +{TAKE_PROFITS[0][0]:.0%}, "
                   f"rest at +{TAKE_PROFITS[1][0]:.0%}")
        if not live:
            st.info("It has no tested setups yet, so it won't trade. Teach it in 🧠 Apprentice → 📚 Teach, then press "
                    "🎓 Learn now - setups that beat random get traded here automatically.")
        for r in active:                                  # runs made before: long-only, 4% stop -> today's settings
            if not r["shorts"] or r["stop"] != STOP or [list(t) for t in r["take_profits"] or []] != \
                    [list(t) for t in TAKE_PROFITS]:
                coin = {pair: c for c, pair in CRYPTO.items()}.get(r["symbol"], r["symbol"])
                lp_store.upsert_run({**run_for(coin, r["symbol"], r["timeframe"], r["capital"]), "name": r["name"],
                                     "active": 1, "source": "apprentice"})
        coins = {pair: coin for coin, pair in CRYPTO.items()}
        c1, c2, c3 = st.columns([4, 1, 1.3])
        tf = c2.selectbox("Timeframe", ["1h", "4h", "1D"], index=1, key="ap_bar_tf")
        cap = c3.number_input("Capital per market $", 100.0, 1e7, 10_000.0, 1000.0, key="ap_bar_cap")
        on = [coins[r["symbol"]] for r in active if r["timeframe"] == tf and r["symbol"] in coins]
        key = f"ap_bar_coins_{tf}"

        def apply():                                       # only when you change the list
            pick = st.session_state[key]
            have = {r["name"] for r in lp_store.runs(active_only=True)
                    if r["strategy"] == STRATEGY_KEY and r["timeframe"] == tf}
            for c in pick:
                if f"Apprentice {c} {tf}" not in have:
                    lp_store.upsert_run({**run_for(c, CRYPTO[c], tf, st.session_state.get("ap_bar_cap", cap)),
                                         "active": 1, "source": "apprentice"})
            for name in have - {f"Apprentice {c} {tf}" for c in pick}:
                lp_store.set_active(name, False)           # closes its open position on the next price check
        if key in st.session_state and sorted(st.session_state[key]) != sorted(on):
            st.session_state[key] = on                     # follow changes made elsewhere (Strategy Lab, runs list)
        c1.multiselect("Markets it paper trades (long and short)", list(CRYPTO), default=on if key not in
                       st.session_state else None, key=key, on_change=apply)
        active = [r for r in lp_store.runs() if r["strategy"] == STRATEGY_KEY and r["active"]]
        if settings.get("broker") == "webull" and settings.get("webull_crypto", "auto") == "webull":
            st.warning("Crypto runs are set to trade at Webull, which can't short spot crypto - shorts will be "
                       "refused. ⚙️ Account → Crypto runs trade in → built-in paper broker.")
        if not active:
            st.caption("Not trading yet - pick markets above.")
            return
        trades = lp_store.table("trades")
        rows = []
        for r in active:
            state = lp_store.load_state(r["name"])
            t = trades[trades.run == r["name"]]
            sig = lp_store.table("signals", r["name"])
            last = sig.iloc[-1] if len(sig) else None
            p = state.position if state else None
            rows.append({
                "Market": f"{coins.get(r['symbol'], r['symbol'])} {r['timeframe']}",
                "Position": "🟢 long" if p and p.side == "long" else "🔴 short" if p else "⚪ waiting",
                "Entry": p.entry_price if p else None,
                "Last candle": "—" if last is None else "🟢 long signal" if last.long_entry else
                               "🔴 short signal" if last.short_entry else "no setup fired",
                "Trades": len(t), "Win rate": f"{(t.pnl_usd > 0).mean() * 100:.0f}%" if len(t) else "-",
                "Closed P&L $": float(t.pnl_usd.sum()) if len(t) else 0.0,
                "Equity $": state.equity if state else r["capital"]})
        df = pd.DataFrame(rows)
        st.dataframe(df.style.format({"Entry": lambda v: "-" if v is None or pd.isna(v) else price(v),
                                      "Closed P&L $": "${:,.2f}", "Equity $": "${:,.2f}"}),
                     hide_index=True, width="stretch")
        st.caption("It enters on a closed candle when one of its tested setups fires (long or short) and exits at the "
                   "stop or the two targets. Details and a Force / Close button per market: Run details below.")


def paper_tab(store: Store, lp_store=None) -> None:
    st.caption("The apprentice trades on its own in the paper account: whenever one of its active setups fires on "
               "a closed candle. As it learns, what it trades changes by itself.")
    if lp_store is not None:
        st.caption("Pick its markets on the 📊 Dashboard (🧠 Apprentice bar at the top) - long and short, 3% stop, "
                   "half off at +3%, the rest at +6%.")
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
