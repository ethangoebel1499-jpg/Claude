"""apprentice - a trading agent you teach with streams, then let it analyse charts and trade.

  python -m apprentice teach <YouTube video or playlist link | transcript file> [--title "..."]
                               a playlist is learned video by video; videos already learned are skipped
  python -m apprentice import-streams   learn from everything the stream study agent already extracted
  python -m apprentice learn [--charts BTC:4h ETH:4h SOL:4h BTC:1D] [--no-mine]
                               test every setup on real prices, keep what works, and learn new setups
                               from the good trades it missed
  python -m apprentice analyze BTC 4h   its call on the latest candle, with reasons
  python -m apprentice list             what it knows

AI model: the same as the stream agent (⚙️ Account: local Ollama or Claude). --no-ai reads lessons
with simple keyword matching instead (rough, but works with no model at all).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from apprentice import brain, teach, youtube  # noqa: E402
from apprentice.store import Store  # noqa: E402
from apprentice.strategy import db_path  # noqa: E402

DEFAULT_CHARTS = ["BTC:4h", "ETH:4h", "SOL:4h", "BTC:1h", "ETH:1h", "SOL:1h", "BTC:1D", "ETH:1D"]


def model(no_ai: bool):
    """The AI that reads lessons (same settings as the stream agent), or None for keyword reading."""
    if no_ai:
        return None
    from streamlearn.__main__ import load_cfg, local_ai
    from streamlearn.extract import backend
    cfg = load_cfg("streamlearn.toml")
    cfg["screen"] = False
    local_ai(cfg)
    return backend(cfg)


def charts(specs: list[str], log=print) -> dict:
    from livepaper.history import CRYPTO, history
    out = {}
    for spec in specs:
        sym, _, tf = spec.partition(":")
        sym, tf = sym.upper(), tf or "4h"
        asset = "crypto" if sym in CRYPTO else "stock"
        try:
            df = history(sym, asset, tf)
            out[f"{sym} {tf}"] = df.iloc[-3000:]
            log(f"  loaded {sym} {tf}: {len(out[f'{sym} {tf}'])} candles")
        except Exception as e:
            log(f"  couldn't load {sym} {tf}: {e}")
    return out


def teach_source(store: Store, source: str, title: str | None, llm, log=print, relearn: bool = False) -> dict:
    total = {"lessons": 0, "concepts": 0, "setups": 0, "skipped": 0, "failed": 0}
    k = youtube.kind(source)
    if k:
        vids = youtube.videos(source)
        log(f"{'Playlist' if k == 'playlist' else 'Video'}: {len(vids)} video(s)")
        for n, v in enumerate(vids, 1):
            if store.has_video(v["id"]) and not relearn:
                log(f"[{n}/{len(vids)}] already learned: {v['title'] or v['id']}")
                total["skipped"] += 1
                continue
            try:
                meta, segs = youtube.captions(v["url"])
            except Exception as e:
                log(f"[{n}/{len(vids)}] skipped {v['title'] or v['id']}: {e}")
                total["failed"] += 1
                continue
            name = meta["title"] or v["title"] or v["id"]
            log(f"[{n}/{len(vids)}] {name}: {len(segs)} caption lines, {segs[-1][0] // 60} min")
            lid = store.add_lesson(name, v["url"], len(segs), v["id"], v["url"], v.get("playlist"))
            try:
                got = teach.learn_lesson(store, lid, segs, name, llm, log)
                store.finish_lesson(lid, "learned", f"{got['concepts']} concepts, {got['setups']} setups")
                total["lessons"] += 1
                total["concepts"] += got["concepts"]
                total["setups"] += got["setups"]
            except Exception as e:
                store.finish_lesson(lid, "failed", str(e))
                log(f"  failed: {e}")
                total["failed"] += 1
    else:
        p = Path(source)
        if p.suffix.lower() in {".mp3", ".m4a", ".wav", ".mp4", ".mkv", ".webm", ".mov"}:
            from streamlearn.transcripts import load
            segs = load(p)
        else:
            segs = teach.segments(p.read_text(encoding="utf-8", errors="replace"))
        name = title or p.stem
        lid = store.add_lesson(name, str(p), len(segs))
        got = teach.learn_lesson(store, lid, segs, name, llm, log)
        store.finish_lesson(lid, "learned", f"{got['concepts']} concepts, {got['setups']} setups")
        total.update(lessons=1, concepts=got["concepts"], setups=got["setups"])
    log(f"RESULT lessons={total['lessons']} concepts={total['concepts']} setups={total['setups']} "
        f"skipped={total['skipped']} failed={total['failed']}")
    return total


def import_streams(store: Store, sl_db: str, llm, log=print) -> dict:
    """Everything the stream study agent pulled out of its streams becomes a lesson here too."""
    from streamlearn.store import Store as SLStore
    sl = SLStore(sl_db)
    streams = sl.table("streams")
    rules, pats = sl.table("rules"), sl.table("patterns")
    got = {"lessons": 0, "concepts": 0, "setups": 0}
    for s in streams.itertuples():
        key = f"stream:{s.id}:{s.start}"
        if store.has_video(key):
            continue
        r = rules[rules.stream_id == s.id]
        p = pats[pats.stream_id == s.id]
        lines = [(int(x.t), f"Rule ({x.category}): {x.rule}. He said: {x.quote}") for x in r.itertuples()]
        lines += [(int(x.t), f"Pattern on {x.asset} {x.timeframe}: {x.pattern_name or x.pattern}, expects break "
                             f"{x.bias}. Break up: {x.if_up}. Break down: {x.if_down}. {x.summary}")
                  for x in p.itertuples()]
        lines.sort()
        if not lines:
            continue
        lid = store.add_lesson(f"Stream: {s.title}", "stream agent", len(lines), key, s.url)
        res = teach.learn_lesson(store, lid, lines, s.title, llm, log)
        store.finish_lesson(lid, "learned", f"{res['concepts']} concepts, {res['setups']} setups")
        got["lessons"] += 1
        got["concepts"] += res["concepts"]
        got["setups"] += res["setups"]
    log(f"RESULT lessons={got['lessons']} concepts={got['concepts']} setups={got['setups']} skipped=0 failed=0")
    return got


def learn(store: Store, specs: list[str], mine: bool = True, log=print) -> None:
    log("Loading charts...")
    cs = charts(specs, log)
    if not cs:
        raise SystemExit("no charts could be loaded")
    if mine:
        log("Looking at the good trades I missed...")
        confirmed, rejected = feedback_marks(store)
        known = {(s["direction"], tuple(sorted(r["rule"] for r in s["rules"]))) for s in store.setups()}
        for d in ("long", "short"):
            for c in brain.mine(cs, d, confirmed, rejected):
                sig = (d, tuple(sorted(r["rule"] for r in c["rules"])))
                if sig in known or not c["held_up"]:
                    continue
                known.add(sig)
                store.add_setup({"name": c["name"], "direction": d, "rules": c["rules"], "stop_atr": 1.5, "rr": 2.0,
                                 "why": f"Found in missed trades: in the first 70% of history {c['train']['trades']} "
                                        f"trades, edge {c['train']['edge']:+.2f}R; on the last 30% (unseen) "
                                        f"{c['test']['trades']} trades, edge {c['test']['edge']:+.2f}R.",
                                 "source": "missed"})
                log(f"  learned a new {d} setup from missed trades: {c['name']}")
    log("Testing every setup on real prices...")
    res = brain.learn(store, cs, log)
    from . import autoreview
    if autoreview.enabled():
        log("Reviewing what it knows (deletes junk, turns off broken setups)...")
        autoreview.run(store, log)
        res = [r for r in res if r["id"] in {s["id"] for s in store.setups()}]
        off = {s["id"] for s in store.setups("off")}
        res = [r for r in res if r["id"] not in off]
    active = sum(1 for r in res if r["status"] in ("active", "pinned"))
    log(f"RESULT tested={len(res)} active={active}")


def _auto_review(store: Store) -> None:
    from . import autoreview
    if autoreview.enabled():
        try:
            autoreview.run(store)
        except Exception as e:
            print(f"auto-review failed: {e}")


def feedback_marks(store: Store) -> tuple[dict, dict]:
    """Your 👍 / 👎 on missed trades: {chart: {times}}."""
    import json
    conf, rej = {}, {}
    for r in store.db.execute("SELECT verdict, context FROM feedback WHERE setup_id IS NULL").fetchall():
        ctx = json.loads(r["context"] or "{}")
        if ctx.get("chart") and ctx.get("time"):
            (conf if r["verdict"] == "good" else rej).setdefault(ctx["chart"], set()).add(ctx["time"])
    return conf, rej


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m apprentice", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("teach")
    t.add_argument("source")
    t.add_argument("--title")
    t.add_argument("--no-ai", action="store_true")
    t.add_argument("--then-learn", action="store_true", help="test what it learned on real prices right after")
    t.add_argument("--relearn", action="store_true", help="learn videos again even if it already learned them")
    i = sub.add_parser("import-streams")
    i.add_argument("--streams-db", default=None)
    i.add_argument("--no-ai", action="store_true")
    i.add_argument("--then-learn", action="store_true")
    le = sub.add_parser("learn")
    le.add_argument("--charts", nargs="*", default=DEFAULT_CHARTS)
    le.add_argument("--no-mine", action="store_true")
    an = sub.add_parser("analyze")
    an.add_argument("symbol")
    an.add_argument("timeframe", nargs="?", default="4h")
    sub.add_parser("list")
    a = ap.parse_args()
    store = Store(a.db or db_path())
    if a.cmd == "teach":
        teach_source(store, a.source, a.title, model(a.no_ai), relearn=a.relearn)
        _auto_review(store)
        if a.then_learn:
            learn(store, DEFAULT_CHARTS)
    elif a.cmd == "import-streams":
        from livepaper import settings
        import_streams(store, a.streams_db or settings.get("streamlearn_db", "streamlearn.db"), model(a.no_ai))
        _auto_review(store)
        if a.then_learn:
            learn(store, DEFAULT_CHARTS)
    elif a.cmd == "learn":
        learn(store, a.charts, not a.no_mine)
    elif a.cmd == "analyze":
        cs = charts([f"{a.symbol}:{a.timeframe}"])
        if not cs:
            raise SystemExit("couldn't load that chart")
        df = next(iter(cs.values()))
        d = brain.decide(df, store.setups(), [dict(r) for r in store.db.execute("SELECT * FROM concepts")])
        print(f"{a.symbol.upper()} {a.timeframe} · candle {d['bar_time']:%Y-%m-%d %H:%M} UTC · {d['action']}"
              + (f" ({d['confidence']}% confident) entry {d['entry']:,.2f} stop {d['stop']:,.2f} "
                 f"target {d['target']:,.2f}" if d["entry"] else ""))
        for r in d["reasons"]:
            print(" ", r)
    elif a.cmd == "list":
        les = store.table("lessons")
        print(f"{len(les)} lessons, {len(store.table('concepts'))} concepts")
        for s in store.setups():
            tot = s["stats"].get("total", {})
            print(f"  [{s['status']:9}] {s['direction']:5} {s['name']} · weight {s['weight']:+.2f} · "
                  f"{tot.get('trades', 0)} trades, edge {tot.get('edge')}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
