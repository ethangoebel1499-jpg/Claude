"""Add a stream: load -> store -> chunk -> extract -> de-duplicate -> save."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from . import transcripts
from .extract import backend as make_backend, mmss
from .store import Store

FILENAME = re.compile(r"^(\d{4}-\d{2}-\d{2})[ _T](\d{2})[-:h]?(\d{2})[ _-]*(.*)$")


def parse_start(text: str, tz: str) -> pd.Timestamp:
    ts = pd.Timestamp(text)
    if ts.tzinfo is None:
        ts = ts.tz_localize(ZoneInfo(tz))
    return ts.tz_convert("UTC")


def from_filename(path: Path, tz: str) -> tuple[pd.Timestamp, str] | None:
    """'2026-10-28 2000 Bitcoin live.txt' -> (start, 'Bitcoin live')."""
    m = FILENAME.match(path.stem)
    if not m:
        return None
    d, hh, mm, title = m.groups()
    return parse_start(f"{d} {hh}:{mm}", tz), (title.strip() or path.stem)


def _clean_num(x):
    try:
        v = float(x)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def add_stream(store: Store, path: str | Path, start: pd.Timestamp, title: str, cfg: dict,
               llm=None, reextract: bool = False, log=print) -> int:
    path = Path(path)
    sid = store.find_stream(title, str(start))
    if sid and not reextract:
        log(f"already added: {title} ({start}); use --reextract to redo")
        return sid
    if sid:
        store.clear_extraction(sid)
        segs = [tuple(x) for x in store.db.execute("SELECT t, text FROM segments WHERE stream_id = ? ORDER BY t",
                                                    (sid,)).fetchall()]
    else:
        segs = transcripts.load(path, cfg.get("whisper_model", "small"))
        sid = store.add_stream(title, str(start), str(path), segs, datetime.now(timezone.utc).isoformat())
    llm = llm or make_backend(cfg)
    parts = transcripts.chunks(segs, cfg.get("chunk_minutes", 8), cfg.get("chunk_overlap", 1))
    log(f"{title}: {len(segs)} lines, {segs[-1][0] // 60} min, {len(parts)} chunks -> extracting with "
        f"{cfg.get('backend', 'ollama')}")
    seen_calls, seen_rules, n_calls, n_rules, n_levels, failed = [], set(), 0, 0, 0, 0
    for i, (_, _, body) in enumerate(parts, 1):
        try:
            res = llm.extract(body, title, str(start.date()))
        except Exception as e:
            log(f"  chunk {i}/{len(parts)} failed: {e}")
            failed += 1
            continue
        for c in res.get("calls", []):
            t = mmss(c.get("time", ""))
            asset = str(c.get("asset", "")).upper().strip()
            if not asset or c.get("direction") not in ("long", "short"):
                continue
            targets = sorted({float(x) for x in c.get("targets", []) if _clean_num(x)})
            key = (asset, c["direction"], _clean_num(c.get("entry")), _clean_num(c.get("stop")), tuple(targets))
            if any(k == key and abs(t - kt) <= 180 for k, kt in seen_calls):
                continue                     # same call caught by two overlapping chunks
            seen_calls.append((key, t))
            store.add_call(sid, {"t": t, "said_at": str(start + pd.Timedelta(seconds=t)), "asset": asset,
                                 "direction": c["direction"], "kind": c.get("kind", "idea"),
                                 "timeframe": c.get("timeframe", ""), "entry": key[2], "stop": key[3],
                                 "targets": targets, "conditions": c.get("conditions", []),
                                 "quote": c.get("quote", "")})
            n_calls += 1
        for r in res.get("rules", []):
            norm = re.sub(r"\W+", " ", r.get("rule", "").lower()).strip()
            if not norm or norm in seen_rules:
                continue
            seen_rules.add(norm)
            store.add_rule(sid, {"t": mmss(r.get("time", "")), "category": r.get("category", "other"),
                                 "rule": r["rule"], "quote": r.get("quote", "")})
            n_rules += 1
        for lv in res.get("levels", []):
            if _clean_num(lv.get("price")):
                store.add_level(sid, {"t": mmss(lv.get("time", "")), "asset": str(lv.get("asset", "")).upper(),
                                      "price": float(lv["price"]), "kind": lv.get("kind", "level")})
                n_levels += 1
        store.commit()
        log(f"  chunk {i}/{len(parts)}: {n_calls} calls, {n_rules} rules, {n_levels} levels so far")
    if failed:
        log(f"{failed} of {len(parts)} chunks failed; the transcript is saved - fix the problem and rerun "
            f"with --reextract to extract it again")
    return sid
