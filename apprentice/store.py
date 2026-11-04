"""The apprentice's memory (apprentice.db on this computer, never pushed to GitHub).

lessons   every stream / video it was taught from (YouTube id, so a playlist isn't learned twice)
concepts  what it learned in words: rules of thumb, definitions, what to look for, mistakes to avoid
setups    trades it knows how to look for, as checkable rules (mcbot.builder rule ids) plus the parts it
          can't check on a chart yet; each with its track record on real prices and a weight
feedback  your 👍 / 👎 on missed trades (would you have taken it?)
ratings   your 1-5 star accuracy rating + comment on anything the AI produced: a setup it learned,
          a concept, a chart call, a missed trade it found
journal   every decision it made on a chart (trade long / short / wait) and how it turned out
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS lessons (id INTEGER PRIMARY KEY, title TEXT, source TEXT, video_id TEXT, url TEXT,
                                    playlist TEXT, lines INTEGER, added TEXT, status TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS concepts (id INTEGER PRIMARY KEY, lesson_id INTEGER, t INTEGER, category TEXT,
                                     text TEXT, quote TEXT);
CREATE TABLE IF NOT EXISTS setups (id INTEGER PRIMARY KEY, name TEXT, direction TEXT, rules TEXT, unchecked TEXT,
                                   stop_atr REAL, rr REAL, timeframe TEXT, why TEXT, quote TEXT, source TEXT,
                                   lesson_id INTEGER, status TEXT, weight REAL, stats TEXT, created TEXT,
                                   updated TEXT);
CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY, setup_id INTEGER, verdict TEXT, note TEXT,
                                     context TEXT, at TEXT);
CREATE TABLE IF NOT EXISTS ratings (id INTEGER PRIMARY KEY, kind TEXT, ref TEXT, stars INTEGER, comment TEXT,
                                    snapshot TEXT, at TEXT);
CREATE TABLE IF NOT EXISTS journal (id INTEGER PRIMARY KEY, at TEXT, symbol TEXT, timeframe TEXT, bar_time TEXT,
                                    action TEXT, confidence REAL, entry REAL, stop REAL, target REAL,
                                    reasons TEXT, outcome TEXT, r REAL);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path = "apprentice.db"):
        self.db = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        if "job" not in {r[1] for r in self.db.execute("PRAGMA table_info(lessons)")}:
            self.db.execute("ALTER TABLE lessons ADD COLUMN job TEXT")     # which background job added it
        self.db.commit()

    # ---- lessons -----------------------------------------------------------------------
    def has_video(self, video_id: str) -> bool:
        return bool(video_id) and self.db.execute(
            "SELECT 1 FROM lessons WHERE video_id = ? AND status = 'learned'", (video_id,)).fetchone() is not None

    def add_lesson(self, title, source, lines, video_id=None, url=None, playlist=None) -> int:
        if video_id:                              # a failed earlier try is replaced
            for (lid,) in self.db.execute("SELECT id FROM lessons WHERE video_id = ?", (video_id,)).fetchall():
                self.forget_lesson(lid)
        import os
        cur = self.db.execute("INSERT INTO lessons (title, source, video_id, url, playlist, lines, added, status, job) "
                              "VALUES (?,?,?,?,?,?,?, 'learning', ?)",
                              (title, source, video_id, url, playlist, lines, now(), os.environ.get("APPRENTICE_JOB")))
        self.db.commit()
        return cur.lastrowid

    def finish_lesson(self, lid: int, status: str = "learned", note: str = ""):
        self.db.execute("UPDATE lessons SET status = ?, note = ? WHERE id = ?", (status, note, lid))
        self.db.commit()

    def forget_lesson(self, lid: int):
        """Remove a lesson and the setups learned from it (ones you pinned are kept)."""
        self.db.execute("DELETE FROM concepts WHERE lesson_id = ?", (lid,))
        self.db.execute("DELETE FROM setups WHERE lesson_id = ? AND source = 'lesson' AND status != 'pinned'", (lid,))
        self.db.execute("DELETE FROM lessons WHERE id = ?", (lid,))
        self.db.commit()

    def job_stopped(self, job: str):
        """A job was stopped: the lesson it was in the middle of is marked stopped (paste it again to redo it)."""
        self.db.execute("UPDATE lessons SET status = 'stopped', note = 'stopped part-way' "
                        "WHERE job = ? AND status = 'learning'", (job,))
        self.db.commit()

    def forget_job(self, job: str) -> int:
        """Delete every lesson a job added, and what was learned only from them."""
        ids = [r[0] for r in self.db.execute("SELECT id FROM lessons WHERE job = ?", (job,)).fetchall()]
        for lid in ids:
            self.forget_lesson(lid)
        return len(ids)

    # ---- knowledge ---------------------------------------------------------------------
    def add_concept(self, lid, c: dict):
        self.db.execute("INSERT INTO concepts (lesson_id, t, category, text, quote) VALUES (?,?,?,?,?)",
                        (lid, int(c.get("t", 0)), c.get("category", "other"), c["text"], c.get("quote", "")))

    def add_setup(self, s: dict) -> int:
        cur = self.db.execute(
            "INSERT INTO setups (name, direction, rules, unchecked, stop_atr, rr, timeframe, why, quote, source, "
            "lesson_id, status, weight, stats, created, updated) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (s["name"], s["direction"], json.dumps(s.get("rules", [])), json.dumps(s.get("unchecked", [])),
             float(s.get("stop_atr") or 1.5), float(s.get("rr") or 2.0), s.get("timeframe", ""), s.get("why", ""),
             s.get("quote", ""), s.get("source", "lesson"), s.get("lesson_id"), s.get("status", "candidate"),
             float(s.get("weight", 0.0)), json.dumps(s.get("stats", {})), now(), now()))
        self.db.commit()
        return cur.lastrowid

    def update_setup(self, sid: int, **fields):
        for k in ("rules", "unchecked", "stats"):
            if k in fields and not isinstance(fields[k], str):
                fields[k] = json.dumps(fields[k])
        fields["updated"] = now()
        sets = ", ".join(f"{k} = ?" for k in fields)
        self.db.execute(f"UPDATE setups SET {sets} WHERE id = ?", [*fields.values(), sid])
        self.db.commit()

    def delete_setup(self, sid: int):
        self.db.execute("DELETE FROM setups WHERE id = ?", (sid,))
        self.db.commit()

    def setups(self, status: str | tuple | None = None) -> list[dict]:
        rows = self.db.execute("SELECT * FROM setups ORDER BY id").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("rules", "unchecked", "stats"):
                d[k] = json.loads(d[k] or ("{}" if k == "stats" else "[]"))
            if status is None or d["status"] == status or (isinstance(status, tuple) and d["status"] in status):
                out.append(d)
        return out

    def feedback(self, setup_id: int | None, verdict: str, note: str = "", context: dict | None = None):
        self.db.execute("INSERT INTO feedback (setup_id, verdict, note, context, at) VALUES (?,?,?,?,?)",
                        (setup_id, verdict, note, json.dumps(context or {}), now()))
        self.db.commit()

    def votes(self, setup_id: int) -> tuple[int, int]:
        """(good, bad) for a setup: 4-5 star ratings count as good, 1-2 as bad."""
        stars = [r[0] for r in self.db.execute("SELECT stars FROM ratings WHERE kind = 'setup' AND ref = ?",
                                                (str(setup_id),)).fetchall()]
        for r in self.db.execute("SELECT stars, snapshot FROM ratings WHERE kind = 'decision'").fetchall():
            if setup_id in (json.loads(r[1] or "{}").get("setups") or []):     # a chart call this setup made
                stars.append(r[0])
        return sum(1 for x in stars if x >= 4), sum(1 for x in stars if x <= 2)

    # ---- your accuracy ratings ----------------------------------------------------------
    def rate(self, kind: str, ref, stars: int, comment: str = "", snapshot: dict | None = None) -> int:
        """kind: 'setup' | 'concept' | 'decision' | 'missed'. Every rating is kept (with its comment),
        so you can see how the AI's accuracy changes as it learns."""
        cur = self.db.execute("INSERT INTO ratings (kind, ref, stars, comment, snapshot, at) VALUES (?,?,?,?,?,?)",
                              (kind, str(ref), int(stars), comment.strip(), json.dumps(snapshot or {}, default=str),
                               now()))
        self.db.commit()
        return cur.lastrowid

    def ratings(self, kind: str | None = None, ref=None) -> list[dict]:
        q, args = "SELECT * FROM ratings", []
        if kind:
            q += " WHERE kind = ?"
            args.append(kind)
            if ref is not None:
                q += " AND ref = ?"
                args.append(str(ref))
        rows = self.db.execute(q + " ORDER BY id DESC", args).fetchall()
        return [{**dict(r), "snapshot": json.loads(r["snapshot"] or "{}")} for r in rows]

    def delete_rating(self, rid: int):
        self.db.execute("DELETE FROM ratings WHERE id = ?", (rid,))
        self.db.commit()

    def accuracy(self) -> dict:
        """Average stars (1-5) and % rated accurate (4-5 stars), overall and per kind."""
        rows = self.db.execute("SELECT kind, stars FROM ratings").fetchall()
        out = {}
        for kind in ("all", "setup", "concept", "decision", "missed"):
            st = [r[1] for r in rows if (kind == "all" and r[0] != "signal") or r[0] == kind]
            out[kind] = {"n": len(st), "avg": round(sum(st) / len(st), 2) if st else None,
                         "accurate_pct": round(100 * sum(1 for x in st if x >= 4) / len(st)) if st else None}
        return out

    def log_decision(self, d: dict) -> int:
        cur = self.db.execute(
            "INSERT INTO journal (at, symbol, timeframe, bar_time, action, confidence, entry, stop, target, reasons) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (now(), d["symbol"], d["timeframe"], str(d["bar_time"]), d["action"], d.get("confidence"), d.get("entry"),
             d.get("stop"), d.get("target"), json.dumps(d.get("reasons", []))))
        self.db.commit()
        return cur.lastrowid

    def table(self, name: str) -> pd.DataFrame:
        return pd.read_sql_query(f"SELECT * FROM {name}", self.db)

    def commit(self):
        self.db.commit()
