"""SQLite storage for streams, extracted calls / rules / levels and call outcomes."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS streams (id INTEGER PRIMARY KEY, title TEXT, start TEXT, source TEXT,
                                    segments INTEGER, duration_s INTEGER, added TEXT);
CREATE TABLE IF NOT EXISTS segments (stream_id INTEGER, t INTEGER, text TEXT);
CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, stream_id INTEGER, t INTEGER, said_at TEXT,
                                  asset TEXT, direction TEXT, kind TEXT, timeframe TEXT, entry REAL,
                                  stop REAL, targets TEXT, conditions TEXT, quote TEXT);
CREATE TABLE IF NOT EXISTS rules (id INTEGER PRIMARY KEY, stream_id INTEGER, t INTEGER, category TEXT,
                                  rule TEXT, quote TEXT);
CREATE TABLE IF NOT EXISTS levels (stream_id INTEGER, t INTEGER, asset TEXT, price REAL, kind TEXT);
CREATE TABLE IF NOT EXISTS reviews (call_id INTEGER PRIMARY KEY, verdict TEXT, note TEXT, corrected TEXT,
                                    reviewed TEXT);
CREATE TABLE IF NOT EXISTS outcomes (call_id INTEGER PRIMARY KEY, status TEXT, entry_time TEXT,
                                     entry_price REAL, exit_time TEXT, exit_price REAL, result TEXT,
                                     r_multiple REAL, ret_pct REAL, ret_24h REAL, ret_72h REAL, ret_7d REAL,
                                     mfe_pct REAL, mae_pct REAL, checked TEXT);
"""


class Store:
    def __init__(self, path: str | Path):
        self.db = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self.db.executescript(SCHEMA)
        # columns added after the first release
        for table, col, kind in (("calls", "summary", "TEXT"), ("calls", "t_start", "INTEGER"),
                                 ("calls", "t_end", "INTEGER"), ("calls", "frame", "TEXT"),
                                 ("streams", "url", "TEXT")):
            have = {r[1] for r in self.db.execute(f"PRAGMA table_info({table})")}
            if col not in have:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {kind}")
        self.db.commit()

    def add_stream(self, title, start, source, segs, added, url=None) -> int:
        cur = self.db.execute("INSERT INTO streams (title, start, source, segments, duration_s, added, url) "
                              "VALUES (?, ?, ?, ?, ?, ?, ?)",
                              (title, start, source, len(segs), segs[-1][0] if segs else 0, added, url))
        sid = cur.lastrowid
        self.db.executemany("INSERT INTO segments VALUES (?, ?, ?)", [(sid, t, s) for t, s in segs])
        self.db.commit()
        return sid

    def find_stream(self, title, start):
        row = self.db.execute("SELECT id FROM streams WHERE title = ? AND start = ?", (title, start)).fetchone()
        return row[0] if row else None

    def clear_extraction(self, sid):
        for t in ("calls", "rules", "levels"):
            self.db.execute(f"DELETE FROM {t} WHERE stream_id = ?", (sid,))
        self.db.commit()

    def add_call(self, sid, c: dict) -> int:
        cur = self.db.execute(
            "INSERT INTO calls (stream_id, t, said_at, asset, direction, kind, timeframe, entry, stop, targets, "
            "conditions, quote, summary, t_start, t_end) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, c["t"], c["said_at"], c["asset"], c["direction"], c["kind"], c["timeframe"], c["entry"], c["stop"],
             json.dumps(c["targets"]), json.dumps(c["conditions"]), c["quote"], c.get("summary", ""),
             c.get("t_start", c["t"]), c.get("t_end", c["t"])))
        return cur.lastrowid

    def set_frame(self, call_id: int, path: str):
        self.db.execute("UPDATE calls SET frame = ? WHERE id = ?", (path, call_id))
        self.db.commit()

    def set_url(self, sid: int, url: str):
        self.db.execute("UPDATE streams SET url = ? WHERE id = ?", (url, sid))
        self.db.commit()

    def review(self, call_id: int, verdict: str, note: str = "", corrected: dict | None = None, when: str = ""):
        """verdict: 'correct' (award) or 'wrong' (punish). Corrections are written onto the call."""
        if corrected:
            sets = ", ".join(f"{k} = ?" for k in corrected)
            vals = [json.dumps(v) if k in ("targets", "conditions") else v for k, v in corrected.items()]
            self.db.execute(f"UPDATE calls SET {sets} WHERE id = ?", [*vals, call_id])
        cols = ("asset", "direction", "kind", "timeframe", "entry", "stop", "targets", "summary", "quote")
        row = self.db.execute(f"SELECT {','.join(cols)} FROM calls WHERE id = ?", (call_id,)).fetchone()
        snapshot = dict(zip(cols, row)) if row else {}
        if snapshot.get("targets"):
            snapshot["targets"] = json.loads(snapshot["targets"])
        # keep a snapshot so the verdict still teaches the agent after a stream is re-extracted
        self.db.execute("INSERT OR REPLACE INTO reviews VALUES (?, ?, ?, ?, ?)",
                        (call_id, verdict, note, json.dumps({"call": snapshot, "changes": corrected or {}}), when))
        self.db.commit()

    def unreview(self, call_id: int):
        self.db.execute("DELETE FROM reviews WHERE call_id = ?", (call_id,))
        self.db.commit()

    def add_rule(self, sid, r: dict):
        self.db.execute("INSERT INTO rules (stream_id, t, category, rule, quote) VALUES (?,?,?,?,?)",
                        (sid, r["t"], r["category"], r["rule"], r["quote"]))

    def add_level(self, sid, lv: dict):
        self.db.execute("INSERT INTO levels VALUES (?,?,?,?,?)", (sid, lv["t"], lv["asset"], lv["price"], lv["kind"]))

    def save_outcome(self, o: dict):
        cols = ",".join(o)
        self.db.execute(f"INSERT OR REPLACE INTO outcomes ({cols}) VALUES ({','.join('?' * len(o))})", list(o.values()))
        self.db.commit()

    def commit(self):
        self.db.commit()

    def table(self, name: str) -> pd.DataFrame:
        return pd.read_sql_query(f"SELECT * FROM {name}", self.db)

    def calls_with_streams(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT c.*, s.title, s.start AS stream_start, s.source, s.url, o.status, o.result, o.r_multiple, "
            "o.ret_pct, o.ret_24h, o.ret_72h, o.ret_7d, o.mfe_pct, o.mae_pct, o.entry_price, o.exit_price, "
            "o.entry_time, o.exit_time, r.verdict, r.note AS review_note "
            "FROM calls c JOIN streams s ON s.id = c.stream_id LEFT JOIN outcomes o ON o.call_id = c.id "
            "LEFT JOIN reviews r ON r.call_id = c.id", self.db)
