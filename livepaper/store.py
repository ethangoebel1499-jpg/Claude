"""SQLite storage for the paper trader and the dashboard."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .core import Fill, State

SCHEMA = """
CREATE TABLE IF NOT EXISTS fills (run TEXT, time TEXT, side TEXT, reason TEXT, price REAL, qty REAL,
                                  fee_usd REAL, order_id TEXT);
CREATE TABLE IF NOT EXISTS trades (run TEXT, side TEXT, entry_time TEXT, entry_price REAL, qty REAL,
                                   exit_time TEXT, exit_price REAL, reason TEXT, targets_hit INTEGER,
                                   pnl_usd REAL, pnl_pct REAL);
CREATE TABLE IF NOT EXISTS equity (run TEXT, time TEXT, equity REAL, price REAL);
CREATE TABLE IF NOT EXISTS signals (run TEXT, bar_time TEXT, long_entry INTEGER, short_entry INTEGER,
                                    checklist TEXT);
CREATE TABLE IF NOT EXISTS events (run TEXT, time TEXT, kind TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS state (run TEXT PRIMARY KEY, json TEXT);
CREATE TABLE IF NOT EXISTS runs (name TEXT PRIMARY KEY, strategy TEXT, symbol TEXT, asset TEXT, timeframe TEXT,
                                 feed TEXT, shorts INTEGER, capital REAL, fraction REAL, stop REAL,
                                 take_profits TEXT, active INTEGER, source TEXT, created TEXT);
CREATE TABLE IF NOT EXISTS quotes (run TEXT PRIMARY KEY, time TEXT, price REAL, change_24h REAL);
CREATE TABLE IF NOT EXISTS commands (id INTEGER PRIMARY KEY, run TEXT, action TEXT, created TEXT, done TEXT,
                                     result TEXT);
"""
RUN_FIELDS = ("name", "strategy", "symbol", "asset", "timeframe", "feed", "shorts", "capital", "fraction", "stop",
              "take_profits", "active", "source", "created", "longs", "auto_sides")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.db = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.db.executescript(SCHEMA)
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(runs)")}
        for col, default in (("longs", 1), ("auto_sides", 0)):    # added later: long / short checkboxes, apprentice auto
            if col not in cols:
                self.db.execute(f"ALTER TABLE runs ADD COLUMN {col} INTEGER DEFAULT {default}")
                if col == "auto_sides":                         # the apprentice picks its own sides from now on
                    self.db.execute("UPDATE runs SET auto_sides = 1 WHERE strategy = 'apprentice'")
        ccols = {r[1] for r in self.db.execute("PRAGMA table_info(commands)")}
        if "size_pct" not in ccols:                              # added later: how much of the account a command uses
            self.db.execute("ALTER TABLE commands ADD COLUMN size_pct REAL")
        self.db.commit()

    def _ins(self, table: str, row: dict):
        cols = ",".join(row)
        self.db.execute(f"INSERT INTO {table} ({cols}) VALUES ({','.join('?' * len(row))})", list(row.values()))
        self.db.commit()

    def fill(self, run, time, side, reason, f: Fill):
        self._ins("fills", {"run": run, "time": str(time), "side": side, "reason": reason, "price": f.price,
                            "qty": f.qty, "fee_usd": f.fee_usd, "order_id": f.order_id})

    def trade(self, t: dict):
        self._ins("trades", t)

    def equity(self, run, time, equity, price):
        self._ins("equity", {"run": run, "time": str(time), "equity": equity, "price": price})

    def signal(self, run, bar_time, long_entry, short_entry, checklist):
        self._ins("signals", {"run": run, "bar_time": str(bar_time), "long_entry": int(bool(long_entry)),
                              "short_entry": int(bool(short_entry)), "checklist": checklist})

    def event(self, run, time, kind, text):
        self._ins("events", {"run": run, "time": str(time), "kind": kind, "text": text})

    def save_state(self, run, state: State):
        self.db.execute("INSERT OR REPLACE INTO state (run, json) VALUES (?, ?)", (run, json.dumps(state.to_dict())))
        self.db.commit()

    def load_state(self, run) -> State | None:
        row = self.db.execute("SELECT json FROM state WHERE run = ?", (run,)).fetchone()
        return State.from_dict(json.loads(row[0])) if row else None

    # ---- runs (what the paper trader trades; the dashboard edits these) -------
    def upsert_run(self, r: dict, *, keep_active: bool = False):
        row = {k: r.get(k) for k in RUN_FIELDS}
        row["take_profits"] = json.dumps(r["take_profits"]) if r.get("take_profits") is not None else None
        row["shorts"], row["active"] = int(bool(r.get("shorts"))), int(r.get("active", 1))
        row["longs"] = int(bool(r.get("longs", True)))
        row["auto_sides"] = int(bool(r.get("auto_sides", False)))
        row["created"] = row["created"] or now_utc()
        old = self.db.execute("SELECT active, created FROM runs WHERE name = ?", (r["name"],)).fetchone()
        if old:
            row["created"] = old[1]
            if keep_active:
                row["active"] = old[0]
        self.db.execute(f"INSERT OR REPLACE INTO runs ({','.join(RUN_FIELDS)}) VALUES ({','.join('?' * len(RUN_FIELDS))})",
                        [row[k] for k in RUN_FIELDS])
        self.db.commit()

    def set_active(self, name: str, active: bool):
        self.db.execute("UPDATE runs SET active = ? WHERE name = ?", (int(active), name))
        self.db.commit()

    def runs(self, active_only: bool = False) -> list[dict]:
        df = pd.read_sql_query("SELECT * FROM runs" + (" WHERE active = 1" if active_only else ""), self.db)
        out = []
        for r in df.to_dict("records"):
            r["take_profits"] = json.loads(r["take_profits"]) if isinstance(r["take_profits"], str) else None
            r["shorts"], r["active"] = bool(r["shorts"]), bool(r["active"])
            r["longs"] = True if r.get("longs") is None or pd.isna(r.get("longs")) else bool(r["longs"])
            r["auto_sides"] = False if r.get("auto_sides") is None or pd.isna(r.get("auto_sides")) else bool(r["auto_sides"])
            if r["stop"] is not None and pd.isna(r["stop"]):
                r["stop"] = None
            out.append(r)
        return out

    # ---- manual orders from the dashboard (the trader carries them out on its next price check)
    def command(self, run: str, action: str, size_pct: float | None = None) -> None:
        """size_pct = percent of the run's account for this trade (None: the run's normal size)."""
        self.db.execute("INSERT INTO commands (run, action, created, size_pct) VALUES (?, ?, ?, ?)",
                        (run, action, now_utc(), size_pct))
        self.db.commit()

    def command_size(self, cid: int) -> float | None:
        r = self.db.execute("SELECT size_pct FROM commands WHERE id = ?", (cid,)).fetchone()
        return float(r[0]) if r and r[0] is not None else None

    def pending_commands(self, run: str) -> list[tuple[int, str]]:
        return self.db.execute("SELECT id, action FROM commands WHERE run = ? AND done IS NULL ORDER BY id",
                               (run,)).fetchall()

    def finish_command(self, cid: int, result: str) -> None:
        self.db.execute("UPDATE commands SET done = ?, result = ? WHERE id = ?", (now_utc(), result, cid))
        self.db.commit()

    def quote(self, run, time, price, change_24h):
        self.db.execute("INSERT OR REPLACE INTO quotes VALUES (?, ?, ?, ?)", (run, str(time), price, change_24h))
        self.db.commit()

    def table(self, name: str, run: str | None = None) -> pd.DataFrame:
        q = f"SELECT * FROM {name}" + (" WHERE run = ?" if run else "")
        return pd.read_sql_query(q, self.db, params=(run,) if run else None)
