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
"""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.executescript(SCHEMA)

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

    def table(self, name: str, run: str | None = None) -> pd.DataFrame:
        q = f"SELECT * FROM {name}" + (" WHERE run = ?" if run else "")
        return pd.read_sql_query(q, self.db, params=(run,) if run else None)
