"""The live loop: one Runner per configured strategy run."""

from __future__ import annotations

import time
import tomllib
import traceback
from pathlib import Path

import pandas as pd

from mcbot import s8
from mcbot.grind import TAKE_PROFITS as G_TP, grind_checklist, grind_signals
from mcbot.witcher import witcher_checklist, witcher_signals

from .core import Exits, PositionManager, SimBroker
from .store import Store, now_utc

STRATEGIES = {"witcher": (witcher_signals, witcher_checklist),
              "grind": (grind_signals, grind_checklist),
              "s8": (s8.signals, s8.checklist)}
# The settings that held up in testing (reports/tuning.md, reports/universe.md)
PRESETS = {"witcher": Exits(0.04, [(0.12, 0.5), (0.24, 0.5)]),
           "grind": Exits(0.06, [(a * 3, s) for a, s in G_TP]),
           "s8": Exits(0.05, s8.TAKE_PROFITS)}


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    for r in cfg.get("run", []):
        r.setdefault("asset", "crypto")
        r.setdefault("shorts", False)
        r.setdefault("capital", 10_000.0)
        r.setdefault("fraction", 1.0)
        r.setdefault("feed", "kraken" if r["asset"] == "crypto" else "webull")
        r.setdefault("qty_step", 1.0 if r["asset"] == "stock" else 1e-6)
        if r["strategy"] not in STRATEGIES:
            raise SystemExit(f"run {r['name']}: unknown strategy {r['strategy']} (use {', '.join(STRATEGIES)})")
    return cfg


def exits_for(r: dict) -> Exits:
    base = PRESETS[r["strategy"]]
    stop = r.get("stop", base.stop)
    tps = [tuple(x) if x[0] != "ride" else (None, x[1]) for x in r["take_profits"]] if "take_profits" in r \
        else base.take_profits
    return Exits(stop or None, tps)


class Runner:
    def __init__(self, r: dict, feed, broker, store: Store):
        self.r, self.feed, self.store = r, feed, store
        self.signals_fn, self.checklist_fn = STRATEGIES[r["strategy"]]
        state = store.load_state(r["name"])
        self.pm = PositionManager(r["name"], exits_for(r), shorts=r["shorts"], broker=broker, store=store,
                                  fraction=r["fraction"], qty_step=r["qty_step"], state=state,
                                  start_equity=r["capital"])
        self.price = None
        self._last_equity_write = 0.0

    def _bars(self) -> pd.DataFrame:
        if self.r["feed"] == "webull":
            return self.feed.bars(self.r["symbol"], self.r["timeframe"], self.r["asset"])
        return self.feed.bars(self.r["symbol"], self.r["timeframe"])

    def _price(self) -> float:
        if self.r["feed"] == "webull":
            return self.feed.price(self.r["symbol"], self.r["asset"])
        return self.feed.price(self.r["symbol"])

    def step(self) -> None:
        name, now = self.r["name"], now_utc()
        bars = self._bars()
        self.price = self._price()
        last = str(bars.index[-1])
        st = self.pm.state
        if st.last_bar is None or last > st.last_bar:
            sig = self.signals_fn(bars)
            row = sig.iloc[-1]
            self.store.signal(name, last, row["long_entry"], row["short_entry"], self.checklist_fn(bars, -1, label=self.r["name"]))
            if st.last_bar is None:          # first start: don't act on an old candle
                st.last_bar = last
                self.store.event(name, now, "start", f"watching {self.r['symbol']} {self.r['timeframe']}")
                self.store.save_state(name, st)
            else:
                self.pm.on_bar_close(row, self.price, now, bar_time=last)
        else:
            self.pm.retry_pending(self.price, now)
        self.pm.on_tick(self.price, self.price, self.price, now)
        if time.time() - self._last_equity_write >= 300:
            self.store.equity(name, now, self.pm.state.equity + self.pm.unrealized(self.price), self.price)
            self._last_equity_write = time.time()


def build(cfg: dict, store: Store):
    runs = cfg.get("run", [])
    app = cfg.get("app", {})
    feeds, client = {}, None
    if any(r["feed"] == "webull" for r in runs) or app.get("broker") == "webull":
        from .brokers import api_client
        wb = cfg.get("webull", {})
        client = api_client(wb.get("region", "us"), wb.get("endpoint", "api.sandbox.webull.com"))
    if any(r["feed"] == "kraken" for r in runs):
        from .feeds import KrakenFeed
        feeds["kraken"] = KrakenFeed()
    if client is not None and any(r["feed"] == "webull" for r in runs):
        from .feeds import WebullFeed
        feeds["webull"] = WebullFeed(client)
    runners: list[Runner] = []
    if app.get("broker", "sim") == "webull":
        from .brokers import WebullBroker
        broker = WebullBroker(client, {r["name"]: (r["symbol"], r["asset"]) for r in runs}, fee=app.get("fee", 0.0))
    else:
        broker = SimBroker(fee=app.get("fee", 0.001), fill="market",
                           market_price=lambda run: next(x.price for x in runners if x.r["name"] == run))
    runners += [Runner(r, feeds[r["feed"]], broker, store) for r in runs]
    return runners


def run_forever(cfg: dict, store: Store) -> None:
    from .sync import sync
    runners = build(cfg, store)
    app = cfg.get("app", {})
    poll, sync_every = app.get("poll_seconds", 30), app.get("sync_minutes", 60) * 60
    last_sync = time.time()
    print(f"livepaper: {len(runners)} runs, broker={app.get('broker', 'sim')}, polling every {poll}s. Ctrl+C to stop.")
    while True:
        for rn in runners:
            try:
                rn.step()
            except Exception as e:  # keep the other runs going; the next poll retries
                store.event(rn.r["name"], now_utc(), "error", f"{type(e).__name__}: {e}")
                traceback.print_exc()
        if app.get("sync_git", False) and time.time() - last_sync >= sync_every:
            try:
                sync(store)
            except Exception as e:
                print(f"[sync] failed: {e}")
            last_sync = time.time()
        time.sleep(poll)
