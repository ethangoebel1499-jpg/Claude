"""The live loop. The runs to trade live in the database (`runs` table): the dashboard's
Strategy Lab adds and removes them, and live.toml's [[run]] entries are copied in at start.
The loop picks up changes on every poll; removing a run closes its open paper position."""

from __future__ import annotations

import time
import tomllib
import traceback
from pathlib import Path

import pandas as pd

from .core import Exits, PositionManager, SimBroker
from .store import Store, now_utc
from .strategies import REGISTRY

# kept for older imports
STRATEGIES = {k: (v["signals"], v["checklist"]) for k, v in REGISTRY.items()}
PRESETS = {k: v["exits"] for k, v in REGISTRY.items()}


def normalize_run(r: dict) -> dict:
    r = dict(r)
    if r["strategy"] not in REGISTRY:
        raise SystemExit(f"run {r.get('name')}: unknown strategy {r['strategy']} (use {', '.join(REGISTRY)})")
    r.setdefault("asset", "crypto")
    r["shorts"] = bool(r.get("shorts", False)) and not REGISTRY[r["strategy"]].get("long_only")
    r["capital"] = float(r.get("capital") or 10_000.0)
    r["fraction"] = float(r.get("fraction") or 1.0)
    if not r.get("feed"):
        r["feed"] = "kraken" if r["asset"] == "crypto" else "yahoo"
    r.setdefault("qty_step", 1.0 if r["asset"] == "stock" else 1e-6)
    r.setdefault("stop", None)
    r.setdefault("take_profits", None)
    return r


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    cfg["run"] = [normalize_run(r) for r in cfg.get("run", [])]
    return cfg


def exits_for(r: dict) -> Exits:
    base = REGISTRY[r["strategy"]]["exits"]
    stop = r.get("stop")
    stop = base.stop if stop is None else (stop or None)          # 0 = no stop
    tps = r.get("take_profits")
    tps = [(None if x[0] in (None, "ride") else float(x[0]), float(x[1])) for x in tps] if tps else base.take_profits
    return Exits(stop, tps)


class Runner:
    def __init__(self, r: dict, feed, broker, store: Store):
        self.r, self.feed, self.store = r, feed, store
        spec = REGISTRY[r["strategy"]]
        self.signals_fn, self.checklist_fn = spec["signals"], spec["checklist"]
        self.pm = PositionManager(r["name"], exits_for(r), shorts=r["shorts"], broker=broker, store=store,
                                  fraction=r["fraction"], qty_step=r["qty_step"], state=store.load_state(r["name"]),
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
        ref = bars.close[bars.index <= pd.Timestamp(now) - pd.Timedelta("24h")]
        change = (self.price / ref.iloc[-1] - 1) * 100 if len(ref) else None
        self.store.quote(name, now, self.price, change)
        last = str(bars.index[-1])
        st = self.pm.state
        if st.last_bar is None or last > st.last_bar:
            sig = self.signals_fn(bars)
            row = sig.iloc[-1]
            self.store.signal(name, last, row["long_entry"], row["short_entry"],
                              self.checklist_fn(bars, -1, label=name))
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

    def stop(self) -> None:
        """Removed from paper trading: close any open position at the market."""
        if self.pm.state.position is not None:
            if self.price is None:
                self.price = self._price()
            self.pm.close_all(self.price, now_utc(), "removed from paper trading")
        self.store.event(self.r["name"], now_utc(), "stop", "removed from paper trading")


class App:
    def __init__(self, cfg: dict, store: Store, feeds: dict | None = None):
        self.cfg, self.store, self.app = cfg, store, cfg.get("app", {})
        self.feeds, self.runners, self.client = dict(feeds or {}), {}, None
        if self.app.get("broker", "sim") == "webull":
            from .brokers import WebullBroker
            self.broker = WebullBroker(self._webull_client(), {}, fee=self.app.get("fee", 0.0))
        else:
            self.broker = SimBroker(fee=self.app.get("fee", 0.001), fill="market",
                                    market_price=lambda run: self.runners[run].price)

    def _webull_client(self):
        if self.client is None:
            from .brokers import api_client
            wb = self.cfg.get("webull", {})
            self.client = api_client(wb.get("region", "us"), wb.get("endpoint", "api.sandbox.webull.com"))
        return self.client

    def feed(self, kind: str):
        if kind not in self.feeds:
            from . import feeds
            if kind == "webull":
                self.feeds[kind] = feeds.WebullFeed(self._webull_client())
            elif kind == "yahoo":
                self.feeds[kind] = feeds.YahooFeed()
            else:
                self.feeds[kind] = feeds.KrakenFeed()
        return self.feeds[kind]

    def sync_runs(self) -> None:
        all_runs = self.store.runs()
        wanted = {r["name"]: normalize_run(r) for r in all_runs if r["active"]}
        for r in all_runs:              # unticked while the trader was off, with a position still open
            st = self.store.load_state(r["name"]) if not r["active"] and r["name"] not in self.runners else None
            if st is not None and st.position is not None:
                rn = Runner(normalize_run(r), self.feed(r["feed"] or "kraken"), self.broker, self.store)
                self.runners[r["name"]] = rn
        for name in list(self.runners):
            if name not in wanted:
                try:
                    self.runners[name].stop()
                finally:
                    del self.runners[name]
        for name, r in wanted.items():
            if name not in self.runners:
                if hasattr(self.broker, "assets"):
                    self.broker.assets[name] = (r["symbol"], r["asset"])
                self.runners[name] = Runner(r, self.feed(r["feed"]), self.broker, self.store)

    def step(self) -> None:
        self.sync_runs()
        for rn in list(self.runners.values()):
            try:
                rn.step()
            except Exception as e:  # keep the other runs going; the next poll retries
                self.store.event(rn.r["name"], now_utc(), "error", f"{type(e).__name__}: {e}")
                traceback.print_exc()


def seed_runs(cfg: dict, store: Store) -> None:
    """Copy live.toml's runs into the database (settings refreshed, on/off state kept)."""
    for r in cfg.get("run", []):
        store.upsert_run({**r, "source": "live.toml"}, keep_active=True)


def build(cfg: dict, store: Store) -> list[Runner]:
    seed_runs(cfg, store)
    app = App(cfg, store)
    app.sync_runs()
    return list(app.runners.values())


def run_forever(cfg: dict, store: Store) -> None:
    from .sync import sync
    seed_runs(cfg, store)
    app = App(cfg, store)
    poll = app.app.get("poll_seconds", 30)
    sync_every, last_sync = app.app.get("sync_minutes", 60) * 60, time.time()
    print(f"livepaper: broker={app.app.get('broker', 'sim')}, polling every {poll}s. Runs are managed in the "
          f"dashboard's Strategy Lab. Ctrl+C to stop.")
    while True:
        app.step()
        if app.app.get("sync_git", False) and time.time() - last_sync >= sync_every:
            try:
                sync(store)
            except Exception as e:
                print(f"[sync] failed: {e}")
            last_sync = time.time()
        time.sleep(poll)
