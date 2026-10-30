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

DEFAULT_MAX_STOP = 0.10
CATCH_UP_AFTER = 120          # seconds without a price check before missed candles are replayed

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


def max_stop() -> float:
    from . import settings
    return float(settings.get("max_stop") or DEFAULT_MAX_STOP)


def exits_for(r: dict) -> Exits:
    base = REGISTRY[r["strategy"]]["exits"]
    stop = r.get("stop")
    stop = base.stop if stop is None else stop
    # safety cap: no trade without a stop, and none wider than the Account tab's maximum
    stop = min(stop, max_stop()) if stop else max_stop()
    tps = r.get("take_profits")
    tps = [(None if x[0] in (None, "ride") else float(x[0]), float(x[1])) for x in tps] if tps else base.take_profits
    return Exits(stop, tps)


def _span(td: pd.Timedelta) -> str:
    m = int(td.total_seconds() // 60)
    return f"{m // 1440}d {m % 1440 // 60}h" if m >= 1440 else f"{m // 60}h {m % 60}m" if m >= 60 else f"{m}m"


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

    def catch_up(self, now: str) -> None:
        """The trader was off (laptop asleep / shut down). A stop order resting at the broker
        is checked in on_tick; with the built-in paper broker, replay the hourly candles that
        were missed - stop first, then targets - at the prices they would have filled."""
        st, p = self.pm.state, self.pm.state.position
        if p is None or st.last_check is None:
            return
        since = pd.Timestamp(st.last_check)
        off = pd.Timestamp(now) - since
        if off.total_seconds() < CATCH_UP_AFTER:
            return
        name = self.r["name"]
        st.last_check = now                     # once per restart, even if the rest of this poll fails
        self.store.save_state(name, st)
        if p.stop_order:
            self.store.event(name, now, "catch-up", f"trader was off {_span(off)}: checking the broker's stop order")
            return
        if not isinstance(self.pm.broker, SimBroker):
            self.store.event(name, now, "warning", f"⚠️ trader was off {_span(off)} with no stop order at the broker: "
                                                    "anything missed is acted on at today's price")
            return
        try:
            hourly = self._bars_tf("1h")
        except Exception:
            hourly = self._bars()
        missed = hourly[hourly.index + pd.Timedelta("1h") > since]
        before = (st.equity, p.targets_hit, p.remaining_qty)
        mode, self.pm.broker.mode = self.pm.broker.mode, "intended"     # fill at the level, not today's price
        try:
            for t, b in missed.iterrows():
                if self.pm.state.position is None:
                    break
                self.pm.on_tick(b.open, b.low, b.high, str(t))
        finally:
            self.pm.broker.mode = mode
        p2 = self.pm.state.position
        changed = p2 is None or (st.equity, p2.targets_hit, p2.remaining_qty) != before
        self.store.event(name, now, "catch-up", f"trader was off {_span(off)}: replayed {len(missed)} hourly candles"
                         + ("" if changed else ", stop and targets not reached"))

    def _bars_tf(self, tf: str) -> pd.DataFrame:
        if self.r["feed"] == "webull":
            return self.feed.bars(self.r["symbol"], tf, self.r["asset"])
        return self.feed.bars(self.r["symbol"], tf)

    def step(self) -> None:
        name, now = self.r["name"], now_utc()
        self.catch_up(now)
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
        self.pm.state.last_check = now          # saved every poll, so a restart knows what it missed
        self.store.save_state(name, self.pm.state)
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
        from . import settings
        # the Account tab's saved settings override live.toml's [app]
        self.cfg, self.store, self.app = cfg, store, {**cfg.get("app", {}), **settings.saved()}
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
            self.client = api_client(self.app.get("webull_region") or wb.get("region"),
                                     self.app.get("webull_endpoint") or wb.get("endpoint"))
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
    poll = int(app.app.get("poll_seconds", 30))
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
