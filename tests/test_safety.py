"""The safety features: a stop order resting at the broker, catching up after the
laptop was off, and the stop-loss cap."""

import types
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from livepaper.core import Exits, Fill, PositionManager, SimBroker
from livepaper.runner import Runner, exits_for
from livepaper.store import Store

SIG = {"long_entry": False, "short_entry": False, "long_exit": False, "short_exit": False}


def sig(**kw):
    return pd.Series({**SIG, **kw})


class StopBroker(SimBroker):
    """Fills market orders at the asked price and keeps resting stop orders like Webull would."""

    def __init__(self, accept=True):
        super().__init__(fee=0.0)
        self.accept, self.stops, self.n_stop = accept, {}, 0   # id -> [side, qty, px, status, fill]

    def place_stop(self, run, side, qty, px):
        if not self.accept:
            return None
        self.n_stop += 1
        coid = f"stop-{self.n_stop}"
        self.stops[coid] = [side, qty, px, "open", None]
        return coid

    def trigger(self, coid, fill_px):              # the market fell through the stop while we were away
        s = self.stops[coid]
        s[3], s[4] = "filled", Fill(fill_px, s[1], 0.0, coid)

    def stop_status(self, run, coid, qty, px):
        s = self.stops[coid]
        return s[4] if s[3] == "filled" else "dead" if s[3] == "cancelled" else "open"

    def cancel_stop(self, run, coid, qty, px):
        s = self.stops[coid]
        if s[3] == "filled":
            return s[4]
        s[3] = "cancelled"
        return True

    def open_stops(self):
        return [(s[1], s[2]) for s in self.stops.values() if s[3] == "open"]


def manager(broker, store=None, tps=((0.10, 0.5), (0.20, 0.5))):
    return PositionManager("R", Exits(0.05, list(tps)), shorts=False, broker=broker, store=store)


def test_entry_places_stop_at_broker_and_resizes_after_target():
    b = StopBroker()
    m = manager(b)
    m.on_bar_close(sig(long_entry=True), 100.0, "t0")
    q = m.state.position.qty
    assert b.open_stops() == [(q, pytest.approx(95.0))]          # whole position, 5% below entry
    m.on_tick(100, 99, 111, "t1")                                # first target: half sold
    assert m.state.position.targets_hit == 1
    assert b.open_stops() == [(pytest.approx(q / 2), pytest.approx(95.0))]   # stop now covers only what's left


def test_stop_filled_while_away_is_logged_at_its_real_price(tmp_path):
    store = Store(tmp_path / "s.db")
    b = StopBroker()
    m = manager(b, store)
    m.on_bar_close(sig(long_entry=True), 100.0, "t0")
    b.trigger(m.state.position.stop_order, 93.10)               # gapped through 95 overnight
    # restart: a fresh manager loads the saved state and checks the broker
    m2 = manager(b, store)
    m2.state = store.load_state("R")
    m2.on_tick(97, 97, 97, "t-morning")                          # price has since bounced
    assert m2.state.position is None
    t = store.table("trades").iloc[-1]
    assert t.reason == "stop (at broker)" and t.exit_price == 93.10
    assert not b.open_stops()


def test_signal_exit_cancels_stop_first_and_never_sells_twice():
    b = StopBroker()
    m = manager(b)
    m.on_bar_close(sig(long_entry=True), 100.0, "t0")
    sold = []
    real_fill = b.fill
    b.fill = lambda *a: sold.append(a) or real_fill(*a)
    b.trigger(m.state.position.stop_order, 94.0)                 # stop filled just before our exit signal
    m.on_bar_close(sig(short_entry=True), 96.0, "t1")
    assert sold == [] and m.trades[-1]["reason"] == "stop (at broker)" and m.trades[-1]["exit_price"] == 94.0


def test_refused_stop_warns_and_program_still_stops_out(tmp_path):
    store = Store(tmp_path / "s.db")
    b = StopBroker(accept=False)
    m = manager(b, store)
    m.on_bar_close(sig(long_entry=True), 100.0, "t0")
    ev = store.table("events")
    assert ev.kind.eq("warning").any() and "only works while the trader is running" in ev.text.iloc[-1]
    m.on_tick(96, 94.5, 96, "t1")
    assert m.trades[-1]["reason"] == "stop"


def test_backstop_sells_if_broker_stop_does_not_trigger():
    b = StopBroker()
    m = manager(b)
    m.on_bar_close(sig(long_entry=True), 100.0, "t0")
    m.on_tick(95, 94.5, 95, "t1")                                # past the stop, the broker's order handles it
    assert m.state.position is not None
    m.on_tick(94, 93.8, 94, "t2")                                # 1% further and still open: sell ourselves
    assert m.state.position is None and not b.open_stops()


class HourFeed:
    def __init__(self, hourly, price):
        self.hourly, self.p = hourly, price

    def bars(self, symbol, timeframe):
        return self.hourly

    def price(self, symbol):
        return self.p


def test_built_in_broker_replays_missed_candles(tmp_path):
    store = Store(tmp_path / "s.db")
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    off_at = now - timedelta(hours=10)
    idx = pd.date_range(off_at - timedelta(hours=40), now - timedelta(hours=1), freq="1h", tz="UTC")
    hourly = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0}, index=idx)
    hourly.loc[off_at + timedelta(hours=3), "low"] = 90.0           # crashed through the stop overnight...
    r = {"name": "R", "strategy": "s8", "symbol": "X", "asset": "crypto", "timeframe": "4h", "shorts": False,
         "capital": 10_000.0, "fraction": 1.0, "feed": "kraken", "qty_step": 1e-6, "stop": 0.05}
    feed = HourFeed(hourly, 104.0)                                   # ...and is back up by morning
    runners = []
    broker = SimBroker(fee=0.0, fill="market", market_price=lambda run: runners[0].price)
    rn = Runner(r, feed, broker, store)
    runners.append(rn)
    rn.price = 100.0
    rn.pm.on_bar_close(sig(long_entry=True), 100.0, str(off_at - timedelta(hours=1)))
    rn.pm.state.last_check = off_at.isoformat()
    rn.pm.state.last_bar = str(idx[-1])
    rn.step()
    t = store.table("trades").iloc[-1]
    assert t.reason == "stop" and t.exit_price == pytest.approx(95.0)    # the stop level, not this morning's 104
    assert "trader was off 10h" in " ".join(store.table("events").text)
    assert rn.pm.state.last_check is not None


@pytest.mark.parametrize("stop,expected", [(None, 0.05), (0, 0.10), (0.30, 0.10), (0.03, 0.03)])
def test_every_trade_has_a_capped_stop(stop, expected, tmp_path, monkeypatch):
    from livepaper import settings
    monkeypatch.setattr(settings, "PATH", tmp_path / "local_settings.json")
    assert exits_for({"strategy": "s8", "stop": stop}).stop == pytest.approx(expected)
    settings.save({"max_stop": 0.04})
    assert exits_for({"strategy": "s8", "stop": stop}).stop == pytest.approx(min(expected, 0.04))


def test_webull_stop_order_shape(monkeypatch):
    from livepaper import brokers
    sent, status = [], {"s": "SUBMITTED"}

    class Res:
        def __init__(self, code, js): self.status_code, self._js, self.text = code, js, str(js)
        def json(self): return self._js

    class Orders:
        def place_order(self, acct, orders): sent.append(orders[0]); return Res(200, {"ok": True})
        def get_order_detail(self, acct, coid): return Res(200, {"status": status["s"], "filled_price": "0.4123"})
        def cancel_order(self, acct, coid): status["s"] = "CANCELLED"; return Res(200, {})

    fake = types.SimpleNamespace(order_v3=Orders(), account_v2=None)
    monkeypatch.setattr(brokers.time, "sleep", lambda s: None)
    monkeypatch.setitem(__import__("sys").modules, "webull.trade.trade_client",
                        types.SimpleNamespace(TradeClient=lambda client: fake))
    monkeypatch.setenv("WEBULL_ACCOUNT_ID", "ACC1")
    b = brokers.WebullBroker(object(), {"run": ("DOGEUSD", "crypto")}, fee=0)
    coid = b.place_stop("run", "sell", 1234.5, 0.41234567)
    o = sent[0]
    assert o["order_type"] == "STOP_LOSS" and o["time_in_force"] == "GTC" and o["side"] == "SELL"
    assert o["stop_price"] == "0.412346" and o["quantity"] == "1234.5" and "e" not in o["stop_price"]
    assert b.stop_status("run", coid, 1234.5, 0.41) == "open"
    status["s"] = "FILLED"
    f = b.stop_status("run", coid, 1234.5, 0.41)
    assert (f.price, f.qty) == (0.4123, 1234.5)
    status["s"] = "SUBMITTED"
    assert b.cancel_stop("run", coid, 1234.5, 0.41) is True
