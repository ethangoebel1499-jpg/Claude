import os
import types

import pytest

from mcbot import s8
from mcbot.backtest import run_backtest
from mcbot.data import synthetic
from mcbot.grind import STOP, TAKE_PROFITS, grind_signals
from mcbot.witcher import witcher_signals
from livepaper.core import Exits, PositionManager, SimBroker, replay
from livepaper.feeds import parse_bars, parse_price
from livepaper.store import Store
from livepaper.runner import Runner, exits_for, load_config


@pytest.fixture(scope="module")
def candles():
    return synthetic(n=3000, seed=11, timeframe="4h")


@pytest.mark.parametrize("fn,shorts,exits", [
    (witcher_signals, True, Exits(0.04, [(0.12, 0.5), (0.24, 0.5)])),
    (s8.signals, False, Exits(0.05, s8.TAKE_PROFITS)),
    (grind_signals, True, Exits(STOP, TAKE_PROFITS)),
])
def test_replay_matches_backtester(candles, fn, shorts, exits):
    sig = fn(candles)
    bt = run_backtest(candles, sig, allow_short=shorts, stop_loss=exits.stop, take_profits=exits.take_profits, fee=0)
    m = replay(candles, sig, PositionManager("t", exits, shorts=shorts, broker=SimBroker(fee=0)))
    assert len(m.trades) == bt.stats["trades"] > 0
    assert m.state.equity == pytest.approx(bt.equity.iloc[-1], rel=1e-9)
    assert [t["reason"] for t in m.trades] == [t.reason for t in bt.trades]


def test_fees_charged_on_both_sides(candles):
    sig = s8.signals(candles)
    free = replay(candles, sig, PositionManager("a", exits_for({"strategy": "s8"}), shorts=False, broker=SimBroker(0)))
    paid = replay(candles, sig, PositionManager("b", exits_for({"strategy": "s8"}), shorts=False, broker=SimBroker(0.001)))
    assert paid.state.equity < free.state.equity


class FakeFeed:
    """Serves historical candles up to a cursor, like a live feed over time."""
    def __init__(self, df, start):
        self.df, self.i = df, start

    def bars(self, symbol, timeframe):
        return self.df.iloc[: self.i]

    def price(self, symbol):
        return float(self.df.open.iloc[self.i])   # next candle's open = "now"


def test_runner_trades_and_survives_restart(tmp_path, candles):
    store = Store(tmp_path / "t.db")
    r = {"name": "W", "strategy": "witcher", "symbol": "X", "asset": "crypto", "timeframe": "4h", "shorts": True,
         "capital": 10_000.0, "fraction": 1.0, "feed": "kraken", "qty_step": 1e-6}
    feed = FakeFeed(candles, 400)
    runners = []
    broker = SimBroker(fee=0.001, fill="market", market_price=lambda run: runners[0].price)
    runners.append(Runner(r, feed, broker, store))
    for _ in range(400):
        runners[0].step()
        feed.i += 1
    trades = store.table("trades")
    assert len(trades) > 0 and len(store.table("fills")) >= 2 * len(trades)
    assert len(store.table("signals")) == 400 and "LONG:" in store.table("signals").checklist.iloc[-1]
    before = store.load_state("W")
    # restart: a new runner picks the state up from the database
    runners2 = []
    broker2 = SimBroker(fee=0.001, fill="market", market_price=lambda run: runners2[0].price)
    runners2.append(Runner(r, feed, broker2, store))
    assert runners2[0].pm.state.equity == pytest.approx(before.equity)
    assert (runners2[0].pm.state.position is None) == (before.position is None)


def test_parse_webull_shapes():
    ms = 1_700_000_000_000
    a = {"data": [{"time": ms, "open": "1", "high": "2", "low": "0.5", "close": "1.5", "volume": "10"}]}
    b = [{"symbol": "AAPL", "result": [{"time": "2024-01-02T14:30:00Z", "open": 1, "high": 2, "low": 0.5, "close": 1.5}]}]
    for js in (a, b):
        df = parse_bars(js)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"] and df.close.iloc[0] == 1.5
    assert parse_price([{"symbol": "BTCUSD", "price": "68000.5"}]) == 68000.5


def test_webull_broker_orders(monkeypatch):
    from livepaper import brokers
    sent = []

    class Res:
        def __init__(self, code, js): self.status_code, self._js, self.text = code, js, str(js)
        def json(self): return self._js

    class Orders:
        def place_order(self, acct, orders): sent.append((acct, orders)); return Res(200, {"ok": True})
        def get_order_detail(self, acct, coid): return Res(200, {"status": "FILLED", "filled_price": "101.5", "filled_quantity": "3"})
        def cancel_order(self, acct, coid): return Res(200, {})

    fake = types.SimpleNamespace(order_v3=Orders(), account_v2=types.SimpleNamespace(
        get_account_list=lambda: Res(200, [{"account_id": "ACC1"}])))
    monkeypatch.setattr(brokers.time, "sleep", lambda s: None)
    monkeypatch.setitem(__import__("sys").modules, "webull.trade.trade_client",
                        types.SimpleNamespace(TradeClient=lambda client: fake))
    monkeypatch.delenv("WEBULL_ACCOUNT_ID", raising=False)
    b = brokers.WebullBroker(object(), {"run": ("AAPL", "stock")}, fee=0)
    f = b.fill("run", "buy", 3, 100.0, "entry")
    acct, (order,) = sent[0]
    assert acct == "ACC1" and order["order_type"] == "MARKET" and order["side"] == "BUY"
    assert order["instrument_type"] == "EQUITY" and order["quantity"] == "3" and order["support_trading_session"] == "CORE"
    assert (f.price, f.qty) == (101.5, 3.0)


def test_example_config_loads():
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "livepaper", "live.example.toml"))
    assert len(cfg["run"]) >= 3 and all(r["feed"] == "kraken" for r in cfg["run"])
    assert exits_for(cfg["run"][0]).take_profits == [(0.12, 0.5), (0.24, 0.5)]


def test_dashboard_renders(tmp_path, candles, monkeypatch):
    from streamlit.testing.v1 import AppTest
    store = Store(tmp_path / "d.db")
    cfg = tmp_path / "live.toml"
    cfg.write_text('[app]\nbroker = "sim"\n[[run]]\nname = "W"\nstrategy = "witcher"\nsymbol = "X"\ntimeframe = "4h"\n')
    r = load_config(cfg)["run"][0]
    feed = FakeFeed(candles, 400)
    runners = []
    broker = SimBroker(fee=0.001, fill="market", market_price=lambda run: runners[0].price)
    runners.append(Runner(r, feed, broker, store))
    for _ in range(150):
        runners[0]._last_equity_write = 0
        runners[0].step()
        feed.i += 1
    monkeypatch.setenv("LIVEPAPER_DB", str(tmp_path / "d.db"))
    monkeypatch.setenv("LIVEPAPER_CONFIG", str(cfg))
    at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "livepaper", "dashboard.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert at.title[0].value.startswith("📈") and len(at.metric) == 4
