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
    from livepaper import settings
    monkeypatch.setattr(settings, "PATH", tmp_path / "local_settings.json")
    monkeypatch.setenv("LOCAL_SETTINGS", str(tmp_path / "local_settings.json"))
    settings.save({"streamlearn_db": str(tmp_path / "streams.db")})
    monkeypatch.setenv("LIVEPAPER_CONFIG", str(cfg))
    at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "livepaper", "dashboard.py"), default_timeout=60)
    import livepaper.history
    monkeypatch.setattr(livepaper.history, "history", lambda symbol, asset, tf: candles)
    at.run()
    assert not at.exception
    assert at.title[0].value.startswith("📈") and [m.label for m in at.metric][:5] == \
        ["Total equity", "Open P&L", "Closed P&L", "Open positions", "Win rate"]
    # Strategy Lab: backtest, then tick "Paper trade this setup"
    at.button[0].click().run()
    assert not at.exception and any(m.label == "Strategy return" for m in at.metric)
    box = next(c for c in at.checkbox if c.label.startswith("📄 Paper trade this setup"))
    box.check().run()
    added = [r for r in store.runs() if r["source"] == "dashboard"]
    assert len(added) == 1 and added[0]["active"] and added[0]["strategy"] == "witcher"
    assert added[0]["take_profits"] == [[0.12, 0.5], [0.24, 0.5]] and added[0]["stop"] == 0.04


def test_runs_added_and_removed_live(tmp_path, candles):
    """Strategy Lab ticks a run on, the trader starts it; unticking closes its position."""
    from livepaper.runner import App
    store = Store(tmp_path / "r.db")
    feed = FakeFeed(candles, 400)
    app = App({"app": {"broker": "sim", "fee": 0.0}}, store, feeds={"kraken": feed})
    run = {"name": "W BTC 4h", "strategy": "witcher", "symbol": "XBTUSD", "asset": "crypto", "timeframe": "4h",
           "shorts": True, "capital": 10_000.0, "active": 1, "source": "dashboard"}
    store.upsert_run(run)
    opened = False
    for _ in range(600):
        app.step()
        feed.i += 1
        if store.load_state("W BTC 4h").position is not None:
            opened = True
            break
    assert opened and "W BTC 4h" in app.runners
    assert store.table("quotes").price.iloc[0] > 0
    store.set_active("W BTC 4h", False)
    app.step()
    assert "W BTC 4h" not in app.runners and store.load_state("W BTC 4h").position is None
    assert store.table("trades").reason.iloc[-1] == "removed from paper trading"


def test_orphan_position_closed_on_restart(tmp_path, candles):
    from livepaper.runner import App
    store = Store(tmp_path / "o.db")
    feed = FakeFeed(candles, 400)
    run = {"name": "W", "strategy": "witcher", "symbol": "X", "asset": "crypto", "timeframe": "4h",
           "shorts": True, "capital": 10_000.0, "active": 1}
    store.upsert_run(run)
    app = App({"app": {"fee": 0.0}}, store, feeds={"kraken": feed})
    while store.load_state("W") is None or store.load_state("W").position is None:
        app.step()
        feed.i += 1
    store.set_active("W", False)                     # unticked while the trader is off
    app2 = App({"app": {"fee": 0.0}}, store, feeds={"kraken": feed})
    app2.step()
    assert store.load_state("W").position is None and "W" not in app2.runners



def test_prices_never_use_exponents():
    from mcbot.fmt import money, price
    for v in (77168.7, 2226.72, 0.5432, 0.0000123456, 1.234e-08, 123456789.123, 3e-12):
        assert "e" not in price(v) and "e" not in money(v)
    assert price(77168.7) == "77,168.70" and price(0.0000123456) == "0.00001235" and money(-0.0042) == "-$0.0042"


def test_account_settings_round_trip(tmp_path, monkeypatch):
    from livepaper import settings
    monkeypatch.setattr(settings, "PATH", tmp_path / "local_settings.json")
    monkeypatch.delenv("WEBULL_APP_KEY", raising=False)
    assert settings.get("broker") == "sim" and settings.saved() == {}
    settings.save({"broker": "webull", "webull_app_key": "abcd1234"})
    assert settings.get("broker") == "webull" and settings.get("webull_app_key") == "abcd1234"
    assert settings.mask("abcd1234").endswith("1234")
    monkeypatch.setenv("WEBULL_APP_KEY", "fromenv")
    assert settings.get("webull_app_key") == "fromenv"
