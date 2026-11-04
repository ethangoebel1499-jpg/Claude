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
    store.command("W", "close")                  # a force-close the trader hasn't done yet (empty result = NaN)
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
    next(b for b in at.button if b.label == "Run backtest").click().run()
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


def test_take_profit_plans():
    import pandas as pd
    from livepaper.core import Exits
    from livepaper.tp_plans import plan_text, tp_rows, tps_from_table
    base = Exits(0.04, [(0.12, 0.5), (0.24, 0.5)])
    rows = tp_rows("Strategy's tested setting", True, base, 0.04)
    assert [r["At"] for r in rows] == [3.0, 6.0]                     # 12% / 24% with a 4% stop = 3R / 6R
    tps = tps_from_table(pd.DataFrame(tp_rows("2 targets: 50 / 50", True, base, 0.04)), True, 0.04)
    assert tps == [(0.04, 0.5), (0.08, 0.5)]                         # 1R and 2R of a 4% stop
    run = tps_from_table(pd.DataFrame(tp_rows("2 targets + runner: 50 / 25 / rest rides", False, base, 0.04)), False, 0.04)
    assert run == [(0.02, 0.5), (0.05, 0.25), (None, 0.25)]
    assert "rides until the exit signal" in plan_text(run, 0.04) and "e+" not in plan_text(run, 0.04)
    bad = pd.DataFrame([{"Target": "TP1", "At": 5.0, "Close %": 60}, {"Target": "TP2", "At": 10.0, "Close %": 60}])
    with pytest.raises(ValueError, match="add up to 100"):
        tps_from_table(bad, False, 0.04)
    with pytest.raises(ValueError, match="need a stop"):
        tps_from_table(pd.DataFrame(tp_rows("1 target (all out)", True, base, 0)), True, 0)


def test_force_trades_get_the_runs_exits(tmp_path, candles):
    store = Store(tmp_path / "f.db")
    r = {"name": "G", "strategy": "grind", "symbol": "X", "asset": "crypto", "timeframe": "4h", "shorts": False,
         "capital": 10_000.0, "fraction": 1.0, "feed": "kraken", "qty_step": 1e-6, "stop": None, "take_profits": None}
    feed = FakeFeed(candles, 400)
    runners = []
    broker = SimBroker(fee=0.001, fill="market", market_price=lambda run: runners[0].price)
    runners.append(Runner(r, feed, broker, store))
    rn = runners[0]
    rn.step()                                              # first start: just watching
    store.command("G", "long")
    rn.step()
    p = rn.pm.state.position
    assert p is not None and p.side == "long"
    entry = p.entry_price
    rn.pm.on_tick(entry, entry * 0.93, entry, "later")        # down 7%: GRIND's 6% stop fires
    assert rn.pm.state.position is None and rn.pm.trades[-1]["reason"] == "stop"
    store.command("G", "short")
    rn.step()
    assert rn.pm.state.position.side == "short"
    store.command("G", "short")
    store.command("G", "close")
    rn.step()
    assert rn.pm.state.position is None
    results = [x[0] for x in store.db.execute("SELECT result FROM commands ORDER BY id")]
    assert results[0].startswith("opened long at ") and results[1].startswith("opened short at ")
    assert results[2:] == ["already short", "closed"]


def test_force_trades_on_webull_say_why_they_failed(tmp_path, candles):
    store = Store(tmp_path / "wf.db")
    r = {"name": "G", "strategy": "grind", "symbol": "XBTUSD", "asset": "crypto", "timeframe": "4h", "shorts": False,
         "capital": 10_000.0, "fraction": 1.0, "feed": "kraken", "qty_step": 1e-6, "stop": None, "take_profits": None}
    feed = FakeFeed(candles, 400)
    holder = []

    class FakeWebull(SimBroker):
        name = "webull"
        assets = {}
        refuse = True
        def fill(self, run, side, qty, price, reason):
            if self.refuse:
                self.last_error, self.rejected = "insufficient buying power", True
                return None
            self.last_error, self.rejected = None, False
            return SimBroker.fill(self, run, side, qty, price, reason)
        def place_stop(self, *a):
            return None
    wb = FakeWebull(fee=0.0, fill="market", market_price=lambda run: holder[0].price)
    holder.append(Runner(r, feed, wb, store))
    rn = holder[0]
    rn.step()
    store.command("G", "short")                              # Webull can't short spot crypto: never sent
    store.command("G", "long")                               # refused, with Webull's reason
    rn.step()
    res = [x[0] for x in store.db.execute("SELECT result FROM commands ORDER BY id")]
    assert res[0].startswith("not sent: webull can't short spot crypto")
    assert res[1] == "refused by webull: insufficient buying power"
    assert rn.pm.state.pending_entry is None                 # a refusal isn't retried every price check
    ev = store.table("events")
    assert ev.text.str.contains("long entry refused by webull: insufficient buying power").any()
    wb.refuse = False
    store.command("G", "long")
    rn.step()
    assert rn.pm.state.position.side == "long"
    store.command("G", "short")                              # refused before closing the long
    rn.step()
    assert rn.pm.state.position is not None and rn.pm.state.position.side == "long"


def test_stock_list_is_searchable_by_name():
    from livepaper import stocks
    names = stocks.labels()
    assert len(names) > 1000 and {"AAPL", "SPY", "BRK-B", "PLTR"} <= set(names)
    assert "Apple" in names["AAPL"] and "e+" not in "".join(names)
    assert len(set(stocks.tickers())) == len(stocks.tickers())          # no duplicates


def test_switching_broker_drops_positions_the_new_broker_doesnt_hold(tmp_path, candles):
    store = Store(tmp_path / "w.db")
    r = {"name": "S", "strategy": "s8", "symbol": "XBTUSD", "asset": "crypto", "timeframe": "4h", "shorts": False,
         "capital": 10_000.0, "fraction": 1.0, "feed": "kraken", "qty_step": 1e-6, "stop": None, "take_profits": None}
    feed = FakeFeed(candles, 400)
    holder = []
    sim = SimBroker(fee=0.0, fill="market", market_price=lambda run: holder[0].price)
    holder.append(Runner(r, feed, sim, store))
    holder[0].step()
    store.command("S", "long")
    holder[0].step()
    assert store.load_state("S").position.broker == "sim"

    class FakeWebull(SimBroker):
        name = "webull"
        assets = {}
    sent = []
    wb = FakeWebull(fee=0.0, fill="market", market_price=lambda run: holder[-1].price)
    wb.fill = lambda *a: sent.append(a) or SimBroker.fill(wb, *a)
    holder.append(Runner(r, feed, wb, store))                  # the trader restarted on Webull
    holder[-1].step()
    assert holder[-1].pm.state.position is None and sent == []  # closed on the books, no order to Webull
    assert "switched from sim to webull" in store.table("trades").reason.iloc[-1]
    from livepaper.brokers import WEBULL_CRYPTO
    assert WEBULL_CRYPTO["XBTUSD"] == "BTCUSD" and WEBULL_CRYPTO["XDGUSD"] == "DOGEUSD"


def test_minute_candles_page_back_through_coinbase_and_cache(tmp_path, monkeypatch):
    import pandas as pd
    from livepaper import history
    monkeypatch.setattr(history, "CANDLES", tmp_path)
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    class Res:
        status_code = 200
        def __init__(self, rows): self.rows = rows
        def raise_for_status(self): pass
        def json(self): return self.rows

    def get(url, params=None, **k):
        calls.append(params)
        t0, t1 = pd.Timestamp(params["start"]), pd.Timestamp(params["end"])
        times = pd.date_range(t0, t1, freq="60s", inclusive="left")
        return Res([[int(t.timestamp()), 99.0, 101.0, 100.0, 100.5, 1.0] for t in times][::-1])   # newest first
    monkeypatch.setattr(history.requests, "get", get)
    df = history.history("XBTUSD", "crypto", "1m")
    assert len(df) >= 7 * 1440 - 2 and (df.index[1] - df.index[0]) == pd.Timedelta("1min")
    assert all(p["granularity"] == 60 for p in calls) and len(calls) >= 33          # paged 300 at a time
    assert (tmp_path / "coinbase_BTC-USD_60.csv").exists()
    n = len(calls)
    history.history("XBTUSD", "crypto", "1m")                                    # cached: only the newest bit
    assert len(calls) - n <= 2
    n = len(calls)
    three = history.history("XBTUSD", "crypto", "3m")                            # 14 days: only the older 7 fetched
    assert 30 <= len(calls) - n <= 38 and (three.index[1] - three.index[0]) == pd.Timedelta("3min")
    five = history.history("XBTUSD", "crypto", "15m")
    assert (five.index[1] - five.index[0]) == pd.Timedelta("15min")


def test_stock_list_falls_back_instead_of_crashing(monkeypatch, tmp_path):
    from livepaper import stocks
    monkeypatch.setattr(stocks, "FILE", tmp_path / "missing.csv")
    stocks.universe.cache_clear(); stocks.labels.cache_clear()
    try:
        assert "SPY" in stocks.labels()
    finally:
        stocks.universe.cache_clear(); stocks.labels.cache_clear()


def test_files_the_app_needs_are_in_git():
    import subprocess
    from healthcheck import DATA
    tracked = set(subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.split())
    assert [f for f in DATA if f not in tracked] == []


def test_custom_strategy_end_to_end(tmp_path, monkeypatch, candles):
    from livepaper import custom
    from livepaper.strategies import REGISTRY, refresh
    monkeypatch.setattr(custom, "PATH", tmp_path / "custom.json")
    spec = {"name": "RSI dip in uptrend", "long": [{"rule": "rsi_below", "a": 40}, {"rule": "ema_above_ema", "a": 20, "b": 50}],
            "short": [], "exit_long": [{"rule": "rsi_above", "a": 65}], "exit_short": [],
            "stop": 0.05, "take_profits": [[0.06, 0.5], [0.12, 0.5]]}
    key = custom.save_strategy(spec)
    refresh()
    try:
        s = REGISTRY[key]
        assert s["label"] == "🛠️ RSI dip in uptrend" and s["long_only"] and s["exits"].stop == 0.05
        sig = s["signals"](candles)
        assert sig.long_entry.any() and not sig.short_entry.any()
        assert "RSI (14) below 40" in s["checklist"](candles, -1, label="X")
        # paper trading it: the live position logic replays it like any other strategy
        m = replay(candles, sig, PositionManager("c", s["exits"], shorts=False, broker=SimBroker(0.001)))
        assert len(m.trades) > 0
        # rename a built-in and the custom one; deleting removes it
        custom.rename("s8", "My reversal")
        custom.rename(key, "Dip buyer")
        refresh()
        assert REGISTRY["s8"]["label"] == "My reversal" and REGISTRY[key]["label"] == "🛠️ Dip buyer"
        custom.rename("s8", "")
        custom.delete_strategy(key)
        refresh()
        assert key not in REGISTRY and REGISTRY["s8"]["label"] == "S8 Long-only reversal"
    finally:
        monkeypatch.undo()
        refresh()


def test_webull_account_without_crypto_moves_crypto_runs_to_the_built_in_broker(tmp_path, candles, monkeypatch):
    from livepaper import settings
    from livepaper.brokers import WebullBroker
    from livepaper.runner import App
    monkeypatch.setattr(settings, "PATH", tmp_path / "settings.json")
    store = Store(tmp_path / "c.db")
    feed = FakeFeed(candles, 400)
    app = App({"app": {"broker": "sim", "fee": 0.0}}, store, feeds={"kraken": feed})

    class TC:                                                  # Webull's SDK raises for crypto on this account
        class order_v3:
            @staticmethod
            def place_order(account, orders):
                if orders[0]["instrument_type"] == "CRYPTO":
                    raise RuntimeError("ServerException: HTTP Status: 417, Code: OPENAPI_PARAM_ERR, Msg: Parameter "
                                       "error, The current account does not support crypto trading., RequestID: x")
                raise AssertionError("no stock orders in this test")
    wb = WebullBroker.__new__(WebullBroker)
    wb.tc, wb.account, wb.assets, wb.fee, wb.wait = TC, "acc", {}, 0.0, 1
    app.broker = wb                                            # the trader is on Webull
    store.upsert_run({"name": "G", "strategy": "grind", "symbol": "XBTUSD", "asset": "crypto", "timeframe": "4h",
                      "shorts": False, "capital": 10_000.0, "active": 1, "source": "dashboard"})
    app.step()
    assert app.runners["G"].pm.broker is wb
    store.command("G", "long")
    app.step()                                                 # refused - the reason is shown, nothing crashes
    res = store.db.execute("SELECT result FROM commands").fetchone()[0]
    assert res == "refused by webull: Parameter error, The current account does not support crypto trading"
    assert settings.get("webull_crypto") == "sim"              # remembered
    assert "doesn't support crypto" in " ".join(store.table("events").text)
    app.step()
    assert app.runners["G"].pm.broker is app.sim               # crypto now trades in the built-in paper broker
    store.command("G", "short")                                # and shorts work there
    app.step()
    assert app.runners["G"].pm.state.position.side == "short"
    assert app.broker_for({"asset": "stock"}) is wb            # stocks stay at Webull
