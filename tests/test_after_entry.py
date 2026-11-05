"""After-entry stats, trigger tuning, and the early exit on a confirmed reversal."""
import numpy as np
import pandas as pd

from apprentice import brain, strategy
from livepaper.core import Exits, PositionManager, SimBroker, replay
from mcbot.backtest import run_backtest
from mcbot.data import synthetic


def _df(n=1500, seed=3):
    return synthetic(n=n, seed=seed, timeframe="4h")


def _sig(df, long_at, reversal_at=()):
    sig = pd.DataFrame(False, index=df.index, columns=["long_entry", "short_entry", "long_exit", "short_exit",
                                                       "long_reversal", "short_reversal"])
    sig.iloc[list(long_at), 0] = True
    sig.iloc[list(reversal_at), 4] = True
    return sig


def _falling(n=40):
    idx = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    c = np.linspace(100, 98, n)                      # drifts down 2%: never reaches a 5% stop
    return pd.DataFrame({"open": c, "high": c + 0.05, "low": c - 0.05, "close": c, "volume": 1.0}, index=idx)


def test_reversal_closes_a_losing_trade_before_the_stop():
    df = _falling()
    sig = _sig(df, [1], [10])
    m = replay(df, sig, PositionManager("t", Exits(0.05, [(0.10, 1.0)]), shorts=False, broker=SimBroker(fee=0)))
    assert m.trades[0]["reason"] == "reversal"
    assert m.trades[0]["exit_time"] == str(df.index[11])           # acted on candle 10's close, at the next open
    res = run_backtest(df, sig, stop_loss=0.05, take_profits=[(0.10, 1.0)], fee=0)
    assert res.trades[0].reason == "reversal" and res.trades[0].exit_time == df.index[11]   # same as live


def test_reversal_leaves_a_winning_trade_alone():
    df = _falling()
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].iloc[::-1].to_numpy()  # rising
    sig = _sig(df, [1], [10])
    res = run_backtest(df, sig, stop_loss=0.05, take_profits=[(0.10, 1.0)], fee=0)
    assert res.trades[0].reason != "reversal"


def test_no_reversal_columns_changes_nothing():
    df = _df()
    sig = _sig(df, range(50, 1400, 97)).drop(columns=["long_reversal", "short_reversal"])
    a = run_backtest(df, sig, stop_loss=0.03, take_profits=[(0.06, 1.0)])
    b = run_backtest(df, sig.assign(long_reversal=False, short_reversal=False), stop_loss=0.03,
                     take_profits=[(0.06, 1.0)])
    assert a.stats == b.stats


def test_after_entry_stats_and_tuning():
    df = _df()
    setup = {"id": 1, "name": "t", "direction": "long", "stop_atr": 1.5, "rr": 2.0,
             "rules": [{"rule": "rsi_below", "a": 40}]}
    a = brain.after_entry(df, setup)
    assert a["trades"] > 10
    assert a["mae_r"] <= 0 <= a["mfe_r"]
    assert 0 <= a["reached_target_pct"] <= a["reached_1r_pct"] <= 100
    both = brain.combine_after({"x": a, "y": a})
    assert both["trades"] == 2 * a["trades"] and both["mfe_r"] == a["mfe_r"]
    assert isinstance(brain.hints(setup, a), list)
    t = brain.tune({"x": df}, setup)
    assert t is None or (t["test"]["edge"] > (t["test"]["was"] or 0) and t["set"])


def test_reversal_exit_only_where_learned(monkeypatch, tmp_path):
    from livepaper import settings
    monkeypatch.setattr(settings, "PATH", tmp_path / "s.json")
    df = _df(400)
    rule = [{"rule": "rsi_below", "a": 45}]
    helped = {"direction": "long", "weight": 0.5, "rules": rule, "stop_atr": 1.5,
              "stats": {"exit_plan": {"4h": {"stop": "swing", "reversal": True}}}}
    plain = {**helped, "stats": {}}
    assert strategy.reversal_mode() == "learned"                         # the default
    sig = strategy.signals(df, [plain])
    assert not sig["use_reversal"].any() and sig["stop_pct"].isna().all()   # nothing learned: 3% stop, no early exit
    sig = strategy.signals(df, [helped])
    fired = sig["long_entry"]
    assert fired.any() and sig["use_reversal"][fired].all()
    assert (sig["stop_pct"][fired].between(0.002, 0.10)).all()           # its own stop, beyond the last low
    settings.save({"apprentice_reversal_exit": "off"})
    assert "long_reversal" not in strategy.signals(df, [helped])


def test_trade_with_its_own_stop_moves_targets_with_it():
    df = _falling()
    sig = _sig(df, [1]).assign(stop_pct=np.nan)
    sig.loc[sig.index[1], "stop_pct"] = 0.01                             # 1% instead of the run's 5%
    m = replay(df, sig, PositionManager("t", Exits(0.05, [(0.10, 1.0)]), shorts=False, broker=SimBroker(fee=0)))
    assert m.trades[0]["reason"] == "stop" and abs(m.trades[0]["exit_price"] / m.trades[0]["entry_price"] - 0.99) < 1e-3
    res = run_backtest(df, sig, stop_loss=0.05, take_profits=[(0.10, 1.0)], fee=0)
    assert res.trades[0].reason == "stop" and res.trades[0].exit_time == pd.Timestamp(m.trades[0]["exit_time"])
    up = df.copy()
    c = np.linspace(100, 110, len(up))
    up["open"], up["high"], up["low"], up["close"] = c, c + 0.05, c - 0.05, c
    res = run_backtest(up, sig, stop_loss=0.05, take_profits=[(0.10, 1.0)], fee=0)
    assert res.trades[0].reason == "target1" and 0.02 <= res.trades[0].pnl_pct < 0.025   # 2R of a 1% stop, not the run's 10%


def test_exit_plan_is_learned_out_of_sample():
    df = _df(3000)
    setup = {"id": 1, "name": "t", "direction": "long", "stop_atr": 1.5, "rr": 2.0,
             "rules": [{"rule": "rsi_below", "a": 40}]}
    p = brain.exit_plan({"x": df}, setup)
    assert p["stop"] in brain.STOP_METHODS and isinstance(p["reversal"], bool)
    if p["stop"] != "fixed":                                             # only switched if it won on both halves
        s, f = p["stops"][p["stop"]], p["stops"]["fixed"]
        assert s["test_pct"] > f["test_pct"]


def test_learned_exits_only_deployed_if_the_whole_strategy_does_better(tmp_path, monkeypatch):
    from apprentice.store import Store as AStore
    from livepaper import settings
    monkeypatch.setattr(settings, "PATH", tmp_path / "s.json")
    a = AStore(tmp_path / "a.db")
    for d in ("long", "short"):
        a.add_setup({"name": f"rsi {d}", "direction": d, "source": "you",
                     "rules": [{"rule": "rsi_below" if d == "long" else "rsi_above", "a": 40 if d == "long" else 60}]})
    charts = {"A 4h": _df(3000, 1), "B 4h": _df(3000, 2), "C 1h": synthetic(n=2000, seed=4, timeframe="1h")}
    log = []
    brain.learn(a, charts, log.append)
    for s in a.setups():
        plans = s["stats"].get("exit_plan") or {}
        if not plans:
            continue
        assert set(plans) <= {"4h", "1h"}                                # learned per timeframe
        for p in plans.values():
            assert "deployed" in p and "learned_stop" in p
            if not p["deployed"]:
                assert p["stop"] == "fixed" and p["reversal"] is False   # not proven: back to the 3% stop
    assert brain.timeframe_of(charts["C 1h"]) == "1h" and brain.timeframe_of(charts["A 4h"]) == "4h"


def test_breakeven_moves_the_stop_to_the_entry_live_and_in_backtests():
    idx = pd.date_range("2026-01-01", periods=40, freq="4h", tz="UTC")
    c = np.r_[np.full(3, 100.0), np.linspace(100, 102, 5), np.linspace(102, 95, 32)]   # up 2%, then down 5%
    df = pd.DataFrame({"open": c, "high": c + 0.05, "low": c - 0.05, "close": c, "volume": 1.0}, index=idx)
    sig = _sig(df, [1]).assign(be_pct=np.nan)
    sig.loc[sig.index[1], "be_pct"] = 0.015
    m = replay(df, sig, PositionManager("t", Exits(0.03, [(0.03, 0.5), (0.06, 0.5)]), shorts=False,
                                        broker=SimBroker(fee=0)))
    assert m.trades[0]["reason"] == "breakeven" and abs(m.trades[0]["exit_price"] - m.trades[0]["entry_price"]) < 0.2
    res = run_backtest(df, sig, stop_loss=0.03, take_profits=[(0.03, 0.5), (0.06, 0.5)], fee=0)
    assert res.trades[0].reason == "breakeven" and abs(res.trades[0].pnl_pct) < 0.002   # instead of -3%
    plain = run_backtest(df, sig.drop(columns="be_pct"), stop_loss=0.03, take_profits=[(0.03, 0.5), (0.06, 0.5)], fee=0)
    assert plain.trades[0].reason == "stop"


def test_breakeven_moves_the_resting_broker_stop_too():
    class Broker(SimBroker):
        name = "webull"
        def __init__(self):
            super().__init__(fee=0); self.orders = []
        def place_stop(self, run, side, qty, price):
            self.orders.append(round(price, 2)); return f"o{len(self.orders)}"
        def cancel_stop(self, run, oid, qty, price):
            return "cancelled"
        def stop_status(self, run, oid, qty, price):
            return "open"
    b = Broker()
    pm = PositionManager("t", Exits(0.03, [(0.03, 0.5), (0.06, 0.5)]), shorts=False, broker=b)
    pm.open("long", 100.0, "t0", be_pct=0.015)
    pm.on_tick(100.0, 99.9, 101.6, "t1")
    assert b.orders == [97.0, 100.0] and pm.state.position.be_armed


def test_no_entry_against_an_opposite_dot_or_divergence(monkeypatch, tmp_path):
    from livepaper import settings
    monkeypatch.setattr(settings, "PATH", tmp_path / "s.json")
    df = _df(1500)
    always = lambda d: [{"direction": d, "weight": 1, "rules": [{"rule": "rsi_above", "a": -1}]}]
    no_long, no_short = strategy.opposite_signals(df)
    on, off = {"no_opposite": True, "breakeven": True}, {"no_opposite": False, "breakeven": False}
    longs_on = strategy.signals(df, always("long"), safety=on)["long_entry"].to_numpy()
    longs_off = strategy.signals(df, always("long"), safety=off)["long_entry"].to_numpy()
    shorts_on = strategy.signals(df, always("short"), safety=on)["short_entry"].to_numpy()
    assert longs_off[200:].all() and no_long.any() and no_short.any()
    assert not (longs_on & no_long).any() and not (shorts_on & no_short).any()
    assert (longs_on == (longs_off & ~no_long)).all()
    on = strategy.signals(df, always("long"), safety=on)
    off = strategy.signals(df, always("long"), safety=off)
    assert np.allclose(on["be_pct"].dropna(), 0.015) and off["be_pct"].isna().all()


def test_a_paused_timeframe_opens_nothing_new_but_still_exits():
    df = _falling()
    sig = _sig(df, [1, 20], [10]).assign(pause=False)
    sig.loc[sig.index[15:], "pause"] = True                              # paused after candle 15
    m = replay(df, sig, PositionManager("t", Exits(0.05, [(0.10, 1.0)]), shorts=False, broker=SimBroker(fee=0)))
    assert len(m.trades) == 1 and m.trades[0]["reason"] == "reversal"   # the open trade still closed; no new one
    res = run_backtest(df, sig, stop_loss=0.05, take_profits=[(0.10, 1.0)], fee=0)
    assert len(res.trades) == 1


def test_learn_pauses_a_losing_timeframe(tmp_path, monkeypatch):
    from apprentice.store import Store as AStore
    from livepaper import settings
    monkeypatch.setattr(settings, "PATH", tmp_path / "s.json")
    monkeypatch.setenv("APPRENTICE_DB", str(tmp_path / "a.db"))
    a = AStore(tmp_path / "a.db")
    sid = a.add_setup({"name": "always long", "direction": "long", "source": "you", "rules": [{"rule": "rsi_above", "a": -1}]})
    a.update_setup(sid, status="pinned", weight=1)
    idx = pd.date_range("2026-01-01", periods=600, freq="1h", tz="UTC")
    c = 100 * np.exp(np.cumsum(np.r_[0, np.full(599, -0.002)]) + 0.01 * np.sin(np.arange(600)))   # grinding down
    down = pd.DataFrame({"open": c, "high": c * 1.004, "low": c * 0.996, "close": c, "volume": 1.0}, index=idx)
    out = brain.timeframe_check(a, {"X 1h": down}, log=lambda m: None)
    assert out["1h"]["trade"] is False and out["1h"]["unseen_pct"] < 0
    assert strategy.timeframe_status("1h", a)["trade"] is False and strategy.timeframe_status("4h", a)["trade"] is True
    assert strategy.signals(down, a.setups(("pinned",)))["pause"].all()   # live: reads the decision from the store


def test_avoid_rules_block_entries_and_are_learned_out_of_sample(tmp_path, monkeypatch):
    from apprentice.store import Store as AStore
    from livepaper import settings
    monkeypatch.setattr(settings, "PATH", tmp_path / "s.json")
    monkeypatch.setenv("APPRENTICE_DB", str(tmp_path / "a.db"))
    df = _df(1500)
    always = [{"direction": "long", "weight": 1, "rules": [{"rule": "rsi_above", "a": -1}]}]
    off = {"no_opposite": False, "breakeven": False}
    plain = strategy.signals(df, always, safety=off, avoid={})["long_entry"].to_numpy()
    blocked = strategy.signals(df, always, safety=off, avoid={"long": [{"rule": "green_candles", "text": "x"}]})
    g = builder_rule(df, "green_candles")
    assert not (blocked["long_entry"].to_numpy() & g).any() and (plain & g).any()
    a = AStore(tmp_path / "a.db")
    sid = a.add_setup({"name": "always", "direction": "long", "source": "you", "rules": always[0]["rules"]})
    a.update_setup(sid, status="pinned", weight=1)
    out = brain.avoid_check(a, {"X 4h": df}, log=lambda m: None)
    for d in ("long", "short"):
        for r in out["4h"][d]:
            assert r["test"]["saved_r"] > 0 and r["train"]["saved_r"] > 0   # only kept if it helped on unseen prices
    assert strategy.avoid_rules("4h", a) == out["4h"]


def builder_rule(df, rule):
    from mcbot import builder
    return builder.rule_series(df, {"rule": rule}).to_numpy(bool)
