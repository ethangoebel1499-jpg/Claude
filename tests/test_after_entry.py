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
