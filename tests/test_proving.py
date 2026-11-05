"""🔬 Proving ground: scan every timeframe, trial the best, decide after 100 trades."""
import numpy as np
import pandas as pd
import pytest

from apprentice import proving
from apprentice.store import Store
from livepaper.core import PositionManager, SimBroker, replay
from livepaper.store import Store as LPStore
from mcbot.data import synthetic


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setenv("APPRENTICE_DB", str(tmp_path / "a.db"))
    from livepaper import settings
    monkeypatch.setattr(settings, "PATH", tmp_path / "s.json")
    a = Store(tmp_path / "a.db")
    sid = a.add_setup({"name": "rsi dip", "direction": "long", "source": "you", "stop_atr": 1.5, "rr": 2.0,
                       "rules": [{"rule": "rsi_below", "a": 40}]})
    a.update_setup(sid, status="pinned", weight=1)
    return a, LPStore(tmp_path / "lp.db"), sid


def test_fees_are_counted_and_hurt_low_timeframes():
    s = {"direction": "long", "stop_atr": 1.5, "rr": 2.0, "rules": [{"rule": "rsi_below", "a": 40}]}
    from apprentice import brain
    for tf in ("1m", "4h"):
        df = synthetic(n=3000, seed=5, timeframe=tf)
        net, risk = proving.net_outcomes(df, s)
        gross = brain.outcomes(df, "long", 1.5, 2.0)
        ok = np.isfinite(net) & np.isfinite(gross)
        assert (net[ok] < gross[ok]).all()                                   # fees always cost something
    r = proving.scan_one(synthetic(n=3000, seed=5, timeframe="4h"), s)
    assert r["trades"] == r["train_trades"] + r["test_trades"] > 0


def test_scan_trial_and_decision(stores, monkeypatch):
    a, lp, sid = stores
    import livepaper.history as H
    monkeypatch.setattr(H, "history", lambda sym, asset, tf: synthetic(n=2000, seed=hash(tf) % 50, timeframe=tf))
    n = proving.scan(a, [("BTC", "crypto"), ("PEPE", "crypto")], ["1m", "4h"], log=lambda m: None)
    assert n == 12 and len(proving.results(a)) == 12          # 2 markets x 2 timeframes x (1 setup + S10 long/short)
    assert proving.pair_for("PEPE", "crypto") == "PEPEUSD" and proving.pair_for("BTC", "crypto") == "XBTUSD"
    # force a candidate and start a trial (S10's own rows are tested separately)
    a.db.execute("DELETE FROM proving_scan WHERE setup_id < 0")
    a.db.execute("UPDATE proving_scan SET trades = 60, test_trades = 20, train_r = 0.2, test_r = 0.1, avg_r = 0.15, "
                 "edge = 0.1 WHERE timeframe = '4h' AND symbol = 'BTC'")
    a.db.commit()
    out = proving.tick(a, lp, max_trials=3, log=lambda m: None)
    assert out["started"] == 1
    run = proving.run_name(sid, "BTC", "4h")
    r = next(x for x in lp.runs() if x["name"] == run)
    assert r["strategy"] == f"{proving.PREFIX}{sid}" and r["active"]
    # the trial strategy is in the registry and trades with the setup's own ATR stop
    reg = proving.refresh({})
    e = reg[f"{proving.PREFIX}{sid}"]
    df = synthetic(n=800, seed=3, timeframe="4h")
    sig = e["signals"](df)
    assert sig["stop_pct"].between(0.0005, 0.10).all() and sig["long_entry"].any()
    m = replay(df, sig, PositionManager("t", e["exits"], shorts=False, broker=SimBroker(fee=0.001)))
    assert len(m.trades) > 0
    # not decided before 100 trades; decided after
    assert proving.tick(a, lp, max_trials=3, log=lambda m: None) == {"kept": 0, "dropped": 0, "started": 0}
    for i in range(100):
        lp.trade({"run": run, "side": "long", "entry_time": f"t{i}", "entry_price": 1, "qty": 1, "exit_time": f"u{i}",
                  "exit_price": 1, "reason": "stop", "targets_hit": 0, "pnl_usd": -5.0, "pnl_pct": -0.05})
    out = proving.tick(a, lp, max_trials=3, log=lambda m: None)
    assert out["dropped"] == 1
    assert not next(x for x in lp.runs() if x["name"] == run)["active"]
    t = proving.trials(a)
    assert t.status.iloc[0] == "dropped"


def test_verdict_keeps_only_profitable_trials():
    win = pd.DataFrame({"pnl_usd": [10.0] * 60 + [-8.0] * 40, "pnl_pct": [0.1] * 60 + [-0.08] * 40})
    lose = pd.DataFrame({"pnl_usd": [10.0] * 40 + [-8.0] * 60, "pnl_pct": [0.1] * 40 + [-0.08] * 60})
    assert proving.verdict(win)[0] is True and proving.verdict(lose)[0] is False


def test_s10_price_action_uses_no_indicators_and_is_scanned(stores, monkeypatch):
    import inspect
    from mcbot import price_action as pa
    src = inspect.getsource(pa)
    code = src.lower().split('"""', 2)[2]
    for word in ("macd", "ema", "sma(", "bollinger", "wavetrend", "volume"):
        assert word not in code, word                    # price action + only RSI and money flow
    assert "_rsi(" in code and "money_flow(" in code
    df = synthetic(n=1500, seed=4, timeframe="4h")
    sig = pa.signals(df)
    assert {"long_entry", "short_entry", "stop_pct"} <= set(sig.columns)
    assert sig.loc[sig.long_entry | sig.short_entry, "stop_pct"].between(0.002, 0.10).all()
    assert "LONG:" in pa.checklist(df) and "Structure:" in pa.checklist(df)
    from livepaper.strategies import REGISTRY
    assert {"s10", "s10_long", "s10_short"} <= set(REGISTRY)
    a, lp, sid = stores
    import livepaper.history as H
    monkeypatch.setattr(H, "history", lambda sym, asset, tf: synthetic(n=1500, seed=1, timeframe=tf))
    proving.scan(a, [("BTC", "crypto")], ["4h"], log=lambda m: None)
    ids = set(proving.results(a).setup_id)
    assert {-10, -11} <= ids
    a.db.execute("UPDATE proving_scan SET trades = 80, test_trades = 20, train_r = 0.3, test_r = 0.2, avg_r = 0.25, "
                 "edge = 0.25 WHERE setup_id = -10")
    a.db.commit()
    proving.tick(a, lp, max_trials=5, log=lambda m: None)
    run = next(r for r in lp.runs() if r["name"] == "Trial S10 longs BTC 4h")
    assert run["strategy"] == "s10_long"


def test_s10_follows_the_breakout_statistics_and_sets_its_own_targets():
    from mcbot import price_action as pa
    p_up, _ = pa.odds("ascending_triangle", "up", None)
    p_dn, _ = pa.odds("ascending_triangle", "down", None)
    assert p_up < pa.MIN_ODDS <= p_dn                      # crypto: ascending triangles broke DOWN most of the time
    assert pa.odds("falling_wedge", "up", "up")[0] >= 80
    df = synthetic(n=2500, seed=11, timeframe="4h")
    sig = pa.signals(df)
    trade = sig.long_entry | sig.short_entry
    assert trade.any()
    t = sig[trade]
    assert (t.tp1_pct > 0).all() and (t.tp2_pct >= t.tp1_pct).all() and (t.tp1_pct >= pa.MIN_RR * t.stop_pct - 1e-9).all()
    assert (t.be_pct == t.tp1_pct).all()                   # breakeven after TP1
    f = pa.frame(df)
    for w in f.why[trade]:                                  # a shape retest, or two reasons at once
        assert "breakout retest" in w or len(w.split(" + ")) >= pa.CONFLUENCE


def test_live_trader_uses_each_trades_own_targets():
    idx = pd.date_range("2026-01-01", periods=30, freq="4h", tz="UTC")
    c = np.linspace(100, 106, 30)
    df = pd.DataFrame({"open": c, "high": c + 0.05, "low": c - 0.05, "close": c, "volume": 1.0}, index=idx)
    sig = pd.DataFrame(False, index=idx, columns=["long_entry", "short_entry", "long_exit", "short_exit"])
    sig["stop_pct"], sig["tp1_pct"], sig["tp2_pct"] = np.nan, np.nan, np.nan
    sig.iloc[1, 0] = True
    sig.loc[idx[1], ["stop_pct", "tp1_pct", "tp2_pct"]] = [0.02, 0.01, 0.04]
    from livepaper.core import Exits
    m = replay(df, sig, PositionManager("t", Exits(0.03, [(0.03, 0.5), (0.06, 0.5)]), shorts=False,
                                        broker=SimBroker(fee=0)))
    fills = m.trades[0]
    assert fills["reason"] == "target2" and abs(fills["exit_price"] / fills["entry_price"] - 1.04) < 0.01



def test_s10_trades_only_with_rsi_and_money_flow_agreeing():
    from mcbot import price_action as pa
    df = synthetic(n=2500, seed=11, timeframe="4h")
    f = pa.frame(df)
    assert (f.mf_rising[f["long"]]).all() and (f.rsi[f["long"]] < pa.RSI_OB).all()
    assert (f.mf_falling[f["short"]]).all() and (f.rsi[f["short"]] > pa.RSI_OS).all()
    pa.CONFIRM = False
    try:
        loose = pa.frame(df)
    finally:
        pa.CONFIRM = True
    assert (loose["long"] | loose["short"]).sum() > (f["long"] | f["short"]).sum()     # it does filter
    assert "money flow" in pa.checklist(df) and "RSI" in pa.checklist(df)
