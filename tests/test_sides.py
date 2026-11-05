"""Side filters: a daily trend switch x a confirmation, picked by Learn per timeframe and side."""
import numpy as np
import pandas as pd

from apprentice import brain, sides, strategy
from mcbot.data import synthetic


def test_trend_uses_only_yesterdays_closed_day():
    df = synthetic(n=24 * 300, seed=3, timeframe="1h")
    tm = sides.trend_masks(df)
    d = sides.daily_candles(df)
    above = (d.close > d.close.rolling(50).mean())
    day = df.index[-1].floor("1D")
    prev = above.loc[:day - pd.Timedelta(days=1)].iloc[-1]                 # yesterday's close decides today
    today = tm["50-day"][0][df.index >= day]
    assert today.all() if prev else not today.any()
    assert set(tm) == set(sides.TRENDS) and set(sides.confirm_masks(df)) == set(sides.CONFIRMS)


def test_symbol_tag_fetches_that_markets_daily_candles(monkeypatch):
    calls = []
    import livepaper.history as H
    daily = synthetic(n=400, seed=1, timeframe="1D")
    monkeypatch.setattr(H, "history", lambda sym, asset, tf: calls.append((sym, asset, tf)) or daily)
    sides._DAILY.clear()
    df = sides.tag(synthetic(n=500, seed=2, timeframe="1h"), "BTC")
    assert sides.daily_candles(df) is daily and calls == [("BTC", "crypto", "1D")]
    sides.daily_candles(df)
    assert len(calls) == 1                                                 # cached for the hour


def test_learn_picks_a_side_filter_only_if_it_holds_up(tmp_path, monkeypatch):
    from apprentice.store import Store
    from livepaper import settings
    monkeypatch.setattr(settings, "PATH", tmp_path / "s.json")
    monkeypatch.setenv("APPRENTICE_DB", str(tmp_path / "a.db"))
    a = Store(tmp_path / "a.db")
    for d in ("long", "short"):
        sid = a.add_setup({"name": d, "direction": d, "source": "you", "rules": [{"rule": "rsi_above", "a": -1}]})
        a.update_setup(sid, status="pinned", weight=1)
    df = synthetic(n=2000, seed=9, timeframe="4h")
    out = brain.timeframe_check(a, {"X 4h": df}, log=lambda m: None)["4h"]
    for side in ("long", "short"):
        f = out[f"{side}_filter"]
        assert (f is None) == (not out[side])
        if f:
            assert f["trend"] in sides.TRENDS and f["confirm"] in sides.CONFIRMS
            assert f["before_pct"] > 0 and f["unseen_pct"] > 0
    sig = strategy.signals(df, a.setups(("pinned",)))
    for side in ("long", "short"):
        col = sig[f"allow_{side}"].to_numpy()
        if not out[side]:
            assert not col.any()
        else:
            assert (col == sides.masks(df, out[f"{side}_filter"], side)).all()
