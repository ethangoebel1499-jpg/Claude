import numpy as np
import pandas as pd

from apprentice import autorate
from apprentice.store import Store


def _df(n=700):
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(3)
    c = 100 + np.cumsum(rng.normal(0, 0.6, n))
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": rng.uniform(1, 5, n)}, index=idx)


def test_rating_a_chart_saves_each_trade_once(tmp_path):
    store = Store(str(tmp_path / "a.db"))
    df = _df()
    setups = [{"id": 1, "name": "t", "direction": "long", "status": "active", "stop_atr": 1.5, "rr": 2.0,
               "rules": [{"rule": "close_above_ema", "a": 21}]}]
    n = autorate.rate_chart(store, "BTC 1h", df, setups)
    assert n["arrows"] > 0
    assert len(store.ratings("signal")) == n["arrows"] and len(store.ratings("missed")) == n["missed"]
    assert all(1 <= r["stars"] <= 5 for r in store.ratings("signal"))
    again = autorate.rate_chart(store, "BTC 1h", df, setups)
    assert again == {"missed": 0, "arrows": 0}


def test_missed_trade_gets_a_verdict_the_apprentice_learns_from(tmp_path):
    store = Store(str(tmp_path / "a.db"))
    n = autorate.rate_chart(store, "ETH 1h", _df(), [])
    fb = store.db.execute("SELECT verdict, note FROM feedback WHERE setup_id IS NULL").fetchall()
    assert len(fb) == n["missed"] and all(r[0] in ("good", "bad") and "rule-based" in r[1] for r in fb)
