"""One fee setting used everywhere."""
import numpy as np
import pandas as pd

from livepaper import fees, settings
from mcbot.backtest import run_backtest
from mcbot.data import synthetic


def test_presets_and_custom(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PATH", tmp_path / "s.json")
    assert fees.preset() == "kraken_spot_taker" and fees.current() == 0.004       # default: Kraken spot taker
    assert fees.round_trip_pct() == 0.8
    settings.save({"fee_preset": "kraken_futures"})
    assert fees.current() == 0.0005
    settings.save({"fee_preset": "custom", "fee": 0.0012})
    assert fees.current() == 0.0012 and "0.12%" in fees.label()


def test_backtests_charge_the_chosen_fee_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PATH", tmp_path / "s.json")
    df = synthetic(n=600, seed=2, timeframe="4h")
    sig = pd.DataFrame(False, index=df.index, columns=["long_entry", "short_entry", "long_exit", "short_exit"])
    sig.iloc[10::40, 0] = True
    kw = dict(stop_loss=0.03, take_profits=[(0.03, 1.0)])
    settings.save({"fee_preset": "kraken_futures"})
    cheap = run_backtest(df, sig, **kw)
    settings.save({"fee_preset": "kraken_spot_taker"})
    dear = run_backtest(df, sig, **kw)
    explicit = run_backtest(df, sig, fee=0.004, **kw)
    assert dear.stats == explicit.stats and cheap.stats["total_return_pct"] > dear.stats["total_return_pct"]


def test_learn_scores_setups_after_fees(tmp_path, monkeypatch):
    from apprentice import brain
    monkeypatch.setattr(settings, "PATH", tmp_path / "s.json")
    df = synthetic(n=1500, seed=4, timeframe="1h")
    s = {"direction": "long", "stop_atr": 1.5, "rr": 2.0, "rules": [{"rule": "rsi_below", "a": 40}]}
    settings.save({"fee_preset": "kraken_futures"})
    low = brain.evaluate(df, s)
    settings.save({"fee_preset": "kraken_spot_taker"})
    high = brain.evaluate(df, s)
    assert high["trades"] == low["trades"] and high["avg_r"] < low["avg_r"]
    tight = brain.evaluate(df, {**s, "stop_atr": 0.5})          # a tight stop pays more fees per R
    assert np.nanmean(brain.fee_r(df, 0.5)) > np.nanmean(brain.fee_r(df, 1.5))
