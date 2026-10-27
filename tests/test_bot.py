import numpy as np
import pandas as pd
import pytest

from mcbot.backtest import run_backtest
from mcbot.bot import PaperBroker, check_stops
from mcbot.data import synthetic
from mcbot.indicators import market_cipher, rsi
from mcbot.strategies import STRATEGIES


@pytest.fixture(scope="module")
def df():
    return synthetic(n=2000, seed=1)


def test_indicators_have_no_lookahead(df):
    full = market_cipher(df)
    cut = market_cipher(df.iloc[:1500])
    cols = ["wt1", "wt2", "money_flow", "rsi", "green_dot", "red_dot", "bull_div", "bear_div"]
    pd.testing.assert_frame_equal(full[cols].iloc[:1500], cut[cols])


def test_wavetrend_produces_dots(df):
    mc = market_cipher(df)
    assert mc["green_dot"].sum() > 0 and mc["red_dot"].sum() > 0
    assert (mc.loc[mc["green_dot"], "wt2"] <= -53).all()
    assert (mc.loc[mc["red_dot"], "wt2"] >= 53).all()


def test_rsi_bounds(df):
    r = rsi(df["close"]).dropna()
    assert r.between(0, 100).all()


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_strategies_backtest(df, name):
    mc = market_cipher(df)
    res = run_backtest(df, STRATEGIES[name](mc), allow_short=True)
    assert len(res.equity) == len(df)
    assert np.isfinite(res.equity).all()
    for t in res.trades:
        assert t.exit_time >= t.entry_time


def test_backtest_fills_next_open():
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    df = pd.DataFrame({"open": [10, 11, 12, 13, 14], "high": [10.5, 11.5, 12.5, 13.5, 14.5],
                       "low": [9.5, 10.5, 11.5, 12.5, 13.5], "close": [10, 11, 12, 13, 14],
                       "volume": 1.0}, index=idx)
    sig = pd.DataFrame(False, index=idx, columns=["long_entry", "long_exit", "short_entry", "short_exit"])
    sig.iloc[0, 0] = True   # entry signal on bar 0 -> fill bar 1 open (11)
    sig.iloc[2, 1] = True   # exit signal on bar 2 -> fill bar 3 open (13)
    res = run_backtest(df, sig, stop_loss=None, take_profit=None, fee=0)
    assert len(res.trades) == 1
    t = res.trades[0]
    assert (t.entry_price, t.exit_price) == (11, 13)
    assert res.stats["final_equity"] == pytest.approx(10_000 * 13 / 11)


def test_paper_broker_round_trip(tmp_path):
    b = PaperBroker(str(tmp_path / "s.json"), start_cash=1000, fee=0)
    b.open("long", 100, 1.0)
    check_stops(b, 94, stop_loss=0.05, take_profit=0.1)
    assert b.position is None
    assert b.cash == pytest.approx(940)
    # state persists
    assert PaperBroker(str(tmp_path / "s.json")).cash == pytest.approx(940)
