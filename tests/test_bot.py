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
    cols = ["wt1", "wt2", "money_flow", "rsi", "green_dot", "red_dot", "bull_div", "bear_div",
            "vwap", "sfp_bull", "sfp_bear", "anchor_trigger_bull", "htf_money_flow", "htf_wt_rising"]
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


def test_money_flow_divergence():
    from mcbot.indicators import mf_convergence
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    # Window of 2: price makes a lower low, money flow makes a higher low.
    df = pd.DataFrame({"high": [10, 10, 10, 10], "low": [5, 5, 4, 4]}, index=idx)
    mf = pd.Series([-10, -10, -5, -5], index=idx)
    out = mf_convergence(df, mf, window=2)
    assert out["bull_mf_div"].iloc[-1]
    assert not out["bear_mf_conv"].iloc[-1]
    # Money flow also makes a lower low -> bearish convergence instead.
    out = mf_convergence(df, pd.Series([-10, -10, -20, -20], index=idx), window=2)
    assert out["bear_mf_conv"].iloc[-1] and not out["bull_mf_div"].iloc[-1]


def test_swing_failure_pattern():
    from mcbot.indicators import swing_failure
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    # Bar 3 wicks below the prior low (9) but closes back above it.
    df = pd.DataFrame({"high": [11, 11, 11, 11], "low": [9, 9.5, 9.5, 8.5],
                       "close": [10, 10, 10, 9.8]}, index=idx)
    bull, bear = swing_failure(df, lookback=3)
    assert bull.iloc[-1] and not bear.iloc[-1]


def test_higher_timeframe_stable_when_window_moves(df):
    from mcbot.indicators import higher_timeframe
    a = higher_timeframe(df.iloc[:1200])
    b = higher_timeframe(df.iloc[1:1200])
    # Same clock-aligned HTF candles regardless of where the window starts
    # (compare after indicator warm-up).
    pd.testing.assert_series_equal(a["htf_ribbon_bull"].iloc[-200:], b["htf_ribbon_bull"].iloc[-200:])


def test_higher_timeframe_groups_bars(df):
    from mcbot.indicators import higher_timeframe
    h = higher_timeframe(df)
    assert h["htf_money_flow"].notna().sum() > len(df) // 2
    # 4 consecutive bars share one HTF value.
    assert h["htf_wt2"].iloc[-400:].nunique() == pytest.approx(100, abs=1)


def test_heikin_ashi_values():
    from mcbot.indicators import heikin_ashi
    idx = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    df = pd.DataFrame({"open": [10, 12], "high": [13, 15], "low": [9, 11], "close": [12, 14],
                       "volume": 1.0}, index=idx)
    ha = heikin_ashi(df)
    assert ha["close"].tolist() == [11, 13]          # (o+h+l+c)/4
    assert ha["open"].tolist() == [11, 11]           # (o0+c0)/2, then prev (ha_open+ha_close)/2
    assert ha["high"].tolist() == [13, 15] and ha["low"].tolist() == [9, 11]


def test_heikin_ashi_signals_fill_at_real_prices(df):
    mc = market_cipher(df, use_heikin_ashi=True)
    assert not np.allclose(mc["close"], df["close"])
    res = run_backtest(df, STRATEGIES["heikin_ashi_trend"](mc))
    assert res.trades
    for t in res.trades[:20]:
        if t.reason == "signal":
            assert t.entry_price in df["open"].values


class FakeExchange:
    """Hourly candles; the clock is 20 minutes into the newest candle."""
    rateLimit = 0

    def __init__(self, n=5):
        self.start = 1_704_067_200_000
        self.n = n
        self.now = self.start + (n - 1) * 3_600_000 + 20 * 60_000

    def parse_timeframe(self, tf):
        return 3600

    def milliseconds(self):
        return self.now

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        return [[self.start + i * 3_600_000, 1, 2, 0.5, 1.5, 10] for i in range(self.n)]


def test_bot_drops_forming_candle():
    from mcbot.bot import closed_candles
    ex = FakeExchange(n=5)
    df = closed_candles(ex, "BTC/USDT", "1h", history=5)
    assert len(df) == 4  # the 5th candle is only 20 minutes old
    ex.now = ex.start + 5 * 3_600_000 + 1  # 5th candle has now closed
    assert len(closed_candles(ex, "BTC/USDT", "1h", history=5)) == 5


def test_stops_on_close_ignores_wicks():
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    # Bar 2 wicks to -10% but closes flat; bar 3 closes -5%.
    df = pd.DataFrame({"open": [100, 100, 100, 100, 95], "high": [101] * 5,
                       "low": [99, 99, 90, 94, 94], "close": [100, 100, 100, 95, 95],
                       "volume": 1.0}, index=idx)
    sig = pd.DataFrame(False, index=idx, columns=["long_entry", "long_exit", "short_entry", "short_exit"])
    sig.iloc[0, 0] = True  # enter at bar 1 open (100)
    intrabar = run_backtest(df, sig, stop_loss=0.03, take_profit=None, fee=0)
    assert intrabar.trades[0].exit_time == idx[2]       # wick stopped it out
    on_close = run_backtest(df, sig, stop_loss=0.03, take_profit=None, fee=0, stops_on_close=True)
    t = on_close.trades[0]
    assert t.exit_time == idx[4] and t.exit_price == 95 and t.reason == "stop"


def _dot_frame(green_at, wt2_vals, mf_vals):
    n = len(wt2_vals)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    green = pd.Series(False, index=idx)
    green.iloc[green_at] = True
    return pd.DataFrame({"wt2": wt2_vals, "money_flow": mf_vals, "green_dot": green,
                         "red_dot": False}, index=idx)


def test_double_dot_waits_for_higher_second_dot():
    from mcbot.strategies import double_dot, double_dot_only
    # Dot 1 at bar 4 (wt2 -70): money flow falling, so skip.
    # Dot 2 at bar 10 (wt2 -55, higher): money flow rising, so enter.
    mf = [5, 4, 3, 2, 1, 1, 1, 2, 3, 4, 5, 6]
    wt2 = [-50, -60, -65, -68, -70, -60, -50, -52, -54, -56, -55, -40]
    mc = _dot_frame([4, 10], wt2, mf)
    sig = double_dot(mc)
    assert not sig["long_entry"].iloc[4]
    assert sig["long_entry"].iloc[10]
    assert double_dot_only(mc)["long_entry"].iloc[10]

    # Second dot LOWER than the first is not the double-dot setup.
    # (double_dot still takes it as a plain dot with momentum.)
    wt2_lower = wt2.copy()
    wt2_lower[10] = -80
    assert not double_dot_only(_dot_frame([4, 10], wt2_lower, mf))["long_entry"].iloc[10]


def test_double_dot_takes_first_dot_when_momentum_lines_up():
    from mcbot.strategies import double_dot, double_dot_only
    mf = [1, 2, 3, 4, 5, 6, 7, 8]
    mc = _dot_frame([4], [-60] * 8, mf)
    assert double_dot(mc)["long_entry"].iloc[4]
    assert not double_dot_only(mc)["long_entry"].iloc[4]


def _one_trade(highs, lows, closes, **kw):
    n = len(highs) + 1
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({"open": [100.0] * 2 + list(closes[:-1]), "high": [100.0] + list(highs),
                       "low": [100.0] + list(lows), "close": [100.0] + list(closes),
                       "volume": 1.0}, index=idx)
    sig = pd.DataFrame(False, index=idx, columns=["long_entry", "long_exit", "short_entry", "short_exit"])
    sig.iloc[0, 0] = True  # enter at bar 1 open = 100
    return run_backtest(df, sig, **kw)


TP = [(0.01, 0.5), (0.02, 0.25), (0.03, 0.25)]


def test_scaled_take_profits_with_leverage():
    from mcbot.backtest import parse_take_profits
    assert parse_take_profits("1:50,2:25,3:25") == TP
    res = _one_trade([101.5, 102.5, 103.5], [99.5, 100.5, 101.5], [101, 102, 103],
                     stop_loss=0.03, take_profits=TP, leverage=5, fee=0.001)
    t = res.trades[0]
    assert t.targets_hit == 3 and t.reason == "target3"
    # 50% at +1%, 25% at +2%, 25% at +3%, 5x, 0.1% fee per side on notional.
    expected = 0.5 * (5 * 0.01 - 0.01) + 0.25 * (5 * 0.02 - 0.01) + 0.25 * (5 * 0.03 - 0.01)
    assert t.pnl_pct == pytest.approx(expected)


def test_target1_then_stop():
    res = _one_trade([101.5, 100], [99.5, 96.5], [101, 97],
                     stop_loss=0.03, take_profits=TP, leverage=5, fee=0.001)
    t = res.trades[0]
    assert t.targets_hit == 1 and t.reason == "stop"
    assert t.pnl_pct == pytest.approx(0.5 * (0.05 - 0.01) + 0.5 * (-0.15 - 0.01))


def test_liquidation_before_wide_stop():
    # 50x: liquidated after ~1.5% adverse move, well before the 3% stop.
    res = _one_trade([100.5], [98], [98.5], stop_loss=0.03, take_profits=TP, leverage=50)
    assert res.trades[0].reason == "liquidated"
    assert res.stats["final_equity"] == 0
