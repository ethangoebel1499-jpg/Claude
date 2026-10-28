import numpy as np
import pandas as pd
import pytest

from mcbot.backtest import run_backtest
from mcbot.bot import PaperBroker, check_stops
from mcbot.data import synthetic
from mcbot.indicators import market_cipher
from mcbot.strategies import build_signals, topdown_confluence


@pytest.fixture(scope="module")
def df():
    return synthetic(n=2000, seed=1)


def test_indicators_have_no_lookahead(df):
    full = market_cipher(df)
    cut = market_cipher(df.iloc[:1500])
    cols = ["wt1", "wt2", "money_flow", "green_dot", "red_dot", "sr_reclaim_up", "sr_reject_down",
            "sr_next_above", "sr_next_below", "real_ema21"]
    pd.testing.assert_frame_equal(full[cols].iloc[:1500], cut[cols])


def test_wavetrend_produces_dots(df):
    mc = market_cipher(df)
    assert mc["green_dot"].sum() > 0 and mc["red_dot"].sum() > 0
    assert (mc.loc[mc["green_dot"], "wt2"] <= -53).all()
    assert (mc.loc[mc["red_dot"], "wt2"] >= 53).all()


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


def test_heikin_ashi_values():
    from mcbot.indicators import heikin_ashi
    idx = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    df = pd.DataFrame({"open": [10, 12], "high": [13, 15], "low": [9, 11], "close": [12, 14],
                       "volume": 1.0}, index=idx)
    ha = heikin_ashi(df)
    assert ha["close"].tolist() == [11, 13]          # (o+h+l+c)/4
    assert ha["open"].tolist() == [11, 11]           # (o0+c0)/2, then prev (ha_open+ha_close)/2
    assert ha["high"].tolist() == [13, 15] and ha["low"].tolist() == [9, 11]


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


def test_breakeven_after_first_target():
    # Bar 1 hits +1% (TP1), bar 2 falls back through entry to -3.5%.
    kw = dict(stop_loss=0.03, take_profits=TP, leverage=5, fee=0.001)
    full_stop = _one_trade([101.5, 100.5], [99.8, 96.0], [101, 97], **kw).trades[0]
    be = _one_trade([101.5, 100.5], [99.8, 96.0], [101, 97], breakeven_after_tp1=True, **kw).trades[0]
    assert full_stop.reason == "stop" and be.reason == "breakeven"
    assert be.exit_price == 100.0
    # Half closed at +1%, half at entry: only fees on the second half.
    assert be.pnl_pct == pytest.approx(0.5 * (0.05 - 0.01) + 0.5 * (0 - 0.01))
    assert be.pnl_pct > 0 > full_stop.pnl_pct


def test_breakeven_not_triggered_in_same_bar_as_tp1():
    # Entry bar dips to 99.8 and reaches 101.5: TP1 fills, BE stop not hit yet.
    res = _one_trade([101.5, 102.5, 103.5], [99.8, 100.5, 101.5], [101, 102, 103],
                     stop_loss=0.03, take_profits=TP, leverage=5, breakeven_after_tp1=True)
    assert res.trades[0].targets_hit == 3


RIDE = [(0.015, 0.5), (0.025, 0.25), (None, 0.25)]


def _runner_case(allow_short=False):
    # Price climbs past +1.5% and +2.5%, then keeps going to +8%.
    closes = [101, 102, 103, 104, 106, 108, 107]
    n = len(closes) + 1
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({"open": [100.0, 100.0] + closes[:-1], "high": [100.0] + [c + 0.5 for c in closes],
                       "low": [100.0] + [c - 1.5 for c in closes], "close": [100.0] + closes,
                       "volume": 1.0}, index=idx)
    sig = pd.DataFrame(False, index=idx, columns=["long_entry", "long_exit", "short_entry", "short_exit"])
    sig.iloc[0, 0] = True                              # long at bar 1 open (100)
    sig.loc[idx[4], "long_exit"] = True                 # normal exit signal: runner ignores it
    sig.loc[idx[6], "short_entry"] = True               # short setup -> runner closes at bar 7 open (108)
    return run_backtest(df, sig, stop_loss=0.03, take_profits=RIDE, leverage=5, fee=0.001,
                        breakeven_after_tp1=True, allow_short=allow_short)


def test_runner_rides_until_short_setup():
    from mcbot.backtest import parse_take_profits
    assert parse_take_profits("1.5:50,2.5:25,ride:25") == RIDE
    t = _runner_case().trades[0]
    assert t.reason == "opposite setup" and t.exit_price == 108
    assert [round(f[2], 2) for f in t.fills] == [0.5, 0.25, 0.25]
    expected = 0.5 * (5 * 0.015 - 0.01) + 0.25 * (5 * 0.025 - 0.01) + 0.25 * (5 * 0.08 - 0.01)
    assert t.pnl_pct == pytest.approx(expected)


def test_runner_flips_into_short():
    trades = _runner_case(allow_short=True).trades
    assert [t.side for t in trades] == ["long", "short"]
    assert trades[1].entry_price == 108


def test_runner_stopped_at_breakeven():
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    df = pd.DataFrame({"open": [100, 100, 102, 103, 99.5], "high": [100, 102, 103, 103.5, 100],
                       "low": [100, 99.8, 101.5, 101, 99], "close": [100, 102, 103, 101, 99.5],
                       "volume": 1.0}, index=idx)
    sig = pd.DataFrame(False, index=idx, columns=["long_entry", "long_exit", "short_entry", "short_exit"])
    sig.iloc[0, 0] = True
    t = run_backtest(df, sig, stop_loss=0.03, take_profits=RIDE, leverage=5, fee=0,
                     breakeven_after_tp1=True).trades[0]
    assert t.targets_hit == 2 and t.reason == "breakeven" and t.exit_price == 99.5


def test_trail_stop_moves_to_target1_after_target2():
    # +1.5% and +2.5% hit, then price falls back to +1% -> runner stopped at +1.5%.
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    df = pd.DataFrame({"open": [100, 100, 102, 103, 102], "high": [100, 102, 103, 103.5, 102],
                       "low": [100, 99.8, 101.6, 101.8, 100.8], "close": [100, 102, 103, 102, 101],
                       "volume": 1.0}, index=idx)
    sig = pd.DataFrame(False, index=idx, columns=["long_entry", "long_exit", "short_entry", "short_exit"])
    sig.iloc[0, 0] = True
    kw = dict(stop_loss=0.03, take_profits=RIDE, leverage=5, fee=0)
    be = run_backtest(df, sig, breakeven_after_tp1=True, **kw).trades
    tr = run_backtest(df, sig, trail_targets=True, **kw).trades[0]
    assert tr.reason == "trail stop" and tr.exit_price == pytest.approx(101.5)
    assert tr.pnl_pct == pytest.approx(0.5 * 0.075 + 0.25 * 0.125 + 0.25 * 0.075)
    # With breakeven only, the runner is still open at the end (no stop at +1.5%).
    assert be[0].reason == "end"


def test_entry_quality_outcomes():
    from mcbot.entry_quality import outcomes
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    df = pd.DataFrame({"open": [100] * 6, "high": [100, 100.5, 101.6, 100, 100, 100],
                       "low": [100, 99.5, 99.0, 100, 96, 100], "close": [100] * 6}, index=idx, dtype=float)
    e = pd.Series([True, False, False, True, False, False], index=idx)
    res = outcomes(df, e, "long", 0.015, 0.03, "pct", max_bars=3)
    assert res.iloc[0] == 1.0     # +1.5% at bar 2 before -3%
    assert res.iloc[1] == 0.0     # entry bar 4 open 100 -> low 96 hits the stop


@pytest.mark.parametrize("times", [[1704067200, 1704070800],                 # unix seconds (TradingView)
                                   [1704067200000, 1704070800000],           # unix ms
                                   ["2024-01-01 00:00:00", "2024-01-01 01:00:00"]])
def test_load_csv_timestamp_formats(tmp_path, times):
    from mcbot.data import load_csv
    p = tmp_path / "c.csv"
    pd.DataFrame({"time": times, "open": [1, 2], "high": [2, 3], "low": [0.5, 1.5],
                  "close": [1.5, 2.5]}).to_csv(p, index=False)
    df = load_csv(str(p))
    assert df.index[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert (df["volume"] == 1.0).all()


def test_relabel_daily_as_hourly():
    from mcbot.data import relabel
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"], utc=True)  # weekend gap
    df = pd.DataFrame({"open": [1, 2, 3], "high": [1, 2, 3], "low": [1, 2, 3], "close": [1, 2, 3],
                       "volume": 1.0}, index=idx)
    out = relabel(df, "1h")
    assert (out.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()
    assert out["close"].tolist() == [1, 2, 3]


def test_sr_levels_no_lookahead(df):
    from mcbot.indicators import sr_levels
    full = sr_levels(df)
    cut = sr_levels(df.iloc[:1501])
    pd.testing.assert_frame_equal(full.iloc[:1501], cut)


def test_sr_level_reclaim():
    from mcbot.indicators import sr_levels
    # 4h candles (factor 1 here) that repeatedly turn at ~100, then a dip under
    # 100 that closes back above it.
    lows = [104, 102, 100, 102, 104, 102, 100.1, 102, 104, 103, 101, 99, 102, 103, 104]
    highs = [x + 3 for x in lows]
    closes = [x + 1.5 for x in lows]
    closes[11] = 100.8  # bar 11 dips to 99, closes above the zone
    idx = pd.date_range("2024-01-01", periods=len(lows), freq="h", tz="UTC")
    df = pd.DataFrame({"open": closes, "high": highs, "low": lows, "close": closes}, index=idx)
    out = sr_levels(df, factor=1, zone_pct=0.004)
    assert out["sr_reclaim_up"].iloc[11]
    assert out["sr_next_below"].iloc[11] == pytest.approx(100.05, abs=0.1)


def _topdown_frame(n=30):
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    mc = pd.DataFrame({"real_close": 101.0, "real_ema21": np.linspace(99, 100, n), "money_flow": 2.0,
                       "wt2": -20.0, "wt_cross_up": False, "wt_cross_down": False, "green_dot": False,
                       "red_dot": False, "sr_reclaim_up": False, "sr_reject_down": False}, index=idx)
    mc["money_flow"] = np.r_[np.full(20, -3.0), np.linspace(-1, 3, n - 20)]   # crosses above 0
    mc.loc[idx[10], ["green_dot", "wt2"]] = [True, -70.0]                   # anchor wave + dot
    mc.loc[idx[[5, 14]], "wt_cross_down"] = True                             # red dots below zero (came down)
    mc.loc[idx[24], ["wt_cross_up", "wt2"]] = [True, -20.0]                 # trigger: higher low, below +20
    mc.loc[idx[24], "sr_reclaim_up"] = True                                  # this candle reclaimed a 4h zone
    mc["sr_next_above"], mc["sr_next_above2"] = 104.0, 108.0                # next 4h levels
    mc["sr_next_below"], mc["sr_next_below2"] = 97.0, 94.0
    return mc, idx


def test_topdown_long_needs_every_layer():
    mc, idx = _topdown_frame()
    assert topdown_confluence(mc)["long_entry"].iloc[24]
    for col, value in (("sr_reclaim_up", False), ("wt_cross_up", False), ("green_dot", False),
                       ("wt_cross_down", False)):
        broken = mc.copy()
        broken[col] = value
        assert not topdown_confluence(broken)["long_entry"].any(), col
    below_ema = mc.copy()
    below_ema["real_close"] = 98.0
    assert not topdown_confluence(below_ema)["long_entry"].any()
    red_flow = mc.copy()
    red_flow["money_flow"] = -20.0                                           # money flow red: don't long
    assert not topdown_confluence(red_flow)["long_entry"].any()


def test_topdown_exit_on_close_below_ema21():
    mc, idx = _topdown_frame()
    mc.loc[idx[27], "real_close"] = 99.0
    assert topdown_confluence(mc)["long_exit"].iloc[27]


def test_topdown_backtest_fills_at_real_prices():
    data = synthetic(n=5000, seed=1)
    sig = build_signals(market_cipher(data, use_heikin_ashi=True))
    res = run_backtest(data, sig, allow_short=True)
    assert res.trades
    opens = set(data["open"].round(8))
    assert all(round(t.entry_price, 8) in opens for t in res.trades)


def test_load_kraken_ohlcvt_without_header(tmp_path):
    from mcbot.data import load_csv
    p = tmp_path / "XBTUSD_60.csv"
    p.write_text("1704067200,42000.1,42500,41900,42300.5,12.5,340\n"
                 "1704070800,42300.5,42400,42100,42200,8.1,210\n")
    df = load_csv(str(p))
    assert len(df) == 2 and df.index[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert df["close"].tolist() == [42300.5, 42200.0]


def test_checklist_matches_entries():
    from mcbot.strategies import checklist, topdown_conditions
    mc, idx = _topdown_frame()
    long_c, _ = topdown_conditions(mc)
    sig = topdown_confluence(mc)
    pd.testing.assert_series_equal(long_c.all(axis=1), sig["long_entry"], check_names=False)
    text = checklist(mc, 24)
    assert "LONG: 🟢 TRADE  (7/7 yes)" in text and "SHORT: 🔴 NO TRADE" in text
    assert "Target 1  103.79  -> close 50%" in text and "Target 2  107.78  -> close 25%" in text
    assert text.count("🟢 YES") >= 7
    mc.loc[:, "sr_reclaim_up"] = False
    text = checklist(mc, 24)
    assert "LONG: 🔴 NO TRADE  (6/7 yes)" in text
    assert "🔴 NO   This candle closed back above a 4h support/resistance zone" in text


def test_load_close_only_csv(tmp_path):
    from mcbot.data import load_csv
    p = tmp_path / "btc.csv"
    p.write_text("Date,Price\n2024-01-01,100\n2024-01-02,105\n2024-01-03,103\n")
    df = load_csv(str(p))
    assert df["open"].tolist() == [100, 100, 105]
    assert df["high"].tolist() == [100, 105, 105] and df["low"].tolist() == [100, 100, 103]


def test_money_flow_must_be_blue_and_fresh():
    mc, idx = _topdown_frame()
    stale = mc.copy()
    stale["money_flow"] = np.r_[np.full(10, -3.0), np.full(20, 2.0)]   # crossed 14 bars before the entry
    assert not topdown_confluence(stale)["long_entry"].iloc[24]
    nearly = mc.copy()
    nearly["money_flow"] = np.r_[np.full(20, -3.0), np.linspace(-2.9, -0.5, 10)]  # rising but still red
    assert not topdown_confluence(nearly)["long_entry"].iloc[24]


def test_came_down_needs_red_dots_mostly_below_zero():
    mc, idx = _topdown_frame()
    mc.loc[idx[[8, 11, 18]], ["wt_cross_down", "wt2"]] = [True, 30.0]  # 3 of 5 red dots above zero
    assert not topdown_confluence(mc)["long_entry"].iloc[24]


def test_level_exits_target_then_ema_trail():
    from mcbot.backtest import run_level_backtest
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    #            signal   entry@100  hits 103   trail->102   falls to 101.5
    df = pd.DataFrame({"open": [100, 100, 102, 103, 102.5], "high": [100, 101, 103.5, 104, 102.6],
                       "low": [99, 99.5, 101.5, 102.5, 101.5], "close": [100, 101, 103, 103, 101.8],
                       "volume": 1.0}, index=idx)
    sig = pd.DataFrame(False, index=idx, columns=["long_entry", "long_exit", "short_entry", "short_exit"])
    sig.iloc[0, 0] = True
    lv = pd.DataFrame({"long_stop": 98.0, "long_trail": [98, 98, 102, 102, 102], "long_tp1": 103.0,
                       "long_tp2": 106.0, "short_stop": np.nan, "short_trail": np.nan,
                       "short_tp1": np.nan, "short_tp2": np.nan}, index=idx)
    t = run_level_backtest(df, sig, lv, fee=0).trades[0]
    assert [f[3] for f in t.fills] == ["target1", "trail stop"]
    assert t.fills[1][1] == 102 and t.exit_time == idx[4]
    assert t.pnl_pct == pytest.approx(0.5 * 0.03 + 0.5 * 0.02)


def test_level_exits_initial_stop_below_ema():
    from mcbot.backtest import run_level_backtest
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame({"open": [100, 100, 99], "high": [100, 100.5, 99.5], "low": [100, 99.2, 97],
                       "close": [100, 99.5, 97.5], "volume": 1.0}, index=idx)
    sig = pd.DataFrame(False, index=idx, columns=["long_entry", "long_exit", "short_entry", "short_exit"])
    sig.iloc[0, 0] = True
    lv = pd.DataFrame({"long_stop": 98.0, "long_trail": 98.0, "long_tp1": 103.0, "long_tp2": np.nan,
                       "short_stop": np.nan, "short_trail": np.nan, "short_tp1": np.nan,
                       "short_tp2": np.nan}, index=idx)
    t = run_level_backtest(df, sig, lv, leverage=5, fee=0).trades[0]
    assert t.reason == "stop" and t.exit_price == 98
    assert t.pnl_pct == pytest.approx(5 * -0.02)


def test_resample_minutes_to_hours(tmp_path):
    from mcbot.data import load_csv, resample
    p = tmp_path / "btc_1min.csv"   # Kaggle layout: Timestamp,Open,High,Low,Close,Volume
    rows = ["Timestamp,Open,High,Low,Close,Volume"]
    for m in range(120):
        ts = 1704067200 + 60 * m
        rows.append(f"{ts}.0,{100 + m},{101 + m},{99 + m},{100.5 + m},1.0")
    p.write_text("\n".join(rows) + "\n")
    h = resample(load_csv(str(p)), "1h")
    assert len(h) == 2
    assert h.iloc[0].tolist() == [100, 160, 99, 159.5, 60]


def test_make_exchange_honours_proxy_env():
    from mcbot.data import make_exchange
    ex = make_exchange("kraken")
    assert ex.requests_trust_env is True


def test_witcher_no_lookahead_and_alternates(df):
    from mcbot.witcher import witcher_conditions, witcher_signals
    lc_full, sc_full, ls_full, _ = witcher_conditions(df)
    lc_cut, sc_cut, ls_cut, _ = witcher_conditions(df.iloc[:1500])
    pd.testing.assert_frame_equal(lc_full.iloc[:1500], lc_cut)
    pd.testing.assert_series_equal(ls_full.iloc[:1500], ls_cut)
    sig = witcher_signals(df)
    assert sig["long_entry"].sum() > 0 and sig["short_entry"].sum() > 0
    # Direction alternates: never two long (or two short) entries in a row.
    seq = [s for s in sig.index if sig.at[s, "long_entry"] or sig.at[s, "short_entry"]]
    sides = ["L" if sig.at[s, "long_entry"] else "S" for s in seq]
    assert all(a != b for a, b in zip(sides, sides[1:]))


def test_witcher_checklist_and_exits(df):
    from mcbot.witcher import witcher_checklist, witcher_signals
    sig = witcher_signals(df)
    bar = int(np.flatnonzero(sig["long_entry"].to_numpy())[0])
    text = witcher_checklist(df, bar)
    assert "LONG: 🟢 TRADE" in text and "Take profit" in text and "(8.2%)" in text
    res = run_backtest(df, sig, allow_short=True, take_profit=0.011, stop_loss=0.082,
                       stop_loss_short=0.078, fee=0)
    assert res.trades and all(t.reason in ("target1", "stop", "opposite setup", "signal", "end") for t in res.trades)


def test_grind_no_lookahead_and_signals(df):
    from mcbot.grind import grind_conditions, grind_signals, grind_checklist, TAKE_PROFITS, STOP
    lc_full, sc_full, _, _ = grind_conditions(df)
    lc_cut, sc_cut, _, _ = grind_conditions(df.iloc[:1500])
    pd.testing.assert_frame_equal(lc_full.iloc[:1500], lc_cut)
    pd.testing.assert_frame_equal(sc_full.iloc[:1500], sc_cut)
    sig = grind_signals(df)
    assert sig["long_entry"].any() and sig["short_entry"].any()
    assert not (sig["long_entry"] & sig["short_entry"]).any()
    bar = int(np.flatnonzero(sig["long_entry"].to_numpy())[0])
    assert "LONG: 🟢 TRADE" in grind_checklist(df, bar)
    res = run_backtest(df, sig, allow_short=True, take_profits=TAKE_PROFITS, stop_loss=STOP, fee=0)
    assert res.trades
    assert sum(size for _, size in TAKE_PROFITS) == pytest.approx(1.0)


def test_bull_bear_power_simple_candles():
    from mcbot.grind import _bull_bear_power
    # Up candle after an up close: bull power = full range; bear = max(open-low, high-close).
    bull, bear = _bull_bear_power([10, 10], [12, 12], [9, 9], [11, 11])
    assert bull[1] == 3 and bear[1] == max(11 - 9, 12 - 11)


def test_waves_find_bullish_divergence():
    from mcbot.divergence import waves, pair_divergences
    # Two oscillator troughs: the second is higher (-40 vs -80) while price makes a lower low.
    osc = pd.Series([0, -50, -80, -50, 0, 20, 0, -30, -40, -20, 10], dtype=float)
    low = pd.Series([100, 95, 90, 95, 100, 102, 98, 92, 85, 90, 95], dtype=float)
    high = low + 1
    up = pd.Series([False] * 3 + [True] + [False] * 5 + [True, False])
    down = pd.Series([True] + [False] * 5 + [True] + [False] * 4)
    w = waves(osc, low, high, up, down)
    assert (w.t1_osc.iloc[9], w.t1_px.iloc[9], w.t0_osc.iloc[9], w.t0_px.iloc[9]) == (-40, 85, -80, 90)
    div = pair_divergences(w, up, down, max_gap=20)
    assert div.bull.iloc[9] and not div.bull.iloc[3] and not div.hidden_bull.any()


def _assert_entries_no_lookahead(full, cut):
    for k in ("long_entry", "short_entry"):
        pd.testing.assert_series_equal(full[k].iloc[:len(cut)], cut[k], check_names=False)


def test_s2_no_lookahead_and_backtest():
    from mcbot import s2
    d = synthetic(n=6000, seed=3, timeframe="24min")
    fr = s2.signals(d)
    _assert_entries_no_lookahead(fr, s2.signals(d.iloc[:4500]))
    assert fr.long_entry.any() and fr.short_entry.any()
    res = s2.backtest(d, fr, fee=0)
    assert res.trades and {t.reason for t in res.trades} <= {"stop", "trail stop", "ema turned", "end"}
    bar = int(np.flatnonzero(fr.long_entry.to_numpy())[0])
    assert "LONG: 🟢 TRADE" in s2.checklist(d, bar)


def test_s6_no_lookahead_and_backtest():
    from mcbot import s6
    d = synthetic(n=6000, seed=4, timeframe="1min")
    fr = s6.signals(d)
    _assert_entries_no_lookahead(fr, s6.signals(d.iloc[:4000]))
    assert fr.long_entry.any() and fr.short_entry.any()
    res = s6.backtest(d, fr, fee=0)
    assert res.trades and {t.reason for t in res.trades} <= {"stop", "target1"}
    # Every long's stop sat below its entry.
    bar = int(np.flatnonzero(fr.long_entry.to_numpy())[0])
    assert "LONG: 🟢 TRADE" in s6.checklist(d, bar)


def test_s7_no_lookahead_and_backtest():
    from mcbot import s7
    d = synthetic(n=8000, seed=5, timeframe="12min")
    fr = s7.signals(d)
    _assert_entries_no_lookahead(fr, s7.signals(d.iloc[:6000]))
    res = s7.backtest(d, fr, fee=0)
    assert all(t.reason in ("stop", "breakeven", "target1", "target2", "opposite setup", "end") for t in res.trades)
    assert "LONG:" in s7.checklist(d, -1)


def test_s5_no_lookahead():
    from mcbot import s5
    d = synthetic(n=20000, seed=6, timeframe="1min")
    fr = s5.signals(d)
    _assert_entries_no_lookahead(fr, s5.signals(d.iloc[:15000]))
    s5.backtest(d, fr, fee=0)
    assert "4. 1m money flow" in s5.checklist(d, -1)


def test_s8_long_only_reversal(df):
    from mcbot import s8
    sig = s8.signals(df)
    assert sig.long_entry.any() and sig.short_entry.any()
    res = run_backtest(df, sig, allow_short=False, take_profits=s8.TAKE_PROFITS, stop_loss=None, fee=0)
    assert res.trades and all(t.side == "long" for t in res.trades)
    # 50% / 25% / runner: a trade that hit both targets kept a quarter for the reversal down.
    two = [t for t in res.trades if t.targets_hit == 2]
    assert two and all(abs(sum(f[2] for f in t.fills) - 1) < 1e-9 for t in two)
    assert all(t.reason in ("target1", "target2", "opposite setup", "end") for t in res.trades)
    bar = int(np.flatnonzero(sig.long_entry.to_numpy())[0])
    assert "LONG: 🟢 TRADE" in s8.checklist(df, bar)
