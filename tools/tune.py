"""Parameter tuning with an honest out-of-sample check.

  python tools/tune.py [--m1 path/to/XBTUSD_1.pkl] [--only S3,S4b,...]

For each strategy a grid of settings is tried. The winner is chosen ONLY on
2017-2021 (train) by total return at a 0.1% fee (at least 30 trades), then scored
on 2022-2026 (test), which the choice never saw. The best test result in
hindsight is shown too, to show how much picking by hindsight flatters a result.
Writes reports/tuning.md and reports/tuning.csv.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mcbot import s2, s5, s6, s7  # noqa: E402
from mcbot.backtest import run_backtest, run_level_backtest  # noqa: E402
from mcbot.data import fill_gaps, load_csv, resample  # noqa: E402
from mcbot.grind import grind_signals  # noqa: E402
from mcbot.witcher import witcher_signals  # noqa: E402

FEE, LOW_FEE = 0.001, 0.0004
SPLITS = {"train": ("2017-01-01", "2022-01-01"), "test": ("2022-01-01", "2027-01-01")}
rows: list[dict] = []


def periods(df):
    return {k: df[(df.index >= a) & (df.index < b)] for k, (a, b) in SPLITS.items()}


def record(strategy, params, df, fr, bt):
    """bt(df_part, fr_part, fee) -> Result. Records train/test stats."""
    r = {"strategy": strategy, **{"params": params}}
    for name, part in periods(df).items():
        f = fr.loc[part.index]
        s = bt(part, f, FEE).stats
        r[f"{name}_ret"], r[f"{name}_n"] = s["total_return_pct"], s["trades"]
        r[f"{name}_win"], r[f"{name}_dd"] = s["win_rate_pct"], s["max_drawdown_pct"]
        r[f"{name}_ret_lowfee"] = bt(part, f, LOW_FEE).stats["total_return_pct"]
    rows.append(r)
    print(f"{strategy:5} {params:70} train {r['train_ret']:+8.1f}% (n{r['train_n']})  "
          f"test {r['test_ret']:+8.1f}% (n{r['test_n']})", flush=True)


def tune_s3(h1):
    for tf in ("2h", "3h", "4h", "6h"):
        df = resample(h1, tf)
        sig = grind_signals(df)
        for stop, m, shorts in itertools.product((0.04, 0.06, 0.09, 0.12), (0.75, 1, 1.5, 2, 3), (True, False)):
            tps = [(0.017 * m, 0.33), (0.023 * m, 0.66), (0.034 * m, 0.01)]
            record("S3", f"BTC {tf} stop {stop:.0%} targets x{m} shorts={shorts}", df, sig,
                   lambda d, f, fee: run_backtest(d, f, allow_short=shorts, take_profits=tps,
                                                  stop_loss=stop, fee=fee))


def tune_s4(hourly):
    for coin, h1 in hourly.items():
        for tf in ("2h", "4h", "6h", "8h", "12h"):
            df = resample(h1, tf)
            sig = witcher_signals(df)
            for stop, (a, b), shorts in itertools.product(
                    (0.02, 0.03, 0.04, 0.06), ((0.03, 0.06), (0.05, 0.10), (0.08, 0.16), (0.12, 0.24)), (True, False)):
                record("S4b", f"{coin} {tf} stop {stop:.0%} targets {a:.0%}/{b:.0%} shorts={shorts}", df, sig,
                       lambda d, f, fee: run_backtest(d, f, allow_short=shorts, stop_loss=stop,
                                                      take_profits=[(a, 0.5), (b, 0.5)], fee=fee))


def tune_s2(m24):
    for anchor, armed, ribbon in itertools.product((40, 53, 60), (6, 12, 24), (False, True)):
        s2.ANCHOR_LEVEL, s2.ARMED_BARS = anchor, armed
        fr = s2.s2_frame(m24, need_ribbon=ribbon)
        for gray, partial, shorts in itertools.product(("after_white", "never"), (0.0, 0.2), (False, True)):
            record("S2", f"anchor -{anchor} armed {armed} ribbon={ribbon} gray_exit={gray} partial {partial:.0%} "
                         f"shorts={shorts}", m24, fr,
                   lambda d, f, fee: s2.s2_backtest(d, f, allow_short=shorts, fee=fee, gray_exit=gray,
                                                    partial=partial if partial else 1.0,
                                                    min_left=0.2 if partial else 1.0))
    s2.ANCHOR_LEVEL, s2.ARMED_BARS = 60, 12


def tune_s7(m12):
    for near, sig_bars, gap in itertools.product((0.003, 0.005, 0.01), (3, 5, 10), (20, 40)):
        s7.NEAR, s7.SIGNAL_BARS, s7.MAX_GAP = near, sig_bars, gap
        for exits, rr in itertools.product(("partial", "single"), (0.0, 1.5)):
            fr = s7.signals(m12, rr=rr, exits=exits)
            for shorts in (False, True):
                record("S7", f"near {near:.1%} signal bars {sig_bars} gap {gap} exits={exits} rr>={rr} "
                             f"shorts={shorts}", m12, fr,
                       lambda d, f, fee: s7.backtest(d, f, allow_short=shorts, fee=fee, exits=exits))
    s7.NEAR, s7.SIGNAL_BARS, s7.MAX_GAP = 0.005, 5, 40


def tune_s5(m1):
    fr = s5.signals(m1)
    c = m1["close"]
    for exits, shorts in itertools.product(("levels 30/70", "all at 2nd level", "2R", "3R", "5R", "3R+BE at 1.5R"),
                                           (False, True)):
        lv = fr[["long_stop", "long_tp1", "long_tp2", "long_trail", "short_stop", "short_tp1", "short_tp2",
                 "short_trail"]].copy()
        sizes, be = (0.3, 0.7), True
        if exits == "all at 2nd level":
            lv["long_tp1"], lv["short_tp1"] = lv.long_tp2, lv.short_tp2
            lv["long_tp2"] = lv["short_tp2"] = np.nan
            sizes, be = (1.0,), False
        elif exits in ("2R", "3R", "5R"):
            k = float(exits[:-1])
            lv["long_tp1"] = c + k * (c - lv.long_stop)
            lv["short_tp1"] = c - k * (lv.short_stop - c)
            lv["long_tp2"] = lv["short_tp2"] = np.nan
            sizes, be = (1.0,), False
        elif exits == "3R+BE at 1.5R":
            lv["long_tp1"], lv["long_tp2"] = c + 1.5 * (c - lv.long_stop), c + 3 * (c - lv.long_stop)
            lv["short_tp1"], lv["short_tp2"] = c - 1.5 * (lv.short_stop - c), c - 3 * (lv.short_stop - c)
            sizes, be = (0.5, 0.5), True
        sig = fr[["long_entry", "short_entry", "long_exit", "short_exit"]]
        frame = pd.concat([sig, lv], axis=1)
        record("S5", f"BTC 1m exits={exits} shorts={shorts}", m1, frame,
               lambda d, f, fee: run_level_backtest(d, f[sig.columns], f[lv.columns], allow_short=shorts,
                                                    fee=fee, sizes=sizes, breakeven_after_tp1=be))


def tune_s6(m1):
    for tf, room in itertools.product(("1min", "3min", "5min", "15min"), (0.0005, 0.002, 0.005)):
        df = m1 if tf == "1min" else resample(m1, tf)
        s6.STOP_ROOM = room
        fr = s6.signals(df)
        for shorts in (False, True):
            record("S6", f"BTC {tf} stop room {room:.2%} shorts={shorts}", df, fr,
                   lambda d, f, fee: s6.backtest(d, f, allow_short=shorts, fee=fee))
    s6.STOP_ROOM = 0.0005


def report():
    df = pd.DataFrame(rows)
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "tuning.csv", index=False)
    lines = ["# Parameter tuning", "",
             "Each setting was chosen **only on 2017-2021** (train) by total return at a 0.1% fee, with at least "
             "30 trades, then scored on **2022-2026** (test), which the choice never saw. "
             "'Hindsight best' is the setting that happened to do best in the test years: what you'd get by "
             "cherry-picking, shown only for comparison.", "",
             "| Strategy | Chosen on 2017-21 | Train | Test 2022-26 @0.1% | Test @0.04% | Test trades | Test max DD | "
             "Hindsight best (test) |",
             "|---|---|---|---|---|---|---|---|"]
    for strat, g in df.groupby("strategy", sort=False):
        ok = g[g.train_n >= 30]
        if ok.empty:
            continue
        best = ok.loc[ok.train_ret.idxmax()]
        hind = g.loc[g.test_ret.idxmax()]
        light = "🟢" if best.test_ret > 0 else "🔴"
        lines.append(f"| {strat} | {best.params} | {best.train_ret:+,.0f}% | {light} {best.test_ret:+,.0f}% | "
                     f"{best.test_ret_lowfee:+,.0f}% | {best.test_n} | {best.test_dd:.0f}% | "
                     f"{hind.test_ret:+,.0f}% ({hind.params}) |")
    lines += ["", "## Settings that were profitable in both periods at 0.1% fee", "",
              "| Strategy | Settings | Train | Test | Test trades | Test max DD |", "|---|---|---|---|---|---|"]
    both = df[(df.train_ret > 0) & (df.test_ret > 0) & (df.train_n >= 30) & (df.test_n >= 20)]
    for _, r in both.sort_values(["strategy", "test_ret"], ascending=[True, False]).groupby("strategy").head(5).iterrows():
        lines.append(f"| {r.strategy} | {r.params} | {r.train_ret:+,.0f}% | {r.test_ret:+,.0f}% | {r.test_n} | "
                     f"{r.test_dd:.0f}% |")
    (out / "tuning.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", help="Kraken 1m BTC candles (pickle from pandas or CSV), needed for S5 and S6")
    ap.add_argument("--only", default="S3,S4b,S2,S7,S5,S6")
    args = ap.parse_args()
    only = args.only.split(",")
    t = time.time()
    if "S3" in only:
        tune_s3(load_csv(ROOT / "datasets/XBTUSD_60.csv"))
    if "S4b" in only:
        tune_s4({c: load_csv(ROOT / f"datasets/{p}_60.csv") for c, p in
                 (("BTC", "XBTUSD"), ("ETH", "ETHUSD"), ("SOL", "SOLUSD"))})
    if "S2" in only:
        tune_s2(load_csv(ROOT / "datasets/XBTUSD_24m.csv"))
    if "S7" in only:
        tune_s7(load_csv(ROOT / "datasets/XBTUSD_12m.csv"))
    if args.m1 and ({"S5", "S6"} & set(only)):
        m1 = pd.read_pickle(args.m1) if args.m1.endswith(".pkl") else load_csv(args.m1)
        m1 = fill_gaps(m1[m1.index >= "2016-12-01"])
        if "S5" in only:
            tune_s5(m1)
        if "S6" in only:
            tune_s6(m1)
    report()
    print(f"DONE in {time.time() - t:.0f}s -> reports/tuning.md", flush=True)
