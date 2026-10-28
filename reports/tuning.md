# Parameter tuning

Each setting was chosen **only on 2017-2021** (train) by total return at a 0.1% fee, with at least 30 trades, then scored on **2022-2026** (test), which the choice never saw. 'Hindsight best' is the setting that happened to do best in the test years: what you'd get by cherry-picking, shown only for comparison.

| Strategy | Chosen on 2017-21 | Train | Test 2022-26 @0.1% | Test @0.04% | Test trades | Test max DD | Hindsight best (test) |
|---|---|---|---|---|---|---|---|
| S3 | BTC 2h stop 6% targets x3 shorts=True | +3,100% | 🟢 +72% | +144% | 292 | -61% | +166% (BTC 4h stop 6% targets x3 shorts=True) |
| S4b | ETH 2h stop 2% targets 12%/24% shorts=True | +1,172% | 🟢 +121% | +173% | 177 | -36% | +657% (SOL 2h stop 6% targets 12%/24% shorts=True) |
| S2 | anchor -40 armed 24 ribbon=True gray_exit=never partial 0% shorts=False | +3,357% | 🟢 +15% | +51% | 229 | -45% | +41% (anchor -53 armed 24 ribbon=False gray_exit=never partial 0% shorts=False) |
| S6 | BTC 15min stop room 0.05% shorts=False | -88% | 🔴 -90% | -52% | 1263 | -90% | -88% (BTC 15min stop room 0.50% shorts=False) |

## Settings that were profitable in both periods at 0.1% fee

| Strategy | Settings | Train | Test | Test trades | Test max DD |
|---|---|---|---|---|---|
| S2 | anchor -53 armed 24 ribbon=False gray_exit=never partial 0% shorts=False | +1,971% | +41% | 231 | -42% |
| S2 | anchor -53 armed 12 ribbon=False gray_exit=never partial 0% shorts=False | +1,796% | +39% | 226 | -44% |
| S2 | anchor -40 armed 24 ribbon=False gray_exit=never partial 0% shorts=False | +2,052% | +29% | 240 | -43% |
| S2 | anchor -40 armed 12 ribbon=False gray_exit=never partial 0% shorts=False | +1,847% | +23% | 236 | -47% |
| S2 | anchor -53 armed 24 ribbon=True gray_exit=never partial 0% shorts=False | +3,229% | +17% | 220 | -43% |
| S3 | BTC 4h stop 6% targets x3 shorts=True | +699% | +166% | 200 | -42% |
| S3 | BTC 3h stop 12% targets x3 shorts=False | +200% | +127% | 101 | -31% |
| S3 | BTC 4h stop 9% targets x3 shorts=True | +313% | +126% | 184 | -44% |
| S3 | BTC 4h stop 12% targets x3 shorts=True | +190% | +105% | 179 | -53% |
| S3 | BTC 4h stop 6% targets x1.5 shorts=True | +172% | +105% | 317 | -30% |
| S4b | ETH 2h stop 4% targets 12%/24% shorts=True | +798% | +187% | 177 | -35% |
| S4b | ETH 2h stop 6% targets 12%/24% shorts=True | +311% | +186% | 177 | -42% |
| S4b | ETH 2h stop 6% targets 8%/16% shorts=True | +242% | +142% | 177 | -36% |
| S4b | ETH 2h stop 4% targets 8%/16% shorts=True | +390% | +128% | 177 | -34% |
| S4b | ETH 2h stop 2% targets 12%/24% shorts=True | +1,172% | +121% | 177 | -36% |

## From the first run (see tuning_run1_log.txt)

- **S5** (levels + multi-timeframe divergence scalp, BTC 1m): every exit tried (levels 30/70, all at the 2nd level, 2R, 3R, 5R; longs only or both) lost **-89% to -99.9%** in both periods at a 0.1% fee. 🔴
- **S7** (range levels + 24m/12m): **none of 66 settings** was profitable in both periods; the setting chosen on 2017-21 (+26%) made -6% in 2022-26. 🔴
