# What price did after each pattern

Real Kraken candles, 2017 to now, 8 coins × 4h, 1D. The detector walks forward candle by candle (no look-ahead). A break = a candle closing beyond a line. **Target** = the pattern's height measured from the break, reached before the stop (price back inside past the opposite line). **Fake** = closed back inside within 3 candles. **Move after 10** = average % in the break's direction 10 candles later.

| Pattern | Breaks | Broke the usual way | Usual-way breaks: target hit | Against-usual breaks: target hit | Fake breaks | Retested the line | Move after 10 candles |
|---|---|---|---|---|---|---|---|
| Ascending triangle | 765 | 36% | 36% | 31% | 26% | 76% | +0.02% |
| Descending triangle | 756 | 37% | 30% | 33% | 24% | 76% | +0.30% |
| Symmetrical triangle | 1273 | 48% | 17% | 17% | 22% | 72% | +0.33% |
| Rising wedge | 884 | 79% | 30% | 38% | 25% | 67% | -0.11% |
| Falling wedge | 760 | 80% | 29% | 35% | 27% | 71% | -0.39% |
| Ascending channel | 783 | 21% | 65% | 56% | 25% | 68% | +0.41% |
| Descending channel | 760 | 17% | 54% | 55% | 25% | 68% | +0.43% |
| Range | 269 | - | - | - | 36% | 76% | +0.48% |
| Bull flag | 18 | 72% | 38% | 40% | 33% | 61% | -3.37% |
| Bear flag | 28 | 79% | 41% | 50% | 46% | 71% | -0.32% |
| Pennant | 1 | 100% | 0% | - | 0% | 100% | -1.18% |
| Broadening pattern | 540 | - | - | - | 30% | 80% | +0.78% |
| Double top | 384 | 43% | 48% | 53% | 35% | 83% | +0.53% |
| Double bottom | 351 | 39% | 64% | 54% | 37% | 80% | -0.04% |
| Head and shoulders | 250 | 40% | 41% | 50% | 33% | 82% | +1.37% |
| Inverse head and shoulders | 275 | 43% | 48% | 43% | 36% | 77% | +1.32% |

## Squeezed near the apex (85%+ of the way to where the lines meet)

| Pattern | Times | Never broke (fizzled) | Broke the usual way | Late break hit target | Late break faked |
|---|---|---|---|---|---|
| Ascending triangle | 144 | 12% | 38% | 14% | 24% |
| Descending triangle | 174 | 10% | 39% | 16% | 18% |
| Symmetrical triangle | 891 | 9% | 49% | 11% | 20% |
| Rising wedge | 53 | 9% | 81% | 8% | 25% |
| Falling wedge | 54 | 13% | 68% | 13% | 23% |
| Pennant | 1 | 0% | 100% | 0% | 0% |

## Chance of breaking UP (95% range, number of breaks)

**On random data** = the same detector on random prices with no real patterns: the share of breaks UP that comes purely from how the lines are drawn (the sloped line is closer to price, so it gets broken first). **Real edge** = real minus random - the part that is actually market behaviour. 

Break = first candle close beyond either line. The range is where the true chance most likely sits given how many breaks were seen. *Trend before* = price higher or lower than 50 candles before the pattern started. Published = Bulkowski, *Encyclopedia of Chart Patterns* / thepatternsite.com (US stocks).

| Pattern | All | On random data | Real edge | Trend before UP | Trend before DOWN | 4h | Daily | Published (stocks) |
|---|---|---|---|---|---|---|---|---|
| Ascending triangle | 36% (32-39%, n=765) | 42% | -6 pts | 35% (31-40%, n=369) | 36% (32-41%, n=396) | 36% (32-40%, n=630) | 36% (28-44%, n=135) | breaks up 63% |
| Descending triangle | 63% (59-66%, n=756) | 57% | +5 pts | 63% (58-68%, n=395) | 62% (57-67%, n=361) | 64% (60-68%, n=619) | 58% (49-66%, n=137) | breaks up 53% |
| Symmetrical triangle | 51% (48-53%, n=1273) | 51% | -0 pts | 49% (45-53%, n=615) | 52% (48-56%, n=658) | 50% (47-53%, n=1097) | 53% (46-61%, n=176) | - |
| Rising wedge | 21% (19-24%, n=884) | 26% | -5 pts | 23% (19-27%, n=397) | 20% (17-24%, n=487) | 19% (17-22%, n=775) | 36% (27-45%, n=109) | breaks down 60% ("close to random") |
| Falling wedge | 80% (77-83%, n=760) | 75% | +6 pts | 85% (81-88%, n=415) | 75% (70-79%, n=345) | 80% (77-83%, n=631) | 82% (75-88%, n=129) | - |
| Ascending channel | 21% (19-24%, n=783) | 25% | -4 pts | 23% (19-28%, n=360) | 20% (16-24%, n=423) | 21% (18-24%, n=675) | 26% (19-35%, n=108) | - |
| Descending channel | 83% (80-86%, n=760) | 79% | +4 pts | 83% (79-86%, n=411) | 84% (80-87%, n=349) | 83% (80-86%, n=660) | 82% (73-88%, n=100) | - |
| Range | 50% (44-56%, n=269) | 51% | -1 pts | 49% (41-57%, n=142) | 50% (42-59%, n=127) | 51% (44-57%, n=231) | 45% (30-60%, n=38) | - |
| Bull flag | 72% (49-88%, n=18) | - | - | 71% (47-87%, n=17) | 100% (21-100%, n=1) | 73% (48-89%, n=15) | 67% (21-94%, n=3) | - |
| Bear flag | 21% (10-40%, n=28) | - | - | 0% (0-79%, n=1) | 22% (11-41%, n=27) | 14% (5-33%, n=22) | 50% (19-81%, n=6) | - |
| Pennant | 100% (21-100%, n=1) | - | - | 100% (21-100%, n=1) | - | 100% (21-100%, n=1) | - | - |
| Broadening pattern | 53% (48-57%, n=540) | 55% | -2 pts | 57% (51-62%, n=309) | 47% (41-54%, n=231) | 54% (50-59%, n=457) | 42% (32-53%, n=83) | - |
| Double top | 57% (52-62%, n=384) | 52% | +5 pts | 52% (45-58%, n=223) | 63% (56-70%, n=161) | 57% (52-63%, n=336) | 52% (38-66%, n=48) | - |
| Double bottom | 39% (34-44%, n=351) | 48% | -9 pts | 39% (32-46%, n=160) | 39% (33-46%, n=191) | 39% (33-44%, n=309) | 43% (29-58%, n=42) | - |
| Head and shoulders | 60% (53-65%, n=250) | 63% | -3 pts | 59% (50-67%, n=128) | 61% (52-69%, n=122) | 61% (55-67%, n=222) | 46% (30-64%, n=28) | - |
| Inverse head and shoulders | 43% (37-48%, n=275) | 36% | +7 pts | 38% (30-47%, n=118) | 46% (38-54%, n=157) | 42% (36-48%, n=246) | 48% (31-66%, n=29) | - |

## Early vs late breaks (triangles, wedges, pennants)

| When it broke | Breaks | Target hit | Fake | Move after 10 candles |
|---|---|---|---|---|
| Before 75% of the way to the apex | 2486 | 37% | 26% | +0.17% |
| 75% or later | 1953 | 15% | 22% | -0.07% |
