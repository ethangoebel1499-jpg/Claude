# Top-down confluence trading bot

A bot that trades one strategy: **top-down confluence**. It combines 4h support/resistance and the 21 EMA on the naked chart with Market Cipher B confirmation, and enters on **1h** candles. You can backtest it, score its entries, paper trade it, or (if you opt in) trade live on any exchange [ccxt](https://github.com/ccxt/ccxt) supports.

> Market Cipher is a closed-source, paid TradingView indicator. This project uses the open-source **VuManChu Cipher B** approximation (WaveTrend, green/red dots, RSI+MFI money flow), so its values are close to the real indicator but not identical.

## The strategy (`mcbot/strategies.py`)
Rules taken from a Market Cipher top-down analysis video and adapted to 4h levels with 1h entries.

**Long**, when all seven lights are green on a closed 1h candle:
1. **Levels (4h):** *this* candle closed back above a support/resistance zone. Zones are where several 4h turning points line up (about 0.4% wide), and they only use *closed* 4h candles.
2. **21 EMA (real candles):** price is closed above the 21 EMA, and the EMA is rising (the video's "white" EMA).
3. **Market Cipher B:**
   - a green-dot **anchor wave** in the last 20 bars,
   - Market Cipher B **came down first**: red dots (bearish wave crosses) printing mostly below the zero line in the last 30 bars,
   - **money flow blue** (above 0) and crossed up within the last 5 bars (never long while it's red),
   - a **trigger wave**: a WaveTrend cross up at a higher low than the anchor, still below +20 (not about to print a red dot).

Every condition is shown as 🟢 YES / 🔴 NO with an overall 🟢 TRADE / 🔴 NO TRADE light, by `signals` and in the bot's log on every closed candle. The entry is literally "all six lights green", so the checklist and the trades can't disagree.

**Exits (from the video, the `backtest` default `--exits video`):**
- **Stop** just below the 21 EMA (0.2% buffer, `--stop-buffer`). For shorts it goes just above the rejected resistance level.
- **Target 1:** close 50% at the next 4h support/resistance level. **Target 2:** close 25% at the level after that.
- The stop **trails the 21 EMA** on every closed candle, which moves it into profit and lets the rest ride.
- A full opposite setup also closes the trade. When the lights are green, `signals` prints the exact stop and target prices.

`--from 2017-01-01` limits the backtest to recent data.

`--exits fixed` keeps the percentage exits (`--stop`, `--tp`, `--trail`).

**Short:** the mirror image. A rejection below a zone, price below a falling 21 EMA, a red-dot anchor, money flow falling into red, and a lower-high trigger wave.

Vague parts I had to interpret: what counts as a level, the EMA colour, "big" anchor waves, "tight" triggers, and money flow "about to cross". Market Cipher B can optionally be calculated on Heikin Ashi candles (on by default, `--no-heikin-ashi` to turn off). Levels and the EMA always use real candles, and trades always fill at real prices.

### Results so far (no real BTC data yet)
Real EUR/USD 1h plus 4 stocks (SPY, GOOGL, GOOG, 002032) with daily candles treated as 1h; a hit means reaching 1 ATR before a 2 ATR stop:

| Layer | Signals | Hit target first | vs. random |
|---|---|---|---|
| Random | 37,459 | 66.2% | — |
| Chart only (level reclaim + rising 21 EMA) | 4,322 | 68.6% | **+2.4 (±0.7)** |
| + Market Cipher B trigger wave | 447 | 65.3% | -0.9 (±2.2) |
| + anchor green dot | 187 | 69.5% | +3.3 (±3.5) |
| Full strategy (+ money flow) | 31 | 61.3% | -4.9 (±8.5) |

The naked-chart part showed a small but statistically solid edge. The approximated Market Cipher B layers didn't improve it on this data, and the full rule set fires too rarely to judge. The real test is BTC 1h data, ideally with the real Market Cipher B values from a TradingView export.

## Usage
```bash
pip install -r requirements.txt

# Backtest with partial take-profits: 50% at +1.5%, 25% at +2.5%, rest rides until the
# opposite setup; stop steps up behind each target; 5x leverage; shorts on.
python main.py backtest --exchange binance --symbol BTC/USDT --limit 5000 --short --leverage 5 --fee 0.0005

# Or the fixed-percentage exits instead of the video's
python main.py backtest --exchange binance --symbol BTC/USDT --limit 5000 --short \
  --exits fixed --leverage 5 --stop 0.03 --tp "1.5:50,2.5:25,ride:25" --trail --fee 0.0005

# Entry quality: how often price hits the target before the stop, vs random entries
python main.py entries --exchange binance --symbol BTC/USDT --limit 5000           # 1.5% target / 3% stop
python main.py entries --csv tradingview_export.csv --mode atr --target-dist 1 --stop-dist 2

# Daily candles from any CSV, treated as 1h candles
python main.py entries --csv SPY_daily.csv --relabel 1h

# Green/red light checklist (every condition YES/NO + TRADE / NO TRADE) for the last closed candle
python main.py signals --exchange binance --symbol BTC/USDT
python main.py signals --csv datasets/btc_daily_close.csv --relabel 1h --date 2016-03-04

# Paper trade (simulated account saved to paper_state.json)
python main.py paper --exchange binance --symbol BTC/USDT

# Live: spot, long-only, needs both the --live flag and API keys
export MCBOT_API_KEY=... MCBOT_API_SECRET=...
python main.py live --exchange binance --symbol BTC/USDT --live --sandbox
```

Saved data: `datasets/XBTUSD_60.csv` holds **real Kraken BTC/USD 1h candles from 2013-10-06 to 2026-09-30** (102,936 candles, headerless Kraken OHLCVT format; `tools/kraken_extract.py` pulled it from Kraken's archive with range requests). `datasets/btc_1h_kraken_recent.csv` holds the last ~30 days of real Kraken BTC/USD 1h candles (2026-09-27 to 2026-10-27). Kraken's public candle API only serves the latest 720 candles.
`datasets/btc_daily_close.csv` holds BTC daily closing prices from 2010-07-18 to 2026-09-15 (5,904 days, `Date,Price`, close only, no open/high/low).

CSV files need a timestamp first column (unix seconds as in TradingView exports, unix ms, or a date) plus open/high/low/close, with volume optional.

## Money flow sensitivity
- `--mf-period N` (default 60): how many candles money flow averages over. **Smaller makes it looser and faster**, so it crosses zero more often and the money flow light turns green more.

Measured on the real data (EUR/USD 1h, BTC daily as 1h, 4 stocks as 1h) with the video's exits at 5×:

| Period | Zero crosses / 1,000 candles | Money flow light green | Signals | Trades | Win % | Avg trade (5×) |
|---|---|---|---|---|---|---|
| 60 (default) | 25 | 5.3% | 2 | 1 | 100% | +4.9% |
| 30 | 40 | 8.4% | 4 | 3 | 0% | -14.0% |
| 20 | 51 | 10.8% | 14 | 10 | 20% | -2.2% |
| 14 | 61 | 13.0% | 26 | 21 | 33% | -1.3% |

The samples are still small.

## Exits and risk (backtest)
- `--stop` and `--tp` levels are **price** moves. At 5x, a 3% stop costs 15% of the account, plus fees charged on the leveraged notional.
- `--tp "1.5:50,2.5:25,ride:25"`: the `ride` portion ignores the 21 EMA exit and stays open until the **opposite setup** fires. With `--short` it flips straight into the short.
- `--trail`: breakeven after target 1, then target 1's level after target 2. `--breakeven` only does the first step. Both take effect from the next candle.
- The whole account is used as margin. A position is liquidated (-100%) if price moves `1/leverage - 0.5%` against it before the stop fills.
- If a stop and a target are both hit in the same candle, the stop is assumed to fill first.

## Candle confirmation
- Entries and signal exits only happen on **closed** candles. The backtest fills at the next candle's open, and the bot drops the still-forming candle.
- On startup, the bot won't trade off a candle that closed a while ago.
- Stops and targets fire intrabar by default. `--stops-on-close` makes them wait for a closed candle too.

## Important caveats
- The paper/live bot uses a single stop and target. Leverage, partial take-profits and trailing are backtest-only for now.
- Bot stops are checked by the bot itself, not placed on the exchange, so they only protect you while it runs.
- This is educational code, not financial advice. Paper trade before risking real money.

## Tests
```bash
python -m pytest -q
```
