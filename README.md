# Top-down confluence trading bot

A bot that trades one strategy: **top-down confluence**. It combines 4h support/resistance and the 21 EMA on the naked chart with Market Cipher B confirmation, and enters on **1h** candles. You can backtest it, score its entries, paper trade it, or (if you opt in) trade live on any exchange [ccxt](https://github.com/ccxt/ccxt) supports.

> Market Cipher is a closed-source, paid TradingView indicator. This project uses the open-source **VuManChu Cipher B** approximation (WaveTrend, green/red dots, RSI+MFI money flow), so its values are close to the real indicator but not identical.

## Strategy catalog
Each strategy gets a number and a label so they stay separate. List them with `python main.py strategies`, and pick one with `--strategy S1`.

| ID | Name | Timeframes | Status |
|---|---|---|---|
| S1 | Top-down confluence | 4h levels, 1h entries | 🟢 ready |
| S2 | 4-hour / 15-minute | 4h bias, 15m entries | 🔴 rules pending (send the transcript) |
| S3 | GRIND BTC (wielkieef) | 45m (from Kraken 15m candles) | 🟢 ready. Port of his "GRIND BTC [45MIN]" strategy, using only its own indicators |

**S3 - GRIND BTC** (`mcbot/grind.py`): a long needs all nine of its conditions (Masanakamura ADX(9) > 12, RSI(100, opens) < 70, MFI > 8, volume > 1.3x its 51-bar average, 50-bar midpoint average, MACD 3/4/7, rising 55-bar SMA, SAR, cumulative volume delta above its 5-bar SMA), *or* its Heikin Ashi scalp signal while ADX agrees. Like the script's backtest it enters whenever conditions hold and reverses on an opposite signal. Exits: 9% stop; take-profits at +1.7% / +2.3% / +3.4% for 33% / 66% / the last ~1%, as TradingView fills them.
```bash
python main.py signals  --strategy S3 --csv datasets/XBTUSD_15.csv --resample 45min
python main.py backtest --strategy S3 --csv datasets/XBTUSD_15.csv --resample 45min --from 2017-01-01 --short --fee 0.0004
```
Real Kraken BTC/USD, 2017 onward, no leverage:

| Timeframe | Fee | Trades | Win % | Total | 2017-2021 | 2022-2026 | Entry edge vs random |
|---|---|---|---|---|---|---|---|
| 45m (native) | 0.04% | 2,380 | 66% | +2,538% | +1,978% | +29% | +2.3 ±0.5 |
| 45m (native) | 0.10% | 2,380 | 66% | +52% | +292% | -61% | |
| 4h | 0.04% | 929 | 81% | +729% | +285% | +113% | +4.7 ±1.1 |
| 4h | 0.10% | 929 | 81% | +172% | +99% | +35% | |

The first strategy here whose entries beat random by a statistically clear margin. Caveats: the author likely tuned it on BTC history, so 2017-2021 is probably flattering; futures funding and slippage aren't modelled; buy and hold made +8,532% over the same period.
| S4 | The Witcher (wielkieef) | 30m (from Kraken 15m candles) | 🟢 ready. Port of his "The Witcher [30MIN]" script (MPL-2.0), using only its own indicators |

**S4 - The Witcher** (`mcbot/witcher.py`): a long needs all nine of its conditions (ADX > 17 with +DI over -DI, RSI(35) < 70, MFI > 10, midpoint average, SAR, Donchian midline, momentum, rising VWMA, range filter), *or* its Heikin Ashi scalp signal while ADX agrees. Direction alternates. Exits: +1.1% take-profit, 8.2% (long) / 7.8% (short) stop, and an opposite signal flips the position.
```bash
python main.py signals  --strategy S4 --csv datasets/XBTUSD_15.csv --resample 30min
python main.py backtest --strategy S4 --csv datasets/XBTUSD_15.csv --resample 30min --from 2017-01-01 --short --fee 0.001
```
On real Kraken BTC 30m data from 2017 onward (no leverage, 0.1% fee): 1,330 trades with a **71% win rate, but -93% total**. The average trade is +0.02% before fees and -0.18% after; entries hit their target first only 0.9 points more often than random (±1.3).

Real Kraken 1h data for other coins: `datasets/ETHUSD_60.csv` (2015-2026), `datasets/SOLUSD_60.csv` (2021-2026), `datasets/XRPUSD_60.csv` (2017-2026); build any slower timeframe with `--resample`.

Real Kraken BTC/USD data for these: `datasets/XBTUSD_15.csv` (15m, 387,797 candles), `datasets/XBTUSD_60.csv` (1h) and `datasets/XBTUSD_240.csv` (4h), all from 2013-10 to 2026-09.

## S1 - Top-down confluence (`mcbot/strategies.py`)
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

**Exits (from the video, the `backtest` default `--exits strategy`):**
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

## Live paper trading + dashboard (`livepaper/`)

Runs on your own computer: trades the tested strategies on live prices in a
paper account, records every fill, and shows it all in a local dashboard. The
live engine uses the same strategy code and exit rules as the backtester; a
test replays history through it and checks it makes the exact same trades.

**Setup (once)**

    git clone <this repo> && cd <repo>
    python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
    pip install -r requirements.txt -r livepaper/requirements.txt
    cp livepaper/live.example.toml live.toml                # Windows: copy livepaper\live.example.toml live.toml

Edit `live.toml`: which strategy, symbol, timeframe and capital per run.

**Run it (two terminals)**

    python -m livepaper run          # the trader: checks prices every 30 s, acts on each closed candle
    python -m livepaper dashboard    # opens the dashboard in your browser (http://localhost:8501)

`python -m livepaper check` tests the data feeds (and Webull keys) without placing orders.

**Dashboard tabs**

- **📊 Dashboard** - every open trade with its current price, the asset's 24h move and move since
  your entry, and *your trade's* P&L in % and $ (green up / red down); account totals; all runs;
  and a collapsible detail panel per run (equity curve, closed trades, latest checklist).
- **🧪 Strategy Lab** - pick any strategy (S1, S3 GRIND original/x3, S4 Witcher original/wide,
  S8), any crypto (top 20 coins) or stock (any Yahoo ticker; `pip install yfinance`), a timeframe
  and a period, adjust the exits if you like, and see the backtest against buy & hold (chart,
  year by year, every trade). Tick **Paper trade this setup** and the trader starts it on its next
  price check; untick it (here or under *Manage paper-trading runs*) and it closes the open
  position and stops. Runs live in the database, so no restart is needed.
- **📜 Activity** - every event and fill.

Stocks in the built-in paper broker use Yahoo prices (about 15 minutes delayed); use the Webull
feed for real-time.

**Brokers**

- `broker = "sim"` (default): built-in paper broker, fills at the live market price. Works
  right away; crypto prices come from Kraken's public API (no account needed).
- `broker = "webull"`: your **Webull paper (sandbox) account** through Webull's official SDK.
  1. In Webull's Developer Center click *Using OpenAPI service in Paper Trading*, then
     *API Management -> My Application* and apply for the Sandbox Trading API.
  2. Put the keys in environment variables (never in `live.toml`):
     `WEBULL_APP_KEY`, `WEBULL_APP_SECRET`, optionally `WEBULL_ACCOUNT_ID`.
  3. Run `python -m livepaper check`, then `python -m livepaper run`.
  Stocks use `feed = "webull"`; stock orders are whole shares, regular hours only.

**Shared with Claude:** with `sync_git = true` the trade record (trades, fills, equity,
signals, events as CSV) is pushed hourly to the `paper-trades` branch, so a Claude session
can compare live results with the backtests. It uses a separate worktree (`.paper-trades/`),
so your working branch is never touched. `python -m livepaper sync` pushes immediately.

Spot crypto can't be shorted, so crypto runs are long-only (the backtests in the reports
include shorts for GRIND and the Witcher; expect different numbers).

## Stream study agent (`streamlearn/`)

Studies livestreams on your own computer: pulls out every trade call (asset, direction,
entry, stop, targets, timeframe, conditions, his exact words and timestamp), every trading
rule he states and every price level he marks, then **checks each call against real Kraken
prices** from the moment he said it, and builds a **playbook** of his recurring rules (with
how many streams repeat each one) plus a **scorecard** (win rate when he gives a stop and
target, "right direction after 7 days" for every call, by asset / kind / direction).

**Setup (once)**

    pip install -r requirements.txt -r streamlearn/requirements.txt
    cp streamlearn/streamlearn.example.toml streamlearn.toml     # set your timezone
    # Local model: install Ollama (https://ollama.com), then:  ollama pull qwen2.5:14b
    # (or set backend = "claude" in streamlearn.toml to use Anthropic's API instead)

**Each stream** - save the transcript (YouTube: "...more" -> "Show transcript", select all,
paste into a .txt) or the audio/video you're allowed to keep, then either:

    python -m streamlearn add "stream.txt" --start "2026-10-28 20:00" --title "BTC live"
    python -m streamlearn watch streams_inbox     # or drop files named "2026-10-28 2000 BTC live.txt" here

Then `python -m streamlearn dashboard` (scorecard, calls with outcomes, playbook, searchable
transcripts), `python -m streamlearn report` (markdown playbook + scorecard), and
`python -m streamlearn sync` to push everything to the `paper-trades` branch so a Claude
session can turn his most repeated rules into strategies to backtest.

Keep it personal: transcripts of paid or members-only streams are for your own study; don't
redistribute them.

## Important caveats
- The paper/live bot uses a single stop and target. Leverage, partial take-profits and trailing are backtest-only for now.
- Bot stops are checked by the bot itself, not placed on the exchange, so they only protect you while it runs.
- This is educational code, not financial advice. Paper trade before risking real money.

## Tests
```bash
python -m pytest -q
```
