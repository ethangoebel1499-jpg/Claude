# Market Cipher trading bot

A simple bot that trades with Market Cipher A/B style signals. You can backtest it, paper trade it, or (if you opt in) trade live on any exchange [ccxt](https://github.com/ccxt/ccxt) supports.

> Market Cipher is a closed-source, paid TradingView indicator. This project uses the open-source **VuManChu Cipher A/B** approximation, which has the same parts and default settings. Its values will be close to the real thing, but not identical.

## Indicators (`mcbot/indicators.py`)
| Component | What it is |
|---|---|
| WaveTrend `wt1`/`wt2` | Main Cipher B wave (channel 9, average 12, MA 3) |
| Green / red dots | WT bullish cross with wt2 ≤ -53 / bearish cross with wt2 ≥ 53 |
| Money flow | RSI+MFI area (period 60, multiplier 150). Above 0 means buying pressure |
| RSI, Stoch RSI | 14-period |
| Divergences | Fractal-based regular WT divergences, flagged on the bar they're confirmed (no lookahead) |
| Money flow divergence / convergence | Compares the last 20 bars of price and money flow with the 20 bars before them (see below) |
| EMA ribbon | Cipher A EMAs 5/11/15/18/21/24/28/34. Bullish when EMA5 > EMA34 |

## Better entries (entry quality)
Your exits (target 1 at half the stop distance) only pay off if an entry hits target 1 before the stop about **75–80%** of the time. A random entry manages about **66%**. Use `entries` to score this for any strategy, with no exits involved:

```bash
python main.py entries --exchange binance --symbol BTC/USDT --timeframe 1h --limit 5000 --compare   # 1.5% target vs 3% stop
python main.py entries --csv tradingview_export.csv --compare --mode atr --target-dist 1 --stop-dist 2
```

On real **EUR/USD 1h data (2017)**, using a target of 1 ATR and a stop of 2 ATR:

| Entry | Hit target first | Signals |
|---|---|---|
| Random | 66% | — |
| Plain green dot (`dots`) | 72% | 292 |
| `second_dot_confirmed`: second dot higher than the first (3–30 bars apart), then a green candle confirms | **82%** | 45 |
| `dot_htf_confirmed`: green dot while the 4h wave is below -40, then a green candle confirms | **84%** | 61 |

Both held up in each half of the year, and with a 2 ATR target and 4 ATR stop.

**But they didn't hold up on stocks.** On real daily data for SPY (1993–2021), GOOGL (2009–2018), GOOG (2004–2009) and 002032 (2004–2016), treated as 1h candles with `--relabel 1h`, about 25,000 bars in total:

| Entry | Hit 1 ATR before 2 ATR stop | Signals |
|---|---|---|
| Random | 66.1% | — |
| `dots` | 64.8% | 732 |
| `second_dot_confirmed` | 60.0% | 140 |
| `dot_htf_confirmed` | 59.4% | 224 |
| best (`double_dot_only`) | 71.2% (±5.8) | 66 |

No entry clearly beat random entries on stocks, and the two EUR/USD winners came out *below* random. The EUR/USD result was most likely luck or specific to that market. Treat every entry as unproven until `entries` shows an edge on the market you actually trade.

```bash
# Daily candles from any CSV, treated as 1h candles
python main.py entries --csv SPY_daily.csv --relabel 1h --compare --mode atr --target-dist 1 --stop-dist 2
```

## Higher-timeframe dot confirmed by a lower-timeframe dot
`mtf_dots`: a green dot on the higher timeframe (`--htf 4` = 4 chart candles, so 1h → 4h) arms the setup once that HTF candle has **closed**. The bot then waits up to 3 HTF candles for a green dot on the chart timeframe and enters on it. A green dot that came *before* the HTF dot doesn't count, and an HTF red dot cancels the setup. It exits on a chart-timeframe red dot (or your stops/targets). Shorts mirror it. The higher timeframe is built from real candles and then converted to Heikin Ashi on its own.

```bash
python main.py backtest --exchange binance --symbol BTC/USDT --timeframe 1h --htf 4 --strategy mtf_dots
```

## Double green dot ("uneven butt cheeks")
- `double_dot`: take a green dot if momentum lines up (money flow rising). If it doesn't, skip it and wait for a **second green dot within 30 bars that sits higher on the wave** (a higher low: the smaller, "uneven" cheek), with money flow rising this time. Exit on a red dot, or when money flow turns negative. Shorts mirror it with a second, lower red dot.
- `double_dot_only`: only takes the second-dot setup and never trades single dots.

## Heikin Ashi candles
All indicators and signals are calculated on **Heikin Ashi** candles by default. These are smoothed candles: close = (O+H+L+C)/4, and open = the average of the previous HA open and close. Backtest and live trades still fill at **real** prices, because Heikin Ashi prices don't exist on the exchange and filling at them would give fake results. Add `--no-heikin-ashi` to use regular candles.

`heikin_ashi_trend`: a green dot in the last 5 bars, confirmed by a strong green HA candle (no lower wick) with money flow rising. It holds until the first red HA candle.

## Strategies (`mcbot/strategies.py`)

### Trader-inspired strategies
Neither trader publishes exact mechanical rules. These are rules a bot can follow, built from public material: Market Cipher B guides for CryptoFace (Market Cipher's creator), and course topics for Jayson Casper (swing failure patterns, VWAP, support to resistance). They're approximations, not their actual systems.

| Strategy | Style | Entry | Exit |
|---|---|---|---|
| `cryptoface_swing` | Long-term | Anchor wave (≤ -60), then a higher-low trigger wave crossing up, with higher-timeframe money flow > 0 and a bullish ribbon | Bearish anchor/trigger, a red dot while the HTF wave falls, or HTF money flow < 0 |
| `cryptoface_scalp` | Scalp | Green dot in the last 3 bars, plus the Cipher B VWAP wave crossing above 0 with money flow rising, in the direction of the HTF wave | Next bearish WaveTrend cross |
| `casper_swing` | Long-term | Swing failure pattern at support (50-bar low) with money flow turning up | Price reaches resistance (50-bar high) or a bearish SFP |
| `casper_scalp` | Scalp | Swing failure pattern (wick below the 20-bar low that closes back above it) while price is under the session VWAP | Price gets back to VWAP |

The higher timeframe is 4× the chart timeframe (1h chart → 4h bias). It only uses completed candles, so there's no lookahead. Everything runs on the **1h chart** (the default) for consistency. "Scalp" vs "swing" here means how quickly a strategy exits, not a different chart timeframe.

### Basic strategies
- `dots`: buy green dots, sell red dots.
- `dots_confirmed` (default): a green dot only counts if money flow is rising, price is above EMA200, and the ribbon is bullish. Exits on a red dot or when money flow flips negative. Shorts are the mirror image.
- `divergence`: a dot that comes within 5 bars of a matching WT divergence.
- `money_flow_div`: buys a bullish money flow divergence once money flow turns up. Exits when price and money flow converge down, or on a bearish divergence.

### Money flow filters (any strategy)
- Money flow conditions in every strategy now mean money flow is **changing direction**: it's rising on this bar after falling within the last 3 bars (curling up), or the mirror for shorts.
- `--mf-turn`: only enter if money flow turned in the trade's direction within the last 3 bars.
- `--convergence`: only enter after a matching money flow divergence or convergence in the last 20 bars, and not while price and money flow converge against the trade.

### Money flow divergence vs convergence
| Signal | Price | Money flow | Meaning |
|---|---|---|---|
| Bullish divergence | lower low | higher low | Sellers are running out, so a bounce is likely |
| Bearish divergence | higher high | lower high | Buyers are running out, so a drop is likely |
| Bullish convergence | higher high | higher high | The uptrend is confirmed |
| Bearish convergence | lower low | lower low | The downtrend is confirmed |

```bash
python examples/money_flow_divergence.py                      # demo data
python examples/money_flow_divergence.py binance BTC/USDT 1h  # real data
```

To add your own strategy, write a function that returns `long_entry/long_exit/short_entry/short_exit` columns and register it in `STRATEGIES`.

## Usage
```bash
pip install -r requirements.txt

# Backtest (fills happen at the next bar's open; 3% stop, 6% target, 0.1% fee by default)
python main.py backtest --exchange binance --symbol BTC/USDT --timeframe 1h --limit 3000 --compare
python main.py backtest --csv my_candles.csv --strategy divergence --short
python main.py backtest --synthetic --compare          # offline demo on random data
python main.py backtest --synthetic --compare --no-heikin-ashi  # same, regular candles

# See the current indicator values and signal
python main.py signals --exchange binance --symbol ETH/USDT --timeframe 1h

# Paper trade (simulated account saved to paper_state.json)
python main.py paper --exchange binance --symbol BTC/USDT --timeframe 1h

# Live trade: spot, long-only, needs both the --live flag and API keys
export MCBOT_API_KEY=... MCBOT_API_SECRET=...
python main.py live --exchange binance --symbol BTC/USDT --timeframe 1h --live --sandbox
```

## Leverage and scaled take-profits
```bash
# 5x, 3% stop, close 50% at +1%, 25% at +2%, the rest at +3% (price moves)
python main.py backtest --exchange binance --symbol BTC/USDT --timeframe 1h \
  --strategy cryptoface_swing --short --leverage 5 --stop 0.03 --tp "1.5:50,2.5:25,ride:25" --trail --fee 0.0005
```
- Stop and target levels are **price** moves. At 5x, a 3% stop costs 15% of the account, plus fees.
- Fees are charged on the leveraged notional. The whole account is used as margin, and a position is liquidated (-100%) if price moves `1/leverage - 0.5%` against it before the stop fills.
- If a stop and a target are both hit in the same candle, the stop is assumed to fill first (conservative).
- Use `ride` for a runner: `--tp "1.5:50,2.5:25,ride:25"` closes 50% at +1.5% and 25% at +2.5%. The last 25% ignores the normal exit signals and stays open until the strategy's **opposite setup** fires (for a long: the full short entry, including money flow turning down and any filters). If `--short` is on, it flips straight into the short. The breakeven stop still protects the runner.
- The opposite setup now always closes a trade, runner or not.
- Add `--trail` to step the stop up behind each target: breakeven after target 1, then target 1's level after target 2. With `1.5:50,2.5:25,ride:25`, the runner locks in at least +1.5%. Like breakeven, it takes effect from the next candle.
- Add `--breakeven` to move the stop to the entry price after the first target fills. It takes effect from the next candle, because within one candle we can't tell whether the high or the low came first.
- Break-even math for this setup: a full win is about +8.25% of the account and a full stop about -15.5%, so you need to win roughly **65%** of trades just to break even.

## Candle confirmation
- **Entries and signal exits only happen on closed candles.** The backtest fills a signal from candle *i* at candle *i+1*'s open. The bot drops the still-forming candle and acts once per newly closed candle.
- On startup, the bot won't trade off a candle that closed a while ago. It waits for the next candle to close.
- **Stop-loss and take-profit fire intrabar by default**, so they can exit mid-candle. Add `--stops-on-close` to only trigger them on a closed candle as well. The risk: in a fast drop, you exit at the next open, possibly far below your stop.

## Important caveats
- The stop-loss and take-profit are checked by the bot itself, not placed as orders on the exchange. They only protect you while the bot is running.
- On real data, the plain `dots` strategy usually loses money once fees are included. Backtest across several symbols and market periods, then paper trade for a while before you risk real money.
- This is educational code, not financial advice.

## Tests
```bash
python -m pytest -q
```
