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

## Strategies (`mcbot/strategies.py`)

### Trader-inspired strategies
Neither trader publishes exact mechanical rules. These are rules a bot can follow, built from public material: Market Cipher B guides for CryptoFace (Market Cipher's creator), and course topics for Jayson Casper (swing failure patterns, VWAP, support to resistance). They're approximations, not their actual systems.

| Strategy | Style | Entry | Exit |
|---|---|---|---|
| `cryptoface_swing` | Long-term | Anchor wave (≤ -60), then a higher-low trigger wave crossing up, with higher-timeframe money flow > 0 and a bullish ribbon | Bearish anchor/trigger, a red dot while the HTF wave falls, or HTF money flow < 0 |
| `cryptoface_scalp` | Scalp | Green dot in the last 3 bars, plus the Cipher B VWAP wave crossing above 0 with money flow rising, in the direction of the HTF wave | Next bearish WaveTrend cross |
| `casper_swing` | Long-term | Swing failure pattern at support (50-bar low) with money flow turning up | Price reaches resistance (50-bar high) or a bearish SFP |
| `casper_scalp` | Scalp | Swing failure pattern (wick below the 20-bar low that closes back above it) while price is under the session VWAP | Price gets back to VWAP |

The higher timeframe is 4× the chart timeframe (1h chart → 4h bias). It only uses completed candles, so there's no lookahead. Run scalps on 5m–15m with tight stops (e.g. `--timeframe 15m --stop 0.008 --target 0.012`) and swings on 4h–1d with wider stops (e.g. `--timeframe 4h --stop 0.06 --target 0.15`). Fees eat scalping strategies, so check `--fee` against your exchange.

### Basic strategies
- `dots`: buy green dots, sell red dots.
- `dots_confirmed` (default): a green dot only counts if money flow is rising, price is above EMA200, and the ribbon is bullish. Exits on a red dot or when money flow flips negative. Shorts are the mirror image.
- `divergence`: a dot that comes within 5 bars of a matching WT divergence.
- `money_flow_div`: buys a bullish money flow divergence once money flow turns up. Exits when price and money flow converge down, or on a bearish divergence.

### Money flow divergence vs convergence
| Signal | Price | Money flow | Meaning |
|---|---|---|---|
| Bullish divergence | lower low | higher low | Sellers are running out, so a bounce is likely |
| Bearish divergence | higher high | lower high | Buyers are running out, so a drop is likely |
| Bullish convergence | higher high | higher high | The uptrend is confirmed |
| Bearish convergence | lower low | lower low | The downtrend is confirmed |

```bash
python examples/money_flow_divergence.py                      # demo data
python examples/money_flow_divergence.py binance BTC/USDT 4h  # real data
```

To add your own strategy, write a function that returns `long_entry/long_exit/short_entry/short_exit` columns and register it in `STRATEGIES`.

## Usage
```bash
pip install -r requirements.txt

# Backtest (fills happen at the next bar's open; 3% stop, 6% target, 0.1% fee by default)
python main.py backtest --exchange binance --symbol BTC/USDT --timeframe 4h --limit 3000 --compare
python main.py backtest --csv my_candles.csv --strategy divergence --short
python main.py backtest --synthetic --compare          # offline demo on random data

# See the current indicator values and signal
python main.py signals --exchange binance --symbol ETH/USDT --timeframe 1h

# Paper trade (simulated account saved to paper_state.json)
python main.py paper --exchange binance --symbol BTC/USDT --timeframe 1h

# Live trade: spot, long-only, needs both the --live flag and API keys
export MCBOT_API_KEY=... MCBOT_API_SECRET=...
python main.py live --exchange binance --symbol BTC/USDT --timeframe 1h --live --sandbox
```

## Important caveats
- The stop-loss and take-profit are checked by the bot itself, not placed as orders on the exchange. They only protect you while the bot is running.
- On real data, the plain `dots` strategy usually loses money once fees are included. Backtest across several symbols and timeframes, then paper trade for a while before you risk real money.
- This is educational code, not financial advice.

## Tests
```bash
python -m pytest -q
```
