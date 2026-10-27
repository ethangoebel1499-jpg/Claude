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
| EMA ribbon | Cipher A EMAs 5/11/15/18/21/24/28/34. Bullish when EMA5 > EMA34 |

## Strategies (`mcbot/strategies.py`)
- `dots`: buy green dots, sell red dots.
- `dots_confirmed` (default): a green dot only counts if money flow is rising, price is above EMA200, and the ribbon is bullish. Exits on a red dot or when money flow flips negative. Shorts are the mirror image.
- `divergence`: a dot that comes within 5 bars of a matching WT divergence.

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
