"""Show where price and money flow converge or diverge.

  python examples/money_flow_divergence.py                     # demo data
  python examples/money_flow_divergence.py binance BTC/USDT 4h # real data
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcbot import data
from mcbot.indicators import market_cipher

if len(sys.argv) == 4:
    df = data.fetch_ohlcv(data.make_exchange(sys.argv[1]), sys.argv[2], sys.argv[3], limit=500)
else:
    df = data.synthetic(n=500)

mc = market_cipher(df)
labels = {
    "bull_mf_div": "BULLISH DIVERGENCE  price lower low,   money flow higher low",
    "bear_mf_div": "BEARISH DIVERGENCE  price higher high, money flow lower high",
    "bull_mf_conv": "BULLISH CONVERGENCE price + money flow both higher highs",
    "bear_mf_conv": "BEARISH CONVERGENCE price + money flow both lower lows",
}

# Print only the first bar of each new event so the list stays readable.
for col, text in labels.items():
    starts = mc[col] & ~mc[col].shift(fill_value=False)
    print(f"\n{text}")
    for ts, row in mc[starts].tail(5).iterrows():
        print(f"  {ts:%Y-%m-%d %H:%M}  close={row.close:>10.2f}  money_flow={row.money_flow:>7.2f}")

last = mc.iloc[-1]
active = [text.split("  ")[0] for col, text in labels.items() if last[col]]
print(f"\nNow: {', '.join(active) or 'nothing'}  (money flow {last.money_flow:.2f})")
