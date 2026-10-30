"""Price history for Strategy Lab backtests.

Crypto: the repo's Kraken files (datasets/, 1h back to 2017 for the top 20 coins) topped up
with Kraken's public API. Stocks: Yahoo Finance through yfinance (pip install yfinance;
hourly for the last ~2 years, daily for decades).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from mcbot.data import load_csv, resample

ROOT = Path(__file__).resolve().parent.parent
TF_MINUTES = {"1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720, "1D": 1440, "2D": 2880}

# display name -> Kraken pair (the repo has 1h history for all of these)
CRYPTO = {"BTC": "XBTUSD", "ETH": "ETHUSD", "SOL": "SOLUSD", "XRP": "XRPUSD", "BNB": "BNBUSD", "DOGE": "XDGUSD",
          "TRX": "TRXUSD", "ADA": "ADAUSD", "LINK": "LINKUSD", "AVAX": "AVAXUSD", "XLM": "XLMUSD", "SUI": "SUIUSD",
          "BCH": "BCHUSD", "LTC": "LTCUSD", "DOT": "DOTUSD", "HBAR": "HBARUSD", "SHIB": "SHIBUSD", "TON": "TONUSD",
          "UNI": "UNIUSD", "XMR": "XMRUSD"}
STOCKS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "BRK-B", "JPM", "LLY",
          "V", "WMT", "XOM", "UNH", "MA", "COST", "NFLX", "AMD", "PLTR"]


def _local(pair: str) -> pd.DataFrame | None:
    for p in (ROOT / "datasets" / f"{pair}_60.csv", ROOT / "datasets" / "crypto" / f"{pair}_60.csv"):
        if p.exists():
            return load_csv(p)[["open", "high", "low", "close", "volume"]]
    return None


def _kraken(pair: str, interval: int) -> pd.DataFrame:
    r = requests.get("https://api.kraken.com/0/public/OHLC", params={"pair": pair, "interval": interval}, timeout=30)
    js = r.json()
    if js.get("error"):
        raise RuntimeError(f"Kraken {pair}: {js['error']}")
    rows = next(v for k, v in js["result"].items() if k != "last")
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    df.index = pd.to_datetime(df.pop("time").astype(int), unit="s", utc=True)
    return df[["open", "high", "low", "close", "volume"]].astype(float).iloc[:-1]   # drop the forming candle


def crypto(pair: str, timeframe: str) -> pd.DataFrame:
    minutes = TF_MINUTES[timeframe]
    frames = []
    local = _local(pair)
    if local is not None:
        frames.append(local)
    try:
        frames.append(_kraken(pair, 60))                    # last ~30 days of 1h
    except Exception:
        if not frames:
            # no local history: Kraken's longest-reaching interval that fits the timeframe
            frames.append(_kraken(pair, 1440 if minutes % 1440 == 0 else 240 if minutes % 240 == 0 else 60))
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    base = int(pd.Series(df.index).diff().median().total_seconds() // 60)
    return df if minutes == base else resample(df, f"{minutes}min")


def stock(symbol: str, timeframe: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("Stock history needs yfinance: pip install yfinance")
    minutes = TF_MINUTES[timeframe]
    intraday = minutes < 1440
    raw = yf.Ticker(symbol).history(period="730d" if intraday else "max", interval="1h" if intraday else "1d",
                                    auto_adjust=True)
    if raw.empty:
        raise RuntimeError(f"no Yahoo data for {symbol}")
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    if minutes in (60, 1440):
        return df
    # stocks trade ~6.5 h a day: group whole bars rather than clock time
    n = minutes // 60 if intraday else minutes // 1440
    g = pd.Series(range(len(df)), index=df.index) // n
    out = df.groupby(g.values).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    out.index = g.drop_duplicates().index            # each group's first bar time
    return out


def history(symbol: str, asset: str, timeframe: str) -> pd.DataFrame:
    return crypto(CRYPTO.get(symbol, symbol), timeframe) if asset == "crypto" else stock(symbol, timeframe)
