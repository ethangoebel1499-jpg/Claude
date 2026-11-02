"""Price history for Strategy Lab backtests.

Crypto, 1h and up: the repo's Kraken files (datasets/, 1h back to 2017 for the top 20 coins) topped
up with Kraken's public API.
Crypto, minute charts (1m-30m): Coinbase's public candles, paged back weeks and saved in
runtime/candles so later runs only fetch what's new; Kraken's last 720 candles if Coinbase can't
be reached or doesn't list the coin.
Stocks: Yahoo Finance through yfinance - 1m for the last 7 days, 5m-30m for the last 60 days,
hourly for ~2 years, daily for decades.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from mcbot.data import load_csv, resample

ROOT = Path(__file__).resolve().parent.parent
TF_MINUTES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "6h": 360,
              "8h": 480, "12h": 720, "1D": 1440, "2D": 2880}
INTRADAY_DAYS = {1: 7, 3: 14, 5: 30, 15: 60, 30: 90}      # how far back minute charts reach by default
CANDLES = ROOT / "runtime" / "candles"

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


def _coinbase(product: str, granularity: int, days: float) -> pd.DataFrame:
    """Coinbase's public candles (300 per request, newest first), cached and topped up."""
    import time as _time
    CANDLES.mkdir(parents=True, exist_ok=True)
    cache = CANDLES / f"coinbase_{product}_{granularity}.csv"
    have = pd.read_csv(cache, index_col=0, parse_dates=True) if cache.exists() else None
    end = pd.Timestamp.now(tz="UTC").floor(f"{granularity}s")
    start = end - pd.Timedelta(days=days)
    gaps = [(start, end)]
    if have is not None and len(have):                     # only what isn't cached yet: older and newer parts
        have.index = pd.to_datetime(have.index, utc=True)
        gaps = [(start, have.index.min())] if start < have.index.min() else []
        gaps.append((have.index.max(), end))
    frames = []
    step = pd.Timedelta(seconds=granularity * 300)
    jobs = [(a, b) for a, b in gaps if b > a]
    while jobs:
        start, t1 = jobs.pop()
        t0 = max(start, t1 - step)
        r = requests.get(f"https://api.exchange.coinbase.com/products/{product}/candles", timeout=30,
                         params={"granularity": granularity, "start": t0.isoformat(), "end": t1.isoformat()},
                         headers={"User-Agent": "paper-trader"})
        if r.status_code == 429:
            _time.sleep(1)
            jobs.append((start, t1))
            continue
        r.raise_for_status()
        rows = r.json()
        if rows:
            frames.append(pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"]))
        if t0 > start:
            jobs.append((start, t0))
        _time.sleep(0.12)                                  # stay under the public rate limit
    new = pd.concat(frames) if frames else pd.DataFrame(columns=["time", "low", "high", "open", "close", "volume"])
    new.index = pd.to_datetime(new.pop("time").astype(int), unit="s", utc=True)
    new = new[["open", "high", "low", "close", "volume"]].astype(float)
    df = pd.concat([have, new]) if have is not None else new
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df.index < end]                                # drop the forming candle
    df.to_csv(cache)
    return df[df.index >= end - pd.Timedelta(days=days)]


def crypto_intraday(pair: str, minutes: int, days: float | None = None) -> pd.DataFrame:
    days = days or INTRADAY_DAYS.get(minutes, 30)
    coin = next((c for c, p in CRYPTO.items() if p == pair), pair.replace("USD", ""))
    gran = 60 if minutes in (1, 3) else 300 if minutes == 5 else 900
    try:
        df = _coinbase(f"{coin}-USD", gran, days)
        if df.empty:
            raise RuntimeError("no candles")
    except Exception:
        df = _kraken(pair, 1 if minutes in (1, 3) else minutes)         # last 720 candles
    base = int(pd.Series(df.index).diff().median().total_seconds() // 60) if len(df) > 1 else minutes
    return df if minutes == base else resample(df, f"{minutes}min").dropna()


def crypto(pair: str, timeframe: str) -> pd.DataFrame:
    minutes = TF_MINUTES[timeframe]
    if minutes < 60:
        return crypto_intraday(pair, minutes)
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
    if minutes < 60:                                     # Yahoo keeps 1m for 7 days, 5m-30m for 60 days
        interval = {1: "1m", 3: "1m", 5: "5m", 15: "15m", 30: "30m"}[minutes]
        raw = yf.Ticker(symbol).history(period="7d" if interval == "1m" else "60d", interval=interval, auto_adjust=True)
        if raw.empty:
            raise RuntimeError(f"no Yahoo data for {symbol}")
        df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
        return resample(df, "3min").dropna() if minutes == 3 else df
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
