"""OHLCV data sources: exchange (via ccxt), CSV, or synthetic for demos."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]


def _to_frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts"] + COLUMNS)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop_duplicates("ts").set_index("ts").astype(float)


def make_exchange(name: str, api_key: str | None = None, secret: str | None = None,
                  sandbox: bool = False):
    import ccxt
    ex = getattr(ccxt, name)({"apiKey": api_key, "secret": secret, "enableRateLimit": True})
    if sandbox:
        ex.set_sandbox_mode(True)
    return ex


def fetch_ohlcv(exchange, symbol: str, timeframe: str = "1h", limit: int = 1000) -> pd.DataFrame:
    """Fetch the most recent `limit` candles, paging backwards as needed."""
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    since = exchange.milliseconds() - limit * tf_ms
    rows: list = []
    while len(rows) < limit:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=min(1000, limit - len(rows)))
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + tf_ms
        if since >= exchange.milliseconds():
            break
        time.sleep(exchange.rateLimit / 1000)
    return _to_frame(rows)


def load_csv(path: str) -> pd.DataFrame:
    """CSV with a timestamp column (first column) and open/high/low/close/volume.

    The timestamp can be unix seconds (TradingView exports), unix milliseconds,
    or a date string. Volume is optional. Headerless Kraken OHLCVT files
    (time, open, high, low, close, volume, trades) are detected automatically.
    Close-only files (e.g. Date,Price) get candles built from consecutive closes.
    """
    df = pd.read_csv(path)
    first = str(df.columns[0]).strip()
    if first.replace(".", "", 1).isdigit():
        # No header row: Kraken OHLCVT export (time, open, high, low, close, volume, trades).
        df = pd.read_csv(path, header=None)
        df.columns = ["time", "open", "high", "low", "close", "volume", "trades"][:df.shape[1]]
    df.columns = [str(c).strip().lower() for c in df.columns]
    ts = df[df.columns[0]]
    if pd.api.types.is_numeric_dtype(ts):
        df.index = pd.to_datetime(ts, unit="s" if ts.max() < 1e11 else "ms", utc=True)
    else:
        df.index = pd.to_datetime(ts, utc=True)
    if "close" not in df.columns and "price" in df.columns:
        df = df.rename(columns={"price": "close"})
    if "open" not in df.columns:
        # Close-only data: build candles from consecutive closes (no wicks).
        df["open"] = df["close"].shift().fillna(df["close"])
        df["high"] = df[["open", "close"]].max(axis=1)
        df["low"] = df[["open", "close"]].min(axis=1)
    if "volume" not in df.columns:
        df["volume"] = 1.0
    return df[COLUMNS].astype(float)


def relabel(df: pd.DataFrame, timeframe: str = "1h") -> pd.DataFrame:
    """Give candles consecutive timestamps `timeframe` apart, keeping their order.

    Lets you treat e.g. daily stock candles as if they were 1h candles: the
    indicators see the same bars, but session VWAP, the higher timeframe and
    bars-per-year all behave as on a 1h chart.
    """
    out = df.copy()
    out.attrs["real_dates"] = df.index  # keep the true dates for display
    out.index = pd.date_range("2000-01-01", periods=len(df), freq=pd.Timedelta(timeframe), tz="UTC")
    return out


def synthetic(n: int = 3000, seed: int = 7, start_price: float = 30_000.0,
              timeframe: str = "1h") -> pd.DataFrame:
    """Random-walk candles with drifting trend regimes. For demos and tests only."""
    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.normal(0, 0.0015, n // 200 + 1), 200)[:n]
    rets = regime + rng.normal(0, 0.008, n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    wiggle = np.abs(rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + wiggle)
    low = np.minimum(open_, close) * (1 - wiggle)
    vol = rng.lognormal(3, 0.5, n)
    idx = pd.date_range("2024-01-01", periods=n, freq=pd.Timedelta(timeframe), tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol},
                        index=idx)
