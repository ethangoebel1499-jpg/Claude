"""Live candles. Each feed returns CLOSED candles (the one still forming is dropped)
indexed by candle start time (UTC), plus the latest price."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import requests

from mcbot.data import resample

TF_MINUTES = {"1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720, "1D": 1440, "2D": 2880}


def closed_only(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    now = pd.Timestamp(datetime.now(timezone.utc))
    return df[df.index + pd.Timedelta(minutes=minutes) <= now]


def to_timeframe(df: pd.DataFrame, base: int, minutes: int) -> pd.DataFrame:
    return df if base == minutes else resample(df, f"{minutes}min")


class KrakenFeed:
    """Kraken's public market data: no account or key needed."""
    URL = "https://api.kraken.com/0/public"
    INTERVALS = (1440, 240, 60, 30, 15, 5, 1)

    def __init__(self, session: requests.Session | None = None):
        self.http = session or requests.Session()

    def _get(self, path: str, **params):
        r = self.http.get(f"{self.URL}/{path}", params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        if js.get("error"):
            raise RuntimeError(f"Kraken: {js['error']}")
        return next(v for k, v in js["result"].items() if k != "last")

    def bars(self, symbol: str, timeframe: str) -> pd.DataFrame:
        minutes = TF_MINUTES[timeframe]
        base = next(i for i in self.INTERVALS if minutes % i == 0)
        rows = self._get("OHLC", pair=symbol, interval=base)
        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
        df.index = pd.to_datetime(df.pop("time").astype(int), unit="s", utc=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        df = closed_only(df, base)                      # drop Kraken's forming candle first
        return closed_only(to_timeframe(df, base, minutes), minutes)

    def price(self, symbol: str) -> float:
        return float(self._get("Ticker", pair=symbol)["c"][0])


class WebullFeed:
    """Webull market data through the official SDK (pip install webull-openapi-python-sdk).
    The response layout is parsed defensively; `python -m livepaper check` prints a raw sample."""
    TIMESPANS = {60: "M60", 120: "M120", 240: "M240", 1440: "D"}

    def __init__(self, api_client):
        from webull.data.data_client import DataClient
        self.dc = DataClient(api_client)
        self._last_call = 0.0

    def _throttle(self):                      # crypto bars: 1 request / second / app key
        wait = 1.05 - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def raw_bars(self, symbol: str, asset: str, timespan: str, count: int = 1200):
        self._throttle()
        if asset == "crypto":
            res = self.dc.crypto_market_data.get_crypto_history_bar(symbol, "US_CRYPTO", timespan, count=str(count))
        else:
            res = self.dc.market_data.get_history_bar(symbol, "US_STOCK", timespan, count=str(count))
        if res.status_code != 200:
            raise RuntimeError(f"Webull bars {symbol}: HTTP {res.status_code} {res.text[:300]}")
        return res.json()

    def bars(self, symbol: str, timeframe: str, asset: str = "stock") -> pd.DataFrame:
        minutes = TF_MINUTES[timeframe]
        base = max(b for b in self.TIMESPANS if minutes % b == 0)
        df = parse_bars(self.raw_bars(symbol, asset, self.TIMESPANS[base]))
        df = closed_only(df, base)
        return closed_only(to_timeframe(df, base, minutes), minutes)

    def price(self, symbol: str, asset: str = "stock") -> float:
        self._throttle()
        if asset == "crypto":
            res = self.dc.crypto_market_data.get_crypto_snapshot(symbol)
        else:
            res = self.dc.market_data.get_snapshot(symbol, "US_STOCK")
        if res.status_code != 200:
            raise RuntimeError(f"Webull snapshot {symbol}: HTTP {res.status_code} {res.text[:300]}")
        return parse_price(res.json())


def _records(js):
    """Find the list of bar / quote records in a Webull response."""
    if isinstance(js, list):
        if js and isinstance(js[0], dict) and not any(k in js[0] for k in ("open", "close", "price")):
            for v in js[0].values():           # e.g. [{"symbol": ..., "result": [bars]}]
                if isinstance(v, list):
                    return v
        return js
    if isinstance(js, dict):
        for key in ("data", "result", "bars", "list", "items"):
            if key in js:
                return _records(js[key])
    return []


def parse_bars(js) -> pd.DataFrame:
    recs = _records(js)
    if not recs:
        raise RuntimeError(f"no bars in Webull response: {str(js)[:300]}")
    df = pd.DataFrame(recs)
    tcol = next(c for c in ("time", "timestamp", "t", "start_time", "date") if c in df.columns)
    t = df[tcol]
    if pd.api.types.is_numeric_dtype(t) or str(t.iloc[0]).isdigit():
        t = pd.to_numeric(t)
        df.index = pd.to_datetime(t, unit="ms" if t.max() > 1e11 else "s", utc=True)
    else:
        df.index = pd.to_datetime(t, utc=True)
    out = df[["open", "high", "low", "close"]].astype(float)
    out["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0) if "volume" in df else 0.0
    return out.sort_index()[~out.index.duplicated()]


def parse_price(js) -> float:
    recs = _records(js) or [js]
    r = recs[0]
    for key in ("price", "last_price", "close", "last", "trade_price"):
        if key in r and r[key] not in (None, ""):
            return float(r[key])
    raise RuntimeError(f"no price in Webull response: {str(js)[:300]}")


class YahooFeed:
    """US stocks from Yahoo Finance (pip install yfinance). Quotes are delayed ~15 minutes
    for most exchanges - fine for 1h+ candles; use Webull for real-time."""

    def bars(self, symbol: str, timeframe: str) -> pd.DataFrame:
        from .history import stock
        return closed_only(stock(symbol, timeframe), min(TF_MINUTES[timeframe], 60 * 24))

    def price(self, symbol: str) -> float:
        import yfinance as yf
        return float(yf.Ticker(symbol).fast_info["last_price"])
