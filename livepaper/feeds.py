"""Live candles. Each feed returns CLOSED candles (the one still forming is dropped)
indexed by candle start time (UTC), plus the latest price."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import requests

from mcbot.data import resample

from .history import TF_MINUTES  # noqa: E402  (one list of timeframes, down to 1 minute)


def closed_only(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    now = pd.Timestamp(datetime.now(timezone.utc))
    return df[df.index + pd.Timedelta(minutes=minutes) <= now]


def to_timeframe(df: pd.DataFrame, base: int, minutes: int) -> pd.DataFrame:
    return df if base == minutes else resample(df, f"{minutes}min")


class KrakenFeed:
    """Kraken's public market data: no account or key needed.
    One feed is shared by every run, so: the same request within CACHE_S seconds is answered from memory (ten runs on
    BTC = one download), requests are paced to stay under Kraken's public limit, and when Kraken stops answering
    (time-outs, "too many requests") it is left alone for PAUSE_S seconds instead of every run waiting on it in turn."""
    URL = "https://api.kraken.com/0/public"
    INTERVALS = (1440, 240, 60, 30, 15, 5, 1)
    TIMEOUT, CACHE_S, GAP_S, PAUSE_S = 10, 25, 1.0, 120

    def __init__(self, session: requests.Session | None = None):
        self.http = session or requests.Session()
        self._cache: dict = {}
        self._last_call = 0.0
        self._paused_until = 0.0

    def _get(self, path: str, **params):
        key = (path, tuple(sorted(params.items())))
        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < self.CACHE_S:
            return hit[1]
        if time.time() < self._paused_until:
            raise RuntimeError(f"Kraken isn't answering - paused for {self._paused_until - time.time():.0f}s more")
        wait = self.GAP_S - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()
        try:
            r = self.http.get(f"{self.URL}/{path}", params=params, timeout=self.TIMEOUT)
            r.raise_for_status()
            js = r.json()
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            self._paused_until = time.time() + self.PAUSE_S
            raise RuntimeError(f"Kraken didn't answer ({type(e).__name__}) - trying again in {self.PAUSE_S}s") from None
        if js.get("error"):
            if any("rate limit" in str(x).lower() or "too many" in str(x).lower() for x in js["error"]):
                self._paused_until = time.time() + self.PAUSE_S
            raise RuntimeError(f"Kraken: {js['error']}")
        out = next(v for k, v in js["result"].items() if k != "last")
        self._cache[key] = (time.time(), out)
        return out

    def bars(self, symbol: str, timeframe: str) -> pd.DataFrame:
        try:
            return self._kraken_bars(symbol, timeframe)
        except Exception as e:                          # Kraken not answering: Coinbase's public candles instead
            try:
                return self.backup().bars(symbol, timeframe)
            except Exception as e2:
                raise RuntimeError(f"{e}; Coinbase backup failed too: {e2}") from None

    def price(self, symbol: str) -> float:
        try:
            return float(self._get("Ticker", pair=symbol)["c"][0])
        except Exception as e:
            try:
                return self.backup().price(symbol)
            except Exception as e2:
                raise RuntimeError(f"{e}; Coinbase backup failed too: {e2}") from None

    def backup(self) -> "CoinbaseFeed":
        if not hasattr(self, "_backup"):
            self._backup = CoinbaseFeed()
        return self._backup

    def _kraken_bars(self, symbol: str, timeframe: str) -> pd.DataFrame:
        minutes = TF_MINUTES[timeframe]
        base = next(i for i in self.INTERVALS if minutes % i == 0)
        rows = self._get("OHLC", pair=symbol, interval=base)
        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
        df.index = pd.to_datetime(df.pop("time").astype(int), unit="s", utc=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        df = closed_only(df, base)                      # drop Kraken's forming candle first
        return closed_only(to_timeframe(df, base, minutes), minutes)


class CoinbaseFeed:
    """Coinbase's public market data (no account or key) - the backup when Kraken doesn't answer. Candles are kept
    on disk (runtime/candles) so after the first download only the newest ones are fetched."""
    GRANULARITIES = (86400, 21600, 3600, 900, 300, 60)
    CACHE_S = 25

    def __init__(self):
        self._cache: dict = {}

    @staticmethod
    def product(symbol: str) -> str:
        from .history import CRYPTO
        coin = next((c for c, p in CRYPTO.items() if p == symbol), symbol[:-3] if symbol.endswith("USD") else symbol)
        return f"{coin}-USD"

    def bars(self, symbol: str, timeframe: str) -> pd.DataFrame:
        from .history import _coinbase
        key = (symbol, timeframe)
        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < self.CACHE_S:
            return hit[1]
        minutes = TF_MINUTES[timeframe]
        gran = next(g for g in self.GRANULARITIES if (minutes * 60) % g == 0)
        days = min(365.0, max(2.0, 720 * minutes / 1440))   # about as many candles as Kraken gives
        df = _coinbase(self.product(symbol), gran, days)
        out = closed_only(to_timeframe(df, gran // 60, minutes), minutes)
        self._cache[key] = (time.time(), out)
        return out

    def price(self, symbol: str) -> float:
        r = requests.get(f"https://api.exchange.coinbase.com/products/{self.product(symbol)}/ticker", timeout=10,
                         headers={"User-Agent": "paper-trader"})
        r.raise_for_status()
        return float(r.json()["price"])


class WebullFeed:
    """Webull market data through the official SDK (pip install webull-openapi-python-sdk).
    The response layout is parsed defensively; `python -m livepaper check` prints a raw sample."""
    TIMESPANS = {1: "M1", 5: "M5", 15: "M15", 30: "M30", 60: "M60", 120: "M120", 240: "M240", 1440: "D"}

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
        return closed_only(stock(symbol, timeframe), min(TF_MINUTES[timeframe], 60 * 24))   # down to 1m

    def price(self, symbol: str) -> float:
        import yfinance as yf
        return float(yf.Ticker(symbol).fast_info["last_price"])
