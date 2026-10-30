"""Webull sandbox (paper) broker through the official SDK.

Keys come from the dashboard's Account tab (local_settings.json) or the environment variables
WEBULL_APP_KEY, WEBULL_APP_SECRET and optional WEBULL_ACCOUNT_ID.
The sandbox host is api.sandbox.webull.com (apply for the Sandbox Trading API in
Webull's Developer Center: API Management -> My Application).
"""

from __future__ import annotations

import time
import uuid

from .core import Fill

FILLED = {"FILLED", "ALL_FILLED", "FINISHED", "COMPLETED"}
DEAD = {"CANCELLED", "CANCELED", "REJECTED", "FAILED", "EXPIRED"}


def api_client(region: str | None = None, endpoint: str | None = None):
    from webull.core.client import ApiClient
    from . import settings
    region = region or settings.get("webull_region", "us")
    endpoint = endpoint or settings.get("webull_endpoint", "api.sandbox.webull.com")
    key, secret = settings.get("webull_app_key"), settings.get("webull_app_secret")
    if not key or not secret:
        raise RuntimeError("No Webull keys yet: add them in the dashboard's Account tab.")
    client = ApiClient(key, secret, region)
    client.add_endpoint(region, endpoint)
    return client


def _find(d, *keys):
    """First matching key anywhere in a nested response."""
    if isinstance(d, dict):
        for k in keys:
            if k in d and d[k] not in (None, ""):
                return d[k]
        for v in d.values():
            r = _find(v, *keys)
            if r is not None:
                return r
    elif isinstance(d, list):
        for v in d:
            r = _find(v, *keys)
            if r is not None:
                return r
    return None


class WebullBroker:
    def __init__(self, client, assets: dict[str, tuple[str, str]], fee: float = 0.0, wait_seconds: int = 20):
        """assets: run name -> (symbol, 'stock' | 'crypto')."""
        from webull.trade.trade_client import TradeClient
        self.tc = TradeClient(client)
        self.assets, self.fee, self.wait = assets, fee, wait_seconds
        from . import settings
        self.account = settings.get("webull_account_id") or self._first_account()

    def _first_account(self) -> str:
        res = self.tc.account_v2.get_account_list()
        if res.status_code != 200:
            raise RuntimeError(f"Webull account list: HTTP {res.status_code} {res.text[:300]}")
        acc = _find(res.json(), "account_id", "accountId")
        if not acc:
            raise RuntimeError(f"no account id in {res.text[:300]}; set WEBULL_ACCOUNT_ID")
        return str(acc)

    def _qty(self, asset: str, qty: float) -> str:
        return f"{qty:.8f}".rstrip("0").rstrip(".") if asset == "crypto" else str(int(qty))

    @staticmethod
    def _px(px: float) -> str:
        return f"{px:.2f}" if px >= 1 else f"{px:.6g}"

    def _detail(self, coid: str):
        det = self.tc.order_v3.get_order_detail(self.account, coid)
        if det.status_code != 200:
            return None, None
        js = det.json()
        return str(_find(js, "status", "order_status") or "").upper(), js

    def _as_fill(self, js, coid: str, qty: float, price: float) -> Fill:
        px = float(_find(js, "filled_price", "avg_filled_price", "avg_price", "filled_avg_price") or price)
        q = float(_find(js, "filled_quantity", "filled_qty") or qty)
        return Fill(px, q, abs(px * q) * self.fee, coid)

    # ---- the stop order that rests at Webull (works with the laptop off) ----------
    def place_stop(self, run: str, side: str, qty: float, stop_price: float) -> str | None:
        """Good-till-cancelled stop-loss order. Returns its id, or None if Webull refuses it."""
        symbol, asset = self.assets[run]
        coid = uuid.uuid4().hex
        order = {"combo_type": "NORMAL", "client_order_id": coid, "symbol": symbol,
                 "instrument_type": "CRYPTO" if asset == "crypto" else "EQUITY", "market": "US",
                 "order_type": "STOP_LOSS", "stop_price": self._px(stop_price), "quantity": self._qty(asset, qty),
                 "side": side.upper(), "time_in_force": "GTC", "entrust_type": "QTY"}
        if asset != "crypto":
            order["support_trading_session"] = "CORE"
        res = self.tc.order_v3.place_order(self.account, [order])
        if res.status_code != 200:
            print(f"[webull] stop order refused ({side} {qty} {symbol} @ {stop_price}): HTTP {res.status_code} "
                  f"{res.text[:300]}")
            return None
        return coid

    def stop_status(self, run: str, coid: str, qty: float, stop_price: float):
        """A Fill if the stop has filled, 'dead' if cancelled / rejected, else 'open'.
        qty / stop_price fill in anything Webull's reply leaves out."""
        status, js = self._detail(coid)
        if status in FILLED:
            return self._as_fill(js, coid, qty, stop_price)
        return "dead" if status in DEAD else "open"

    def cancel_stop(self, run: str, coid: str, qty: float, stop_price: float):
        """Cancel before we sell ourselves. Returns a Fill if it filled first, else True."""
        self.tc.order_v3.cancel_order(self.account, coid)
        for _ in range(5):
            st = self.stop_status(run, coid, qty, stop_price)
            if st != "open":
                return st if isinstance(st, Fill) else True
            time.sleep(1)
        return True

    def fill(self, run: str, side: str, qty: float, price: float, reason: str) -> Fill | None:
        symbol, asset = self.assets[run]
        coid = uuid.uuid4().hex
        order = {"combo_type": "NORMAL", "client_order_id": coid, "symbol": symbol,
                 "instrument_type": "CRYPTO" if asset == "crypto" else "EQUITY", "market": "US",
                 "order_type": "MARKET", "quantity": self._qty(asset, qty), "side": side.upper(), "time_in_force": "DAY", "entrust_type": "QTY"}
        if asset != "crypto":
            order["support_trading_session"] = "CORE"
        res = self.tc.order_v3.place_order(self.account, [order])
        if res.status_code != 200:
            print(f"[webull] order rejected ({reason} {side} {qty} {symbol}): HTTP {res.status_code} {res.text[:300]}")
            return None
        deadline = time.time() + self.wait
        while time.time() < deadline:
            time.sleep(2)
            status, js = self._detail(coid)
            if status is None:
                continue
            if status in FILLED:
                return self._as_fill(js, coid, qty, price)
            if status in DEAD:
                print(f"[webull] order {coid} {status}: {str(js)[:300]}")
                return None
        print(f"[webull] order {coid} not filled within {self.wait}s; cancelling and retrying later")
        self.tc.order_v3.cancel_order(self.account, coid)
        return None
