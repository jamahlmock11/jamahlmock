"""Authenticated and public Kalshi REST client."""

from __future__ import annotations

import base64
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import httpx
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

logger = logging.getLogger(__name__)


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def dollars(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


class KalshiClient:
    def __init__(
        self,
        base_url: str,
        api_key_id: str | None = None,
        private_key_pem: str | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key_id = api_key_id
        self._private_key = None
        if private_key_pem:
            self._private_key = serialization.load_pem_private_key(
                private_key_pem.encode() if isinstance(private_key_pem, str) else private_key_pem,
                password=None,
                backend=default_backend(),
            )
        self._http = httpx.Client(timeout=timeout)

    @property
    def authenticated(self) -> bool:
        return bool(self.api_key_id and self._private_key)

    def close(self) -> None:
        self._http.close()

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        assert self._private_key is not None
        path_no_query = path.split("?")[0]
        message = f"{timestamp_ms}{method.upper()}{path_no_query}".encode()
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode()

    def _headers(self, method: str, full_path: str) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if not self.authenticated:
            return headers
        ts = str(int(time.time() * 1000))
        headers.update(
            {
                "KALSHI-ACCESS-KEY": self.api_key_id or "",
                "KALSHI-ACCESS-TIMESTAMP": ts,
                "KALSHI-ACCESS-SIGNATURE": self._sign(ts, method, full_path),
            }
        )
        return headers

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json_body: dict | None = None,
    ) -> Any:
        if not path.startswith("/"):
            path = "/" + path
        url = self.base_url + path
        sign_path = urlparse(url).path
        headers = self._headers(method, sign_path)
        resp = self._http.request(method, url, params=params, json=json_body, headers=headers)
        if resp.status_code >= 400:
            raise RuntimeError(f"Kalshi {method} {path} -> {resp.status_code}: {resp.text[:500]}")
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    def get(self, path: str, **params: Any) -> Any:
        return self.request("GET", path, params=params or None)

    def get_markets(
        self,
        *,
        series_ticker: str | None = None,
        status: str = "open",
        limit: int = 200,
        cursor: str | None = None,
        event_ticker: str | None = None,
    ) -> dict:
        params: dict[str, Any] = {"status": status, "limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        if event_ticker:
            params["event_ticker"] = event_ticker
        return self.get("/markets", **params)

    def iter_markets(self, series_ticker: str, status: str = "open", limit: int = 200):
        cursor = None
        while True:
            page = self.get_markets(
                series_ticker=series_ticker, status=status, limit=limit, cursor=cursor
            )
            for m in page.get("markets", []):
                yield m
            cursor = page.get("cursor") or ""
            if not cursor:
                break

    def get_orderbook(self, ticker: str, depth: int = 10) -> dict:
        return self.get(f"/markets/{ticker}/orderbook", depth=depth)

    def get_market(self, ticker: str) -> dict:
        data = self.get(f"/markets/{ticker}")
        return data.get("market") or data

    def get_trades(
        self,
        *,
        ticker: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
        min_ts: int | None = None,
    ) -> dict:
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if cursor:
            params["cursor"] = cursor
        if min_ts is not None:
            params["min_ts"] = min_ts
        return self.get("/markets/trades", **params)

    def iter_trades(self, *, ticker: str, limit: int = 100, max_pages: int = 5):
        cursor = None
        pages = 0
        while pages < max_pages:
            page = self.get_trades(ticker=ticker, limit=limit, cursor=cursor)
            for trade in page.get("trades") or []:
                yield trade
            cursor = page.get("cursor") or ""
            pages += 1
            if not cursor:
                break

    def websocket_url(self) -> str:
        if "demo" in self.base_url:
            return "wss://demo-api.kalshi.co/trade-api/ws/v2"
        return "wss://api.elections.kalshi.com/trade-api/ws/v2"

    def get_balance(self) -> dict:
        return self.get("/portfolio/balance")

    def get_positions(self, **params: Any) -> dict:
        return self.get("/portfolio/positions", **params)

    def create_order(
        self,
        *,
        ticker: str,
        side: str,
        action: str,
        count: int,
        yes_price_dollars: str | None = None,
        no_price_dollars: str | None = None,
        time_in_force: str = "immediate_or_cancel",
        client_order_id: str | None = None,
        self_trade_prevention_type: str = "taker_at_cross",
        reduce_only: bool = False,
    ) -> dict:
        """Place an order — prefers Kalshi V2 events API (bid/ask + fixed-point price)."""
        cid = client_order_id or str(uuid.uuid4())

        def _v2_book_side_and_price() -> tuple[str, str]:
            """Map legacy yes/no + buy/sell to V2 bid/ask on the YES book."""
            if side == "yes":
                book_side = "bid" if action == "buy" else "ask"
                price = yes_price_dollars
            else:
                # NO orders quote via complementary YES price on the single book.
                book_side = "ask" if action == "buy" else "bid"
                price = no_price_dollars
                if price is not None:
                    price = f"{1.0 - float(price):.4f}"
            if price is None:
                raise ValueError(f"missing price for {action} {side}")
            return book_side, price

        try:
            book_side, price = _v2_book_side_and_price()
            v2 = {
                "ticker": ticker,
                "client_order_id": cid,
                "side": book_side,
                "count": f"{count:.2f}",
                "price": price,
                "time_in_force": time_in_force,
                "self_trade_prevention_type": self_trade_prevention_type,
                "reduce_only": reduce_only,
                "post_only": False,
                "cancel_order_on_pause": True,
            }
            return self.request("POST", "/portfolio/events/orders", json_body=v2)
        except (RuntimeError, ValueError) as exc:
            logger.warning("V2 order failed (%s), trying legacy /portfolio/orders", exc)

        body: dict[str, Any] = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": "limit",
            "client_order_id": cid,
        }
        if yes_price_dollars is not None:
            body["yes_price"] = int(round(float(yes_price_dollars) * 100))
        if no_price_dollars is not None:
            body["no_price"] = int(round(float(no_price_dollars) * 100))
        return self.request("POST", "/portfolio/orders", json_body=body)

    def get_brti(self, index_id: str = "BRTI") -> float | None:
        """Fetch CF Benchmarks BRTI via Kalshi authenticated passthrough."""
        from kalshi_bot.data.cfbenchmarks import parse_brti_payload

        if not self.authenticated:
            return None
        try:
            data = self.get("/cfbenchmarks/values", id=index_id)
        except Exception as exc:
            logger.warning("BRTI passthrough failed: %s", exc)
            return None
        value = parse_brti_payload(data)
        if value is None:
            logger.warning("unrecognized BRTI payload: %s", str(data)[:300])
        return value


def normalize_market(raw: dict) -> dict[str, Any]:
    """Normalize a Kalshi market dict into strategy-friendly fields."""
    close = _parse_ts(raw.get("close_time"))
    open_t = _parse_ts(raw.get("open_time"))
    floor = raw.get("floor_strike")
    yes_bid = dollars(raw.get("yes_bid_dollars"))
    yes_ask = dollars(raw.get("yes_ask_dollars"))
    no_bid = dollars(raw.get("no_bid_dollars"))
    no_ask = dollars(raw.get("no_ask_dollars"))
    last = dollars(raw.get("last_price_dollars"))
    series = (raw.get("event_ticker") or "").split("-")[0]
    # Infer series ticker more reliably from event ticker prefix
    event_ticker = raw.get("event_ticker") or ""
    if event_ticker.startswith("KXBTC15M"):
        series = "KXBTC15M"
    elif event_ticker.startswith("KXBTCD"):
        series = "KXBTCD"
    return {
        "ticker": raw.get("ticker"),
        "event_ticker": event_ticker,
        "series_ticker": series,
        "title": raw.get("title"),
        "status": raw.get("status"),
        "strike": float(floor) if floor is not None else None,
        "strike_type": raw.get("strike_type"),
        "close_time": close,
        "open_time": open_t,
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
        "last": last,
        "volume": float(raw.get("volume_fp") or 0),
        "open_interest": float(raw.get("open_interest_fp") or 0),
        "rules_primary": raw.get("rules_primary") or "",
        "raw": raw,
    }
