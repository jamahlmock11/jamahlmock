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
    ) -> dict:
        """Place an order via legacy portfolio endpoint (widely supported).

        Prefer fixed-point dollar prices when provided.
        """
        body: dict[str, Any] = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": "limit",
            "client_order_id": client_order_id or str(uuid.uuid4()),
        }
        # Convert dollar string "0.42" -> cents int for legacy endpoint.
        if yes_price_dollars is not None:
            body["yes_price"] = int(round(float(yes_price_dollars) * 100))
        if no_price_dollars is not None:
            body["no_price"] = int(round(float(no_price_dollars) * 100))
        # time_in_force mapping for legacy: IOC via expiration_ts near-now is awkward;
        # many stacks use post-only false and rely on fill. Keep field if accepted.
        try:
            return self.request("POST", "/portfolio/orders", json_body=body)
        except RuntimeError:
            # Fallback to V2 events orders shape
            v2 = {
                "ticker": ticker,
                "side": "yes" if side == "yes" else "no",
                "action": action,
                "count": f"{count:.2f}",
                "type": "limit",
                "client_order_id": body["client_order_id"],
                "time_in_force": time_in_force,
            }
            if yes_price_dollars is not None and side == "yes":
                v2["yes_price_dollars"] = yes_price_dollars
            if no_price_dollars is not None and side == "no":
                v2["no_price_dollars"] = no_price_dollars
            return self.request("POST", "/portfolio/events/orders", json_body=v2)

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
