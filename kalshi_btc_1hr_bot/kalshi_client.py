"""Kalshi REST API client for KXBTCD markets."""

from __future__ import annotations

import base64
import logging
import time
from datetime import datetime, timezone
from typing import Any, Iterator

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from kalshi_btc_1hr_bot.config import BotConfig
from kalshi_btc_1hr_bot.utils import parse_ts

logger = logging.getLogger(__name__)


class KalshiClient:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.base_url = config.kalshi_base_url.rstrip("/")
        self._client = httpx.Client(timeout=20.0)
        self._private_key = None
        if config.kalshi_api_key_id and config.kalshi_private_key_pem:
            pem = config.kalshi_private_key_pem.replace("\\n", "\n")
            self._private_key = serialization.load_pem_private_key(pem.encode(), password=None)

    def close(self) -> None:
        self._client.close()

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        if self._private_key is None:
            raise RuntimeError("Kalshi credentials not configured")
        msg = f"{timestamp_ms}{method.upper()}{path}".encode()
        sig = self._private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode()

    def _request(self, method: str, path: str, *, params: dict | None = None, json: dict | None = None) -> Any:
        url = f"{self.base_url}{path}"
        headers: dict[str, str] = {}
        if self._private_key is not None:
            ts = str(int(time.time() * 1000))
            sign_path = path.split("?")[0]
            headers = {
                "KALSHI-ACCESS-KEY": self.config.kalshi_api_key_id,
                "KALSHI-ACCESS-TIMESTAMP": ts,
                "KALSHI-ACCESS-SIGNATURE": self._sign(ts, method, sign_path),
            }
        resp = self._client.request(method, url, params=params, json=json, headers=headers)
        resp.raise_for_status()
        return resp.json()

    def iter_markets(self, series_ticker: str, *, status: str = "open") -> Iterator[dict]:
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"series_ticker": series_ticker, "status": status, "limit": 200}
            if cursor:
                params["cursor"] = cursor
            data = self._request("GET", "/markets", params=params)
            for m in data.get("markets", []):
                yield m
            cursor = data.get("cursor")
            if not cursor:
                break

    def get_orderbook(self, ticker: str) -> dict:
        return self._request("GET", f"/markets/{ticker}/orderbook")

    def place_order(
        self,
        *,
        ticker: str,
        side: str,
        action: str,
        count: int,
        price_cents: int,
        order_type: str = "limit",
    ) -> dict:
        body = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": order_type,
            f"{side}_price": price_cents,
        }
        return self._request("POST", "/portfolio/orders", json=body)

    @property
    def authenticated(self) -> bool:
        return self._private_key is not None

    def get_brti(self, index_id: str = "BRTI") -> float | None:
        """Fetch CF Benchmarks BRTI via Kalshi authenticated passthrough."""
        from kalshi_btc_1hr_bot.brti import parse_brti_payload

        if not self.authenticated:
            return None
        try:
            data = self._request("GET", "/cfbenchmarks/values", params={"id": index_id})
        except Exception as exc:
            logger.warning("BRTI passthrough failed: %s", exc)
            return None
        value = parse_brti_payload(data)
        if value is None:
            logger.warning("unrecognized BRTI payload")
        return value


def normalize_market(raw: dict) -> dict:
    """Extract normalized fields from a Kalshi market payload."""
    strike = raw.get("floor_strike") or raw.get("cap_strike")
    if strike is None:
        subtitle = str(raw.get("subtitle") or "")
        for token in subtitle.replace(",", "").split():
            try:
                strike = float(token.replace("$", ""))
                break
            except ValueError:
                continue

    yes_bid = _price_to_float(raw.get("yes_bid"))
    yes_ask = _price_to_float(raw.get("yes_ask"))
    no_bid = _price_to_float(raw.get("no_bid"))
    no_ask = _price_to_float(raw.get("no_ask"))

    return {
        "ticker": raw.get("ticker"),
        "series_ticker": raw.get("series_ticker") or raw.get("event_ticker", "")[:7],
        "strike": float(strike) if strike is not None else None,
        "close_time": parse_ts(raw.get("close_time") or raw.get("expiration_time")),
        "open_time": parse_ts(raw.get("open_time")),
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
        "volume": float(raw.get("volume") or 0),
        "raw": raw,
    }


def _price_to_float(val: Any) -> float | None:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        v = float(val)
        return v / 100.0 if v > 1.0 else v
    try:
        v = float(val)
        return v / 100.0 if v > 1.0 else v
    except (TypeError, ValueError):
        return None


def is_hourly_market(raw: dict) -> bool:
    """Filter out daily KXBTCD buckets."""
    meta = raw.get("product_metadata") or {}
    cadence = str(meta.get("cadence") or "").lower()
    if cadence == "hourly":
        return True
    if cadence == "daily":
        return False
    open_t = parse_ts(raw.get("open_time"))
    close_t = parse_ts(raw.get("close_time"))
    if open_t and close_t:
        span = (close_t - open_t).total_seconds()
        return 50 * 60 <= span <= 70 * 60
    return True
