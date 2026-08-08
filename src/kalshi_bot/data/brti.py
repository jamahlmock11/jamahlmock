"""BRTI spot resolution with public fallbacks."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
import yfinance as yf

from kalshi_bot.data.kalshi_client import KalshiClient

logger = logging.getLogger(__name__)


@dataclass
class SpotSnapshot:
    brti: float
    source: str
    is_official: bool


def _kraken_btc() -> float:
    with httpx.Client(timeout=10.0) as client:
        data = client.get("https://api.kraken.com/0/public/Ticker", params={"pair": "XBTUSD"}).json()
    return float(data["result"]["XXBTZUSD"]["c"][0])


def _coinbase_btc() -> float:
    with httpx.Client(timeout=10.0) as client:
        data = client.get("https://api.coinbase.com/v2/prices/BTC-USD/spot").json()
    return float(data["data"]["amount"])


def resolve_spot(client: KalshiClient, fallback_btc: float | None = None) -> SpotSnapshot:
    """Prefer authenticated CF Benchmarks BRTI; else public BTC proxy.

    Settlement is on BRTI 60s average. Using a proxy is fine for scanning,
    but live trading should use the official passthrough.
    """
    brti = client.get_brti()
    if brti is not None and brti > 0:
        return SpotSnapshot(brti=brti, source="cfbenchmarks_brti", is_official=True)

    for name, fn in (("kraken_xbtusd", _kraken_btc), ("coinbase_btc_usd", _coinbase_btc)):
        try:
            px = fn()
            if px > 0:
                logger.warning("using %s proxy for BRTI (not official settlement index)", name)
                return SpotSnapshot(brti=float(px), source=name, is_official=False)
        except Exception as exc:
            logger.debug("spot source %s failed: %s", name, exc)

    if fallback_btc is not None and fallback_btc > 0:
        return SpotSnapshot(brti=fallback_btc, source="provided_fallback", is_official=False)

    try:
        t = yf.Ticker("BTC-USD")
        px = t.info.get("regularMarketPrice") or float(t.fast_info["last_price"])
        logger.warning("using BTC-USD proxy for BRTI (not official settlement index)")
        return SpotSnapshot(brti=float(px), source="yahoo_btc_usd", is_official=False)
    except Exception as exc:
        raise RuntimeError(f"unable to resolve BTC spot: {exc}") from exc
