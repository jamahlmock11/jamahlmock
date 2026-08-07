"""BRTI spot resolution with public fallbacks."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import yfinance as yf

from kalshi_bot.data.kalshi_client import KalshiClient

logger = logging.getLogger(__name__)


@dataclass
class SpotSnapshot:
    brti: float
    source: str
    is_official: bool


def resolve_spot(client: KalshiClient, fallback_btc: float | None = None) -> SpotSnapshot:
    """Prefer authenticated CF Benchmarks BRTI; else public BTC proxy.

    Settlement is on BRTI 60s average. Using a proxy is fine for scanning,
    but live trading should use the official passthrough.
    """
    brti = client.get_brti()
    if brti is not None and brti > 0:
        return SpotSnapshot(brti=brti, source="cfbenchmarks_brti", is_official=True)

    if fallback_btc is not None and fallback_btc > 0:
        return SpotSnapshot(brti=fallback_btc, source="provided_fallback", is_official=False)

    try:
        t = yf.Ticker("BTC-USD")
        px = t.info.get("regularMarketPrice") or float(t.fast_info["last_price"])
        logger.warning("using BTC-USD proxy for BRTI (not official settlement index)")
        return SpotSnapshot(brti=float(px), source="yahoo_btc_usd", is_official=False)
    except Exception as exc:
        raise RuntimeError(f"unable to resolve BTC spot: {exc}") from exc
