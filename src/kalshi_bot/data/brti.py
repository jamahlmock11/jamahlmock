"""BRTI spot resolution with CF Benchmarks official sources and fallbacks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import httpx
import yfinance as yf

from kalshi_bot.config import BrtiConfig, Settings
from kalshi_bot.data.cfbenchmarks import fetch_brti_direct_api, fetch_brti_public_summary
from kalshi_bot.data.kalshi_client import KalshiClient

logger = logging.getLogger(__name__)


@dataclass
class SpotSnapshot:
    brti: float
    source: str
    is_official: bool
    updated_at: datetime | None = None


def _kraken_btc() -> float:
    with httpx.Client(timeout=10.0) as client:
        data = client.get("https://api.kraken.com/0/public/Ticker", params={"pair": "XBTUSD"}).json()
    return float(data["result"]["XXBTZUSD"]["c"][0])


def _coinbase_btc() -> float:
    with httpx.Client(timeout=10.0) as client:
        data = client.get("https://api.coinbase.com/v2/prices/BTC-USD/spot").json()
    return float(data["data"]["amount"])


def _quote_to_snapshot(quote) -> SpotSnapshot:
    return SpotSnapshot(
        brti=quote.value,
        source=quote.source,
        is_official=quote.is_official,
        updated_at=quote.updated_at,
    )


def resolve_spot(
    client: KalshiClient,
    fallback_btc: float | None = None,
    *,
    brti_cfg: BrtiConfig | None = None,
    settings: Settings | None = None,
) -> SpotSnapshot:
    """Resolve BRTI for both 15m and 1h bots.

    Priority:
    1. CF Benchmarks public BRTI page (https://www.cfbenchmarks.com/data/indices/BRTI)
    2. Kalshi authenticated CF Benchmarks passthrough
    3. Direct CF Benchmarks API credentials (if configured)
    4. Exchange BTC proxies (scanning only; not settlement grade)
    5. Provided fallback / Yahoo BTC-USD
    """
    cfg = brti_cfg or BrtiConfig()
    settings = settings or Settings()

    if cfg.prefer_official:
        if cfg.public_summary_enabled:
            quote = fetch_brti_public_summary(index_id=cfg.index_id)
            if quote is not None:
                logger.info(
                    "using official CF Benchmarks BRTI from %s (%.2f)",
                    quote.source,
                    quote.value,
                )
                return _quote_to_snapshot(quote)

        brti = client.get_brti()
        if brti is not None and brti > 0:
            return SpotSnapshot(brti=brti, source="cfbenchmarks_kalshi_passthrough", is_official=True)

        username = settings.cf_benchmarks_api_username or cfg.cf_benchmarks_username
        api_key = settings.cf_benchmarks_api_key or cfg.cf_benchmarks_api_key
        if username and api_key:
            quote = fetch_brti_direct_api(
                username=username,
                api_key=api_key,
                index_id=cfg.index_id,
            )
            if quote is not None:
                return _quote_to_snapshot(quote)

    if not cfg.allow_exchange_proxy:
        raise RuntimeError("official BRTI unavailable and exchange proxy disabled")

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
