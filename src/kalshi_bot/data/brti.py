"""Multi-asset spot resolution with CF Benchmarks official sources and fallbacks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import httpx
import yfinance as yf

from kalshi_bot.config import BrtiConfig, Settings
from kalshi_bot.data.cfbenchmarks import fetch_brti_direct_api, fetch_brti_public_summary
from kalshi_bot.data.kalshi_client import KalshiClient
from kalshi_bot.data.markets_15m import Series15mSpec, get_series_spec

logger = logging.getLogger(__name__)


@dataclass
class SpotSnapshot:
    brti: float
    source: str
    is_official: bool
    updated_at: datetime | None = None
    asset: str = "BTC"
    series_ticker: str | None = None


def _kraken_btc() -> float:
    with httpx.Client(timeout=10.0) as client:
        data = client.get("https://api.kraken.com/0/public/Ticker", params={"pair": "XBTUSD"}).json()
    return float(data["result"]["XXBTZUSD"]["c"][0])


def _coinbase_btc() -> float:
    with httpx.Client(timeout=10.0) as client:
        data = client.get("https://api.coinbase.com/v2/prices/BTC-USD/spot").json()
    return float(data["data"]["amount"])


def _quote_to_snapshot(quote, *, asset: str = "BTC", series_ticker: str | None = None) -> SpotSnapshot:
    return SpotSnapshot(
        brti=quote.value,
        source=quote.source,
        is_official=quote.is_official,
        updated_at=quote.updated_at,
        asset=asset,
        series_ticker=series_ticker,
    )


def _kraken_ticker(pair: str) -> float:
    with httpx.Client(timeout=10.0) as client:
        data = client.get("https://api.kraken.com/0/public/Ticker", params={"pair": pair}).json()
    if data.get("error"):
        raise RuntimeError(f"Kraken ticker error: {data['error']}")
    for key, value in (data.get("result") or {}).items():
        return float(value["c"][0])
    raise RuntimeError(f"Kraken ticker empty for {pair}")


def _yahoo_spot(symbol: str) -> float:
    t = yf.Ticker(symbol)
    px = t.info.get("regularMarketPrice") or float(t.fast_info["last_price"])
    if px is None or px <= 0:
        raise RuntimeError(f"yahoo price unavailable for {symbol}")
    return float(px)


def resolve_spot(
    client: KalshiClient,
    fallback_btc: float | None = None,
    *,
    brti_cfg: BrtiConfig | None = None,
    settings: Settings | None = None,
    series_ticker: str = "KXBTC15M",
) -> SpotSnapshot:
    """Resolve settlement spot for a 15M or hourly market series."""
    spec = get_series_spec(series_ticker)
    if spec is None:
        return _resolve_btc_spot(
            client,
            fallback_btc,
            brti_cfg=brti_cfg,
            settings=settings,
        )
    return resolve_series_spot(
        client,
        spec,
        fallback=fallback_btc,
        brti_cfg=brti_cfg,
        settings=settings,
    )


def resolve_series_spot(
    client: KalshiClient,
    spec: Series15mSpec,
    *,
    fallback: float | None = None,
    brti_cfg: BrtiConfig | None = None,
    settings: Settings | None = None,
) -> SpotSnapshot:
    """Resolve spot for any registered 15M series."""
    cfg = brti_cfg or BrtiConfig()
    settings = settings or Settings()
    index_id = spec.cf_benchmarks_index or cfg.index_id

    if cfg.prefer_official and index_id:
        if spec.asset == "BTC":
            brti = client.get_brti(index_id=index_id)
            if brti is not None and brti > 0:
                return SpotSnapshot(
                    brti=brti,
                    source="cfbenchmarks_kalshi_passthrough",
                    is_official=True,
                    asset=spec.asset,
                    series_ticker=spec.ticker,
                )
        else:
            username = settings.cf_benchmarks_api_username or cfg.cf_benchmarks_username
            api_key = settings.cf_benchmarks_api_key or cfg.cf_benchmarks_api_key
            if username and api_key:
                quote = fetch_brti_direct_api(
                    username=username,
                    api_key=api_key,
                    index_id=index_id,
                )
                if quote is not None:
                    return _quote_to_snapshot(
                        quote,
                        asset=spec.asset,
                        series_ticker=spec.ticker,
                    )

        if cfg.public_summary_enabled:
            quote = fetch_brti_public_summary(index_id=index_id)
            if quote is not None:
                logger.info(
                    "using official CF Benchmarks %s from %s (%.4f)",
                    index_id,
                    quote.source,
                    quote.value,
                )
                return _quote_to_snapshot(
                    quote,
                    asset=spec.asset,
                    series_ticker=spec.ticker,
                )

    if not cfg.allow_exchange_proxy:
        raise RuntimeError(
            f"official spot unavailable for {spec.ticker} and exchange proxy disabled"
        )

    if spec.kraken_pair:
        try:
            px = _kraken_ticker(spec.kraken_pair)
            if px > 0:
                logger.warning(
                    "using kraken:%s proxy for %s (not official settlement index)",
                    spec.kraken_pair,
                    spec.ticker,
                )
                return SpotSnapshot(
                    brti=float(px),
                    source=f"kraken_{spec.kraken_pair.lower()}",
                    is_official=False,
                    asset=spec.asset,
                    series_ticker=spec.ticker,
                )
        except Exception as exc:
            logger.debug("kraken spot for %s failed: %s", spec.ticker, exc)

    if spec.asset == "BTC":
        for name, fn in (("kraken_xbtusd", _kraken_btc), ("coinbase_btc_usd", _coinbase_btc)):
            try:
                px = fn()
                if px > 0:
                    logger.warning("using %s proxy for BRTI (not official settlement index)", name)
                    return SpotSnapshot(
                        brti=float(px),
                        source=name,
                        is_official=False,
                        asset=spec.asset,
                        series_ticker=spec.ticker,
                    )
            except Exception as exc:
                logger.debug("spot source %s failed: %s", name, exc)

    if spec.yahoo_symbol:
        try:
            px = _yahoo_spot(spec.yahoo_symbol)
            logger.warning(
                "using yahoo:%s proxy for %s (not official settlement index)",
                spec.yahoo_symbol,
                spec.ticker,
            )
            return SpotSnapshot(
                brti=float(px),
                source=f"yahoo_{spec.yahoo_symbol}",
                is_official=False,
                asset=spec.asset,
                series_ticker=spec.ticker,
            )
        except Exception as exc:
            logger.debug("yahoo spot for %s failed: %s", spec.ticker, exc)

    if fallback is not None and fallback > 0:
        return SpotSnapshot(
            brti=fallback,
            source="provided_fallback",
            is_official=False,
            asset=spec.asset,
            series_ticker=spec.ticker,
        )

    raise RuntimeError(f"unable to resolve spot for {spec.ticker}")


def _resolve_btc_spot(
    client: KalshiClient,
    fallback_btc: float | None = None,
    *,
    brti_cfg: BrtiConfig | None = None,
    settings: Settings | None = None,
) -> SpotSnapshot:
    """Resolve BRTI for legacy hourly bots."""
    from kalshi_bot.data.markets_15m import CRYPTO_15M_SERIES

    return resolve_series_spot(
        client,
        CRYPTO_15M_SERIES["KXBTC15M"],
        fallback=fallback_btc,
        brti_cfg=brti_cfg,
        settings=settings,
    )
