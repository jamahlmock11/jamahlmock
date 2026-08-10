"""Kalshi 15-minute market series registry and discovery."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

SeriesKind = Literal["crypto_up_down", "commodity_up_down", "index_up_down", "head_to_head"]


@dataclass(frozen=True)
class Series15mSpec:
    """Metadata for a Kalshi 15-minute up/down series."""

    ticker: str
    asset: str
    kind: SeriesKind
    cf_benchmarks_index: str | None = None
    kraken_pair: str | None = None
    yahoo_symbol: str | None = None
    # For commodities/financials settled on Pyth or other feeds.
    proxy_label: str | None = None


# Core crypto 15M up/down markets (CF Benchmarks RTI settlement).
CRYPTO_15M_SERIES: dict[str, Series15mSpec] = {
    "KXBTC15M": Series15mSpec(
        ticker="KXBTC15M",
        asset="BTC",
        kind="crypto_up_down",
        cf_benchmarks_index="BRTI",
        kraken_pair="XBTUSD",
        yahoo_symbol="BTC-USD",
    ),
    "KXETH15M": Series15mSpec(
        ticker="KXETH15M",
        asset="ETH",
        kind="crypto_up_down",
        cf_benchmarks_index="ETHUSD_RTI",
        kraken_pair="ETHUSD",
        yahoo_symbol="ETH-USD",
    ),
    "KXSOL15M": Series15mSpec(
        ticker="KXSOL15M",
        asset="SOL",
        kind="crypto_up_down",
        cf_benchmarks_index="SOLUSD_RTI",
        kraken_pair="SOLUSD",
        yahoo_symbol="SOL-USD",
    ),
    "KXXRP15M": Series15mSpec(
        ticker="KXXRP15M",
        asset="XRP",
        kind="crypto_up_down",
        cf_benchmarks_index="XRPUSD_RTI",
        kraken_pair="XRPUSD",
        yahoo_symbol="XRP-USD",
    ),
    "KXDOGE15M": Series15mSpec(
        ticker="KXDOGE15M",
        asset="DOGE",
        kind="crypto_up_down",
        cf_benchmarks_index="DOGEUSD_RTI",
        kraken_pair="XDGUSD",
        yahoo_symbol="DOGE-USD",
    ),
    "KXBNB15M": Series15mSpec(
        ticker="KXBNB15M",
        asset="BNB",
        kind="crypto_up_down",
        cf_benchmarks_index="BNBUSD_RTI",
        kraken_pair=None,
        yahoo_symbol="BNB-USD",
    ),
    "KXHYPE15M": Series15mSpec(
        ticker="KXHYPE15M",
        asset="HYPE",
        kind="crypto_up_down",
        cf_benchmarks_index="HYPEUSD_RTI",
        kraken_pair=None,
        yahoo_symbol=None,
    ),
    "KXADA15M": Series15mSpec(
        ticker="KXADA15M",
        asset="ADA",
        kind="crypto_up_down",
        cf_benchmarks_index="ADAUSD_RTI",
        kraken_pair="ADAUSD",
        yahoo_symbol="ADA-USD",
    ),
    "KXBCH15M": Series15mSpec(
        ticker="KXBCH15M",
        asset="BCH",
        kind="crypto_up_down",
        cf_benchmarks_index="BCHUSD_RTI",
        kraken_pair="BCHUSD",
        yahoo_symbol="BCH-USD",
    ),
    "KXZEC15M": Series15mSpec(
        ticker="KXZEC15M",
        asset="ZEC",
        kind="crypto_up_down",
        cf_benchmarks_index="ZECUSD_RTI",
        kraken_pair="ZECUSD",
        yahoo_symbol="ZEC-USD",
    ),
    "KXNEAR15M": Series15mSpec(
        ticker="KXNEAR15M",
        asset="NEAR",
        kind="crypto_up_down",
        cf_benchmarks_index="NEARUSD_RTI",
        kraken_pair="NEARUSD",
        yahoo_symbol="NEAR-USD",
    ),
    "KXTON15M": Series15mSpec(
        ticker="KXTON15M",
        asset="TON",
        kind="crypto_up_down",
        cf_benchmarks_index="TONUSD_RTI",
        kraken_pair=None,
        yahoo_symbol=None,
    ),
}

# Non-crypto 15M up/down (Pyth or index settlement — proxy pricing only).
OTHER_15M_SERIES: dict[str, Series15mSpec] = {
    "KXGOLD15M": Series15mSpec(
        ticker="KXGOLD15M",
        asset="GOLD",
        kind="commodity_up_down",
        yahoo_symbol="GC=F",
        proxy_label="yahoo_gold_futures",
    ),
    "KXSILVER15M": Series15mSpec(
        ticker="KXSILVER15M",
        asset="SILVER",
        kind="commodity_up_down",
        yahoo_symbol="SI=F",
        proxy_label="yahoo_silver_futures",
    ),
    "KXWTI15M": Series15mSpec(
        ticker="KXWTI15M",
        asset="WTI",
        kind="commodity_up_down",
        yahoo_symbol="CL=F",
        proxy_label="yahoo_wti_futures",
    ),
    "KXINX15M": Series15mSpec(
        ticker="KXINX15M",
        asset="SPX",
        kind="index_up_down",
        yahoo_symbol="^GSPC",
        proxy_label="yahoo_spx",
    ),
    "KXNDQ15M": Series15mSpec(
        ticker="KXNDQ15M",
        asset="NDX",
        kind="index_up_down",
        yahoo_symbol="^NDX",
        proxy_label="yahoo_ndx",
    ),
    "KXCRYPTOCOMP15M": Series15mSpec(
        ticker="KXCRYPTOCOMP15M",
        asset="CRYPTOCOMP",
        kind="head_to_head",
    ),
    "KXCRYPTOLEAD15M": Series15mSpec(
        ticker="KXCRYPTOLEAD15M",
        asset="CRYPTOLEAD",
        kind="head_to_head",
    ),
}

ALL_15M_SERIES: dict[str, Series15mSpec] = {**CRYPTO_15M_SERIES, **OTHER_15M_SERIES}

_SERIES_RE = re.compile(r"^(KX[A-Z0-9]+15M)")


def parse_series_ticker(event_ticker: str | None) -> str | None:
    """Extract series ticker from a Kalshi event or market ticker."""
    if not event_ticker:
        return None
    match = _SERIES_RE.match(event_ticker.upper())
    return match.group(1) if match else None


def get_series_spec(series_ticker: str) -> Series15mSpec | None:
    return ALL_15M_SERIES.get(series_ticker.upper())


def is_up_down_15m(series_ticker: str) -> bool:
    spec = get_series_spec(series_ticker)
    return spec is not None and spec.kind in ("crypto_up_down", "commodity_up_down", "index_up_down")


def default_crypto_15m_tickers() -> list[str]:
    """Primary liquid crypto 15M series (Kalshi's main lineup)."""
    return [
        "KXBTC15M",
        "KXETH15M",
        "KXSOL15M",
        "KXXRP15M",
        "KXDOGE15M",
        "KXBNB15M",
        "KXHYPE15M",
    ]


def resolve_enabled_series(
    *,
    configured: list[str] | None = None,
    auto_discover: bool = False,
    include_commodities: bool = True,
    include_head_to_head: bool = False,
) -> list[Series15mSpec]:
    """Resolve which 15M series to scan."""
    if configured:
        specs: list[Series15mSpec] = []
        for ticker in configured:
            spec = get_series_spec(ticker)
            if spec is None:
                continue
            if spec.kind == "head_to_head" and not include_head_to_head:
                continue
            if spec.kind in ("commodity_up_down", "index_up_down") and not include_commodities:
                continue
            specs.append(spec)
        return specs

    if auto_discover:
        specs = []
        for spec in ALL_15M_SERIES.values():
            if spec.kind == "head_to_head" and not include_head_to_head:
                continue
            if spec.kind in ("commodity_up_down", "index_up_down") and not include_commodities:
                continue
            specs.append(spec)
        return sorted(specs, key=lambda s: s.ticker)

    return [CRYPTO_15M_SERIES[t] for t in default_crypto_15m_tickers() if t in CRYPTO_15M_SERIES]
