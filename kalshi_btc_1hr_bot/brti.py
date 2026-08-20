"""CF Benchmarks BRTI — official Kalshi settlement index.

Primary source: https://www.cfbenchmarks.com/data/indices/BRTI
"""

from __future__ import annotations

import base64
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

CFB_BASE = "https://www.cfbenchmarks.com"
CFB_API = f"{CFB_BASE}/api/v1"
CFB_INDEX_PAGE = f"{CFB_BASE}/data/indices/BRTI"
_BUILD_ID_RE = re.compile(r"/_next/static/([^/]+)/_buildManifest")


@dataclass(frozen=True)
class BrtiQuote:
    value: float
    source: str
    is_official: bool
    updated_at: datetime | None = None


def parse_brti_payload(data: Any) -> float | None:
    """Extract the latest BRTI value from CF Benchmarks or Kalshi envelopes."""
    if data is None:
        return None
    if isinstance(data, (int, float, str)):
        try:
            val = float(data)
            return val if val > 0 else None
        except (TypeError, ValueError):
            return None
    if not isinstance(data, dict):
        return None
    if "value" in data and data["value"] not in (None, ""):
        try:
            val = float(data["value"])
            return val if val > 0 else None
        except (TypeError, ValueError):
            pass
    for key in ("data", "payload", "values"):
        nested = data.get(key)
        if nested is None:
            continue
        if isinstance(nested, list):
            for item in reversed(nested):
                val = parse_brti_payload(item)
                if val is not None:
                    return val
        else:
            val = parse_brti_payload(nested)
            if val is not None:
                return val
    return None


def _next_build_id(client: httpx.Client) -> str | None:
    try:
        html = client.get(CFB_INDEX_PAGE, follow_redirects=True).text
    except Exception as exc:
        logger.debug("CF Benchmarks build id fetch failed: %s", exc)
        return None
    match = _BUILD_ID_RE.search(html)
    return match.group(1) if match else None


def fetch_brti_public_summary(
    *,
    index_id: str = "BRTI",
    timeout: float = 10.0,
) -> BrtiQuote | None:
    """Fetch the official displayed RTI from the CF Benchmarks index page."""
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        build_id = _next_build_id(client)
        if not build_id:
            return None
        url = f"{CFB_BASE}/_next/data/{build_id}/data/indices/{index_id}.json"
        try:
            resp = client.get(url, params={"externalIndexId": index_id})
            resp.raise_for_status()
            summary = resp.json().get("pageProps", {}).get("indexSummary") or {}
            value = parse_brti_payload(summary)
            if value is None:
                return None
            updated_at = None
            last_updated = summary.get("lastUpdated")
            if last_updated is not None:
                try:
                    updated_at = datetime.fromtimestamp(float(last_updated) / 1000.0, tz=timezone.utc)
                except (TypeError, ValueError, OSError):
                    updated_at = None
            return BrtiQuote(
                value=value,
                source="cfbenchmarks_public_rti",
                is_official=True,
                updated_at=updated_at,
            )
        except Exception as exc:
            logger.warning("CF Benchmarks public BRTI summary failed: %s", exc)
            return None


def fetch_brti_direct_api(
    *,
    username: str,
    api_key: str,
    index_id: str = "BRTI",
    timeout: float = 10.0,
) -> BrtiQuote | None:
    """Fetch BRTI via licensed CF Benchmarks REST credentials."""
    creds = base64.b64encode(f"{username}:{api_key}".encode()).decode()
    headers = {"Authorization": f"Basic {creds}", "Accept": "application/json"}
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{CFB_API}/values", params={"id": index_id}, headers=headers)
            resp.raise_for_status()
            value = parse_brti_payload(resp.json())
            if value is None:
                return None
            return BrtiQuote(
                value=value,
                source="cfbenchmarks_api",
                is_official=True,
            )
    except Exception as exc:
        logger.warning("CF Benchmarks direct API failed: %s", exc)
        return None


def resolve_brti(
    *,
    kalshi_client=None,
    index_id: str = "BRTI",
    prefer_official: bool = True,
    allow_exchange_proxy: bool = True,
) -> BrtiQuote:
    """Resolve BRTI with official sources first, exchange proxies last.

    Priority:
    1. CF Benchmarks public page (https://www.cfbenchmarks.com/data/indices/BRTI)
    2. Kalshi authenticated /cfbenchmarks passthrough
    3. Licensed CF Benchmarks REST API (env credentials)
    4. Kraken / Coinbase exchange proxies (not settlement grade)
    """
    if prefer_official:
        quote = fetch_brti_public_summary(index_id=index_id)
        if quote is not None:
            logger.info("BRTI from %s: %.2f", quote.source, quote.value)
            return quote

        if kalshi_client is not None:
            brti = kalshi_client.get_brti(index_id=index_id)
            if brti is not None and brti > 0:
                return BrtiQuote(
                    value=brti,
                    source="cfbenchmarks_kalshi_passthrough",
                    is_official=True,
                )

        username = os.getenv("CF_BENCHMARKS_API_USERNAME", "")
        api_key = os.getenv("CF_BENCHMARKS_API_KEY", "")
        if username and api_key:
            quote = fetch_brti_direct_api(username=username, api_key=api_key, index_id=index_id)
            if quote is not None:
                return quote

    if not allow_exchange_proxy:
        raise RuntimeError("official BRTI unavailable and exchange proxy disabled")

    for name, fetch in (("kraken_xbtusd", _kraken_btc), ("coinbase_btc_usd", _coinbase_btc)):
        try:
            px = fetch()
            if px > 0:
                logger.warning("using %s proxy for BRTI (not official settlement index)", name)
                return BrtiQuote(value=px, source=name, is_official=False)
        except Exception as exc:
            logger.debug("BRTI proxy %s failed: %s", name, exc)

    raise RuntimeError("unable to resolve BRTI")


def _kraken_btc() -> float:
    with httpx.Client(timeout=10.0) as client:
        data = client.get("https://api.kraken.com/0/public/Ticker", params={"pair": "XBTUSD"}).json()
    for key, val in (data.get("result") or {}).items():
        if key == "last":
            continue
        return float(val["c"][0])
    raise RuntimeError("kraken ticker empty")


def _coinbase_btc() -> float:
    with httpx.Client(timeout=10.0) as client:
        data = client.get("https://api.coinbase.com/v2/prices/BTC-USD/spot").json()
    return float(data["data"]["amount"])
