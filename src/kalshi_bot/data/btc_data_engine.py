"""Multi-feed BTC data engine for 15-minute mispricing."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Sequence

import httpx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BtcFeedQuote:
    source: str
    price: float
    age_seconds: float


@dataclass
class BtcMarketSnapshot:
    """Real-time BTC market state from multiple feeds."""

    reference_price: float
    reference_source: str
    is_official: bool
    feeds: tuple[BtcFeedQuote, ...]
    cross_exchange_agreement: float  # 0..1
    momentum_1m: float
    momentum_3m: float
    momentum_5m: float
    acceleration: float
    volume_ratio: float  # recent vs prior window
    annualized_vol: float
  # seconds since last successful refresh
    data_age_seconds: float
    stale: bool

    @property
    def order_flow_label(self) -> str:
        if self.momentum_1m > 0.0015 and self.acceleration > 0:
            return "BULLISH"
        if self.momentum_1m < -0.0015 and self.acceleration < 0:
            return "BEARISH"
        return "NEUTRAL"

    @property
    def volatility_label(self) -> str:
        if self.annualized_vol > 0.65:
            return "HIGH"
        if self.annualized_vol < 0.35:
            return "LOW"
        return "NORMAL"


def _fetch_kraken_ticker() -> float:
    with httpx.Client(timeout=10.0) as client:
        resp = client.get("https://api.kraken.com/0/public/Ticker", params={"pair": "XBTUSD"})
        resp.raise_for_status()
        data = resp.json()
    for key, val in (data.get("result") or {}).items():
        if key == "last":
            continue
        return float(val["c"][0])
    raise RuntimeError("kraken ticker empty")


def _fetch_coinbase_ticker() -> float:
    with httpx.Client(timeout=10.0) as client:
        resp = client.get("https://api.coinbase.com/v2/prices/BTC-USD/spot")
        resp.raise_for_status()
        return float(resp.json()["data"]["amount"])


def _fetch_kraken_closes(interval_minutes: int, limit: int) -> np.ndarray:
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(
            "https://api.kraken.com/0/public/OHLC",
            params={"pair": "XBTUSD", "interval": interval_minutes},
        )
        resp.raise_for_status()
        data = resp.json()
    for key, val in (data.get("result") or {}).items():
        if key == "last":
            continue
        rows = val[-limit:]
        return np.array([float(r[4]) for r in rows], dtype=float)
    raise RuntimeError("kraken ohlc empty")


def _momentum(closes: np.ndarray, bars: int) -> float:
    if len(closes) < bars + 1:
        return 0.0
    return float((closes[-1] - closes[-1 - bars]) / max(closes[-1 - bars], 1.0))


def _volume_ratio(closes: np.ndarray, window: int = 5) -> float:
    """Proxy volume from absolute returns when exchange volume unavailable."""
    if len(closes) < window * 2 + 1:
        return 1.0
    rets = np.abs(np.diff(closes))
    recent = float(np.mean(rets[-window:]))
    prior = float(np.mean(rets[-window * 2 : -window]))
    if prior <= 0:
        return 1.0
    return recent / prior


def _cross_exchange_agreement(prices: Sequence[float]) -> float:
    if len(prices) < 2:
        return 1.0
    median = float(np.median(prices))
    if median <= 0:
        return 0.0
    max_dev_pp = max(abs(p - median) / median * 100 for p in prices)
    return max(0.0, 1.0 - max_dev_pp / 0.50)  # 50bp full disagreement


class BtcDataEngine:
    """Polls BTC feeds and maintains short-horizon features."""

    def __init__(self, *, stale_after_seconds: float = 30.0) -> None:
        self.stale_after_seconds = stale_after_seconds
        self._last_snapshot: BtcMarketSnapshot | None = None
        self._last_fetch: float = 0.0

    def refresh(
        self,
        *,
        reference_price: float,
        reference_source: str,
        is_official: bool,
        annualized_vol: float,
        min_interval_seconds: float = 2.0,
    ) -> BtcMarketSnapshot:
        now = time.time()
        if (
            self._last_snapshot is not None
            and now - self._last_fetch < min_interval_seconds
        ):
            return self._last_snapshot

        feeds: list[BtcFeedQuote] = [
            BtcFeedQuote(reference_source, reference_price, 0.0),
        ]
        prices = [reference_price]
        t0 = time.time()
        for name, fetch in (("kraken", _fetch_kraken_ticker), ("coinbase", _fetch_coinbase_ticker)):
            try:
                px = fetch()
                feeds.append(BtcFeedQuote(name, px, time.time() - t0))
                prices.append(px)
            except Exception as exc:
                logger.debug("%s feed failed: %s", name, exc)

        try:
            closes_1m = _fetch_kraken_closes(1, 30)
            mom_1m = _momentum(closes_1m, 1)
            mom_3m = _momentum(closes_1m, 3)
            mom_5m = _momentum(closes_1m, 5)
            vol_ratio = _volume_ratio(closes_1m)
            if len(closes_1m) >= 4:
                accel = mom_1m - _momentum(closes_1m[:-1], 1)
            else:
                accel = 0.0
        except Exception:
            mom_1m = mom_3m = mom_5m = accel = 0.0
            vol_ratio = 1.0

        agreement = _cross_exchange_agreement(prices)
        age = time.time() - t0
        stale = age > self.stale_after_seconds or agreement < 0.5

        snap = BtcMarketSnapshot(
            reference_price=reference_price,
            reference_source=reference_source,
            is_official=is_official,
            feeds=tuple(feeds),
            cross_exchange_agreement=agreement,
            momentum_1m=mom_1m,
            momentum_3m=mom_3m,
            momentum_5m=mom_5m,
            acceleration=accel,
            volume_ratio=vol_ratio,
            annualized_vol=annualized_vol,
            data_age_seconds=age,
            stale=stale,
        )
        self._last_snapshot = snap
        self._last_fetch = now
        return snap
