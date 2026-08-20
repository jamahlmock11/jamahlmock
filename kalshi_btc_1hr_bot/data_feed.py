"""BTC data feed: Binance spot, funding rate, and BRTI proxy."""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field

import httpx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    spot: float
    vwap: float
    funding_rate: float
    annualized_vol: float
    mu_5m: float
    mu_15m: float
    mu_30m: float
    closes_1m: np.ndarray = field(repr=False)
    source: str = "binance"
    timestamp: float = field(default_factory=time.time)


class DataFeed:
    """Fetches BTC price, funding rate, and computes momentum/VWAP features."""

    def __init__(self, *, cache_seconds: float = 3.0) -> None:
        self.cache_seconds = cache_seconds
        self._last: MarketData | None = None
        self._last_fetch = 0.0
        self._client = httpx.Client(timeout=15.0)

    def close(self) -> None:
        self._client.close()

    def refresh(self) -> MarketData:
        now = time.time()
        if self._last and now - self._last_fetch < self.cache_seconds:
            return self._last

        spot, source = self._fetch_spot()
        funding = self._fetch_funding_rate()
        closes = self._fetch_closes(limit=60)
        vwap = self._compute_vwap(closes) if len(closes) else spot
        vol = self._estimate_vol(closes)
        mu_5m = self._momentum(closes, 5)
        mu_15m = self._momentum(closes, 15)
        mu_30m = self._momentum(closes, 30)

        self._last = MarketData(
            spot=spot,
            vwap=vwap,
            funding_rate=funding,
            annualized_vol=vol,
            mu_5m=mu_5m,
            mu_15m=mu_15m,
            mu_30m=mu_30m,
            closes_1m=closes,
            source=source,
            timestamp=now,
        )
        self._last_fetch = now
        return self._last

    def _fetch_spot(self) -> tuple[float, str]:
        for name, fetch in (
            ("binance", self._fetch_binance_spot),
            ("kraken", self._fetch_kraken_spot),
            ("coinbase", self._fetch_coinbase_spot),
        ):
            try:
                return fetch(), name
            except Exception as exc:
                logger.debug("%s spot failed: %s", name, exc)
        raise RuntimeError("all spot feeds failed")

    def _fetch_closes(self, limit: int = 60) -> np.ndarray:
        for fetch in (self._fetch_binance_klines, self._fetch_kraken_closes):
            try:
                return fetch(limit=limit)
            except Exception as exc:
                logger.debug("klines fetch failed: %s", exc)
        return np.array([], dtype=float)

    def _fetch_binance_spot(self) -> float:
        resp = self._client.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": "BTCUSDT"},
        )
        resp.raise_for_status()
        return float(resp.json()["price"])

    def _fetch_kraken_spot(self) -> float:
        resp = self._client.get(
            "https://api.kraken.com/0/public/Ticker",
            params={"pair": "XBTUSD"},
        )
        resp.raise_for_status()
        for key, val in (resp.json().get("result") or {}).items():
            if key == "last":
                continue
            return float(val["c"][0])
        raise RuntimeError("kraken ticker empty")

    def _fetch_coinbase_spot(self) -> float:
        resp = self._client.get("https://api.coinbase.com/v2/prices/BTC-USD/spot")
        resp.raise_for_status()
        return float(resp.json()["data"]["amount"])

    def _fetch_funding_rate(self) -> float:
        try:
            resp = self._client.get(
                "https://fapi.binance.com/fapi/v1/premiumIndex",
                params={"symbol": "BTCUSDT"},
            )
            resp.raise_for_status()
            return float(resp.json().get("lastFundingRate", 0.0))
        except Exception as exc:
            logger.debug("funding rate fetch failed: %s", exc)
            return 0.0

    def _fetch_binance_klines(self, limit: int = 60) -> np.ndarray:
        resp = self._client.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "1m", "limit": limit},
        )
        resp.raise_for_status()
        rows = resp.json()
        return np.array([float(r[4]) for r in rows], dtype=float)

    def _fetch_kraken_closes(self, limit: int = 60) -> np.ndarray:
        resp = self._client.get(
            "https://api.kraken.com/0/public/OHLC",
            params={"pair": "XBTUSD", "interval": 1},
        )
        resp.raise_for_status()
        for key, val in (resp.json().get("result") or {}).items():
            if key == "last":
                continue
            rows = val[-limit:]
            return np.array([float(r[4]) for r in rows], dtype=float)
        raise RuntimeError("kraken ohlc empty")

    @staticmethod
    def _compute_vwap(closes: np.ndarray) -> float:
        if len(closes) == 0:
            return 0.0
        # Proxy VWAP with volume-weighted closes (equal weight when volume unavailable)
        return float(np.mean(closes))

    @staticmethod
    def _momentum(closes: np.ndarray, bars: int) -> float:
        if len(closes) < bars + 1:
            return 0.0
        prev = closes[-1 - bars]
        if prev <= 0:
            return 0.0
        return float((closes[-1] - prev) / prev)

    @staticmethod
    def _estimate_vol(closes: np.ndarray) -> float:
        if len(closes) < 5:
            return 0.50
        rets = np.diff(np.log(closes))
        per_min = float(np.std(rets))
        annual = per_min * math.sqrt(525600)  # minutes per year
        return max(annual, 0.05)


class SyntheticPriceGenerator:
    """Generate synthetic hourly BTC paths for backtesting."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.regimes = [0.30, 0.50, 0.80, 1.20]
        self.current_regime = 0.50
        self.funding = 0.0001

    def next_hour_path(
        self,
        *,
        spot0: float,
        n_steps: int = 3600,
        dt: float = 1.0,
    ) -> tuple[np.ndarray, float, float]:
        """Return (price path, final funding rate, vwap)."""
        if self.rng.random() < 0.05:
            self.current_regime = float(self.rng.choice(self.regimes))
        vol = self.current_regime
        mu = self.rng.normal(0, 0.00002)
        prices = np.empty(n_steps + 1)
        prices[0] = spot0
        for i in range(1, n_steps + 1):
            # Mean reversion toward VWAP proxy (running mean)
            vwap = float(np.mean(prices[:i]))
            mr = -0.0001 * (prices[i - 1] - vwap) / vwap
            shock = self.rng.normal(0, vol * math.sqrt(dt / (365.25 * 24 * 3600)))
            prices[i] = prices[i - 1] * math.exp(mu + mr + shock)
        self.funding = float(np.clip(self.funding + self.rng.normal(0, 0.00002), -0.001, 0.001))
        return prices, self.funding, float(np.mean(prices))
