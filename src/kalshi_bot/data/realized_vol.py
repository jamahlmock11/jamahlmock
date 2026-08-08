"""Short-horizon BTC realized volatility from public spot history."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import httpx
import numpy as np

logger = logging.getLogger(__name__)

SECONDS_PER_YEAR = 365.25 * 24 * 3600


@dataclass(frozen=True)
class RealizedVolEstimate:
    """Annualized realized vol estimate for short-horizon forecasting."""

    annualized_vol: float
    horizon_vol: float
    horizon_seconds: float
    n_returns: int
    bar_seconds: int
    source: str
    spot: float

    @property
    def is_reliable(self) -> bool:
        return self.n_returns >= 30 and self.annualized_vol > 0.01


def _kraken_ohlc(pair: str = "XBTUSD", interval_minutes: int = 1, limit: int = 240) -> tuple[float, np.ndarray]:
    """Return (last_close, close_array) from Kraken OHLC."""
    url = "https://api.kraken.com/0/public/OHLC"
    with httpx.Client(timeout=20.0) as client:
        resp = client.get(url, params={"pair": pair, "interval": interval_minutes})
        resp.raise_for_status()
        data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"Kraken OHLC error: {data['error']}")
    result = data.get("result") or {}
    # Key is pair-dependent (XXBTZUSD etc.)
    series = None
    for key, value in result.items():
        if key == "last":
            continue
        series = value
        break
    if not series:
        raise RuntimeError("Kraken OHLC empty")
    closes = np.array([float(row[4]) for row in series[-limit:]], dtype=float)
    return float(closes[-1]), closes


def estimate_realized_vol(
    *,
    horizon_seconds: float,
    bar_minutes: int = 1,
    lookback_bars: int = 180,
) -> RealizedVolEstimate:
    """Estimate annualized and horizon vol from recent 1-minute BTC closes.

    Uses log-return sample std, scaled by sqrt(seconds/year).
    Conservative floor avoids near-zero vol in thin tapes.
    """
    horizon_seconds = max(float(horizon_seconds), 1.0)
    try:
        spot, closes = _kraken_ohlc(interval_minutes=bar_minutes, limit=lookback_bars + 5)
        if len(closes) < 10:
            raise RuntimeError("insufficient OHLC history")
        log_rets = np.diff(np.log(closes))
        # Drop pathological spikes (>5% in 1m) from sample
        clipped = log_rets[np.abs(log_rets) < 0.05]
        if len(clipped) < 8:
            clipped = log_rets
        bar_seconds = bar_minutes * 60
        sigma_bar = float(np.std(clipped, ddof=1))
        annualized = sigma_bar * math.sqrt(SECONDS_PER_YEAR / bar_seconds)
        annualized = float(np.clip(annualized, 0.05, 3.0))
        horizon_vol = annualized * math.sqrt(horizon_seconds / SECONDS_PER_YEAR)
        return RealizedVolEstimate(
            annualized_vol=annualized,
            horizon_vol=horizon_vol,
            horizon_seconds=horizon_seconds,
            n_returns=int(len(clipped)),
            bar_seconds=bar_seconds,
            source="kraken_ohlc",
            spot=spot,
        )
    except Exception as exc:
        logger.warning("realized vol fetch failed: %s — using conservative fallback", exc)
        # Weekend/offline fallback: mid BTC realized (~45% ann)
        annualized = 0.45
        return RealizedVolEstimate(
            annualized_vol=annualized,
            horizon_vol=annualized * math.sqrt(horizon_seconds / SECONDS_PER_YEAR),
            horizon_seconds=horizon_seconds,
            n_returns=0,
            bar_seconds=bar_minutes * 60,
            source="fallback_conservative",
            spot=0.0,
        )
