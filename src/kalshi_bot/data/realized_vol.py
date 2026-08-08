"""Short-horizon BTC realized volatility from public spot history."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import httpx
import numpy as np

logger = logging.getLogger(__name__)

SECONDS_PER_YEAR = 365.25 * 24 * 3600
# BTC short windows can print unrealistically low vol in quiet hours.
# Soft floor; ensemble still blends with options IV when available.
MIN_ANNUALIZED_VOL = 0.15
MAX_ANNUALIZED_VOL = 3.0


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


def _kraken_ohlc(pair: str = "XBTUSD", interval_minutes: int = 1, limit: int = 720) -> tuple[float, np.ndarray]:
    """Return (last_close, close_array) from Kraken OHLC."""
    url = "https://api.kraken.com/0/public/OHLC"
    with httpx.Client(timeout=20.0) as client:
        resp = client.get(url, params={"pair": pair, "interval": interval_minutes})
        resp.raise_for_status()
        data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"Kraken OHLC error: {data['error']}")
    result = data.get("result") or {}
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


def _ann_vol_from_closes(closes: np.ndarray, bar_seconds: int) -> tuple[float, int]:
    if len(closes) < 10:
        raise RuntimeError("insufficient OHLC history")
    log_rets = np.diff(np.log(closes))
    clipped = log_rets[np.abs(log_rets) < 0.05]
    if len(clipped) < 8:
        clipped = log_rets
    sigma_bar = float(np.std(clipped, ddof=1))
    annualized = sigma_bar * math.sqrt(SECONDS_PER_YEAR / bar_seconds)
    return annualized, int(len(clipped))


def estimate_realized_vol(
    *,
    horizon_seconds: float,
    bar_minutes: int = 1,
    lookback_bars: int = 180,
) -> RealizedVolEstimate:
    """Estimate annualized and horizon vol from recent BTC closes.

    Blends 1-minute and 5-minute windows, then applies a BTC-aware floor so
    quiet hours do not produce unrealistically low short-horizon risk.
    """
    horizon_seconds = max(float(horizon_seconds), 1.0)
    try:
        spot, closes_1m = _kraken_ohlc(interval_minutes=1, limit=max(lookback_bars, 180) + 5)
        ann_1m, n_1m = _ann_vol_from_closes(closes_1m, 60)
        try:
            _, closes_5m = _kraken_ohlc(interval_minutes=5, limit=288)
            ann_5m, n_5m = _ann_vol_from_closes(closes_5m, 300)
        except Exception:
            ann_5m, n_5m = ann_1m, 0

        # Robust blend: weight recent 1m more, but keep 5m as stabilizer.
        if n_5m >= 20:
            blended = 0.55 * ann_1m + 0.45 * ann_5m
            n_returns = n_1m + n_5m
            source = "kraken_ohlc_1m_5m"
        else:
            blended = ann_1m
            n_returns = n_1m
            source = "kraken_ohlc_1m"

        annualized = float(np.clip(blended, MIN_ANNUALIZED_VOL, MAX_ANNUALIZED_VOL))
        horizon_vol = annualized * math.sqrt(horizon_seconds / SECONDS_PER_YEAR)
        return RealizedVolEstimate(
            annualized_vol=annualized,
            horizon_vol=horizon_vol,
            horizon_seconds=horizon_seconds,
            n_returns=n_returns,
            bar_seconds=bar_minutes * 60,
            source=source,
            spot=spot,
        )
    except Exception as exc:
        logger.warning("realized vol fetch failed: %s — using conservative fallback", exc)
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
