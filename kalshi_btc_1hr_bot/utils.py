"""Shared utilities."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

from scipy.stats import norm


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def years_to_expiry(close: datetime, now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    secs = max((close - now).total_seconds(), 1.0)
    return secs / (365.25 * 24 * 3600)


def gbm_prob_above(
    spot: float,
    strike: float,
    time_years: float,
    vol: float,
    drift: float = 0.0,
) -> float:
    """P(S_T > K) under GBM with optional drift."""
    if spot <= 0 or strike <= 0 or time_years <= 0 or vol <= 0:
        return 0.5
    d2 = (math.log(spot / strike) + (drift - 0.5 * vol * vol) * time_years) / (
        vol * math.sqrt(time_years)
    )
    return float(norm.cdf(d2))


def effective_vol_for_averaging(vol: float, window_seconds: int = 60) -> float:
    """Reduce effective vol for 60-second settlement averaging."""
    # Averaging over n prints reduces variance by ~1/sqrt(n)
    n = max(window_seconds, 1)
    return vol / math.sqrt(n)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def clamp_prob(p: float) -> float:
    return min(max(p, 0.001), 0.999)


def quadratic_fee(price: float, fee_rate: float = 0.07) -> float:
    """Kalshi quadratic fee per contract (rounded up to cent)."""
    raw = fee_rate * price * (1.0 - price)
    return math.ceil(raw * 100) / 100.0
