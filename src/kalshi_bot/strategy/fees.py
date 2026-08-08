"""Kalshi fee model (quadratic) for edge-after-cost."""

from __future__ import annotations

import math


def quadratic_fee_per_contract(
    price: float,
    *,
    fee_rate: float = 0.07,
    fee_multiplier: float = 1.0,
) -> float:
    """Expected Kalshi quadratic taker fee for one contract at price P.

    fee ≈ ceil_cent(fee_rate * fee_multiplier * P * (1-P))
    """
    if price <= 0 or price >= 1:
        return 0.0
    raw = fee_rate * fee_multiplier * price * (1.0 - price)
    # Round up to next cent
    return math.ceil(raw * 100 - 1e-12) / 100.0


def total_fees(price: float, contracts: int, fee_rate: float = 0.07, fee_multiplier: float = 1.0) -> float:
    return contracts * quadratic_fee_per_contract(
        price, fee_rate=fee_rate, fee_multiplier=fee_multiplier
    )
