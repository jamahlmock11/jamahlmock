"""Kalshi hourly card strike selection — the 3 brackets shown around spot."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from kalshi_btc_1hr_bot import config


def select_kalshi_card_markets(
    markets: list[dict[str, Any]],
    spot: float,
    *,
    n: int | None = None,
) -> list[dict[str, Any]]:
    """Return the n strikes on the current hour's Kalshi card (closest to spot).

    Each input row must include: ticker, strike, close_time, secs_left.
    """
    n = n or config.KALSHI_CARD_PICKS
    if not markets or n <= 0:
        return []

    by_close: dict[datetime, list[dict[str, Any]]] = defaultdict(list)
    for row in markets:
        close = row.get("close_time")
        if close is not None:
            by_close[close].append(row)

    if not by_close:
        ranked = sorted(markets, key=lambda m: abs(float(m["strike"]) - spot))
        return ranked[:n]

    # Current trading hour = soonest expiry still inside the bot window
    current_hour = min(by_close.values(), key=lambda group: min(m["secs_left"] for m in group))
    ranked = sorted(current_hour, key=lambda m: abs(float(m["strike"]) - spot))
    return ranked[:n]


def kalshi_card_tickers(
    markets: list[dict[str, Any]],
    spot: float,
    *,
    n: int | None = None,
) -> set[str]:
    return {str(m["ticker"]) for m in select_kalshi_card_markets(markets, spot, n=n) if m.get("ticker")}
