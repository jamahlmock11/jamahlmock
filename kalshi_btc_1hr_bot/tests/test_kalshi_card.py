"""Kalshi card strike selection tests."""

from __future__ import annotations

from datetime import datetime, timezone

from kalshi_btc_1hr_bot.kalshi_card import select_kalshi_card_markets


def _row(ticker: str, strike: float, secs: float, close: datetime) -> dict:
    return {
        "ticker": ticker,
        "strike": strike,
        "secs_left": secs,
        "close_time": close,
        "market": {},
    }


def test_select_kalshi_card_picks_three_nearest_to_spot():
    close = datetime(2026, 8, 21, 16, 0, tzinfo=timezone.utc)
    markets = [
        _row("A", 77599.99, 1500, close),
        _row("B", 77699.99, 1500, close),
        _row("C", 77499.99, 1500, close),
        _row("D", 77799.99, 1500, close),
        _row("E", 77399.99, 1500, close),
    ]
    spot = 77635.0
    card = select_kalshi_card_markets(markets, spot, n=3)
    assert [m["ticker"] for m in card] == ["A", "B", "C"]


def test_select_kalshi_card_uses_current_hour_bucket():
    close1 = datetime(2026, 8, 21, 16, 0, tzinfo=timezone.utc)
    close2 = datetime(2026, 8, 21, 17, 0, tzinfo=timezone.utc)
    markets = [
        _row("near", 65000, 800, close1),
        _row("far-hour", 65000, 4000, close2),
    ]
    card = select_kalshi_card_markets(markets, 65000, n=3)
    assert len(card) == 1
    assert card[0]["ticker"] == "near"
