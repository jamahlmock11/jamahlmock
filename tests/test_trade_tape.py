"""Tests for Kalshi trade tape buffer and service."""

import time
from unittest.mock import MagicMock

from kalshi_bot.data.kalshi_trade_tape import (
    KalshiTradeTapeService,
    TradeTapeBuffer,
    normalize_trade,
)


def test_normalize_trade_yes_price():
    raw = {
        "ticker": "KXBTC15M-TEST",
        "yes_price": 62,
        "taker_side": "yes",
        "count": 5,
        "created_time": "2026-01-01T12:00:00Z",
    }
    t = normalize_trade(raw)
    assert t["ticker"] == "KXBTC15M-TEST"
    assert t["yes_price"] == 0.62
    assert t["side"] == "yes"
    assert t["quantity"] == 5.0


def test_buffer_stats_buy_pressure():
    buf = TradeTapeBuffer()
    now = time.time()
    for i in range(5):
        buf.add(
            "T1",
            {"ts": now - 10 + i, "yes_price": 0.5, "side": "yes", "quantity": 2.0},
        )
    for i in range(3):
        buf.add(
            "T1",
            {"ts": now - 5 + i, "yes_price": 0.51, "side": "no", "quantity": 1.0},
        )
    stats = buf.stats("T1", stale_after=9999.0)
    assert stats.trades_per_second > 0
    assert stats.buy_pressure > 0  # more yes volume
    assert stats.volume_1m > 0
    assert stats.last_price == 0.51


def test_service_refresh_from_rest():
    import time

    client = MagicMock()
    now = time.time()
    client.iter_trades.return_value = [
        {
            "ticker": "KXBTC15M-A",
            "yes_price": 55,
            "taker_side": "no",
            "count": 1,
            "created_time": now,
        }
    ]
    svc = KalshiTradeTapeService(client)
    trades = svc.refresh_ticker("KXBTC15M-A")
    assert len(trades) == 1
    assert svc.recent_trades("KXBTC15M-A")
    assert client.iter_trades.call_count >= 1
