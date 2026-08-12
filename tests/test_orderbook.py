"""Tests for live orderbook cache."""

import time

from kalshi_bot.data.kalshi_orderbook import OrderbookCache


def test_orderbook_snapshot_and_quote():
    cache = OrderbookCache(stale_after=9999.0)
    cache.apply_snapshot(
        "T1",
        {
            "yes": [[55, 10.0], [54, 5.0]],
            "no": [[44, 8.0]],
        },
    )
    quote = cache.quote("T1")
    assert quote is not None
    assert quote.yes_bid == 0.55
    assert quote.yes_ask == 0.56
    assert quote.source == "ws"
    assert quote.stale is False


def test_orderbook_delta_updates_level():
    cache = OrderbookCache(stale_after=9999.0)
    cache.apply_snapshot("T1", {"yes": [[50, 5.0]], "no": []})
    cache.apply_delta(
        "T1",
        {"side": "yes", "price_dollars": "0.51", "delta_fp": "3.0"},
    )
    quote = cache.quote("T1")
    assert quote is not None
    assert quote.yes_bid == 0.51


def test_orderbook_delta_removes_level():
    cache = OrderbookCache(stale_after=9999.0)
    cache.apply_snapshot("T1", {"yes": [[50, 5.0]], "no": []})
    cache.apply_delta("T1", {"side": "yes", "price_dollars": "0.50", "delta_fp": "0"})
    quote = cache.quote("T1")
    assert quote is None


def test_to_orderbook_dict_shape():
    cache = OrderbookCache()
    cache.apply_snapshot("T1", {"yes": [[55, 2.0]], "no": [[44, 1.0]]})
    ob = cache.to_orderbook_dict("T1")
    assert ob is not None
    assert "orderbook" in ob
    assert ob["orderbook"]["yes"][0][0] == 55
