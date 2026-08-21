"""Tests for Kalshi Pro depth / order-flow analytics."""

from __future__ import annotations

from kalshi_btc_1hr_bot.orderbook_analytics import (
    build_pro_analytics,
    implied_ask_ladder,
    order_book_imbalance,
    parse_orderbook_levels,
    pro_analytics_to_dict,
)


def _sample_book() -> dict:
    return {
        "orderbook_fp": {
            "yes_dollars": [["0.5200", "100"], ["0.5100", "200"], ["0.5000", "50"]],
            "no_dollars": [["0.4600", "80"], ["0.4500", "120"], ["0.4400", "60"]],
        }
    }


def test_parse_orderbook_levels():
    yes, no = parse_orderbook_levels(_sample_book())
    assert yes[0] == (0.52, 100)
    assert no[0] == (0.46, 80)


def test_implied_ask_ladder():
    _, no = parse_orderbook_levels(_sample_book())
    asks = implied_ask_ladder(no)
    assert asks[0][0] == 0.54  # 1 - 0.46


def test_build_pro_analytics_execution_score():
    pro = build_pro_analytics("T", _sample_book(), side="yes", source="rest", latency_ms=100)
    assert pro.yes_bid == 0.52
    assert pro.yes_ask == 0.54
    assert pro.execution_score >= 50
    payload = pro_analytics_to_dict(pro)
    assert payload["execution"]["score"] >= 50
    assert payload["top_of_book"]["yes_ask_cents"] == 54


def test_order_book_imbalance():
    yes, no = parse_orderbook_levels(_sample_book())
    obi = order_book_imbalance(yes, no, levels=3)
    assert -1.0 <= obi <= 1.0
