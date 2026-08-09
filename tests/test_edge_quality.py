"""Tests for edge quality tier system."""

import pytest

from kalshi_bot.config import TierEdgeConfig
from kalshi_bot.strategy.tiered_edge import (
    EdgeQuality,
    EDGE_ACTION_LABEL,
    classify_edge_quality,
    confirmation_passes,
    should_trade_for_quality,
)


@pytest.mark.parametrize(
    "net_dollars,expected",
    [
        (0.30, EdgeQuality.EXCEPTIONAL),
        (0.17, EdgeQuality.STRONG),
        (0.12, EdgeQuality.CONDITIONAL),
        (0.06, EdgeQuality.EXPERIMENTAL),
        (0.04, EdgeQuality.NO_TRADE),
    ],
)
def test_edge_quality_bands(net_dollars, expected):
    result = classify_edge_quality(net_dollars)
    assert result.quality == expected
    assert result.action_label == EDGE_ACTION_LABEL[expected]


def test_exceptional_trades_without_confirmation():
    edge = classify_edge_quality(0.25)
    ok, reason = should_trade_for_quality(
        edge,
        model_confidence=0.5,
        model_agrees=False,
        data_fresh=True,
        liquidity_ok=True,
        spread_ok=True,
        no_manipulation=True,
        net_ev_positive=True,
        net_edge_dollars=0.20,
    )
    assert ok
    assert edge.quality == EdgeQuality.EXCEPTIONAL


def test_conditional_requires_confirmation():
    edge = classify_edge_quality(0.12)
    assert edge.requires_confirmation
    ok, _ = should_trade_for_quality(
        edge,
        model_confidence=0.45,
        model_agrees=False,
        data_fresh=True,
        liquidity_ok=True,
        spread_ok=True,
        no_manipulation=True,
        net_ev_positive=True,
        net_edge_dollars=0.08,
    )
    assert not ok
    ok2, _ = should_trade_for_quality(
        edge,
        model_confidence=0.80,
        model_agrees=True,
        data_fresh=True,
        liquidity_ok=True,
        spread_ok=True,
        no_manipulation=True,
        net_ev_positive=True,
        net_edge_dollars=0.08,
    )
    assert ok2


def test_experimental_smaller_size_multiplier():
    edge = classify_edge_quality(0.06)
    assert edge.quality == EdgeQuality.EXPERIMENTAL
    assert edge.size_multiplier == 0.50


def test_no_trade_below_5_cents():
    edge = classify_edge_quality(0.04)
    assert not edge.trades_allowed
    ok, _ = should_trade_for_quality(
        edge,
        model_confidence=0.9,
        model_agrees=True,
        data_fresh=True,
        liquidity_ok=True,
        spread_ok=True,
        no_manipulation=True,
        net_ev_positive=True,
        net_edge_dollars=0.02,
    )
    assert not ok
