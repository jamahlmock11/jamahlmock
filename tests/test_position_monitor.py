"""Tests for pre-expiry position exit logic."""

from __future__ import annotations

from kalshi_bot.execution.position_monitor import (
    drawdown_pct,
    net_exit_edge_dollars,
    parse_open_positions,
    should_exit_position,
)


def test_parse_open_positions_yes_and_no():
    payload = {
        "market_positions": [
            {
                "ticker": "KXBTC15M-26AUG092230-30",
                "position_fp": "2.00",
                "market_exposure_dollars": "0.80",
            },
            {
                "ticker": "KXBTCD-26AUG0923-T65000",
                "position_fp": "-1.00",
                "market_exposure_dollars": "0.45",
            },
            {
                "ticker": "KXBTC15M-flat",
                "position_fp": "0.00",
                "market_exposure_dollars": "0.00",
            },
        ]
    }
    positions = parse_open_positions(payload)
    assert len(positions) == 2
    assert positions[0].side == "yes"
    assert positions[0].contracts == 2
    assert positions[0].avg_entry_price == 0.4
    assert positions[1].side == "no"
    assert positions[1].contracts == 1


def test_drawdown_pct_triggers_at_45_percent():
    # Bought YES at 40¢, now bid 22¢ → 45% loss
    assert drawdown_pct(avg_entry_price=0.40, exit_bid=0.22) == 0.45


def test_drawdown_pct_zero_when_profitable():
    assert drawdown_pct(avg_entry_price=0.40, exit_bid=0.50) == 0.0


def test_should_exit_on_drawdown():
    ok, reason = should_exit_position(
        drawdown=0.46,
        net_edge=0.05,
        max_drawdown_pct=0.45,
        exit_on_edge_flip=True,
    )
    assert ok is True
    assert "drawdown" in reason


def test_should_exit_on_edge_flip():
    ok, reason = should_exit_position(
        drawdown=0.10,
        net_edge=-0.02,
        max_drawdown_pct=0.45,
        exit_on_edge_flip=True,
    )
    assert ok is True
    assert "edge flipped" in reason


def test_should_hold_when_edge_positive_and_drawdown_small():
    ok, _ = should_exit_position(
        drawdown=0.10,
        net_edge=0.03,
        max_drawdown_pct=0.45,
        exit_on_edge_flip=True,
    )
    assert ok is False


def test_net_exit_edge_negative_when_model_below_bid():
    # Model says 30¢ fair, can only sell at 35¢ bid → negative net after fees
    net = net_exit_edge_dollars(model_prob_side=0.30, exit_bid=0.35)
    assert net < 0
