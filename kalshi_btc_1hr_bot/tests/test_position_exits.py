"""Tests for take-profit / stop-loss exit logic."""

from __future__ import annotations

from kalshi_btc_1hr_bot.config import ExitConfig
from kalshi_btc_1hr_bot.position_exits import (
    bid_for_side,
    compute_exit_levels,
    evaluate_exit,
)


def test_compute_exit_levels():
    cfg = ExitConfig(take_profit_pct=0.50, stop_loss_pct=0.40)
    lv = compute_exit_levels(0.40, cfg)
    assert lv.take_profit_price == 0.6
    assert lv.stop_loss_price == 0.24


def test_evaluate_take_profit():
    cfg = ExitConfig(enabled=True, take_profit_pct=0.50, stop_loss_pct=0.40)
    lv = compute_exit_levels(0.40, cfg)
    signal = evaluate_exit(
        entry_price=0.40,
        bid_price=0.61,
        contracts=2,
        cfg=cfg,
        levels=lv,
    )
    assert signal is not None
    assert signal.reason == "take_profit"
    assert abs(signal.pnl_total - 0.42) < 1e-9


def test_evaluate_stop_loss():
    cfg = ExitConfig(enabled=True, take_profit_pct=0.50, stop_loss_pct=0.40)
    lv = compute_exit_levels(0.40, cfg)
    signal = evaluate_exit(
        entry_price=0.40,
        bid_price=0.20,
        contracts=1,
        cfg=cfg,
        levels=lv,
    )
    assert signal is not None
    assert signal.reason == "stop_loss"
    assert signal.pnl_total == -0.20


def test_evaluate_no_signal_in_band():
    cfg = ExitConfig(enabled=True, take_profit_pct=0.50, stop_loss_pct=0.40)
    assert evaluate_exit(entry_price=0.40, bid_price=0.45, contracts=1, cfg=cfg) is None


def test_bid_for_side_yes():
    market = {"yes_bid": 0.55, "yes_ask": 0.58}
    assert bid_for_side(market, "yes") == 0.55


def test_bid_for_side_no_fallback():
    market = {"no_ask": 0.62}
    assert bid_for_side(market, "no") == 0.61
