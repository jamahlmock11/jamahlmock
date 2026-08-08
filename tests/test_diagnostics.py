"""Tests for diagnostic system and rejection codes."""

from datetime import datetime, timedelta, timezone

import pytest

from kalshi_bot.config import TierEdgeConfig, V6Config
from kalshi_bot.strategy.decision_record import pick_primary_rejection
from kalshi_bot.strategy.rejection_codes import RejectionCode
from kalshi_bot.strategy.tiered_edge import SetupTier, classify_tier, estimate_slippage
from kalshi_bot.strategy.v6_evaluator import _evaluate_side
from kalshi_bot.strategy.v6_upgrades import V6IntelligenceEngine


def test_rejection_code_priority():
    codes = [RejectionCode.EDGE_TOO_SMALL, RejectionCode.MODEL_CONFLICT]
    assert pick_primary_rejection(codes) == RejectionCode.MODEL_CONFLICT


def test_evaluate_side_uses_executable_ask():
    side = _evaluate_side(
        side="YES",
        model_prob=0.67,
        ask=0.54,
        min_edge=0.20,
        spread=0.02,
        liquidity_score=0.5,
    )
    assert side.raw_edge_dollars == pytest.approx(0.13)
    assert RejectionCode.EDGE_TOO_SMALL in side.rejection_codes
    assert side.executable_ask == 0.54


def test_evaluate_side_net_edge_includes_fees():
    side = _evaluate_side(
        side="YES",
        model_prob=0.80,
        ask=0.50,
        min_edge=0.20,
        spread=0.04,
        liquidity_score=0.5,
    )
    assert side.raw_edge_dollars == pytest.approx(0.30)
    assert side.estimated_fee > 0
    assert side.net_edge_dollars < side.raw_edge_dollars


def test_tier_classification():
    cfg = TierEdgeConfig()
    a_plus = classify_tier(
        net_edge_dollars=0.25,
        model_confidence=0.80,
        data_fresh=True,
        liquidity_ok=True,
        spread_ok=True,
        model_agrees=True,
        no_conflicts=True,
        config=cfg,
    )
    assert a_plus.tier == SetupTier.A_PLUS

    below = classify_tier(
        net_edge_dollars=0.08,
        model_confidence=0.80,
        data_fresh=True,
        liquidity_ok=True,
        spread_ok=True,
        model_agrees=True,
        no_conflicts=True,
        config=cfg,
    )
    assert below.tier == SetupTier.NONE


def test_audited_evaluation_produces_record():
    cfg = V6Config(require_pattern_evidence=False)
    engine = V6IntelligenceEngine(cfg)
    close = datetime.now(timezone.utc) + timedelta(minutes=8)
    open_t = datetime.now(timezone.utc) - timedelta(minutes=7)
    market = {
        "ticker": "KXBTC15M-TEST",
        "series_ticker": "KXBTC15M",
        "strike": 65000.0,
        "close_time": close,
        "open_time": open_t,
        "yes_bid": 0.48,
        "yes_ask": 0.50,
        "no_ask": 0.52,
    }
    for _ in range(30):
        engine.update_spot(65100.0)
    decision = engine.evaluate(
        market,
        spot=65100.0,
        vol=0.55,
        options_prob=0.58,
        spot_source="test",
        spot_is_official=True,
        record_diagnostics=False,
    )
    assert decision.audit_record is not None
    assert decision.audit_record.yes_side.executable_ask == 0.50
    assert decision.audit_record.no_side.executable_ask == 0.52
    assert decision.audit_record.primary_rejection != RejectionCode.NONE or decision.verdict != "NO_TRADE"
    text = decision.audit_record.summary_text()
    assert "MARKET:" in text
    assert "EDGE" in text


def test_slippage_estimate_positive():
    slip = estimate_slippage(spread=0.04, liquidity_score=0.3)
    assert slip > 0
