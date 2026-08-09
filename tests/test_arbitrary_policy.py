"""Tests for Arbitrary independent judgment policy."""

from __future__ import annotations

from kalshi_bot.config import ArbitraryPolicyConfig, BotActionConfig
from kalshi_bot.strategy.arbitrary_policy import EdgeChaseGuard, evaluate_arbitrary
from kalshi_bot.strategy.bot_action import BotAction
from kalshi_bot.strategy.probability_calibration import ProbabilityCalibrator


def test_overpriced_favorite_blocks_yes_but_can_trade_no():
    verdict = evaluate_arbitrary(
        ticker="KXBTC15M-TEST",
        model_prob_yes=0.45,
        model_prob_yes_lo=0.43,
        model_prob_yes_hi=0.47,
        yes_ask=0.65,
        yes_bid=0.63,
        no_ask=0.34,
        seconds_to_expiry=600,
        min_seconds_to_expiry=60,
        max_seconds_to_expiry=840,
        base_min_edge_pp=8.0,
        confidence=0.80,
        disagreement_pp=4.0,
        sufficient_evidence=True,
        bot_action_cfg=BotActionConfig(conditional_min_gap_pp=15.0),
        policy_cfg=ArbitraryPolicyConfig(require_calibration_for_conditional=False),
    )
    assert verdict.verdict == "TRADE_NO"
    assert verdict.yes.is_favorite is True
    assert verdict.yes.bot_action == BotAction.NO_TRADE
    assert any("overpriced_favorite" in b for b in verdict.yes.blockers)
    assert "fading overpriced favorite" in " ".join(verdict.reasons)


def test_underpriced_underdog_can_buy_yes():
    verdict = evaluate_arbitrary(
        ticker="KXBTC15M-UNDERDOG",
        model_prob_yes=0.62,
        model_prob_yes_lo=0.60,
        model_prob_yes_hi=0.64,
        yes_ask=0.35,
        yes_bid=0.33,
        no_ask=0.67,
        seconds_to_expiry=300,
        min_seconds_to_expiry=60,
        max_seconds_to_expiry=840,
        base_min_edge_pp=5.0,
        confidence=0.85,
        disagreement_pp=3.0,
        sufficient_evidence=True,
        bot_action_cfg=BotActionConfig(),
        policy_cfg=ArbitraryPolicyConfig(),
    )
    assert verdict.verdict == "TRADE_YES"
    assert verdict.yes.is_underdog is True
    assert any("underpriced underdog" in r for r in verdict.reasons)


def test_edge_chase_guard_blocks_after_price_runs():
    guard = EdgeChaseGuard(ttl_seconds=120.0)
    policy = ArbitraryPolicyConfig(chase_min_gap_decay_pp=2.0, chase_max_ask_rise=0.01)
    kwargs = dict(
        ticker="KXBTC15M-CHASE",
        model_prob_yes=0.62,
        model_prob_yes_lo=0.60,
        model_prob_yes_hi=0.64,
        yes_ask=0.35,
        yes_bid=0.33,
        no_ask=0.67,
        seconds_to_expiry=300,
        min_seconds_to_expiry=60,
        max_seconds_to_expiry=840,
        base_min_edge_pp=5.0,
        confidence=0.85,
        disagreement_pp=3.0,
        sufficient_evidence=True,
        chase_guard=guard,
        policy_cfg=policy,
    )
    first = evaluate_arbitrary(**kwargs)
    assert first.verdict == "TRADE_YES"
    kwargs["yes_ask"] = 0.40
    second = evaluate_arbitrary(**kwargs)
    assert second.verdict == "NO_TRADE"
    assert second.chase_blocked is True


def test_uncalibrated_model_shrinks_toward_fifty_fifty():
    calibrator = ProbabilityCalibrator(min_trades_per_bucket=3)
    verdict = evaluate_arbitrary(
        ticker="KXBTC15M-UNCAL",
        model_prob_yes=0.70,
        model_prob_yes_lo=0.68,
        model_prob_yes_hi=0.72,
        yes_ask=0.60,
        yes_bid=0.58,
        no_ask=0.42,
        seconds_to_expiry=300,
        min_seconds_to_expiry=60,
        max_seconds_to_expiry=840,
        base_min_edge_pp=5.0,
        confidence=0.85,
        disagreement_pp=3.0,
        sufficient_evidence=True,
        calibrator=calibrator,
        policy_cfg=ArbitraryPolicyConfig(uncalibrated_shrink=0.5),
    )
    assert verdict.calibrated is False
    assert verdict.yes.model_probability < 0.70
