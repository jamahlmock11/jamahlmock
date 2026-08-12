"""Tests for mispricing pipeline (settlement prob → edge → trade filter)."""

from datetime import datetime, timedelta, timezone

import pytest

from kalshi_bot.config import Rules15mConfig, V6Config
from kalshi_bot.data.btc_data_engine import BtcFeedQuote, BtcMarketSnapshot
from kalshi_bot.strategy.mispricing_engine import evaluate_mispricing
from kalshi_bot.strategy.settlement_probability import SettlementProbability
from kalshi_bot.strategy.time_buckets import TimeBucket, classify_time_bucket
from kalshi_bot.strategy.trade_filter import filter_trade
from kalshi_bot.strategy.v6_upgrades import compute_microstructure


def _btc_snapshot(**kwargs) -> BtcMarketSnapshot:
    defaults = dict(
        reference_price=65000.0,
        reference_source="test",
        is_official=True,
        feeds=(BtcFeedQuote("test", 65000.0, 0.0),),
        cross_exchange_agreement=0.95,
        momentum_1m=0.001,
        momentum_3m=0.002,
        momentum_5m=0.003,
        acceleration=0.0001,
        volume_ratio=1.1,
        annualized_vol=0.45,
        data_age_seconds=1.0,
        stale=False,
    )
    defaults.update(kwargs)
    return BtcMarketSnapshot(**defaults)


def _settlement(prob: float = 0.78) -> SettlementProbability:
    return SettlementProbability(
        prob_above_strike=prob,
        prob_below_strike=1.0 - prob,
        raw_prob=prob,
        calibrated=False,
        gbm_prob=prob,
        monte_carlo_prob=prob,
        momentum_adjustment=0.0,
        confidence=0.75,
        disagreement_pp=5.0,
    )


def test_time_bucket_classification():
    assert classify_time_bucket(12 * 60) == TimeBucket.BUCKET_15_10
    assert classify_time_bucket(8 * 60) == TimeBucket.BUCKET_10_7
    assert classify_time_bucket(6 * 60) == TimeBucket.BUCKET_7_5
    assert classify_time_bucket(4 * 60) == TimeBucket.BUCKET_5_3
    assert classify_time_bucket(2 * 60) == TimeBucket.BUCKET_3_1


def test_mispricing_finds_large_edge():
    micro = compute_microstructure(yes_bid=0.58, yes_ask=0.61)
    opp = evaluate_mispricing(
        ticker="KXBTC15M-TEST",
        strike=65000.0,
        seconds_to_expiry=8 * 60,
        settlement=_settlement(0.784),
        yes_ask=0.61,
        no_ask=0.42,
        micro=micro,
        order_flow_label="BULLISH",
        volatility_label="NORMAL",
    )
    assert opp.yes.raw_edge_dollars == pytest.approx(0.174, abs=0.01)
    assert opp.yes.net_edge_dollars > 0.10


def test_trade_filter_buy_yes_on_strong_edge():
    micro = compute_microstructure(yes_bid=0.58, yes_ask=0.61)
    opp = evaluate_mispricing(
        ticker="KXBTC15M-TEST",
        strike=65000.0,
        seconds_to_expiry=8 * 60,
        settlement=_settlement(0.784),
        yes_ask=0.61,
        no_ask=0.42,
        micro=micro,
        order_flow_label="BULLISH",
        volatility_label="NORMAL",
    )
    decision = filter_trade(
        opp,
        btc=_btc_snapshot(),
        micro=micro,
        max_spread=0.08,
        min_liquidity_score=0.10,
        risk_allows=True,
        kelly_contracts=2,
    )
    from kalshi_bot.strategy.mispricing_engine import TradeAction

    assert decision.action == TradeAction.BUY_YES
    assert decision.net_edge_dollars > 0.10


def test_trade_filter_no_trade_insufficient_edge():
    micro = compute_microstructure(yes_bid=0.66, yes_ask=0.68)
    opp = evaluate_mispricing(
        ticker="KXBTC15M-TEST",
        strike=65000.0,
        seconds_to_expiry=8 * 60,
        settlement=_settlement(0.72),
        yes_ask=0.68,
        no_ask=0.34,
        micro=micro,
        order_flow_label="NEUTRAL",
        volatility_label="NORMAL",
    )
    decision = filter_trade(
        opp,
        btc=_btc_snapshot(),
        micro=micro,
        max_spread=0.08,
        min_liquidity_score=0.10,
        risk_allows=True,
        kelly_contracts=1,
    )
    from kalshi_bot.strategy.mispricing_engine import TradeAction

    assert decision.action == TradeAction.NO_TRADE
    assert "INSUFFICIENT EDGE" in decision.reason.upper() or "edge" in decision.reason.lower()


def test_rules_config_mispricing_mode():
    rules = Rules15mConfig.model_validate({"enabled": True, "mode": "mispricing"})
    assert rules.mode == "mispricing"
    assert rules.arbitrary.enabled is False
