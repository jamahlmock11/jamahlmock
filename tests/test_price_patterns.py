"""Tests for price patterns and independent YES/NO trade filter."""

from kalshi_bot.data.btc_data_engine import BtcFeedQuote, BtcMarketSnapshot
from kalshi_bot.strategy.mispricing_engine import MispricingOpportunity, SideMispricing, TradeAction
from kalshi_bot.strategy.settlement_probability import SettlementProbability
from kalshi_bot.strategy.price_patterns import PricePattern, detect_price_pattern
from kalshi_bot.strategy.trade_filter import filter_trade
from kalshi_bot.strategy.v6_upgrades import MicrostructureSnapshot


def _btc(**kwargs) -> BtcMarketSnapshot:
    defaults = dict(
        reference_price=65000.0,
        reference_source="test",
        is_official=True,
        feeds=(),
        cross_exchange_agreement=0.9,
        momentum_1m=0.002,
        momentum_3m=0.0015,
        momentum_5m=0.001,
        acceleration=0.0001,
        volume_ratio=1.0,
        annualized_vol=0.4,
        data_age_seconds=1.0,
        stale=False,
    )
    defaults.update(kwargs)
    return BtcMarketSnapshot(**defaults)


def _settlement(prob_yes: float = 0.30) -> SettlementProbability:
    return SettlementProbability(
        prob_above_strike=prob_yes,
        prob_below_strike=1.0 - prob_yes,
        raw_prob=prob_yes,
        calibrated=False,
        gbm_prob=prob_yes,
        monte_carlo_prob=prob_yes,
        momentum_adjustment=0.0,
        confidence=0.75,
        disagreement_pp=2.0,
    )


def _side(side: str, prob: float, ask: float, net: float) -> SideMispricing:
    return SideMispricing(
        side=side,
        model_probability=prob,
        fair_value=prob,
        executable_ask=ask,
        spread=0.02,
        fee=0.02,
        slippage=0.01,
        raw_edge_dollars=net + 0.03,
        net_edge_dollars=net,
        expected_value=net,
    )


def test_detect_drift_pattern():
    btc = _btc(momentum_1m=0.002, momentum_3m=0.0018, acceleration=0.0001)
    p = detect_price_pattern(btc, spot=65100, strike=65000, seconds_to_expiry=600)
    assert p.pattern == PricePattern.DRIFT


def test_filter_takes_no_when_yes_lacks_edge():
    micro = MicrostructureSnapshot(
        bid_ask_imbalance=0.0,
        depth_bid_10=100,
        depth_ask_10=100,
        whale_detected=False,
        whale_side=None,
        cancel_to_new_ratio=0.5,
        spread=0.02,
        spread_change=0.0,
        trades_per_second=0.2,
        liquidity_score=0.4,
    )
    opp = MispricingOpportunity(
        ticker="T",
        strike=65000,
        seconds_to_expiry=600,
        settlement=_settlement(0.25),
        yes=_side("YES", 0.25, 0.30, 0.02),  # insufficient
        no=_side("NO", 0.75, 0.68, 0.20),  # qualifies
        best_side="NO",
        best_net_edge=0.20,
        kalshi_stale=False,
        liquidity_label="GOOD",
        order_flow_label="NEUTRAL",
        volatility_label="NORMAL",
        confidence_label="HIGH",
    )
    dec = filter_trade(
        opp,
        btc=_btc(),
        micro=micro,
        max_spread=0.08,
        min_liquidity_score=0.15,
        bucket_overrides={"15_10": {"min_net_edge_dollars": 0.18, "min_raw_edge_dollars": 0.22}},
        kelly_contracts=1,
    )
    assert dec.action == TradeAction.BUY_NO
    assert dec.side == "NO"
    assert dec.executable_price == 0.68
