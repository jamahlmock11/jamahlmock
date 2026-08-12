"""Guards when 15m trading rules are disabled or misconfigured."""

from __future__ import annotations

from datetime import datetime

from kalshi_bot.strategy.decision_record import (
    DataFreshness,
    FilterCheck,
    MarketEvaluationRecord,
    SideEvaluation,
)
from kalshi_bot.strategy.rejection_codes import RejectionCode


def _blank_side(side: str) -> SideEvaluation:
    return SideEvaluation(
        side=side,
        model_probability=0.5,
        executable_ask=None,
        raw_edge_dollars=0.0,
        estimated_fee=0.0,
        estimated_slippage=0.0,
        net_edge_dollars=0.0,
        expected_value_per_contract=0.0,
        passes_edge_threshold=False,
        passes_net_ev=False,
        rejection_codes=[RejectionCode.RULES_NOT_CONFIGURED],
    )


def rules_not_configured_record(
    *,
    ticker: str,
    series: str,
    now: datetime,
    secs: float,
    mins: float,
    spot: float,
    spot_source: str,
    strike: float,
    filter_checks: list[FilterCheck],
) -> MarketEvaluationRecord:
    yes_side = _blank_side("YES")
    no_side = _blank_side("NO")
    filter_checks.append(
        FilterCheck(
            "rules",
            False,
            RejectionCode.RULES_NOT_CONFIGURED,
            "15-minute bot trading rules are disabled",
        )
    )
    return MarketEvaluationRecord(
        ticker=ticker,
        series=series,
        evaluated_at=now,
        seconds_to_expiry=secs,
        minutes_to_expiry=mins,
        spot=spot,
        spot_source=spot_source,
        strike=strike,
        model_prob_up=0.5,
        model_prob_down=0.5,
        model_confidence=0.0,
        model_disagreement_pp=0.0,
        monte_carlo_prob=0.5,
        options_implied_prob=None,
        calibrated=False,
        yes_bid=None,
        yes_ask=None,
        no_ask=None,
        spread=0.0,
        yes_side=yes_side,
        no_side=no_side,
        best_side=None,
        best_net_edge=0.0,
        liquidity_score=0.0,
        bid_ask_imbalance=0.0,
        order_book_depth_bid=0.0,
        order_book_depth_ask=0.0,
        data_freshness=DataFreshness(
            cf_benchmark="MISSING",
            btc_spot="MISSING",
            order_book="MISSING",
            options_smile="MISSING",
        ),
        filter_checks=filter_checks,
        all_rejection_codes=[RejectionCode.RULES_NOT_CONFIGURED],
        setup_tier="NONE",
        opportunity_score=0.0,
        verdict="NO_TRADE",
        primary_rejection=RejectionCode.RULES_NOT_CONFIGURED,
        contracts=0,
        regime="",
        explainability=0.0,
        edge_quality="NO_TRADE",
        edge_action="🔴 No trade",
        trade_reason="Configure config/rules_15m.yaml and set enabled: true",
    )
