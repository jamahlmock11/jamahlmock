"""Mispricing-based market evaluation for KXBTC15M."""

from __future__ import annotations

from datetime import datetime, timezone

from kalshi_bot.config import Rules15mConfig, V6Config
from kalshi_bot.data.btc_data_engine import BtcDataEngine, BtcMarketSnapshot
from kalshi_bot.strategy.decision_record import (
    DataFreshness,
    FilterCheck,
    MarketEvaluationRecord,
    SideEvaluation,
    pick_primary_rejection,
)
from kalshi_bot.strategy.mispricing_engine import MispricingOpportunity, TradeAction, evaluate_mispricing
from kalshi_bot.strategy.rejection_codes import RejectionCode
from kalshi_bot.strategy.settlement_probability import estimate_settlement_probability
from kalshi_bot.strategy.time_buckets import TimeBucket, classify_time_bucket
from kalshi_bot.strategy.trade_filter import TradeDecision, filter_trade
from kalshi_bot.strategy.v6_upgrades import (
    V6IntelligenceEngine,
    compute_microstructure,
    compute_price_action,
    detect_regime,
)


def _side_eval(side_mispricing, rejection: RejectionCode) -> SideEvaluation:
    m = side_mispricing
    codes: list[RejectionCode] = []
    if m.executable_ask is None:
        codes.append(RejectionCode.MISSING_DATA)
    elif m.net_edge_dollars <= 0:
        codes.append(RejectionCode.EXPECTED_VALUE_NEGATIVE)
    if rejection != RejectionCode.NONE:
        codes.append(rejection)
    return SideEvaluation(
        side=m.side,
        model_probability=m.model_probability,
        executable_ask=m.executable_ask,
        raw_edge_dollars=m.raw_edge_dollars,
        estimated_fee=m.fee,
        estimated_slippage=m.slippage,
        net_edge_dollars=m.net_edge_dollars,
        expected_value_per_contract=m.expected_value,
        passes_edge_threshold=m.raw_edge_dollars > 0,
        passes_net_ev=m.net_edge_dollars > 0,
        rejection_codes=codes,
    )


def _verdict_from_action(action: TradeAction, side: str | None) -> str:
    if action == TradeAction.BUY_YES:
        return "TRADE_YES"
    if action == TradeAction.BUY_NO:
        return "TRADE_NO"
    return "NO_TRADE"


def evaluate_market_mispricing(
    engine: V6IntelligenceEngine,
    market: dict,
    *,
    spot: float,
    spot_source: str,
    spot_is_official: bool,
    vol: float,
    btc: BtcMarketSnapshot,
    options_prob: float | None = None,
    now: datetime | None = None,
    fee_rate: float = 0.07,
    recent_trades: list[dict] | None = None,
    kalshi_stale: bool = False,
    orderbook: dict | None = None,
    orderbook_source: str = "rest",
) -> tuple[MarketEvaluationRecord, MispricingOpportunity, TradeDecision]:
    """Full mispricing pipeline for one KXBTC15M market."""
    config: V6Config = engine.config
    rules: Rules15mConfig = engine.rules
    now = now or datetime.now(timezone.utc)
    engine.update_spot(spot)

    ticker = str(market.get("ticker") or "")
    series = str(market.get("series_ticker") or config.series_ticker)
    strike = float(market.get("strike") or spot)
    close = market.get("close_time")
    open_t = market.get("open_time")
    yes_bid = market.get("yes_bid")
    yes_ask = market.get("yes_ask")
    no_ask = market.get("no_ask")
    if no_ask is None and yes_bid is not None:
        no_ask = max(0.0, 1.0 - yes_bid)

    secs = max((close - now).total_seconds(), 0) if close else 0
    mins = secs / 60.0
    filter_checks: list[FilterCheck] = []
    all_rejections: list[RejectionCode] = []

    orderbook_data = orderbook
    ob_status = "MISSING"
    if orderbook_data is not None:
        ob_status = orderbook_source.upper()
    elif engine.client and ticker:
        try:
            orderbook_data = engine.client.get_orderbook(ticker, depth=10)
            ob_status = "REST"
        except Exception:
            ob_status = "MISSING"

    micro = compute_microstructure(
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        orderbook=orderbook_data,
        prev_spread=engine._prev_spread,
        prev_depth=engine._prev_depth,
        recent_trades=recent_trades,
    )
    engine._prev_spread = micro.spread
    engine._prev_depth = (micro.depth_bid_10, micro.depth_ask_10)
    pa = compute_price_action(list(engine._price_history))
    regime = detect_regime(pa, micro)

    if kalshi_stale:
        kalshi_stale_flag = True
    else:
        kalshi_stale_flag = micro.spread <= 0 and yes_ask is None

    settlement = estimate_settlement_probability(
        spot=spot,
        strike=strike,
        seconds_to_expiry=secs,
        annualized_vol=vol,
        btc=btc,
        options_prob=options_prob,
        calibrator=engine.calibrator,
        monte_carlo_sims=config.monte_carlo_sims,
    )

    opp = evaluate_mispricing(
        ticker=ticker,
        strike=strike,
        seconds_to_expiry=secs,
        settlement=settlement,
        yes_ask=yes_ask,
        no_ask=no_ask,
        micro=micro,
        order_flow_label=btc.order_flow_label,
        volatility_label=btc.volatility_label,
        kalshi_stale=kalshi_stale_flag,
        fee_rate=fee_rate,
    )

    allow, risk_reason = engine.risk.allow_trade(settlement.confidence)
    kelly_side = opp.yes if opp.best_side == "YES" else opp.no
    kelly_contracts = 0
    if kelly_side.executable_ask:
        kelly_contracts = engine.risk.kelly_size(
            prob=kelly_side.model_probability,
            price=kelly_side.executable_ask,
            bankroll=config.bankroll_usd,
            confidence=settlement.confidence,
        )

    q = rules.quality
    max_spread = q.max_spread if q.max_spread is not None else 0.08
    min_liq = q.min_liquidity_score if q.min_liquidity_score is not None else 0.15

    decision = filter_trade(
        opp,
        btc=btc,
        micro=micro,
        max_spread=max_spread,
        min_liquidity_score=min_liq,
        bucket_overrides=rules.time_buckets or None,
        min_seconds=config.min_seconds_to_expiry,
        max_seconds=config.max_seconds_to_expiry,
        risk_allows=allow,
        risk_reason=risk_reason,
        kelly_contracts=kelly_contracts,
    )

    verdict = _verdict_from_action(decision.action, decision.side)
    if decision.rejection != RejectionCode.NONE:
        all_rejections.append(decision.rejection)
    filter_checks.append(
        FilterCheck(
            "trade_filter",
            decision.rejection == RejectionCode.NONE,
            decision.rejection if decision.rejection != RejectionCode.NONE else None,
            decision.reason,
        )
    )

    bucket = classify_time_bucket(secs, min_seconds=config.min_seconds_to_expiry, max_seconds=config.max_seconds_to_expiry)
    filter_checks.append(
        FilterCheck(
            "time_bucket",
            bucket not in (TimeBucket.TOO_EARLY, TimeBucket.TOO_LATE),
            detail=bucket.value,
        )
    )

    data_freshness = DataFreshness(
        cf_benchmark="FRESH" if spot_is_official else "PROXY",
        btc_spot="STALE" if btc.stale else "FRESH",
        order_book=ob_status,
        options_smile="FRESH" if options_prob is not None else "MISSING",
    )

    yes_side = _side_eval(opp.yes, decision.rejection if opp.best_side == "YES" else RejectionCode.NONE)
    no_side = _side_eval(opp.no, decision.rejection if opp.best_side == "NO" else RejectionCode.NONE)

    primary = (
        RejectionCode.NONE
        if verdict != "NO_TRADE"
        else pick_primary_rejection(all_rejections) if all_rejections else decision.rejection
    )

    edge_action = decision.action.value
    if decision.action == TradeAction.NO_TRADE:
        edge_action = f"🔴 {decision.reason}"
    elif decision.action == TradeAction.WAIT:
        edge_action = f"🟡 {decision.reason}"

    record = MarketEvaluationRecord(
        ticker=ticker,
        series=series,
        evaluated_at=now,
        seconds_to_expiry=secs,
        minutes_to_expiry=mins,
        spot=spot,
        spot_source=spot_source,
        strike=strike,
        model_prob_up=settlement.prob_above_strike,
        model_prob_down=settlement.prob_below_strike,
        model_confidence=settlement.confidence,
        model_disagreement_pp=settlement.disagreement_pp,
        monte_carlo_prob=settlement.monte_carlo_prob,
        options_implied_prob=options_prob,
        calibrated=settlement.calibrated,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_ask=no_ask,
        spread=micro.spread,
        yes_side=yes_side,
        no_side=no_side,
        best_side=decision.side,
        best_net_edge=opp.best_net_edge,
        liquidity_score=micro.liquidity_score,
        bid_ask_imbalance=micro.bid_ask_imbalance,
        order_book_depth_bid=micro.depth_bid_10,
        order_book_depth_ask=micro.depth_ask_10,
        data_freshness=data_freshness,
        filter_checks=filter_checks,
        all_rejection_codes=list(dict.fromkeys(all_rejections)),
        setup_tier=decision.time_bucket,
        opportunity_score=opp.best_net_edge * 100,
        verdict=verdict,
        primary_rejection=primary,
        contracts=decision.contracts,
        regime=regime.value,
        explainability=settlement.confidence,
        edge_quality=decision.action.value,
        edge_action=edge_action,
        trade_reason=decision.reason,
    )
    return record, opp, decision
