"""Audited V6 market evaluation with structured rejection codes."""

from __future__ import annotations

from datetime import datetime, timezone

from kalshi_bot.config import Rules15mConfig, V6Config
from kalshi_bot.data.kalshi_client import KalshiClient
from kalshi_bot.strategy.arbitrary_policy import EdgeChaseGuard, evaluate_arbitrary
from kalshi_bot.strategy.decision_record import (
    DataFreshness,
    FilterCheck,
    MarketEvaluationRecord,
    SideEvaluation,
    pick_primary_rejection,
)
from kalshi_bot.strategy.fees import quadratic_fee_per_contract
from kalshi_bot.strategy.rejection_codes import RejectionCode
from kalshi_bot.strategy.tiered_edge import (
    EdgeQuality,
    classify_edge_quality,
    classify_tier,
    estimate_slippage,
    opportunity_score,
    should_trade_for_quality,
)
from kalshi_bot.strategy.v6_upgrades import (
    V6IntelligenceEngine,
    assess_market_quality,
    compute_microstructure,
    compute_price_action,
    compute_time_features,
    detect_manipulation,
    detect_regime,
    explainability_score,
    institutional_flow_score,
    monte_carlo_binary,
    multi_model_ensemble,
    passes_strict_edge,
    strike_gravity_bias,
    strict_edge_gap_dollars,
)


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


def _min_edge_floor(rules: Rules15mConfig) -> float:
    if rules.strict_edge.min_gap_dollars is not None:
        return rules.strict_edge.min_gap_dollars
    if rules.tiers.edge_experimental is not None:
        return rules.tiers.edge_experimental
    return 0.0


def _rules_not_configured_record(
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


def _evaluate_side(
    *,
    side: str,
    model_prob: float,
    ask: float | None,
    min_edge: float,
    fee_rate: float = 0.07,
    spread: float = 0.0,
    liquidity_score: float = 0.0,
) -> SideEvaluation:
    if ask is None or not (0 < ask < 1):
        return SideEvaluation(
            side=side,
            model_probability=model_prob,
            executable_ask=None,
            raw_edge_dollars=0.0,
            estimated_fee=0.0,
            estimated_slippage=0.0,
            net_edge_dollars=0.0,
            expected_value_per_contract=0.0,
            passes_edge_threshold=False,
            passes_net_ev=False,
            rejection_codes=[RejectionCode.MISSING_DATA],
        )

    raw = strict_edge_gap_dollars(model_prob, ask)
    fee = quadratic_fee_per_contract(ask, fee_rate=fee_rate)
    slip = estimate_slippage(spread=spread, liquidity_score=liquidity_score)
    net = raw - fee - slip
    ev = model_prob - ask - fee - slip
    passes_edge = raw >= min_edge
    passes_ev = net > 0

    rejections: list[RejectionCode] = []
    if not passes_edge:
        rejections.append(RejectionCode.EDGE_TOO_SMALL)
    if not passes_ev:
        rejections.append(RejectionCode.EXPECTED_VALUE_NEGATIVE)

    return SideEvaluation(
        side=side,
        model_probability=model_prob,
        executable_ask=ask,
        raw_edge_dollars=raw,
        estimated_fee=fee,
        estimated_slippage=slip,
        net_edge_dollars=net,
        expected_value_per_contract=ev,
        passes_edge_threshold=passes_edge,
        passes_net_ev=passes_ev,
        rejection_codes=rejections,
    )


def evaluate_market_audited(
    engine: V6IntelligenceEngine,
    market: dict,
    *,
    spot: float,
    spot_source: str,
    spot_is_official: bool,
    vol: float,
    options_prob: float | None = None,
    now: datetime | None = None,
    fee_rate: float = 0.07,
) -> MarketEvaluationRecord:
    """Full audited evaluation with structured rejection codes."""
    config = engine.config
    rules = engine.rules
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

    all_rejections: list[RejectionCode] = []
    filter_checks: list[FilterCheck] = []

    # --- Timing ---
    secs = max((close - now).total_seconds(), 0) if close else 0
    mins = secs / 60.0
    open_secs = 0.0
    if open_t and close:
        open_secs = max((now - open_t).total_seconds(), 0)

    if not rules.enabled:
        return _rules_not_configured_record(
            ticker=ticker,
            series=series,
            now=now,
            secs=secs,
            mins=mins,
            spot=spot,
            spot_source=spot_source,
            strike=strike,
            filter_checks=filter_checks,
        )

    tier_cfg = rules.tiers
    quality_rules = rules.quality

    timing_ok = (
        secs >= config.min_seconds_to_expiry
        and secs <= config.max_seconds_to_expiry
        and open_secs >= config.min_open_seconds
    )
    if not timing_ok:
        if secs < config.min_seconds_to_expiry:
            detail = f"too close ({secs:.0f}s < {config.min_seconds_to_expiry:.0f}s)"
        elif secs > config.max_seconds_to_expiry:
            detail = f"too early ({secs:.0f}s > {config.max_seconds_to_expiry:.0f}s)"
        else:
            detail = f"market too new ({open_secs:.0f}s < {config.min_open_seconds:.0f}s)"
        all_rejections.append(RejectionCode.TIMING_RESTRICTION)
        filter_checks.append(
            FilterCheck("timing", False, RejectionCode.TIMING_RESTRICTION, detail)
        )
    else:
        filter_checks.append(FilterCheck("timing", True, detail=f"{mins:.1f}m remaining"))

    # --- Missing data ---
    if not yes_ask and not yes_bid:
        all_rejections.append(RejectionCode.MISSING_DATA)
        filter_checks.append(
            FilterCheck("book", False, RejectionCode.MISSING_DATA, "no yes bid/ask")
        )

    # --- Data freshness ---
    cf_status = "FRESH" if spot_is_official else "PROXY"
    spot_status = "FRESH" if spot > 0 else "MISSING"
    data_freshness = DataFreshness(
        cf_benchmark=cf_status,
        btc_spot=spot_status,
        order_book="PENDING",
        options_smile="FRESH" if options_prob is not None else "MISSING",
    )

    # --- Order book ---
    orderbook = None
    ob_error = None
    if engine.client and ticker:
        try:
            orderbook = engine.client.get_orderbook(ticker, depth=10)
            data_freshness = DataFreshness(
                cf_benchmark=data_freshness.cf_benchmark,
                btc_spot=data_freshness.btc_spot,
                order_book="FRESH",
                options_smile=data_freshness.options_smile,
            )
        except Exception as exc:
            ob_error = str(exc)
            data_freshness = DataFreshness(
                cf_benchmark=data_freshness.cf_benchmark,
                btc_spot=data_freshness.btc_spot,
                order_book="MISSING",
                options_smile=data_freshness.options_smile,
            )

    micro = compute_microstructure(
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        orderbook=orderbook,
        prev_spread=engine._prev_spread,
        prev_depth=engine._prev_depth,
    )
    engine._prev_spread = micro.spread
    engine._prev_depth = (micro.depth_bid_10, micro.depth_ask_10)

    pa = compute_price_action(list(engine._price_history))
    if close:
        compute_time_features(close, now=now)
    regime = detect_regime(pa, micro)
    manip = detect_manipulation(micro, pa)

    max_disagreement_pp = (
        quality_rules.max_model_disagreement_pp
        if quality_rules.max_model_disagreement_pp is not None
        else config.max_model_disagreement_pp
    )
    ensemble = multi_model_ensemble(
        spot=spot,
        strike=strike,
        vol=vol,
        seconds_to_expiry=secs,
        market_yes=yes_ask or 0.5,
        micro=micro,
        price_action=pa,
        options_prob=options_prob,
        max_disagreement_pp=max_disagreement_pp,
    )

    mc_mean, _, _ = monte_carlo_binary(
        spot=spot, strike=strike, vol=vol, seconds=secs, n_sims=config.monte_carlo_sims
    )

    gravity = strike_gravity_bias(spot, strike, secs, pa.momentum_1m)
    inst_flow = institutional_flow_score(micro)
    raw_prob = (
        ensemble.consensus_prob * engine.weights.weights["ensemble"]
        + mc_mean * engine.weights.weights["monte_carlo"]
        + (options_prob or ensemble.consensus_prob) * 0.1
        + gravity
        + inst_flow * 0.03
    )
    raw_prob = max(0.01, min(0.99, raw_prob))
    model_prob, calibrated = engine.calibrator.calibrate(raw_prob)

    spread_limit = quality_rules.max_spread if quality_rules.max_spread is not None else 1.0
    liquidity_floor = (
        quality_rules.min_liquidity_score
        if quality_rules.min_liquidity_score is not None
        else 0.0
    )
    quality = assess_market_quality(
        micro=micro,
        price_action=pa,
        ensemble=ensemble,
        spread_limit=spread_limit,
        min_liquidity=liquidity_floor,
        manipulation_flag=manip,
    )

    # --- Individual filter checks ---
    if quality_rules.max_spread is not None:
        spread_ok = micro.spread <= quality_rules.max_spread
        if not spread_ok:
            all_rejections.append(RejectionCode.SPREAD_TOO_WIDE)
            filter_checks.append(
                FilterCheck(
                    "spread",
                    False,
                    RejectionCode.SPREAD_TOO_WIDE,
                    f"{micro.spread*100:.1f}¢ > {quality_rules.max_spread*100:.0f}¢",
                )
            )
        else:
            filter_checks.append(FilterCheck("spread", True, detail=f"{micro.spread*100:.1f}¢"))
    else:
        spread_ok = True
        filter_checks.append(FilterCheck("spread", True, detail="no spread rule configured"))

    if quality_rules.min_liquidity_score is not None:
        liq_ok = micro.liquidity_score >= quality_rules.min_liquidity_score
        if not liq_ok:
            all_rejections.append(RejectionCode.INSUFFICIENT_LIQUIDITY)
            filter_checks.append(
                FilterCheck(
                    "liquidity",
                    False,
                    RejectionCode.INSUFFICIENT_LIQUIDITY,
                    f"score={micro.liquidity_score:.2f}",
                )
            )
        else:
            filter_checks.append(
                FilterCheck("liquidity", True, detail=f"score={micro.liquidity_score:.2f}")
            )
    else:
        liq_ok = True
        filter_checks.append(
            FilterCheck("liquidity", True, detail="no liquidity rule configured")
        )

    if not ensemble.models_agree:
        # Diagnostic only — not a hard block (model conflict handled per tier).
        filter_checks.append(
            FilterCheck(
                "model_agreement",
                False,
                RejectionCode.MODEL_CONFLICT,
                f"disagreement across models",
            )
        )
    else:
        filter_checks.append(FilterCheck("model_agreement", True))

    if manip:
        all_rejections.append(RejectionCode.MANIPULATION_SUSPECTED)
        filter_checks.append(
            FilterCheck("manipulation", False, RejectionCode.MANIPULATION_SUSPECTED)
        )

    if pa.breakout_signal == "fake_breakout":
        filter_checks.append(
            FilterCheck("breakout", False, RejectionCode.FAKE_BREAKOUT)
        )

    # Risk
    allow, risk_reason = engine.risk.allow_trade(ensemble.agreement_score)
    if not allow:
        code = RejectionCode.KILL_SWITCH if "kill" in risk_reason else (
            RejectionCode.COOLDOWN if "cooldown" in risk_reason else RejectionCode.RISK_LIMIT
        )
        all_rejections.append(code)
        filter_checks.append(FilterCheck("risk", False, code, risk_reason))

    feat_dict = {
        "bid_ask_imbalance": micro.bid_ask_imbalance,
        "momentum_1m": pa.momentum_1m,
        "liquidity_score": micro.liquidity_score,
        "regime": regime.value,
    }
    pattern_min = quality_rules.min_pattern_examples
    if pattern_min is None:
        pattern_min = config.min_pattern_examples
    pattern_n, _ = engine.journal.similar_setups(
        feat_dict, min_examples=pattern_min
    )
    if (
        quality_rules.require_pattern_evidence
        and quality_rules.min_pattern_examples is not None
        and pattern_n < quality_rules.min_pattern_examples
    ):
        all_rejections.append(RejectionCode.PATTERN_EVIDENCE_INSUFFICIENT)
        filter_checks.append(
            FilterCheck(
                "pattern",
                False,
                RejectionCode.PATTERN_EVIDENCE_INSUFFICIENT,
                f"{pattern_n}<{quality_rules.min_pattern_examples}",
            )
        )

    # --- Both sides evaluated independently ---
    min_edge = _min_edge_floor(rules)
    yes_side = _evaluate_side(
        side="YES",
        model_prob=model_prob,
        ask=yes_ask,
        min_edge=min_edge,
        fee_rate=fee_rate,
        spread=micro.spread,
        liquidity_score=micro.liquidity_score,
    )
    no_side = _evaluate_side(
        side="NO",
        model_prob=1.0 - model_prob,
        ask=no_ask,
        min_edge=min_edge,
        fee_rate=fee_rate,
        spread=micro.spread,
        liquidity_score=micro.liquidity_score,
    )

    # Best side by net edge (risk-adjusted EV ranking)
    candidates: list[tuple[str, SideEvaluation]] = [
        ("YES", yes_side),
        ("NO", no_side),
    ]
    best_side: str | None = None
    best_eval: SideEvaluation | None = None
    best_net = max(yes_side.net_edge_dollars, no_side.net_edge_dollars)
    for name, ev in candidates:
        if ev.executable_ask is None:
            continue
        if best_eval is None or ev.net_edge_dollars > best_eval.net_edge_dollars:
            best_eval = ev
            best_side = name

    # Hard blockers — risk/timing/data only (NOT model conflict)
    hard_blockers = {
        RejectionCode.KILL_SWITCH,
        RejectionCode.RISK_LIMIT,
        RejectionCode.COOLDOWN,
        RejectionCode.TIMING_RESTRICTION,
        RejectionCode.MISSING_DATA,
    }
    has_hard_block = bool(hard_blockers & set(all_rejections))

    # Edge quality tier (separates frequency from quality)
    best_net_for_tier = best_eval.net_edge_dollars if best_eval else 0.0
    best_raw = best_eval.raw_edge_dollars if best_eval else 0.0

    if (
        not quality.tradeable
        and tier_cfg.edge_conditional is not None
        and best_raw < tier_cfg.edge_conditional
    ):
        all_rejections.append(RejectionCode.QUALITY_SCORE_TOO_HIGH)

    edge_quality = classify_edge_quality(
        best_net_for_tier,
        raw_edge_dollars=best_raw,
        config=tier_cfg,
    )

    tier_result = classify_tier(
        net_edge_dollars=best_net_for_tier,
        model_confidence=ensemble.agreement_score,
        data_fresh=spot_is_official and data_freshness.order_book != "MISSING",
        liquidity_ok=liq_ok,
        spread_ok=spread_ok,
        model_agrees=ensemble.models_agree,
        no_conflicts=not manip and pa.breakout_signal != "fake_breakout",
        config=tier_cfg,
    )

    opp_score = opportunity_score(
        net_edge_dollars=best_net_for_tier,
        model_confidence=ensemble.agreement_score,
        momentum_confirmation=pa.momentum_1m,
        order_flow_confirmation=micro.bid_ask_imbalance,
        liquidity_score=micro.liquidity_score,
        spread=micro.spread,
        data_fresh=spot_is_official,
        model_agrees=ensemble.models_agree,
    )

    # --- Final verdict: Arbitrary policy (both sides, calibration, no chase) ---
    verdict = "NO_TRADE"
    contracts = 0
    side_rejections: list[RejectionCode] = []
    trade_reason = ""

    arb = evaluate_arbitrary(
        ticker=ticker,
        model_prob_yes=model_prob,
        model_prob_yes_lo=max(0.01, model_prob - 0.04),
        model_prob_yes_hi=min(0.99, model_prob + 0.04),
        yes_ask=yes_ask,
        yes_bid=yes_bid,
        no_ask=no_ask,
        seconds_to_expiry=secs,
        min_seconds_to_expiry=config.min_seconds_to_expiry,
        max_seconds_to_expiry=config.max_seconds_to_expiry,
        base_min_edge_pp=_min_edge_floor(rules) * 100.0,
        fee_rate=fee_rate,
        confidence=ensemble.agreement_score,
        disagreement_pp=(
            max(v.probability for v in ensemble.votes) - min(v.probability for v in ensemble.votes)
        )
        * 100,
        sufficient_evidence=ensemble.models_agree,
        calibrator=engine.calibrator,
        chase_guard=engine.chase_guard,
        policy_cfg=rules.arbitrary,
        bot_action_cfg=rules.bot_action,
    ) if rules.arbitrary.enabled else None

    if arb is not None and arb.chase_blocked:
        all_rejections.append(RejectionCode.EDGE_TOO_SMALL)
        filter_checks.append(
            FilterCheck("edge_chase", False, RejectionCode.EDGE_TOO_SMALL, "edge decayed or price chased")
        )

    if not has_hard_block and best_eval and best_side:
        net_ev_ok = best_eval.passes_net_ev
        can_trade, trade_reason = should_trade_for_quality(
            edge_quality,
            model_confidence=ensemble.agreement_score,
            model_agrees=ensemble.models_agree,
            data_fresh=spot_is_official,
            liquidity_ok=liq_ok,
            spread_ok=spread_ok,
            no_manipulation=not manip,
            net_ev_positive=net_ev_ok,
            net_edge_dollars=best_net_for_tier,
            config=tier_cfg,
        )

        arbitrary_ok = (
            arb is not None
            and arb.verdict != "NO_TRADE"
            and arb.chosen_side == best_side
        )
        if can_trade and tier_cfg.enabled_for_live and arbitrary_ok:
            verdict = f"TRADE_{best_side}"
        elif can_trade and tier_cfg.enabled_for_live and arb is not None and arb.verdict == "NO_TRADE":
            all_rejections.extend(
                [RejectionCode.EDGE_TOO_SMALL]
                if any("overpriced_favorite" in b for b in arb.blockers)
                else [RejectionCode.LOW_CONFIDENCE]
            )
            trade_reason = "; ".join(arb.blockers[:3]) or trade_reason
        elif not can_trade:
            if edge_quality.quality == EdgeQuality.NO_TRADE:
                all_rejections.append(RejectionCode.EDGE_TOO_SMALL)
            elif not ensemble.models_agree and edge_quality.requires_confirmation:
                all_rejections.append(RejectionCode.MODEL_CONFLICT)
            elif not net_ev_ok:
                all_rejections.append(RejectionCode.EXPECTED_VALUE_NEGATIVE)
            else:
                all_rejections.append(RejectionCode.LOW_CONFIDENCE)
            side_rejections = list(best_eval.rejection_codes)

    if verdict == "NO_TRADE":
        if not side_rejections and best_eval:
            side_rejections = list(best_eval.rejection_codes)
        all_rejections.extend(side_rejections)
        if edge_quality.quality == EdgeQuality.NO_TRADE and RejectionCode.EDGE_TOO_SMALL not in all_rejections:
            all_rejections.append(RejectionCode.EDGE_TOO_SMALL)

    primary = pick_primary_rejection(all_rejections) if verdict == "NO_TRADE" else RejectionCode.NONE

    explain = explainability_score(
        ensemble=ensemble,
        quality=quality,
        pattern_support=pattern_n,
        strict_edge_gap=best_eval.raw_edge_dollars if best_eval else 0.0,
    )

    if verdict != "NO_TRADE" and best_eval and best_eval.executable_ask:
        prob = model_prob if best_side == "YES" else 1.0 - model_prob
        base_contracts = engine.risk.kelly_size(
            prob=prob,
            price=best_eval.executable_ask,
            bankroll=config.bankroll_usd,
            confidence=explain,
        )
        contracts = max(0, int(base_contracts * edge_quality.size_multiplier))
        if edge_quality.quality == EdgeQuality.EXPERIMENTAL:
            contracts = min(contracts, tier_cfg.experimental_max_contracts)

    return MarketEvaluationRecord(
        ticker=ticker,
        series=series,
        evaluated_at=now,
        seconds_to_expiry=secs,
        minutes_to_expiry=mins,
        spot=spot,
        spot_source=spot_source,
        strike=strike,
        model_prob_up=model_prob,
        model_prob_down=1.0 - model_prob,
        model_confidence=ensemble.agreement_score,
        model_disagreement_pp=(
            max(v.probability for v in ensemble.votes)
            - min(v.probability for v in ensemble.votes)
        )
        * 100,
        monte_carlo_prob=mc_mean,
        options_implied_prob=options_prob,
        calibrated=calibrated,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_ask=no_ask,
        spread=micro.spread,
        yes_side=yes_side,
        no_side=no_side,
        best_side=best_side,
        best_net_edge=best_net,
        liquidity_score=micro.liquidity_score,
        bid_ask_imbalance=micro.bid_ask_imbalance,
        order_book_depth_bid=micro.depth_bid_10,
        order_book_depth_ask=micro.depth_ask_10,
        data_freshness=data_freshness,
        filter_checks=filter_checks,
        all_rejection_codes=list(dict.fromkeys(all_rejections)),
        setup_tier=edge_quality.quality.value,
        opportunity_score=opp_score,
        verdict=verdict,
        primary_rejection=primary,
        contracts=contracts,
        regime=regime.value,
        explainability=explain,
        edge_quality=edge_quality.quality.value,
        edge_action=edge_quality.action_label,
        trade_reason=trade_reason,
    )
