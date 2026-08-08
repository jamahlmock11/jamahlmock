"""Evidence-gated trade decisions for Kalshi BTC 1-hour markets.

Default action is NO_TRADE. A trade is emitted only when:
- Gap tier clears the bot-action matrix (Strong BUY / Conditional / No trade)
- Ensemble confidence clears the floor
- Post-fee edge clears the minimum
- Model disagreement stays below the ceiling
- Book quality (spread / liquidity) is acceptable
- Horizon is inside the 1h trading window

Gap tier (raw, before fees): gap_pp = (model_prob − market_price) × 100
  ≥20pp → Strong BUY candidate
  ≥15pp → Only if other signals confirm
  <15pp → No trade
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from kalshi_bot.config import BotActionConfig, ForecastGateConfig, SeriesConfig, SmileConfig
from kalshi_bot.data.realized_vol import RealizedVolEstimate
from kalshi_bot.models.forecast import EnsembleForecast, ForecastAction, forecast_prob_above
from kalshi_bot.models.smile import VolSmile
from kalshi_bot.strategy.bot_action import (
    BotAction,
    GapAssessment,
    assess_buy_gap,
    other_signals_confirm,
)
from kalshi_bot.strategy.fees import quadratic_fee_per_contract
from kalshi_bot.strategy.mispricing import Side


class DecisionVerdict(str, Enum):
    TRADE = "TRADE"
    NO_TRADE = "NO_TRADE"


@dataclass(frozen=True)
class TradeDecision:
    verdict: DecisionVerdict
    action: ForecastAction
    side: Side | None
    ticker: str
    strike: float
    spot: float
    kalshi_price: float | None
    forecast_prob: float
    edge_pp: float
    edge_after_fees_pp: float
    expected_value_per_contract: float
    confidence: float
    disagreement_pp: float
    seconds_to_expiry: float
    forecast: EnsembleForecast
    reasons: tuple[str, ...]
    blockers: tuple[str, ...]
    bot_action: BotAction = BotAction.NO_TRADE
    gap_pp: float = 0.0

    @property
    def should_trade(self) -> bool:
        return self.verdict == DecisionVerdict.TRADE


def _seconds_to_expiry(close: datetime | None, now: datetime | None = None) -> float:
    if close is None:
        return 0.0
    now = now or datetime.now(timezone.utc)
    if close.tzinfo is None:
        close = close.replace(tzinfo=timezone.utc)
    return (close - now).total_seconds()


def _empty_decision(
    *,
    ticker: str,
    strike: float,
    spot: float,
    forecast: EnsembleForecast,
    blockers: list[str],
    reasons: list[str],
    kalshi_price: float | None = None,
    confidence: float = 0.0,
    secs: float = 0.0,
    bot_action: BotAction = BotAction.NO_TRADE,
    gap_pp: float = 0.0,
    edge_pp: float = 0.0,
    edge_after: float = 0.0,
    ev: float = 0.0,
    side: Side | None = None,
) -> TradeDecision:
    return TradeDecision(
        verdict=DecisionVerdict.NO_TRADE,
        action=ForecastAction.NO_TRADE,
        side=side,
        ticker=ticker,
        strike=strike,
        spot=spot,
        kalshi_price=kalshi_price,
        forecast_prob=forecast.probability_yes,
        edge_pp=edge_pp,
        edge_after_fees_pp=edge_after,
        expected_value_per_contract=ev,
        confidence=confidence,
        disagreement_pp=forecast.disagreement_pp,
        seconds_to_expiry=secs,
        forecast=forecast,
        reasons=tuple(reasons),
        blockers=tuple(dict.fromkeys(blockers)),
        bot_action=bot_action,
        gap_pp=gap_pp,
    )


def evaluate_forecast_market(
    market: dict,
    *,
    spot: float,
    smile: VolSmile | None,
    series_cfg: SeriesConfig,
    smile_cfg: SmileConfig,
    gates: ForecastGateConfig,
    fee_rate: float = 0.07,
    fee_multiplier: float = 1.0,
    now: datetime | None = None,
    realized: RealizedVolEstimate | None = None,
    spot_is_official: bool = False,
    bot_action_cfg: BotActionConfig | None = None,
) -> TradeDecision:
    """Return TRADE or NO_TRADE for one KXBTCD market."""
    now = now or datetime.now(timezone.utc)
    action_cfg = bot_action_cfg or BotActionConfig()
    ticker = str(market.get("ticker") or "")
    strike = market.get("strike")
    close = market.get("close_time")
    yes_bid = market.get("yes_bid")
    yes_ask = market.get("yes_ask")
    no_ask = market.get("no_ask")
    volume = float(market.get("volume") or 0.0)

    blockers: list[str] = []
    reasons: list[str] = []

    if strike is None or close is None or spot <= 0:
        blockers.append("missing strike/close/spot")
        empty = forecast_prob_above(
            spot=max(spot, 1.0),
            strike=float(strike or spot or 1.0),
            close_time=close or now,
            smile=smile,
            rate=smile_cfg.risk_free_rate,
            dividend=smile_cfg.dividend_yield,
            now=now,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
        )
        return _empty_decision(
            ticker=ticker,
            strike=float(strike or 0.0),
            spot=spot,
            forecast=empty,
            blockers=blockers,
            reasons=reasons,
        )

    secs = _seconds_to_expiry(close, now)
    if secs < gates.min_seconds_to_expiry:
        blockers.append(f"too close to expiry ({secs:.0f}s < {gates.min_seconds_to_expiry:.0f}s)")
    if secs > gates.max_seconds_to_expiry:
        blockers.append(
            f"outside last-{gates.max_seconds_to_expiry/60:.0f}m window "
            f"({secs:.0f}s > {gates.max_seconds_to_expiry:.0f}s)"
        )

    if yes_bid is not None and yes_ask is not None:
        spread = yes_ask - yes_bid
        if spread > gates.max_spread:
            blockers.append(f"spread too wide ({spread*100:.1f}¢ > {gates.max_spread*100:.1f}¢)")
    else:
        blockers.append("incomplete top-of-book")

    if volume < gates.min_volume:
        blockers.append(f"insufficient volume ({volume:.0f} < {gates.min_volume:.0f})")

    forecast = forecast_prob_above(
        spot=spot,
        strike=float(strike),
        close_time=close,
        smile=smile,
        rate=smile_cfg.risk_free_rate,
        dividend=smile_cfg.dividend_yield,
        now=now,
        realized=realized,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        include_equal=True,
    )
    reasons.extend(forecast.evidence_notes)

    confidence = forecast.confidence
    effective_min_edge = max(gates.min_edge_pp, series_cfg.min_edge_pp)
    if not spot_is_official:
        confidence = max(0.0, confidence - gates.proxy_spot_confidence_penalty)
        effective_min_edge *= gates.proxy_spot_edge_multiplier
        reasons.append(
            f"proxy spot: confidence -{gates.proxy_spot_confidence_penalty:.2f}, "
            f"edge floor x{gates.proxy_spot_edge_multiplier:.2f}"
        )

    if confidence < gates.min_confidence:
        blockers.append(
            f"confidence {confidence:.2f} below floor {gates.min_confidence:.2f}"
        )
    if forecast.disagreement_pp > gates.max_disagreement_pp:
        blockers.append(
            f"disagreement {forecast.disagreement_pp:.1f}pp above max {gates.max_disagreement_pp:.1f}pp"
        )
    if not forecast.sufficient_evidence:
        blockers.append("ensemble evidence insufficient")

    p = forecast.probability_yes
    p_yes_conservative = forecast.probability_lo
    p_no_conservative = 1.0 - forecast.probability_hi

    # Deep ITM YES: spot already through strike by buffer — NO mean-reversion
    # trades need extra edge (quiet-tape vol floors invent false reversals).
    itm_cushion = spot - float(strike)
    deep_itm_yes = itm_cushion >= gates.deep_itm_buffer_usd
    deep_otm_yes = (-itm_cushion) >= gates.deep_itm_buffer_usd

    # Candidate: side, price, raw_gap_pp, edge_after, ev, ForecastAction, BotAction, GapAssessment
    candidates: list[
        tuple[Side, float, float, float, float, ForecastAction, BotAction, GapAssessment]
    ] = []
    gap_rejects: list[str] = []

    if yes_ask is not None and 0 < yes_ask < 1:
        assessment = assess_buy_gap(p, yes_ask, config=action_cfg)
        fee = quadratic_fee_per_contract(yes_ask, fee_rate=fee_rate, fee_multiplier=fee_multiplier)
        edge_after = (p_yes_conservative - yes_ask - fee) * 100
        ev = p_yes_conservative - yes_ask - fee
        need = effective_min_edge + (gates.deep_itm_extra_edge_pp if deep_otm_yes else 0.0)

        if assessment.action == BotAction.NO_TRADE:
            gap_rejects.append(
                f"YES gap {assessment.gap_pp:.1f}pp → {assessment.label} "
                f"(need ≥{action_cfg.conditional_min_gap_pp:.0f}pp)"
            )
        elif assessment.action == BotAction.CONDITIONAL:
            ok, fails = other_signals_confirm(
                confidence=confidence,
                disagreement_pp=forecast.disagreement_pp,
                sufficient_evidence=forecast.sufficient_evidence,
                config=action_cfg,
            )
            if not ok:
                gap_rejects.extend(fails)
            elif edge_after >= need:
                candidates.append(
                    (
                        Side.YES,
                        yes_ask,
                        assessment.gap_pp,
                        edge_after,
                        ev,
                        ForecastAction.TRADE_YES,
                        assessment.action,
                        assessment,
                    )
                )
            else:
                gap_rejects.append(
                    f"YES conditional gap {assessment.gap_pp:.1f}pp but post-fee edge "
                    f"{edge_after:.1f}pp < {need:.1f}pp"
                )
        else:  # STRONG_BUY
            reasons.append(
                f"YES {assessment.label}: model={p*100:.0f}% market={yes_ask*100:.0f}¢ "
                f"gap={assessment.gap_pp:.1f}pp"
            )
            if edge_after >= need:
                candidates.append(
                    (
                        Side.YES,
                        yes_ask,
                        assessment.gap_pp,
                        edge_after,
                        ev,
                        ForecastAction.TRADE_YES,
                        assessment.action,
                        assessment,
                    )
                )
            else:
                gap_rejects.append(
                    f"YES strong gap {assessment.gap_pp:.1f}pp but post-fee edge "
                    f"{edge_after:.1f}pp < {need:.1f}pp"
                )

    if no_ask is None and yes_bid is not None:
        no_ask = max(0.0, 1.0 - yes_bid)
    if no_ask is not None and 0 < no_ask < 1:
        q = 1.0 - p
        assessment = assess_buy_gap(q, no_ask, config=action_cfg)
        fee = quadratic_fee_per_contract(no_ask, fee_rate=fee_rate, fee_multiplier=fee_multiplier)
        edge_after = (p_no_conservative - no_ask - fee) * 100
        ev = p_no_conservative - no_ask - fee
        need = effective_min_edge + (gates.deep_itm_extra_edge_pp if deep_itm_yes else 0.0)

        if assessment.action == BotAction.NO_TRADE:
            gap_rejects.append(
                f"NO gap {assessment.gap_pp:.1f}pp → {assessment.label} "
                f"(need ≥{action_cfg.conditional_min_gap_pp:.0f}pp)"
            )
        elif assessment.action == BotAction.CONDITIONAL:
            ok, fails = other_signals_confirm(
                confidence=confidence,
                disagreement_pp=forecast.disagreement_pp,
                sufficient_evidence=forecast.sufficient_evidence,
                config=action_cfg,
            )
            if not ok:
                gap_rejects.extend(fails)
            elif edge_after >= need:
                candidates.append(
                    (
                        Side.NO,
                        no_ask,
                        assessment.gap_pp,
                        edge_after,
                        ev,
                        ForecastAction.TRADE_NO,
                        assessment.action,
                        assessment,
                    )
                )
            else:
                gap_rejects.append(
                    f"NO conditional gap {assessment.gap_pp:.1f}pp but post-fee edge "
                    f"{edge_after:.1f}pp < {need:.1f}pp"
                )
        else:  # STRONG_BUY
            reasons.append(
                f"NO {assessment.label}: model={q*100:.0f}% market={no_ask*100:.0f}¢ "
                f"gap={assessment.gap_pp:.1f}pp"
            )
            if edge_after >= need:
                candidates.append(
                    (
                        Side.NO,
                        no_ask,
                        assessment.gap_pp,
                        edge_after,
                        ev,
                        ForecastAction.TRADE_NO,
                        assessment.action,
                        assessment,
                    )
                )
            else:
                gap_rejects.append(
                    f"NO strong gap {assessment.gap_pp:.1f}pp but post-fee edge "
                    f"{edge_after:.1f}pp < {need:.1f}pp"
                )

    if not candidates:
        blockers.append(
            f"no side clears gap-tier + conservative post-fee floor "
            f"(≥{action_cfg.conditional_min_gap_pp:.0f}pp raw / {effective_min_edge:.1f}pp post-fee)"
        )
        blockers.extend(gap_rejects)

    min_edge = effective_min_edge
    if smile is not None and smile.is_synthetic:
        min_edge = max(min_edge, 25.0)
        reasons.append("synthetic smile: elevated edge floor to 25pp")

    viable = [c for c in candidates if c[3] >= min_edge]
    if candidates and not viable:
        blockers.append(f"edge below effective floor ({min_edge:.1f}pp)")
    if blockers or not viable:
        best = max(candidates, key=lambda c: c[3]) if candidates else None
        # Surface the best raw gap assessment even on NO_TRADE for logging.
        best_gap = best[2] if best else 0.0
        best_bot = best[6] if best else BotAction.NO_TRADE
        if best is None and yes_ask is not None and 0 < yes_ask < 1:
            ya = assess_buy_gap(p, yes_ask, config=action_cfg)
            best_gap, best_bot = ya.gap_pp, ya.action
        return _empty_decision(
            ticker=ticker,
            strike=float(strike),
            spot=spot,
            forecast=forecast,
            blockers=blockers,
            reasons=reasons,
            kalshi_price=best[1] if best else (yes_ask if yes_ask is not None else no_ask),
            confidence=confidence,
            secs=secs,
            bot_action=best_bot,
            gap_pp=best_gap,
            edge_pp=best[2] if best else best_gap,
            edge_after=best[3] if best else 0.0,
            ev=best[4] if best else 0.0,
            side=best[0] if best else None,
        )

    side, price, gap_pp, edge_after, ev, action, bot_action, assessment = max(
        viable, key=lambda c: c[4]
    )
    reasons.append(
        f"{bot_action.label}: gap={gap_pp:.1f}pp; conservative EV ${ev:.3f}/contract; "
        f"conf={confidence:.2f}; disagreement={forecast.disagreement_pp:.1f}pp"
    )
    return TradeDecision(
        verdict=DecisionVerdict.TRADE,
        action=action,
        side=side,
        ticker=ticker,
        strike=float(strike),
        spot=spot,
        kalshi_price=price,
        forecast_prob=p if side == Side.YES else 1.0 - p,
        edge_pp=gap_pp,
        edge_after_fees_pp=edge_after,
        expected_value_per_contract=ev,
        confidence=confidence,
        disagreement_pp=forecast.disagreement_pp,
        seconds_to_expiry=secs,
        forecast=forecast,
        reasons=tuple(reasons),
        blockers=(),
        bot_action=bot_action,
        gap_pp=gap_pp,
    )
