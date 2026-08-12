"""Trade filter — probability → fair value → executable → net edge → risk → decision."""

from __future__ import annotations

from dataclasses import dataclass

from kalshi_bot.data.btc_data_engine import BtcMarketSnapshot
from kalshi_bot.strategy.mispricing_engine import MispricingOpportunity, TradeAction
from kalshi_bot.strategy.rejection_codes import RejectionCode
from kalshi_bot.strategy.time_buckets import TimeBucket, bucket_policy, classify_time_bucket
from kalshi_bot.strategy.v6_upgrades import MicrostructureSnapshot


@dataclass(frozen=True)
class TradeDecision:
    action: TradeAction
    side: str | None  # YES | NO
    contracts: int
    reason: str
    rejection: RejectionCode
    time_bucket: str
    net_edge_dollars: float
    fair_value: float
    executable_price: float | None
    confidence: float


def filter_trade(
    opp: MispricingOpportunity,
    *,
    btc: BtcMarketSnapshot,
    micro: MicrostructureSnapshot,
    max_spread: float,
    min_liquidity_score: float,
    bucket_overrides: dict[str, dict] | None = None,
    min_seconds: float = 60.0,
    max_seconds: float = 900.0,
    risk_allows: bool = True,
    risk_reason: str = "",
    kelly_contracts: int = 0,
) -> TradeDecision:
    bucket = classify_time_bucket(opp.seconds_to_expiry, min_seconds=min_seconds, max_seconds=max_seconds)
    policy = bucket_policy(bucket, bucket_overrides)

    def _no(action: TradeAction, reason: str, code: RejectionCode) -> TradeDecision:
        return TradeDecision(
            action=action,
            side=None,
            contracts=0,
            reason=reason,
            rejection=code,
            time_bucket=bucket.value,
            net_edge_dollars=opp.best_net_edge,
            fair_value=opp.fair_value_yes,
            executable_price=None,
            confidence=opp.settlement.confidence,
        )

    if policy is None:
        return _no(TradeAction.NO_TRADE, f"outside time window ({bucket.value})", RejectionCode.TIMING_RESTRICTION)
    if btc.stale:
        return _no(TradeAction.NO_TRADE, "BTC data stale — stop trading", RejectionCode.STALE_DATA)
    if opp.kalshi_stale:
        return _no(TradeAction.NO_TRADE, "Kalshi price stale", RejectionCode.STALE_DATA)
    if not risk_allows:
        return _no(TradeAction.NO_TRADE, risk_reason or "risk limit", RejectionCode.RISK_LIMIT)
    if micro.spread > max_spread:
        return _no(
            TradeAction.NO_TRADE,
            f"spread {micro.spread*100:.1f}¢ > {max_spread*100:.0f}¢",
            RejectionCode.SPREAD_TOO_WIDE,
        )
    if micro.liquidity_score < min_liquidity_score:
        return _no(
            TradeAction.NO_TRADE,
            f"liquidity {micro.liquidity_score:.2f} below floor",
            RejectionCode.INSUFFICIENT_LIQUIDITY,
        )
    if opp.settlement.confidence < policy.min_confidence:
        return _no(
            TradeAction.WAIT if policy.allow_wait else TradeAction.NO_TRADE,
            f"confidence {opp.settlement.confidence:.0%} < {policy.min_confidence:.0%} ({policy.label})",
            RejectionCode.LOW_CONFIDENCE,
        )

    best = opp.best_mispricing()
    if best is None or best.executable_ask is None:
        return _no(TradeAction.NO_TRADE, "no executable ask", RejectionCode.MISSING_DATA)

    if best.net_edge_dollars < policy.min_net_edge_dollars:
        if policy.allow_wait and best.net_edge_dollars > 0:
            return _no(
                TradeAction.WAIT,
                f"net edge {best.net_edge_dollars*100:.1f}¢ < {policy.min_net_edge_dollars*100:.0f}¢ ({policy.label})",
                RejectionCode.EDGE_TOO_SMALL,
            )
        return _no(
            TradeAction.NO_TRADE,
            f"NO TRADE — INSUFFICIENT EDGE ({best.net_edge_dollars*100:.1f}¢ net, need {policy.min_net_edge_dollars*100:.0f}¢)",
            RejectionCode.EDGE_TOO_SMALL,
        )
    if best.raw_edge_dollars < policy.min_raw_edge_dollars:
        return _no(
            TradeAction.NO_TRADE,
            f"raw edge {best.raw_edge_dollars*100:.1f}¢ below {policy.label} floor",
            RejectionCode.EDGE_TOO_SMALL,
        )
    if best.expected_value <= 0:
        return _no(TradeAction.NO_TRADE, "expected value ≤ 0 after costs", RejectionCode.EXPECTED_VALUE_NEGATIVE)

    action = TradeAction.BUY_YES if best.side == "YES" else TradeAction.BUY_NO
    contracts = max(0, int(kelly_contracts * policy.size_multiplier))
    return TradeDecision(
        action=action,
        side=best.side,
        contracts=contracts,
        reason=f"{action.value} — net edge {best.net_edge_dollars*100:.1f}¢ ({policy.label})",
        rejection=RejectionCode.NONE,
        time_bucket=bucket.value,
        net_edge_dollars=best.net_edge_dollars,
        fair_value=best.fair_value,
        executable_price=best.executable_ask,
        confidence=opp.settlement.confidence,
    )
