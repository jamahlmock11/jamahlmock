"""Edge quality tiers — separate trade frequency from trade quality.

| Edge (raw or net) | Tier         | Action                              |
|-------------------|--------------|-------------------------------------|
| ≥ 20¢             | EXCEPTIONAL  | 🟢 Exceptional opportunity          |
| 15–20¢            | STRONG       | 🟢 Strong trade                     |
| 8–15¢             | CONDITIONAL  | 🟡 Trade only with strong confirm.  |
| 5–8¢              | EXPERIMENTAL | 🟠 Small/experimental trade         |
| < 5¢              | NO_TRADE     | 🔴 No trade                         |
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from kalshi_bot.config import TierEdgeConfig


class EdgeQuality(str, Enum):
    EXCEPTIONAL = "EXCEPTIONAL"
    STRONG = "STRONG"
    CONDITIONAL = "CONDITIONAL"
    EXPERIMENTAL = "EXPERIMENTAL"
    NO_TRADE = "NO_TRADE"


EDGE_ACTION_LABEL: dict[EdgeQuality, str] = {
    EdgeQuality.EXCEPTIONAL: "🟢 Exceptional opportunity",
    EdgeQuality.STRONG: "🟢 Strong trade",
    EdgeQuality.CONDITIONAL: "🟡 Trade only with strong confirmation",
    EdgeQuality.EXPERIMENTAL: "🟠 Small/experimental trade",
    EdgeQuality.NO_TRADE: "🔴 No trade",
}


class SetupTier(str, Enum):
    A_PLUS = "A_PLUS"
    A = "A"
    B = "B"
    NONE = "NONE"


@dataclass(frozen=True)
class EdgeQualityAssessment:
    quality: EdgeQuality
    net_edge_dollars: float
    raw_edge_dollars: float
    action_label: str
    trades_allowed: bool
    requires_confirmation: bool
    size_multiplier: float
    detail: str


@dataclass(frozen=True)
class TierAssessment:
    tier: SetupTier
    min_edge_required: float
    net_edge_dollars: float
    qualifies: bool
    detail: str


def _tier_threshold(cfg: TierEdgeConfig, name: str) -> float | None:
    return getattr(cfg, name)


def classify_edge_quality(
    net_edge_dollars: float,
    *,
    raw_edge_dollars: float | None = None,
    config: TierEdgeConfig | None = None,
) -> EdgeQualityAssessment:
    """Classify edge into quality tier. Uses raw edge when configured."""
    cfg = config or TierEdgeConfig()
    raw = raw_edge_dollars if raw_edge_dollars is not None else net_edge_dollars
    tier_edge = raw if cfg.use_raw_edge_for_tiers else net_edge_dollars
    cents = tier_edge * 100

    exceptional = _tier_threshold(cfg, "edge_exceptional")
    strong = _tier_threshold(cfg, "edge_strong")
    conditional = _tier_threshold(cfg, "edge_conditional")
    experimental = _tier_threshold(cfg, "edge_experimental")
    if None in (exceptional, strong, conditional, experimental):
        return EdgeQualityAssessment(
            quality=EdgeQuality.NO_TRADE,
            net_edge_dollars=net_edge_dollars,
            raw_edge_dollars=raw,
            action_label=EDGE_ACTION_LABEL[EdgeQuality.NO_TRADE],
            trades_allowed=False,
            requires_confirmation=False,
            size_multiplier=0.0,
            detail="tier thresholds not configured",
        )

    if cents >= exceptional * 100:
        q = EdgeQuality.EXCEPTIONAL
    elif cents >= strong * 100:
        q = EdgeQuality.STRONG
    elif cents >= conditional * 100:
        q = EdgeQuality.CONDITIONAL
    elif cents >= experimental * 100:
        q = EdgeQuality.EXPERIMENTAL
    else:
        q = EdgeQuality.NO_TRADE

    if q == EdgeQuality.NO_TRADE:
        return EdgeQualityAssessment(
            quality=q,
            net_edge_dollars=net_edge_dollars,
            raw_edge_dollars=raw,
            action_label=EDGE_ACTION_LABEL[q],
            trades_allowed=False,
            requires_confirmation=False,
            size_multiplier=0.0,
            detail=f"edge={cents:.1f}¢ < {experimental*100:.0f}¢ floor",
        )

    requires_conf = q == EdgeQuality.CONDITIONAL
    multipliers = {
        EdgeQuality.EXCEPTIONAL: cfg.size_multiplier_exceptional,
        EdgeQuality.STRONG: cfg.size_multiplier_strong,
        EdgeQuality.CONDITIONAL: cfg.size_multiplier_conditional,
        EdgeQuality.EXPERIMENTAL: cfg.size_multiplier_experimental,
    }
    return EdgeQualityAssessment(
        quality=q,
        net_edge_dollars=net_edge_dollars,
        raw_edge_dollars=raw,
        action_label=EDGE_ACTION_LABEL[q],
        trades_allowed=True,
        requires_confirmation=requires_conf,
        size_multiplier=multipliers[q],
        detail=f"{'raw' if cfg.use_raw_edge_for_tiers else 'net'}={cents:.1f}¢ → {q.value}",
    )


def confirmation_passes(
    *,
    model_confidence: float,
    model_agrees: bool,
    data_fresh: bool,
    liquidity_ok: bool,
    spread_ok: bool,
    no_manipulation: bool,
    config: TierEdgeConfig | None = None,
) -> tuple[bool, str]:
    """Confirmation gate for CONDITIONAL tier (relaxed)."""
    cfg = config or TierEdgeConfig()
    if cfg.conditional_requires_model_agree and not model_agrees:
        return False, "models disagree"
    min_conf = _tier_threshold(cfg, "conditional_min_confidence")
    if min_conf is not None and model_confidence < min_conf:
        return False, f"confidence {model_confidence:.2f} < {min_conf:.2f}"
    if not liquidity_ok:
        return False, "liquidity insufficient"
    if not spread_ok:
        return False, "spread too wide"
    if not no_manipulation:
        return False, "manipulation suspected"
    return True, "confirmed"


def _net_ev_ok_for_tier(
    edge: EdgeQualityAssessment,
    net_edge: float,
    config: TierEdgeConfig,
) -> bool:
    min_strong = _tier_threshold(config, "min_net_edge_strong")
    min_experimental = _tier_threshold(config, "min_net_edge_experimental")
    if edge.quality in (EdgeQuality.EXCEPTIONAL, EdgeQuality.STRONG):
        return net_edge >= (min_strong if min_strong is not None else 0.0)
    if edge.quality == EdgeQuality.EXPERIMENTAL:
        return net_edge >= (min_experimental if min_experimental is not None else 0.0)
    # Conditional: small negative net ok if raw edge is solid
    experimental = _tier_threshold(config, "edge_experimental")
    floor = min_experimental if min_experimental is not None else 0.0
    raw_floor = experimental if experimental is not None else 0.0
    return net_edge >= floor or (
        config.use_raw_edge_for_tiers and edge.raw_edge_dollars >= raw_floor
    )


def should_trade_for_quality(
    edge: EdgeQualityAssessment,
    *,
    model_confidence: float,
    model_agrees: bool,
    data_fresh: bool,
    liquidity_ok: bool,
    spread_ok: bool,
    no_manipulation: bool,
    net_ev_positive: bool,
    net_edge_dollars: float,
    config: TierEdgeConfig | None = None,
) -> tuple[bool, str]:
    """Decide if edge quality tier permits a trade."""
    cfg = config or TierEdgeConfig()
    if not edge.trades_allowed:
        return False, edge.detail

    net_ok = _net_ev_ok_for_tier(edge, net_edge_dollars, cfg)
    experimental = _tier_threshold(cfg, "edge_experimental")
    if not net_ok and not (
        cfg.use_raw_edge_for_tiers
        and experimental is not None
        and edge.raw_edge_dollars >= experimental
    ):
        return False, f"net EV {net_edge_dollars*100:.1f}¢ below floor"

    if edge.quality in (EdgeQuality.EXCEPTIONAL, EdgeQuality.STRONG):
        if not no_manipulation:
            return False, "manipulation suspected"
        return True, edge.action_label

    if edge.quality == EdgeQuality.CONDITIONAL:
        ok, reason = confirmation_passes(
            model_confidence=model_confidence,
            model_agrees=model_agrees,
            data_fresh=data_fresh,
            liquidity_ok=liquidity_ok,
            spread_ok=spread_ok,
            no_manipulation=no_manipulation,
            config=cfg,
        )
        return ok, reason if ok else f"conditional: {reason}"

    if edge.quality == EdgeQuality.EXPERIMENTAL:
        if not spread_ok or not liquidity_ok:
            return False, "experimental: spread/liquidity"
        if not no_manipulation:
            return False, "manipulation suspected"
        return True, edge.action_label

    return False, "no trade tier"


def classify_tier(
    *,
    net_edge_dollars: float,
    model_confidence: float,
    data_fresh: bool,
    liquidity_ok: bool,
    spread_ok: bool,
    model_agrees: bool,
    no_conflicts: bool,
    config: TierEdgeConfig,
    raw_edge_dollars: float | None = None,
) -> TierAssessment:
    eq = classify_edge_quality(
        net_edge_dollars, raw_edge_dollars=raw_edge_dollars, config=config
    )
    mapping = {
        EdgeQuality.EXCEPTIONAL: SetupTier.A_PLUS,
        EdgeQuality.STRONG: SetupTier.A,
        EdgeQuality.CONDITIONAL: SetupTier.B,
        EdgeQuality.EXPERIMENTAL: SetupTier.B,
        EdgeQuality.NO_TRADE: SetupTier.NONE,
    }
    tier = mapping[eq.quality]
    ok, _ = should_trade_for_quality(
        eq,
        model_confidence=model_confidence,
        model_agrees=model_agrees,
        data_fresh=data_fresh,
        liquidity_ok=liquidity_ok,
        spread_ok=spread_ok,
        no_manipulation=no_conflicts,
        net_ev_positive=net_edge_dollars > 0,
        net_edge_dollars=net_edge_dollars,
        config=config,
    )
    return TierAssessment(tier, eq.net_edge_dollars, net_edge_dollars, ok, eq.detail)


def opportunity_score(
    *,
    net_edge_dollars: float,
    model_confidence: float,
    momentum_confirmation: float,
    order_flow_confirmation: float,
    liquidity_score: float,
    spread: float,
    data_fresh: bool,
    model_agrees: bool,
) -> float:
    score = net_edge_dollars * 100 * 2.0
    score += model_confidence * 30.0
    score += abs(momentum_confirmation) * 10.0
    score += abs(order_flow_confirmation) * 15.0
    score += liquidity_score * 10.0
    if data_fresh:
        score += 5.0
    if model_agrees:
        score += 5.0
    score -= spread * 100 * 1.5
    return round(max(0.0, score), 2)


def estimate_slippage(
    *,
    spread: float,
    liquidity_score: float,
    contracts: int = 1,
) -> float:
    half_spread = spread / 2.0
    depth_penalty = max(0.0, 0.015 * (1.0 - liquidity_score))
    size_penalty = 0.003 * max(0, contracts - 1)
    return round(half_spread + depth_penalty + size_penalty, 4)
