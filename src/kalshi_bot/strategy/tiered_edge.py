"""Edge quality tiers — separate trade frequency from trade quality.

| Net edge | Tier        | Action                              |
|----------|-------------|-------------------------------------|
| ≥ 25¢    | EXCEPTIONAL | 🟢 Exceptional opportunity          |
| 20–25¢   | STRONG      | 🟢 Strong trade                     |
| 15–20¢   | CONDITIONAL | 🟡 Trade only with strong confirm.  |
| 10–15¢   | EXPERIMENTAL| 🟠 Small/experimental trade         |
| < 10¢    | NO_TRADE    | 🔴 No trade                         |
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from kalshi_bot.config import TierEdgeConfig


class EdgeQuality(str, Enum):
    EXCEPTIONAL = "EXCEPTIONAL"      # ≥25¢
    STRONG = "STRONG"                # 20–25¢
    CONDITIONAL = "CONDITIONAL"        # 15–20¢
    EXPERIMENTAL = "EXPERIMENTAL"    # 10–15¢
    NO_TRADE = "NO_TRADE"            # <10¢


EDGE_ACTION_LABEL: dict[EdgeQuality, str] = {
    EdgeQuality.EXCEPTIONAL: "🟢 Exceptional opportunity",
    EdgeQuality.STRONG: "🟢 Strong trade",
    EdgeQuality.CONDITIONAL: "🟡 Trade only with strong confirmation",
    EdgeQuality.EXPERIMENTAL: "🟠 Small/experimental trade",
    EdgeQuality.NO_TRADE: "🔴 No trade",
}


# Legacy alias kept for diagnostics compatibility
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


def classify_edge_quality(
    net_edge_dollars: float,
    *,
    raw_edge_dollars: float | None = None,
    config: TierEdgeConfig | None = None,
) -> EdgeQualityAssessment:
    """Classify net edge into quality tier (frequency ≠ quality)."""
    cfg = config or TierEdgeConfig()
    raw = raw_edge_dollars if raw_edge_dollars is not None else net_edge_dollars
    cents = net_edge_dollars * 100

    if cents >= cfg.edge_exceptional * 100:
        q = EdgeQuality.EXCEPTIONAL
        return EdgeQualityAssessment(
            quality=q,
            net_edge_dollars=net_edge_dollars,
            raw_edge_dollars=raw,
            action_label=EDGE_ACTION_LABEL[q],
            trades_allowed=True,
            requires_confirmation=False,
            size_multiplier=cfg.size_multiplier_exceptional,
            detail=f"net={cents:.1f}¢ ≥ {cfg.edge_exceptional*100:.0f}¢ exceptional",
        )
    if cents >= cfg.edge_strong * 100:
        q = EdgeQuality.STRONG
        return EdgeQualityAssessment(
            quality=q,
            net_edge_dollars=net_edge_dollars,
            raw_edge_dollars=raw,
            action_label=EDGE_ACTION_LABEL[q],
            trades_allowed=True,
            requires_confirmation=False,
            size_multiplier=cfg.size_multiplier_strong,
            detail=f"net={cents:.1f}¢ in strong band {cfg.edge_strong*100:.0f}–{cfg.edge_exceptional*100:.0f}¢",
        )
    if cents >= cfg.edge_conditional * 100:
        q = EdgeQuality.CONDITIONAL
        return EdgeQualityAssessment(
            quality=q,
            net_edge_dollars=net_edge_dollars,
            raw_edge_dollars=raw,
            action_label=EDGE_ACTION_LABEL[q],
            trades_allowed=True,
            requires_confirmation=True,
            size_multiplier=cfg.size_multiplier_conditional,
            detail=f"net={cents:.1f}¢ in conditional band {cfg.edge_conditional*100:.0f}–{cfg.edge_strong*100:.0f}¢",
        )
    if cents >= cfg.edge_experimental * 100:
        q = EdgeQuality.EXPERIMENTAL
        return EdgeQualityAssessment(
            quality=q,
            net_edge_dollars=net_edge_dollars,
            raw_edge_dollars=raw,
            action_label=EDGE_ACTION_LABEL[q],
            trades_allowed=True,
            requires_confirmation=False,
            size_multiplier=cfg.size_multiplier_experimental,
            detail=f"net={cents:.1f}¢ in experimental band {cfg.edge_experimental*100:.0f}–{cfg.edge_conditional*100:.0f}¢",
        )

    q = EdgeQuality.NO_TRADE
    return EdgeQualityAssessment(
        quality=q,
        net_edge_dollars=net_edge_dollars,
        raw_edge_dollars=raw,
        action_label=EDGE_ACTION_LABEL[q],
        trades_allowed=False,
        requires_confirmation=False,
        size_multiplier=0.0,
        detail=f"net={cents:.1f}¢ < {cfg.edge_experimental*100:.0f}¢ floor",
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
    """Strong confirmation gate for CONDITIONAL (15–20¢) tier."""
    cfg = config or TierEdgeConfig()
    if not model_agrees:
        return False, "models disagree"
    if model_confidence < cfg.conditional_min_confidence:
        return False, f"confidence {model_confidence:.2f} < {cfg.conditional_min_confidence:.2f}"
    if not data_fresh:
        return False, "data not fresh"
    if not liquidity_ok:
        return False, "liquidity insufficient"
    if not spread_ok:
        return False, "spread too wide"
    if not no_manipulation:
        return False, "manipulation suspected"
    return True, "confirmed"


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
    config: TierEdgeConfig | None = None,
) -> tuple[bool, str]:
    """Decide if edge quality tier permits a trade."""
    if not edge.trades_allowed:
        return False, edge.detail
    if not net_ev_positive:
        return False, "net EV ≤ 0 after fees/slippage"
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
            config=config,
        )
        return ok, reason if ok else f"conditional confirmation failed: {reason}"
    if edge.quality == EdgeQuality.EXPERIMENTAL:
        if not spread_ok or not liquidity_ok:
            return False, "experimental tier needs acceptable spread/liquidity"
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
) -> TierAssessment:
    """Legacy tier mapping for diagnostics (maps edge quality → A+/A/B)."""
    eq = classify_edge_quality(net_edge_dollars, config=config)
    mapping = {
        EdgeQuality.EXCEPTIONAL: SetupTier.A_PLUS,
        EdgeQuality.STRONG: SetupTier.A,
        EdgeQuality.CONDITIONAL: SetupTier.B,
        EdgeQuality.EXPERIMENTAL: SetupTier.B,
        EdgeQuality.NO_TRADE: SetupTier.NONE,
    }
    tier = mapping[eq.quality]
    qualifies = eq.trades_allowed and eq.quality != EdgeQuality.NO_TRADE
    if eq.quality == EdgeQuality.CONDITIONAL:
        ok, _ = confirmation_passes(
            model_confidence=model_confidence,
            model_agrees=model_agrees,
            data_fresh=data_fresh,
            liquidity_ok=liquidity_ok,
            spread_ok=spread_ok,
            no_manipulation=no_conflicts,
            config=config,
        )
        qualifies = ok
    return TierAssessment(tier, eq.net_edge_dollars, net_edge_dollars, qualifies, eq.detail)


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
    """Rank opportunities. Does NOT override hard risk limits."""
    score = net_edge_dollars * 100 * 2.0
    score += model_confidence * 30.0
    score += abs(momentum_confirmation) * 10.0
    score += abs(order_flow_confirmation) * 15.0
    score += liquidity_score * 10.0
    if data_fresh:
        score += 5.0
    if model_agrees:
        score += 8.0
    score -= spread * 100 * 2.0
    if not model_agrees:
        score -= 10.0  # soft penalty, not hard block
    return round(max(0.0, score), 2)


def estimate_slippage(
    *,
    spread: float,
    liquidity_score: float,
    contracts: int = 1,
) -> float:
    """Conservative slippage estimate in dollars per contract."""
    half_spread = spread / 2.0
    depth_penalty = max(0.0, 0.02 * (1.0 - liquidity_score))
    size_penalty = 0.005 * max(0, contracts - 1)
    return round(half_spread + depth_penalty + size_penalty, 4)
