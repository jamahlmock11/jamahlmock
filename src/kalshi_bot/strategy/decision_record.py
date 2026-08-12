"""Structured trade decision records for audit and diagnostics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from kalshi_bot.strategy.rejection_codes import RejectionCode


@dataclass
class DataFreshness:
    cf_benchmark: str  # FRESH | STALE | MISSING | PROXY
    btc_spot: str
    order_book: str
    options_smile: str


@dataclass
class SideEvaluation:
    """Executable analysis for one side (YES or NO)."""

    side: str  # YES | NO
    model_probability: float
    executable_ask: float | None
    raw_edge_dollars: float
    estimated_fee: float
    estimated_slippage: float
    net_edge_dollars: float
    expected_value_per_contract: float
    passes_edge_threshold: bool
    passes_net_ev: bool
    rejection_codes: list[RejectionCode] = field(default_factory=list)


@dataclass
class FilterCheck:
    """Result of one gate in the decision tree."""

    name: str
    passed: bool
    rejection_code: RejectionCode | None = None
    detail: str = ""


@dataclass
class MarketEvaluationRecord:
    """Complete audit trail for one market evaluation."""

    # Identity
    ticker: str
    series: str
    evaluated_at: datetime
    seconds_to_expiry: float
    minutes_to_expiry: float

    # Spot / strike
    spot: float
    spot_source: str
    strike: float

    # Model
    model_prob_up: float
    model_prob_down: float
    model_confidence: float
    model_disagreement_pp: float
    monte_carlo_prob: float
    options_implied_prob: float | None
    calibrated: bool

    # Market prices (executable)
    yes_bid: float | None
    yes_ask: float | None
    no_ask: float | None
    spread: float

    # Both sides
    yes_side: SideEvaluation
    no_side: SideEvaluation
    best_side: str | None  # YES | NO | None
    best_net_edge: float

    # Microstructure
    liquidity_score: float
    bid_ask_imbalance: float
    order_book_depth_bid: float
    order_book_depth_ask: float

    # Data quality
    data_freshness: DataFreshness

    # Filter audit
    filter_checks: list[FilterCheck]
    all_rejection_codes: list[RejectionCode]

    # Tier / scoring (paper analysis)
    setup_tier: str  # A_PLUS | A | B | NONE
    opportunity_score: float

    # Final
    verdict: str  # TRADE_YES | TRADE_NO | NO_TRADE
    primary_rejection: RejectionCode
    contracts: int = 0
    regime: str = ""
    explainability: float = 0.0
    edge_quality: str = "NO_TRADE"
    edge_action: str = "🔴 No trade"
    trade_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["evaluated_at"] = self.evaluated_at.isoformat()
        d["primary_rejection"] = self.primary_rejection.value
        d["all_rejection_codes"] = [c.value for c in self.all_rejection_codes]
        for side_key in ("yes_side", "no_side"):
            d[side_key]["rejection_codes"] = [
                c.value for c in getattr(self, side_key).rejection_codes
            ]
        for fc in d["filter_checks"]:
            if fc.get("rejection_code"):
                fc["rejection_code"] = fc["rejection_code"].value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def summary_text(self) -> str:
        """Human-readable decision record matching audit spec."""
        lines = [
            f"MARKET: {self.ticker}",
            "",
            "MODEL:",
            f"  Probability UP   = {self.model_prob_up * 100:.1f}%",
            f"  Probability DOWN = {self.model_prob_down * 100:.1f}%",
            f"  Confidence       = {self.model_confidence * 100:.1f}%",
            f"  Disagreement     = {self.model_disagreement_pp:.1f}pp",
            "",
            "MARKET:",
            f"  YES ASK = {(self.yes_ask or 0) * 100:.0f}¢",
            f"  NO ASK  = {(self.no_ask or 0) * 100:.0f}¢",
            f"  Spread  = {self.spread * 100:.1f}¢",
            "",
            "EDGE (executable):",
            f"  YES raw = {self.yes_side.raw_edge_dollars * 100:+.1f}¢  "
            f"net = {self.yes_side.net_edge_dollars * 100:+.1f}¢  "
            f"fee = {self.yes_side.estimated_fee * 100:.1f}¢",
            f"  NO  raw = {self.no_side.raw_edge_dollars * 100:+.1f}¢  "
            f"net = {self.no_side.net_edge_dollars * 100:+.1f}¢  "
            f"fee = {self.no_side.estimated_fee * 100:.1f}¢",
            "",
            "DATA:",
            f"  CF Benchmark = {self.data_freshness.cf_benchmark}",
            f"  BTC spot     = {self.data_freshness.btc_spot} ({self.spot_source})",
            f"  Order book   = {self.data_freshness.order_book}",
            "",
            f"LIQUIDITY: {'PASS' if self.liquidity_score >= 0.15 else 'FAIL'} "
            f"(score={self.liquidity_score:.2f})",
            f"SPREAD:    {'PASS' if self.spread <= 0.08 else 'FAIL'}",
            f"TIMING:    {self.minutes_to_expiry:.1f}m remaining",
            f"CONFIDENCE: {self.model_confidence * 100:.0f}%",
            "",
            f"TIER: {self.setup_tier}  |  {self.edge_action}",
            f"OPPORTUNITY SCORE: {self.opportunity_score:.2f}",
            "",
            f"FINAL: {self.verdict}",
            f"REASON: {self.primary_rejection.value}",
        ]
        if self.all_rejection_codes:
            codes = ", ".join(c.value for c in self.all_rejection_codes if c != RejectionCode.NONE)
            if codes:
                lines.append(f"ALL REJECTIONS: {codes}")
        return "\n".join(lines)


def pick_primary_rejection(codes: list[RejectionCode]) -> RejectionCode:
    """Select the most informative primary rejection from a list."""
    priority = [
        RejectionCode.RULES_NOT_CONFIGURED,
        RejectionCode.KILL_SWITCH,
        RejectionCode.RISK_LIMIT,
        RejectionCode.COOLDOWN,
        RejectionCode.MISSING_DATA,
        RejectionCode.STALE_DATA,
        RejectionCode.API_ERROR,
        RejectionCode.MODEL_CONFLICT,
        RejectionCode.MODEL_UNAVAILABLE,
        RejectionCode.MANIPULATION_SUSPECTED,
        RejectionCode.QUALITY_SCORE_TOO_HIGH,
        RejectionCode.SPREAD_TOO_WIDE,
        RejectionCode.INSUFFICIENT_LIQUIDITY,
        RejectionCode.FAKE_BREAKOUT,
        RejectionCode.LOW_CONFIDENCE,
        RejectionCode.PATTERN_EVIDENCE_INSUFFICIENT,
        RejectionCode.EXPECTED_VALUE_NEGATIVE,
        RejectionCode.EDGE_TOO_SMALL,
        RejectionCode.TIMING_RESTRICTION,
        RejectionCode.PRICE_DATA_ERROR,
    ]
    code_set = set(codes)
    for code in priority:
        if code in code_set:
            return code
    return RejectionCode.EDGE_TOO_SMALL if codes else RejectionCode.NONE
