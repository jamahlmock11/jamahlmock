"""Standardized machine-readable rejection codes for trade decisions."""

from __future__ import annotations

from enum import Enum


class RejectionCode(str, Enum):
    """Primary reason a market evaluation did not produce a trade."""

    NONE = "NONE"  # trade allowed
    NO_MARKET = "NO_MARKET"
    STALE_DATA = "STALE_DATA"
    MISSING_DATA = "MISSING_DATA"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    EDGE_TOO_SMALL = "EDGE_TOO_SMALL"
    SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
    INSUFFICIENT_LIQUIDITY = "INSUFFICIENT_LIQUIDITY"
    TIMING_RESTRICTION = "TIMING_RESTRICTION"
    RISK_LIMIT = "RISK_LIMIT"
    DUPLICATE_POSITION = "DUPLICATE_POSITION"
    COOLDOWN = "COOLDOWN"
    API_ERROR = "API_ERROR"
    PRICE_DATA_ERROR = "PRICE_DATA_ERROR"
    MODEL_CONFLICT = "MODEL_CONFLICT"
    EXPECTED_VALUE_NEGATIVE = "EXPECTED_VALUE_NEGATIVE"
    MANIPULATION_SUSPECTED = "MANIPULATION_SUSPECTED"
    FAKE_BREAKOUT = "FAKE_BREAKOUT"
    PATTERN_EVIDENCE_INSUFFICIENT = "PATTERN_EVIDENCE_INSUFFICIENT"
    QUALITY_SCORE_TOO_HIGH = "QUALITY_SCORE_TOO_HIGH"
    KILL_SWITCH = "KILL_SWITCH"


# Human-readable descriptions for reports.
REJECTION_DESCRIPTIONS: dict[RejectionCode, str] = {
    RejectionCode.NONE: "Trade passed all gates",
    RejectionCode.NO_MARKET: "No open market in evaluation window",
    RejectionCode.STALE_DATA: "Price or benchmark data is stale",
    RejectionCode.MISSING_DATA: "Required field missing (book, strike, close)",
    RejectionCode.MODEL_UNAVAILABLE: "Probability model could not run",
    RejectionCode.LOW_CONFIDENCE: "Ensemble confidence below floor",
    RejectionCode.EDGE_TOO_SMALL: "Raw/net edge below minimum threshold",
    RejectionCode.SPREAD_TOO_WIDE: "Bid-ask spread exceeds limit",
    RejectionCode.INSUFFICIENT_LIQUIDITY: "Order book depth too thin",
    RejectionCode.TIMING_RESTRICTION: "Outside allowed seconds-to-expiry window",
    RejectionCode.RISK_LIMIT: "Exposure or position limit reached",
    RejectionCode.DUPLICATE_POSITION: "Already holding this ticker",
    RejectionCode.COOLDOWN: "Post-loss cooldown active",
    RejectionCode.API_ERROR: "Kalshi or data API failure",
    RejectionCode.PRICE_DATA_ERROR: "Invalid or extreme executable price",
    RejectionCode.MODEL_CONFLICT: "Multi-model disagreement above ceiling",
    RejectionCode.EXPECTED_VALUE_NEGATIVE: "Net edge after fees/slippage ≤ 0",
    RejectionCode.MANIPULATION_SUSPECTED: "Manipulation detector flagged market",
    RejectionCode.FAKE_BREAKOUT: "Price action suggests fake breakout",
    RejectionCode.PATTERN_EVIDENCE_INSUFFICIENT: "Not enough similar historical setups",
    RejectionCode.QUALITY_SCORE_TOO_HIGH: "Do-not-trade composite score too high",
    RejectionCode.KILL_SWITCH: "Daily loss or consecutive-loss kill switch active",
}
