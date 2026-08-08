"""Gap-tiered bot action policy for YES/NO buys.

Canonical rule (raw gap before fees):

    gap_pp = (model_probability − market_price) × 100

Reference matrix at model probability = 60%:

| Market YES | Gap  | Bot action                          |
|------------|------|-------------------------------------|
| 35¢        | 25pp | Strong BUY candidate                |
| 40¢        | 20pp | Strong BUY candidate                |
| 45¢        | 15pp | Only if other signals confirm       |
| 50¢        | 10pp | No trade                            |
| 55¢        |  5pp | No trade                            |

Thresholds (defaults):
- gap ≥ 20pp → STRONG_BUY
- 15pp ≤ gap < 20pp → CONDITIONAL
- gap < 15pp → NO_TRADE
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from kalshi_bot.config import BotActionConfig


class BotAction(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    CONDITIONAL = "CONDITIONAL"
    NO_TRADE = "NO_TRADE"

    @property
    def label(self) -> str:
        return {
            BotAction.STRONG_BUY: "Strong BUY candidate",
            BotAction.CONDITIONAL: "Only if other signals confirm",
            BotAction.NO_TRADE: "No trade",
        }[self]


@dataclass(frozen=True)
class GapAssessment:
    """Result of classifying model vs market gap."""

    model_probability: float
    market_price: float
    gap_pp: float
    action: BotAction

    @property
    def label(self) -> str:
        return self.action.label


def raw_gap_pp(model_probability: float, market_price: float) -> float:
    """Raw edge in percentage points: model − executable market price."""
    # Round to 0.01pp to avoid binary float artifacts (e.g. 0.60−0.40 → 19.999…).
    return round((model_probability - market_price) * 100.0, 2)


def classify_buy_gap(
    gap_pp: float,
    *,
    config: BotActionConfig | None = None,
    strong_buy_min_gap_pp: float | None = None,
    conditional_min_gap_pp: float | None = None,
) -> BotAction:
    """Map a raw gap (pp) onto the bot-action tier."""
    cfg = config or BotActionConfig()
    strong = (
        strong_buy_min_gap_pp
        if strong_buy_min_gap_pp is not None
        else cfg.strong_buy_min_gap_pp
    )
    conditional = (
        conditional_min_gap_pp
        if conditional_min_gap_pp is not None
        else cfg.conditional_min_gap_pp
    )
    if gap_pp >= strong:
        return BotAction.STRONG_BUY
    if gap_pp >= conditional:
        return BotAction.CONDITIONAL
    return BotAction.NO_TRADE


def assess_buy_gap(
    model_probability: float,
    market_price: float,
    *,
    config: BotActionConfig | None = None,
) -> GapAssessment:
    """Full assessment: model vs market → gap + action tier."""
    gap = raw_gap_pp(model_probability, market_price)
    return GapAssessment(
        model_probability=model_probability,
        market_price=market_price,
        gap_pp=gap,
        action=classify_buy_gap(gap, config=config),
    )


def other_signals_confirm(
    *,
    confidence: float,
    disagreement_pp: float,
    sufficient_evidence: bool,
    config: BotActionConfig | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Whether secondary signals confirm a CONDITIONAL-tier entry.

    STRONG_BUY does not require this elevated bar; CONDITIONAL does.
    """
    cfg = config or BotActionConfig()
    failures: list[str] = []
    if not sufficient_evidence:
        failures.append("ensemble evidence insufficient for conditional entry")
    if confidence < cfg.conditional_min_confidence:
        failures.append(
            f"conditional confidence {confidence:.2f} < "
            f"{cfg.conditional_min_confidence:.2f}"
        )
    if disagreement_pp > cfg.conditional_max_disagreement_pp:
        failures.append(
            f"conditional disagreement {disagreement_pp:.1f}pp > "
            f"{cfg.conditional_max_disagreement_pp:.1f}pp"
        )
    return (not failures, tuple(failures))
