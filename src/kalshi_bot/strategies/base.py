"""Strategy engine interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from kalshi_bot.platform.observation import ObservationBundle


@dataclass
class MarketDecision:
    ticker: str
    series: str
    signal: str
    reason: str
    model_prob_yes: float
    model_prob_no: float
    fair_yes: float
    fair_no: float
    executable_yes: float | None
    executable_no: float | None
    raw_edge: float
    net_edge: float
    confidence: float
    regime: str
    strike: float
    seconds_to_expiry: float
    btc_spot: float
    yes_price: float | None
    no_price: float | None
    why_trade: str
    why_not_trade: str
    observations: ObservationBundle = field(default_factory=ObservationBundle)
    features: dict[str, Any] = field(default_factory=dict)
    execute_verdict: str | None = None  # TRADE_YES | TRADE_NO | None
    contracts: int = 0


class StrategyEngine(ABC):
    name: str
    series: str

    @abstractmethod
    def evaluate_all(self) -> list[MarketDecision]:
        ...
