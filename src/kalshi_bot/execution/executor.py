"""Order execution: paper and live."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from kalshi_bot.config import BotConfig
from kalshi_bot.data.kalshi_client import KalshiClient
from kalshi_bot.execution.risk import RiskManager
from kalshi_bot.strategy.mispricing import Mispricing, Side

logger = logging.getLogger(__name__)


@dataclass
class Fill:
    ticker: str
    side: str
    price: float
    contracts: int
    mode: str
    ts: float
    edge_after_fees_pp: float
    options_prob: float
    raw: dict[str, Any] = field(default_factory=dict)


class Executor:
    def __init__(self, client: KalshiClient, config: BotConfig, risk: RiskManager) -> None:
        self.client = client
        self.config = config
        self.risk = risk
        self.fills: list[Fill] = []

    def execute(self, mis: Mispricing, contracts: int, *, ignore_cooldown: bool = False) -> Fill | None:
        if contracts <= 0:
            logger.info("skip %s: size=0 (bankroll too small for Kelly)", mis.ticker)
            return None
        ok, reason = self.risk.allow(mis, ignore_cooldown=ignore_cooldown)
        if not ok:
            logger.info("skip %s: %s", mis.ticker, reason)
            return None

        mode = self.config.execution.mode
        price_str = f"{mis.kalshi_price:.4f}"
        action = "buy"
        side = mis.side.value

        if mode == "paper" or self.config.execution.dry_run or not self.client.authenticated:
            fill = Fill(
                ticker=mis.ticker,
                side=side,
                price=mis.kalshi_price,
                contracts=contracts,
                mode="paper",
                ts=time.time(),
                edge_after_fees_pp=mis.edge_after_fees_pp,
                options_prob=mis.options_prob,
                raw={"reason": mis.reason},
            )
            self.fills.append(fill)
            self.risk.register_fill(mis, contracts)
            logger.info(
                "PAPER FILL %s %s x%d @ %.4f edge=%.1fpp | %s",
                side.upper(),
                mis.ticker,
                contracts,
                mis.kalshi_price,
                mis.edge_after_fees_pp,
                mis.reason,
            )
            return fill

        body_kwargs: dict[str, Any] = {
            "ticker": mis.ticker,
            "side": side,
            "action": action,
            "count": contracts,
            "time_in_force": self.config.execution.time_in_force,
        }
        if side == Side.YES.value:
            body_kwargs["yes_price_dollars"] = price_str
        else:
            body_kwargs["no_price_dollars"] = price_str

        resp = self.client.create_order(**body_kwargs)
        fill = Fill(
            ticker=mis.ticker,
            side=side,
            price=mis.kalshi_price,
            contracts=contracts,
            mode="live",
            ts=time.time(),
            edge_after_fees_pp=mis.edge_after_fees_pp,
            options_prob=mis.options_prob,
            raw=resp or {},
        )
        self.fills.append(fill)
        self.risk.register_fill(mis, contracts)
        logger.info(
            "LIVE ORDER %s %s x%d @ %.4f edge=%.1fpp",
            side.upper(),
            mis.ticker,
            contracts,
            mis.kalshi_price,
            mis.edge_after_fees_pp,
        )
        return fill

    def close_position(
        self,
        *,
        ticker: str,
        side: str,
        contracts: int,
        price: float,
        reason: str,
    ) -> Fill | None:
        """Sell an open position at the current bid (early exit)."""
        if contracts <= 0 or price <= 0:
            return None

        mode = self.config.execution.mode
        price_str = f"{price:.4f}"

        if mode == "paper" or self.config.execution.dry_run or not self.client.authenticated:
            fill = Fill(
                ticker=ticker,
                side=side,
                price=price,
                contracts=contracts,
                mode="paper",
                ts=time.time(),
                edge_after_fees_pp=0.0,
                options_prob=0.0,
                raw={"action": "sell", "reason": reason},
            )
            self.fills.append(fill)
            logger.info(
                "PAPER EXIT %s %s x%d @ %.4f | %s",
                side.upper(),
                ticker,
                contracts,
                price,
                reason,
            )
            return fill

        body_kwargs: dict[str, Any] = {
            "ticker": ticker,
            "side": side,
            "action": "sell",
            "count": contracts,
            "time_in_force": self.config.execution.time_in_force,
        }
        if side == Side.YES.value:
            body_kwargs["yes_price_dollars"] = price_str
        else:
            body_kwargs["no_price_dollars"] = price_str

        resp = self.client.create_order(**body_kwargs)
        fill = Fill(
            ticker=ticker,
            side=side,
            price=price,
            contracts=contracts,
            mode="live",
            ts=time.time(),
            edge_after_fees_pp=0.0,
            options_prob=0.0,
            raw=resp or {"action": "sell", "reason": reason},
        )
        self.fills.append(fill)
        logger.info(
            "LIVE EXIT %s %s x%d @ %.4f | %s",
            side.upper(),
            ticker,
            contracts,
            price,
            reason,
        )
        return fill
