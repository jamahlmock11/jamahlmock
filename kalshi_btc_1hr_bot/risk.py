"""Risk management: bankroll caps, drawdown stop, position limits."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from kalshi_btc_1hr_bot.config import BotConfig, RiskConfig


@dataclass
class RiskState:
    daily_pnl: float = 0.0
    day_start_balance: float = 0.0
    open_positions: int = 0
    traded_tickers: set[str] = field(default_factory=set)
    last_trade_ts: float = 0.0
    day_start: float = field(default_factory=time.time)


class RiskManager:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.risk: RiskConfig = config.risk
        self.state = RiskState()

    def reset_daily_if_needed(self) -> None:
        now = time.time()
        if now - self.state.day_start > 86400:
            self.state.daily_pnl = 0.0
            self.state.day_start_balance = 0.0
            self.state.day_start = now
            self.state.traded_tickers.clear()

    def sync_from_journal(self, journal: Any, *, balance_usd: float | None = None) -> None:
        """Restore daily PnL and day-start bankroll from journal + live balance."""
        from kalshi_btc_1hr_bot.trade_journal import daily_pnl_today

        self.reset_daily_if_needed()
        self.state.daily_pnl = daily_pnl_today(journal)
        if balance_usd is not None and balance_usd > 0:
            if self.state.day_start_balance <= 0:
                self.state.day_start_balance = balance_usd
            self.config.sizing.bankroll_usd = balance_usd

    def allow_trade(
        self,
        *,
        ticker: str,
        seconds_to_expiry: float,
    ) -> tuple[bool, str]:
        self.reset_daily_if_needed()
        bankroll = self.state.day_start_balance or self.config.sizing.bankroll_usd
        stop = bankroll * self.config.risk.daily_loss_stop_pct

        if self.state.daily_pnl <= -stop:
            return False, "daily_loss_stop"
        if self.state.open_positions >= self.risk.max_open_positions:
            return False, "max_open_positions"
        if ticker in self.state.traded_tickers:
            return False, "already_traded"
        if seconds_to_expiry < self.risk.min_seconds_to_expiry:
            return False, "too_close_to_expiry"
        if seconds_to_expiry > self.risk.max_seconds_to_expiry:
            return False, "too_early_in_window"
        now = time.time()
        if now - self.state.last_trade_ts < self.risk.cooldown_seconds:
            return False, "cooldown"
        return True, "ok"

    def register_trade(self, ticker: str, cost: float) -> None:
        self.state.open_positions += 1
        self.state.traded_tickers.add(ticker)
        self.state.last_trade_ts = time.time()
        self.state.daily_pnl -= cost

    def record_pnl(self, pnl: float) -> None:
        self.state.daily_pnl += pnl

    def release_position(self, ticker: str) -> None:
        self.state.open_positions = max(0, self.state.open_positions - 1)
        self.state.traded_tickers.discard(ticker)

    def close_position(self, ticker: str, proceeds_usd: float) -> None:
        """Release an open position and credit sale proceeds to daily PnL."""
        self.release_position(ticker)
        self.state.daily_pnl += proceeds_usd
