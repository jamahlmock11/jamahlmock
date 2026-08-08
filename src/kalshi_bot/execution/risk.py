"""Position sizing and risk gates."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from kalshi_bot.config import BotConfig, RiskConfig, SeriesConfig
from kalshi_bot.strategy.mispricing import Mispricing, Side


@dataclass
class RiskState:
    open_positions: int = 0
    exposure_usd: float = 0.0
    last_trade_ts: float = 0.0
    traded_tickers: set[str] = field(default_factory=set)


def kelly_contracts(
    mis: Mispricing,
    *,
    bankroll: float,
    kelly_fraction: float,
    max_contracts: int,
    max_notional: float,
    max_loss: float,
) -> int:
    """Fractional Kelly sizing for a binary contract.

    For buy at price c with win prob p:
      b = (1-c)/c  (net odds)
      f* = (p*b - (1-p)) / b = (p - c) / (1 - c)
    """
    c = mis.kalshi_price
    p = mis.options_prob
    if c <= 0 or c >= 1 or p <= c:
        return 0
    f_star = (p - c) / (1.0 - c)
    f = max(0.0, min(1.0, f_star * kelly_fraction))
    dollars = min(bankroll * f, max_notional, max_loss / max(c, 1e-6))
    contracts = int(dollars / c)
    return max(0, min(contracts, max_contracts))


class RiskManager:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.state = RiskState()

    def series_cfg(self, ticker_series: str) -> SeriesConfig | None:
        for s in self.config.series:
            if s.ticker == ticker_series:
                return s
        return None

    def allow(self, mis: Mispricing, *, ignore_cooldown: bool = False) -> tuple[bool, str]:
        risk: RiskConfig = self.config.risk
        now = time.time()
        if not ignore_cooldown and now - self.state.last_trade_ts < risk.cooldown_seconds:
            return False, "cooldown"
        if self.state.open_positions >= risk.max_open_positions:
            return False, "max_open_positions"
        if mis.ticker in self.state.traded_tickers:
            return False, "already_traded_ticker"
        if mis.seconds_to_expiry < risk.min_seconds_to_expiry:
            return False, "too_close_to_expiry"
        # Hourly markets: refuse entries outside the last-20-minute window.
        if mis.series == "KXBTCD" and mis.seconds_to_expiry > risk.max_seconds_to_expiry_1h:
            return False, "outside_last_20m_window"
        # Extreme probs rarely have real edge after microstructure
        if mis.options_prob < 0.03 or mis.options_prob > 0.97:
            return False, "prob_extreme"
        if mis.kalshi_price < 0.02 or mis.kalshi_price > 0.98:
            return False, "price_extreme"
        return True, "ok"

    def size(self, mis: Mispricing) -> int:
        sc = self.series_cfg(mis.series)
        if sc is None:
            return 0
        risk = self.config.risk
        remaining = max(0.0, risk.max_exposure_usd - self.state.exposure_usd)
        notional_cap = min(sc.max_notional_usd, remaining)
        return kelly_contracts(
            mis,
            bankroll=risk.bankroll_usd,
            kelly_fraction=risk.kelly_fraction,
            max_contracts=sc.max_contracts,
            max_notional=notional_cap,
            max_loss=risk.max_loss_per_trade_usd,
        )

    def register_fill(self, mis: Mispricing, contracts: int) -> None:
        self.state.open_positions += 1
        self.state.exposure_usd += contracts * mis.kalshi_price
        self.state.last_trade_ts = time.time()
        self.state.traded_tickers.add(mis.ticker)
