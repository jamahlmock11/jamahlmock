"""Live trading safety gate — blocks orders when data or systems are unhealthy."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from kalshi_bot.platform.observation import DataQuality, ObservationBundle


@dataclass
class PlatformStatus:
    trading_enabled: bool
    live_mode: bool
    balance_usd: float | None
    available_balance_usd: float | None
    open_exposure_usd: float
    daily_pnl_usd: float
    daily_loss_limit_usd: float
    open_positions: int
    api_connected: bool
    market_data_connected: bool
    last_update_at: float
    last_order_confirmed_at: float | None
    block_reason: str | None
    model_version: str
    candidate_model_version: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trading_enabled": self.trading_enabled,
            "live_mode": self.live_mode,
            "status_label": "LIVE" if self.trading_enabled and self.live_mode else "DISABLED",
            "balance_usd": self.balance_usd,
            "available_balance_usd": self.available_balance_usd,
            "open_exposure_usd": self.open_exposure_usd,
            "daily_pnl_usd": self.daily_pnl_usd,
            "daily_loss_limit_usd": self.daily_loss_limit_usd,
            "open_positions": self.open_positions,
            "api_connected": self.api_connected,
            "market_data_connected": self.market_data_connected,
            "last_update_age_s": round(time.time() - self.last_update_at, 1),
            "last_order_confirmed_at": self.last_order_confirmed_at,
            "block_reason": self.block_reason,
            "model_version": self.model_version,
            "candidate_model_version": self.candidate_model_version,
        }


class LiveSafetyGate:
    """Central kill switch and health checks before any live order."""

    def __init__(
        self,
        *,
        trading_enabled: bool,
        live_mode: bool,
        max_data_age_seconds: float = 12.0,
        daily_loss_limit_usd: float = 50.0,
        model_version: str = "production-v1",
    ) -> None:
        self.trading_enabled = trading_enabled
        self.live_mode = live_mode
        self.max_data_age_seconds = max_data_age_seconds
        self.daily_loss_limit_usd = daily_loss_limit_usd
        self.model_version = model_version
        self.candidate_model_version: str | None = None
        self._daily_pnl_usd = 0.0
        self._open_exposure_usd = 0.0
        self._open_positions = 0
        self._api_ok = True
        self._market_data_ok = True
        self._last_update = time.time()
        self._last_order_confirmed: float | None = None
        self._block_reason: str | None = None
        self._balance_usd: float | None = None

    def update_connectivity(self, *, api_ok: bool, market_data_ok: bool) -> None:
        self._api_ok = api_ok
        self._market_data_ok = market_data_ok
        self._last_update = time.time()
        if not api_ok:
            self._block_reason = "Kalshi API unavailable"
        elif not market_data_ok:
            self._block_reason = "Market data feed unhealthy"
        elif self._block_reason in ("Kalshi API unavailable", "Market data feed unhealthy"):
            self._block_reason = None

    def update_balance(self, balance_usd: float | None) -> None:
        self._balance_usd = balance_usd
        self._last_update = time.time()

    def update_exposure(self, *, exposure_usd: float, open_positions: int) -> None:
        self._open_exposure_usd = exposure_usd
        self._open_positions = open_positions

    def record_order_confirmed(self) -> None:
        self._last_order_confirmed = time.time()

    def record_pnl(self, pnl_usd: float) -> None:
        self._daily_pnl_usd += pnl_usd

    def allow_new_orders(self, observations: ObservationBundle | None = None) -> tuple[bool, str]:
        if not self.trading_enabled:
            return False, "Trading disabled by configuration (platform.trading_enabled=false)"
        if not self.live_mode:
            return False, "Not in live execution mode"
        if not self._api_ok:
            return False, "Kalshi API connection failed"
        if not self._market_data_ok:
            return False, "Market data connection unhealthy"
        if self._daily_pnl_usd <= -self.daily_loss_limit_usd:
            return False, f"Daily loss limit reached (${self._daily_loss_limit_usd:.2f})"
        if observations is not None:
            q = observations.worst_quality()
            if q in (DataQuality.STALE, DataQuality.MISSING, DataQuality.ERROR):
                return False, f"Data quality {q.value} — stop new orders"
        return True, "ok"

    @property
    def _daily_loss_limit_usd(self) -> float:
        return self.daily_loss_limit_usd

    def status(self) -> PlatformStatus:
        allow, reason = self.allow_new_orders()
        return PlatformStatus(
            trading_enabled=self.trading_enabled,
            live_mode=self.live_mode,
            balance_usd=self._balance_usd,
            available_balance_usd=self._balance_usd,
            open_exposure_usd=self._open_exposure_usd,
            daily_pnl_usd=self._daily_pnl_usd,
            daily_loss_limit_usd=self.daily_loss_limit_usd,
            open_positions=self._open_positions,
            api_connected=self._api_ok,
            market_data_connected=self._market_data_ok,
            last_update_at=self._last_update,
            last_order_confirmed_at=self._last_order_confirmed,
            block_reason=None if allow else reason,
            model_version=self.model_version,
            candidate_model_version=self.candidate_model_version,
        )
