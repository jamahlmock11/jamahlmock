"""Per-side trade gate evaluation for the 15m mispricing dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from kalshi_bot.config import ArbitraryPolicyConfig, TradeGatesConfig
from kalshi_bot.strategy.arbitrary_policy import market_yes_mid
from kalshi_bot.strategy.time_buckets import BucketPolicy, TimeBucket, bucket_policy, classify_time_bucket


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


@dataclass(frozen=True)
class TradeGate:
    name: str
    status: GateStatus
    detail: str
    side: str | None = None  # YES | NO | None for global gates


@dataclass(frozen=True)
class TradeGatesResult:
    gates: tuple[TradeGate, ...]
    yes_passes_all: bool
    no_passes_all: bool
    ready_side: str | None
    position_detail: str
    crowd_yes_pct: float
    crowd_direction: str
    uncertainty_pct: float
    min_net_ev: float
    time_bucket: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gates": [
                {
                    "name": g.name,
                    "status": g.status.value,
                    "detail": g.detail,
                    "side": g.side,
                }
                for g in self.gates
            ],
            "yes_passes_all": self.yes_passes_all,
            "no_passes_all": self.no_passes_all,
            "ready_side": self.ready_side,
            "position_detail": self.position_detail,
            "crowd_yes_pct": round(self.crowd_yes_pct, 1),
            "crowd_direction": self.crowd_direction,
            "uncertainty_pct": round(self.uncertainty_pct, 1),
            "min_net_ev": self.min_net_ev,
            "time_bucket": self.time_bucket,
        }


def _fmt_ev(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.3f}"


def evaluate_trade_gates(
    *,
    model_prob_yes: float,
    yes_net_ev: float,
    no_net_ev: float,
    yes_ask: float | None,
    yes_bid: float | None,
    no_ask: float | None,
    seconds_to_expiry: float,
    uncertainty_pct: float,
    contracts: int = 0,
    min_seconds: float = 60.0,
    max_seconds: float = 900.0,
    bucket_overrides: dict[str, dict] | None = None,
    gates_cfg: TradeGatesConfig | None = None,
    arbitrary_cfg: ArbitraryPolicyConfig | None = None,
) -> TradeGatesResult:
    """Evaluate the gate checklist shown on the 15m trade dashboard."""
    cfg = gates_cfg or TradeGatesConfig()
    arb = arbitrary_cfg or ArbitraryPolicyConfig(enabled=False)
    gates: list[TradeGate] = []

    mins = seconds_to_expiry / 60.0
    min_mins = cfg.min_minutes_to_expiry
    max_mins = cfg.max_minutes_to_expiry
    time_ok = min_mins <= mins <= max_mins
    gates.append(
        TradeGate(
            name="Time to expiry",
            status=GateStatus.PASS if time_ok else GateStatus.FAIL,
            detail=(
                f"{mins:.1f} min to expiry (allowed {min_mins:g}–{max_mins:g} min)"
                if time_ok
                else (
                    f"{mins:.1f} min — outside {min_mins:g}–{max_mins:g} min window"
                )
            ),
        )
    )

    uncertainty_ok = uncertainty_pct <= cfg.uncertainty_cap * 100.0
    gates.append(
        TradeGate(
            name="Uncertainty",
            status=GateStatus.PASS if uncertainty_ok else GateStatus.FAIL,
            detail=(
                f"{uncertainty_pct:.1f}% ≤ {cfg.uncertainty_cap * 100:.0f}% cap"
                if uncertainty_ok
                else f"{uncertainty_pct:.1f}% > {cfg.uncertainty_cap * 100:.0f}% cap — too uncertain"
            ),
        )
    )

    crowd_yes = market_yes_mid(yes_bid, yes_ask)
    if crowd_yes is None and yes_ask is not None:
        crowd_yes = float(yes_ask)
    crowd_yes_pct = (crowd_yes or 0.5) * 100.0
    crowd_no_pct = 100.0 - crowd_yes_pct
    crowd_direction = "UP" if crowd_yes_pct >= 50 else "DOWN"

    yes_align_ok = model_prob_yes >= cfg.yes_alignment_min_forecast
    gates.append(
        TradeGate(
            name="BUY YES alignment",
            status=GateStatus.PASS if yes_align_ok else GateStatus.FAIL,
            side="YES",
            detail=(
                f"forecast {model_prob_yes * 100:.0f}% ≥ {cfg.yes_alignment_min_forecast * 100:.0f}%"
                if yes_align_ok
                else f"forecast {model_prob_yes * 100:.0f}% < {cfg.yes_alignment_min_forecast * 100:.0f}%"
            ),
        )
    )

    bucket = classify_time_bucket(seconds_to_expiry, min_seconds=min_seconds, max_seconds=max_seconds)
    policy = bucket_policy(bucket, bucket_overrides)
    min_net_ev = policy.min_net_edge_dollars if policy is not None else 0.12

    yes_ev_ok = yes_net_ev >= min_net_ev
    gates.append(
        TradeGate(
            name="BUY YES NET EV",
            status=GateStatus.PASS if yes_ev_ok else GateStatus.FAIL,
            side="YES",
            detail=(
                f"{_fmt_ev(yes_net_ev)} ≥ {min_net_ev:.3f} required"
                if yes_ev_ok
                else f"{_fmt_ev(yes_net_ev)} < {min_net_ev:.3f} required"
            ),
        )
    )

    no_align_threshold = cfg.no_alignment_min_crowd_pct * 100.0
    no_align_ok = crowd_no_pct >= no_align_threshold
    gates.append(
        TradeGate(
            name="BUY NO alignment",
            status=GateStatus.PASS if no_align_ok else GateStatus.FAIL,
            side="NO",
            detail=(
                f"crowd DOWN {crowd_no_pct:.0f}% (need BUY NO alignment ≥ {no_align_threshold:.0f}%)"
                if no_align_ok
                else (
                    f"crowd {crowd_direction} {crowd_yes_pct:.0f}% "
                    f"(need BUY NO alignment ≥ {no_align_threshold:.0f}%)"
                )
            ),
        )
    )

    no_ev_ok = no_net_ev >= min_net_ev
    gates.append(
        TradeGate(
            name="BUY NO NET EV",
            status=GateStatus.PASS if no_ev_ok else GateStatus.FAIL,
            side="NO",
            detail=(
                f"{_fmt_ev(no_net_ev)} ≥ {min_net_ev:.3f} required"
                if no_ev_ok
                else f"{_fmt_ev(no_net_ev)} < {min_net_ev:.3f} required"
            ),
        )
    )

    global_ok = time_ok and uncertainty_ok
    yes_passes = global_ok and yes_align_ok and yes_ev_ok
    no_passes = global_ok and no_align_ok and no_ev_ok

    ready_side: str | None = None
    if yes_passes and not no_passes:
        ready_side = "YES"
    elif no_passes and not yes_passes:
        ready_side = "NO"
    elif yes_passes and no_passes:
        ready_side = "YES" if yes_net_ev >= no_net_ev else "NO"

    if ready_side and contracts > 0:
        position_detail = f"{ready_side} x{contracts} contracts"
        position_status = GateStatus.PASS
    elif ready_side:
        position_detail = f"Ready: BUY {ready_side} (size pending Kelly)"
        position_status = GateStatus.WARN
    else:
        position_detail = "Waiting for a side to clear all gates"
        position_status = GateStatus.WARN

    gates.append(
        TradeGate(
            name="Position size",
            status=position_status,
            detail=position_detail,
        )
    )

    return TradeGatesResult(
        gates=tuple(gates),
        yes_passes_all=yes_passes,
        no_passes_all=no_passes,
        ready_side=ready_side,
        position_detail=position_detail,
        crowd_yes_pct=crowd_yes_pct,
        crowd_direction=crowd_direction,
        uncertainty_pct=uncertainty_pct,
        min_net_ev=min_net_ev,
        time_bucket=bucket.value if bucket not in (TimeBucket.TOO_EARLY, TimeBucket.TOO_LATE) else bucket.value,
    )
