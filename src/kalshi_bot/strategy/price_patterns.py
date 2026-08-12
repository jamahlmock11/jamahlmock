"""Short-horizon BTC price patterns for 15m settlement context."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from kalshi_bot.data.btc_data_engine import BtcMarketSnapshot


class PricePattern(str, Enum):
    DRIFT = "Drift"
    HAMMER = "Hammer"
    FADE = "Fade"
    NONE = "None"


@dataclass(frozen=True)
class PatternAssessment:
    pattern: PricePattern
    finish_side: str  # YES | NO | NEUTRAL — which settlement side BTC is leaning toward
    confidence: float  # 0..1
    detail: str


def detect_price_pattern(
    btc: BtcMarketSnapshot,
    *,
    spot: float,
    strike: float,
    seconds_to_expiry: float,
) -> PatternAssessment:
    """Classify Drift / Hammer / Fade from live BTC microstructure."""
    m1 = btc.momentum_1m
    m3 = btc.momentum_3m
    m5 = btc.momentum_5m
    accel = btc.acceleration
    vol_ratio = btc.volume_ratio

    above_strike = spot >= strike
    finish_side = "YES" if above_strike else "NO"
    if abs(spot - strike) / max(spot, 1.0) < 0.0003:
        finish_side = "NEUTRAL"

    aligned_short_med = (m1 > 0 and m3 > 0) or (m1 < 0 and m3 < 0)
    low_accel = abs(accel) < 0.0004
    steady_vel = 0.0004 <= abs(m1) <= 0.004

    # Drift — steady grind toward the finish side
    if aligned_short_med and low_accel and steady_vel:
        lean = "YES" if m1 > 0 else "NO"
        conf = min(0.9, 0.5 + abs(m1) * 80 + (0.15 if lean == finish_side else 0))
        return PatternAssessment(
            PricePattern.DRIFT,
            lean,
            conf,
            f"aligned trend m1={m1*100:.2f}% m3={m3*100:.2f}%, low acceleration",
        )

    # Hammer — late velocity + acceleration spike (often near strike)
    near_strike = abs(spot - strike) / max(spot, 1.0) < 0.0015
    late = seconds_to_expiry < 420
    spike = abs(accel) >= 0.0008 or vol_ratio >= 1.35
    strong_m1 = abs(m1) >= 0.0012
    if spike and strong_m1 and (late or near_strike):
        lean = "YES" if m1 > 0 else "NO"
        conf = min(0.95, 0.55 + abs(accel) * 200 + abs(m1) * 40)
        return PatternAssessment(
            PricePattern.HAMMER,
            lean,
            conf,
            f"velocity spike m1={m1*100:.2f}% accel={accel*100:.3f}% vol×{vol_ratio:.2f}",
        )

    # Fade — momentum losing steam or reversing against the move
    reversal = (m1 > 0 and m3 < 0) or (m1 < 0 and m3 > 0)
    decel_against = (m1 > 0 and accel < -0.0003) or (m1 < 0 and accel > 0.0003)
    slowing = abs(m1) < abs(m3) * 0.6 and abs(m3) > 0.0005
    if reversal or decel_against or slowing:
        lean = "NO" if m1 > 0 else "YES"  # fading up move → bearish lean, etc.
        conf = min(0.85, 0.45 + abs(m3 - m1) * 60)
        return PatternAssessment(
            PricePattern.FADE,
            lean,
            conf,
            f"weakening move m1={m1*100:.2f}% m3={m3*100:.2f}% accel={accel*100:.3f}%",
        )

    return PatternAssessment(PricePattern.NONE, finish_side, 0.35, "no clear pattern")
