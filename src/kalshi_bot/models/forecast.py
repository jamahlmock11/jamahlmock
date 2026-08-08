"""Institutional 1-hour BTC probability forecast ensemble.

Combines:
1. Options-implied (risk-neutral) digital from the IBIT/Deribit smile
2. Physical-measure digital from short-horizon realized volatility

Accuracy > frequency: disagreement, stale inputs, and thin books force NO TRADE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from scipy.stats import norm

from kalshi_bot.data.realized_vol import RealizedVolEstimate, estimate_realized_vol
from kalshi_bot.models.black_scholes import BSInputs, digital_call_probability
from kalshi_bot.models.probability import ImpliedProbResult, options_implied_prob_above, years_to_expiry
from kalshi_bot.models.smile import VolSmile


class ForecastAction(str, Enum):
    TRADE_YES = "TRADE_YES"
    TRADE_NO = "TRADE_NO"
    NO_TRADE = "NO_TRADE"


@dataclass(frozen=True)
class ComponentForecast:
    name: str
    probability_yes: float
    vol_used: float
    weight: float
    reliable: bool
    note: str = ""


@dataclass(frozen=True)
class EnsembleForecast:
    """Calibrated P(S_T >= K) with uncertainty metadata."""

    probability_yes: float
    probability_lo: float
    probability_hi: float
    disagreement_pp: float
    confidence: float  # 0..1
    components: list[ComponentForecast]
    options: ImpliedProbResult | None
    realized: RealizedVolEstimate | None
    spot: float
    strike: float
    seconds_to_expiry: float
    evidence_notes: tuple[str, ...]

    @property
    def sufficient_evidence(self) -> bool:
        return self.confidence >= 0.55 and len([c for c in self.components if c.reliable]) >= 2


def _physical_digital(
    *,
    spot: float,
    strike: float,
    time_years: float,
    vol: float,
    drift: float = 0.0,
) -> float:
    """Physical-measure P(S_T > K) under GBM with optional drift μ."""
    if spot <= 0 or strike <= 0 or time_years <= 0 or vol <= 0:
        return 0.5
    # Under physical measure: d2_phys = [ln(S/K) + (μ - σ²/2)T] / (σ√T)
    d2 = (math.log(spot / strike) + (drift - 0.5 * vol * vol) * time_years) / (
        vol * math.sqrt(time_years)
    )
    return float(norm.cdf(d2))


def _blend(components: list[ComponentForecast]) -> tuple[float, float]:
    """Return (weighted mean, range in probability points)."""
    usable = [c for c in components if c.reliable and c.weight > 0]
    if not usable:
        usable = [c for c in components if c.weight > 0] or components
    wsum = sum(c.weight for c in usable) or 1.0
    mean = sum(c.probability_yes * c.weight for c in usable) / wsum
    probs = [c.probability_yes for c in usable]
    spread = (max(probs) - min(probs)) * 100 if len(probs) >= 2 else 100.0
    return mean, spread


def _confidence(
    *,
    components: list[ComponentForecast],
    disagreement_pp: float,
    smile: VolSmile | None,
    realized: RealizedVolEstimate | None,
    seconds: float,
    spread_dollars: float | None,
) -> tuple[float, list[str]]:
    notes: list[str] = []
    score = 1.0
    reliable = [c for c in components if c.reliable]
    if len(reliable) < 2:
        score -= 0.45
        notes.append("fewer than 2 reliable forecast components")
    if disagreement_pp > 12.0:
        score -= 0.35
        notes.append(f"model disagreement high ({disagreement_pp:.1f}pp)")
    elif disagreement_pp > 8.0:
        score -= 0.15
        notes.append(f"model disagreement elevated ({disagreement_pp:.1f}pp)")
    if smile is not None and smile.is_synthetic:
        score -= 0.40
        notes.append("options smile is synthetic (demo-only)")
    if smile is not None and smile.age_seconds > 6 * 3600:
        score -= 0.20
        notes.append(f"options smile stale ({smile.age_seconds/3600:.1f}h)")
    if realized is not None and not realized.is_reliable:
        score -= 0.25
        notes.append(f"realized vol unreliable (source={realized.source}, n={realized.n_returns})")
    if seconds < 120:
        score -= 0.25
        notes.append("too close to expiry for robust forecast")
    if seconds > 55 * 60:
        score -= 0.10
        notes.append("horizon longer than preferred 1h window")
    if spread_dollars is not None and spread_dollars > 0.08:
        score -= 0.20
        notes.append(f"Kalshi spread wide ({spread_dollars*100:.1f}¢)")
    return max(0.0, min(1.0, score)), notes


def forecast_prob_above(
    *,
    spot: float,
    strike: float,
    close_time: datetime,
    smile: VolSmile | None,
    rate: float = 0.045,
    dividend: float = 0.0,
    now: datetime | None = None,
    realized: RealizedVolEstimate | None = None,
    yes_bid: float | None = None,
    yes_ask: float | None = None,
    include_equal: bool = True,
) -> EnsembleForecast:
    """Build an ensemble P(settle >= strike) for a KXBTCD-style contract."""
    now = now or datetime.now(timezone.utc)
    if close_time.tzinfo is None:
        close_time = close_time.replace(tzinfo=timezone.utc)
    seconds = max((close_time - now).total_seconds(), 1.0)
    t_years = years_to_expiry(close_time, now)

    if realized is None:
        realized = estimate_realized_vol(horizon_seconds=seconds)

    components: list[ComponentForecast] = []
    options_result: ImpliedProbResult | None = None

    # --- Component A: options-implied digital ---
    if smile is not None:
        try:
            options_result = options_implied_prob_above(
                spot_btc=spot,
                strike_btc=strike,
                close_time=close_time,
                smile=smile,
                rate=rate,
                dividend=dividend,
                now=now,
                include_equal=include_equal,
            )
            # For ultra-short horizons, blend smile IV toward realized to avoid
            # equity-option term-structure contamination.
            smile_iv = options_result.vol_used
            mix = 0.55 if not smile.is_synthetic else 0.15
            blended_iv = mix * smile_iv + (1.0 - mix) * realized.annualized_vol
            blended_iv = max(blended_iv, 0.05)
            p_opt = digital_call_probability(
                BSInputs(
                    spot=spot,
                    strike=strike,
                    time_years=t_years,
                    vol=blended_iv,
                    rate=rate,
                    dividend=dividend,
                )
            )
            components.append(
                ComponentForecast(
                    name="options_smile",
                    probability_yes=p_opt,
                    vol_used=blended_iv,
                    weight=0.55 if not smile.is_synthetic else 0.15,
                    reliable=not smile.is_synthetic and smile.age_seconds < 12 * 3600,
                    note=f"smile_iv={smile_iv:.3f} blended_iv={blended_iv:.3f}",
                )
            )
        except Exception as exc:
            components.append(
                ComponentForecast(
                    name="options_smile",
                    probability_yes=0.5,
                    vol_used=0.0,
                    weight=0.0,
                    reliable=False,
                    note=f"failed: {exc}",
                )
            )

    # --- Component B: realized-vol physical digital (μ≈0 conservative) ---
    p_rv = _physical_digital(
        spot=spot,
        strike=strike,
        time_years=t_years,
        vol=max(realized.annualized_vol, 0.05),
        drift=0.0,
    )
    components.append(
        ComponentForecast(
            name="realized_vol",
            probability_yes=p_rv,
            vol_used=realized.annualized_vol,
            weight=0.45 if realized.is_reliable else 0.20,
            reliable=realized.is_reliable,
            note=f"source={realized.source} n={realized.n_returns}",
        )
    )

    # --- Component C: pure BS ATM-anchor using max(smile ATM, realized) ---
    anchor_iv = realized.annualized_vol
    if smile is not None and not smile.is_synthetic:
        anchor_iv = 0.5 * smile.atm_iv + 0.5 * realized.annualized_vol
    p_anchor = _physical_digital(
        spot=spot,
        strike=strike,
        time_years=t_years,
        vol=max(anchor_iv, 0.05),
        drift=0.0,
    )
    components.append(
        ComponentForecast(
            name="vol_anchor",
            probability_yes=p_anchor,
            vol_used=anchor_iv,
            weight=0.20,
            reliable=realized.is_reliable or (smile is not None and not smile.is_synthetic),
            note="conservative μ=0 digital",
        )
    )

    mean, disagreement_pp = _blend(components)
    spread = None
    if yes_bid is not None and yes_ask is not None:
        spread = max(0.0, yes_ask - yes_bid)

    conf, notes = _confidence(
        components=components,
        disagreement_pp=disagreement_pp,
        smile=smile,
        realized=realized,
        seconds=seconds,
        spread_dollars=spread,
    )

    # Uncertainty band: half the model disagreement, floored
    band = max(0.02, disagreement_pp / 200.0)
    return EnsembleForecast(
        probability_yes=float(mean),
        probability_lo=float(max(0.0, mean - band)),
        probability_hi=float(min(1.0, mean + band)),
        disagreement_pp=float(disagreement_pp),
        confidence=float(conf),
        components=components,
        options=options_result,
        realized=realized,
        spot=spot,
        strike=strike,
        seconds_to_expiry=seconds,
        evidence_notes=tuple(notes),
    )
