"""Volatility smile construction and interpolation in BTC spot space."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass
class SmilePoint:
    strike_btc: float
    log_moneyness: float
    iv: float
    source_strike_ibit: float | None = None
    weight: float = 1.0


@dataclass
class VolSmile:
    """IV smile in BTC spot coordinates, keyed by log-moneyness."""

    asof_ts: float
    spot_btc: float
    spot_ibit: float
    btc_per_share: float
    expiry: str
    t_years: float
    points: list[SmilePoint] = field(default_factory=list)
    atm_iv: float = 0.0
    is_synthetic: bool = False

    def __post_init__(self) -> None:
        if self.points and not self.atm_iv:
            self.atm_iv = self.iv_at_strike(self.spot_btc)

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.asof_ts)

    def iv_at_log_m(self, k: float) -> float:
        if not self.points:
            raise ValueError("empty smile")
        xs = np.array([p.log_moneyness for p in self.points], dtype=float)
        ys = np.array([p.iv for p in self.points], dtype=float)
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        # Deduplicate x
        uniq_x, uniq_idx = np.unique(xs, return_index=True)
        uniq_y = ys[uniq_idx]
        if len(uniq_x) == 1:
            return float(uniq_y[0])
        # Clamp outside domain to wings
        if k <= uniq_x[0]:
            return float(uniq_y[0])
        if k >= uniq_x[-1]:
            return float(uniq_y[-1])
        spline = PchipInterpolator(uniq_x, uniq_y, extrapolate=False)
        return float(spline(k))

    def iv_at_strike(self, strike_btc: float) -> float:
        if strike_btc <= 0 or self.spot_btc <= 0:
            raise ValueError("invalid strike/spot")
        return self.iv_at_log_m(math.log(strike_btc / self.spot_btc))

    def to_dict(self) -> dict:
        return {
            "asof_ts": self.asof_ts,
            "spot_btc": self.spot_btc,
            "spot_ibit": self.spot_ibit,
            "btc_per_share": self.btc_per_share,
            "expiry": self.expiry,
            "t_years": self.t_years,
            "atm_iv": self.atm_iv,
            "is_synthetic": self.is_synthetic,
            "points": [asdict(p) for p in self.points],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VolSmile":
        points = [SmilePoint(**p) for p in data.get("points", [])]
        return cls(
            asof_ts=float(data["asof_ts"]),
            spot_btc=float(data["spot_btc"]),
            spot_ibit=float(data["spot_ibit"]),
            btc_per_share=float(data["btc_per_share"]),
            expiry=str(data["expiry"]),
            t_years=float(data["t_years"]),
            points=points,
            atm_iv=float(data.get("atm_iv") or 0.0),
            is_synthetic=bool(data.get("is_synthetic", False)),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "VolSmile":
        return cls.from_dict(json.loads(Path(path).read_text()))


def build_smile_from_ibit_chain(
    *,
    strikes_ibit: list[float],
    ivs: list[float],
    weights: list[float] | None,
    spot_ibit: float,
    spot_btc: float,
    expiry: str,
    t_years: float,
    asof_ts: float | None = None,
) -> VolSmile:
    """Translate IBIT smile into BTC strike space.

    Relative (log) moves are preserved: an IBIT strike K_i maps to
    K_btc = K_i / (S_ibit / S_btc) = K_i * S_btc / S_ibit.
    Equivalently, log-moneyness is identical in both spaces.
    """
    if spot_ibit <= 0 or spot_btc <= 0:
        raise ValueError("spots must be positive")
    btc_per_share = spot_ibit / spot_btc
    w = weights or [1.0] * len(strikes_ibit)
    points: list[SmilePoint] = []
    for k_i, iv, weight in zip(strikes_ibit, ivs, w, strict=True):
        if iv <= 0 or k_i <= 0:
            continue
        k_btc = k_i / btc_per_share
        lm = math.log(k_btc / spot_btc)
        points.append(
            SmilePoint(
                strike_btc=k_btc,
                log_moneyness=lm,
                iv=float(iv),
                source_strike_ibit=float(k_i),
                weight=float(weight),
            )
        )
    if len(points) < 2:
        raise ValueError(f"need >=2 smile points, got {len(points)}")
    points.sort(key=lambda p: p.log_moneyness)
    smile = VolSmile(
        asof_ts=asof_ts or time.time(),
        spot_btc=spot_btc,
        spot_ibit=spot_ibit,
        btc_per_share=btc_per_share,
        expiry=expiry,
        t_years=t_years,
        points=points,
    )
    smile.atm_iv = smile.iv_at_strike(spot_btc)
    return smile


def synthetic_smile(
    spot_btc: float,
    atm_iv: float = 0.55,
    skew: float = -0.08,
    smile_curvature: float = 0.12,
    t_years: float = 7 / 365,
) -> VolSmile:
    """Deterministic smile for offline demos / tests."""
    spot_ibit = spot_btc * 0.00056
    strikes = np.linspace(0.7, 1.35, 27) * spot_btc
    points: list[SmilePoint] = []
    for k in strikes:
        m = math.log(k / spot_btc)
        iv = max(0.05, atm_iv + skew * m + smile_curvature * m * m)
        points.append(SmilePoint(strike_btc=float(k), log_moneyness=m, iv=iv))
    return VolSmile(
        asof_ts=time.time(),
        spot_btc=spot_btc,
        spot_ibit=spot_ibit,
        btc_per_share=spot_ibit / spot_btc,
        expiry="synthetic",
        t_years=t_years,
        points=points,
        atm_iv=atm_iv,
        is_synthetic=True,
    )
