"""Black-Scholes primitives for digital (binary) probabilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm


@dataclass(frozen=True)
class BSInputs:
    spot: float
    strike: float
    time_years: float
    vol: float
    rate: float = 0.0
    dividend: float = 0.0


def _validate(inp: BSInputs) -> None:
    if inp.spot <= 0 or inp.strike <= 0:
        raise ValueError("spot and strike must be positive")
    if inp.time_years <= 0:
        raise ValueError("time_years must be positive")
    if inp.vol <= 0:
        raise ValueError("vol must be positive")


def d1(inp: BSInputs) -> float:
    _validate(inp)
    return (
        math.log(inp.spot / inp.strike)
        + (inp.rate - inp.dividend + 0.5 * inp.vol * inp.vol) * inp.time_years
    ) / (inp.vol * math.sqrt(inp.time_years))


def d2(inp: BSInputs) -> float:
    return d1(inp) - inp.vol * math.sqrt(inp.time_years)


def digital_call_probability(inp: BSInputs) -> float:
    """Risk-neutral P(S_T > K) under Black-Scholes = N(d2).

    For cash-or-nothing digitals discounted, multiply by e^{-rT}; Kalshi
    binaries pay $1 at expiry with no discounting in contract space, so we
    return the undiscounted risk-neutral probability N(d2).
    """
    return float(norm.cdf(d2(inp)))


def digital_put_probability(inp: BSInputs) -> float:
    """Risk-neutral P(S_T < K) = N(-d2)."""
    return float(norm.cdf(-d2(inp)))


def call_price(inp: BSInputs) -> float:
    _validate(inp)
    disc_q = math.exp(-inp.dividend * inp.time_years)
    disc_r = math.exp(-inp.rate * inp.time_years)
    return disc_q * inp.spot * norm.cdf(d1(inp)) - disc_r * inp.strike * norm.cdf(d2(inp))


def put_price(inp: BSInputs) -> float:
    _validate(inp)
    disc_q = math.exp(-inp.dividend * inp.time_years)
    disc_r = math.exp(-inp.rate * inp.time_years)
    return disc_r * inp.strike * norm.cdf(-d2(inp)) - disc_q * inp.spot * norm.cdf(-d1(inp))


def implied_vol_from_price(
    option_price: float,
    spot: float,
    strike: float,
    time_years: float,
    rate: float = 0.0,
    dividend: float = 0.0,
    is_call: bool = True,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float | None:
    """Bracketed Newton / bisection IV solver."""
    if option_price <= 0 or time_years <= 0 or spot <= 0 or strike <= 0:
        return None

    lo, hi = 1e-4, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        inp = BSInputs(spot, strike, time_years, mid, rate, dividend)
        model = call_price(inp) if is_call else put_price(inp)
        if abs(model - option_price) < tol:
            return mid
        if model > option_price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
