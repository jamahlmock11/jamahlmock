"""Crowd forecast system tests."""

from __future__ import annotations

import time

import numpy as np

from kalshi_btc_1hr_bot.config import CROWD_MIN_QUORUM
from kalshi_btc_1hr_bot.crowd_forecast import CrowdForecastSystem, crowd_summary
from kalshi_btc_1hr_bot.data_feed import FundingRate, MarketData
from kalshi_btc_1hr_bot.forecast import ForecastEnsemble
from kalshi_btc_1hr_bot.model import ForecastModel, build_market_state


def _sample_history() -> list:
    now = time.time()
    prices = np.linspace(64800, 65000, 120)
    return [(now - (len(prices) - 1 - i) * 60, float(prices[i])) for i in range(len(prices))]


def _sample_data() -> MarketData:
    return MarketData(
        spot=65000,
        vwap=64950,
        funding_rate=0.0001,
        annualized_vol=0.5,
        mu_5m=0.0001,
        mu_15m=0.00008,
        mu_30m=0.00005,
        closes_1m=np.linspace(64800, 65000, 60),
        price_history=_sample_history(),
        funding=FundingRate(0.0001),
        is_official=True,
    )


def test_crowd_has_many_members():
    system = CrowdForecastSystem()
    model = ForecastModel()
    state = build_market_state(
        spot=65000,
        strike=64500,
        seconds_remaining=1200,
        price_history=_sample_history(),
        vwap=64950,
        funding=FundingRate(0.0001),
    )
    output = model.forecast(state)
    crowd = system.forecast(output, state, _sample_data())
    assert len(crowd.members) >= 10
    assert crowd.synthesis in ("weighted", "median", "trimmed", "blend")


def test_crowd_quorum_counting():
    system = CrowdForecastSystem()
    model = ForecastModel()
    state = build_market_state(
        spot=66000,
        strike=64500,
        seconds_remaining=1200,
        price_history=_sample_history(),
        vwap=65500,
        funding=FundingRate(0.0001),
    )
    output = model.forecast(state)
    crowd = system.forecast(output, state, _sample_data())
    assert crowd.yes_votes + crowd.no_votes == len(crowd.members)
    assert crowd.quorum_required == CROWD_MIN_QUORUM


def test_forecast_ensemble_uses_crowd():
    ensemble = ForecastEnsemble()
    state = build_market_state(
        spot=65000,
        strike=64500,
        seconds_remaining=1200,
        price_history=_sample_history(),
        vwap=64950,
        funding=FundingRate(0.0001),
    )
    result = ensemble.forecast(state, _sample_data())
    assert result.crowd is not None
    assert result.p_fair == result.crowd.prob_yes
    assert len(result.votes) <= 4
    summary = crowd_summary(result.crowd)
    assert "members" in summary
    assert summary["quorum_met"] in (True, False)
