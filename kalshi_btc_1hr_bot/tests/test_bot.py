"""Tests for the 1-hour bot."""

from __future__ import annotations

import math

import numpy as np
import pytest

from kalshi_btc_1hr_bot.config import BotConfig, EdgeConfig
from kalshi_btc_1hr_bot.data_feed import MarketData, SyntheticPriceGenerator
from kalshi_btc_1hr_bot.edge import TradeSide, evaluate_edge
from kalshi_btc_1hr_bot.model import HourlyForecastModel
from kalshi_btc_1hr_bot.sizing import kelly_contracts
from kalshi_btc_1hr_bot.utils import gbm_prob_above, quadratic_fee


def _sample_data() -> MarketData:
    return MarketData(
        spot=65000.0,
        vwap=64950.0,
        funding_rate=0.0001,
        annualized_vol=0.50,
        mu_5m=0.001,
        mu_15m=0.0008,
        mu_30m=0.0005,
        closes_1m=np.linspace(64800, 65000, 30),
    )


def test_gbm_prob_above():
    p = gbm_prob_above(65000, 64000, 1 / 365.25 / 24, 0.5)
    assert p > 0.5


def test_quadratic_fee():
    fee = quadratic_fee(0.50)
    assert fee > 0
    assert fee <= 0.02


def test_model_forecast():
    model = HourlyForecastModel()
    result = model.forecast(
        spot=65000,
        strike=64500,
        seconds_to_expiry=1200,
        data=_sample_data(),
    )
    assert 0 < result.p_fair < 1
    assert len(result.layers) == 5
    assert result.confidence > 0


def test_edge_calculation():
    edge = evaluate_edge(
        p_fair=0.65,
        yes_ask=0.40,
        no_ask=0.65,
        edge_cfg=EdgeConfig(min_edge=0.025),
    )
    assert edge.side == TradeSide.YES
    assert edge.net_edge > 0
    assert edge.should_trade


def test_edge_below_threshold():
    edge = evaluate_edge(
        p_fair=0.55,
        yes_ask=0.52,
        no_ask=0.52,
        edge_cfg=EdgeConfig(min_edge=0.025),
    )
    assert not edge.should_trade


def test_kelly_sizing():
    cfg = BotConfig()
    contracts = kelly_contracts(
        win_prob=0.65,
        price=0.40,
        sizing=cfg.sizing,
        confidence=0.8,
    )
    assert contracts > 0


def test_synthetic_generator():
    gen = SyntheticPriceGenerator(seed=42)
    path, funding, vwap = gen.next_hour_path(spot0=65000)
    assert len(path) == 3601
    assert path[0] == 65000
    assert vwap > 0


def test_backtest_runs():
    from kalshi_btc_1hr_bot.backtest import run_synthetic_backtest

    report = run_synthetic_backtest(n_markets=20, seed=42)
    assert report.markets_simulated == 20
    assert isinstance(report.total_pnl, float)
