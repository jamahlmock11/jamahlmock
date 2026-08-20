"""Tests for the 1-hour bot."""

from __future__ import annotations

import time

import numpy as np

from kalshi_btc_1hr_bot.config import BotConfig
from kalshi_btc_1hr_bot.data_feed import FundingRate, MarketData, SyntheticPriceGenerator
from kalshi_btc_1hr_bot.edge import evaluate_edge, vwap_fill_price
from kalshi_btc_1hr_bot.model import ForecastModel, build_market_state
from kalshi_btc_1hr_bot.sizing import kelly_contracts
from kalshi_btc_1hr_bot.utils import gbm_prob_above, quadratic_fee


def _sample_history() -> list:
    now = time.time()
    prices = np.linspace(64800, 65000, 120)
    return [(now - (len(prices) - 1 - i) * 60, float(prices[i])) for i in range(len(prices))]


def test_gbm_prob_above():
    p = gbm_prob_above(65000, 64000, 1 / 365.25 / 24, 0.5)
    assert p > 0.5


def test_quadratic_fee():
    fee = quadratic_fee(0.50)
    assert fee > 0
    assert fee <= 0.02


def test_model_forecast():
    model = ForecastModel()
    state = build_market_state(
        spot=65000,
        strike=64500,
        seconds_remaining=1200,
        price_history=_sample_history(),
        vwap=64950,
        funding=FundingRate(0.0001),
    )
    result = model.forecast(state)
    assert 0 < result.p_calibrated < 1
    assert 0 < result.p_gbm < 1
    assert result.vol_regime in ("low", "medium", "high")
    assert len(result.features) == 18


def test_edge_calculation():
    edge = evaluate_edge(
        p_fair=0.65,
        yes_ask=0.40,
        no_ask=0.65,
        yes_bid=0.38,
        no_bid=0.63,
        min_edge=2.5,
    )
    assert edge.side == "yes"
    assert edge.edge_cents > 0
    assert edge.should_trade


def test_edge_below_threshold():
    edge = evaluate_edge(
        p_fair=0.55,
        yes_ask=0.52,
        no_ask=0.52,
        yes_bid=0.50,
        no_bid=0.50,
        min_edge=2.5,
    )
    assert not edge.should_trade


def test_vwap_fill_price():
    book = [(0.40, 10), (0.42, 20), (0.45, 30)]
    px = vwap_fill_price(book, 25)
    expected = (0.40 * 10 + 0.42 * 15) / 25
    assert abs(px - expected) < 1e-9


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
