"""Tests for the 1-hour bot."""

from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np

from kalshi_btc_1hr_bot.brti import BrtiQuote, resolve_brti
from kalshi_btc_1hr_bot.config import BotConfig
from kalshi_btc_1hr_bot.data_feed import FundingRate, MarketData, SyntheticPriceGenerator
from kalshi_btc_1hr_bot.edge import evaluate_edge, vwap_fill_price
from kalshi_btc_1hr_bot.ensemble import ModelVote, combine_models
from kalshi_btc_1hr_bot.evidence import (
    MarketCandidate,
    directional_evidence,
    evaluate_edge_with_evidence,
    evidence_score,
    select_best_from_top_markets,
)
from kalshi_btc_1hr_bot.forecast import ForecastEnsemble
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
    assert len(result.features) == 18


def test_forecast_ensemble():
    ensemble = ForecastEnsemble()
    state = build_market_state(
        spot=65000,
        strike=64500,
        seconds_remaining=1200,
        price_history=_sample_history(),
        vwap=64950,
        funding=FundingRate(0.0001),
        is_official_brti=True,
    )
    from kalshi_btc_1hr_bot.data_feed import MarketData
    import numpy as np

    data = MarketData(
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
    result = ensemble.forecast(state, data)
    assert 0 < result.p_fair < 1
    assert len(result.crowd.members) >= 10
    assert result.agreement_score > 0
    assert result.crowd.quorum_required >= 5


def test_directional_evidence_picks_above():
    votes = [
        ModelVote("a", 0.70, 0.4, 0.9),
        ModelVote("b", 0.65, 0.3, 0.8),
        ModelVote("c", 0.55, 0.2, 0.7),
        ModelVote("d", 0.45, 0.1, 0.6),  # below — lower rank
    ]
    d = directional_evidence(votes, n=4)
    assert d.side == "yes"
    assert d.above_score > d.below_score
    assert len(d.top_votes) == 4


def test_directional_evidence_picks_below():
    votes = [
        ModelVote("a", 0.30, 0.4, 0.9),
        ModelVote("b", 0.35, 0.3, 0.8),
        ModelVote("c", 0.40, 0.2, 0.7),
        ModelVote("d", 0.60, 0.1, 0.6),
    ]
    d = directional_evidence(votes, n=4)
    assert d.side == "no"
    assert d.below_score > d.above_score


def test_crowd_favorite_gate_blocks_below_dynamic_floor():
    from kalshi_btc_1hr_bot.crowd_forecast import CrowdForecast, CrowdMember
    from kalshi_btc_1hr_bot.dynamic_gates import resolve_dynamic_thresholds

    members = tuple(
        CrowdMember("m", 0.60, 1.0, 0.9, "model") for _ in range(8)
    )
    crowd = CrowdForecast(
        prob_yes=0.60,
        prob_no=0.40,
        consensus_side="yes",
        confidence=0.7,
        agreement_score=0.8,
        uncertainty=0.2,
        quorum_count=8,
        quorum_required=5,
        quorum_met=True,
        yes_votes=8,
        no_votes=0,
        synthesis="blend",
        members=members,
        top_votes=members[:4],
        disagreeing=(),
    )
    th = resolve_dynamic_thresholds(1800, vol_regime="med", agreement_score=0.8, crowd_side_prob=0.60)
    assert not crowd.side_met("yes", min_favorite=th.min_crowd_favorite)
    assert crowd.side_pct("yes") == 60.0

    votes = [ModelVote("m", 0.60, 0.5, 0.9)]
    direction = directional_evidence(votes, n=1)
    from kalshi_btc_1hr_bot.forecast import ForecastEnsembleOutput
    from kalshi_btc_1hr_bot.ensemble import EnsembleResult
    from kalshi_btc_1hr_bot.model import ModelOutput
    import numpy as np

    mo = ModelOutput(0.6, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5, 0.0, 0.0, 0.0, "low", 0.0, np.zeros(18))
    ens = EnsembleResult(0.6, 0.4, 0.7, 0.2, 0.8, tuple(votes))
    forecast = ForecastEnsembleOutput(0.6, 0.7, 0.8, 0.2, mo, ens, crowd, True)
    edge = evaluate_edge_with_evidence(
        0.6, 0.40, 0.65, 0.38, 0.63, direction, thresholds=th, forecast=forecast
    )
    assert not edge.should_trade
    assert "Crowd" in edge.reason


def test_evaluate_edge_with_evidence():
    votes = [
        ModelVote("a", 0.72, 0.5, 0.9),
        ModelVote("b", 0.68, 0.3, 0.8),
        ModelVote("c", 0.66, 0.15, 0.7),
        ModelVote("d", 0.64, 0.05, 0.6),
    ]
    direction = directional_evidence(votes, n=4)
    edge = evaluate_edge_with_evidence(
        0.70, 0.40, 0.65, 0.38, 0.63, direction, min_edge=2.5, min_evidence_margin=0.01
    )
    assert direction.side == "yes"
    assert edge.should_trade
    assert edge.side == "yes"


def test_select_best_from_top_markets():
    from kalshi_btc_1hr_bot.edge import TradeSignal
    from kalshi_btc_1hr_bot.forecast import ForecastEnsembleOutput
    from kalshi_btc_1hr_bot.ensemble import EnsembleResult
    from kalshi_btc_1hr_bot.model import ModelOutput
    import numpy as np

    def _cand(ticker: str, edge_cents: float, ev_score: float):
        from kalshi_btc_1hr_bot.crowd_forecast import CrowdForecast, CrowdForecastSystem
        from kalshi_btc_1hr_bot.model import ForecastModel, build_market_state

        votes = (ModelVote("m", 0.7, 0.5, 0.9),)
        direction = directional_evidence(votes, n=1)
        mo = ModelOutput(
            0.7, 0.3, 0.65, 0.68, 0.67, 0.66, 0.69, 0.5, 0.0, 0.0, 0.0, "low", 0.0, np.zeros(18)
        )
        ens = EnsembleResult(0.7, 0.3, 0.8, 0.1, 0.9, votes)
        state = build_market_state(
            spot=65000, strike=65000, seconds_remaining=1200, price_history=_sample_history()
        )
        crowd = CrowdForecastSystem().forecast(mo, state, None)
        forecast = ForecastEnsembleOutput(0.7, 0.8, 0.9, 0.1, mo, ens, crowd, True)
        return MarketCandidate(
            ticker=ticker,
            strike=65000,
            secs_left=1200,
            forecast=forecast,
            direction=direction,
            edge=TradeSignal(True, "yes", 0.7, 0.4, edge_cents, 0.3, "ok"),
            evidence_score=ev_score,
            market={},
        )

    cands = [
        _cand("A", 5.0, 0.10),
        _cand("B", 6.0, 0.25),
        _cand("C", 5.5, 0.30),
        _cand("D", 4.0, 0.40),
        _cand("E", 7.0, 0.05),
    ]
    best = select_best_from_top_markets(cands, n=4)
    assert best is not None
    # Top 4 by edge: E, B, C, A — highest evidence among those is C
    assert best.ticker == "C"


def test_combine_models():
    votes = [
        ModelVote("a", 0.60, 0.5, 0.8),
        ModelVote("b", 0.58, 0.3, 0.7),
        ModelVote("c", 0.62, 0.2, 0.75),
    ]
    result = combine_models(votes)
    assert result.prob_yes > 0.5
    assert result.agreement_score >= 0.5


def test_brti_resolve_fallback():
    with patch("kalshi_btc_1hr_bot.brti.fetch_brti_public_summary", return_value=None):
        quote = resolve_brti(prefer_official=True, allow_exchange_proxy=True)
    assert quote.value > 0
    assert quote.source in ("kraken_xbtusd", "coinbase_btc_usd", "cfbenchmarks_public_rti")


def test_brti_public_quote():
    quote = BrtiQuote(value=65000.0, source="test", is_official=True)
    assert quote.is_official


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
    assert edge.should_trade


def test_vwap_fill_price():
    book = [(0.40, 10), (0.42, 20), (0.45, 30)]
    px = vwap_fill_price(book, 25)
    expected = (0.40 * 10 + 0.42 * 15) / 25
    assert abs(px - expected) < 1e-9


def test_kelly_sizing_one_dollar_cap():
    from kalshi_btc_1hr_bot.config import SizingConfig

    sizing = SizingConfig(bankroll_usd=1.0, max_trade_usd=1.0, max_bankroll_pct=1.0)
    contracts = kelly_contracts(win_prob=0.65, price=0.40, sizing=sizing, confidence=0.8)
    assert contracts >= 1
    assert contracts * 0.40 <= 1.0


def test_kelly_sizing():
    cfg = BotConfig()
    contracts = kelly_contracts(
        win_prob=0.65,
        price=0.40,
        sizing=cfg.sizing,
        confidence=0.8,
    )
    assert contracts > 0


def test_backtest_runs():
    from kalshi_btc_1hr_bot.backtest import run_synthetic_backtest

    report = run_synthetic_backtest(n_markets=20, seed=42)
    assert report.markets_simulated == 20
