"""Tests for the institutional 1-hour forecasting / evidence gates."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from kalshi_bot.config import ForecastGateConfig, SeriesConfig, SmileConfig
from kalshi_bot.data.realized_vol import RealizedVolEstimate
from kalshi_bot.models.forecast import forecast_prob_above
from kalshi_bot.models.smile import synthetic_smile
from kalshi_bot.strategy.decision import DecisionVerdict, evaluate_forecast_market
from kalshi_bot.strategy.mispricing import Side


def _reliable_rv(spot: float = 65000.0, ann: float = 0.55) -> RealizedVolEstimate:
    return RealizedVolEstimate(
        annualized_vol=ann,
        horizon_vol=ann * (3000 / (365.25 * 24 * 3600)) ** 0.5,
        horizon_seconds=3000,
        n_returns=120,
        bar_seconds=60,
        source="test",
        spot=spot,
    )


def test_forecast_defaults_to_no_trade_when_fair():
    spot = 65000.0
    smile = synthetic_smile(spot, atm_iv=0.5)
    smile.is_synthetic = False
    close = datetime.now(timezone.utc) + timedelta(minutes=40)
    market = {
        "ticker": "KXBTCD-FAIR",
        "series_ticker": "KXBTCD",
        "strike": 65000.0,
        "close_time": close,
        "yes_ask": 0.50,
        "yes_bid": 0.48,
        "no_ask": 0.52,
        "volume": 500.0,
        "strike_type": "greater",
    }
    with patch("kalshi_bot.models.forecast.estimate_realized_vol", return_value=_reliable_rv(spot)):
        decision = evaluate_forecast_market(
            market,
            spot=spot,
            smile=smile,
            series_cfg=SeriesConfig(ticker="KXBTCD", min_edge_pp=10.0),
            smile_cfg=SmileConfig(),
            gates=ForecastGateConfig(min_edge_pp=10.0, min_confidence=0.55),
        )
    assert decision.verdict == DecisionVerdict.NO_TRADE
    assert decision.action.value == "NO_TRADE"


def test_large_mispricing_can_trade_yes_with_evidence():
    spot = 65000.0
    strike = 65100.0
    smile = synthetic_smile(spot, atm_iv=0.9)
    smile.is_synthetic = False
    close = datetime.now(timezone.utc) + timedelta(minutes=45)
    market = {
        "ticker": "KXBTCD-EDGE-YES",
        "series_ticker": "KXBTCD",
        "strike": strike,
        "close_time": close,
        "yes_ask": 0.08,
        "yes_bid": 0.05,
        "no_ask": 0.95,
        "volume": 2000.0,
        "strike_type": "greater",
    }
    with patch("kalshi_bot.models.forecast.estimate_realized_vol", return_value=_reliable_rv(spot, 0.9)):
        decision = evaluate_forecast_market(
            market,
            spot=spot,
            smile=smile,
            series_cfg=SeriesConfig(ticker="KXBTCD", min_edge_pp=8.0),
            smile_cfg=SmileConfig(),
            gates=ForecastGateConfig(
                min_edge_pp=8.0,
                min_confidence=0.50,
                max_disagreement_pp=20.0,
                max_spread=0.10,
            ),
        )
    assert decision.verdict == DecisionVerdict.TRADE
    assert decision.side == Side.YES
    assert decision.expected_value_per_contract > 0


def test_wide_spread_blocks_trade():
    spot = 65000.0
    smile = synthetic_smile(spot, atm_iv=0.9)
    smile.is_synthetic = False
    close = datetime.now(timezone.utc) + timedelta(minutes=45)
    market = {
        "ticker": "KXBTCD-WIDE",
        "series_ticker": "KXBTCD",
        "strike": 65100.0,
        "close_time": close,
        "yes_ask": 0.20,
        "yes_bid": 0.05,  # 15¢ spread
        "no_ask": 0.95,
        "volume": 2000.0,
        "strike_type": "greater",
    }
    with patch("kalshi_bot.models.forecast.estimate_realized_vol", return_value=_reliable_rv(spot, 0.9)):
        decision = evaluate_forecast_market(
            market,
            spot=spot,
            smile=smile,
            series_cfg=SeriesConfig(ticker="KXBTCD", min_edge_pp=5.0),
            smile_cfg=SmileConfig(),
            gates=ForecastGateConfig(min_edge_pp=5.0, max_spread=0.06, min_confidence=0.4),
        )
    assert decision.verdict == DecisionVerdict.NO_TRADE
    assert any("spread" in b.lower() for b in decision.blockers)


def test_proxy_spot_raises_bar_to_no_trade():
    spot = 65000.0
    smile = synthetic_smile(spot, atm_iv=0.9)
    smile.is_synthetic = False
    close = datetime.now(timezone.utc) + timedelta(minutes=45)
    # Mild edge that would pass with official spot but fail with proxy multiplier
    market = {
        "ticker": "KXBTCD-PROXY",
        "series_ticker": "KXBTCD",
        "strike": 65050.0,
        "close_time": close,
        "yes_ask": 0.30,
        "yes_bid": 0.27,
        "no_ask": 0.73,
        "volume": 2000.0,
        "strike_type": "greater",
    }
    with patch("kalshi_bot.models.forecast.estimate_realized_vol", return_value=_reliable_rv(spot, 0.55)):
        official = evaluate_forecast_market(
            market,
            spot=spot,
            smile=smile,
            series_cfg=SeriesConfig(ticker="KXBTCD", min_edge_pp=5.0),
            smile_cfg=SmileConfig(),
            gates=ForecastGateConfig(
                min_edge_pp=5.0,
                min_confidence=0.50,
                max_disagreement_pp=25.0,
                proxy_spot_edge_multiplier=1.5,
                proxy_spot_confidence_penalty=0.30,
            ),
            spot_is_official=True,
        )
        proxy = evaluate_forecast_market(
            market,
            spot=spot,
            smile=smile,
            series_cfg=SeriesConfig(ticker="KXBTCD", min_edge_pp=5.0),
            smile_cfg=SmileConfig(),
            gates=ForecastGateConfig(
                min_edge_pp=5.0,
                min_confidence=0.50,
                max_disagreement_pp=25.0,
                proxy_spot_edge_multiplier=1.5,
                proxy_spot_confidence_penalty=0.30,
            ),
            spot_is_official=False,
        )
    assert proxy.confidence <= official.confidence
    # Proxy path must be at least as conservative
    if official.verdict == DecisionVerdict.NO_TRADE:
        assert proxy.verdict == DecisionVerdict.NO_TRADE


def test_gap_below_15pp_forces_no_trade_even_with_postfee_edge():
    """Matrix: 60% model vs 50¢ YES (10pp) → No trade, regardless of fee EV."""
    from kalshi_bot.config import BotActionConfig
    from kalshi_bot.strategy.bot_action import BotAction
    from kalshi_bot.strategy.decision import DecisionVerdict, evaluate_forecast_market

    spot = 65000.0
    smile = synthetic_smile(spot, atm_iv=0.9)
    smile.is_synthetic = False
    close = datetime.now(timezone.utc) + timedelta(minutes=45)

    # Force forecast probability ≈ 0.60 via a patched ensemble.
    market = {
        "ticker": "KXBTCD-GAP10",
        "series_ticker": "KXBTCD",
        "strike": 65000.0,
        "close_time": close,
        "yes_ask": 0.50,
        "yes_bid": 0.48,
        "no_ask": 0.52,
        "volume": 2000.0,
        "strike_type": "greater",
    }

    class _FakeForecast:
        probability_yes = 0.60
        probability_lo = 0.58
        probability_hi = 0.62
        disagreement_pp = 2.0
        confidence = 0.85
        components = []
        options = None
        realized = None
        spot = 65000.0
        strike = 65000.0
        seconds_to_expiry = 2700.0
        evidence_notes = ("test",)
        sufficient_evidence = True

    with patch(
        "kalshi_bot.strategy.decision.forecast_prob_above",
        return_value=_FakeForecast(),
    ):
        decision = evaluate_forecast_market(
            market,
            spot=spot,
            smile=smile,
            series_cfg=SeriesConfig(ticker="KXBTCD", min_edge_pp=5.0),
            smile_cfg=SmileConfig(),
            gates=ForecastGateConfig(min_edge_pp=5.0, min_confidence=0.50, max_spread=0.10),
            spot_is_official=True,
            bot_action_cfg=BotActionConfig(),
        )
    assert decision.verdict == DecisionVerdict.NO_TRADE
    assert decision.bot_action == BotAction.NO_TRADE
    assert decision.gap_pp == 10.0
    assert any("No trade" in b or "gap" in b.lower() for b in decision.blockers)


def test_gap_20pp_is_strong_buy_candidate():
    """Matrix: 60% model vs 40¢ YES (20pp) → Strong BUY when other gates clear."""
    from kalshi_bot.config import BotActionConfig
    from kalshi_bot.strategy.bot_action import BotAction
    from kalshi_bot.strategy.decision import DecisionVerdict, evaluate_forecast_market

    spot = 65000.0
    smile = synthetic_smile(spot, atm_iv=0.9)
    smile.is_synthetic = False
    close = datetime.now(timezone.utc) + timedelta(minutes=45)
    market = {
        "ticker": "KXBTCD-GAP20",
        "series_ticker": "KXBTCD",
        "strike": 65000.0,
        "close_time": close,
        "yes_ask": 0.40,
        "yes_bid": 0.38,
        "no_ask": 0.62,
        "volume": 2000.0,
        "strike_type": "greater",
    }

    class _FakeForecast:
        probability_yes = 0.60
        probability_lo = 0.58
        probability_hi = 0.62
        disagreement_pp = 2.0
        confidence = 0.85
        components = []
        options = None
        realized = None
        spot = 65000.0
        strike = 65000.0
        seconds_to_expiry = 2700.0
        evidence_notes = ("test",)
        sufficient_evidence = True

    with patch(
        "kalshi_bot.strategy.decision.forecast_prob_above",
        return_value=_FakeForecast(),
    ):
        decision = evaluate_forecast_market(
            market,
            spot=spot,
            smile=smile,
            series_cfg=SeriesConfig(ticker="KXBTCD", min_edge_pp=5.0),
            smile_cfg=SmileConfig(),
            gates=ForecastGateConfig(
                min_edge_pp=5.0,
                min_confidence=0.50,
                max_spread=0.10,
                max_disagreement_pp=20.0,
            ),
            spot_is_official=True,
            bot_action_cfg=BotActionConfig(),
        )
    assert decision.verdict == DecisionVerdict.TRADE
    assert decision.side == Side.YES
    assert decision.bot_action == BotAction.STRONG_BUY
    assert decision.gap_pp == 20.0
