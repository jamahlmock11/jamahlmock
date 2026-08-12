"""Tests for microstructure calibration and time-bucket analytics."""

import tempfile
from pathlib import Path

from kalshi_bot.calibration.microstructure import MicrostructureCalibrator
from kalshi_bot.calibration.time_bucket_analytics import SettledTrade, time_bucket_performance
from kalshi_bot.learning.settlement_ingestion import SettlementIngestor


def test_microstructure_calibrator_uplift():
    with tempfile.TemporaryDirectory() as tmp:
        cal = MicrostructureCalibrator(str(Path(tmp) / "micro.db"))
        # When micro agrees with YES side (positive imbalance), predictions more accurate
        for _ in range(6):
            cal.record(
                ticker="T1",
                side="yes",
                prediction=0.75,
                won=True,
                bid_ask_imbalance=0.4,
                liquidity_score=0.6,
            )
        for _ in range(6):
            cal.record(
                ticker="T2",
                side="yes",
                prediction=0.75,
                won=False,
                bid_ask_imbalance=-0.4,
                liquidity_score=0.6,
            )
        report = cal.report(min_samples=5)
        assert report["n_total"] == 12
        assert report["n_micro_agrees"] == 6
        assert report["n_micro_disagrees"] == 6
        assert report["agree_brier"] is not None
        assert report["disagree_brier"] is not None
        assert report["agree_brier"] < report["disagree_brier"]


def test_time_bucket_performance_groups():
    trades = [
        SettledTrade("A", "yes", 0.7, True, 0.5, 0.12, 720.0),
        SettledTrade("B", "yes", 0.68, False, -0.5, 0.10, 480.0),
        SettledTrade("C", "no", 0.3, True, 0.4, 0.08, 240.0),
    ]
    rows = time_bucket_performance(trades)
    assert len(rows) == 5
    assert any(r["n_trades"] > 0 for r in rows)


def test_settlement_settled_trades_after_ingest():
    with tempfile.TemporaryDirectory() as tmp:
        micro = MicrostructureCalibrator(str(Path(tmp) / "micro.db"))
        ingestor = SettlementIngestor(str(Path(tmp) / "pending.db"), micro)
        ingestor.record_entry(
            ticker="KXBTC15M-TEST",
            side="yes",
            entry_price=0.55,
            contracts=10,
            prediction=0.72,
            confidence=0.8,
            features={"bid_ask_imbalance": 0.3, "liquidity_score": 0.5},
            seconds_to_expiry=420.0,
            volatility=0.45,
            net_edge=0.12,
            reason="test",
        )
        from unittest.mock import MagicMock

        from kalshi_bot.config import Rules15mConfig, V6Config
        from kalshi_bot.strategy.v6_upgrades import V6IntelligenceEngine

        client = MagicMock()
        client.get_market.return_value = {
            "ticker": "KXBTC15M-TEST",
            "status": "finalized",
            "result": "yes",
        }
        engine = V6IntelligenceEngine(
            V6Config(journal_path=str(Path(tmp) / "journal.db")),
            rules=Rules15mConfig(),
        )
        ingestor.ingest(client, engine)
        settled = ingestor.settled_trades()
        assert len(settled) == 1
        assert settled[0]["won"] is True
        assert micro.report()["n_total"] == 1
