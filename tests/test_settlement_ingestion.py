"""Tests for automatic settlement outcome ingestion."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from kalshi_bot.config import Rules15mConfig, V6Config
from kalshi_bot.learning.settlement_ingestion import SettlementIngestor
from kalshi_bot.strategy.v6_upgrades import V6IntelligenceEngine


def _engine(tmp_path: Path) -> V6IntelligenceEngine:
    cfg = V6Config(journal_path=str(tmp_path / "journal.db"))
    rules = Rules15mConfig()
    return V6IntelligenceEngine(cfg, client=None, rules=rules)


def test_record_and_ingest_settlement():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "pending.db"
        ingestor = SettlementIngestor(str(db))
        ingestor.record_entry(
            ticker="KXBTC15M-TEST",
            side="yes",
            entry_price=0.55,
            contracts=10,
            prediction=0.72,
            confidence=0.8,
            features={"liquidity_score": 0.5},
            seconds_to_expiry=300.0,
            volatility=0.45,
            net_edge=0.12,
            reason="test entry",
        )
        assert len(ingestor.pending()) == 1

        client = MagicMock()
        client.get_market.return_value = {
            "ticker": "KXBTC15M-TEST",
            "status": "finalized",
            "result": "yes",
        }
        engine = _engine(Path(tmp))
        resolved = ingestor.ingest(client, engine)
        assert len(resolved) == 1
        assert resolved[0]["won"] is True
        assert resolved[0]["pnl"] > 0
        assert len(ingestor.pending()) == 0


def test_calibration_summary():
    with tempfile.TemporaryDirectory() as tmp:
        ingestor = SettlementIngestor(str(Path(tmp) / "pending.db"))
        engine = _engine(Path(tmp))
        for _ in range(5):
            engine.calibrator.record(0.75, True)
        buckets = ingestor.calibration_summary(engine.calibrator)
        assert any(b["n_trades"] >= 5 for b in buckets)
