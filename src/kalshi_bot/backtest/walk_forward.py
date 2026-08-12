"""Chronological walk-forward backtest harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WalkForwardFold:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class WalkForwardReport:
    folds: list[WalkForwardFold] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["Walk-forward report", f"folds={len(self.folds)}"]
        for i, f in enumerate(self.folds):
            lines.append(f"  fold {i+1}: test {f.test_start}→{f.test_end} {f.metrics}")
        if self.aggregate:
            lines.append(f"aggregate: {self.aggregate}")
        return "\n".join(lines)


def run_walk_forward(
    *,
    decision_store_path: str = "data/decisions.db",
    n_folds: int = 5,
) -> WalkForwardReport:
    """Walk-forward validation using stored decision snapshots.

    Requires resolved outcomes in the decision store. Never shuffles time series.
    """
    # Outcome join + metric computation hooks into DecisionStore in future iteration.
    report = WalkForwardReport()
    report.aggregate = {
        "status": "awaiting_resolved_outcomes",
        "decision_store": decision_store_path,
        "n_folds": n_folds,
    }
    return report
