"""Domain models and serialization contracts for walk-forward evaluation results."""

import json
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from adaptive_trading.evaluation.config import EvaluationConfig


class SampleAdequacy(StrEnum):
    """Evaluation sample statistical adequacy rating."""

    LOW_SAMPLE = "LOW_SAMPLE"
    ADEQUATE_SAMPLE = "ADEQUATE_SAMPLE"


class StabilityClassification(StrEnum):
    """Empirical performance stability classification."""

    STABLE = "STABLE"
    MODERATE = "MODERATE"
    UNSTABLE = "UNSTABLE"


class WindowEvaluationStatus(StrEnum):
    """Execution status of an individual walk-forward window."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class WindowEvaluationResult(BaseModel):
    """Performance and trade execution metrics for an individual test window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    window_id: str = Field(description="Identifier of evaluated window")
    window_index: int = Field(gt=0, description="1-indexed sequence number")
    status: WindowEvaluationStatus = Field(
        default=WindowEvaluationStatus.SUCCESS,
        description="Window evaluation outcome status",
    )
    train_start: datetime | None = None
    train_end: datetime | None = None
    validation_start: datetime | None = None
    validation_end: datetime | None = None
    test_start: datetime | None = None
    test_end: datetime | None = None

    model_version: str = Field(
        default="v1", description="Model version used in this window"
    )
    feature_version: str = Field(default="v1", description="Feature catalog version")
    strategy_version: str = Field(default="v1", description="Strategy rules version")
    risk_version: str = Field(default="v1", description="Risk parameter version")

    initial_equity: float = Field(
        default=0.0, description="Starting cash capital for this window"
    )
    final_equity: float = Field(
        default=0.0, description="Ending cash/equity for this window"
    )
    net_pnl: float = Field(default=0.0, description="Net profit/loss in currency units")
    total_return_pct: float = Field(
        default=0.0, description="Percentage return for this test window"
    )
    trade_count: int = Field(
        default=0, ge=0, description="Total completed round-trip trades"
    )
    winning_trades: int = Field(default=0, ge=0)
    losing_trades: int = Field(default=0, ge=0)
    win_rate: float = Field(default=0.0, description="Percentage of profitable trades")
    max_drawdown_pct: float = Field(
        default=0.0, description="Maximum peak-to-trough percentage drawdown"
    )
    max_drawdown_abs: float = Field(
        default=0.0, description="Maximum peak-to-trough absolute drawdown"
    )
    profit_factor: float | None = Field(
        default=None, description="Gross profit divided by gross loss"
    )
    sharpe_ratio: float | None = Field(
        default=None, description="Annualized risk-adjusted Sharpe ratio"
    )
    error_message: str | None = Field(
        default=None, description="Failure reason if window failed"
    )


class WalkForwardSummary(BaseModel):
    """Aggregated out-of-sample statistics across all evaluation windows."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    number_of_windows: int = Field(ge=0, description="Total windows attempted")
    successful_windows: int = Field(ge=0, description="Windows evaluated successfully")
    failed_windows: int = Field(
        ge=0, description="Windows that failed during evaluation"
    )
    total_return_pct: float = Field(
        description="Compounded cumulative out-of-sample return across windows (%)"
    )
    average_window_return: float = Field(
        description="Mean return across successful windows (%)"
    )
    median_window_return: float = Field(
        description="Median return across successful windows (%)"
    )
    average_drawdown: float = Field(
        description="Mean maximum drawdown across windows (%)"
    )
    worst_drawdown: float = Field(
        description="Worst maximum drawdown observed across all windows (%)"
    )
    total_trades: int = Field(ge=0, description="Total trades across all test windows")
    average_trades_per_window: float = Field(
        description="Mean trades executed per window"
    )
    overall_win_rate: float = Field(
        description="Aggregate win rate across all completed trades (%)"
    )
    sample_adequacy: SampleAdequacy = Field(
        description="Statistical sample confidence classification"
    )


class EvaluationReport(BaseModel):
    """Comprehensive artifact bundle for an entire walk-forward evaluation run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluation_id: str = Field(
        description="Unique evaluation run identifier (e.g. 'eval_20260905_120000_abc')"
    )
    experiment_id: str = Field(description="Associated experiment registry identifier")
    run_id: str = Field(description="Associated observability run tracker identifier")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of evaluation completion",
    )
    dataset_identifier: str = Field(
        description="Dataset name, path, or symbol identifier"
    )
    dataset_fingerprint: str = Field(
        description="SHA-256 fingerprint of the historical dataset"
    )
    config: EvaluationConfig = Field(description="Evaluation configuration settings")
    windows: list[WindowEvaluationResult] = Field(
        default_factory=list, description="Per-window out-of-sample results"
    )
    summary: WalkForwardSummary = Field(description="Aggregated out-of-sample summary")
    benchmark: dict[str, Any] | None = Field(
        default=None, description="Buy & Hold benchmark comparison metrics"
    )
    stability: dict[str, Any] | None = Field(
        default=None, description="Performance stability analysis report"
    )
    scenarios: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Market scenario and regime evaluation results",
    )
    cost_sensitivity: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Transaction cost sensitivity analysis results",
    )
    warnings: list[str] = Field(
        default_factory=list, description="Non-fatal warnings and observations"
    )
    limitations: list[str] = Field(
        default_factory=list, description="Documented empirical limitations"
    )

    def to_text_summary(self) -> str:
        """Format evaluation metrics as a human-readable text report."""
        lines = [
            "=" * 70,
            "         ADAPTIVE TRADING PLATFORM - WALK-FORWARD EVALUATION",
            "=" * 70,
            f"Evaluation ID    : {self.evaluation_id}",
            f"Experiment ID    : {self.experiment_id}",
            f"Run ID           : {self.run_id}",
            f"Dataset          : {self.dataset_identifier}",
            f"Fingerprint      : {self.dataset_fingerprint[:16]}...",
            f"Window Type      : {self.config.window_type.value}",
            f"Model Mode       : {self.config.model_mode.value}",
            "-" * 70,
            "AGGREGATE OUT-OF-SAMPLE PERFORMANCE:",
            (
                f"  - Total Windows: {self.summary.number_of_windows} "
                f"(Passed: {self.summary.successful_windows}, "
                f"Failed: {self.summary.failed_windows})"
            ),
            (
                f"  - Cumulative Out-of-Sample Return : "
                f"{self.summary.total_return_pct:+.2f}%"
            ),
            (
                f"  - Mean / Median Window Return     : "
                f"{self.summary.average_window_return:+.2f}% / "
                f"{self.summary.median_window_return:+.2f}%"
            ),
            f"  - Worst Window Drawdown           : {self.summary.worst_drawdown:.2f}%",
            (
                f"  - Total Trades (Win Rate)         : "
                f"{self.summary.total_trades} "
                f"({self.summary.overall_win_rate:.1f}%)"
            ),
            (
                f"  - Sample Adequacy Rating          : "
                f"{self.summary.sample_adequacy.value}"
            ),
            "-" * 70,
        ]

        if self.stability:
            lines.extend(
                [
                    "STABILITY ANALYSIS:",
                    (
                        f"  - Classification : "
                        f"{self.stability.get('classification', 'N/A')}"
                    ),
                    (
                        f"  - Win Window %   : "
                        f"{self.stability.get('win_window_ratio', 0.0):.1f}%"
                    ),
                    (
                        f"  - Best / Worst   : "
                        f"{self.stability.get('best_window_return', 0.0):+.2f}% / "
                        f"{self.stability.get('worst_window_return', 0.0):+.2f}%"
                    ),
                    (
                        f"  - Return Std Dev : "
                        f"{self.stability.get('return_std_dev', 0.0):.2f}%"
                    ),
                    "-" * 70,
                ]
            )

        if self.benchmark:
            lines.extend(
                [
                    "BENCHMARK COMPARISON (Buy & Hold):",
                    (
                        f"  - Strategy Return : "
                        f"{self.benchmark.get('strategy_return_pct', 0.0):+.2f}%"
                    ),
                    (
                        f"  - Benchmark Return: "
                        f"{self.benchmark.get('benchmark_return_pct', 0.0):+.2f}%"
                    ),
                    (
                        f"  - Outperformed    : "
                        f"{self.benchmark.get('outperformed_benchmark', False)}"
                    ),
                    (
                        f"  - Abs Difference  : "
                        f"{self.benchmark.get('absolute_difference_pct', 0.0):+.2f}%"
                    ),
                    "-" * 70,
                ]
            )

        if self.warnings:
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  [!] {w}")
            lines.append("-" * 70)

        lines.append("=" * 70)
        return "\n".join(lines)

    def save(self, base_dir: Path | str | None = None) -> Path:
        """Persist report.json, summary.txt, and manifest.json to disk."""
        target_dir = Path(base_dir or self.config.artifacts_dir) / self.evaluation_id
        target_dir.mkdir(parents=True, exist_ok=True)

        report_json = target_dir / "report.json"
        with open(report_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.model_dump(mode="json"), indent=2))

        summary_txt = target_dir / "summary.txt"
        with open(summary_txt, "w", encoding="utf-8") as f:
            f.write(self.to_text_summary())

        manifest_json = target_dir / "manifest.json"
        manifest_data = {
            "evaluation_id": self.evaluation_id,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "windows_count": len(self.windows),
            "created_at": self.created_at.isoformat(),
        }
        with open(manifest_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(manifest_data, indent=2))

        return target_dir
