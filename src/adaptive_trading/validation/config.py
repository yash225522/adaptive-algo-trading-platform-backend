"""Configuration contracts for validation rules, thresholds, and policies."""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ValidationPolicy(str, Enum):
    """Execution gating policies governing how validation findings are handled."""

    STRICT = "STRICT"  # Any WARNING, ERROR, or CRITICAL blocks execution
    NORMAL = "NORMAL"  # Only ERROR or CRITICAL blocks execution (Warnings allowed)
    LENIENT = "LENIENT"  # Only CRITICAL blocks execution (Diagnostic mode)


class ValidationConfig(BaseModel):
    """Configuration settings and anomaly thresholds for validation checks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy: ValidationPolicy = Field(
        default=ValidationPolicy.NORMAL,
        description="Validation gating policy for pre-run checks",
    )
    max_reasonable_return: float = Field(
        default=0.50,
        description=(
            "Single-candle percentage price change threshold for warning (0.50 = 50%)"
        ),
    )
    allow_zero_volume: bool = Field(
        default=True,
        description="Whether zero volume produces a WARNING instead of ERROR",
    )
    strict_timezone: bool = Field(
        default=True,
        description="Whether timestamps must explicitly be timezone-aware (e.g. UTC)",
    )
    expected_symbol: str | None = Field(
        default=None,
        description="Optional expected ticker symbol for single-instrument datasets",
    )
    expected_timeframe: str | None = Field(
        default=None,
        description="Optional expected timeframe string (e.g. '5m', '1m', '15m')",
    )
    check_lookahead_leakage: bool = Field(
        default=True,
        description="Whether to perform rolling-window and future look-ahead checks",
    )
    check_target_leakage: bool = Field(
        default=True,
        description="Whether to check that target columns are absent from feature sets",
    )
    check_train_test_overlap: bool = Field(
        default=True,
        description=(
            "Whether to verify chronological non-overlap between train and test splits"
        ),
    )
    check_scaler_isolation: bool = Field(
        default=True,
        description=(
            "Whether to verify scaler preprocessing was fitted only on training data"
        ),
    )
    save_artifacts: bool = Field(
        default=True,
        description="Whether to persist validation reports as artifacts on disk",
    )
    artifacts_dir: Path = Field(
        default=Path("artifacts/validation"),
        description="Base directory where validation report artifacts are stored",
    )
