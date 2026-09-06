"""Configuration contracts for advanced walk-forward evaluation and backtesting."""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from adaptive_trading.validation.config import ValidationPolicy


class WindowType(StrEnum):
    """Time-series window progression strategy."""

    ROLLING = "ROLLING"
    EXPANDING = "EXPANDING"


class ModelMode(StrEnum):
    """Model management strategy across walk-forward windows."""

    STATIC_MODEL = "STATIC_MODEL"
    RETRAIN_PER_WINDOW = "RETRAIN_PER_WINDOW"


class FailurePolicy(StrEnum):
    """Handling policy when an individual evaluation window fails."""

    FAIL_FAST = "FAIL_FAST"
    CONTINUE = "CONTINUE"


class CostScenario(StrEnum):
    """Pre-defined transaction cost sensitivity scenarios."""

    LOW_COST = "LOW_COST"
    NORMAL_COST = "NORMAL_COST"
    HIGH_COST = "HIGH_COST"
    CUSTOM = "CUSTOM"


class EvaluationConfig(BaseModel):
    """Configuration settings for walk-forward evaluation and sensitivity runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    train_window: int | float = Field(
        default=50,
        description="Training window size in samples (>1) or fraction (0.0 < x < 1.0)",
    )
    validation_window: int | float | None = Field(
        default=None,
        description="Optional validation window size in samples or fraction",
    )
    test_window: int | float = Field(
        default=10,
        description="Out-of-sample test window size in samples or fraction",
    )
    step_size: int | float | None = Field(
        default=None,
        description="Step size to advance each fold; defaults to test_window if None",
    )
    window_type: WindowType = Field(
        default=WindowType.ROLLING,
        description="Window progression type (ROLLING or EXPANDING)",
    )
    model_mode: ModelMode = Field(
        default=ModelMode.STATIC_MODEL,
        description="Whether to use a pre-existing model or retrain per window",
    )
    failure_policy: FailurePolicy = Field(
        default=FailurePolicy.CONTINUE,
        description="Behavior when an individual window evaluation encounters an error",
    )
    purge_period: int = Field(
        default=0,
        ge=0,
        description="Embargo/purge candle count between split partitions",
    )
    min_train_samples: int = Field(
        default=20,
        ge=5,
        description="Minimum training samples threshold",
    )
    min_trades_for_adequacy: int = Field(
        default=5,
        ge=1,
        description="Minimum trades required to rate sample as ADEQUATE_SAMPLE",
    )
    min_windows_for_adequacy: int = Field(
        default=3,
        ge=1,
        description="Minimum windows required to rate sample as ADEQUATE_SAMPLE",
    )
    initial_capital: float = Field(
        default=100_000.0,
        gt=0,
        description="Starting cash capital for each window backtest simulation",
    )
    commission_bps: float = Field(
        default=3.0,
        ge=0,
        description="Commission rate in basis points (1 bp = 0.01% = 0.0001)",
    )
    slippage_bps: float = Field(
        default=5.0,
        ge=0,
        description="Slippage rate in basis points",
    )
    fixed_quantity: float = Field(
        default=1.0,
        gt=0,
        description="Fixed order size for backtest execution",
    )
    validation_policy: ValidationPolicy = Field(
        default=ValidationPolicy.NORMAL,
        description="Step 21 pre-run data gating policy",
    )
    save_artifacts: bool = Field(
        default=True,
        description="Whether to persist evaluation reports and summary to disk",
    )
    artifacts_dir: Path = Field(
        default=Path("artifacts/evaluations"),
        description="Base directory where evaluation artifacts are stored",
    )

    @model_validator(mode="after")
    def validate_fraction_sums(self) -> "EvaluationConfig":
        """Validate fractional parameters do not exceed dataset bounds."""
        fractions: list[float] = []
        if isinstance(self.train_window, float) and 0.0 < self.train_window < 1.0:
            fractions.append(self.train_window)
        if (
            isinstance(self.validation_window, float)
            and 0.0 < self.validation_window < 1.0
        ):
            fractions.append(self.validation_window)
        if isinstance(self.test_window, float) and 0.0 < self.test_window < 1.0:
            fractions.append(self.test_window)

        if fractions and sum(fractions) >= 1.0:
            raise ValueError(
                f"Sum of fractional window sizes ({sum(fractions):.2f}) "
                "must be strictly less than 1.0"
            )
        return self

