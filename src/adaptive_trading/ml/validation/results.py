"""Data contracts and containers for walk-forward validation results."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from adaptive_trading.ml.validation.config import WalkForwardConfig


class OOSPrediction(BaseModel):
    """Single out-of-sample prediction record generated during validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime | None = Field(
        default=None, description="Observation timestamp"
    )
    symbol: str = Field(default="", description="Asset ticker symbol")
    actual_label: int = Field(description="True ground truth binary label")
    predicted_label: int = Field(description="Model predicted binary label")
    probability_up: float = Field(
        description="Predicted probability for positive class"
    )
    fold_number: int = Field(
        description="Validation fold index that generated this prediction"
    )


class WalkForwardFoldResult(BaseModel):
    """Validation performance metrics and metadata for a single fold."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fold_number: int
    train_start: datetime | None = None
    train_end: datetime | None = None
    validation_start: datetime | None = None
    validation_end: datetime | None = None
    train_samples: int
    validation_samples: int
    class_distribution: dict[str, int]
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None = None
    confusion_matrix: list[list[int]]
    baseline_metrics: dict[str, float]


class AggregateMetrics(BaseModel):
    """Statistical summary of metrics across all validation folds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_folds: int
    total_validation_samples: int
    mean_accuracy: float
    std_accuracy: float
    mean_precision: float
    std_precision: float
    mean_recall: float
    std_recall: float
    mean_f1: float
    std_f1: float
    mean_roc_auc: float | None = None
    mean_baseline_accuracy: float


class WalkForwardResult(BaseModel):
    """Complete container encapsulating an entire walk-forward experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str
    model_name: str
    feature_version: str
    target_version: str
    config: WalkForwardConfig
    fold_results: list[WalkForwardFoldResult]
    aggregate_metrics: AggregateMetrics
    oos_predictions: list[OOSPrediction] = Field(default_factory=list)
    created_at: datetime
