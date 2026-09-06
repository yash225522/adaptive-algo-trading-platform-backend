"""Machine learning baseline modeling and walk-forward validation package."""

from adaptive_trading.ml.artifacts import ArtifactError, ModelArtifactManager
from adaptive_trading.ml.config import MLConfig
from adaptive_trading.ml.datasets import (
    ChronologicalSplitter,
    DatasetSplit,
    DatasetSplitError,
)
from adaptive_trading.ml.evaluator import EvaluationReport, ModelEvaluator
from adaptive_trading.ml.models import BaseMLModel, LogisticRegressionModel
from adaptive_trading.ml.preprocessing import MLPreprocessor, PreprocessingError
from adaptive_trading.ml.trainer import MLTrainer
from adaptive_trading.ml.validation import (
    AggregateMetrics,
    OOSPrediction,
    WalkForwardConfig,
    WalkForwardFoldResult,
    WalkForwardResult,
    WalkForwardRunner,
    WalkForwardSplit,
    WalkForwardSplitter,
    WindowType,
)

__all__ = [
    "AggregateMetrics",
    "ArtifactError",
    "BaseMLModel",
    "ChronologicalSplitter",
    "DatasetSplit",
    "DatasetSplitError",
    "EvaluationReport",
    "LogisticRegressionModel",
    "MLConfig",
    "MLPreprocessor",
    "MLTrainer",
    "ModelArtifactManager",
    "ModelEvaluator",
    "OOSPrediction",
    "PreprocessingError",
    "WalkForwardConfig",
    "WalkForwardFoldResult",
    "WalkForwardResult",
    "WalkForwardRunner",
    "WalkForwardSplit",
    "WalkForwardSplitter",
    "WindowType",
]
