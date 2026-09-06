"""Target and label generation package for supervised machine learning."""

from adaptive_trading.targets.dataset import DatasetBuilder
from adaptive_trading.targets.definitions import (
    TARGET_VERSION,
    TargetConfig,
)
from adaptive_trading.targets.generator import TargetGenerator
from adaptive_trading.targets.models import (
    DirectionLabel,
    Target,
    TrainingExample,
)

__all__ = [
    "DatasetBuilder",
    "DirectionLabel",
    "TARGET_VERSION",
    "Target",
    "TargetConfig",
    "TargetGenerator",
    "TrainingExample",
]
