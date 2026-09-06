"""Feature engineering and transformation pipeline package."""

from adaptive_trading.features.calculator import FeatureCalculator
from adaptive_trading.features.definitions import (
    FEATURE_CATALOG_V1,
    FEATURE_SET_VERSION,
    MAX_LOOKBACK_PERIODS_V1,
    FeatureDefinition,
)
from adaptive_trading.features.pipeline import FeaturePipeline
from adaptive_trading.features.validators import (
    FeatureValidationError,
    FeatureValidator,
)

__all__ = [
    "FEATURE_CATALOG_V1",
    "FEATURE_SET_VERSION",
    "FeatureCalculator",
    "FeatureDefinition",
    "FeaturePipeline",
    "FeatureValidationError",
    "FeatureValidator",
    "MAX_LOOKBACK_PERIODS_V1",
]
