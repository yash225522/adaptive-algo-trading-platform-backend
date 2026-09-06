"""Data & Model Quality Validation layer.

Validates historical datasets, features, models, and prevents leakage.
"""

from adaptive_trading.validation.config import (
    ValidationConfig,
    ValidationPolicy,
)
from adaptive_trading.validation.data_validator import DataValidator
from adaptive_trading.validation.exceptions import (
    DataLeakageError,
    DatasetValidationError,
    FeatureValidationError,
    ModelValidationError,
    ValidationError,
    ValidationGateError,
)
from adaptive_trading.validation.feature_validator import FeatureValidator
from adaptive_trading.validation.gate import (
    validate_for_backtest,
    validate_for_replay,
    validate_for_training,
)
from adaptive_trading.validation.leakage import LeakageValidator
from adaptive_trading.validation.model_validator import ModelValidator
from adaptive_trading.validation.report import (
    ValidationReport,
    ValidationResult,
)
from adaptive_trading.validation.rules import (
    ValidationRuleId,
    ValidationSeverity,
    ValidationStatus,
)
from adaptive_trading.validation.schema import DatasetSchemaValidator

__all__ = [
    "ValidationConfig",
    "ValidationPolicy",
    "ValidationSeverity",
    "ValidationStatus",
    "ValidationRuleId",
    "ValidationResult",
    "ValidationReport",
    "ValidationError",
    "DatasetValidationError",
    "FeatureValidationError",
    "ModelValidationError",
    "DataLeakageError",
    "ValidationGateError",
    "DatasetSchemaValidator",
    "DataValidator",
    "FeatureValidator",
    "LeakageValidator",
    "ModelValidator",
    "validate_for_backtest",
    "validate_for_replay",
    "validate_for_training",
]
