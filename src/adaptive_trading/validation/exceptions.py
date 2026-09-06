"""Exception hierarchy for the data and model quality validation subsystem."""


class ValidationError(Exception):
    """Base exception for all validation errors."""


class DatasetValidationError(ValidationError):
    """Raised when a market dataset fails schema, integrity, or time-series checks."""


class FeatureValidationError(ValidationError):
    """Raised when feature vectors or matrix fail structural or numeric validation."""


class ModelValidationError(ValidationError):
    """Raised when an ML model artifact, schema, or output contract fails validation."""


class DataLeakageError(ValidationError):
    """Raised when look-ahead, target, or train/test split leakage is detected."""


class ValidationGateError(ValidationError):
    """Raised when a pre-run validation gate rejects execution due to policy rules."""
