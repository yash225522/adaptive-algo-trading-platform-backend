"""Exception hierarchy for the evaluation and walk-forward subsystem."""


class EvaluationError(Exception):
    """Base exception for all evaluation errors."""


class InsufficientDataError(EvaluationError):
    """Raised when dataset contains fewer samples than required by windows."""


class TemporalOrderError(EvaluationError):
    """Raised when train/validation/test chronological ordering is violated."""


class WindowEvaluationError(EvaluationError):
    """Raised when an individual evaluation window fails during execution."""


class ConfigurationFreezeError(EvaluationError):
    """Raised if configuration modification is attempted on a frozen test window."""


class BenchmarkEvaluationError(EvaluationError):
    """Raised when benchmark return or drawdown calculation fails."""


class StabilityAnalysisError(EvaluationError):
    """Raised when statistical performance stability analysis fails."""

