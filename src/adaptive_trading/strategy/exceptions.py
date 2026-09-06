"""Custom exception hierarchy for the strategy engine layer."""


class StrategyError(Exception):
    """Base exception for all strategy-related errors."""


class StrategyConfigError(StrategyError):
    """Raised when strategy threshold constraints are violated."""


class InvalidPredictionError(StrategyError):
    """Raised when an input prediction object fails validation."""


class DuplicateSignalError(StrategyError):
    """Raised when duplicate prediction timestamps are detected in a batch."""
