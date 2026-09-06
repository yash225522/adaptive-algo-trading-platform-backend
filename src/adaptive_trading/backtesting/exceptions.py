"""Custom exception hierarchy for the backtesting engine layer."""


class BacktestError(Exception):
    """Base exception for all backtesting errors."""


class BacktestConfigError(BacktestError):
    """Raised when backtesting configuration parameters are invalid."""


class InsufficientDataError(BacktestError):
    """Raised when candles or signals are missing or insufficient for backtesting."""


class ExecutionError(BacktestError):
    """Raised when simulated order execution encounters an error."""
