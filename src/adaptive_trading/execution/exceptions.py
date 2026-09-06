"""Custom exceptions for the broker execution and paper trading layer."""


class BrokerError(Exception):
    """Base exception for all broker execution errors."""


class OrderValidationError(BrokerError):
    """Raised when an order request fails validation constraints."""


class InsufficientFundsError(BrokerError):
    """Raised when an account has insufficient cash or buying power."""


class OrderNotFoundError(BrokerError):
    """Raised when a requested order ID cannot be located."""


class ExecutionModeError(BrokerError):
    """Raised when execution mode configuration is invalid."""


class LiveTradingNotEnabledError(BrokerError):
    """Raised when live broker order placement is attempted while disabled."""
