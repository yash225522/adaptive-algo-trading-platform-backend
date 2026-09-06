"""Custom exceptions for the risk management engine."""


class RiskError(Exception):
    """Base exception for all risk-related errors."""


class RiskConfigError(RiskError):
    """Raised when risk configuration settings are invalid."""


class RiskCheckError(RiskError):
    """Raised when an error occurs during risk check evaluation."""
