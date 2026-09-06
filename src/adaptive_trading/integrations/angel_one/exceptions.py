"""Custom exceptions for Angel One SmartAPI integration."""


class AngelOneError(Exception):
    """Base exception for all Angel One SmartAPI errors."""


class AngelOneConfigurationError(AngelOneError):
    """Raised when Angel One credentials or settings are missing or invalid."""


class AngelOneAuthenticationError(AngelOneError):
    """Raised when Angel One authentication or session generation fails."""


class AngelOneSessionError(AngelOneError):
    """Raised when an active session is missing, invalid, or expired."""


class AngelOneHistoricalDataError(AngelOneError):
    """Raised when historical candle fetching or processing fails."""
