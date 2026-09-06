"""Custom exceptions for instrument master management and resolution."""


class InstrumentError(Exception):
    """Base exception for all instrument management errors."""


class InstrumentNotFoundError(InstrumentError):
    """Raised when no instrument matches the specified query criteria."""


class InstrumentResolutionError(InstrumentError):
    """Raised when an instrument query is ambiguous or matches multiple contracts."""


class InstrumentImportError(InstrumentError):
    """Raised when fetching, parsing, or storing the instrument master fails."""
