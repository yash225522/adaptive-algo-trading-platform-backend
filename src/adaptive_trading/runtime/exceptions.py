"""Custom exceptions for the event loop and paper trading runtime."""


class RuntimeError_(Exception):
    """Base exception for all runtime errors."""


class EventProcessingError(RuntimeError_):
    """Raised when processing a specific market event fails."""


class WarmupIncompleteError(RuntimeError_):
    """Raised when feature generation is attempted before warmup period finishes."""


class DuplicateEventError(RuntimeError_):
    """Raised when an identical market event is submitted in strict mode."""


class InvalidEventOrderError(RuntimeError_):
    """Raised when market events arrive out of chronological order."""


class RuntimeConfigError(RuntimeError_):
    """Raised when runtime configuration validation fails."""
