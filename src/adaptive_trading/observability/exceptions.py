"""Exceptions for observability, health checks, and run tracking."""


class ObservabilityError(Exception):
    """Base exception for all observability and monitoring errors."""


class RunTrackingError(ObservabilityError):
    """Raised when an operation on a run record fails."""


class InvalidRunTransitionError(RunTrackingError):
    """Raised when an invalid run lifecycle state transition is attempted."""


class RunNotFoundError(RunTrackingError):
    """Raised when the requested run ID is not found in storage."""


class HealthCheckError(ObservabilityError):
    """Raised when a system component health check encounters a fatal error."""


class MetricsError(ObservabilityError):
    """Raised when metric registration or calculation fails."""
