"""Exceptions for the experiment, dataset, model, and versioning subsystems."""


class ExperimentError(Exception):
    """Base exception for all experiment management errors."""


class ExperimentNotFoundError(ExperimentError):
    """Raised when the specified experiment identifier cannot be located."""


class InvalidExperimentStateError(ExperimentError):
    """Raised when an illegal lifecycle state transition is attempted."""


class IncompatibleVersionError(ExperimentError):
    """Raised when model, feature, or dataset versions are incompatible."""


class ReproducibilityError(ExperimentError):
    """Raised when an experiment fails reproducibility verification."""


class FingerprintMismatchError(ReproducibilityError):
    """Raised when calculated component fingerprints do not match manifest."""
