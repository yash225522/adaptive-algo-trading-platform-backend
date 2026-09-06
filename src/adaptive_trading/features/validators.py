"""Validation utilities for generated feature vectors."""

import math
from collections.abc import Sequence
from datetime import datetime

from adaptive_trading.domain.prediction import FeatureVector
from adaptive_trading.features.definitions import FEATURE_SET_VERSION


class FeatureValidationError(Exception):
    """Raised when a feature vector or sequence fails validation."""


class FeatureValidator:
    """Validates structural and numeric integrity of FeatureVectors."""

    def __init__(self, expected_version: str = FEATURE_SET_VERSION) -> None:
        self.expected_version = expected_version

    def validate_vector(
        self, vector: FeatureVector, allow_nan: bool = False
    ) -> list[str]:
        """Validate an individual FeatureVector.

        Args:
            vector: The FeatureVector instance to validate.
            allow_nan: If False, flags any NaN values as errors.

        Returns:
            list[str]: List of error message strings (empty if valid).
        """
        errors: list[str] = []

        # 1. Feature version check
        if vector.feature_version != self.expected_version:
            errors.append(
                f"Feature version mismatch: expected '{self.expected_version}', "
                f"got '{vector.feature_version}'"
            )

        # 2. Timezone-awareness check
        if (
            vector.timestamp.tzinfo is None
            or vector.timestamp.tzinfo.utcoffset(vector.timestamp) is None
        ):
            errors.append("FeatureVector timestamp must be timezone-aware")

        # 3. Numeric values check
        for fname, fval in vector.features.items():
            if math.isinf(fval):
                errors.append(
                    f"Feature '{fname}' has infinite value ({fval}) "
                    f"at {vector.timestamp}"
                )
            elif math.isnan(fval) and not allow_nan:
                errors.append(f"Feature '{fname}' has NaN value at {vector.timestamp}")

        return errors

    def validate_sequence(
        self,
        vectors: Sequence[FeatureVector],
        allow_nan: bool = False,
    ) -> list[str]:
        """Validate a sequence of FeatureVectors for integrity and chronology.

        Args:
            vectors: Sequence of FeatureVectors.
            allow_nan: If False, rejects any vectors with NaN values.

        Returns:
            list[str]: List of validation error strings (empty if valid).
        """
        if not vectors:
            return []

        errors: list[str] = []
        seen_timestamps: set[datetime] = set()

        for idx, vec in enumerate(vectors):
            # Check individual vector
            vec_errors = self.validate_vector(vec, allow_nan=allow_nan)
            errors.extend(vec_errors)

            # Check duplicates
            if vec.timestamp in seen_timestamps:
                errors.append(
                    f"Duplicate timestamp in feature sequence: {vec.timestamp}"
                )
            else:
                seen_timestamps.add(vec.timestamp)

            # Check strict chronology
            if idx > 0:
                prev_vec = vectors[idx - 1]
                if vec.timestamp <= prev_vec.timestamp:
                    errors.append(
                        f"Non-chronological order at index {idx}: "
                        f"{vec.timestamp} is not strictly after {prev_vec.timestamp}"
                    )

        return errors
