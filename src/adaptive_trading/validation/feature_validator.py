"""Feature quality validator: schema, missing/infinite values, and alignment."""

import math
from collections.abc import Sequence

import pandas as pd

from adaptive_trading.domain.market import Candle
from adaptive_trading.domain.prediction import FeatureVector
from adaptive_trading.features.definitions import (
    FEATURE_CATALOG_V1,
    FEATURE_SET_VERSION,
)
from adaptive_trading.validation.config import ValidationConfig
from adaptive_trading.validation.report import ValidationResult
from adaptive_trading.validation.rules import (
    ValidationRuleId,
    ValidationSeverity,
    ValidationStatus,
)


class FeatureValidator:
    """Validates structural completeness and numeric sanity of features."""

    def __init__(
        self,
        config: ValidationConfig | None = None,
        expected_version: str = FEATURE_SET_VERSION,
        expected_features: Sequence[str] | None = None,
    ) -> None:
        self.config = config or ValidationConfig()
        self.expected_version = expected_version
        self.expected_features = expected_features or list(FEATURE_CATALOG_V1.keys())

    def validate_feature_vectors(
        self,
        vectors: Sequence[FeatureVector],
        reference_candles: Sequence[Candle] | None = None,
    ) -> list[ValidationResult]:
        """Validate a sequence of FeatureVectors and check alignment."""
        results: list[ValidationResult] = []

        if not vectors:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.FEATURE_SCHEMA.value,
                    severity=ValidationSeverity.ERROR,
                    status=ValidationStatus.FAIL,
                    message="Feature vector sequence is empty",
                    details={"count": 0},
                )
            )
            return results

        # 1. Feature Schema & Version Check
        version_mismatches = 0
        missing_expected_features: set[str] = set()

        for v in vectors:
            if v.feature_version != self.expected_version:
                version_mismatches += 1
            present = set(v.features.keys())
            expected = set(self.expected_features)
            missing = expected - present
            if missing:
                missing_expected_features.update(missing)

        if version_mismatches > 0:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.FEATURE_SCHEMA.value,
                    severity=ValidationSeverity.ERROR,
                    status=ValidationStatus.FAIL,
                    message=(
                        f"Found {version_mismatches} vectors with mismatched "
                        f"version (expected '{self.expected_version}')"
                    ),
                    details={"version_mismatches": version_mismatches},
                )
            )
        elif missing_expected_features:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.FEATURE_SCHEMA.value,
                    severity=ValidationSeverity.ERROR,
                    status=ValidationStatus.FAIL,
                    message=(
                        f"Feature vectors missing expected features: "
                        f"{sorted(missing_expected_features)}"
                    ),
                    details={"missing_features": list(missing_expected_features)},
                )
            )
        else:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.FEATURE_SCHEMA.value,
                    severity=ValidationSeverity.INFO,
                    status=ValidationStatus.PASS,
                    message=(
                        f"Feature schema complete ({len(self.expected_features)} "
                        f"features, v'{self.expected_version}')"
                    ),
                    details={"feature_count": len(self.expected_features)},
                )
            )

        # 2. Missing (NaN) & Infinite Values Check
        nan_occurrences: list[str] = []
        inf_occurrences: list[str] = []

        for idx, v in enumerate(vectors):
            for fname, fval in v.features.items():
                if math.isinf(fval):
                    inf_occurrences.append(
                        f"Vector {idx} ({v.timestamp}): '{fname}' = {fval}"
                    )
                elif math.isnan(fval):
                    nan_occurrences.append(
                        f"Vector {idx} ({v.timestamp}): '{fname}' is NaN"
                    )

        if inf_occurrences:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.FEATURE_INFINITE_VALUES.value,
                    severity=ValidationSeverity.CRITICAL,
                    status=ValidationStatus.FAIL,
                    message=f"Detected {len(inf_occurrences)} infinite feature values",
                    details={
                        "infinite_values": inf_occurrences[:10],
                        "total": len(inf_occurrences),
                    },
                )
            )
        else:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.FEATURE_INFINITE_VALUES.value,
                    severity=ValidationSeverity.INFO,
                    status=ValidationStatus.PASS,
                    message="Zero infinite values detected in features",
                    details={},
                )
            )

        if nan_occurrences:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.FEATURE_MISSING_VALUES.value,
                    severity=ValidationSeverity.ERROR,
                    status=ValidationStatus.FAIL,
                    message=f"Detected {len(nan_occurrences)} NaN feature values",
                    details={
                        "nan_occurrences": nan_occurrences[:10],
                        "total": len(nan_occurrences),
                    },
                )
            )
        else:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.FEATURE_MISSING_VALUES.value,
                    severity=ValidationSeverity.INFO,
                    status=ValidationStatus.PASS,
                    message="Zero NaN missing values in feature vectors",
                    details={},
                )
            )

        # 3. Timestamp Alignment against Candles
        if reference_candles:
            candle_ts_set = {c.timestamp for c in reference_candles}
            vector_ts_list = [v.timestamp for v in vectors]
            misaligned: list[str] = []

            for ts in vector_ts_list:
                if ts not in candle_ts_set:
                    misaligned.append(str(ts))

            if misaligned:
                results.append(
                    ValidationResult(
                        rule_id=ValidationRuleId.FEATURE_TIMESTAMP_ALIGNMENT.value,
                        severity=ValidationSeverity.ERROR,
                        status=ValidationStatus.FAIL,
                        message=(
                            f"Found {len(misaligned)} feature vectors with "
                            "timestamps absent from candles"
                        ),
                        details={"misaligned_timestamps": misaligned[:10]},
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        rule_id=ValidationRuleId.FEATURE_TIMESTAMP_ALIGNMENT.value,
                        severity=ValidationSeverity.INFO,
                        status=ValidationStatus.PASS,
                        message="All feature vector timestamps align with candles",
                        details={"aligned_vectors": len(vectors)},
                    )
                )

        return results

    def validate_features_df(
        self,
        df: pd.DataFrame,
        reference_candles: Sequence[Candle] | None = None,
    ) -> list[ValidationResult]:
        """Validate a tabular DataFrame of extracted features."""
        results: list[ValidationResult] = []

        if df.empty:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.FEATURE_SCHEMA.value,
                    severity=ValidationSeverity.ERROR,
                    status=ValidationStatus.FAIL,
                    message="Feature DataFrame is empty",
                    details={"rows": 0},
                )
            )
            return results

        # Schema & required feature columns
        feature_cols = [c for c in self.expected_features if c in df.columns]
        missing_cols = [c for c in self.expected_features if c not in df.columns]

        if missing_cols:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.FEATURE_SCHEMA.value,
                    severity=ValidationSeverity.ERROR,
                    status=ValidationStatus.FAIL,
                    message=f"Missing feature columns: {missing_cols}",
                    details={"missing_columns": missing_cols},
                )
            )
        else:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.FEATURE_SCHEMA.value,
                    severity=ValidationSeverity.INFO,
                    status=ValidationStatus.PASS,
                    message=f"All {len(self.expected_features)} features present",
                    details={"features": feature_cols},
                )
            )

        # Missing and infinite values
        nan_counts = {
            col: int(df[col].isna().sum())
            for col in feature_cols
            if int(df[col].isna().sum()) > 0
        }
        inf_counts = {}
        for col in feature_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            infinities = int(series.isin([float("inf"), float("-inf")]).sum())
            if infinities > 0:
                inf_counts[col] = infinities

        if inf_counts:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.FEATURE_INFINITE_VALUES.value,
                    severity=ValidationSeverity.CRITICAL,
                    status=ValidationStatus.FAIL,
                    message=f"Infinite values detected in columns: {inf_counts}",
                    details={"infinite_counts": inf_counts},
                )
            )
        else:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.FEATURE_INFINITE_VALUES.value,
                    severity=ValidationSeverity.INFO,
                    status=ValidationStatus.PASS,
                    message="Zero infinite values in feature DataFrame",
                    details={},
                )
            )

        if nan_counts:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.FEATURE_MISSING_VALUES.value,
                    severity=ValidationSeverity.ERROR,
                    status=ValidationStatus.FAIL,
                    message=f"NaN values in post-warmup features: {nan_counts}",
                    details={"nan_counts": nan_counts},
                )
            )
        else:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.FEATURE_MISSING_VALUES.value,
                    severity=ValidationSeverity.INFO,
                    status=ValidationStatus.PASS,
                    message="Zero NaN values in feature columns",
                    details={},
                )
            )

        # Timestamp alignment
        if "timestamp" in df.columns and reference_candles:
            df_ts = set(pd.to_datetime(df["timestamp"]).dt.tz_localize(None))
            candle_ts = {c.timestamp.replace(tzinfo=None) for c in reference_candles}
            diff = df_ts - candle_ts
            if diff:
                results.append(
                    ValidationResult(
                        rule_id=ValidationRuleId.FEATURE_TIMESTAMP_ALIGNMENT.value,
                        severity=ValidationSeverity.ERROR,
                        status=ValidationStatus.FAIL,
                        message=f"{len(diff)} feature rows missing in candle data",
                        details={"missing_count": len(diff)},
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        rule_id=ValidationRuleId.FEATURE_TIMESTAMP_ALIGNMENT.value,
                        severity=ValidationSeverity.INFO,
                        status=ValidationStatus.PASS,
                        message="Feature timestamps align with candle dataset",
                        details={"rows": len(df)},
                    )
                )

        return results
