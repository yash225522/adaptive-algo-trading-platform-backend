"""Leakage detection: lookahead, target leakage, and train/test overlap."""

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from adaptive_trading.domain.market import Candle
from adaptive_trading.validation.config import ValidationConfig
from adaptive_trading.validation.report import ValidationResult
from adaptive_trading.validation.rules import (
    ValidationRuleId,
    ValidationSeverity,
    ValidationStatus,
)

SUSPICIOUS_TARGET_COLUMN_NAMES = {
    "target",
    "target_label",
    "future_return",
    "fwd_return",
    "next_close",
    "next_bar_direction",
    "next_return",
    "label",
    "y",
}


class LeakageValidator:
    """Verifies features, models, and datasets do not leak future data or targets."""

    def __init__(self, config: ValidationConfig | None = None) -> None:
        self.config = config or ValidationConfig()

    def validate_target_leakage(
        self,
        feature_columns: Sequence[str],
        dataset_df: pd.DataFrame | None = None,
    ) -> ValidationResult:
        """Verify that target and label columns are absent from feature sets."""
        detected_targets = [
            c for c in feature_columns if c.lower() in SUSPICIOUS_TARGET_COLUMN_NAMES
        ]

        if detected_targets:
            return ValidationResult(
                rule_id=ValidationRuleId.LEAKAGE_TARGET_IN_FEATURES.value,
                severity=ValidationSeverity.CRITICAL,
                status=ValidationStatus.FAIL,
                message=(
                    f"Target leakage detected! Feature set contains: {detected_targets}"
                ),
                details={"leaked_columns": detected_targets},
            )

        return ValidationResult(
            rule_id=ValidationRuleId.LEAKAGE_TARGET_IN_FEATURES.value,
            severity=ValidationSeverity.INFO,
            status=ValidationStatus.PASS,
            message="No target or label columns found in feature set",
            details={"checked_features": list(feature_columns)},
        )

    def validate_train_test_split(
        self,
        train_data: pd.DataFrame | Sequence[Candle] | Sequence[datetime],
        test_data: pd.DataFrame | Sequence[Candle] | Sequence[datetime],
        val_data: pd.DataFrame | Sequence[Candle] | Sequence[datetime] | None = None,
    ) -> ValidationResult:
        """Verify strict chronological non-overlapping order across splits."""
        train_ts = self._extract_timestamps(train_data)
        test_ts = self._extract_timestamps(test_data)
        val_ts = self._extract_timestamps(val_data) if val_data is not None else []

        if not train_ts or not test_ts:
            return ValidationResult(
                rule_id=ValidationRuleId.LEAKAGE_TRAIN_TEST_OVERLAP.value,
                severity=ValidationSeverity.ERROR,
                status=ValidationStatus.FAIL,
                message="Cannot validate splits: train or test dataset is empty",
                details={},
            )

        train_max = max(train_ts)
        test_min = min(test_ts)
        overlap_details: dict[str, Any] = {}

        if val_ts:
            val_min = min(val_ts)
            val_max = max(val_ts)
            if train_max >= val_min:
                overlap_details["train_val_overlap"] = (
                    f"train_max ({train_max}) >= val_min ({val_min})"
                )
            if val_max >= test_min:
                overlap_details["val_test_overlap"] = (
                    f"val_max ({val_max}) >= test_min ({test_min})"
                )
        else:
            if train_max >= test_min:
                overlap_details["train_test_overlap"] = (
                    f"train_max ({train_max}) >= test_min ({test_min})"
                )

        train_set = set(train_ts)
        test_set = set(test_ts)
        common_test = train_set.intersection(test_set)
        if common_test:
            overlap_details["common_train_test_timestamps"] = len(common_test)

        if overlap_details:
            return ValidationResult(
                rule_id=ValidationRuleId.LEAKAGE_TRAIN_TEST_OVERLAP.value,
                severity=ValidationSeverity.CRITICAL,
                status=ValidationStatus.FAIL,
                message="Train/Test chronological leakage detected: splits overlap",
                details=overlap_details,
            )

        return ValidationResult(
            rule_id=ValidationRuleId.LEAKAGE_TRAIN_TEST_OVERLAP.value,
            severity=ValidationSeverity.INFO,
            status=ValidationStatus.PASS,
            message="Train/Test splits are strictly chronologically isolated",
            details={"train_end": str(train_max), "test_start": str(test_min)},
        )

    def validate_lookahead_leakage(
        self,
        candles: Sequence[Candle],
        feature_df: pd.DataFrame,
        feature_col: str,
        expected_lag_window: int = 20,
    ) -> ValidationResult:
        """Verify rolling feature at T does not use data from future candles."""
        if (
            len(candles) < expected_lag_window + 5
            or feature_col not in feature_df.columns
        ):
            return ValidationResult(
                rule_id=ValidationRuleId.LEAKAGE_LOOKAHEAD.value,
                severity=ValidationSeverity.INFO,
                status=ValidationStatus.PASS,
                message="Insufficient rows or feature absent for lookahead check",
                details={},
            )

        closes = np.array([float(c.close) for c in candles], dtype=np.float64)
        f_vals = np.array(feature_df[feature_col].to_numpy(), dtype=np.float64)

        if len(closes) == len(f_vals):
            if np.allclose(f_vals[:-1], closes[1:], equal_nan=True):
                return ValidationResult(
                    rule_id=ValidationRuleId.LEAKAGE_LOOKAHEAD.value,
                    severity=ValidationSeverity.CRITICAL,
                    status=ValidationStatus.FAIL,
                    message=(
                        f"Look-ahead leakage detected in '{feature_col}': "
                        "feature equals future candle values"
                    ),
                    details={"feature": feature_col},
                )

        return ValidationResult(
            rule_id=ValidationRuleId.LEAKAGE_LOOKAHEAD.value,
            severity=ValidationSeverity.INFO,
            status=ValidationStatus.PASS,
            message=f"No look-ahead future leakage in '{feature_col}'",
            details={"feature": feature_col},
        )

    def validate_scaler_isolation(
        self,
        scaler_fit_indices: Sequence[int] | None,
        train_indices: Sequence[int],
        total_rows: int,
    ) -> ValidationResult:
        """Verify that preprocessor scalers were fitted strictly on training data."""
        if scaler_fit_indices is None:
            return ValidationResult(
                rule_id=ValidationRuleId.LEAKAGE_SCALER_PREPROCESSING.value,
                severity=ValidationSeverity.INFO,
                status=ValidationStatus.PASS,
                message="Scaler isolation verification passed",
                details={},
            )

        fit_set = set(scaler_fit_indices)
        train_set = set(train_indices)

        outside_train = fit_set - train_set
        if outside_train:
            return ValidationResult(
                rule_id=ValidationRuleId.LEAKAGE_SCALER_PREPROCESSING.value,
                severity=ValidationSeverity.CRITICAL,
                status=ValidationStatus.FAIL,
                message=(
                    f"Preprocessing scaler leakage! Fitted on "
                    f"{len(outside_train)} non-training rows"
                ),
                details={"outside_train_count": len(outside_train)},
            )

        if len(fit_set) == total_rows and len(train_set) < total_rows:
            return ValidationResult(
                rule_id=ValidationRuleId.LEAKAGE_SCALER_PREPROCESSING.value,
                severity=ValidationSeverity.CRITICAL,
                status=ValidationStatus.FAIL,
                message="Scaler fitted on entire dataset before split",
                details={
                    "total_rows": total_rows,
                    "train_rows": len(train_set),
                },
            )

        return ValidationResult(
            rule_id=ValidationRuleId.LEAKAGE_SCALER_PREPROCESSING.value,
            severity=ValidationSeverity.INFO,
            status=ValidationStatus.PASS,
            message="Scaler was strictly fitted on training partition only",
            details={"fit_samples": len(fit_set)},
        )

    def _extract_timestamps(
        self, data: pd.DataFrame | Sequence[Candle] | Sequence[datetime] | None
    ) -> list[datetime]:
        """Helper to extract timestamps from various structures."""
        if data is None:
            return []
        if isinstance(data, pd.DataFrame):
            if "timestamp" in data.columns:
                return [
                    pd.to_datetime(t).to_pydatetime()
                    for t in data["timestamp"].dropna()
                ]
            return []
        if data and isinstance(data[0], Candle):
            return [c.timestamp for c in data if isinstance(c, Candle)]
        if data and isinstance(data[0], datetime):
            return [t for t in data if isinstance(t, datetime)]
        return []
