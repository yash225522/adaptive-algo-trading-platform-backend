"""Model validator: artifact integrity, fingerprint, dimensions, and output contract."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from adaptive_trading.experiments.fingerprint import compute_file_fingerprint
from adaptive_trading.experiments.models import ModelVersion
from adaptive_trading.ml.artifacts import ModelArtifactManager
from adaptive_trading.validation.config import ValidationConfig
from adaptive_trading.validation.report import ValidationResult
from adaptive_trading.validation.rules import (
    ValidationRuleId,
    ValidationSeverity,
    ValidationStatus,
)


class ModelValidator:
    """Validates ML model artifacts, registry hashes, and prediction contracts."""

    def __init__(
        self,
        config: ValidationConfig | None = None,
        artifact_manager: ModelArtifactManager | None = None,
    ) -> None:
        self.config = config or ValidationConfig()
        self.artifact_manager = artifact_manager or ModelArtifactManager()

    def validate_artifact(
        self,
        artifact_path: Path | str,
        expected_fingerprint: str | None = None,
    ) -> tuple[list[ValidationResult], Any]:
        """Validate artifact presence, readability, and SHA-256 fingerprint."""
        results: list[ValidationResult] = []
        p = Path(artifact_path)
        loaded_model = None

        if not p.exists():
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.MODEL_EXISTENCE.value,
                    severity=ValidationSeverity.CRITICAL,
                    status=ValidationStatus.FAIL,
                    message=f"Model artifact path does not exist: '{p}'",
                    details={"path": str(p)},
                )
            )
            return results, None

        results.append(
            ValidationResult(
                rule_id=ValidationRuleId.MODEL_EXISTENCE.value,
                severity=ValidationSeverity.INFO,
                status=ValidationStatus.PASS,
                message=f"Model artifact exists at '{p}'",
                details={"path": str(p)},
            )
        )

        target_file = p if p.is_file() else (p / "model.joblib")
        if not target_file.is_file():
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.MODEL_ARTIFACT_READABLE.value,
                    severity=ValidationSeverity.CRITICAL,
                    status=ValidationStatus.FAIL,
                    message=f"Model file 'model.joblib' not found in '{p}'",
                    details={"directory": str(p)},
                )
            )
            return results, None

        try:
            loaded_model = joblib.load(target_file)
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.MODEL_ARTIFACT_READABLE.value,
                    severity=ValidationSeverity.INFO,
                    status=ValidationStatus.PASS,
                    message=f"Model successfully loaded from '{target_file.name}'",
                    details={"class": type(loaded_model).__name__},
                )
            )
        except Exception as exc:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.MODEL_ARTIFACT_READABLE.value,
                    severity=ValidationSeverity.CRITICAL,
                    status=ValidationStatus.FAIL,
                    message=f"Failed to deserialize model artifact: {exc}",
                    details={"error": str(exc)},
                )
            )
            return results, None

        actual_fp = compute_file_fingerprint(target_file)
        if expected_fingerprint and actual_fp != expected_fingerprint:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.MODEL_FINGERPRINT_MATCH.value,
                    severity=ValidationSeverity.CRITICAL,
                    status=ValidationStatus.FAIL,
                    message=(
                        f"Model artifact fingerprint mismatch! Expected "
                        f"'{expected_fingerprint[:8]}', got '{actual_fp[:8]}'"
                    ),
                    details={"expected": expected_fingerprint, "actual": actual_fp},
                )
            )
        elif expected_fingerprint:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.MODEL_FINGERPRINT_MATCH.value,
                    severity=ValidationSeverity.INFO,
                    status=ValidationStatus.PASS,
                    message=(
                        f"Model artifact SHA-256 fingerprint verified ({actual_fp[:8]})"
                    ),
                    details={"fingerprint": actual_fp},
                )
            )

        return results, loaded_model

    def validate_version_compatibility(
        self,
        model_version: ModelVersion | str,
        runtime_feature_version: str,
        runtime_feature_names: list[str] | None = None,
    ) -> list[ValidationResult]:
        """Validate cross-version compatibility between model and feature pipeline."""
        results: list[ValidationResult] = []

        model_feat_ver = (
            model_version.feature_version
            if isinstance(model_version, ModelVersion)
            else "v1"
        )
        model_id = (
            model_version.model_id
            if isinstance(model_version, ModelVersion)
            else str(model_version)
        )

        if model_feat_ver != runtime_feature_version:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.MODEL_FEATURE_COMPATIBILITY.value,
                    severity=ValidationSeverity.CRITICAL,
                    status=ValidationStatus.FAIL,
                    message=(
                        f"Model '{model_id}' requires feature version "
                        f"'{model_feat_ver}', but runtime uses "
                        f"'{runtime_feature_version}'"
                    ),
                    details={
                        "model_feature_version": model_feat_ver,
                        "runtime_feature_version": runtime_feature_version,
                    },
                )
            )
        else:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.MODEL_FEATURE_COMPATIBILITY.value,
                    severity=ValidationSeverity.INFO,
                    status=ValidationStatus.PASS,
                    message=(
                        f"Model and runtime feature versions match "
                        f"('{runtime_feature_version}')"
                    ),
                    details={"feature_version": runtime_feature_version},
                )
            )

        return results

    def validate_input_contract(
        self,
        features: np.ndarray | list[list[float]],
        expected_feature_count: int,
        expected_feature_names: list[str] | None = None,
        actual_feature_names: list[str] | None = None,
    ) -> list[ValidationResult]:
        """Validate input matrix dimensions, feature names, and absence of NaNs/Infs."""
        results: list[ValidationResult] = []
        arr = np.asarray(features)

        if arr.ndim != 2:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.MODEL_INPUT_CONTRACT.value,
                    severity=ValidationSeverity.CRITICAL,
                    status=ValidationStatus.FAIL,
                    message=(
                        f"Expected 2D feature matrix (samples, features), "
                        f"got ndim={arr.ndim}"
                    ),
                    details={"ndim": arr.ndim, "shape": arr.shape},
                )
            )
            return results

        n_samples, n_feats = arr.shape

        if n_feats != expected_feature_count:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.MODEL_INPUT_CONTRACT.value,
                    severity=ValidationSeverity.CRITICAL,
                    status=ValidationStatus.FAIL,
                    message=(
                        f"Feature dimension mismatch: model expects "
                        f"{expected_feature_count} features, received {n_feats}"
                    ),
                    details={"expected": expected_feature_count, "actual": n_feats},
                )
            )
        else:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.MODEL_INPUT_CONTRACT.value,
                    severity=ValidationSeverity.INFO,
                    status=ValidationStatus.PASS,
                    message=f"Input feature dimensions match ({n_feats} features)",
                    details={"samples": n_samples, "features": n_feats},
                )
            )

        if expected_feature_names and actual_feature_names:
            if expected_feature_names != actual_feature_names:
                results.append(
                    ValidationResult(
                        rule_id=ValidationRuleId.MODEL_INPUT_CONTRACT.value,
                        severity=ValidationSeverity.CRITICAL,
                        status=ValidationStatus.FAIL,
                        message="Feature column names or ordering mismatch",
                        details={
                            "expected_names": expected_feature_names,
                            "actual_names": actual_feature_names,
                        },
                    )
                )

        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())
        if nan_count > 0 or inf_count > 0:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.MODEL_INPUT_CONTRACT.value,
                    severity=ValidationSeverity.CRITICAL,
                    status=ValidationStatus.FAIL,
                    message=f"Model input contains: {nan_count} NaNs, {inf_count} Infs",
                    details={"nan_count": nan_count, "inf_count": inf_count},
                )
            )

        return results

    def validate_output_contract(
        self,
        predictions: Any,
        is_probability: bool = True,
    ) -> list[ValidationResult]:
        """Validate inference outputs (finite values, probabilities in [0.0, 1.0])."""
        results: list[ValidationResult] = []

        if predictions is None:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.MODEL_OUTPUT_CONTRACT.value,
                    severity=ValidationSeverity.CRITICAL,
                    status=ValidationStatus.FAIL,
                    message="Model prediction output is None",
                    details={},
                )
            )
            return results

        arr = np.asarray(predictions)
        if arr.size == 0:
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.MODEL_OUTPUT_CONTRACT.value,
                    severity=ValidationSeverity.CRITICAL,
                    status=ValidationStatus.FAIL,
                    message="Model prediction output array is empty",
                    details={},
                )
            )
            return results

        if np.isnan(arr).any() or np.isinf(arr).any():
            results.append(
                ValidationResult(
                    rule_id=ValidationRuleId.MODEL_OUTPUT_CONTRACT.value,
                    severity=ValidationSeverity.CRITICAL,
                    status=ValidationStatus.FAIL,
                    message="Model prediction output contains NaN or infinite values",
                    details={"shape": arr.shape},
                )
            )
            return results

        if is_probability:
            if np.any(arr < 0.0) or np.any(arr > 1.0):
                results.append(
                    ValidationResult(
                        rule_id=ValidationRuleId.MODEL_OUTPUT_CONTRACT.value,
                        severity=ValidationSeverity.CRITICAL,
                        status=ValidationStatus.FAIL,
                        message="Probability predictions out of bounds [0.0, 1.0]",
                        details={
                            "min_val": float(np.min(arr)),
                            "max_val": float(np.max(arr)),
                        },
                    )
                )
                return results

        results.append(
            ValidationResult(
                rule_id=ValidationRuleId.MODEL_OUTPUT_CONTRACT.value,
                severity=ValidationSeverity.INFO,
                status=ValidationStatus.PASS,
                message=f"Model output contract verified ({arr.size} predictions)",
                details={"output_count": arr.size},
            )
        )
        return results
