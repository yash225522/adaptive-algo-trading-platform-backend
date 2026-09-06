"""Pre-run execution gates for backtesting, paper replay, and model training."""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from adaptive_trading.domain.market import Candle
from adaptive_trading.validation.config import ValidationConfig, ValidationPolicy
from adaptive_trading.validation.data_validator import DataValidator
from adaptive_trading.validation.exceptions import ValidationGateError
from adaptive_trading.validation.feature_validator import FeatureValidator
from adaptive_trading.validation.leakage import LeakageValidator
from adaptive_trading.validation.model_validator import ModelValidator
from adaptive_trading.validation.report import ValidationReport, ValidationResult


def validate_for_backtest(
    candles: Sequence[Candle] | pd.DataFrame,
    feature_df: pd.DataFrame | None = None,
    model_artifact_path: Path | str | None = None,
    config: ValidationConfig | None = None,
    policy: ValidationPolicy | None = None,
) -> ValidationReport:
    """Execute pre-run gating checks before launching a backtest simulation."""
    cfg = config or ValidationConfig()
    active_policy = policy or cfg.policy
    checks: list[ValidationResult] = []

    # 1. Dataset Validation
    data_validator = DataValidator(config=cfg)
    data_results, dataset_fp = data_validator.validate_dataset(candles)
    checks.extend(data_results)

    # 2. Feature & Leakage Validation
    if feature_df is not None:
        feat_validator = FeatureValidator(config=cfg)
        feat_results = feat_validator.validate_features_df(feature_df)
        checks.extend(feat_results)

        leakage_validator = LeakageValidator(config=cfg)
        target_res = leakage_validator.validate_target_leakage(list(feature_df.columns))
        checks.append(target_res)

    # 3. Model Compatibility Validation
    if model_artifact_path:
        model_validator = ModelValidator(config=cfg)
        model_results, _ = model_validator.validate_artifact(model_artifact_path)
        checks.extend(model_results)

    report = ValidationReport.create(
        checks=checks,
        dataset_identifier="backtest_dataset",
        policy=active_policy,
        dataset_fingerprint=dataset_fp,
    )

    if cfg.save_artifacts:
        report.save(base_dir=cfg.artifacts_dir)

    if not report.is_allowed(active_policy):
        err_msg = (
            f"Backtest execution rejected by validation gate "
            f"({report.overall_status.value} under {active_policy.value} policy).\n"
            f"Details:\n{report.to_text_summary()}"
        )
        raise ValidationGateError(err_msg)

    return report


def validate_for_replay(
    candles: Sequence[Candle] | pd.DataFrame,
    expected_symbol: str | None = None,
    model_artifact_path: Path | str | None = None,
    config: ValidationConfig | None = None,
    policy: ValidationPolicy | None = None,
) -> ValidationReport:
    """Execute pre-run gating checks before launching paper trading replay."""
    cfg = config or ValidationConfig()
    active_policy = policy or cfg.policy
    checks: list[ValidationResult] = []

    # 1. Dataset Validation
    data_validator = DataValidator(config=cfg)
    data_results, dataset_fp = data_validator.validate_dataset(
        candles, expected_symbol=expected_symbol
    )
    checks.extend(data_results)

    # 2. Model Validation
    if model_artifact_path:
        model_validator = ModelValidator(config=cfg)
        model_results, _ = model_validator.validate_artifact(model_artifact_path)
        checks.extend(model_results)

    report = ValidationReport.create(
        checks=checks,
        dataset_identifier="paper_replay_dataset",
        policy=active_policy,
        dataset_fingerprint=dataset_fp,
    )

    if cfg.save_artifacts:
        report.save(base_dir=cfg.artifacts_dir)

    if not report.is_allowed(active_policy):
        err_msg = (
            f"Paper replay simulation rejected by validation gate "
            f"({report.overall_status.value} under {active_policy.value} policy).\n"
            f"Details:\n{report.to_text_summary()}"
        )
        raise ValidationGateError(err_msg)

    return report


def validate_for_training(
    dataset_df: pd.DataFrame,
    train_indices: Sequence[int] | None = None,
    test_indices: Sequence[int] | None = None,
    scaler_fit_indices: Sequence[int] | None = None,
    config: ValidationConfig | None = None,
    policy: ValidationPolicy | None = None,
) -> ValidationReport:
    """Execute pre-run gating checks before training an ML model."""
    cfg = config or ValidationConfig()
    active_policy = policy or cfg.policy
    checks: list[ValidationResult] = []

    # 1. Schema & Feature Checks
    feat_validator = FeatureValidator(config=cfg)
    feat_results = feat_validator.validate_features_df(dataset_df)
    checks.extend(feat_results)

    # 2. Target Leakage in Features
    leakage_validator = LeakageValidator(config=cfg)
    target_res = leakage_validator.validate_target_leakage(list(dataset_df.columns))
    checks.append(target_res)

    # 3. Train/Test Overlap Check
    if (
        train_indices is not None
        and test_indices is not None
        and "timestamp" in dataset_df.columns
    ):
        train_df = dataset_df.iloc[list(train_indices)]
        test_df = dataset_df.iloc[list(test_indices)]
        split_res = leakage_validator.validate_train_test_split(train_df, test_df)
        checks.append(split_res)

    # 4. Scaler Fit Isolation Check
    if train_indices is not None and scaler_fit_indices is not None:
        scaler_res = leakage_validator.validate_scaler_isolation(
            scaler_fit_indices=scaler_fit_indices,
            train_indices=train_indices,
            total_rows=len(dataset_df),
        )
        checks.append(scaler_res)

    report = ValidationReport.create(
        checks=checks,
        dataset_identifier="ml_training_dataset",
        policy=active_policy,
    )

    if cfg.save_artifacts:
        report.save(base_dir=cfg.artifacts_dir)

    if not report.is_allowed(active_policy):
        err_msg = (
            f"ML model training rejected by validation gate "
            f"({report.overall_status.value} under {active_policy.value} policy).\n"
            f"Details:\n{report.to_text_summary()}"
        )
        raise ValidationGateError(err_msg)

    return report
