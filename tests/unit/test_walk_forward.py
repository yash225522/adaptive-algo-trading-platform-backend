"""Unit tests for walk-forward validation splitter, runner, and metrics."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaptive_trading.ml.config import MLConfig
from adaptive_trading.ml.validation.config import (
    WalkForwardConfig,
    WindowType,
)
from adaptive_trading.ml.validation.metrics import compute_aggregate_metrics
from adaptive_trading.ml.validation.results import WalkForwardFoldResult
from adaptive_trading.ml.validation.runner import WalkForwardRunner
from adaptive_trading.ml.validation.splitter import WalkForwardSplitter

UTC_TZ = timezone.utc


@pytest.fixture
def sample_ml_dataframe() -> pd.DataFrame:
    """Create a deterministic synthetic dataset with chronological timestamps."""
    n_samples = 100
    base_time = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)

    timestamps = [base_time + timedelta(minutes=5 * i) for i in range(n_samples)]
    np.random.seed(42)

    # Synthetic features
    f1 = np.sin(np.linspace(0, 10, n_samples)) + np.random.normal(0, 0.1, n_samples)
    f2 = np.cos(np.linspace(0, 10, n_samples)) + np.random.normal(0, 0.1, n_samples)
    f3 = np.random.normal(0, 1, n_samples)

    # Deterministic binary labels (UP if f1 + f2 > 0 else DOWN)
    labels = np.where(f1 + f2 > 0, "UP", "DOWN")

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "NIFTY",
            "feature_version": "v1",
            "target_version": "v1",
            "ret_1": f1,
            "ret_5": f2,
            "volatility": f3,
            "future_return": f1 * 0.01,
            "label": labels,
        }
    )


def test_walk_forward_config_validation() -> None:
    """Test WalkForwardConfig bounds and parameter validations."""
    config = WalkForwardConfig(
        initial_train_size=0.50,
        validation_size=0.10,
        step_size=0.10,
        gap=2,
        window_type=WindowType.EXPANDING,
    )
    assert config.initial_train_size == 0.50
    assert config.gap == 2

    # Incompatible fraction bounds rejected
    with pytest.raises(ValueError, match="cannot exceed 1.0"):
        WalkForwardConfig(
            initial_train_size=0.80,
            validation_size=0.30,
        )


def test_splitter_chronological_expanding(
    sample_ml_dataframe: pd.DataFrame,
) -> None:
    """Test expanding window walk-forward splitting."""
    config = WalkForwardConfig(
        initial_train_size=50,
        validation_size=10,
        step_size=10,
        window_type=WindowType.EXPANDING,
        min_train_samples=20,
    )
    splitter = WalkForwardSplitter(config=config)
    splits = splitter.split(sample_ml_dataframe)

    assert len(splits) == 5  # (100 - 50) / 10 = 5 folds

    for i, split in enumerate(splits, start=1):
        assert split.fold_index == i
        # Expanding window: train_indices always start at 0
        assert split.train_indices[0] == 0
        assert split.train_samples == 50 + (i - 1) * 10
        assert split.validation_samples == 10

        # Temporal integrity: max train timestamp < min validation timestamp
        assert (
            split.train_end is not None
            and split.validation_start is not None
            and split.train_end < split.validation_start
        )
        assert max(split.train_indices) < min(split.validation_indices)


def test_splitter_rolling_window(sample_ml_dataframe: pd.DataFrame) -> None:
    """Test rolling window walk-forward splitting."""
    config = WalkForwardConfig(
        initial_train_size=40,
        validation_size=10,
        step_size=10,
        window_type=WindowType.ROLLING,
        min_train_samples=20,
    )
    splitter = WalkForwardSplitter(config=config)
    splits = splitter.split(sample_ml_dataframe)

    assert len(splits) == 6

    for split in splits:
        # Rolling window: fixed train sample count
        assert split.train_samples == 40
        assert split.validation_samples == 10
        assert (
            split.train_end is not None
            and split.validation_start is not None
            and split.train_end < split.validation_start
        )


def test_splitter_with_gap(sample_ml_dataframe: pd.DataFrame) -> None:
    """Test temporal purge gap between train and validation."""
    gap_bars = 3
    config = WalkForwardConfig(
        initial_train_size=50,
        validation_size=10,
        step_size=10,
        gap=gap_bars,
        min_train_samples=20,
    )
    splitter = WalkForwardSplitter(config=config)
    splits = splitter.split(sample_ml_dataframe)

    for split in splits:
        # Gap index difference: val_start - train_end == gap + 1
        assert min(split.validation_indices) - max(split.train_indices) == gap_bars + 1


def test_splitter_raises_on_small_dataset() -> None:
    """Test splitter errors on insufficient dataset size."""
    small_df = pd.DataFrame({"timestamp": [1, 2, 3], "label": ["UP", "DOWN", "UP"]})
    config = WalkForwardConfig(min_train_samples=20)
    splitter = WalkForwardSplitter(config=config)

    with pytest.raises(ValueError, match="smaller than minimum required"):
        splitter.split(small_df)


def test_leakage_future_data_isolation(
    sample_ml_dataframe: pd.DataFrame,
) -> None:
    """Verify modifying future records in fold 2 does NOT change fold 1 results."""
    config = WalkForwardConfig(
        initial_train_size=50,
        validation_size=10,
        step_size=10,
        min_train_samples=20,
    )
    runner = WalkForwardRunner(config=config)

    # 1. Run baseline on original data
    result_orig = runner.run(dataset_df=sample_ml_dataframe, save_artifacts=False)
    fold_1_orig = result_orig.fold_results[0]

    # 2. Corrupt future records (index 60 onwards)
    modified_df = sample_ml_dataframe.copy()
    modified_df.loc[60:, "ret_1"] = 9999.9
    modified_df.loc[60:, "volatility"] = -9999.9

    result_mod = runner.run(dataset_df=modified_df, save_artifacts=False)
    fold_1_mod = result_mod.fold_results[0]

    # Fold 1 metrics must be 100% bit-exact and identical
    assert fold_1_orig.accuracy == fold_1_mod.accuracy
    assert fold_1_orig.f1 == fold_1_mod.f1
    assert fold_1_orig.confusion_matrix == fold_1_mod.confusion_matrix


def test_walk_forward_runner_end_to_end(
    sample_ml_dataframe: pd.DataFrame, tmp_path: Path
) -> None:
    """Test full walk-forward execution, metric calculation, and artifact export."""
    config = WalkForwardConfig(
        initial_train_size=0.50,
        validation_size=0.10,
        step_size=0.10,
        min_train_samples=20,
    )
    runner = WalkForwardRunner(config=config)
    result = runner.run(
        dataset_df=sample_ml_dataframe,
        save_artifacts=True,
        artifacts_base_dir=tmp_path,
    )

    assert result.aggregate_metrics.total_folds == 5
    assert result.aggregate_metrics.total_validation_samples == 50
    assert 0.0 <= result.aggregate_metrics.mean_accuracy <= 1.0
    assert 0.0 <= result.aggregate_metrics.mean_f1 <= 1.0

    # Verify Out-of-Sample Predictions
    assert len(result.oos_predictions) == 50
    for pred in result.oos_predictions:
        assert pred.symbol == "NIFTY"
        assert pred.actual_label in (0, 1)
        assert pred.predicted_label in (0, 1)
        assert 0.0 <= pred.probability_up <= 1.0
        assert 1 <= pred.fold_number <= 5

    # Verify Artifact files written to disk
    exp_dir = tmp_path / result.experiment_id
    assert exp_dir.exists()
    assert (exp_dir / "metadata.json").is_file()
    assert (exp_dir / "metrics.json").is_file()
    assert (exp_dir / "predictions.csv").is_file()


def test_majority_baseline_per_fold(
    sample_ml_dataframe: pd.DataFrame,
) -> None:
    """Test majority baseline accuracy is evaluated for each fold."""
    config = WalkForwardConfig(
        initial_train_size=50,
        validation_size=10,
        step_size=10,
        min_train_samples=20,
    )
    runner = WalkForwardRunner(config=config)
    result = runner.run(dataset_df=sample_ml_dataframe, save_artifacts=False)

    for fold in result.fold_results:
        assert "accuracy" in fold.baseline_metrics
        assert "f1" in fold.baseline_metrics
        assert 0.0 <= fold.baseline_metrics["accuracy"] <= 1.0


def test_reproducibility(sample_ml_dataframe: pd.DataFrame) -> None:
    """Test identical configuration and dataset produce identical results."""
    config = WalkForwardConfig(
        initial_train_size=50,
        validation_size=10,
        step_size=10,
        min_train_samples=20,
    )
    ml_config = MLConfig(random_state=42)

    runner1 = WalkForwardRunner(config=config, ml_config=ml_config)
    result1 = runner1.run(dataset_df=sample_ml_dataframe, save_artifacts=False)

    runner2 = WalkForwardRunner(config=config, ml_config=ml_config)
    result2 = runner2.run(dataset_df=sample_ml_dataframe, save_artifacts=False)

    assert (
        result1.aggregate_metrics.mean_accuracy
        == result2.aggregate_metrics.mean_accuracy
    )
    assert result1.aggregate_metrics.mean_f1 == result2.aggregate_metrics.mean_f1


def test_compute_aggregate_metrics_calculation() -> None:
    """Test statistical aggregation logic."""
    fold1 = WalkForwardFoldResult(
        fold_number=1,
        train_samples=50,
        validation_samples=10,
        class_distribution={"UP": 6, "DOWN": 4},
        accuracy=0.60,
        precision=0.60,
        recall=0.70,
        f1=0.65,
        roc_auc=0.65,
        confusion_matrix=[[2, 2], [2, 4]],
        baseline_metrics={"accuracy": 0.50},
    )
    fold2 = WalkForwardFoldResult(
        fold_number=2,
        train_samples=60,
        validation_samples=10,
        class_distribution={"UP": 5, "DOWN": 5},
        accuracy=0.80,
        precision=0.80,
        recall=0.80,
        f1=0.80,
        roc_auc=0.85,
        confusion_matrix=[[4, 1], [1, 4]],
        baseline_metrics={"accuracy": 0.50},
    )

    agg = compute_aggregate_metrics([fold1, fold2])
    assert agg.total_folds == 2
    assert agg.total_validation_samples == 20
    assert agg.mean_accuracy == pytest.approx(0.70)
    assert agg.mean_f1 == pytest.approx(0.725)
    assert agg.mean_roc_auc == pytest.approx(0.75)
    assert agg.mean_baseline_accuracy == pytest.approx(0.50)
