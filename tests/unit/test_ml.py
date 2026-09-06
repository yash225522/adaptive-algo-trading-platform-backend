"""Unit tests for ML baseline modeling, splitting, evaluation, and artifacts."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaptive_trading.common.config import TimeFrame
from adaptive_trading.domain.market import Candle
from adaptive_trading.features.pipeline import FeaturePipeline
from adaptive_trading.ml.artifacts import ModelArtifactManager
from adaptive_trading.ml.cli import run_training_pipeline
from adaptive_trading.ml.config import MLConfig
from adaptive_trading.ml.datasets import (
    ChronologicalSplitter,
    DatasetSplitError,
)
from adaptive_trading.ml.evaluator import ModelEvaluator
from adaptive_trading.ml.models import LogisticRegressionModel
from adaptive_trading.ml.preprocessing import (
    MLPreprocessor,
    PreprocessingError,
)
from adaptive_trading.ml.trainer import MLTrainer
from adaptive_trading.targets.dataset import DatasetBuilder
from adaptive_trading.targets.definitions import TargetConfig
from adaptive_trading.targets.generator import TargetGenerator

UTC_TZ = timezone.utc


def make_synthetic_ml_dataset(n_samples: int = 50) -> pd.DataFrame:
    """Generate a clean synthetic tabular training dataset for testing."""
    start_time = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    records = []
    for i in range(n_samples):
        ts = start_time + timedelta(minutes=5 * i)
        row = {
            "timestamp": ts,
            "symbol": "NIFTY",
            "feature_1": float(np.sin(i / 5.0)),
            "feature_2": float(np.cos(i / 5.0)),
            "feature_3": float(i * 0.1),
            "future_return": 0.005 if i % 2 == 0 else -0.005,
            "label": "UP" if i % 2 == 0 else "DOWN",
            "feature_version": "v1",
            "target_version": "v1",
        }
        records.append(row)
    return pd.DataFrame.from_records(records)


def test_chronological_split() -> None:
    """Test strict chronological partitioning without shuffle."""
    df = make_synthetic_ml_dataset(n_samples=20)
    splitter = ChronologicalSplitter(train_ratio=0.80)
    split = splitter.split(df)

    assert split.train_sample_count == 16
    assert split.test_sample_count == 4
    assert split.train_end < split.test_start
    assert split.train_df["timestamp"].iloc[-1] < split.test_df["timestamp"].iloc[0]

    # Test error on empty df
    with pytest.raises(DatasetSplitError, match="empty DataFrame"):
        splitter.split(pd.DataFrame())

    # Test error on invalid train_ratio
    with pytest.raises(ValueError, match="train_ratio"):
        ChronologicalSplitter(train_ratio=1.5)


def test_preprocessing_feature_selection_and_target_exclusion() -> None:
    """Verify target columns and identifiers are strictly excluded from features."""
    df = make_synthetic_ml_dataset(n_samples=20)
    preprocessor = MLPreprocessor()
    X_df, y = preprocessor.extract_features_and_labels(df)

    # Assert target columns are excluded
    assert "future_return" not in X_df.columns
    assert "label" not in X_df.columns
    assert "timestamp" not in X_df.columns
    assert "symbol" not in X_df.columns
    assert "feature_version" not in X_df.columns
    assert "target_version" not in X_df.columns

    # Assert features are present
    assert set(X_df.columns) == {"feature_1", "feature_2", "feature_3"}
    assert len(y) == 20
    assert set(np.unique(y)).issubset({0, 1})

    # Test error on unexpected label
    bad_df = df.copy()
    bad_df.loc[0, "label"] = "INVALID"
    with pytest.raises(PreprocessingError, match="Unexpected label"):
        preprocessor.extract_features_and_labels(bad_df)


def test_scaler_isolation_no_leakage() -> None:
    """Verify StandardScaler is fitted strictly on train data."""
    df = make_synthetic_ml_dataset(n_samples=30)
    splitter = ChronologicalSplitter(train_ratio=0.80)
    split = splitter.split(df)

    preprocessor = MLPreprocessor()
    X_train, y_train = preprocessor.fit_transform(split.train_df)

    assert preprocessor.scaler is not None
    train_mean = np.copy(preprocessor.scaler.mean_)

    # Mutate test data dramatically
    mutated_test_df = split.test_df.copy()
    mutated_test_df["feature_1"] = mutated_test_df["feature_1"] * 1000.0

    X_test, y_test = preprocessor.transform(mutated_test_df)

    # Assert scaler parameters remained identical
    np.testing.assert_array_equal(preprocessor.scaler.mean_, train_mean)


def test_model_fit_predict_probabilities() -> None:
    """Test LogisticRegressionModel fitting, prediction, and probability shapes."""
    df = make_synthetic_ml_dataset(n_samples=40)
    preprocessor = MLPreprocessor()
    X, y = preprocessor.fit_transform(df)

    config = MLConfig(C=1.0, max_iter=500, random_state=42)
    model = LogisticRegressionModel(config=config)
    assert model.model_name == "logistic_regression"

    model.fit(X, y)
    preds = model.predict(X)
    probas = model.predict_proba(X)

    assert preds.shape == (40,)
    assert probas.shape == (40, 2)
    assert np.all(probas >= 0.0) and np.all(probas <= 1.0)
    # Sum of probabilities equals 1.0
    np.testing.assert_allclose(probas.sum(axis=1), np.ones(40), rtol=1e-5)


def test_model_evaluator_and_baseline() -> None:
    """Test evaluation report calculation and majority-class comparison."""
    evaluator = ModelEvaluator()
    y_true = np.array([1, 1, 1, 0, 0], dtype=int)
    y_pred = np.array([1, 1, 0, 0, 1], dtype=int)
    y_proba = np.array(
        [
            [0.2, 0.8],
            [0.3, 0.7],
            [0.6, 0.4],
            [0.9, 0.1],
            [0.4, 0.6],
        ],
        dtype=float,
    )

    report = evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        model_name="logistic_regression",
        y_train=np.array([1, 1, 1, 1, 0], dtype=int),  # Majority is 1 (UP)
    )

    assert report.sample_count == 5
    assert report.accuracy == 3 / 5  # 0.60
    assert report.roc_auc is not None
    assert report.baseline_metrics["majority_class"] == 1.0
    assert report.baseline_metrics["accuracy"] == 3 / 5  # 3 ones out of 5


def test_evaluator_handles_single_class_gracefully() -> None:
    """Test that single-class test set does not crash ROC-AUC calculation."""
    evaluator = ModelEvaluator()
    y_true = np.array([1, 1, 1, 1], dtype=int)
    y_pred = np.array([1, 1, 1, 1], dtype=int)
    y_proba = np.array([[0.1, 0.9]] * 4, dtype=float)

    report = evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )
    assert report.roc_auc is None  # Gracefully None


def test_artifact_save_load_roundtrip(tmp_path: Path) -> None:
    """Test saving and loading model artifact roundtrip."""
    df = make_synthetic_ml_dataset(n_samples=30)
    trainer = MLTrainer(artifact_manager=ModelArtifactManager(base_dir=tmp_path))

    model, report, artifact_dir = trainer.train_and_evaluate(
        dataset_df=df,
        save_artifact=True,
        artifact_version="test_v1",
    )
    assert artifact_dir is not None
    assert (artifact_dir / "model.joblib").exists()
    assert (artifact_dir / "metadata.json").exists()

    # Load back
    loaded_model, loaded_prep, metadata = trainer.artifact_manager.load_artifact(
        model_name="logistic_regression",
        artifact_version="test_v1",
    )
    assert metadata["model_name"] == "logistic_regression"
    assert metadata["artifact_version"] == "test_v1"

    # Verify prediction consistency
    X_test, _ = loaded_prep.transform(df.iloc[-5:])
    preds_orig = model.predict(X_test)
    preds_loaded = loaded_model.predict(X_test)
    np.testing.assert_array_equal(preds_orig, preds_loaded)


def test_end_to_end_ml_trainer_on_candles(tmp_path: Path) -> None:
    """Test end-to-end flow from Candle domain models to trained ML model."""
    start_time = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    candles = []
    for i in range(40):
        ts = start_time + timedelta(minutes=5 * i)
        price = 100.0 + (i % 5) * 2.0
        c = Candle(
            timestamp=ts,
            symbol="NIFTY",
            timeframe=TimeFrame.FIVE_MINUTES,
            open=price,
            high=price + 1.0,
            low=price - 1.0,
            close=price,
            volume=1000.0,
        )
        candles.append(c)

    # 1. Features
    feature_vectors = FeaturePipeline().generate_feature_vectors(
        candles, drop_warmup=True
    )
    # 2. Targets
    targets = TargetGenerator(config=TargetConfig(horizon=3)).generate_targets(candles)
    # 3. Dataset
    dataset_df = DatasetBuilder().build_training_dataframe(feature_vectors, targets)

    # 4. Train
    trainer = MLTrainer(artifact_manager=ModelArtifactManager(base_dir=tmp_path))
    model, report, _ = trainer.train_and_evaluate(dataset_df, save_artifact=True)

    assert report.sample_count > 0
    assert 0.0 <= report.accuracy <= 1.0


def test_cli_execution(tmp_path: Path) -> None:
    """Test CLI runner on the actual sample CSV file."""
    csv_file = Path("data/sample/nifty_5m_ml_sample.csv")
    exit_code = run_training_pipeline(
        csv_file=csv_file,
        train_ratio=0.80,
        horizon=3,
        save_artifact=False,
    )
    assert exit_code == 0
