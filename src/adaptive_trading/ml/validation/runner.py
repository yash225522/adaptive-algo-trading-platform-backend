"""Walk-forward cross-validation orchestrator and experiment runner."""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from adaptive_trading.ml.config import MLConfig
from adaptive_trading.ml.evaluator import ModelEvaluator
from adaptive_trading.ml.models import LogisticRegressionModel
from adaptive_trading.ml.preprocessing import MLPreprocessor
from adaptive_trading.ml.validation.config import WalkForwardConfig
from adaptive_trading.ml.validation.metrics import compute_aggregate_metrics
from adaptive_trading.ml.validation.results import (
    OOSPrediction,
    WalkForwardFoldResult,
    WalkForwardResult,
)
from adaptive_trading.ml.validation.splitter import WalkForwardSplitter

logger = logging.getLogger(__name__)


class WalkForwardRunner:
    """Orchestrates walk-forward training, evaluation, and artifact generation."""

    def __init__(
        self,
        config: WalkForwardConfig | None = None,
        ml_config: MLConfig | None = None,
        splitter: WalkForwardSplitter | None = None,
        evaluator: ModelEvaluator | None = None,
    ) -> None:
        self.config = config or WalkForwardConfig()
        self.ml_config = ml_config or MLConfig()
        self.splitter = splitter or WalkForwardSplitter(config=self.config)
        self.evaluator = evaluator or ModelEvaluator()

    def run(
        self,
        dataset_df: pd.DataFrame,
        experiment_id: str | None = None,
        save_artifacts: bool = True,
        artifacts_base_dir: Path | str = "artifacts/experiments",
    ) -> WalkForwardResult:
        """Execute full walk-forward validation across chronological folds.

        Args:
            dataset_df: Input aligned dataset with features, labels, and timestamps.
            experiment_id: Optional unique identifier for the experiment run.
            save_artifacts: Whether to persist experiment metrics & OOS predictions.
            artifacts_base_dir: Output root directory for experiment artifacts.

        Returns:
            WalkForwardResult: Validation results across all folds.
        """
        if dataset_df.empty:
            raise ValueError("Cannot run walk-forward validation on empty dataset")

        # Sort chronologically if timestamp is present
        if "timestamp" in dataset_df.columns:
            df = dataset_df.sort_values("timestamp").reset_index(drop=True)
        else:
            df = dataset_df.reset_index(drop=True)

        feature_version = (
            str(df["feature_version"].iloc[0])
            if "feature_version" in df.columns
            else "unknown"
        )
        target_version = (
            str(df["target_version"].iloc[0])
            if "target_version" in df.columns
            else "unknown"
        )

        splits = self.splitter.split(df)
        fold_results: list[WalkForwardFoldResult] = []
        oos_predictions: list[OOSPrediction] = []

        logger.info(
            "Starting walk-forward validation across %d folds (%d total observations)",
            len(splits),
            len(df),
        )

        for split in splits:
            logger.info(
                "Processing fold %d/%d: train=%d samples, val=%d samples",
                split.fold_index,
                len(splits),
                split.train_samples,
                split.validation_samples,
            )

            train_df = df.iloc[split.train_indices]
            val_df = df.iloc[split.validation_indices]

            # 1. Independent preprocessing fitted strictly on training fold
            preprocessor = MLPreprocessor(
                positive_class=self.ml_config.positive_class,
                negative_class=self.ml_config.negative_class,
            )
            X_train_scaled, y_train = preprocessor.fit_transform(train_df)

            # 2. Fit model on training fold
            model = LogisticRegressionModel(config=self.ml_config)
            model.fit(X_train_scaled, y_train)

            # 3. Transform validation fold using train-fitted scaler
            X_val_scaled, y_val = preprocessor.transform(val_df)

            # 4. Predict validation fold
            y_pred = model.predict(X_val_scaled)
            y_proba = model.predict_proba(X_val_scaled)

            # 5. Evaluate fold metrics (majority baseline computed from y_train)
            report = self.evaluator.evaluate(
                y_true=y_val,
                y_pred=y_pred,
                y_proba=y_proba,
                model_name=model.model_name,
                y_train=y_train,
            )

            fold_results.append(
                WalkForwardFoldResult(
                    fold_number=split.fold_index,
                    train_start=split.train_start,
                    train_end=split.train_end,
                    validation_start=split.validation_start,
                    validation_end=split.validation_end,
                    train_samples=split.train_samples,
                    validation_samples=split.validation_samples,
                    class_distribution=report.class_distribution,
                    accuracy=report.accuracy,
                    precision=report.precision,
                    recall=report.recall,
                    f1=report.f1,
                    roc_auc=report.roc_auc,
                    confusion_matrix=report.confusion_matrix,
                    baseline_metrics=report.baseline_metrics,
                )
            )

            # 6. Collect out-of-sample predictions
            for i, val_idx in enumerate(split.validation_indices):
                row = df.iloc[val_idx]
                ts = row["timestamp"] if "timestamp" in row else None
                sym = str(row["symbol"]) if "symbol" in row else ""
                prob_up = (
                    float(y_proba[i, 1])
                    if y_proba is not None and y_proba.shape[1] > 1
                    else float(y_proba[i, 0])
                    if y_proba is not None
                    else 0.0
                )

                oos_predictions.append(
                    OOSPrediction(
                        timestamp=ts,
                        symbol=sym,
                        actual_label=int(y_val[i]),
                        predicted_label=int(y_pred[i]),
                        probability_up=prob_up,
                        fold_number=split.fold_index,
                    )
                )

        aggregate_metrics = compute_aggregate_metrics(fold_results)

        exp_id = experiment_id or (
            f"wf_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_"
            f"{uuid.uuid4().hex[:6]}"
        )

        result = WalkForwardResult(
            experiment_id=exp_id,
            model_name="logistic_regression",
            feature_version=feature_version,
            target_version=target_version,
            config=self.config,
            fold_results=fold_results,
            aggregate_metrics=aggregate_metrics,
            oos_predictions=oos_predictions,
            created_at=datetime.now(timezone.utc),
        )

        if save_artifacts:
            self._save_experiment_artifacts(result, artifacts_base_dir)

        return result

    def _save_experiment_artifacts(
        self,
        result: WalkForwardResult,
        artifacts_base_dir: Path | str,
    ) -> Path:
        """Save experiment metrics, predictions, and metadata to disk."""
        exp_dir = Path(artifacts_base_dir) / result.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save metadata.json
        metadata = {
            "experiment_id": result.experiment_id,
            "model_name": result.model_name,
            "feature_version": result.feature_version,
            "target_version": result.target_version,
            "created_at": result.created_at.isoformat(),
            "config": result.config.model_dump(),
            "total_folds": result.aggregate_metrics.total_folds,
            "total_validation_samples": (
                result.aggregate_metrics.total_validation_samples
            ),
        }
        with open(exp_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 2. Save metrics.json
        metrics_data = {
            "aggregate_metrics": result.aggregate_metrics.model_dump(),
            "fold_results": [f.model_dump() for f in result.fold_results],
        }
        with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2, default=str)

        # 3. Save predictions.csv
        preds_df = pd.DataFrame([p.model_dump() for p in result.oos_predictions])
        preds_df.to_csv(exp_dir / "predictions.csv", index=False)

        logger.info("Saved walk-forward experiment artifacts to %s", exp_dir)
        return exp_dir
