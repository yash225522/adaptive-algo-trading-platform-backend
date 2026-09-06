"""End-to-end ML training and evaluation orchestrator."""

import logging
from pathlib import Path

import pandas as pd

from adaptive_trading.ml.artifacts import ModelArtifactManager
from adaptive_trading.ml.config import MLConfig
from adaptive_trading.ml.datasets import ChronologicalSplitter
from adaptive_trading.ml.evaluator import EvaluationReport, ModelEvaluator
from adaptive_trading.ml.models import BaseMLModel, LogisticRegressionModel
from adaptive_trading.ml.preprocessing import MLPreprocessor

logger = logging.getLogger(__name__)


class MLTrainer:
    """Orchestrates chronological splitting, training, evaluation, and artifacts."""

    def __init__(
        self,
        config: MLConfig | None = None,
        artifact_manager: ModelArtifactManager | None = None,
        evaluator: ModelEvaluator | None = None,
    ) -> None:
        self.config = config or MLConfig()
        self.artifact_manager = artifact_manager or ModelArtifactManager()
        self.evaluator = evaluator or ModelEvaluator()

    def train_and_evaluate(
        self,
        dataset_df: pd.DataFrame,
        save_artifact: bool = True,
        artifact_version: str = "v1",
    ) -> tuple[BaseMLModel, EvaluationReport, Path | None]:
        """Execute the full training pipeline.

        Args:
            dataset_df: Aligned tabular training DataFrame.
            save_artifact: Whether to persist model and metadata to disk.
            artifact_version: Subfolder version for the artifact.

        Returns:
            tuple: (model, report, artifact_dir)
        """
        if dataset_df.empty:
            raise ValueError("Training dataset DataFrame is empty")

        # Extract version identifiers
        feature_version = (
            str(dataset_df["feature_version"].iloc[0])
            if "feature_version" in dataset_df.columns
            else "unknown"
        )
        target_version = (
            str(dataset_df["target_version"].iloc[0])
            if "target_version" in dataset_df.columns
            else "unknown"
        )

        # 1. Chronological Train/Test Split
        splitter = ChronologicalSplitter(train_ratio=self.config.train_ratio)
        split = splitter.split(dataset_df)

        # 2. Training-only Preprocessing & Scaling
        preprocessor = MLPreprocessor(
            positive_class=self.config.positive_class,
            negative_class=self.config.negative_class,
        )
        X_train, y_train = preprocessor.fit_transform(split.train_df)
        X_test, y_test = preprocessor.transform(split.test_df)

        # 3. Model Training
        model: BaseMLModel = LogisticRegressionModel(config=self.config)
        model.fit(X_train, y_train)

        # 4. Evaluation on unseen chronological test partition
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        report = self.evaluator.evaluate(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            model_name=model.model_name,
            y_train=y_train,
        )

        logger.info(
            "Evaluation complete for %s (Accuracy: %.4f, F1: %.4f, ROC-AUC: %s)",
            model.model_name,
            report.accuracy,
            report.f1,
            f"{report.roc_auc:.4f}" if report.roc_auc is not None else "N/A",
        )

        # 5. Persist artifact if requested
        artifact_dir: Path | None = None
        if save_artifact:
            split_metadata = {
                "train_ratio": self.config.train_ratio,
                "train_samples": split.train_sample_count,
                "test_samples": split.test_sample_count,
                "train_start": split.train_start.isoformat(),
                "train_end": split.train_end.isoformat(),
                "test_start": split.test_start.isoformat(),
                "test_end": split.test_end.isoformat(),
            }
            artifact_dir = self.artifact_manager.save_artifact(
                model=model,
                preprocessor=preprocessor,
                report=report,
                feature_version=feature_version,
                target_version=target_version,
                split_metadata=split_metadata,
                artifact_version=artifact_version,
            )

        return model, report, artifact_dir
