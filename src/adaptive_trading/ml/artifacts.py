"""Model serialization, artifact management, and metadata storage."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from adaptive_trading.ml.evaluator import EvaluationReport
from adaptive_trading.ml.models import BaseMLModel
from adaptive_trading.ml.preprocessing import MLPreprocessor

logger = logging.getLogger(__name__)


class ArtifactError(Exception):
    """Raised when saving or loading model artifacts fails."""


class ModelArtifactManager:
    """Manages saving and loading of ML model binaries and accompanying metadata."""

    def __init__(self, base_dir: Path | str = "artifacts/models") -> None:
        self.base_dir = Path(base_dir)

    def save_artifact(
        self,
        model: BaseMLModel,
        preprocessor: MLPreprocessor,
        report: EvaluationReport,
        feature_version: str,
        target_version: str,
        split_metadata: dict[str, Any],
        artifact_version: str = "v1",
    ) -> Path:
        """Save model bundle and metadata to disk.

        Args:
            model: Fitted BaseMLModel instance.
            preprocessor: Fitted MLPreprocessor instance.
            report: EvaluationReport instance.
            feature_version: Feature set version identifier.
            target_version: Target definition version identifier.
            split_metadata: Dictionary with split timestamps and sample counts.
            artifact_version: Artifact subfolder version name.

        Returns:
            Path: Destination directory containing saved model and metadata.
        """
        dest_dir = self.base_dir / model.model_name / artifact_version
        dest_dir.mkdir(parents=True, exist_ok=True)

        bundle = {
            "model": model,
            "preprocessor": preprocessor,
            "feature_names": preprocessor.feature_names,
        }

        # 1. Save binary model bundle
        model_path = dest_dir / "model.joblib"
        try:
            joblib.dump(bundle, model_path)
            logger.info("Saved model artifact to %s", model_path)
        except Exception as e:
            raise ArtifactError(f"Failed to serialize model bundle: {e}") from e

        # 2. Save structured JSON metadata
        metadata = {
            "model_name": model.model_name,
            "artifact_version": artifact_version,
            "feature_version": feature_version,
            "target_version": target_version,
            "feature_names": preprocessor.feature_names,
            "training_timestamp": datetime.now(timezone.utc).isoformat(),
            "split": split_metadata,
            "metrics": report.model_dump(),
        }

        metadata_path = dest_dir / "metadata.json"
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.info("Saved model metadata to %s", metadata_path)
        except Exception as e:
            raise ArtifactError(f"Failed to write metadata JSON: {e}") from e

        return dest_dir

    def load_artifact(
        self,
        model_name: str = "logistic_regression",
        artifact_version: str = "v1",
    ) -> tuple[BaseMLModel, MLPreprocessor, dict[str, Any]]:
        """Load model bundle and metadata from disk.

        Args:
            model_name: Name of model architecture.
            artifact_version: Artifact version subfolder.

        Returns:
            tuple: (model, preprocessor, metadata)
        """
        artifact_dir = self.base_dir / model_name / artifact_version
        model_path = artifact_dir / "model.joblib"
        metadata_path = artifact_dir / "metadata.json"

        if not model_path.exists():
            raise ArtifactError(f"Model binary not found at {model_path}")
        if not metadata_path.exists():
            raise ArtifactError(f"Metadata file not found at {metadata_path}")

        try:
            bundle = joblib.load(model_path)
            model = bundle["model"]
            preprocessor = bundle["preprocessor"]
        except Exception as e:
            raise ArtifactError(f"Failed to load model bundle: {e}") from e

        try:
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            raise ArtifactError(f"Failed to load metadata JSON: {e}") from e

        return model, preprocessor, metadata
