"""Model evaluation metrics and baseline comparison."""

import logging

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class EvaluationReport(BaseModel):
    """Structured container for test set evaluation and baseline comparison."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str
    sample_count: int
    class_distribution: dict[str, int]
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    confusion_matrix: list[list[int]]
    baseline_metrics: dict[str, float]


class ModelEvaluator:
    """Computes classification metrics and compares against majority baseline."""

    def evaluate(
        self,
        y_true: npt.NDArray[np.int_],
        y_pred: npt.NDArray[np.int_],
        y_proba: npt.NDArray[np.float64] | None = None,
        model_name: str = "logistic_regression",
        y_train: npt.NDArray[np.int_] | None = None,
    ) -> EvaluationReport:
        """Calculate classification metrics on unseen test set.

        Args:
            y_true: Ground truth binary labels (0 or 1).
            y_pred: Predicted binary labels.
            y_proba: Predicted probability distribution array.
            model_name: Model identifier string.
            y_train: Optional training labels to determine majority class.

        Returns:
            EvaluationReport: Performance report.
        """
        n_samples = len(y_true)
        if n_samples == 0:
            raise ValueError("Cannot evaluate on empty test set")

        # Class counts
        n_up = int(np.sum(y_true == 1))
        n_down = int(np.sum(y_true == 0))
        class_dist = {"UP (1)": n_up, "DOWN (0)": n_down}

        # Core classification metrics
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0.0))
        rec = float(recall_score(y_true, y_pred, zero_division=0.0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0.0))

        # ROC-AUC calculation (handles single-class test set gracefully)
        roc_auc: float | None = None
        unique_classes = np.unique(y_true)
        if len(unique_classes) > 1 and y_proba is not None and y_proba.shape[1] > 1:
            try:
                roc_auc = float(roc_auc_score(y_true, y_proba[:, 1]))
            except ValueError as e:
                logger.warning("ROC-AUC calculation failed: %s", e)
                roc_auc = None

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

        # Majority class baseline
        # Use training set majority class if available, else test set majority class
        ref_y = y_train if y_train is not None and len(y_train) > 0 else y_true
        majority_label = int(np.bincount(ref_y).argmax())
        majority_preds = np.full_like(y_true, fill_value=majority_label)

        baseline_metrics: dict[str, float] = {
            "majority_class": float(majority_label),
            "accuracy": float(accuracy_score(y_true, majority_preds)),
            "precision": float(
                precision_score(y_true, majority_preds, zero_division=0.0)
            ),
            "recall": float(recall_score(y_true, majority_preds, zero_division=0.0)),
            "f1": float(f1_score(y_true, majority_preds, zero_division=0.0)),
        }

        return EvaluationReport(
            model_name=model_name,
            sample_count=n_samples,
            class_distribution=class_dist,
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            roc_auc=roc_auc,
            confusion_matrix=cm,
            baseline_metrics=baseline_metrics,
        )
