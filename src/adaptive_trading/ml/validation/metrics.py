"""Metric calculation and aggregation across walk-forward validation folds."""

import numpy as np

from adaptive_trading.ml.validation.results import (
    AggregateMetrics,
    WalkForwardFoldResult,
)


def compute_aggregate_metrics(
    fold_results: list[WalkForwardFoldResult],
) -> AggregateMetrics:
    """Calculate summary statistics and averages across validation folds.

    Args:
        fold_results: List of evaluated WalkForwardFoldResult objects.

    Returns:
        AggregateMetrics: Aggregated statistical summary.
    """
    if not fold_results:
        raise ValueError("Cannot calculate aggregate metrics for empty fold list")

    total_folds = len(fold_results)
    total_val_samples = sum(f.validation_samples for f in fold_results)

    accuracies = [f.accuracy for f in fold_results]
    precisions = [f.precision for f in fold_results]
    recalls = [f.recall for f in fold_results]
    f1s = [f.f1 for f in fold_results]
    roc_aucs = [f.roc_auc for f in fold_results if f.roc_auc is not None]
    baselines = [f.baseline_metrics.get("accuracy", 0.0) for f in fold_results]

    mean_roc = float(np.mean(roc_aucs)) if roc_aucs else None

    return AggregateMetrics(
        total_folds=total_folds,
        total_validation_samples=total_val_samples,
        mean_accuracy=float(np.mean(accuracies)),
        std_accuracy=float(np.std(accuracies)),
        mean_precision=float(np.mean(precisions)),
        std_precision=float(np.std(precisions)),
        mean_recall=float(np.mean(recalls)),
        std_recall=float(np.std(recalls)),
        mean_f1=float(np.mean(f1s)),
        std_f1=float(np.std(f1s)),
        mean_roc_auc=mean_roc,
        mean_baseline_accuracy=float(np.mean(baselines)),
    )
