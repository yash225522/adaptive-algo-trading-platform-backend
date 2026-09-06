"""Command line interface for training and evaluating baseline ML models."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from adaptive_trading.data.readers.csv_reader import CSVMarketDataReader
from adaptive_trading.data.validators.market_data import MarketDataValidator
from adaptive_trading.features.pipeline import FeaturePipeline
from adaptive_trading.ml.config import MLConfig
from adaptive_trading.ml.trainer import MLTrainer
from adaptive_trading.ml.validation.config import WalkForwardConfig, WindowType
from adaptive_trading.ml.validation.runner import WalkForwardRunner
from adaptive_trading.targets.dataset import DatasetBuilder
from adaptive_trading.targets.definitions import TargetConfig
from adaptive_trading.targets.generator import TargetGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_ml_dataset_from_csv(
    csv_path: Path,
    horizon: int = 3,
) -> pd.DataFrame:
    """Load, validate, compute features, and generate target labels from CSV."""
    reader = CSVMarketDataReader()
    raw_records = reader.read_records(csv_path)
    validator = MarketDataValidator()
    valid_candles, _ = validator.validate_batch(raw_records)

    if not valid_candles:
        raise ValueError(f"No valid candles extracted from CSV: {csv_path}")

    # 1. Compute Features
    feature_pipeline = FeaturePipeline()
    feature_vectors = feature_pipeline.generate_feature_vectors(
        valid_candles, drop_warmup=True
    )

    # 2. Generate Targets
    target_config = TargetConfig(horizon=horizon)
    target_generator = TargetGenerator(config=target_config)
    target_series = target_generator.generate_targets(valid_candles)

    # 3. Align Dataset
    dataset_builder = DatasetBuilder()
    return dataset_builder.build_training_dataframe(
        feature_vectors=feature_vectors,
        targets=target_series,
    )


def run_training_pipeline(
    csv_file: Path | str,
    train_ratio: float = 0.80,
    horizon: int = 3,
    save_artifact: bool = True,
    artifact_version: str = "v1",
) -> int:
    """Run full baseline ML pipeline on a market data CSV file."""
    csv_path = Path(csv_file)
    if not csv_path.exists():
        logger.error("Input CSV file not found: %s", csv_path)
        return 1

    print("=" * 60)
    print("      ADAPTIVE TRADING PLATFORM - ML BASELINE TRAINING      ")
    print("=" * 60)
    print(f"Dataset File     : {csv_path}")
    print(f"Prediction Horizon: {horizon} bars")
    print(f"Train/Test Ratio : {train_ratio:.0%} / {1 - train_ratio:.0%}")
    print("-" * 60)

    try:
        dataset_df = build_ml_dataset_from_csv(csv_path, horizon=horizon)
    except Exception as exc:
        logger.error("Failed to build ML dataset: %s", exc)
        return 1

    print(f"Aligned Samples  : {len(dataset_df)}")

    # Train and Evaluate
    config = MLConfig(train_ratio=train_ratio)
    trainer = MLTrainer(config=config)
    _, report, artifact_dir = trainer.train_and_evaluate(
        dataset_df=dataset_df,
        save_artifact=save_artifact,
        artifact_version=artifact_version,
    )

    print("-" * 60)
    print(f"Model            : {report.model_name}")
    print(f"Test Samples     : {report.sample_count}")
    print(f"Class Dist       : {report.class_distribution}")
    print(f"Accuracy         : {report.accuracy:.4f}")
    print(f"Precision        : {report.precision:.4f}")
    print(f"Recall           : {report.recall:.4f}")
    print(f"F1 Score         : {report.f1:.4f}")
    roc_str = f"{report.roc_auc:.4f}" if report.roc_auc is not None else "N/A"
    print(f"ROC-AUC          : {roc_str}")
    print(f"Confusion Matrix : {report.confusion_matrix}")
    print("-" * 60)
    print(f"Baseline Acc     : {report.baseline_metrics.get('accuracy', 0.0):.4f}")
    print(f"Baseline F1      : {report.baseline_metrics.get('f1', 0.0):.4f}")
    print("-" * 60)
    if artifact_dir:
        print(f"Artifact Saved   : {artifact_dir}")
    print("=" * 60)
    return 0


def run_walk_forward_pipeline(
    csv_file: Path | str,
    initial_train_size: float = 0.50,
    validation_size: float = 0.10,
    step_size: float | None = None,
    gap: int = 0,
    window_type_str: str = "EXPANDING",
    horizon: int = 3,
    save_artifacts: bool = True,
) -> int:
    """Run full walk-forward validation on a market data CSV file."""
    csv_path = Path(csv_file)
    if not csv_path.exists():
        logger.error("Input CSV file not found: %s", csv_path)
        return 1

    print("=" * 60)
    print("      ADAPTIVE TRADING PLATFORM - WALK-FORWARD VALIDATION   ")
    print("=" * 60)
    print(f"Dataset File     : {csv_path}")
    print(f"Prediction Horizon: {horizon} bars")
    print(f"Window Type      : {window_type_str}")
    print(f"Initial Train    : {initial_train_size}")
    print(f"Validation Size  : {validation_size}")
    print(f"Gap (Purge)      : {gap} bars")
    print("-" * 60)

    try:
        dataset_df = build_ml_dataset_from_csv(csv_path, horizon=horizon)
    except Exception as exc:
        logger.error("Failed to build ML dataset: %s", exc)
        return 1

    print(f"Aligned Samples  : {len(dataset_df)}")

    # Configure Walk-Forward Runner
    try:
        min_train = min(10, max(5, int(len(dataset_df) * initial_train_size)))
        wf_config = WalkForwardConfig(
            initial_train_size=initial_train_size,
            validation_size=validation_size,
            step_size=step_size,
            gap=gap,
            window_type=WindowType(window_type_str),
            min_train_samples=min_train,
        )
        runner = WalkForwardRunner(config=wf_config)
        result = runner.run(dataset_df=dataset_df, save_artifacts=save_artifacts)
    except Exception as exc:
        logger.error("Walk-forward validation failed: %s", exc)
        print(f"WALK-FORWARD ERROR: {exc}")
        print("=" * 60)
        return 1

    agg = result.aggregate_metrics
    print("-" * 60)
    print(f"Experiment ID    : {result.experiment_id}")
    print(f"Total Folds      : {agg.total_folds}")
    print(f"Total Val Samples: {agg.total_validation_samples}")
    print(f"Mean Accuracy    : {agg.mean_accuracy:.4f} (+/- {agg.std_accuracy:.4f})")
    print(f"Mean Precision   : {agg.mean_precision:.4f} (+/- {agg.std_precision:.4f})")
    print(f"Mean Recall      : {agg.mean_recall:.4f} (+/- {agg.std_recall:.4f})")
    print(f"Mean F1 Score    : {agg.mean_f1:.4f} (+/- {agg.std_f1:.4f})")
    roc_str = f"{agg.mean_roc_auc:.4f}" if agg.mean_roc_auc is not None else "N/A"
    print(f"Mean ROC-AUC     : {roc_str}")
    print("-" * 60)
    print(f"Majority Baseline: {agg.mean_baseline_accuracy:.4f}")
    print(f"OOS Predictions  : {len(result.oos_predictions)} rows")
    print("=" * 60)
    return 0


def main() -> None:
    """CLI entrypoint supporting train and walk-forward subcommands."""
    parser = argparse.ArgumentParser(
        description="Adaptive Trading ML Training & Walk-Forward Validation"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Subcommand: train
    train_parser = subparsers.add_parser(
        "train", help="Train single chronological baseline model"
    )
    train_parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="data/sample/nifty_5m_ml_sample.csv",
        help="Path to input market data CSV file",
    )
    train_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="Train split ratio (default: 0.80)",
    )
    train_parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="Forward target horizon in bars (default: 3)",
    )
    train_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save model artifact to disk",
    )
    train_parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Artifact version subfolder (default: v1)",
    )

    # Subcommand: walk-forward
    wf_parser = subparsers.add_parser(
        "walk-forward", help="Run walk-forward cross-validation"
    )
    wf_parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="data/sample/nifty_5m_ml_sample.csv",
        help="Path to input market data CSV file",
    )
    wf_parser.add_argument(
        "--initial-train-size",
        type=float,
        default=0.50,
        help="Initial training size fraction (default: 0.50)",
    )
    wf_parser.add_argument(
        "--validation-size",
        type=float,
        default=0.10,
        help="Validation window size fraction (default: 0.10)",
    )
    wf_parser.add_argument(
        "--step-size",
        type=float,
        default=None,
        help="Step increment fraction (defaults to validation-size)",
    )
    wf_parser.add_argument(
        "--gap",
        type=int,
        default=0,
        help="Purge gap between train and validation (default: 0)",
    )
    wf_parser.add_argument(
        "--window-type",
        choices=["EXPANDING", "ROLLING"],
        default="EXPANDING",
        help="Window strategy: EXPANDING or ROLLING (default: EXPANDING)",
    )
    wf_parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="Forward target horizon in bars (default: 3)",
    )
    wf_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save experiment artifacts",
    )

    # Fallback default options when run without subcommands
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="data/sample/nifty_5m_ml_sample.csv",
        help="Path to input market data CSV file",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="Train split ratio (default: 0.80)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="Forward target horizon in bars (default: 3)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save model artifact to disk",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Artifact version subfolder (default: v1)",
    )

    args = parser.parse_args()

    if args.command == "walk-forward":
        exit_code = run_walk_forward_pipeline(
            csv_file=args.file,
            initial_train_size=args.initial_train_size,
            validation_size=args.validation_size,
            step_size=args.step_size,
            gap=args.gap,
            window_type_str=args.window_type,
            horizon=args.horizon,
            save_artifacts=not args.no_save,
        )
    else:
        # Default to single train pipeline
        exit_code = run_training_pipeline(
            csv_file=args.file,
            train_ratio=args.train_ratio,
            horizon=args.horizon,
            save_artifact=not args.no_save,
            artifact_version=args.version,
        )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
