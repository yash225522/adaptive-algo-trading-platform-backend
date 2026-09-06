"""Command-line interface for data quality, feature, model, and leakage validation."""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from adaptive_trading.validation.config import ValidationConfig, ValidationPolicy
from adaptive_trading.validation.data_validator import DataValidator
from adaptive_trading.validation.feature_validator import FeatureValidator
from adaptive_trading.validation.leakage import LeakageValidator
from adaptive_trading.validation.model_validator import ModelValidator
from adaptive_trading.validation.report import ValidationReport, ValidationResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_csv_as_dataframe(path: Path) -> pd.DataFrame:
    """Helper to load a CSV file into a pandas DataFrame."""
    if not path.is_file():
        raise FileNotFoundError(f"File not found: '{path}'")
    return pd.read_csv(path)


def cmd_validate_dataset(
    dataset_path: Path,
    policy: ValidationPolicy = ValidationPolicy.NORMAL,
    expected_symbol: str | None = None,
    expected_timeframe: str | None = None,
    as_json: bool = False,
) -> int:
    """Validate a historical market dataset."""
    try:
        df = _load_csv_as_dataframe(dataset_path)
    except Exception as exc:
        print(f"Error loading dataset: {exc}", file=sys.stderr)
        return 1

    cfg = ValidationConfig(
        policy=policy,
        expected_symbol=expected_symbol,
        expected_timeframe=expected_timeframe,
    )
    validator = DataValidator(config=cfg)
    checks, dataset_fp = validator.validate_dataset(
        df,
        expected_symbol=expected_symbol,
        expected_timeframe=expected_timeframe,
    )

    report = ValidationReport.create(
        checks=checks,
        dataset_identifier=str(dataset_path.name),
        policy=policy,
        dataset_fingerprint=dataset_fp,
    )

    if as_json:
        print(json.dumps(report.model_dump(mode="json"), indent=2))
    else:
        print(report.to_text_summary())

    return 0 if report.is_allowed(policy) else 1


def cmd_validate_features(
    features_path: Path,
    policy: ValidationPolicy = ValidationPolicy.NORMAL,
    as_json: bool = False,
) -> int:
    """Validate extracted feature tabular dataset."""
    try:
        df = _load_csv_as_dataframe(features_path)
    except Exception as exc:
        print(f"Error loading features file: {exc}", file=sys.stderr)
        return 1

    cfg = ValidationConfig(policy=policy)
    feat_val = FeatureValidator(config=cfg)
    checks = feat_val.validate_features_df(df)

    leak_val = LeakageValidator(config=cfg)
    target_chk = leak_val.validate_target_leakage(list(df.columns))
    checks.append(target_chk)

    report = ValidationReport.create(
        checks=checks,
        dataset_identifier=str(features_path.name),
        policy=policy,
    )

    if as_json:
        print(json.dumps(report.model_dump(mode="json"), indent=2))
    else:
        print(report.to_text_summary())

    return 0 if report.is_allowed(policy) else 1


def cmd_validate_model(
    model_path: Path,
    expected_fingerprint: str | None = None,
    policy: ValidationPolicy = ValidationPolicy.NORMAL,
    as_json: bool = False,
) -> int:
    """Validate an ML model artifact."""
    cfg = ValidationConfig(policy=policy)
    val = ModelValidator(config=cfg)
    checks, _ = val.validate_artifact(
        model_path, expected_fingerprint=expected_fingerprint
    )

    report = ValidationReport.create(
        checks=checks,
        dataset_identifier=str(model_path.name),
        policy=policy,
    )

    if as_json:
        print(json.dumps(report.model_dump(mode="json"), indent=2))
    else:
        print(report.to_text_summary())

    return 0 if report.is_allowed(policy) else 1


def cmd_validate_run(
    dataset_path: Path,
    features_path: Path | None = None,
    model_path: Path | None = None,
    policy: ValidationPolicy = ValidationPolicy.NORMAL,
    as_json: bool = False,
) -> int:
    """Perform full end-to-end validation across data, features, and model."""
    try:
        df = _load_csv_as_dataframe(dataset_path)
    except Exception as exc:
        print(f"Error loading dataset: {exc}", file=sys.stderr)
        return 1

    cfg = ValidationConfig(policy=policy)
    checks: list[ValidationResult] = []

    # 1. Data Validation
    data_val = DataValidator(config=cfg)
    data_checks, dataset_fp = data_val.validate_dataset(df)
    checks.extend(data_checks)

    # 2. Features Validation
    if features_path:
        try:
            feat_df = _load_csv_as_dataframe(features_path)
            feat_val = FeatureValidator(config=cfg)
            checks.extend(feat_val.validate_features_df(feat_df))

            leak_val = LeakageValidator(config=cfg)
            checks.append(leak_val.validate_target_leakage(list(feat_df.columns)))
        except Exception as exc:
            print(f"Warning: could not validate features: {exc}", file=sys.stderr)

    # 3. Model Validation
    if model_path:
        model_val = ModelValidator(config=cfg)
        model_checks, _ = model_val.validate_artifact(model_path)
        checks.extend(model_checks)

    report = ValidationReport.create(
        checks=checks,
        dataset_identifier=str(dataset_path.name),
        policy=policy,
        dataset_fingerprint=dataset_fp,
    )

    if as_json:
        print(json.dumps(report.model_dump(mode="json"), indent=2))
    else:
        print(report.to_text_summary())

    return 0 if report.is_allowed(policy) else 1


def create_validation_parser() -> argparse.ArgumentParser:
    """Construct subparser tree for validation commands."""
    parser = argparse.ArgumentParser(
        description="Adaptive Trading Platform Data & Model Quality Validation CLI"
    )
    subparsers = parser.add_subparsers(
        dest="subcommand", help="Validation target action"
    )

    # dataset
    p_data = subparsers.add_parser("dataset", help="Validate market dataset")
    p_data.add_argument("dataset", type=Path, help="Path to candle CSV file")
    p_data.add_argument("--symbol", type=str, default=None, help="Expected symbol")
    p_data.add_argument(
        "--timeframe", type=str, default=None, help="Expected timeframe"
    )
    p_data.add_argument(
        "--policy",
        choices=["STRICT", "NORMAL", "LENIENT"],
        default="NORMAL",
        help="Validation gating policy",
    )
    p_data.add_argument("--json", action="store_true", help="JSON output format")

    # features
    p_feat = subparsers.add_parser("features", help="Validate feature dataset")
    p_feat.add_argument("features", type=Path, help="Path to feature CSV file")
    p_feat.add_argument(
        "--policy",
        choices=["STRICT", "NORMAL", "LENIENT"],
        default="NORMAL",
        help="Validation gating policy",
    )
    p_feat.add_argument("--json", action="store_true", help="JSON output format")

    # model
    p_mod = subparsers.add_parser("model", help="Validate model artifact")
    p_mod.add_argument("model", type=Path, help="Path to model directory or file")
    p_mod.add_argument(
        "--fingerprint",
        type=str,
        default=None,
        help="Expected artifact SHA-256",
    )
    p_mod.add_argument(
        "--policy",
        choices=["STRICT", "NORMAL", "LENIENT"],
        default="NORMAL",
        help="Validation gating policy",
    )
    p_mod.add_argument("--json", action="store_true", help="JSON output format")

    # run (all)
    p_run = subparsers.add_parser("run", help="Run full multi-layer validation")
    p_run.add_argument("dataset", type=Path, help="Path to candle CSV file")
    p_run.add_argument(
        "--features", type=Path, default=None, help="Optional feature CSV"
    )
    p_run.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional model artifact path",
    )
    p_run.add_argument(
        "--policy",
        choices=["STRICT", "NORMAL", "LENIENT"],
        default="NORMAL",
        help="Validation gating policy",
    )
    p_run.add_argument("--json", action="store_true", help="JSON output format")

    return parser


def main() -> None:
    """Entrypoint for validation CLI."""
    parser = create_validation_parser()
    args = parser.parse_args()

    policy = (
        ValidationPolicy(args.policy)
        if hasattr(args, "policy")
        else ValidationPolicy.NORMAL
    )

    if args.subcommand == "dataset":
        code = cmd_validate_dataset(
            dataset_path=args.dataset,
            policy=policy,
            expected_symbol=args.symbol,
            expected_timeframe=args.timeframe,
            as_json=args.json,
        )
    elif args.subcommand == "features":
        code = cmd_validate_features(
            features_path=args.features,
            policy=policy,
            as_json=args.json,
        )
    elif args.subcommand == "model":
        code = cmd_validate_model(
            model_path=args.model,
            expected_fingerprint=args.fingerprint,
            policy=policy,
            as_json=args.json,
        )
    elif args.subcommand == "run":
        code = cmd_validate_run(
            dataset_path=args.dataset,
            features_path=args.features,
            model_path=args.model,
            policy=policy,
            as_json=args.json,
        )
    else:
        parser.print_help()
        code = 1

    sys.exit(code)


if __name__ == "__main__":
    main()
