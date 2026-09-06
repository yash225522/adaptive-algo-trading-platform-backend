"""Command line interface for Advanced Walk-Forward Evaluation."""

import argparse
import json
import logging
from pathlib import Path
import sys

import pandas as pd

from adaptive_trading.data.readers.csv_reader import CSVMarketDataReader
from adaptive_trading.data.validators.market_data import MarketDataValidator
from adaptive_trading.domain.market import Candle
from adaptive_trading.evaluation.config import EvaluationConfig, WindowType
from adaptive_trading.evaluation.models import EvaluationReport
from adaptive_trading.evaluation.walk_forward import WalkForwardEvaluator
from adaptive_trading.validation.config import ValidationPolicy

logger = logging.getLogger(__name__)


def load_candles(file_path: Path | str) -> list[Candle]:
    """Read and validate candles from CSV file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    reader = CSVMarketDataReader()
    raw_records = reader.read_records(path)
    validator = MarketDataValidator()
    valid_candles, _ = validator.validate_batch(raw_records)

    if not valid_candles:
        raise ValueError(f"No valid candles could be loaded from {path}")

    return valid_candles


def cmd_walk_forward(
    dataset_file: Path | str,
    window_type: str = "rolling",
    train_window: int = 30,
    validation_window: int = 0,
    test_window: int = 10,
    step_size: int | None = None,
    purge_period: int = 0,
    policy: str = "NORMAL",
    artifacts_dir: Path | str = "artifacts/evaluations",
    as_json: bool = False,
) -> int:
    """Execute walk-forward evaluation from CLI."""
    try:
        candles = load_candles(dataset_file)
    except Exception as exc:
        logger.error("Failed to load dataset: %s", exc)
        if as_json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            print(f"Error loading dataset: {exc}", file=sys.stderr)
        return 1

    w_type = (
        WindowType.EXPANDING
        if window_type.lower() == "expanding"
        else WindowType.ROLLING
    )
    val_policy = (
        ValidationPolicy(policy.upper())
        if policy.upper() in ValidationPolicy.__members__
        else ValidationPolicy.NORMAL
    )

    config = EvaluationConfig(
        train_window=train_window,
        validation_window=validation_window if validation_window > 0 else None,
        test_window=test_window,
        step_size=step_size or test_window,
        window_type=w_type,
        purge_period=purge_period,
        validation_policy=val_policy,
        artifacts_dir=Path(artifacts_dir),
    )

    evaluator = WalkForwardEvaluator(config=config)
    try:
        report = evaluator.evaluate(
            candles=candles,
            dataset_identifier=Path(dataset_file).name,
        )
    except Exception as exc:
        logger.error("Evaluation execution failed: %s", exc)
        if as_json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    if as_json:
        print(json.dumps(report.model_dump(mode="json"), indent=2))
    else:
        print(report.to_text_summary())

    return 0


def cmd_show(
    evaluation_id: str,
    artifacts_dir: Path | str = "artifacts/evaluations",
    as_json: bool = False,
) -> int:
    """Display comprehensive evaluation report summary."""
    report_file = Path(artifacts_dir) / evaluation_id / "report.json"
    if not report_file.is_file():
        print(
            f"Error: Evaluation '{evaluation_id}' not found in {artifacts_dir}",
            file=sys.stderr,
        )
        return 1

    try:
        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        report = EvaluationReport.model_validate(data)
    except Exception as exc:
        print(f"Error reading evaluation report: {exc}", file=sys.stderr)
        return 1

    if as_json:
        print(json.dumps(report.model_dump(mode="json"), indent=2))
    else:
        print(report.to_text_summary())

    return 0


def cmd_windows(
    evaluation_id: str,
    artifacts_dir: Path | str = "artifacts/evaluations",
    as_json: bool = False,
) -> int:
    """Display per-window breakdown of an evaluation run."""
    report_file = Path(artifacts_dir) / evaluation_id / "report.json"
    if not report_file.is_file():
        print(
            f"Error: Evaluation '{evaluation_id}' not found in {artifacts_dir}",
            file=sys.stderr,
        )
        return 1

    try:
        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        report = EvaluationReport.model_validate(data)
    except Exception as exc:
        print(f"Error reading evaluation report: {exc}", file=sys.stderr)
        return 1

    if as_json:
        print(
            json.dumps(
                [w.model_dump(mode="json") for w in report.windows],
                indent=2,
            )
        )
        return 0

    print("=" * 70)
    print(f"   WALK-FORWARD WINDOWS BREAKDOWN: {evaluation_id}")
    print("=" * 70)
    print(
        f"{'WINDOW':<10} {'STATUS':<10} {'TRADES':<8} {'WIN RATE':<10} "
        f"{'RETURN':<10} {'MAX DD':<10}"
    )
    print("-" * 70)
    for w in report.windows:
        print(
            f"{w.window_id:<10} {w.status.value:<10} {w.trade_count:<8} "
            f"{w.win_rate:>6.1f}%   {w.total_return_pct:>+7.2f}%   "
            f"{w.max_drawdown_pct:>6.2f}%"
        )
    print("=" * 70)
    return 0


def cmd_compare(
    eval_a: str,
    eval_b: str,
    artifacts_dir: Path | str = "artifacts/evaluations",
    as_json: bool = False,
) -> int:
    """Compare two evaluation runs side by side."""
    file_a = Path(artifacts_dir) / eval_a / "report.json"
    file_b = Path(artifacts_dir) / eval_b / "report.json"

    if not file_a.is_file():
        print(f"Error: Evaluation '{eval_a}' not found", file=sys.stderr)
        return 1
    if not file_b.is_file():
        print(f"Error: Evaluation '{eval_b}' not found", file=sys.stderr)
        return 1

    try:
        with open(file_a, "r", encoding="utf-8") as f:
            rep_a = EvaluationReport.model_validate(json.load(f))
        with open(file_b, "r", encoding="utf-8") as f:
            rep_b = EvaluationReport.model_validate(json.load(f))
    except Exception as exc:
        print(f"Error loading evaluation reports: {exc}", file=sys.stderr)
        return 1

    comp_dict = {
        "evaluation_a": {
            "id": rep_a.evaluation_id,
            "window_type": rep_a.config.window_type.value,
            "windows_count": len(rep_a.windows),
            "total_return_pct": rep_a.summary.total_return_pct,
            "worst_drawdown": rep_a.summary.worst_drawdown,
            "total_trades": rep_a.summary.total_trades,
            "win_rate": rep_a.summary.overall_win_rate,
            "stability": rep_a.stability.get("classification")
            if rep_a.stability
            else None,
        },
        "evaluation_b": {
            "id": rep_b.evaluation_id,
            "window_type": rep_b.config.window_type.value,
            "windows_count": len(rep_b.windows),
            "total_return_pct": rep_b.summary.total_return_pct,
            "worst_drawdown": rep_b.summary.worst_drawdown,
            "total_trades": rep_b.summary.total_trades,
            "win_rate": rep_b.summary.overall_win_rate,
            "stability": rep_b.stability.get("classification")
            if rep_b.stability
            else None,
        },
        "differences": {
            "return_diff_pct": round(
                rep_a.summary.total_return_pct
                - rep_b.summary.total_return_pct,
                4,
            ),
            "drawdown_diff_pct": round(
                rep_a.summary.worst_drawdown
                - rep_b.summary.worst_drawdown,
                4,
            ),
        },
    }

    if as_json:
        print(json.dumps(comp_dict, indent=2))
        return 0

    print("=" * 70)
    print("         WALK-FORWARD EVALUATION COMPARISON")
    print("=" * 70)
    print(f"{'METRIC':<28} {eval_a:<20} {eval_b:<20}")
    print("-" * 70)
    print(
        f"{'Window Type':<28} {rep_a.config.window_type.value:<20} "
        f"{rep_b.config.window_type.value:<20}"
    )
    print(
        f"{'Total Windows':<28} {len(rep_a.windows):<20} {len(rep_b.windows):<20}"
    )
    print(
        f"{'Out-of-Sample Return':<28} {rep_a.summary.total_return_pct:>+7.2f}% "
        f"             {rep_b.summary.total_return_pct:>+7.2f}%"
    )
    print(
        f"{'Worst Drawdown':<28} {rep_a.summary.worst_drawdown:>7.2f}% "
        f"             {rep_b.summary.worst_drawdown:>7.2f}%"
    )
    print(
        f"{'Total Trades':<28} {rep_a.summary.total_trades:<20} "
        f"{rep_b.summary.total_trades:<20}"
    )
    print(
        f"{'Win Rate':<28} {rep_a.summary.overall_win_rate:>6.1f}% "
        f"              {rep_b.summary.overall_win_rate:>6.1f}%"
    )
    stab_a = (
        rep_a.stability.get("classification", "N/A")
        if rep_a.stability
        else "N/A"
    )
    stab_b = (
        rep_b.stability.get("classification", "N/A")
        if rep_b.stability
        else "N/A"
    )
    print(f"{'Stability':<28} {stab_a:<20} {stab_b:<20}")
    print("=" * 70)
    return 0


def create_evaluation_parser() -> argparse.ArgumentParser:
    """Build argument parser for walk-forward evaluation commands."""
    parser = argparse.ArgumentParser(
        description="Adaptive Trading Platform Walk-Forward Evaluation CLI"
    )
    subparsers = parser.add_subparsers(
        dest="subcommand", help="Evaluation action"
    )

    # walk-forward
    p_wf = subparsers.add_parser(
        "walk-forward", help="Run walk-forward evaluation"
    )
    p_wf.add_argument("--dataset", required=True, help="Path to candles CSV")
    p_wf.add_argument(
        "--window-type",
        choices=["rolling", "expanding"],
        default="rolling",
        help="Window progression type",
    )
    p_wf.add_argument(
        "--train-window",
        type=int,
        default=30,
        help="Training window candle size",
    )
    p_wf.add_argument(
        "--validation-window",
        type=int,
        default=0,
        help="Validation window size",
    )
    p_wf.add_argument(
        "--test-window",
        type=int,
        default=10,
        help="Test window candle size",
    )
    p_wf.add_argument(
        "--step-size",
        type=int,
        default=None,
        help="Step size forward",
    )
    p_wf.add_argument(
        "--purge",
        type=int,
        default=0,
        help="Purge embargo period",
    )
    p_wf.add_argument(
        "--policy",
        choices=["STRICT", "NORMAL", "LENIENT"],
        default="NORMAL",
        help="Step 21 validation policy",
    )
    p_wf.add_argument(
        "--artifacts-dir",
        default="artifacts/evaluations",
        help="Output directory",
    )
    p_wf.add_argument(
        "--json", action="store_true", help="Format output as JSON"
    )

    # show
    p_show = subparsers.add_parser(
        "show", help="Display evaluation summary"
    )
    p_show.add_argument("evaluation_id", help="Evaluation ID to display")
    p_show.add_argument(
        "--artifacts-dir",
        default="artifacts/evaluations",
        help="Artifacts directory",
    )
    p_show.add_argument(
        "--json", action="store_true", help="Format output as JSON"
    )

    # windows
    p_win = subparsers.add_parser(
        "windows", help="Display per-window breakdown"
    )
    p_win.add_argument("evaluation_id", help="Evaluation ID to display")
    p_win.add_argument(
        "--artifacts-dir",
        default="artifacts/evaluations",
        help="Artifacts directory",
    )
    p_win.add_argument(
        "--json", action="store_true", help="Format output as JSON"
    )

    # compare
    p_comp = subparsers.add_parser(
        "compare", help="Compare two evaluation runs"
    )
    p_comp.add_argument("eval_a", help="First evaluation ID")
    p_comp.add_argument("eval_b", help="Second evaluation ID")
    p_comp.add_argument(
        "--artifacts-dir",
        default="artifacts/evaluations",
        help="Artifacts directory",
    )
    p_comp.add_argument(
        "--json", action="store_true", help="Format output as JSON"
    )

    return parser


def main() -> None:
    """CLI dispatcher entry point."""
    parser = create_evaluation_parser()
    args = parser.parse_args()

    if args.subcommand == "walk-forward":
        code = cmd_walk_forward(
            dataset_file=args.dataset,
            window_type=args.window_type,
            train_window=args.train_window,
            validation_window=args.validation_window,
            test_window=args.test_window,
            step_size=args.step_size,
            purge_period=args.purge,
            policy=args.policy,
            artifacts_dir=args.artifacts_dir,
            as_json=args.json,
        )
    elif args.subcommand == "show":
        code = cmd_show(
            evaluation_id=args.evaluation_id,
            artifacts_dir=args.artifacts_dir,
            as_json=args.json,
        )
    elif args.subcommand == "windows":
        code = cmd_windows(
            evaluation_id=args.evaluation_id,
            artifacts_dir=args.artifacts_dir,
            as_json=args.json,
        )
    elif args.subcommand == "compare":
        code = cmd_compare(
            eval_a=args.eval_a,
            eval_b=args.eval_b,
            artifacts_dir=args.artifacts_dir,
            as_json=args.json,
        )
    else:
        parser.print_help()
        code = 1

    sys.exit(code)


if __name__ == "__main__":
    main()

