"""Top-level CLI dispatcher for the Adaptive Trading Platform."""

import argparse
import sys
from pathlib import Path

from adaptive_trading.evaluation.cli import (
    cmd_compare as cmd_eval_compare,
)
from adaptive_trading.evaluation.cli import (
    cmd_show as cmd_eval_show,
)
from adaptive_trading.evaluation.cli import (
    cmd_walk_forward,
)
from adaptive_trading.evaluation.cli import (
    cmd_windows as cmd_eval_windows,
)
from adaptive_trading.experiments.cli import (
    cmd_compare,
    cmd_create,
    cmd_manifest,
    cmd_verify,
)
from adaptive_trading.experiments.cli import (
    cmd_list as cmd_exp_list,
)
from adaptive_trading.experiments.cli import (
    cmd_show as cmd_exp_show,
)
from adaptive_trading.observability.cli import (
    cmd_events,
    cmd_health,
    cmd_metrics,
    cmd_run_details,
    cmd_runs,
)
from adaptive_trading.validation.cli import (
    cmd_validate_dataset,
    cmd_validate_features,
    cmd_validate_model,
    cmd_validate_run,
)
from adaptive_trading.validation.config import ValidationPolicy


def build_parser() -> argparse.ArgumentParser:
    """Construct top-level CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m adaptive_trading",
        description="Adaptive Trading Platform CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Subsystem")

    # 1. Monitoring subsystem
    p_mon = subparsers.add_parser(
        "monitoring", help="Observability and health monitoring"
    )
    mon_sub = p_mon.add_subparsers(dest="mon_cmd", help="Monitoring action")

    # health
    p_health = mon_sub.add_parser("health", help="Evaluate system health")
    p_health.add_argument("--json", action="store_true", help="JSON output format")

    # metrics
    p_metrics = mon_sub.add_parser("metrics", help="Display metrics")
    p_metrics.add_argument("--json", action="store_true", help="JSON output format")

    # runs
    p_runs = mon_sub.add_parser("runs", help="List execution runs")
    p_runs.add_argument("--limit", type=int, default=20, help="Max runs to show")
    p_runs.add_argument("--json", action="store_true", help="JSON output format")

    # run
    p_run = mon_sub.add_parser("run", help="Display details for a run")
    p_run.add_argument("run_id", type=str, help="Run ID")
    p_run.add_argument("--json", action="store_true", help="JSON output format")

    # events
    p_events = mon_sub.add_parser("events", help="Query lifecycle events for a run")
    p_events.add_argument("run_id", type=str, help="Run ID")
    p_events.add_argument("--type", type=str, default=None, help="Filter by event type")
    p_events.add_argument("--json", action="store_true", help="JSON output format")

    # 2. Experiments subsystem
    p_exp = subparsers.add_parser(
        "experiments", help="Experiment and version management"
    )
    exp_sub = p_exp.add_subparsers(dest="exp_cmd", help="Experiment action")

    # experiments list
    p_exp_l = exp_sub.add_parser("list", help="List tracked experiments")
    p_exp_l.add_argument(
        "--limit", type=int, default=20, help="Max experiments to list"
    )
    p_exp_l.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("artifacts/experiments"),
        help="Experiments directory",
    )
    p_exp_l.add_argument("--json", action="store_true", help="JSON output format")

    # experiments show
    p_exp_s = exp_sub.add_parser("show", help="Show experiment details")
    p_exp_s.add_argument("experiment_id", type=str, help="Experiment ID")
    p_exp_s.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("artifacts/experiments"),
        help="Experiments directory",
    )
    p_exp_s.add_argument("--json", action="store_true", help="JSON output format")

    # experiments manifest
    p_exp_m = exp_sub.add_parser("manifest", help="Display experiment manifest")
    p_exp_m.add_argument("experiment_id", type=str, help="Experiment ID")
    p_exp_m.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("artifacts/experiments"),
        help="Experiments directory",
    )
    p_exp_m.add_argument("--json", action="store_true", help="JSON output format")

    # experiments compare
    p_exp_c = exp_sub.add_parser("compare", help="Compare two experiments")
    p_exp_c.add_argument("exp_a", type=str, help="First experiment ID")
    p_exp_c.add_argument("exp_b", type=str, help="Second experiment ID")
    p_exp_c.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("artifacts/experiments"),
        help="Experiments directory",
    )
    p_exp_c.add_argument("--json", action="store_true", help="JSON output format")

    # experiments verify
    p_exp_v = exp_sub.add_parser("verify", help="Verify experiment reproducibility")
    p_exp_v.add_argument("experiment_id", type=str, help="Experiment ID")
    p_exp_v.add_argument(
        "--target",
        type=str,
        default=None,
        help="Optional target experiment ID to compare against",
    )
    p_exp_v.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("artifacts/experiments"),
        help="Experiments directory",
    )
    p_exp_v.add_argument("--json", action="store_true", help="JSON output format")

    # experiments create
    p_exp_cr = exp_sub.add_parser(
        "create", help="Create and run a paper replay experiment"
    )
    p_exp_cr.add_argument(
        "--candles",
        type=Path,
        default=Path("data/sample/nifty_5m_ml_sample.csv"),
        help="Candle data CSV path",
    )
    p_exp_cr.add_argument(
        "--name",
        type=str,
        default="cli_paper_replay_exp",
        help="Experiment name",
    )
    p_exp_cr.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("artifacts/experiments"),
        help="Experiments directory",
    )
    p_exp_cr.add_argument("--json", action="store_true", help="JSON output format")

    # 3. Validation subsystem
    p_val = subparsers.add_parser(
        "validation", help="Data, feature, model, and leakage quality validation"
    )
    val_sub = p_val.add_subparsers(dest="val_cmd", help="Validation target")

    # validation dataset
    p_val_d = val_sub.add_parser("dataset", help="Validate market dataset")
    p_val_d.add_argument("dataset", type=Path, help="Path to candle CSV")
    p_val_d.add_argument("--symbol", type=str, default=None, help="Expected symbol")
    p_val_d.add_argument(
        "--timeframe", type=str, default=None, help="Expected timeframe"
    )
    p_val_d.add_argument(
        "--policy",
        choices=["STRICT", "NORMAL", "LENIENT"],
        default="NORMAL",
        help="Validation gating policy",
    )
    p_val_d.add_argument("--json", action="store_true", help="JSON output format")

    # validation features
    p_val_f = val_sub.add_parser("features", help="Validate feature dataset")
    p_val_f.add_argument("features", type=Path, help="Path to feature CSV")
    p_val_f.add_argument(
        "--policy",
        choices=["STRICT", "NORMAL", "LENIENT"],
        default="NORMAL",
        help="Validation gating policy",
    )
    p_val_f.add_argument("--json", action="store_true", help="JSON output format")

    # validation model
    p_val_m = val_sub.add_parser("model", help="Validate model artifact")
    p_val_m.add_argument("model", type=Path, help="Path to model artifact")
    p_val_m.add_argument(
        "--fingerprint", type=str, default=None, help="Expected SHA-256 fingerprint"
    )
    p_val_m.add_argument(
        "--policy",
        choices=["STRICT", "NORMAL", "LENIENT"],
        default="NORMAL",
        help="Validation gating policy",
    )
    p_val_m.add_argument("--json", action="store_true", help="JSON output format")

    # validation run
    p_val_r = val_sub.add_parser("run", help="Run full multi-layer validation")
    p_val_r.add_argument("dataset", type=Path, help="Path to candle CSV")
    p_val_r.add_argument(
        "--features", type=Path, default=None, help="Optional feature CSV"
    )
    p_val_r.add_argument(
        "--model", type=Path, default=None, help="Optional model artifact path"
    )
    p_val_r.add_argument(
        "--policy",
        choices=["STRICT", "NORMAL", "LENIENT"],
        default="NORMAL",
        help="Validation gating policy",
    )
    p_val_r.add_argument("--json", action="store_true", help="JSON output format")

    # 4. Evaluation subsystem
    p_eval = subparsers.add_parser(
        "evaluate", help="Advanced walk-forward evaluation and backtesting"
    )
    eval_sub = p_eval.add_subparsers(dest="eval_cmd", help="Evaluation action")

    # evaluate walk-forward
    p_eval_wf = eval_sub.add_parser("walk-forward", help="Run walk-forward evaluation")
    p_eval_wf.add_argument("--dataset", required=True, help="Path to candles CSV")
    p_eval_wf.add_argument(
        "--window-type",
        choices=["rolling", "expanding"],
        default="rolling",
        help="Window progression type",
    )
    p_eval_wf.add_argument(
        "--train-window",
        type=int,
        default=30,
        help="Training window candle size",
    )
    p_eval_wf.add_argument(
        "--validation-window",
        type=int,
        default=0,
        help="Validation window size",
    )
    p_eval_wf.add_argument(
        "--test-window",
        type=int,
        default=10,
        help="Test window candle size",
    )
    p_eval_wf.add_argument(
        "--step-size",
        type=int,
        default=None,
        help="Step size forward",
    )
    p_eval_wf.add_argument(
        "--purge",
        type=int,
        default=0,
        help="Purge embargo period",
    )
    p_eval_wf.add_argument(
        "--policy",
        choices=["STRICT", "NORMAL", "LENIENT"],
        default="NORMAL",
        help="Step 21 validation policy",
    )
    p_eval_wf.add_argument(
        "--artifacts-dir",
        default="artifacts/evaluations",
        help="Output directory",
    )
    p_eval_wf.add_argument("--json", action="store_true", help="JSON output format")

    # evaluate show
    p_eval_s = eval_sub.add_parser("show", help="Display evaluation summary")
    p_eval_s.add_argument("evaluation_id", help="Evaluation ID")
    p_eval_s.add_argument(
        "--artifacts-dir",
        default="artifacts/evaluations",
        help="Artifacts directory",
    )
    p_eval_s.add_argument("--json", action="store_true", help="JSON output format")

    # evaluate windows
    p_eval_w = eval_sub.add_parser("windows", help="Display per-window breakdown")
    p_eval_w.add_argument("evaluation_id", help="Evaluation ID")
    p_eval_w.add_argument(
        "--artifacts-dir",
        default="artifacts/evaluations",
        help="Artifacts directory",
    )
    p_eval_w.add_argument("--json", action="store_true", help="JSON output format")

    # evaluate compare
    p_eval_c = eval_sub.add_parser("compare", help="Compare two evaluations")
    p_eval_c.add_argument("eval_a", help="First evaluation ID")
    p_eval_c.add_argument("eval_b", help="Second evaluation ID")
    p_eval_c.add_argument(
        "--artifacts-dir",
        default="artifacts/evaluations",
        help="Artifacts directory",
    )
    p_eval_c.add_argument("--json", action="store_true", help="JSON output format")

    return parser


def main() -> None:
    """Main CLI entry point for python -m adaptive_trading."""
    parser = build_parser()
    args = parser.parse_args()

    if args.subcommand == "monitoring":
        if args.mon_cmd == "health":
            code = cmd_health(as_json=args.json)
        elif args.mon_cmd == "metrics":
            code = cmd_metrics(as_json=args.json)
        elif args.mon_cmd == "runs":
            code = cmd_runs(limit=args.limit, as_json=args.json)
        elif args.mon_cmd == "run":
            code = cmd_run_details(run_id=args.run_id, as_json=args.json)
        elif args.mon_cmd == "events":
            code = cmd_events(
                run_id=args.run_id, event_type=args.type, as_json=args.json
            )
        else:
            parser.print_help()
            code = 1
    elif args.subcommand == "experiments":
        if args.exp_cmd == "list":
            code = cmd_exp_list(
                base_dir=args.experiments_dir,
                limit=args.limit,
                as_json=args.json,
            )
        elif args.exp_cmd == "show":
            code = cmd_exp_show(
                experiment_id=args.experiment_id,
                base_dir=args.experiments_dir,
                as_json=args.json,
            )
        elif args.exp_cmd == "manifest":
            code = cmd_manifest(
                experiment_id=args.experiment_id,
                base_dir=args.experiments_dir,
                as_json=args.json,
            )
        elif args.exp_cmd == "compare":
            code = cmd_compare(
                exp_a_id=args.exp_a,
                exp_b_id=args.exp_b,
                base_dir=args.experiments_dir,
                as_json=args.json,
            )
        elif args.exp_cmd == "verify":
            code = cmd_verify(
                experiment_id=args.experiment_id,
                target_exp_id=args.target,
                base_dir=args.experiments_dir,
                as_json=args.json,
            )
        elif args.exp_cmd == "create":
            code = cmd_create(
                candles_path=args.candles,
                name=args.name,
                base_dir=args.experiments_dir,
                as_json=args.json,
            )
        else:
            parser.print_help()
            code = 1
    elif args.subcommand == "validation":
        policy = (
            ValidationPolicy(args.policy)
            if hasattr(args, "policy")
            else ValidationPolicy.NORMAL
        )
        if args.val_cmd == "dataset":
            code = cmd_validate_dataset(
                dataset_path=args.dataset,
                policy=policy,
                expected_symbol=args.symbol,
                expected_timeframe=args.timeframe,
                as_json=args.json,
            )
        elif args.val_cmd == "features":
            code = cmd_validate_features(
                features_path=args.features,
                policy=policy,
                as_json=args.json,
            )
        elif args.val_cmd == "model":
            code = cmd_validate_model(
                model_path=args.model,
                expected_fingerprint=args.fingerprint,
                policy=policy,
                as_json=args.json,
            )
        elif args.val_cmd == "run":
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
    elif args.subcommand == "evaluate":
        if args.eval_cmd == "walk-forward":
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
        elif args.eval_cmd == "show":
            code = cmd_eval_show(
                evaluation_id=args.evaluation_id,
                artifacts_dir=args.artifacts_dir,
                as_json=args.json,
            )
        elif args.eval_cmd == "windows":
            code = cmd_eval_windows(
                evaluation_id=args.evaluation_id,
                artifacts_dir=args.artifacts_dir,
                as_json=args.json,
            )
        elif args.eval_cmd == "compare":
            code = cmd_eval_compare(
                eval_a=args.eval_a,
                eval_b=args.eval_b,
                artifacts_dir=args.artifacts_dir,
                as_json=args.json,
            )
        else:
            parser.print_help()
            code = 1
    else:
        parser.print_help()
        code = 1

    sys.exit(code)


if __name__ == "__main__":
    main()
