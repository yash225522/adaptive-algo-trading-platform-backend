"""Command-line interface for experiment tracking, comparison, and reproducibility."""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from adaptive_trading.common.config import TimeFrame
from adaptive_trading.domain.market import Candle
from adaptive_trading.experiments.comparison import ExperimentComparator
from adaptive_trading.experiments.experiment import ExperimentRunner
from adaptive_trading.experiments.registry import ExperimentRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_candles_from_csv(path: Path) -> list[Candle]:
    """Helper to parse a CSV file into Candle objects."""
    df = pd.read_csv(path)
    candles: list[Candle] = []
    for _, row in df.iterrows():
        ts = (
            pd.to_datetime(row["timestamp"]).to_pydatetime()
            if "timestamp" in row
            else datetime.now(timezone.utc)
        )
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        sym = str(row.get("symbol", "NIFTY"))
        c = Candle(
            timestamp=ts,
            symbol=sym,
            timeframe=TimeFrame.FIVE_MINUTES,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0.0)),
        )
        candles.append(c)
    return candles


def cmd_list(
    base_dir: Path = Path("artifacts/experiments"),
    limit: int = 20,
    as_json: bool = False,
) -> int:
    """List tracked experiments."""
    registry = ExperimentRegistry(base_dir=base_dir)
    experiments = registry.list_experiments(limit=limit)

    if as_json:
        print(json.dumps([e.model_dump(mode="json") for e in experiments], indent=2))
        return 0

    print("=" * 80)
    print("      ADAPTIVE TRADING PLATFORM - EXPERIMENTS REGISTRY       ")
    print("=" * 80)
    print(f"{'EXPERIMENT ID':<24} {'NAME':<20} {'TYPE':<14} {'STATUS':<12} {'CREATED'}")
    print("-" * 80)
    for exp in experiments:
        created_str = exp.created_at.strftime("%Y-%m-%d %H:%M")
        print(
            f"{exp.experiment_id:<24} {exp.name[:18]:<20} "
            f"{exp.experiment_type.value:<14} {exp.status.value:<12} {created_str}"
        )
    print("=" * 80)
    return 0


def cmd_show(
    experiment_id: str,
    base_dir: Path = Path("artifacts/experiments"),
    as_json: bool = False,
) -> int:
    """Show details of a specific experiment."""
    registry = ExperimentRegistry(base_dir=base_dir)
    exp = registry.get_experiment(experiment_id)
    if exp is None:
        print(f"Error: Experiment '{experiment_id}' not found", file=sys.stderr)
        return 1

    manifest = registry.get_manifest(experiment_id)

    if as_json:
        payload = {
            "experiment": exp.model_dump(mode="json"),
            "manifest": manifest.model_dump(mode="json") if manifest else None,
        }
        print(json.dumps(payload, indent=2))
        return 0

    print("=" * 65)
    print("      ADAPTIVE TRADING PLATFORM - EXPERIMENT DETAILS        ")
    print("=" * 65)
    print(f"ID               : {exp.experiment_id}")
    print(f"Name             : {exp.name}")
    print(f"Description      : {exp.description or 'N/A'}")
    print(f"Type             : {exp.experiment_type.value}")
    print(f"Status           : {exp.status.value}")
    print(f"Created At       : {exp.created_at.isoformat()}")
    if manifest:
        print("-" * 65)
        ds_fp = manifest.dataset.fingerprint[:8]
        ds_sym = manifest.dataset.symbol
        ds_rows = manifest.dataset.row_count
        print(f"Dataset          : {ds_sym} ({ds_rows} rows, fp={ds_fp})")
        feat_fp = manifest.features.fingerprint[:8]
        feat_name = manifest.features.feature_set_name
        feat_ver = manifest.features.feature_version
        print(f"Features         : {feat_name} (v{feat_ver}, fp={feat_fp})")
        art_fp = manifest.model.artifact_fingerprint[:8] or "N/A"
        mod_name = manifest.model.model_name
        mod_ver = manifest.model.model_version
        print(f"Model            : {mod_name} ({mod_ver}, fp={art_fp})")
        print(
            f"Strategy         : {manifest.strategy.strategy_name} "
            f"(fp={manifest.strategy.fingerprint[:8]})"
        )
        print(
            f"Risk Config      : v{manifest.risk.risk_version} "
            f"(fp={manifest.risk.fingerprint[:8]})"
        )
        print(
            f"Execution Mode   : {manifest.execution.execution_mode} "
            f"(fp={manifest.execution.fingerprint[:8]})"
        )
        print(
            f"Runtime Mode     : {manifest.runtime.runtime_mode} "
            f"(warmup={manifest.runtime.warmup_period})"
        )
        if manifest.result:
            r = manifest.result
            print("-" * 65)
            print(f"Final Equity     : {r.final_equity:,.2f}")
            print(f"Net P&L          : {r.net_pnl:+,.2f}")
            print(f"Trade Count      : {r.trade_count}")
            print(f"Max Drawdown     : {r.max_drawdown:.2%}")
            if r.win_rate is not None:
                print(f"Win Rate         : {r.win_rate:.2%}")
    print("=" * 65)
    return 0


def cmd_manifest(
    experiment_id: str,
    base_dir: Path = Path("artifacts/experiments"),
    as_json: bool = False,
) -> int:
    """Output the reproducibility manifest for an experiment."""
    registry = ExperimentRegistry(base_dir=base_dir)
    manifest = registry.get_manifest(experiment_id)
    if manifest is None:
        print(
            f"Error: Manifest for experiment '{experiment_id}' not found",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(manifest.model_dump(mode="json"), indent=2))
    return 0


def cmd_compare(
    exp_a_id: str,
    exp_b_id: str,
    base_dir: Path = Path("artifacts/experiments"),
    as_json: bool = False,
) -> int:
    """Compare two experiment manifests and display diffs."""
    registry = ExperimentRegistry(base_dir=base_dir)
    man_a = registry.get_manifest(exp_a_id)
    if man_a is None:
        print(f"Error: Manifest '{exp_a_id}' not found", file=sys.stderr)
        return 1

    man_b = registry.get_manifest(exp_b_id)
    if man_b is None:
        print(f"Error: Manifest '{exp_b_id}' not found", file=sys.stderr)
        return 1

    comparison = ExperimentComparator.compare(man_a, man_b)

    if as_json:
        print(json.dumps(comparison.model_dump(mode="json"), indent=2))
        return 0

    print("=" * 70)
    print("      ADAPTIVE TRADING PLATFORM - EXPERIMENT COMPARISON      ")
    print("=" * 70)
    print(f"Experiment A     : {man_a.experiment_id}")
    print(f"Experiment B     : {man_b.experiment_id}")
    print(f"Identical Config : {comparison.is_identical_config}")
    print(f"Identical Result : {comparison.is_identical_result}")
    print("-" * 70)
    print("CONFIGURATION DIFFERENCES:")
    if comparison.config_diffs:
        for comp_name, diffs in comparison.config_diffs.items():
            print(f"  - {comp_name.upper()}:")
            print(f"      A: {diffs['a']}")
            print(f"      B: {diffs['b']}")
    else:
        print("  (No configuration differences detected)")

    print("-" * 70)
    print("PERFORMANCE DELTAS (B - A):")
    if comparison.performance_diffs:
        for metric, val in comparison.performance_diffs.items():
            if isinstance(val, float):
                print(f"  - {metric:<22}: {val:+,.2f}")
            else:
                print(f"  - {metric:<22}: {val}")
    else:
        print("  (No performance results available)")
    print("=" * 70)
    return 0


def cmd_verify(
    experiment_id: str,
    target_exp_id: str | None = None,
    base_dir: Path = Path("artifacts/experiments"),
    as_json: bool = False,
) -> int:
    """Verify reproducibility of an experiment against itself or a target."""
    registry = ExperimentRegistry(base_dir=base_dir)
    manifest = registry.get_manifest(experiment_id)
    if manifest is None:
        print(f"Error: Manifest '{experiment_id}' not found", file=sys.stderr)
        return 1

    target_manifest = registry.get_manifest(target_exp_id) if target_exp_id else None
    level, reasons = ExperimentComparator.verify_reproducibility(
        manifest, target_manifest
    )

    if as_json:
        print(
            json.dumps(
                {
                    "experiment_id": experiment_id,
                    "target_experiment_id": target_exp_id,
                    "reproducibility_level": level.value,
                    "reasons": reasons,
                },
                indent=2,
            )
        )
        return 0

    print("=" * 60)
    print("      REPRODUCIBILITY VERIFICATION REPORT                   ")
    print("=" * 60)
    print(f"Experiment ID    : {experiment_id}")
    if target_exp_id:
        print(f"Target ID        : {target_exp_id}")
    print(f"Status           : {level.value}")
    print("-" * 60)
    print("DIAGNOSTIC DETAILS:")
    for r in reasons:
        print(f"  - {r}")
    print("=" * 60)
    return 0


def cmd_create(
    candles_path: Path,
    name: str = "cli_paper_replay_exp",
    base_dir: Path = Path("artifacts/experiments"),
    as_json: bool = False,
) -> int:
    """Run an end-to-end replay experiment from candle CSV data."""
    if not candles_path.is_file():
        print(f"Error: Candle file '{candles_path}' not found", file=sys.stderr)
        return 1

    candles = _load_candles_from_csv(candles_path)
    registry = ExperimentRegistry(base_dir=base_dir)
    runner = ExperimentRunner(registry=registry)

    exp, manifest, result = runner.run_paper_replay_experiment(
        candles=candles,
        name=name,
    )

    if as_json:
        print(json.dumps(manifest.model_dump(mode="json"), indent=2))
        return 0

    print(f"Experiment '{exp.experiment_id}' ({exp.name}) completed successfully!")
    pnl_str = (
        f"Result Net P&L: {result.net_pnl:+,.2f} | "
        f"Final Equity: {result.final_equity:,.2f}"
    )
    print(pnl_str)
    return 0


def create_experiments_parser() -> argparse.ArgumentParser:
    """Build argument subparser tree for experiments commands."""
    parser = argparse.ArgumentParser(
        description="Adaptive Trading Platform Experiments & Reproducibility CLI"
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Experiment action")

    # list
    p_list = subparsers.add_parser("list", help="List tracked experiments")
    p_list.add_argument(
        "--limit", type=int, default=20, help="Maximum experiments to list"
    )
    p_list.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("artifacts/experiments"),
        help="Path to experiments directory",
    )
    p_list.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )

    # show
    p_show = subparsers.add_parser("show", help="Show experiment details")
    p_show.add_argument("experiment_id", type=str, help="Experiment ID")
    p_show.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("artifacts/experiments"),
        help="Path to experiments directory",
    )
    p_show.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )

    # manifest
    p_man = subparsers.add_parser("manifest", help="Show reproducibility manifest")
    p_man.add_argument("experiment_id", type=str, help="Experiment ID")
    p_man.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("artifacts/experiments"),
        help="Path to experiments directory",
    )
    p_man.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )

    # compare
    p_comp = subparsers.add_parser("compare", help="Compare two experiments")
    p_comp.add_argument("exp_a", type=str, help="First experiment ID")
    p_comp.add_argument("exp_b", type=str, help="Second experiment ID")
    p_comp.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("artifacts/experiments"),
        help="Path to experiments directory",
    )
    p_comp.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )

    # verify
    p_ver = subparsers.add_parser(
        "verify", help="Verify reproducibility of an experiment"
    )
    p_ver.add_argument("experiment_id", type=str, help="Experiment ID")
    p_ver.add_argument(
        "--target",
        type=str,
        default=None,
        help="Optional target experiment ID to compare against",
    )
    p_ver.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("artifacts/experiments"),
        help="Path to experiments directory",
    )
    p_ver.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )

    # create
    p_create = subparsers.add_parser(
        "create", help="Execute and record a replay experiment"
    )
    p_create.add_argument(
        "--candles",
        type=Path,
        default=Path("data/sample/nifty_5m_ml_sample.csv"),
        help="Path to candle CSV file",
    )
    p_create.add_argument(
        "--name",
        type=str,
        default="cli_paper_replay_exp",
        help="Experiment name",
    )
    p_create.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("artifacts/experiments"),
        help="Path to experiments directory",
    )
    p_create.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )

    return parser


def main() -> None:
    """Entry point for experiment management CLI."""
    parser = create_experiments_parser()
    args = parser.parse_args()

    if args.subcommand == "list":
        code = cmd_list(
            base_dir=args.experiments_dir, limit=args.limit, as_json=args.json
        )
    elif args.subcommand == "show":
        code = cmd_show(
            experiment_id=args.experiment_id,
            base_dir=args.experiments_dir,
            as_json=args.json,
        )
    elif args.subcommand == "manifest":
        code = cmd_manifest(
            experiment_id=args.experiment_id,
            base_dir=args.experiments_dir,
            as_json=args.json,
        )
    elif args.subcommand == "compare":
        code = cmd_compare(
            exp_a_id=args.exp_a,
            exp_b_id=args.exp_b,
            base_dir=args.experiments_dir,
            as_json=args.json,
        )
    elif args.subcommand == "verify":
        code = cmd_verify(
            experiment_id=args.experiment_id,
            target_exp_id=args.target,
            base_dir=args.experiments_dir,
            as_json=args.json,
        )
    elif args.subcommand == "create":
        code = cmd_create(
            candles_path=args.candles,
            name=args.name,
            base_dir=args.experiments_dir,
            as_json=args.json,
        )
    else:
        parser.print_help()
        code = 1

    sys.exit(code)


if __name__ == "__main__":
    main()
