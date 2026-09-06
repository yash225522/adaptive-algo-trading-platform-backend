"""Command line interface for System Health, Metrics, and Run Tracking."""

import argparse
import json
import logging
import sys
from pathlib import Path

from adaptive_trading.observability.events import (
    filter_events,
    load_run_events,
)
from adaptive_trading.observability.health import HealthChecker
from adaptive_trading.observability.metrics import global_metrics
from adaptive_trading.observability.run_tracker import RunTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_health(as_json: bool = False) -> int:
    """Evaluate and display system health status."""
    checker = HealthChecker()
    report = checker.check_all()

    if as_json:
        print(json.dumps(report.model_dump(mode="json"), indent=2))
        return 0

    print("=" * 60)
    print("      ADAPTIVE TRADING PLATFORM - SYSTEM HEALTH             ")
    print("=" * 60)
    print(f"Overall Status   : {report.status.value}")
    print(f"Timestamp        : {report.checked_at.isoformat()}")
    print("-" * 60)
    print(f"{'Component':<18} {'Status':<12} {'Details'}")
    print("-" * 60)
    for name, comp in report.components.items():
        print(f"{name:<18} {comp.status.value:<12} {comp.message}")
    print("=" * 60)
    return 0


def cmd_metrics(as_json: bool = False) -> int:
    """Display registered operational metrics and latency summaries."""
    snap = global_metrics.snapshot()

    if as_json:
        print(json.dumps(snap, indent=2))
        return 0

    print("=" * 60)
    print("      ADAPTIVE TRADING PLATFORM - RUNTIME METRICS           ")
    print("=" * 60)
    print("COUNTERS:")
    for name, c in snap.get("counters", {}).items():
        print(f"  - {name:<32}: {c.get('total', 0)}")
    print("-" * 60)
    print("GAUGES:")
    for name, g in snap.get("gauges", {}).items():
        print(f"  - {name:<32}: {g.get('values', {})}")
    print("-" * 60)
    print("HISTOGRAMS:")
    for name, h in snap.get("histograms", {}).items():
        stats = h.get("stats", {}).get("default", {})
        print(
            f"  - {name:<32}: count={stats.get('count', 0)} "
            f"mean={stats.get('mean', 0.0):.2f}ms "
            f"p95={stats.get('p95', 0.0):.2f}ms"
        )
    print("=" * 60)
    return 0


def cmd_runs(
    runs_dir: Path = Path("artifacts/runs"),
    limit: int = 20,
    as_json: bool = False,
) -> int:
    """List recently recorded execution runs."""
    tracker = RunTracker(base_dir=runs_dir)
    runs = tracker.list_runs(limit=limit)

    if as_json:
        print(json.dumps([r.model_dump(mode="json") for r in runs], indent=2))
        return 0

    print("=" * 70)
    print("      ADAPTIVE TRADING PLATFORM - EXECUTION RUNS            ")
    print("=" * 70)
    print(f"{'RUN ID':<30} {'TYPE':<14} {'STATUS':<12} {'DURATION':<10}")
    print("-" * 70)
    for r in runs:
        dur = f"{r.duration_seconds:.2f}s" if r.duration_seconds else "N/A"
        print(f"{r.run_id:<30} {r.run_type.value:<14} {r.status.value:<12} {dur:<10}")
    print("=" * 70)
    return 0


def cmd_run_details(
    run_id: str,
    runs_dir: Path = Path("artifacts/runs"),
    as_json: bool = False,
) -> int:
    """Display detailed metrics and metadata for a specific run."""
    tracker = RunTracker(base_dir=runs_dir)
    record = tracker.get_run(run_id)

    # Fallback to runtime artifacts if not found in runs_dir
    if record is None:
        runtime_meta = Path("artifacts/runtime") / run_id / "metadata.json"
        if runtime_meta.is_file():
            with open(runtime_meta, "r", encoding="utf-8") as f:
                meta_dict = json.load(f)
            if as_json:
                print(json.dumps(meta_dict, indent=2))
                return 0

            print("=" * 60)
            print("      ADAPTIVE TRADING PLATFORM - RUN DETAILS               ")
            print("=" * 60)
            print(f"Run ID           : {meta_dict.get('run_id')}")
            print(f"Mode             : {meta_dict.get('mode')}")
            print(f"Started At       : {meta_dict.get('started_at')}")
            print(f"Ended At         : {meta_dict.get('ended_at')}")
            print(
                f"Data Span        : {meta_dict.get('data_start')} to "
                f"{meta_dict.get('data_end')}"
            )
            print("=" * 60)
            return 0

        print(f"Error: Run '{run_id}' not found", file=sys.stderr)
        return 1

    if as_json:
        print(json.dumps(record.model_dump(mode="json"), indent=2))
        return 0

    print("=" * 60)
    print("      ADAPTIVE TRADING PLATFORM - RUN DETAILS               ")
    print("=" * 60)
    print(f"Run ID           : {record.run_id}")
    print(f"Type             : {record.run_type.value}")
    print(f"Status           : {record.status.value}")
    print(f"Started At       : {record.started_at.isoformat()}")
    ended_str = record.ended_at.isoformat() if record.ended_at else "N/A"
    print(f"Ended At         : {ended_str}")
    dur = f"{record.duration_seconds:.2f}s" if record.duration_seconds else "N/A"
    print(f"Duration         : {dur}")
    print("-" * 60)
    print(
        f"Model            : {record.model_name or 'N/A'} "
        f"({record.model_version or 'N/A'})"
    )
    print(
        f"Strategy         : {record.strategy_name or 'N/A'} "
        f"({record.strategy_version or 'N/A'})"
    )
    if record.summary:
        s = record.summary
        print("-" * 60)
        print(f"Events Processed : {s.events_processed}")
        print(f"Predictions      : {s.predictions_generated}")
        print(f"Signals (Gen/Rej): {s.signals_generated} / {s.risk_rejections}")
        print(f"Orders (Sub/Fill): {s.orders_submitted} / {s.orders_filled}")
        print(f"Orders Rejected  : {s.orders_rejected}")
        print(f"Errors           : {s.errors}")
        print("-" * 60)
        print(f"Final Equity     : {s.final_equity:,.2f}")
        print(f"Net P&L          : {s.net_pnl:+,.2f}")
        print(f"Max Drawdown     : {s.max_drawdown:.2%}")
    if record.error_summary:
        print("-" * 60)
        print(f"Error Summary    : {record.error_summary}")
    print("=" * 60)
    return 0


def cmd_events(
    run_id: str,
    event_type: str | None = None,
    as_json: bool = False,
) -> int:
    """Inspect and query event records from a past run."""
    events = load_run_events(run_id)
    if event_type:
        events = filter_events(events, event_type=event_type)

    if as_json:
        print(json.dumps([e.model_dump(mode="json") for e in events], indent=2))
        return 0

    print("=" * 70)
    print(f"      EVENT TRACE FOR RUN: {run_id}")
    print("=" * 70)
    print(f"{'TIMESTAMP':<28} {'TYPE':<16} {'SYMBOL':<8} {'CORRELATION ID'}")
    print("-" * 70)
    for e in events:
        print(
            f"{e.timestamp.isoformat():<28} {e.event_type.value:<16} "
            f"{e.symbol:<8} {e.correlation_id[:8]}"
        )
    print("=" * 70)
    return 0


def create_monitoring_parser() -> argparse.ArgumentParser:
    """Create subparser tree for monitoring commands."""
    parser = argparse.ArgumentParser(
        description="Adaptive Trading Platform Observability & Monitoring CLI"
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Monitoring action")

    # health
    p_health = subparsers.add_parser("health", help="Check system health")
    p_health.add_argument("--json", action="store_true", help="JSON output")

    # metrics
    p_metrics = subparsers.add_parser("metrics", help="Display metrics")
    p_metrics.add_argument("--json", action="store_true", help="JSON output")

    # runs
    p_runs = subparsers.add_parser("runs", help="List execution runs")
    p_runs.add_argument("--limit", type=int, default=20, help="Maximum runs to display")
    p_runs.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("artifacts/runs"),
        help="Path to runs directory",
    )
    p_runs.add_argument("--json", action="store_true", help="JSON output")

    # run
    p_run = subparsers.add_parser("run", help="Display details for a run")
    p_run.add_argument("run_id", type=str, help="Run ID to inspect")
    p_run.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("artifacts/runs"),
        help="Path to runs directory",
    )
    p_run.add_argument("--json", action="store_true", help="JSON output")

    # events
    p_events = subparsers.add_parser("events", help="Query lifecycle events")
    p_events.add_argument("run_id", type=str, help="Run ID or path")
    p_events.add_argument("--type", type=str, default=None, help="Filter by EventType")
    p_events.add_argument("--json", action="store_true", help="JSON output")

    return parser


def main() -> None:
    """Entry point for monitoring CLI."""
    parser = create_monitoring_parser()
    args = parser.parse_args()

    if args.subcommand == "health":
        code = cmd_health(as_json=args.json)
    elif args.subcommand == "metrics":
        code = cmd_metrics(as_json=args.json)
    elif args.subcommand == "runs":
        code = cmd_runs(runs_dir=args.runs_dir, limit=args.limit, as_json=args.json)
    elif args.subcommand == "run":
        code = cmd_run_details(
            run_id=args.run_id, runs_dir=args.runs_dir, as_json=args.json
        )
    elif args.subcommand == "events":
        code = cmd_events(run_id=args.run_id, event_type=args.type, as_json=args.json)
    else:
        parser.print_help()
        code = 1

    sys.exit(code)


if __name__ == "__main__":
    main()
