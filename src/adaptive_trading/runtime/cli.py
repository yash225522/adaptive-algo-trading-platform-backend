"""Command line interface for the Paper Trading Event Loop and Replay Runner."""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib

from adaptive_trading.execution.config import ExecutionConfig
from adaptive_trading.execution.paper_broker import (
    PaperBroker,
    StaticMarketDataProvider,
)
from adaptive_trading.execution.service import ExecutionService
from adaptive_trading.features.pipeline import FeaturePipeline
from adaptive_trading.portfolio.state import PortfolioManager
from adaptive_trading.risk.config import RiskConfig
from adaptive_trading.risk.engine import RiskEngine
from adaptive_trading.runtime.config import RuntimeConfig, RuntimeMode
from adaptive_trading.runtime.event_loop import (
    EventLoop,
    MLPredictionService,
    PredictionService,
    SimplePredictionService,
)
from adaptive_trading.runtime.runner import HistoricalDataProvider, ReplayRunner
from adaptive_trading.strategy.engine import StrategyEngine
from adaptive_trading.strategy.rules import ProbabilityStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_replay(
    candles_file: Path,
    model_artifact_dir: Path | None = None,
    initial_cash: float = 100_000.0,
    warmup_period: int = 20,
    output_dir: Path = Path("artifacts/runtime"),
) -> int:
    """Run historical market event replay through EventLoop."""
    if not candles_file.is_file():
        print(f"Error: Candles file '{candles_file}' does not exist", file=sys.stderr)
        return 1

    config = RuntimeConfig(
        mode=RuntimeMode.REPLAY,
        initial_cash=initial_cash,
        warmup_period=warmup_period,
    )

    # 1. Setup Prediction Service
    prediction_service: PredictionService | None = None
    if model_artifact_dir is not None and model_artifact_dir.is_dir():
        try:
            model_path = model_artifact_dir / "model.joblib"
            meta_path = model_artifact_dir / "metadata.json"
            if model_path.is_file():
                bundle = joblib.load(model_path)
                meta = {}
                if meta_path.is_file():
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                prediction_service = MLPredictionService(
                    model=bundle["model"],
                    preprocessor=bundle["preprocessor"],
                    model_version=meta.get("model_name", "v1"),
                )
                logger.info("Loaded ML model artifact from %s", model_artifact_dir)
            else:
                prediction_service = SimplePredictionService(fixed_probability_up=0.65)
        except Exception as exc:
            logger.warning(
                "Could not load ML artifact (%s), using SimplePredictionService: %s",
                model_artifact_dir,
                exc,
            )
            prediction_service = SimplePredictionService(fixed_probability_up=0.65)
    else:
        prediction_service = SimplePredictionService(fixed_probability_up=0.65)

    # 2. Setup Components
    feature_pipeline = FeaturePipeline()
    strategy_engine = StrategyEngine(strategy=ProbabilityStrategy())
    risk_engine = RiskEngine(
        config=RiskConfig(fixed_quantity=1.0, max_position_pct=0.20)
    )
    portfolio_manager = PortfolioManager(initial_capital=initial_cash)
    paper_broker = PaperBroker(
        config=ExecutionConfig(initial_cash=initial_cash),
        market_data_provider=StaticMarketDataProvider(),
    )
    execution_service = ExecutionService(broker=paper_broker)

    event_loop = EventLoop(
        config=config,
        feature_pipeline=feature_pipeline,
        prediction_service=prediction_service,
        strategy_engine=strategy_engine,
        risk_engine=risk_engine,
        execution_service=execution_service,
        portfolio_manager=portfolio_manager,
    )

    runner = ReplayRunner(event_loop=event_loop, config=config)
    provider = HistoricalDataProvider(file_path=candles_file, sort=True)

    out_run_dir = runner.run(provider=provider, artifacts_base_dir=output_dir)

    stats = event_loop.state.stats
    account = paper_broker.get_account()

    print("=" * 60)
    print("      ADAPTIVE TRADING PLATFORM - REPLAY SUMMARY            ")
    print("=" * 60)
    print(f"Mode             : {config.mode.value}")
    print(f"Candles File     : {candles_file}")
    print(f"Artifacts Saved  : {out_run_dir}")
    print("-" * 60)
    print(f"Events Processed : {stats.market_events_processed}")
    print(f"Predictions      : {stats.predictions_generated}")
    print(f"Signals (Gen/Rej): {stats.signals_generated} / {stats.signals_rejected}")
    print(f"Orders (Sub/Fill): {stats.orders_submitted} / {stats.orders_filled}")
    print(f"Orders Rejected  : {stats.orders_rejected}")
    print(f"Runtime Errors   : {stats.errors}")
    print("-" * 60)
    print(f"Initial Cash     : {initial_cash:,.2f}")
    print(f"Final Cash       : {account.cash:,.2f}")
    print(f"Final Equity     : {account.equity:,.2f}")
    print(f"Realized P&L     : {account.realized_pnl:+,.2f}")
    print(f"Unrealized P&L   : {account.unrealized_pnl:+,.2f}")
    print("=" * 60)
    return 0


def cmd_status(run_dir: Path) -> int:
    """Display summary of a past runtime execution run."""
    if not run_dir.is_dir():
        print(f"Error: Run directory '{run_dir}' not found", file=sys.stderr)
        return 1

    stats_file = run_dir / "runtime_stats.json"
    state_file = run_dir / "final_state.json"
    meta_file = run_dir / "metadata.json"

    if not stats_file.is_file() or not state_file.is_file():
        print(f"Error: Missing summary files in '{run_dir}'", file=sys.stderr)
        return 1

    with open(stats_file, "r", encoding="utf-8") as f:
        stats = json.load(f)
    with open(state_file, "r", encoding="utf-8") as f:
        state = json.load(f)
    meta = {}
    if meta_file.is_file():
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

    print("=" * 60)
    print("      ADAPTIVE TRADING PLATFORM - RUNTIME STATUS            ")
    print("=" * 60)
    print(f"Run ID           : {meta.get('run_id', run_dir.name)}")
    print(f"Mode             : {meta.get('mode', 'UNKNOWN')}")
    print(f"Data Span        : {meta.get('data_start')} to {meta.get('data_end')}")
    print("-" * 60)
    print(f"Events Processed : {stats.get('market_events_processed', 0)}")
    print(f"Predictions      : {stats.get('predictions_generated', 0)}")
    print(
        f"Signals (Gen/Rej): {stats.get('signals_generated', 0)} / "
        f"{stats.get('signals_rejected', 0)}"
    )
    print(
        f"Orders (Sub/Fill): {stats.get('orders_submitted', 0)} / "
        f"{stats.get('orders_filled', 0)}"
    )
    print(f"Orders Rejected  : {stats.get('orders_rejected', 0)}")
    print(f"Errors           : {stats.get('errors', 0)}")
    print("-" * 60)
    print(f"Cash Balance     : {state.get('cash', 0.0):,.2f}")
    print(f"Total Equity     : {state.get('equity', 0.0):,.2f}")
    print(f"Realized P&L     : {state.get('realized_pnl', 0.0):+,.2f}")
    print(f"Unrealized P&L   : {state.get('unrealized_pnl', 0.0):+,.2f}")
    print("=" * 60)
    return 0


def main() -> None:
    """Entry point for runtime CLI."""
    parser = argparse.ArgumentParser(
        description="Adaptive Trading Platform Runtime & Event Loop CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # 1. replay
    replay_parser = subparsers.add_parser(
        "replay", help="Run historical market event replay"
    )
    replay_parser.add_argument(
        "--candles",
        "-c",
        type=Path,
        required=True,
        help="Path to candles CSV file",
    )
    replay_parser.add_argument(
        "--model-dir",
        "-m",
        type=Path,
        default=None,
        help="Path to trained model artifact directory",
    )
    replay_parser.add_argument(
        "--initial-cash",
        type=float,
        default=100_000.0,
        help="Initial simulation cash balance",
    )
    replay_parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup period (candles)",
    )
    replay_parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("artifacts/runtime"),
        help="Destination directory for artifacts",
    )

    # 2. status
    status_parser = subparsers.add_parser(
        "status", help="Inspect runtime status from run directory"
    )
    status_parser.add_argument(
        "--run-dir",
        "-r",
        type=Path,
        required=True,
        help="Path to run output directory",
    )

    args = parser.parse_args()

    if args.command == "replay":
        code = cmd_replay(
            candles_file=args.candles,
            model_artifact_dir=args.model_dir,
            initial_cash=args.initial_cash,
            warmup_period=args.warmup,
            output_dir=args.output_dir,
        )
    elif args.command == "status":
        code = cmd_status(run_dir=args.run_dir)
    else:
        parser.print_help()
        code = 1

    sys.exit(code)


if __name__ == "__main__":
    main()
