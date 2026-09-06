"""Command line interface for evaluating strategies and generating trading signals."""

import argparse
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from adaptive_trading.strategy.config import StrategyConfig
from adaptive_trading.strategy.engine import StrategyEngine
from adaptive_trading.strategy.models import (
    SignalStatistics,
    StrategyPrediction,
    TradingSignal,
)
from adaptive_trading.strategy.rules import ProbabilityStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_predictions_from_csv(
    file_path: Path | str,
) -> list[StrategyPrediction]:
    """Load and parse predictions from a CSV file into StrategyPredictions."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Predictions file is empty: {path}")

    required_cols = {"probability_up"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Predictions file missing required columns. Expected {required_cols}, "
            f"found {list(df.columns)}"
        )

    # Convert timestamps
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    else:
        df["timestamp"] = [datetime.now(timezone.utc) for _ in range(len(df))]

    predictions: list[StrategyPrediction] = []
    for _, row in df.iterrows():
        ts: datetime = row["timestamp"].to_pydatetime()
        sym = str(row.get("symbol", "UNKNOWN"))
        prob = float(row["probability_up"])
        pred_cls = str(row.get("predicted_label", row.get("predicted_class", "")))
        model_nm = str(row.get("model_name", "logistic_regression"))
        model_ver = str(row.get("model_version", "v1"))

        predictions.append(
            StrategyPrediction(
                timestamp=ts,
                symbol=sym,
                predicted_class=pred_cls,
                probability_up=prob,
                model_name=model_nm,
                model_version=model_ver,
            )
        )

    return predictions


def save_signals_to_csv(
    signals: list[TradingSignal],
    output_dir: Path | str,
) -> Path:
    """Persist generated TradingSignal records to a CSV file."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = out_dir / "signals.csv"

    records = [s.model_dump() for s in signals]
    df = pd.DataFrame(records)
    df.to_csv(csv_file, index=False)
    logger.info("Saved %d trading signals to %s", len(signals), csv_file)
    return csv_file


def run_strategy_generation(
    predictions_file: Path | str,
    long_threshold: float = 0.60,
    short_threshold: float = 0.40,
    strategy_name: str = "probability_threshold",
    strategy_version: str = "v1",
    output_dir: Path | str | None = None,
    save_artifact: bool = True,
) -> int:
    """Execute strategy evaluation on predictions and print summary statistics."""
    pred_path = Path(predictions_file)
    if not pred_path.exists():
        logger.error("Predictions file not found: %s", pred_path)
        return 1

    try:
        predictions = load_predictions_from_csv(pred_path)
    except Exception as exc:
        logger.error("Failed to load predictions: %s", exc)
        return 1

    config = StrategyConfig(
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        long_probability_threshold=long_threshold,
        short_probability_threshold=short_threshold,
    )
    strategy = ProbabilityStrategy(config=config)
    engine = StrategyEngine(strategy=strategy)

    try:
        signals = engine.generate_signals(predictions)
    except Exception as exc:
        logger.error("Signal generation failed: %s", exc)
        return 1

    stats: SignalStatistics = engine.calculate_statistics(signals)

    print("=" * 60)
    print("      ADAPTIVE TRADING PLATFORM - STRATEGY ENGINE           ")
    print("=" * 60)
    print(f"Strategy Name    : {strategy.name}")
    print(f"Strategy Version : {strategy.version}")
    print(f"Long Threshold   : {long_threshold:.4f}")
    print(f"Short Threshold  : {short_threshold:.4f}")
    print("-" * 60)
    print(f"Total Predictions: {stats.total_predictions:,}")
    print(f"LONG Signals     : {stats.long_signals:,} ({stats.long_percentage:.1f}%)")
    print(f"SHORT Signals    : {stats.short_signals:,} ({stats.short_percentage:.1f}%)")
    print(
        f"NO_TRADE Signals : {stats.no_trade_signals:,} "
        f"({stats.no_trade_percentage:.1f}%)"
    )
    print("-" * 60)

    if save_artifact and signals:
        run_id = (
            f"strat_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_"
            f"{uuid.uuid4().hex[:6]}"
        )
        target_dir = Path(output_dir or f"artifacts/strategy/{run_id}")
        saved_file = save_signals_to_csv(signals, target_dir)
        print(f"Artifact Saved   : {saved_file}")

    print("=" * 60)
    return 0


def main() -> None:
    """CLI entrypoint for Strategy Engine."""
    parser = argparse.ArgumentParser(
        description="Adaptive Trading Strategy Engine - Signal Generation"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    gen_parser = subparsers.add_parser(
        "generate", help="Generate trading signals from ML predictions"
    )
    gen_parser.add_argument(
        "--file",
        "-f",
        type=str,
        required=True,
        help="Path to CSV file containing ML predictions",
    )
    gen_parser.add_argument(
        "--long-threshold",
        type=float,
        default=0.60,
        help="Probability threshold for LONG signals (default: 0.60)",
    )
    gen_parser.add_argument(
        "--short-threshold",
        type=float,
        default=0.40,
        help="Probability threshold for SHORT signals (default: 0.40)",
    )
    gen_parser.add_argument(
        "--strategy-name",
        type=str,
        default="probability_threshold",
        help="Strategy name (default: probability_threshold)",
    )
    gen_parser.add_argument(
        "--strategy-version",
        type=str,
        default="v1",
        help="Strategy version (default: v1)",
    )
    gen_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory to save signals artifact",
    )
    gen_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save signal artifact to disk",
    )

    args = parser.parse_args()

    if args.command == "generate":
        exit_code = run_strategy_generation(
            predictions_file=args.file,
            long_threshold=args.long_threshold,
            short_threshold=args.short_threshold,
            strategy_name=args.strategy_name,
            strategy_version=args.strategy_version,
            output_dir=args.output_dir,
            save_artifact=not args.no_save,
        )
    else:
        parser.print_help()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
