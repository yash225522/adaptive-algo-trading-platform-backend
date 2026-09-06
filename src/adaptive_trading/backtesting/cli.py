"""Command line interface for historical strategy backtesting."""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from adaptive_trading.backtesting.config import BacktestConfig
from adaptive_trading.backtesting.engine import BacktestEngine
from adaptive_trading.data.readers.csv_reader import CSVMarketDataReader
from adaptive_trading.data.validators.market_data import MarketDataValidator
from adaptive_trading.domain.market import Candle
from adaptive_trading.strategy.models import SignalAction, TradingSignal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_signals_from_csv(file_path: Path | str) -> list[TradingSignal]:
    """Load TradingSignal objects from a generated signals CSV file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Signals file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Signals file is empty: {path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    signals: list[TradingSignal] = []

    for _, row in df.iterrows():
        ts: datetime = row["timestamp"].to_pydatetime()
        signals.append(
            TradingSignal(
                signal_id=str(row.get("signal_id", "")),
                timestamp=ts,
                symbol=str(row["symbol"]),
                action=SignalAction(str(row["action"])),
                confidence=float(row.get("confidence", 0.0)),
                strategy_name=str(row.get("strategy_name", "")),
                strategy_version=str(row.get("strategy_version", "")),
                reason=str(row.get("reason", "")),
                model_name=str(row.get("model_name", "")),
                model_version=str(row.get("model_version", "")),
            )
        )

    return signals


def load_candles_from_csv(file_path: Path | str) -> list[Candle]:
    """Load and validate Candle objects from a market data CSV file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Candles file not found: {path}")

    reader = CSVMarketDataReader()
    raw_records = reader.read_records(path)
    validator = MarketDataValidator()
    valid_candles, _ = validator.validate_batch(raw_records)

    if not valid_candles:
        raise ValueError(f"No valid candles found in {path}")

    return valid_candles


def run_backtest_pipeline(
    candles_file: Path | str,
    signals_file: Path | str,
    initial_capital: float = 100_000.0,
    commission_bps: float = 3.0,
    slippage_bps: float = 5.0,
    quantity: float = 1.0,
    output_dir: Path | str | None = None,
    save_artifacts: bool = True,
) -> int:
    """Run full backtesting pipeline and print formatted performance metrics."""
    try:
        candles = load_candles_from_csv(candles_file)
        signals = load_signals_from_csv(signals_file)
    except Exception as exc:
        logger.error("Failed to load backtest inputs: %s", exc)
        return 1

    config = BacktestConfig(
        initial_capital=initial_capital,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        fixed_quantity=quantity,
    )
    engine = BacktestEngine(config=config)

    try:
        base_dir = output_dir or "artifacts/backtests"
        result = engine.run(
            candles=candles,
            signals=signals,
            save_artifacts=save_artifacts,
            artifacts_base_dir=base_dir,
        )
    except Exception as exc:
        logger.error("Backtest simulation failed: %s", exc)
        return 1

    m = result.metrics
    pf_str = f"{m.profit_factor:.2f}" if m.profit_factor is not None else "N/A"

    print("=" * 60)
    print("      ADAPTIVE TRADING PLATFORM - BACKTEST RESULTS          ")
    print("=" * 60)
    print(f"Strategy         : {result.strategy_name} ({result.strategy_version})")
    print(f"Model            : {result.model_name} ({result.model_version})")
    start_str = (
        result.start_timestamp.strftime("%Y-%m-%d %H:%M")
        if result.start_timestamp
        else "N/A"
    )
    end_str = (
        result.end_timestamp.strftime("%Y-%m-%d %H:%M")
        if result.end_timestamp
        else "N/A"
    )
    print(f"Period           : {start_str} to {end_str}")
    print("-" * 60)
    print(f"Initial Capital  : {m.initial_capital:,.2f}")
    print(f"Final Equity     : {m.final_equity:,.2f}")
    print(f"Net P&L          : {m.net_pnl:+,.2f}")
    print(f"Total Return     : {m.total_return_pct:+.2f}%")
    print(f"Max Drawdown     : {m.max_drawdown_abs:,.2f} ({m.max_drawdown_pct:.2f}%)")
    print("-" * 60)
    print(f"Total Trades     : {m.total_trades}")
    print(f"Winning / Losing : {m.winning_trades} / {m.losing_trades}")
    print(f"Win Rate         : {m.win_rate:.1f}%")
    print(f"Profit Factor    : {pf_str}")
    print(f"Avg Trade P&L    : {m.average_trade_pnl:+,.2f}")
    print(f"Largest Win/Loss : {m.largest_winner:+,.2f} / {m.largest_loser:+,.2f}")
    print("-" * 60)
    if save_artifacts:
        print(f"Artifacts Saved  : artifacts/backtests/{result.backtest_id}")
    print("=" * 60)
    return 0


def main() -> None:
    """CLI entrypoint for backtesting execution."""
    parser = argparse.ArgumentParser(description="Adaptive Trading Backtesting Engine")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    run_parser = subparsers.add_parser(
        "run", help="Run backtesting simulation on candles and signals"
    )
    run_parser.add_argument(
        "--candles",
        "-c",
        type=str,
        required=True,
        help="Path to CSV file containing historical candles",
    )
    run_parser.add_argument(
        "--signals",
        "-s",
        type=str,
        required=True,
        help="Path to CSV file containing trading signals",
    )
    run_parser.add_argument(
        "--initial-capital",
        type=float,
        default=100_000.0,
        help="Initial capital (default: 100,000.0)",
    )
    run_parser.add_argument(
        "--commission-bps",
        type=float,
        default=3.0,
        help="Commission in basis points (default: 3.0)",
    )
    run_parser.add_argument(
        "--slippage-bps",
        type=float,
        default=5.0,
        help="Slippage in basis points (default: 5.0)",
    )
    run_parser.add_argument(
        "--quantity",
        "-q",
        type=float,
        default=1.0,
        help="Fixed trade quantity (default: 1.0)",
    )
    run_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for backtest artifacts",
    )
    run_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save backtest artifacts to disk",
    )

    args = parser.parse_args()

    if args.command == "run":
        exit_code = run_backtest_pipeline(
            candles_file=args.candles,
            signals_file=args.signals,
            initial_capital=args.initial_capital,
            commission_bps=args.commission_bps,
            slippage_bps=args.slippage_bps,
            quantity=args.quantity,
            output_dir=args.output_dir,
            save_artifacts=not args.no_save,
        )
    else:
        parser.print_help()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
