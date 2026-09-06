"""Command line interface for evaluating risk rules on strategy signals."""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from adaptive_trading.data.readers.csv_reader import CSVMarketDataReader
from adaptive_trading.data.validators.market_data import MarketDataValidator
from adaptive_trading.domain.market import Candle
from adaptive_trading.portfolio.state import PortfolioManager
from adaptive_trading.risk.config import RiskConfig
from adaptive_trading.risk.engine import RiskEngine
from adaptive_trading.risk.models import RiskDecision
from adaptive_trading.strategy.models import SignalAction, TradingSignal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_signals_from_csv(file_path: Path | str) -> list[TradingSignal]:
    """Load TradingSignal objects from CSV."""
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
    """Load and validate Candle objects from CSV."""
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


def run_risk_evaluation(
    signals_file: Path | str,
    candles_file: Path | str,
    initial_capital: float = 100_000.0,
    max_position_pct: float = 0.20,
    max_daily_loss_pct: float = 0.02,
    max_drawdown_pct: float = 0.10,
    max_open_positions: int = 5,
    fixed_quantity: float = 1.0,
    output_file: Path | str | None = None,
) -> int:
    """Run risk engine evaluation on signals and market candles."""
    try:
        signals = load_signals_from_csv(signals_file)
        candles = load_candles_from_csv(candles_file)
    except Exception as exc:
        logger.error("Failed to load inputs for risk evaluation: %s", exc)
        return 1

    config = RiskConfig(
        max_position_pct=max_position_pct,
        max_daily_loss_pct=max_daily_loss_pct,
        max_drawdown_pct=max_drawdown_pct,
        max_open_positions=max_open_positions,
        fixed_quantity=fixed_quantity,
    )
    engine = RiskEngine(config=config)
    portfolio = PortfolioManager(initial_capital=initial_capital)

    sorted_candles = sorted(candles, key=lambda c: c.timestamp)
    sorted_signals = sorted(signals, key=lambda s: s.timestamp)
    candle_map = {c.timestamp: c for c in sorted_candles}

    decisions: list[RiskDecision] = []

    for sig in sorted_signals:
        candle = candle_map.get(sig.timestamp)
        price = float(candle.close) if candle else 100.0

        if candle:
            portfolio.mark_to_market(
                timestamp=candle.timestamp,
                current_prices={candle.symbol: float(candle.close)},
            )

        state = portfolio.get_state(timestamp=sig.timestamp)
        decision = engine.evaluate(
            signal=sig,
            portfolio_state=state,
            current_price=price,
            timestamp=sig.timestamp,
        )
        decisions.append(decision)

    approved_count = sum(1 for d in decisions if d.approved)
    rejected_count = len(decisions) - approved_count

    print("=" * 60)
    print("      ADAPTIVE TRADING PLATFORM - RISK ENGINE AUDIT         ")
    print("=" * 60)
    print(f"Total Signals Evaluated: {len(decisions)}")
    print(f"Approved Trades        : {approved_count}")
    print(f"Rejected Trades        : {rejected_count}")
    print("-" * 60)

    for i, d in enumerate(decisions[:10], start=1):
        status = "APPROVED" if d.approved else "REJECTED"
        print(
            f"[{i:02d}] {d.timestamp.strftime('%Y-%m-%d %H:%M')} | "
            f"{d.symbol:<6} | {d.action.value:<5} | Qty: {d.approved_quantity:.2f} | "
            f"{status:<8} | {d.reason}"
        )

    if len(decisions) > 10:
        print(f"... and {len(decisions) - 10} more decisions.")

    print("-" * 60)

    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        decisions_data = [
            {
                "decision_id": d.decision_id,
                "timestamp": d.timestamp.isoformat(),
                "symbol": d.symbol,
                "action": d.action.value,
                "requested_quantity": d.requested_quantity,
                "approved_quantity": d.approved_quantity,
                "approved": d.approved,
                "reason": d.reason,
            }
            for d in decisions
        ]
        pd.DataFrame(decisions_data).to_csv(out_path, index=False)
        print(f"Decisions saved to: {out_path}")

    print("=" * 60)
    return 0


def main() -> None:
    """CLI entrypoint for risk evaluation."""
    parser = argparse.ArgumentParser(
        description="Adaptive Trading Platform Risk Engine CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate risk rules on trading signals"
    )
    eval_parser.add_argument(
        "--signals",
        "-s",
        type=str,
        required=True,
        help="Path to strategy signals CSV",
    )
    eval_parser.add_argument(
        "--candles",
        "-c",
        type=str,
        required=True,
        help="Path to historical candles CSV",
    )
    eval_parser.add_argument(
        "--initial-capital",
        type=float,
        default=100_000.0,
        help="Starting capital for portfolio evaluation",
    )
    eval_parser.add_argument(
        "--max-position-pct",
        type=float,
        default=0.20,
        help="Max position percentage (default: 0.20)",
    )
    eval_parser.add_argument(
        "--max-daily-loss-pct",
        type=float,
        default=0.02,
        help="Max daily loss percentage (default: 0.02)",
    )
    eval_parser.add_argument(
        "--max-drawdown-pct",
        type=float,
        default=0.10,
        help="Max drawdown percentage (default: 0.10)",
    )
    eval_parser.add_argument(
        "--max-open-positions",
        type=int,
        default=5,
        help="Max concurrent open positions (default: 5)",
    )
    eval_parser.add_argument(
        "--fixed-quantity",
        type=float,
        default=1.0,
        help="Fixed baseline trade quantity (default: 1.0)",
    )
    eval_parser.add_argument(
        "--output-file",
        "-o",
        type=str,
        default=None,
        help="Output CSV file path for risk decisions",
    )

    args = parser.parse_args()

    if args.command == "evaluate":
        code = run_risk_evaluation(
            signals_file=args.signals,
            candles_file=args.candles,
            initial_capital=args.initial_capital,
            max_position_pct=args.max_position_pct,
            max_daily_loss_pct=args.max_daily_loss_pct,
            max_drawdown_pct=args.max_drawdown_pct,
            max_open_positions=args.max_open_positions,
            fixed_quantity=args.fixed_quantity,
            output_file=args.output_file,
        )
    else:
        parser.print_help()
        code = 1

    sys.exit(code)


if __name__ == "__main__":
    main()
