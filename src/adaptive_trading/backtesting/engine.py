"""Deterministic chronological backtest engine with Risk Engine integration."""

import json
import logging
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from adaptive_trading.backtesting.config import BacktestConfig
from adaptive_trading.backtesting.exceptions import InsufficientDataError
from adaptive_trading.backtesting.execution import SimulatedExecutionHandler
from adaptive_trading.backtesting.metrics import calculate_performance_metrics
from adaptive_trading.backtesting.models import (
    BacktestOrder,
    BacktestResult,
    PerformanceMetrics,
    PositionSide,
)
from adaptive_trading.backtesting.portfolio import PortfolioTracker
from adaptive_trading.domain.market import Candle
from adaptive_trading.domain.trading import OrderSide
from adaptive_trading.portfolio.models import PositionSide as PortfolioPositionSide
from adaptive_trading.portfolio.state import PortfolioManager
from adaptive_trading.risk.config import RiskConfig
from adaptive_trading.risk.engine import RiskEngine
from adaptive_trading.risk.models import RiskDecision
from adaptive_trading.strategy.models import SignalAction, TradingSignal

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Simulates trading strategies across chronological historical candles."""

    def __init__(
        self,
        config: BacktestConfig | None = None,
        execution_handler: SimulatedExecutionHandler | None = None,
        risk_engine: RiskEngine | None = None,
    ) -> None:
        self.config = config or BacktestConfig()
        self.execution_handler = execution_handler or SimulatedExecutionHandler(
            config=self.config
        )
        self.risk_engine = risk_engine or RiskEngine(
            config=RiskConfig(fixed_quantity=self.config.fixed_quantity)
        )

    def run(
        self,
        candles: Sequence[Candle],
        signals: Sequence[TradingSignal],
        backtest_id: str | None = None,
        save_artifacts: bool = True,
        artifacts_base_dir: Path | str = "artifacts/backtests",
    ) -> BacktestResult:
        """Run full backtesting simulation.

        Args:
            candles: Sequence of historical market candles.
            signals: Sequence of generated strategy trading signals.
            backtest_id: Optional unique run identifier.
            save_artifacts: Whether to write result files to disk.
            artifacts_base_dir: Directory where backtest runs are persisted.

        Returns:
            BacktestResult: Complete result with trades, equity curve, and metrics.

        Raises:
            InsufficientDataError: If candles or signals are missing.
        """
        if not candles:
            raise InsufficientDataError("Cannot run backtest with empty candles list")
        if not signals:
            raise InsufficientDataError("Cannot run backtest with empty signals list")

        sorted_candles = sorted(candles, key=lambda c: c.timestamp)
        sorted_signals = sorted(signals, key=lambda s: s.timestamp)

        signal_map: dict[datetime, TradingSignal] = {
            s.timestamp: s for s in sorted_signals
        }

        portfolio = PortfolioTracker(config=self.config)
        portfolio_manager = PortfolioManager(
            initial_capital=self.config.initial_capital
        )
        entry_bar_indices: dict[str, int] = {}
        risk_decisions: list[RiskDecision] = []

        logger.info(
            "Starting backtest simulation: %d candles, %d signals, capital=%.2f",
            len(sorted_candles),
            len(sorted_signals),
            self.config.initial_capital,
        )

        for i, candle in enumerate(sorted_candles):
            symbol = candle.symbol

            # Check if previous candle generated a signal
            # that executes at current candle open
            if i > 0:
                prev_candle = sorted_candles[i - 1]
                pending_signal = signal_map.get(prev_candle.timestamp)

                if pending_signal and pending_signal.symbol == symbol:
                    # Update portfolio state as of prev_candle close
                    state_before_signal = portfolio_manager.get_state(
                        prev_candle.timestamp
                    )
                    risk_decision = self.risk_engine.evaluate(
                        signal=pending_signal,
                        portfolio_state=state_before_signal,
                        current_price=float(prev_candle.close),
                        timestamp=prev_candle.timestamp,
                    )
                    risk_decisions.append(risk_decision)

                    if risk_decision.approved and risk_decision.approved_quantity > 0:
                        self._process_signal_execution(
                            signal=pending_signal,
                            approved_quantity=risk_decision.approved_quantity,
                            execution_candle=candle,
                            portfolio=portfolio,
                            portfolio_manager=portfolio_manager,
                            current_bar_index=i,
                            entry_bar_indices=entry_bar_indices,
                        )
                    else:
                        logger.debug(
                            "Signal %s at %s rejected by risk engine: %s",
                            pending_signal.signal_id,
                            prev_candle.timestamp,
                            risk_decision.reason,
                        )

            # Mark portfolio to market at current candle close
            portfolio.mark_to_market(
                timestamp=candle.timestamp,
                current_prices={symbol: float(candle.close)},
            )
            portfolio_manager.mark_to_market(
                timestamp=candle.timestamp,
                current_prices={symbol: float(candle.close)},
            )

        metrics: PerformanceMetrics = calculate_performance_metrics(
            trades=portfolio.trades,
            equity_curve=portfolio.equity_curve,
            initial_capital=self.config.initial_capital,
        )

        first_signal = sorted_signals[0]
        bt_id = backtest_id or (
            f"bt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_"
            f"{uuid.uuid4().hex[:6]}"
        )

        result = BacktestResult(
            backtest_id=bt_id,
            start_timestamp=sorted_candles[0].timestamp,
            end_timestamp=sorted_candles[-1].timestamp,
            strategy_name=first_signal.strategy_name,
            strategy_version=first_signal.strategy_version,
            model_name=first_signal.model_name,
            model_version=first_signal.model_version,
            config=self.config.model_dump(),
            metrics=metrics,
            trades=portfolio.trades,
            equity_curve=portfolio.equity_curve,
            created_at=datetime.now(timezone.utc),
        )

        if save_artifacts:
            self._save_backtest_artifacts(
                result=result,
                risk_decisions=risk_decisions,
                artifacts_base_dir=artifacts_base_dir,
            )

        return result

    def _process_signal_execution(
        self,
        signal: TradingSignal,
        approved_quantity: float,
        execution_candle: Candle,
        portfolio: PortfolioTracker,
        portfolio_manager: PortfolioManager,
        current_bar_index: int,
        entry_bar_indices: dict[str, int],
    ) -> None:
        """Handle signal execution, position transitions, and reversals."""
        symbol = signal.symbol
        pos = portfolio.get_position(symbol)
        qty = approved_quantity

        if signal.action == SignalAction.LONG:
            if pos.side == PositionSide.SHORT:
                # 1. Close SHORT position (BUY to cover)
                exit_order = BacktestOrder(
                    timestamp=execution_candle.timestamp,
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=pos.quantity,
                    signal_id=signal.signal_id,
                )
                exit_fill = self.execution_handler.simulate_fill(
                    exit_order, execution_candle
                )
                bars_held = current_bar_index - entry_bar_indices.get(
                    symbol, current_bar_index - 1
                )
                portfolio.close_position(
                    symbol, exit_fill, signal=signal, holding_bars=bars_held
                )
                portfolio_manager.close_position(
                    symbol=symbol,
                    exit_price=exit_fill.price,
                    exit_timestamp=execution_candle.timestamp,
                    exit_commission=exit_fill.commission,
                )

                # 2. Open LONG position (BUY)
                entry_order = BacktestOrder(
                    timestamp=execution_candle.timestamp,
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=qty,
                    signal_id=signal.signal_id,
                )
                entry_fill = self.execution_handler.simulate_fill(
                    entry_order, execution_candle
                )
                portfolio.open_position(entry_fill, side=PositionSide.LONG)
                portfolio_manager.open_position(
                    symbol=symbol,
                    side=PortfolioPositionSide.LONG,
                    quantity=qty,
                    price=entry_fill.price,
                    timestamp=execution_candle.timestamp,
                    commission=entry_fill.commission,
                )
                entry_bar_indices[symbol] = current_bar_index

            elif pos.side == PositionSide.FLAT:
                # Open LONG position (BUY)
                entry_order = BacktestOrder(
                    timestamp=execution_candle.timestamp,
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=qty,
                    signal_id=signal.signal_id,
                )
                entry_fill = self.execution_handler.simulate_fill(
                    entry_order, execution_candle
                )
                portfolio.open_position(entry_fill, side=PositionSide.LONG)
                portfolio_manager.open_position(
                    symbol=symbol,
                    side=PortfolioPositionSide.LONG,
                    quantity=qty,
                    price=entry_fill.price,
                    timestamp=execution_candle.timestamp,
                    commission=entry_fill.commission,
                )
                entry_bar_indices[symbol] = current_bar_index

        elif signal.action == SignalAction.SHORT:
            if pos.side == PositionSide.LONG:
                # 1. Close LONG position (SELL to close)
                exit_order = BacktestOrder(
                    timestamp=execution_candle.timestamp,
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=pos.quantity,
                    signal_id=signal.signal_id,
                )
                exit_fill = self.execution_handler.simulate_fill(
                    exit_order, execution_candle
                )
                bars_held = current_bar_index - entry_bar_indices.get(
                    symbol, current_bar_index - 1
                )
                portfolio.close_position(
                    symbol, exit_fill, signal=signal, holding_bars=bars_held
                )
                portfolio_manager.close_position(
                    symbol=symbol,
                    exit_price=exit_fill.price,
                    exit_timestamp=execution_candle.timestamp,
                    exit_commission=exit_fill.commission,
                )

                # 2. Open SHORT position (SELL)
                entry_order = BacktestOrder(
                    timestamp=execution_candle.timestamp,
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=qty,
                    signal_id=signal.signal_id,
                )
                entry_fill = self.execution_handler.simulate_fill(
                    entry_order, execution_candle
                )
                portfolio.open_position(entry_fill, side=PositionSide.SHORT)
                portfolio_manager.open_position(
                    symbol=symbol,
                    side=PortfolioPositionSide.SHORT,
                    quantity=qty,
                    price=entry_fill.price,
                    timestamp=execution_candle.timestamp,
                    commission=entry_fill.commission,
                )
                entry_bar_indices[symbol] = current_bar_index

            elif pos.side == PositionSide.FLAT:
                # Open SHORT position (SELL)
                entry_order = BacktestOrder(
                    timestamp=execution_candle.timestamp,
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=qty,
                    signal_id=signal.signal_id,
                )
                entry_fill = self.execution_handler.simulate_fill(
                    entry_order, execution_candle
                )
                portfolio.open_position(entry_fill, side=PositionSide.SHORT)
                portfolio_manager.open_position(
                    symbol=symbol,
                    side=PortfolioPositionSide.SHORT,
                    quantity=qty,
                    price=entry_fill.price,
                    timestamp=execution_candle.timestamp,
                    commission=entry_fill.commission,
                )
                entry_bar_indices[symbol] = current_bar_index

    def _save_backtest_artifacts(
        self,
        result: BacktestResult,
        risk_decisions: list[RiskDecision],
        artifacts_base_dir: Path | str,
    ) -> Path:
        """Persist metadata, metrics, trades, risk decisions, and equity."""
        out_dir = Path(artifacts_base_dir) / result.backtest_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. metadata.json
        metadata = {
            "backtest_id": result.backtest_id,
            "strategy_name": result.strategy_name,
            "strategy_version": result.strategy_version,
            "model_name": result.model_name,
            "model_version": result.model_version,
            "start_timestamp": (
                result.start_timestamp.isoformat() if result.start_timestamp else None
            ),
            "end_timestamp": (
                result.end_timestamp.isoformat() if result.end_timestamp else None
            ),
            "created_at": result.created_at.isoformat(),
            "config": result.config,
        }
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 2. metrics.json
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(result.metrics.model_dump(), f, indent=2)

        # 3. trades.csv
        trades_df = pd.DataFrame([t.model_dump() for t in result.trades])
        trades_df.to_csv(out_dir / "trades.csv", index=False)

        # 4. equity_curve.csv
        equity_df = pd.DataFrame([e.model_dump() for e in result.equity_curve])
        equity_df.to_csv(out_dir / "equity_curve.csv", index=False)

        # 5. risk_decisions.csv
        if risk_decisions:
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
                for d in risk_decisions
            ]
            pd.DataFrame(decisions_data).to_csv(
                out_dir / "risk_decisions.csv", index=False
            )

            # Also persist under artifacts/risk/<backtest_id>/
            risk_dir = Path("artifacts/risk") / result.backtest_id
            risk_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(decisions_data).to_csv(
                risk_dir / "risk_decisions.csv", index=False
            )

        logger.info("Saved backtest artifacts to %s", out_dir)
        return out_dir
