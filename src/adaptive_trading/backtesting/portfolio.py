"""Portfolio tracking, cash accounting, position state machine, and equity curve."""

import logging
from datetime import datetime

from adaptive_trading.backtesting.config import BacktestConfig
from adaptive_trading.backtesting.models import (
    BacktestFill,
    BacktestPosition,
    BacktestTrade,
    EquityPoint,
    PositionSide,
)
from adaptive_trading.strategy.models import TradingSignal

logger = logging.getLogger(__name__)


class PortfolioTracker:
    """Maintains cash, open positions, realized/unrealized P&L, and equity history."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()
        self.initial_capital = self.config.initial_capital
        self.cash = self.initial_capital
        self.realized_pnl = 0.0
        self.positions: dict[str, BacktestPosition] = {}
        self.trades: list[BacktestTrade] = []
        self.equity_curve: list[EquityPoint] = []
        self._entry_commissions: dict[str, float] = {}

    def get_position(self, symbol: str) -> BacktestPosition:
        """Retrieve active position for symbol or return FLAT default."""
        if symbol not in self.positions:
            self.positions[symbol] = BacktestPosition(symbol=symbol)
        return self.positions[symbol]

    def close_position(
        self,
        symbol: str,
        exit_fill: BacktestFill,
        signal: TradingSignal | None = None,
        holding_bars: int = 1,
    ) -> BacktestTrade | None:
        """Close an existing LONG or SHORT position and record a completed trade."""
        pos = self.get_position(symbol)
        if pos.side == PositionSide.FLAT or pos.quantity == 0:
            return None

        entry_comm = self._entry_commissions.get(symbol, 0.0)
        total_comm = entry_comm + exit_fill.commission
        total_slippage = (
            (pos.entry_price * (self.config.slippage_bps * 1e-4)) + exit_fill.slippage
        ) * pos.quantity

        if pos.side == PositionSide.LONG:
            gross_pnl = (exit_fill.price - pos.entry_price) * pos.quantity
            net_pnl = gross_pnl - total_comm
            self.cash += (exit_fill.price * pos.quantity) - exit_fill.commission
            trade_side = PositionSide.LONG
        elif pos.side == PositionSide.SHORT:
            gross_pnl = (pos.entry_price - exit_fill.price) * pos.quantity
            net_pnl = gross_pnl - total_comm
            self.cash -= (exit_fill.price * pos.quantity) + exit_fill.commission
            trade_side = PositionSide.SHORT
        else:
            return None

        self.realized_pnl += net_pnl
        cost_basis = pos.entry_price * pos.quantity
        return_pct = (net_pnl / cost_basis * 100.0) if cost_basis > 0 else 0.0

        trade = BacktestTrade(
            symbol=symbol,
            side=trade_side,
            entry_timestamp=pos.entry_timestamp or exit_fill.timestamp,
            exit_timestamp=exit_fill.timestamp,
            entry_price=round(pos.entry_price, 6),
            exit_price=round(exit_fill.price, 6),
            quantity=pos.quantity,
            gross_pnl=round(gross_pnl, 6),
            commission=round(total_comm, 6),
            slippage_cost=round(total_slippage, 6),
            net_pnl=round(net_pnl, 6),
            return_pct=round(return_pct, 4),
            holding_bars=holding_bars,
            strategy_name=signal.strategy_name if signal else "",
            strategy_version=signal.strategy_version if signal else "",
        )

        self.trades.append(trade)
        # Reset position to FLAT
        self.positions[symbol] = BacktestPosition(
            symbol=symbol,
            side=PositionSide.FLAT,
            realized_pnl=self.realized_pnl,
        )
        self._entry_commissions.pop(symbol, None)

        logger.debug(
            "Closed %s trade on %s: PnL=%.2f (Net=%.2f)",
            trade_side.value,
            symbol,
            gross_pnl,
            net_pnl,
        )
        return trade

    def open_position(
        self,
        fill: BacktestFill,
        side: PositionSide,
    ) -> None:
        """Open a new LONG or SHORT position."""
        symbol = fill.symbol
        if side == PositionSide.LONG:
            self.cash -= (fill.price * fill.quantity) + fill.commission
        elif side == PositionSide.SHORT:
            self.cash += (fill.price * fill.quantity) - fill.commission

        self.positions[symbol] = BacktestPosition(
            symbol=symbol,
            side=side,
            quantity=fill.quantity,
            entry_price=fill.price,
            entry_timestamp=fill.timestamp,
            unrealized_pnl=0.0,
            realized_pnl=self.realized_pnl,
        )
        self._entry_commissions[symbol] = fill.commission

        logger.debug(
            "Opened %s position on %s: qty=%.2f @ %.2f",
            side.value,
            symbol,
            fill.quantity,
            fill.price,
        )

    def mark_to_market(
        self,
        timestamp: datetime,
        current_prices: dict[str, float],
    ) -> EquityPoint:
        """Revalue open positions against current candle closing prices."""
        total_unrealized = 0.0
        total_market_value = 0.0

        for symbol, pos in self.positions.items():
            if pos.side == PositionSide.FLAT or pos.quantity == 0:
                continue

            current_price = current_prices.get(symbol, pos.entry_price)
            if pos.side == PositionSide.LONG:
                unrealized = (current_price - pos.entry_price) * pos.quantity
                mkt_val = current_price * pos.quantity
            elif pos.side == PositionSide.SHORT:
                unrealized = (pos.entry_price - current_price) * pos.quantity
                mkt_val = (pos.entry_price * pos.quantity) + unrealized
            else:
                unrealized = 0.0
                mkt_val = 0.0

            # Update position in-place
            self.positions[symbol] = pos.model_copy(
                update={"unrealized_pnl": round(unrealized, 6)}
            )
            total_unrealized += unrealized
            total_market_value += mkt_val

        current_equity = self.initial_capital + self.realized_pnl + total_unrealized

        point = EquityPoint(
            timestamp=timestamp,
            cash=round(self.cash, 4),
            market_value=round(total_market_value, 4),
            realized_pnl=round(self.realized_pnl, 4),
            unrealized_pnl=round(total_unrealized, 4),
            equity=round(current_equity, 4),
        )
        self.equity_curve.append(point)
        return point
