"""Simulated order execution with realistic slippage and commission calculations."""

import logging

from adaptive_trading.backtesting.config import BacktestConfig
from adaptive_trading.backtesting.exceptions import ExecutionError
from adaptive_trading.backtesting.models import BacktestFill, BacktestOrder
from adaptive_trading.domain.market import Candle
from adaptive_trading.domain.trading import OrderSide

logger = logging.getLogger(__name__)


class SimulatedExecutionHandler:
    """Simulates market order execution against historical candle prices."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def simulate_fill(
        self,
        order: BacktestOrder,
        candle: Candle,
    ) -> BacktestFill:
        """Simulate an order fill at candle open with slippage and commission.

        Args:
            order: BacktestOrder submitted for simulation.
            candle: Historical candle where execution occurs.

        Returns:
            BacktestFill: Confirmed execution fill.

        Raises:
            ExecutionError: If candle price is invalid or symbols mismatch.
        """
        if order.symbol != candle.symbol:
            raise ExecutionError(
                f"Symbol mismatch: Order symbol '{order.symbol}' != "
                f"Candle symbol '{candle.symbol}'"
            )

        if candle.open <= 0:
            raise ExecutionError(
                f"Invalid candle open price ({candle.open}) for {candle.symbol}"
            )

        base_price = float(candle.open)
        slippage_rate = self.config.slippage_bps * 1e-4
        commission_rate = self.config.commission_bps * 1e-4

        if order.side == OrderSide.BUY:
            fill_price = base_price * (1.0 + slippage_rate)
            slippage_offset = fill_price - base_price
        elif order.side == OrderSide.SELL:
            fill_price = base_price * (1.0 - slippage_rate)
            slippage_offset = base_price - fill_price
        else:
            raise ExecutionError(f"Unsupported order side: {order.side}")

        trade_value = fill_price * order.quantity
        commission = trade_value * commission_rate

        fill = BacktestFill(
            order_id=order.order_id,
            timestamp=candle.timestamp,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=round(fill_price, 6),
            commission=round(commission, 6),
            slippage=round(slippage_offset, 6),
        )

        logger.debug(
            "Executed fill for %s %s: qty=%.2f price=%.2f comm=%.2f slip=%.2f",
            fill.side.value,
            fill.symbol,
            fill.quantity,
            fill.price,
            fill.commission,
            fill.slippage,
        )
        return fill
