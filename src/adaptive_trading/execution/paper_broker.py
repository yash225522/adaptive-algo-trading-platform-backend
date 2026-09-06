"""Simulated Paper Broker executing orders against synthetic or replay market prices."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Protocol

from adaptive_trading.execution.broker import Broker
from adaptive_trading.execution.config import ExecutionConfig
from adaptive_trading.execution.exceptions import OrderNotFoundError
from adaptive_trading.execution.models import (
    ExecutionFill,
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    PaperAccount,
    PaperPosition,
)
from adaptive_trading.portfolio.models import PositionSide

logger = logging.getLogger(__name__)


class MarketDataProvider(Protocol):
    """Protocol for fetching market prices during paper execution."""

    def get_latest_price(self, symbol: str) -> float | None:
        """Return the current market price for a symbol, or None if unavailable."""
        ...


class StaticMarketDataProvider:
    """Basic in-memory market price provider for paper simulations and testing."""

    def __init__(self, prices: dict[str, float] | None = None) -> None:
        self.prices: dict[str, float] = dict(prices or {})

    def set_price(self, symbol: str, price: float) -> None:
        """Set or update the market price for a symbol."""
        self.prices[symbol] = float(price)

    def get_latest_price(self, symbol: str) -> float | None:
        """Get the current price for a symbol."""
        return self.prices.get(symbol)


class PaperBroker(Broker):
    """Simulated execution broker maintaining paper account balances and positions."""

    def __init__(
        self,
        config: ExecutionConfig | None = None,
        market_data_provider: MarketDataProvider | None = None,
    ) -> None:
        self.config = config or ExecutionConfig()
        self.market_data_provider = market_data_provider or StaticMarketDataProvider()
        self.cash = self.config.initial_cash
        self.realized_pnl = 0.0
        self.positions: dict[str, PaperPosition] = {}
        self.orders: dict[str, Order] = {}
        self.client_order_map: dict[str, str] = {}
        self.fills: list[ExecutionFill] = []
        self._entry_commissions: dict[str, float] = {}

    def submit_order(self, request: OrderRequest) -> Order:
        """Submit a paper market order, validate, execute fill, and update state."""
        now = datetime.now(timezone.utc)

        # 1. Idempotency Check
        if request.client_order_id in self.client_order_map:
            existing_order_id = self.client_order_map[request.client_order_id]
            logger.info(
                "Duplicate client_order_id '%s' returning existing order '%s'",
                request.client_order_id,
                existing_order_id,
            )
            return self.orders[existing_order_id]

        order_id = uuid.uuid4().hex
        self.client_order_map[request.client_order_id] = order_id

        # 2. Input Validation
        validation_error = self._validate_request(request)
        if validation_error:
            rejected_order = Order(
                order_id=order_id,
                client_order_id=request.client_order_id,
                symbol=request.symbol,
                exchange=request.exchange,
                side=request.side,
                quantity=request.quantity,
                filled_quantity=0.0,
                remaining_quantity=request.quantity,
                order_type=request.order_type,
                requested_price=request.price,
                average_fill_price=None,
                status=OrderStatus.REJECTED,
                rejection_reason=validation_error,
                created_at=request.timestamp,
                updated_at=now,
            )
            self.orders[order_id] = rejected_order
            return rejected_order

        # 3. Market Price Discovery
        base_price = self.market_data_provider.get_latest_price(request.symbol)
        if base_price is None or base_price <= 0:
            if request.price is not None and request.price > 0:
                base_price = request.price
            else:
                rejected_order = Order(
                    order_id=order_id,
                    client_order_id=request.client_order_id,
                    symbol=request.symbol,
                    exchange=request.exchange,
                    side=request.side,
                    quantity=request.quantity,
                    filled_quantity=0.0,
                    remaining_quantity=request.quantity,
                    order_type=request.order_type,
                    requested_price=request.price,
                    average_fill_price=None,
                    status=OrderStatus.REJECTED,
                    rejection_reason=f"No market price available for {request.symbol}",
                    created_at=request.timestamp,
                    updated_at=now,
                )
                self.orders[order_id] = rejected_order
                return rejected_order

        # 4. Slippage & Fill Price Calculation
        slippage_rate = self.config.slippage_bps * 1e-4
        commission_rate = self.config.commission_bps * 1e-4

        if request.side == OrderSide.BUY:
            fill_price = base_price * (1.0 + slippage_rate)
            slippage_offset = fill_price - base_price
        else:
            fill_price = base_price * (1.0 - slippage_rate)
            slippage_offset = base_price - fill_price

        commission = fill_price * request.quantity * commission_rate

        # 5. Insufficient Funds Check for BUY opening/increasing
        current_pos = self.positions.get(request.symbol)
        is_closing_short = (
            current_pos is not None
            and current_pos.side == PositionSide.SHORT
            and current_pos.quantity > 0
        )
        if request.side == OrderSide.BUY and not is_closing_short:
            required_funds = (fill_price * request.quantity) + commission
            if required_funds > self.cash:
                rejected_order = Order(
                    order_id=order_id,
                    client_order_id=request.client_order_id,
                    symbol=request.symbol,
                    exchange=request.exchange,
                    side=request.side,
                    quantity=request.quantity,
                    filled_quantity=0.0,
                    remaining_quantity=request.quantity,
                    order_type=request.order_type,
                    requested_price=request.price,
                    average_fill_price=None,
                    status=OrderStatus.REJECTED,
                    rejection_reason=(
                        f"Insufficient funds: required {required_funds:.2f}, "
                        f"available cash {self.cash:.2f}"
                    ),
                    created_at=request.timestamp,
                    updated_at=now,
                )
                self.orders[order_id] = rejected_order
                return rejected_order

        # 6. Execute Simulated Fill and Update Positions & Account
        fill = ExecutionFill(
            order_id=order_id,
            timestamp=request.timestamp,
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            price=round(fill_price, 6),
            commission=round(commission, 6),
            slippage=round(slippage_offset, 6),
        )
        self.fills.append(fill)

        self._process_fill(fill)

        filled_order = Order(
            order_id=order_id,
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            exchange=request.exchange,
            side=request.side,
            quantity=request.quantity,
            filled_quantity=request.quantity,
            remaining_quantity=0.0,
            order_type=request.order_type,
            requested_price=request.price,
            average_fill_price=round(fill_price, 6),
            status=OrderStatus.FILLED,
            rejection_reason=None,
            created_at=request.timestamp,
            updated_at=now,
        )
        self.orders[order_id] = filled_order
        return filled_order

    def _validate_request(self, request: OrderRequest) -> str | None:
        """Validate request invariants."""
        if not request.symbol or len(request.symbol.strip()) == 0:
            return "Symbol is required"
        if not request.exchange or len(request.exchange.strip()) == 0:
            return "Exchange is required"
        if request.quantity <= 0:
            return f"Quantity must be positive, got {request.quantity}"
        if request.side not in (OrderSide.BUY, OrderSide.SELL):
            return f"Unsupported order side: {request.side}"
        if request.order_type not in (OrderType.MARKET, OrderType.LIMIT):
            return f"Unsupported order type: {request.order_type}"
        return None

    def _process_fill(self, fill: ExecutionFill) -> None:
        """Update account balances and position records upon fill execution."""
        symbol = fill.symbol
        pos = self.positions.get(
            symbol,
            PaperPosition(
                symbol=symbol,
                side=PositionSide.FLAT,
                quantity=0.0,
                average_entry_price=0.0,
                market_price=fill.price,
            ),
        )

        if fill.side == OrderSide.BUY:
            if pos.side == PositionSide.SHORT:
                # 1. Close / cover SHORT position
                entry_comm = self._entry_commissions.get(symbol, 0.0)
                gross_pnl = (pos.average_entry_price - fill.price) * pos.quantity
                net_pnl = gross_pnl - entry_comm - fill.commission
                self.realized_pnl += net_pnl
                self.cash -= (fill.price * pos.quantity) + fill.commission
                self._entry_commissions.pop(symbol, None)

                # Reset to FLAT
                self.positions[symbol] = PaperPosition(
                    symbol=symbol,
                    side=PositionSide.FLAT,
                    quantity=0.0,
                    average_entry_price=0.0,
                    market_price=fill.price,
                    realized_pnl=self.realized_pnl,
                )

                # If fill quantity exceeds existing short quantity (reversal into LONG)
                excess_qty = fill.quantity - pos.quantity
                if excess_qty > 0:
                    comm_excess = (
                        fill.price * excess_qty * (self.config.commission_bps * 1e-4)
                    )
                    self.cash -= (fill.price * excess_qty) + comm_excess
                    self.positions[symbol] = PaperPosition(
                        symbol=symbol,
                        side=PositionSide.LONG,
                        quantity=excess_qty,
                        average_entry_price=fill.price,
                        market_price=fill.price,
                        realized_pnl=self.realized_pnl,
                    )
                    self._entry_commissions[symbol] = comm_excess
            elif pos.side == PositionSide.LONG:
                # Add to LONG
                total_qty = pos.quantity + fill.quantity
                total_cost = (pos.average_entry_price * pos.quantity) + (
                    fill.price * fill.quantity
                )
                avg_price = total_cost / total_qty
                self.cash -= (fill.price * fill.quantity) + fill.commission
                self._entry_commissions[symbol] = (
                    self._entry_commissions.get(symbol, 0.0) + fill.commission
                )
                self.positions[symbol] = PaperPosition(
                    symbol=symbol,
                    side=PositionSide.LONG,
                    quantity=total_qty,
                    average_entry_price=round(avg_price, 6),
                    market_price=fill.price,
                    realized_pnl=self.realized_pnl,
                )
            else:
                # Open new LONG from FLAT
                self.cash -= (fill.price * fill.quantity) + fill.commission
                self._entry_commissions[symbol] = fill.commission
                self.positions[symbol] = PaperPosition(
                    symbol=symbol,
                    side=PositionSide.LONG,
                    quantity=fill.quantity,
                    average_entry_price=fill.price,
                    market_price=fill.price,
                    realized_pnl=self.realized_pnl,
                )

        elif fill.side == OrderSide.SELL:
            if pos.side == PositionSide.LONG:
                # 1. Close / exit LONG position
                entry_comm = self._entry_commissions.get(symbol, 0.0)
                gross_pnl = (fill.price - pos.average_entry_price) * pos.quantity
                net_pnl = gross_pnl - entry_comm - fill.commission
                self.realized_pnl += net_pnl
                self.cash += (fill.price * pos.quantity) - fill.commission
                self._entry_commissions.pop(symbol, None)

                # Reset to FLAT
                self.positions[symbol] = PaperPosition(
                    symbol=symbol,
                    side=PositionSide.FLAT,
                    quantity=0.0,
                    average_entry_price=0.0,
                    market_price=fill.price,
                    realized_pnl=self.realized_pnl,
                )

                # If fill quantity exceeds existing long quantity (reversal into SHORT)
                excess_qty = fill.quantity - pos.quantity
                if excess_qty > 0:
                    comm_excess = (
                        fill.price * excess_qty * (self.config.commission_bps * 1e-4)
                    )
                    self.cash += (fill.price * excess_qty) - comm_excess
                    self.positions[symbol] = PaperPosition(
                        symbol=symbol,
                        side=PositionSide.SHORT,
                        quantity=excess_qty,
                        average_entry_price=fill.price,
                        market_price=fill.price,
                        realized_pnl=self.realized_pnl,
                    )
                    self._entry_commissions[symbol] = comm_excess
            elif pos.side == PositionSide.SHORT:
                # Add to SHORT
                total_qty = pos.quantity + fill.quantity
                total_cost = (pos.average_entry_price * pos.quantity) + (
                    fill.price * fill.quantity
                )
                avg_price = total_cost / total_qty
                self.cash += (fill.price * fill.quantity) - fill.commission
                self._entry_commissions[symbol] = (
                    self._entry_commissions.get(symbol, 0.0) + fill.commission
                )
                self.positions[symbol] = PaperPosition(
                    symbol=symbol,
                    side=PositionSide.SHORT,
                    quantity=total_qty,
                    average_entry_price=round(avg_price, 6),
                    market_price=fill.price,
                    realized_pnl=self.realized_pnl,
                )
            else:
                # Open new SHORT from FLAT
                self.cash += (fill.price * fill.quantity) - fill.commission
                self._entry_commissions[symbol] = fill.commission
                self.positions[symbol] = PaperPosition(
                    symbol=symbol,
                    side=PositionSide.SHORT,
                    quantity=fill.quantity,
                    average_entry_price=fill.price,
                    market_price=fill.price,
                    realized_pnl=self.realized_pnl,
                )

    def cancel_order(self, order_id: str) -> Order:
        """Cancel an existing order if still open/pending."""
        if order_id not in self.orders:
            raise OrderNotFoundError(f"Order ID '{order_id}' not found")

        order = self.orders[order_id]
        if order.status in (OrderStatus.PENDING, OrderStatus.OPEN):
            updated = order.model_copy(
                update={
                    "status": OrderStatus.CANCELLED,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            self.orders[order_id] = updated
            return updated
        return order

    def get_order(self, order_id: str) -> Order | None:
        """Query order by broker order ID."""
        return self.orders.get(order_id)

    def get_order_by_client_id(self, client_order_id: str) -> Order | None:
        """Query order by client idempotency ID."""
        order_id = self.client_order_map.get(client_order_id)
        return self.orders.get(order_id) if order_id else None

    def get_positions(self) -> list[PaperPosition]:
        """Query all active positions."""
        return list(self.positions.values())

    def get_account(self) -> PaperAccount:
        """Revalue positions and return current PaperAccount state."""
        total_unrealized = 0.0
        updated_positions: dict[str, PaperPosition] = {}

        for sym, pos in self.positions.items():
            if pos.side == PositionSide.FLAT or pos.quantity == 0:
                updated_positions[sym] = pos.model_copy(
                    update={"unrealized_pnl": 0.0, "quantity": 0.0}
                )
                continue

            current_mkt_price = (
                self.market_data_provider.get_latest_price(sym)
                or pos.market_price
                or pos.average_entry_price
            )

            if pos.side == PositionSide.LONG:
                unrealized = (
                    current_mkt_price - pos.average_entry_price
                ) * pos.quantity
            elif pos.side == PositionSide.SHORT:
                unrealized = (
                    pos.average_entry_price - current_mkt_price
                ) * pos.quantity
            else:
                unrealized = 0.0

            updated_pos = pos.model_copy(
                update={
                    "market_price": round(current_mkt_price, 4),
                    "unrealized_pnl": round(unrealized, 4),
                }
            )
            updated_positions[sym] = updated_pos
            total_unrealized += unrealized

        current_equity = self.config.initial_cash + self.realized_pnl + total_unrealized
        buying_power = max(0.0, self.cash)

        return PaperAccount(
            cash=round(self.cash, 4),
            equity=round(current_equity, 4),
            buying_power=round(buying_power, 4),
            realized_pnl=round(self.realized_pnl, 4),
            unrealized_pnl=round(total_unrealized, 4),
            positions=updated_positions,
        )
