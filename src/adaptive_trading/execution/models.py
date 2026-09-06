"""Broker-independent data contracts for orders, fills, and paper accounts."""

import uuid
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from adaptive_trading.portfolio.models import PositionSide


class OrderSide(StrEnum):
    """Side of a market order."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    """Type of market order execution."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class ProductType(StrEnum):
    """Product classification for trade execution."""

    INTRADAY = "INTRADAY"
    DELIVERY = "DELIVERY"


class OrderStatus(StrEnum):
    """Lifecycle status states of an order."""

    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class OrderRequest(BaseModel):
    """Execution request submitted to a broker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    client_order_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Client-assigned idempotency key",
    )
    symbol: str = Field(min_length=1, description="Market ticker symbol")
    exchange: str = Field(default="NSE", min_length=1, description="Target exchange")
    side: OrderSide = Field(description="Order side (BUY/SELL)")
    quantity: float = Field(gt=0, description="Order quantity")
    order_type: OrderType = Field(
        default=OrderType.MARKET, description="Order execution type"
    )
    price: float | None = Field(
        default=None, gt=0, description="Limit price if LIMIT order"
    )
    product_type: ProductType = Field(
        default=ProductType.INTRADAY, description="Product type"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timezone-aware request creation timestamp",
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("OrderRequest timestamp must be timezone-aware")
        return v


class ExecutionFill(BaseModel):
    """Record of a trade fill executed by a broker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fill_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Unique execution fill identifier",
    )
    order_id: str = Field(description="Associated broker order identifier")
    timestamp: datetime = Field(description="Timezone-aware execution timestamp")
    symbol: str = Field(min_length=1, description="Market ticker symbol")
    side: OrderSide = Field(description="Executed order side")
    quantity: float = Field(gt=0, description="Executed quantity")
    price: float = Field(gt=0, description="Execution fill price")
    commission: float = Field(
        default=0.0, ge=0, description="Total commission fees charged"
    )
    slippage: float = Field(
        default=0.0, description="Applied slippage offset from market price"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("ExecutionFill timestamp must be timezone-aware")
        return v


class Order(BaseModel):
    """Broker-independent representation of an active or completed order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    order_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Broker-assigned order identifier",
    )
    client_order_id: str = Field(description="Client-provided idempotency key")
    symbol: str = Field(min_length=1, description="Market ticker symbol")
    exchange: str = Field(min_length=1, description="Target exchange")
    side: OrderSide = Field(description="Order side")
    quantity: float = Field(gt=0, description="Total requested quantity")
    filled_quantity: float = Field(
        default=0.0, ge=0, description="Filled quantity so far"
    )
    remaining_quantity: float = Field(ge=0, description="Remaining unfilled quantity")
    order_type: OrderType = Field(description="Order type")
    requested_price: float | None = Field(
        default=None, description="Requested limit price if applicable"
    )
    average_fill_price: float | None = Field(
        default=None, description="Volume-weighted average fill price"
    )
    status: OrderStatus = Field(
        default=OrderStatus.PENDING, description="Current order lifecycle status"
    )
    rejection_reason: str | None = Field(
        default=None, description="Reason if order was rejected or failed"
    )
    created_at: datetime = Field(description="Timezone-aware creation timestamp")
    updated_at: datetime = Field(description="Timezone-aware last update timestamp")

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("Order timestamps must be timezone-aware")
        return v


class PaperPosition(BaseModel):
    """Represents a simulated asset position held in a paper account."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1, description="Market ticker symbol")
    side: PositionSide = Field(
        default=PositionSide.FLAT, description="Position side (LONG/SHORT/FLAT)"
    )
    quantity: float = Field(default=0.0, ge=0.0, description="Position size")
    average_entry_price: float = Field(
        default=0.0, ge=0.0, description="Average entry price"
    )
    market_price: float = Field(default=0.0, ge=0.0, description="Current market price")
    unrealized_pnl: float = Field(
        default=0.0, description="Unrealized mark-to-market P&L"
    )
    realized_pnl: float = Field(default=0.0, description="Accumulated realized P&L")


class PaperAccount(BaseModel):
    """Simulated trading account state snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cash: float = Field(description="Available unencumbered cash balance")
    equity: float = Field(description="Total account equity (cash + positions)")
    buying_power: float = Field(description="Available buying power")
    realized_pnl: float = Field(description="Total cumulative realized P&L")
    unrealized_pnl: float = Field(description="Total open unrealized P&L")
    positions: dict[str, PaperPosition] = Field(
        default_factory=dict, description="Active and closed symbol positions"
    )
