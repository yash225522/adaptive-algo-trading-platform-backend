"""Domain models and data contracts for backtesting simulation and trade records."""

import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from adaptive_trading.domain.trading import OrderSide, OrderType


class PositionSide(StrEnum):
    """Side of an open or simulated market position."""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class BacktestOrder(BaseModel):
    """Simulated order submitted to the execution handler."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    order_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Unique identifier for the order",
    )
    timestamp: datetime = Field(
        description="Timezone-aware timestamp of order creation"
    )
    symbol: str = Field(min_length=1, description="Market ticker symbol")
    side: OrderSide = Field(description="Order side (BUY/SELL)")
    quantity: float = Field(gt=0, description="Order quantity")
    order_type: OrderType = Field(
        default=OrderType.MARKET, description="Order execution type"
    )
    signal_id: str | None = Field(
        default=None, description="Identifier of triggering signal"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("Order timestamp must be timezone-aware")
        return v


class BacktestFill(BaseModel):
    """Simulated execution fill confirmation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fill_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Unique identifier for the fill",
    )
    order_id: str = Field(description="Associated order identifier")
    timestamp: datetime = Field(description="Timezone-aware timestamp of execution")
    symbol: str = Field(min_length=1, description="Market ticker symbol")
    side: OrderSide = Field(description="Executed order side (BUY/SELL)")
    quantity: float = Field(gt=0, description="Filled quantity")
    price: float = Field(gt=0, description="Executed fill price with slippage")
    commission: float = Field(
        default=0.0, ge=0, description="Total commission fees charged"
    )
    slippage: float = Field(default=0.0, description="Price slippage offset applied")

    @field_validator("timestamp")
    @classmethod
    def validate_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("Fill timestamp must be timezone-aware")
        return v


class BacktestPosition(BaseModel):
    """Tracks current active or closed position in a portfolio."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1, description="Market ticker symbol")
    side: PositionSide = Field(
        default=PositionSide.FLAT, description="Position side (LONG/SHORT/FLAT)"
    )
    quantity: float = Field(default=0.0, ge=0.0, description="Position size")
    entry_price: float = Field(
        default=0.0, ge=0.0, description="Average entry fill price"
    )
    entry_timestamp: datetime | None = Field(
        default=None, description="Timezone-aware entry timestamp"
    )
    unrealized_pnl: float = Field(
        default=0.0, description="Unrealized mark-to-market P&L"
    )
    realized_pnl: float = Field(default=0.0, description="Accumulated realized P&L")


class EquityPoint(BaseModel):
    """Snapshot of portfolio equity at a discrete time point."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime = Field(description="Timezone-aware timestamp of equity point")
    cash: float = Field(description="Available cash balance")
    market_value: float = Field(description="Current market value of positions")
    realized_pnl: float = Field(description="Cumulative realized P&L")
    unrealized_pnl: float = Field(description="Current unrealized P&L")
    equity: float = Field(description="Total portfolio equity (cash + positions)")

    @field_validator("timestamp")
    @classmethod
    def validate_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("EquityPoint timestamp must be timezone-aware")
        return v


class BacktestTrade(BaseModel):
    """Completed round-trip trade record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trade_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Unique identifier for the trade",
    )
    symbol: str = Field(min_length=1, description="Market ticker symbol")
    side: PositionSide = Field(description="Trade direction (LONG/SHORT)")
    entry_timestamp: datetime = Field(description="Timezone-aware entry timestamp")
    exit_timestamp: datetime = Field(description="Timezone-aware exit timestamp")
    entry_price: float = Field(gt=0, description="Average entry price")
    exit_price: float = Field(gt=0, description="Average exit price")
    quantity: float = Field(gt=0, description="Traded quantity")
    gross_pnl: float = Field(description="Gross profit/loss before fees")
    commission: float = Field(
        default=0.0, ge=0, description="Total commission fees incurred"
    )
    slippage_cost: float = Field(
        default=0.0, description="Total slippage cost incurred"
    )
    net_pnl: float = Field(description="Net profit/loss after fees and slippage")
    return_pct: float = Field(description="Percentage return on trade entry capital")
    holding_bars: int = Field(
        default=1, ge=0, description="Number of bars position was held"
    )
    strategy_name: str = Field(default="", description="Strategy name")
    strategy_version: str = Field(default="", description="Strategy version")

    @field_validator("entry_timestamp", "exit_timestamp")
    @classmethod
    def validate_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("Trade timestamps must be timezone-aware")
        return v


class PerformanceMetrics(BaseModel):
    """Structured performance metrics calculated from a backtest run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    initial_capital: float
    final_equity: float
    net_pnl: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    scratch_trades: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    profit_factor: float | None = None
    average_trade_pnl: float
    average_winner: float
    average_loser: float
    largest_winner: float
    largest_loser: float
    max_drawdown_abs: float
    max_drawdown_pct: float


class BacktestResult(BaseModel):
    """Complete container encapsulating an entire backtest execution run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    backtest_id: str
    start_timestamp: datetime | None = None
    end_timestamp: datetime | None = None
    strategy_name: str = ""
    strategy_version: str = ""
    model_name: str = ""
    model_version: str = ""
    config: dict[str, object]
    metrics: PerformanceMetrics
    trades: list[BacktestTrade] = Field(default_factory=list)
    equity_curve: list[EquityPoint] = Field(default_factory=list)
    created_at: datetime
