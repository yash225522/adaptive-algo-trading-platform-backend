"""Runtime operational statistics, state management, and checkpoints."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from adaptive_trading.execution.models import ExecutionFill, Order
from adaptive_trading.risk.models import RiskDecision
from adaptive_trading.runtime.clock import TradingClock
from adaptive_trading.strategy.models import StrategyPrediction, TradingSignal


class RuntimeStats(BaseModel):
    """Operational metrics tracking the volume of processed and rejected items."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    market_events_processed: int = 0
    predictions_generated: int = 0
    signals_generated: int = 0
    signals_rejected: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    errors: int = 0


class RuntimeCheckpoint(BaseModel):
    """Serializable snapshot of runtime state for recovery and auditing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime = Field(description="Simulated timestamp of checkpoint")
    last_event_id: str = Field(
        description="Identifier of most recently processed event"
    )
    state_version: str = Field(default="v1", description="Checkpoint schema version")
    stats: RuntimeStats = Field(description="Operational statistics at checkpoint")
    account_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of current cash, equity, and positions",
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timezone_aware(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone-aware."""
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("RuntimeCheckpoint timestamp must be timezone-aware")
        return v


class RuntimeState:
    """Active mutable state container for the EventLoop orchestrator."""

    def __init__(self, initial_time: datetime | None = None) -> None:
        self.clock = TradingClock(initial_time=initial_time)
        self.current_timestamp: datetime | None = initial_time
        self.last_market_price: dict[str, float] = {}
        self.last_prediction: StrategyPrediction | None = None
        self.last_signal: TradingSignal | None = None
        self.last_risk_decision: RiskDecision | None = None
        self.last_order: Order | None = None
        self.last_fill: ExecutionFill | None = None
        self.stats = RuntimeStats()
        self.seen_events: set[tuple[str, datetime]] = set()

    def is_duplicate(self, symbol: str, timestamp: datetime) -> bool:
        """Check if an event for (symbol, timestamp) has already been processed."""
        key = (symbol, timestamp)
        if key in self.seen_events:
            return True
        self.seen_events.add(key)
        return False
