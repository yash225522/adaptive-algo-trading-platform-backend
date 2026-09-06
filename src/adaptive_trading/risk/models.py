"""Data models for risk decisions and risk check results."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from adaptive_trading.strategy.models import SignalAction


class RiskCheckResult(BaseModel):
    """Result of an individual risk limit check."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, description="Name of the risk check")
    passed: bool = Field(description="True if the check passed")
    reason: str | None = Field(
        default=None, description="Explanation if check failed or warning"
    )
    metric_value: float | None = Field(
        default=None, description="Observed portfolio or position metric value"
    )
    limit_value: float | None = Field(
        default=None, description="Configured limit threshold"
    )


class RiskDecision(BaseModel):
    """Structured decision produced by the risk management engine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Unique identifier for the risk decision",
    )
    timestamp: datetime = Field(
        description="Timezone-aware timestamp of the risk decision"
    )
    symbol: str = Field(min_length=1, description="Market ticker symbol")
    action: SignalAction = Field(description="Evaluated trade action")
    requested_quantity: float = Field(
        ge=0.0, description="Quantity requested by the strategy"
    )
    approved_quantity: float = Field(
        ge=0.0, description="Quantity approved by the risk engine"
    )
    approved: bool = Field(description="True if trade is permitted (quantity > 0)")
    reason: str = Field(
        default="", description="Summary rationale for approval or rejection"
    )
    risk_checks: list[RiskCheckResult] = Field(
        default_factory=list, description="Ordered audit log of risk checks"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("RiskDecision timestamp must be timezone-aware")
        return v
