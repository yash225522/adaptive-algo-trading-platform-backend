"""Trace context propagation, latency timers, and audit trail reconstruction."""

import contextvars
import time
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from adaptive_trading.observability.metrics import Histogram
from adaptive_trading.runtime.events import EventRecord, EventType

# Context variable for ambient trace propagation
_current_trace_context: contextvars.ContextVar["TraceContext | None"] = (
    contextvars.ContextVar("current_trace_context", default=None)
)


class TraceContext(BaseModel):
    """Context holding correlation identifiers for distributed tracing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(default="", description="Unique execution run identifier")
    correlation_id: str = Field(
        default="", description="Correlation identifier connecting causality chain"
    )
    event_id: str = Field(default="", description="Active event identifier")
    component: str = Field(default="", description="Active executing component")


class LatencyTimer:
    """Context manager measuring execution latency and recording into a Histogram."""

    def __init__(
        self,
        histogram: Histogram | None = None,
        labels: dict[str, str] | None = None,
    ) -> None:
        self.histogram = histogram
        self.labels = labels
        self.duration_ms: float = 0.0
        self._start_time: float = 0.0

    def __enter__(self) -> "LatencyTimer":
        self._start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        elapsed = (time.perf_counter() - self._start_time) * 1000.0
        self.duration_ms = elapsed
        if self.histogram is not None:
            self.histogram.record(elapsed, labels=self.labels)


class AuditTrace(BaseModel):
    """Structured decision lineage reconstructed from correlated lifecycle events."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: str
    symbol: str
    timestamp: datetime
    market_event: dict[str, Any] | None = None
    prediction: dict[str, Any] | None = None
    signal: dict[str, Any] | None = None
    risk_decision: dict[str, Any] | None = None
    order: dict[str, Any] | None = None
    fill: dict[str, Any] | None = None
    portfolio_update: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class AuditTracer:
    """Reconstructs and verifies end-to-end decision lineage across EventRecords."""

    @staticmethod
    def trace_correlation_id(
        events: Sequence[EventRecord], correlation_id: str
    ) -> AuditTrace | None:
        """Extract and structure all lifecycle events sharing a correlation ID."""
        matching = [e for e in events if e.correlation_id == correlation_id]
        if not matching:
            return None

        sym = matching[0].symbol
        ts = matching[0].timestamp

        trace_dict: dict[str, Any] = {
            "correlation_id": correlation_id,
            "symbol": sym,
            "timestamp": ts,
        }

        for ev in matching:
            if ev.event_type == EventType.MARKET_DATA:
                trace_dict["market_event"] = ev.payload
            elif ev.event_type == EventType.PREDICTION:
                trace_dict["prediction"] = ev.payload
            elif ev.event_type == EventType.SIGNAL:
                trace_dict["signal"] = ev.payload
            elif ev.event_type == EventType.RISK_DECISION:
                trace_dict["risk_decision"] = ev.payload
            elif ev.event_type == EventType.ORDER:
                trace_dict["order"] = ev.payload
            elif ev.event_type == EventType.FILL:
                trace_dict["fill"] = ev.payload
            elif ev.event_type == EventType.PORTFOLIO_UPDATE:
                trace_dict["portfolio_update"] = ev.payload
            elif ev.event_type == EventType.ERROR:
                trace_dict["error"] = ev.payload

        return AuditTrace(**trace_dict)

    @classmethod
    def trace_order_id(
        cls, events: Sequence[EventRecord], order_id: str
    ) -> AuditTrace | None:
        """Find correlation ID for an order ID and return full audit trace."""
        for ev in events:
            if ev.event_type in (EventType.ORDER, EventType.FILL):
                if (
                    ev.payload.get("order_id") == order_id
                    or ev.payload.get("client_order_id") == order_id
                ):
                    return cls.trace_correlation_id(events, ev.correlation_id)
        return None
