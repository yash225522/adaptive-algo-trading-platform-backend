"""Unit tests for Observability, Metrics, Run Tracking, and Health Checks."""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from adaptive_trading.common.config import TimeFrame
from adaptive_trading.domain.market import Candle
from adaptive_trading.observability.cli import (
    cmd_health,
    cmd_metrics,
    cmd_run_details,
    cmd_runs,
)
from adaptive_trading.observability.config import ObservabilityConfig
from adaptive_trading.observability.exceptions import (
    InvalidRunTransitionError,
    RunNotFoundError,
)
from adaptive_trading.observability.health import (
    HealthChecker,
    HealthStatus,
)
from adaptive_trading.observability.logging import (
    SensitiveDataFilter,
    StructuredJsonFormatter,
    redact_data,
    redact_text,
    setup_logging,
)
from adaptive_trading.observability.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
)
from adaptive_trading.observability.run_tracker import (
    RunStatus,
    RunSummary,
    RunTracker,
    RunType,
)
from adaptive_trading.observability.tracing import (
    AuditTracer,
    LatencyTimer,
)
from adaptive_trading.runtime.config import RuntimeConfig
from adaptive_trading.runtime.event_loop import (
    EventLoop,
    SimplePredictionService,
)
from adaptive_trading.runtime.events import EventRecord, EventType, MarketEvent

UTC_TZ = timezone.utc


def test_structured_json_formatter() -> None:
    """Verify StructuredJsonFormatter outputs valid JSON with standard fields."""
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Trading signal generated",
        args=(),
        exc_info=None,
    )
    setattr(record, "run_id", "run_123")
    setattr(record, "symbol", "NIFTY")

    formatted = formatter.format(record)
    parsed = json.loads(formatted)

    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test_logger"
    assert parsed["message"] == "Trading signal generated"
    assert parsed["run_id"] == "run_123"
    assert parsed["symbol"] == "NIFTY"
    assert "timestamp" in parsed


def test_sensitive_data_redaction() -> None:
    """Verify masking of passwords, JWTs, API keys, and bearer tokens."""
    raw_text = (
        "User logged in with password=MySecretPass! and "
        "api_key: 'xyz789' and Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    redacted = redact_text(raw_text)

    assert "MySecretPass!" not in redacted
    assert "xyz789" not in redacted
    assert "[REDACTED]" in redacted
    assert "[REDACTED_JWT]" in redacted

    data = {
        "user": "trader1",
        "password": "secret_password",
        "api_key": "secret_key_123",
        "nested": {
            "jwt_token": "token123",
            "safe_field": 42,
        },
    }
    cleaned = redact_data(data)
    assert cleaned["password"] == "[REDACTED]"
    assert cleaned["api_key"] == "[REDACTED]"
    assert cleaned["nested"]["jwt_token"] == "[REDACTED]"
    assert cleaned["nested"]["safe_field"] == 42


def test_logging_filter_scrubs_records() -> None:
    """Verify SensitiveDataFilter scrubs log record message in place."""
    filt = SensitiveDataFilter()
    record = logging.LogRecord(
        name="auth_logger",
        level=logging.INFO,
        pathname="auth.py",
        lineno=20,
        msg="Login with password=supersecret and token=abc12345",
        args=(),
        exc_info=None,
    )
    filt.filter(record)
    assert "supersecret" not in record.msg
    assert "abc12345" not in record.msg
    assert "[REDACTED]" in record.msg


def test_counter_metric() -> None:
    """Test Counter increment, labels, reset, and validation."""
    c = Counter(name="orders_total", description="Total orders submitted")
    c.increment()
    c.increment(2.0, labels={"symbol": "NIFTY", "side": "BUY"})
    c.increment(1.0, labels={"symbol": "BANKNIFTY", "side": "SELL"})

    assert c.total_value() == 4.0
    assert c.get_value() == 1.0
    assert c.get_value({"symbol": "NIFTY", "side": "BUY"}) == 2.0

    with pytest.raises(ValueError):
        c.increment(-1.0)

    snap = c.snapshot()
    assert snap["total"] == 4.0

    c.reset()
    assert c.total_value() == 0.0


def test_gauge_metric() -> None:
    """Test Gauge set, get, reset, and snapshot."""
    g = Gauge(name="portfolio_equity", description="Current equity")
    g.set(100_500.0)
    assert g.get_value() == 100_500.0

    g.set(101_200.0, labels={"account": "paper_main"})
    assert g.get_value({"account": "paper_main"}) == 101_200.0

    snap = g.snapshot()
    assert snap["type"] == "gauge"

    g.reset()
    assert g.get_value() == 0.0


def test_histogram_metric() -> None:
    """Test Histogram distribution statistics and percentiles."""
    h = Histogram(name="latency_ms", description="Execution latency")
    for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
        h.record(v)

    stats = h.get_stats()
    assert stats["count"] == 5.0
    assert stats["sum"] == 150.0
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0
    assert stats["mean"] == 30.0
    assert stats["p50"] == 30.0

    snap = h.snapshot()
    assert snap["type"] == "histogram"

    h.reset()
    assert h.get_stats()["count"] == 0.0


def test_latency_timer_records_to_histogram() -> None:
    """Test LatencyTimer measures execution time into a Histogram."""
    h = Histogram(name="calc_time")
    with LatencyTimer(histogram=h):
        time.sleep(0.01)

    stats = h.get_stats()
    assert stats["count"] == 1.0
    assert stats["mean"] >= 5.0  # At least 5ms elapsed


def test_metrics_registry_operations() -> None:
    """Test MetricsRegistry registering, snapshotting, and resetting."""
    reg = MetricsRegistry()
    c = reg.counter("events_count")
    g = reg.gauge("active_cash")
    h = reg.histogram("pred_latency")

    c.increment(5)
    g.set(50000.0)
    h.record(2.5)

    snap = reg.snapshot()
    assert "events_count" in snap["counters"]
    assert "active_cash" in snap["gauges"]
    assert "pred_latency" in snap["histograms"]

    reg.reset()
    assert reg.counter("events_count").total_value() == 0.0


def test_run_tracker_lifecycle_happy_path(tmp_path: Path) -> None:
    """Test RunTracker lifecycle: CREATED -> RUNNING -> COMPLETED."""
    tracker = RunTracker(base_dir=tmp_path)
    run = tracker.create_run(
        run_type=RunType.PAPER_REPLAY,
        model_name="logistic_regression",
        model_version="v1",
    )
    assert run.status == RunStatus.CREATED
    assert (tmp_path / run.run_id / "metadata.json").is_file()

    running = tracker.start_run(run.run_id)
    assert running.status == RunStatus.RUNNING

    summary = RunSummary(
        events_processed=100,
        predictions_generated=80,
        signals_generated=80,
        orders_submitted=2,
        orders_filled=2,
        final_equity=105_000.0,
        net_pnl=5_000.0,
    )
    completed = tracker.complete_run(run.run_id, summary=summary)
    assert completed.status == RunStatus.COMPLETED
    assert completed.duration_seconds is not None
    assert completed.summary is not None
    assert completed.summary.final_equity == 105_000.0

    retrieved = tracker.get_run(run.run_id)
    assert retrieved is not None
    assert retrieved.status == RunStatus.COMPLETED


def test_run_tracker_invalid_transition(tmp_path: Path) -> None:
    """Verify invalid state transitions raise InvalidRunTransitionError."""
    tracker = RunTracker(base_dir=tmp_path)
    run = tracker.create_run(run_type=RunType.BACKTEST)

    # Cannot transition directly from CREATED to COMPLETED
    with pytest.raises(InvalidRunTransitionError):
        tracker.complete_run(run.run_id)

    # Non-existent run
    with pytest.raises(RunNotFoundError):
        tracker.start_run("non_existent_id")


def test_run_tracker_failed_and_cancelled(tmp_path: Path) -> None:
    """Test transitioning runs to FAILED and CANCELLED."""
    tracker = RunTracker(base_dir=tmp_path)

    # Test FAILED
    run1 = tracker.create_run(run_type=RunType.ML_EXPERIMENT)
    tracker.start_run(run1.run_id)
    failed = tracker.fail_run(
        run1.run_id, error=ValueError("Model convergence failure")
    )
    assert failed.status == RunStatus.FAILED
    assert "Model convergence failure" in (failed.error_summary or "")

    # Test CANCELLED
    run2 = tracker.create_run(run_type=RunType.PAPER_TRADING)
    tracker.start_run(run2.run_id)
    cancelled = tracker.cancel_run(run2.run_id, reason="User requested interrupt")
    assert cancelled.status == RunStatus.CANCELLED
    assert "User requested interrupt" in (cancelled.error_summary or "")


def test_health_checker() -> None:
    """Test HealthChecker evaluating components and producing reports."""
    checker = HealthChecker()
    report = checker.check_all()

    assert report.status in (
        HealthStatus.HEALTHY,
        HealthStatus.DEGRADED,
        HealthStatus.UNHEALTHY,
    )
    assert "Market Data" in report.components
    assert "Paper Broker" in report.components
    assert "Runtime" in report.components
    assert report.components["Paper Broker"].status == HealthStatus.HEALTHY


def test_audit_tracer_full_lineage() -> None:
    """Verify AuditTracer reconstructs 7-step decision trace by correlation ID."""
    corr_id = "corr_abc123"
    t0 = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)

    events = [
        EventRecord(
            correlation_id=corr_id,
            event_type=EventType.MARKET_DATA,
            timestamp=t0,
            symbol="NIFTY",
            payload={"close": 25000.0},
        ),
        EventRecord(
            correlation_id=corr_id,
            event_type=EventType.PREDICTION,
            timestamp=t0,
            symbol="NIFTY",
            payload={"probability_up": 0.75},
        ),
        EventRecord(
            correlation_id=corr_id,
            event_type=EventType.SIGNAL,
            timestamp=t0,
            symbol="NIFTY",
            payload={"action": "LONG", "confidence": 0.75},
        ),
        EventRecord(
            correlation_id=corr_id,
            event_type=EventType.RISK_DECISION,
            timestamp=t0,
            symbol="NIFTY",
            payload={"approved": True, "approved_quantity": 1.0},
        ),
        EventRecord(
            correlation_id=corr_id,
            event_type=EventType.ORDER,
            timestamp=t0,
            symbol="NIFTY",
            payload={"order_id": "ord_1", "quantity": 1.0, "status": "FILLED"},
        ),
        EventRecord(
            correlation_id=corr_id,
            event_type=EventType.FILL,
            timestamp=t0,
            symbol="NIFTY",
            payload={"order_id": "ord_1", "fill_price": 25001.0},
        ),
        EventRecord(
            correlation_id=corr_id,
            event_type=EventType.PORTFOLIO_UPDATE,
            timestamp=t0,
            symbol="NIFTY",
            payload={"cash": 74999.0, "equity": 100000.0},
        ),
    ]

    trace = AuditTracer.trace_correlation_id(events, corr_id)
    assert trace is not None
    assert trace.market_event is not None
    assert trace.prediction is not None
    assert trace.prediction["probability_up"] == 0.75
    assert trace.signal is not None
    assert trace.signal["action"] == "LONG"
    assert trace.risk_decision is not None
    assert trace.risk_decision["approved"] is True
    assert trace.order is not None
    assert trace.order["order_id"] == "ord_1"
    assert trace.fill is not None
    assert trace.fill["fill_price"] == 25001.0
    assert trace.portfolio_update is not None
    assert trace.portfolio_update["cash"] == 74999.0

    # Test lookup by order ID
    trace_by_order = AuditTracer.trace_order_id(events, "ord_1")
    assert trace_by_order is not None
    assert trace_by_order.correlation_id == corr_id


def test_observability_no_behavioral_changes() -> None:
    """Verify that adding logging produces zero change in trading output."""
    t0 = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    candles = [
        Candle(
            timestamp=t0 + timedelta(minutes=5 * i),
            symbol="NIFTY",
            timeframe=TimeFrame.FIVE_MINUTES,
            open=100.0 + i,
            high=102.0 + i,
            low=99.0 + i,
            close=101.0 + i,
            volume=1000.0,
        )
        for i in range(10)
    ]

    orig_handlers = list(logging.getLogger().handlers)

    def run_loop(with_logging: bool) -> tuple[float, float, int]:
        if with_logging:
            setup_logging(ObservabilityConfig(log_level="DEBUG"))
        cfg = RuntimeConfig(warmup_period=5, initial_cash=100_000.0)
        loop = EventLoop(
            config=cfg,
            prediction_service=SimplePredictionService(fixed_probability_up=0.75),
        )
        for c in candles:
            loop.process_market_event(MarketEvent.from_candle(c))
        acct = loop.execution_service.broker.get_account()
        return (acct.cash, acct.equity, loop.state.stats.orders_filled)

    try:
        res_without = run_loop(with_logging=False)
        res_with = run_loop(with_logging=True)
    finally:
        root_logger = logging.getLogger()
        for h in list(root_logger.handlers):
            root_logger.removeHandler(h)
        for h in orig_handlers:
            root_logger.addHandler(h)

    assert res_without == res_with


def test_observability_cli_commands(tmp_path: Path) -> None:
    """Test CLI commands: health, metrics, runs, run, events."""
    assert cmd_health(as_json=False) == 0
    assert cmd_health(as_json=True) == 0
    assert cmd_metrics(as_json=False) == 0
    assert cmd_metrics(as_json=True) == 0

    runs_dir = tmp_path / "runs"
    tracker = RunTracker(base_dir=runs_dir)
    run = tracker.create_run(run_type=RunType.PAPER_REPLAY)
    tracker.start_run(run.run_id)
    tracker.complete_run(
        run.run_id,
        summary=RunSummary(events_processed=10, final_equity=100_000.0),
    )

    assert cmd_runs(runs_dir=runs_dir, as_json=False) == 0
    assert cmd_runs(runs_dir=runs_dir, as_json=True) == 0
    assert cmd_run_details(run_id=run.run_id, runs_dir=runs_dir, as_json=False) == 0
    assert cmd_run_details(run_id=run.run_id, runs_dir=runs_dir, as_json=True) == 0
