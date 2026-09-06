"""Unit tests for the Paper Trading Event Loop, TradingClock, and ReplayRunner."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from adaptive_trading.common.config import TimeFrame
from adaptive_trading.domain.market import Candle
from adaptive_trading.execution.exceptions import LiveTradingNotEnabledError
from adaptive_trading.risk.config import RiskConfig
from adaptive_trading.risk.engine import RiskEngine
from adaptive_trading.runtime.cli import cmd_replay, cmd_status
from adaptive_trading.runtime.clock import TradingClock
from adaptive_trading.runtime.config import RuntimeConfig, RuntimeMode
from adaptive_trading.runtime.event_loop import (
    EventLoop,
    SimplePredictionService,
)
from adaptive_trading.runtime.events import EventType, MarketEvent
from adaptive_trading.runtime.exceptions import (
    EventProcessingError,
    InvalidEventOrderError,
)
from adaptive_trading.runtime.runner import (
    HistoricalDataProvider,
    ReplayRunner,
)

UTC_TZ = timezone.utc


def make_test_candles(count: int = 30, base_price: float = 100.0) -> list[Candle]:
    """Generate deterministic synthetic candles."""
    base_time = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    return [
        Candle(
            timestamp=base_time + timedelta(minutes=5 * i),
            symbol="NIFTY",
            timeframe=TimeFrame.FIVE_MINUTES,
            open=base_price + i * 1.5,
            high=base_price + i * 1.5 + 1.0,
            low=base_price + i * 1.5 - 0.5,
            close=base_price + i * 1.5 + 0.5,
            volume=1000.0 + i * 10,
        )
        for i in range(count)
    ]


def test_trading_clock_advance_and_backwards_error() -> None:
    """Test TradingClock advancement and chronological error checking."""
    t0 = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    t1 = datetime(2026, 1, 2, 9, 20, tzinfo=UTC_TZ)
    t_past = datetime(2026, 1, 2, 9, 10, tzinfo=UTC_TZ)

    clock = TradingClock(initial_time=t0)
    assert clock.now() == t0

    clock.advance_to(t1)
    assert clock.now() == t1

    with pytest.raises(InvalidEventOrderError):
        clock.advance_to(t_past)


def test_runtime_config_live_mode_rejected() -> None:
    """Verify LIVE mode raises LiveTradingNotEnabledError."""
    with pytest.raises(LiveTradingNotEnabledError):
        RuntimeConfig(mode=RuntimeMode.LIVE)


def test_market_event_conversion_roundtrip() -> None:
    """Test conversion between Candle and MarketEvent."""
    candle = make_test_candles(count=1)[0]
    ev = MarketEvent.from_candle(candle)
    assert ev.symbol == "NIFTY"
    assert ev.close == candle.close

    candle_rt = ev.to_candle()
    assert candle_rt.timestamp == candle.timestamp
    assert candle_rt.symbol == candle.symbol


def test_warmup_period_behavior() -> None:
    """Verify no predictions or orders occur during initial warmup period."""
    candles = make_test_candles(count=15)
    config = RuntimeConfig(warmup_period=10)
    pred_service = SimplePredictionService(fixed_probability_up=0.75)

    event_loop = EventLoop(
        config=config,
        prediction_service=pred_service,
    )

    # Process first 5 candles (warmup = 10 -> no predictions)
    for c in candles[:5]:
        event_loop.process_market_event(MarketEvent.from_candle(c))

    assert event_loop.state.stats.market_events_processed == 5
    assert event_loop.state.stats.predictions_generated == 0
    assert event_loop.state.stats.orders_submitted == 0

    # Process next 10 candles (past warmup -> predictions and orders start)
    for c in candles[5:15]:
        event_loop.process_market_event(MarketEvent.from_candle(c))

    assert event_loop.state.stats.market_events_processed == 15
    assert event_loop.state.stats.predictions_generated > 0


def test_event_loop_event_ordering_trace() -> None:
    """Verify complete lifecycle event ordering for a trade."""
    candles = make_test_candles(count=10)
    config = RuntimeConfig(warmup_period=5)
    pred_service = SimplePredictionService(fixed_probability_up=0.80)

    event_loop = EventLoop(
        config=config,
        prediction_service=pred_service,
    )

    for c in candles:
        event_loop.process_market_event(MarketEvent.from_candle(c))

    # Find the events for a single bar that produced a trade
    trade_events = [e for e in event_loop.events if e.event_type == EventType.FILL]
    assert len(trade_events) > 0
    corr_id = trade_events[0].correlation_id

    bar_records = [e for e in event_loop.events if e.correlation_id == corr_id]
    types = [e.event_type for e in bar_records]

    expected_sequence = [
        EventType.MARKET_DATA,
        EventType.PREDICTION,
        EventType.SIGNAL,
        EventType.RISK_DECISION,
        EventType.ORDER,
        EventType.FILL,
        EventType.PORTFOLIO_UPDATE,
    ]
    assert types == expected_sequence


def test_event_loop_no_trade_signal() -> None:
    """Verify NO_TRADE strategy signal generates no order or fill."""
    candles = make_test_candles(count=8)
    config = RuntimeConfig(warmup_period=5)
    # Neutral probability 0.50 yields NO_TRADE
    pred_service = SimplePredictionService(fixed_probability_up=0.50)

    event_loop = EventLoop(
        config=config,
        prediction_service=pred_service,
    )

    for c in candles:
        event_loop.process_market_event(MarketEvent.from_candle(c))

    assert event_loop.state.stats.predictions_generated > 0
    assert event_loop.state.stats.signals_generated > 0
    assert event_loop.state.stats.orders_submitted == 0
    assert event_loop.state.stats.orders_filled == 0


def test_event_loop_risk_rejection() -> None:
    """Verify risk limit rejection records RISK_DECISION but no orders."""
    candles = make_test_candles(count=8)
    config = RuntimeConfig(warmup_period=5)
    pred_service = SimplePredictionService(fixed_probability_up=0.80)

    # Configure risk engine with tight limit
    risk_engine = RiskEngine(
        config=RiskConfig(
            max_position_pct=0.00001,
            allow_quantity_reduction=False,
        )
    )

    event_loop = EventLoop(
        config=config,
        prediction_service=pred_service,
        risk_engine=risk_engine,
    )

    for c in candles:
        event_loop.process_market_event(MarketEvent.from_candle(c))

    assert event_loop.state.stats.signals_rejected > 0
    assert event_loop.state.stats.orders_submitted == 0


def test_event_loop_deduplication() -> None:
    """Verify identical market event is dropped when deduplication is enabled."""
    candle = make_test_candles(count=1)[0]
    ev = MarketEvent.from_candle(candle)

    event_loop = EventLoop(config=RuntimeConfig(deduplicate_events=True))
    records1 = event_loop.process_market_event(ev)
    records2 = event_loop.process_market_event(ev)

    assert len(records1) > 0
    assert len(records2) == 0
    assert event_loop.state.stats.market_events_processed == 1


def test_event_loop_fail_fast_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify fail-fast policy raises EventProcessingError on internal failure."""
    candle = make_test_candles(count=1)[0]
    ev = MarketEvent.from_candle(candle)

    event_loop = EventLoop(
        config=RuntimeConfig(warmup_period=1, fail_fast=True),
        prediction_service=SimplePredictionService(),
    )

    def mock_fail(*args: object, **kwargs: object) -> None:
        raise ValueError("Boom")

    monkeypatch.setattr(
        event_loop.feature_pipeline, "generate_feature_vectors", mock_fail
    )

    with pytest.raises(EventProcessingError, match="Boom"):
        event_loop.process_market_event(ev)


def test_replay_runner_end_to_end_artifacts(tmp_path: Path) -> None:
    """Test ReplayRunner execution and persistence of all 5 artifacts."""
    candles = make_test_candles(count=10)
    config = RuntimeConfig(warmup_period=5, initial_cash=100_000.0)
    pred_service = SimplePredictionService(fixed_probability_up=0.75)

    event_loop = EventLoop(
        config=config,
        prediction_service=pred_service,
    )
    runner = ReplayRunner(event_loop=event_loop, config=config)
    provider = HistoricalDataProvider(candles=candles)

    out_dir = runner.run(provider=provider, artifacts_base_dir=tmp_path)

    assert (out_dir / "metadata.json").is_file()
    assert (out_dir / "events.jsonl").is_file()
    assert (out_dir / "runtime_stats.json").is_file()
    assert (out_dir / "checkpoint.json").is_file()
    assert (out_dir / "final_state.json").is_file()

    with open(out_dir / "runtime_stats.json", "r", encoding="utf-8") as f:
        stats = json.load(f)
    assert stats["market_events_processed"] == 10
    assert stats["orders_filled"] > 0


def test_determinism_across_identical_runs(tmp_path: Path) -> None:
    """Verify two independent runs on identical data produce identical state."""
    candles = make_test_candles(count=12)

    def run_simulation(dest: Path) -> tuple[float, int, int]:
        cfg = RuntimeConfig(warmup_period=5, initial_cash=100_000.0)
        loop = EventLoop(
            config=cfg,
            prediction_service=SimplePredictionService(fixed_probability_up=0.75),
        )
        r = ReplayRunner(event_loop=loop, config=cfg)
        out = r.run(HistoricalDataProvider(candles=candles), artifacts_base_dir=dest)
        with open(out / "final_state.json", "r", encoding="utf-8") as f:
            st = json.load(f)
        return (st["equity"], loop.state.stats.orders_filled, len(loop.events))

    res1 = run_simulation(tmp_path / "run1")
    res2 = run_simulation(tmp_path / "run2")

    assert res1 == res2


def test_runtime_cli_replay_and_status(tmp_path: Path) -> None:
    """Test CLI replay and status subcommands."""
    ts0 = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    candles = [
        Candle(
            timestamp=ts0 + timedelta(minutes=5 * i),
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
    candles_file = tmp_path / "candles.csv"
    pd.DataFrame(
        [
            {
                "timestamp": c.timestamp.isoformat(),
                "symbol": c.symbol,
                "timeframe": "5m",
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in candles
        ]
    ).to_csv(candles_file, index=False)

    out_base = tmp_path / "runtime_out"
    code_replay = cmd_replay(
        candles_file=candles_file,
        initial_cash=100_000.0,
        warmup_period=5,
        output_dir=out_base,
    )
    assert code_replay == 0

    run_dirs = list(out_base.glob("run_*"))
    assert len(run_dirs) == 1
    code_status = cmd_status(run_dir=run_dirs[0])
    assert code_status == 0
