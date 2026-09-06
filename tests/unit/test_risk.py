"""Unit tests for the Risk & Portfolio Engine, limits, sizing, and CLI."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from adaptive_trading.backtesting.config import BacktestConfig
from adaptive_trading.backtesting.engine import BacktestEngine
from adaptive_trading.common.config import TimeFrame
from adaptive_trading.domain.market import Candle
from adaptive_trading.portfolio.models import PortfolioState, Position, PositionSide
from adaptive_trading.risk.cli import run_risk_evaluation
from adaptive_trading.risk.config import RiskConfig
from adaptive_trading.risk.engine import RiskEngine
from adaptive_trading.strategy.models import SignalAction, TradingSignal

UTC_TZ = timezone.utc


@pytest.fixture
def base_portfolio_state() -> PortfolioState:
    """Create a baseline healthy PortfolioState."""
    ts = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    return PortfolioState(
        timestamp=ts,
        cash=100_000.0,
        equity=100_000.0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        portfolio_exposure=0.0,
        peak_equity=100_000.0,
        daily_start_equity=100_000.0,
        current_date=ts.date(),
        positions={},
    )


def test_risk_config_validation() -> None:
    """Test RiskConfig boundary validations."""
    config = RiskConfig(
        max_position_pct=0.25,
        max_daily_loss_pct=0.03,
        max_drawdown_pct=0.15,
        max_open_positions=3,
        fixed_quantity=5.0,
    )
    assert config.max_position_pct == 0.25
    assert config.max_open_positions == 3

    with pytest.raises(ValueError):
        RiskConfig(max_open_positions=0)

    with pytest.raises(ValueError):
        RiskConfig(fixed_quantity=-1.0)


def test_position_size_limit_rejection_and_reduction(
    base_portfolio_state: PortfolioState,
) -> None:
    """Test capping or rejecting trades that exceed maximum position value/pct."""
    ts = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    signal = TradingSignal(
        signal_id="s1",
        timestamp=ts,
        symbol="NIFTY",
        action=SignalAction.LONG,
        confidence=0.8,
        strategy_name="prob",
        strategy_version="v1",
        reason="LONG test",
    )

    # 1. With allow_quantity_reduction=True (Max pos = 20% of 100k = 20,000)
    # Price = 1,000; Fixed qty = 30 -> Value = 30,000 > 20,000 -> Should reduce to 20.0
    config_reduce = RiskConfig(
        max_position_pct=0.20,
        fixed_quantity=30.0,
        allow_quantity_reduction=True,
    )
    engine_reduce = RiskEngine(config=config_reduce)
    decision = engine_reduce.evaluate(
        signal=signal,
        portfolio_state=base_portfolio_state,
        current_price=1000.0,
    )
    assert decision.approved is True
    assert decision.approved_quantity == 20.0
    assert "reduced" in decision.reason

    # 2. With allow_quantity_reduction=False -> Should reject completely
    config_reject = RiskConfig(
        max_position_pct=0.20,
        fixed_quantity=30.0,
        allow_quantity_reduction=False,
    )
    engine_reject = RiskEngine(config=config_reject)
    decision_rej = engine_reject.evaluate(
        signal=signal,
        portfolio_state=base_portfolio_state,
        current_price=1000.0,
    )
    assert decision_rej.approved is False
    assert decision_rej.approved_quantity == 0.0
    assert "exceeds" in decision_rej.reason


def test_exposure_limit_check(base_portfolio_state: PortfolioState) -> None:
    """Test total gross exposure constraint."""
    ts = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    signal = TradingSignal(
        signal_id="s1",
        timestamp=ts,
        symbol="BANKNIFTY",
        action=SignalAction.LONG,
        confidence=0.8,
        strategy_name="prob",
        strategy_version="v1",
        reason="LONG test",
    )

    # Portfolio already has 90,000 exposure out of 100,000 max (100% of 100k)
    state_exposed = base_portfolio_state.model_copy(
        update={"portfolio_exposure": 90_000.0}
    )

    # Request trade with value 20,000 (price=2000, qty=10)
    # Remaining capacity = 10,000 -> Should reduce qty to 5.0
    config = RiskConfig(
        max_portfolio_exposure_pct=1.0,
        max_position_pct=None,
        fixed_quantity=10.0,
        allow_quantity_reduction=True,
    )
    engine = RiskEngine(config=config)
    decision = engine.evaluate(
        signal=signal,
        portfolio_state=state_exposed,
        current_price=2000.0,
    )
    assert decision.approved is True
    assert decision.approved_quantity == 5.0


def test_daily_loss_limit_blocks_open_allows_close(
    base_portfolio_state: PortfolioState,
) -> None:
    """Verify daily loss limit blocks new risk but allows closing trades."""
    ts = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)

    # Current equity = 97,000 vs Daily start = 100,000 (Loss = 3.0% > 2.0% limit)
    loss_state = base_portfolio_state.model_copy(
        update={
            "equity": 97_000.0,
            "daily_start_equity": 100_000.0,
            "positions": {
                "NIFTY": Position(
                    symbol="NIFTY",
                    side=PositionSide.LONG,
                    quantity=1.0,
                    entry_price=100.0,
                    market_value=97.0,
                )
            },
        }
    )
    config = RiskConfig(max_daily_loss_pct=0.02)
    engine = RiskEngine(config=config)

    # 1. New risk trade on another symbol -> REJECTED
    open_signal = TradingSignal(
        signal_id="s_open",
        timestamp=ts,
        symbol="BANKNIFTY",
        action=SignalAction.LONG,
        confidence=0.8,
        strategy_name="prob",
        strategy_version="v1",
        reason="Open new risk",
    )
    decision_open = engine.evaluate(open_signal, loss_state, current_price=1000.0)
    assert decision_open.approved is False
    assert "Daily loss" in decision_open.reason

    # 2. Risk-reducing close trade on NIFTY -> ALLOWED
    close_signal = TradingSignal(
        signal_id="s_close",
        timestamp=ts,
        symbol="NIFTY",
        action=SignalAction.SHORT,
        confidence=0.8,
        strategy_name="prob",
        strategy_version="v1",
        reason="Close existing position",
    )
    decision_close = engine.evaluate(close_signal, loss_state, current_price=97.0)
    assert decision_close.approved is True
    assert decision_close.approved_quantity > 0


def test_drawdown_limit_blocks_open_allows_close(
    base_portfolio_state: PortfolioState,
) -> None:
    """Verify max drawdown guard blocks new risk but allows closing trades."""
    ts = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)

    # Peak equity = 120,000, current equity = 105,000 (Drawdown = 12.5% > 10% limit)
    dd_state = base_portfolio_state.model_copy(
        update={
            "equity": 105_000.0,
            "peak_equity": 120_000.0,
            "positions": {
                "NIFTY": Position(
                    symbol="NIFTY",
                    side=PositionSide.SHORT,
                    quantity=1.0,
                    entry_price=100.0,
                    market_value=105.0,
                )
            },
        }
    )
    config = RiskConfig(max_drawdown_pct=0.10)
    engine = RiskEngine(config=config)

    # 1. New risk on new symbol -> REJECTED
    open_sig = TradingSignal(
        signal_id="s_open",
        timestamp=ts,
        symbol="RELIANCE",
        action=SignalAction.LONG,
        confidence=0.8,
        strategy_name="prob",
        strategy_version="v1",
        reason="Open new risk",
    )
    decision_open = engine.evaluate(open_sig, dd_state, current_price=2500.0)
    assert decision_open.approved is False
    assert "Drawdown" in decision_open.reason

    # 2. Closing short on NIFTY -> ALLOWED
    close_sig = TradingSignal(
        signal_id="s_close",
        timestamp=ts,
        symbol="NIFTY",
        action=SignalAction.LONG,
        confidence=0.8,
        strategy_name="prob",
        strategy_version="v1",
        reason="Close existing position",
    )
    decision_close = engine.evaluate(close_sig, dd_state, current_price=105.0)
    assert decision_close.approved is True


def test_open_positions_limit(base_portfolio_state: PortfolioState) -> None:
    """Test maximum concurrent open positions count limit."""
    ts = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)

    # Portfolio has 2 open positions (limit = 2)
    maxed_state = base_portfolio_state.model_copy(
        update={
            "positions": {
                "SYM1": Position(
                    symbol="SYM1",
                    side=PositionSide.LONG,
                    quantity=1.0,
                    entry_price=10.0,
                ),
                "SYM2": Position(
                    symbol="SYM2",
                    side=PositionSide.LONG,
                    quantity=1.0,
                    entry_price=10.0,
                ),
            }
        }
    )
    config = RiskConfig(max_open_positions=2)
    engine = RiskEngine(config=config)

    # 1. Open 3rd symbol -> REJECTED
    sig_sym3 = TradingSignal(
        signal_id="s3",
        timestamp=ts,
        symbol="SYM3",
        action=SignalAction.LONG,
        confidence=0.8,
        strategy_name="prob",
        strategy_version="v1",
        reason="Open new symbol",
    )
    res = engine.evaluate(sig_sym3, maxed_state, current_price=50.0)
    assert res.approved is False
    assert "reached maximum limit" in res.reason

    # 2. Reversal or closing on SYM1 -> ALLOWED
    sig_sym1 = TradingSignal(
        signal_id="s1",
        timestamp=ts,
        symbol="SYM1",
        action=SignalAction.SHORT,
        confidence=0.8,
        strategy_name="prob",
        strategy_version="v1",
        reason="Close SYM1",
    )
    res_close = engine.evaluate(sig_sym1, maxed_state, current_price=10.0)
    assert res_close.approved is True


def test_duplicate_direction_rejection(
    base_portfolio_state: PortfolioState,
) -> None:
    """Test duplicate directional signal does not generate new order."""
    ts = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    long_state = base_portfolio_state.model_copy(
        update={
            "positions": {
                "NIFTY": Position(
                    symbol="NIFTY",
                    side=PositionSide.LONG,
                    quantity=1.0,
                    entry_price=100.0,
                )
            }
        }
    )
    engine = RiskEngine()
    sig = TradingSignal(
        signal_id="s_dup",
        timestamp=ts,
        symbol="NIFTY",
        action=SignalAction.LONG,
        confidence=0.8,
        strategy_name="prob",
        strategy_version="v1",
        reason="Duplicate long",
    )
    decision = engine.evaluate(sig, long_state, current_price=105.0)
    assert decision.approved is False
    assert "already LONG" in decision.reason


def test_no_lookahead_risk_evaluation(
    base_portfolio_state: PortfolioState,
) -> None:
    """Verify risk decision strictly uses state available at evaluation timestamp."""
    ts = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    sig = TradingSignal(
        signal_id="s_test",
        timestamp=ts,
        symbol="NIFTY",
        action=SignalAction.LONG,
        confidence=0.8,
        strategy_name="prob",
        strategy_version="v1",
        reason="Lookahead test",
    )
    engine = RiskEngine(config=RiskConfig(fixed_quantity=2.0))
    decision = engine.evaluate(sig, base_portfolio_state, current_price=100.0)

    assert decision.approved is True
    assert decision.approved_quantity == 2.0
    assert decision.timestamp == ts


def test_determinism_across_runs(
    base_portfolio_state: PortfolioState,
) -> None:
    """Verify repeated evaluations with same state produce identical decisions."""
    ts = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    sig = TradingSignal(
        signal_id="s1",
        timestamp=ts,
        symbol="NIFTY",
        action=SignalAction.LONG,
        confidence=0.75,
        strategy_name="prob",
        strategy_version="v1",
        reason="Determinism test",
    )
    engine1 = RiskEngine()
    engine2 = RiskEngine()

    dec1 = engine1.evaluate(sig, base_portfolio_state, 100.0)
    dec2 = engine2.evaluate(sig, base_portfolio_state, 100.0)

    assert dec1.approved == dec2.approved
    assert dec1.approved_quantity == dec2.approved_quantity
    assert dec1.reason == dec2.reason


def test_end_to_end_backtest_with_risk_engine(tmp_path: Path) -> None:
    """Test full BacktestEngine integrated with RiskEngine and risk artifacts."""
    base_time = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    candles = [
        Candle(
            timestamp=base_time + timedelta(minutes=5 * i),
            symbol="NIFTY",
            timeframe=TimeFrame.FIVE_MINUTES,
            open=100.0 + i * 2,
            high=102.0 + i * 2,
            low=99.0 + i * 2,
            close=101.0 + i * 2,
            volume=1000,
        )
        for i in range(5)
    ]

    signals = [
        TradingSignal(
            signal_id="s1",
            timestamp=candles[0].timestamp,
            symbol="NIFTY",
            action=SignalAction.LONG,
            confidence=0.8,
            strategy_name="prob",
            strategy_version="v1",
            reason="LONG",
        )
    ]

    risk_engine = RiskEngine(
        config=RiskConfig(fixed_quantity=1.0, max_position_pct=0.20)
    )
    backtest_engine = BacktestEngine(
        config=BacktestConfig(initial_capital=100_000.0),
        risk_engine=risk_engine,
    )

    out_dir = tmp_path / "bt_risk_out"
    result = backtest_engine.run(
        candles=candles,
        signals=signals,
        artifacts_base_dir=out_dir,
        save_artifacts=True,
    )

    assert result.metrics.initial_capital == 100_000.0
    run_dir = list(out_dir.glob("bt_*"))[0]
    assert (run_dir / "risk_decisions.csv").is_file()

    dec_df = pd.read_csv(run_dir / "risk_decisions.csv")
    assert not dec_df.empty
    assert bool(dec_df.iloc[0]["approved"]) is True
    assert dec_df.iloc[0]["approved_quantity"] == 1.0


def test_risk_cli_evaluate(tmp_path: Path) -> None:
    """Test risk CLI evaluate command execution."""
    ts0 = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    ts1 = datetime(2026, 1, 2, 9, 20, tzinfo=UTC_TZ)

    candles_file = tmp_path / "candles.csv"
    pd.DataFrame(
        {
            "timestamp": [ts0.isoformat(), ts1.isoformat()],
            "symbol": ["NIFTY", "NIFTY"],
            "timeframe": ["5m", "5m"],
            "open": [100.0, 105.0],
            "high": [102.0, 107.0],
            "low": [99.0, 104.0],
            "close": [101.0, 106.0],
            "volume": [1000, 1200],
        }
    ).to_csv(candles_file, index=False)

    signals_file = tmp_path / "signals.csv"
    pd.DataFrame(
        {
            "signal_id": ["s1"],
            "timestamp": [ts0.isoformat()],
            "symbol": ["NIFTY"],
            "action": ["LONG"],
            "confidence": [0.75],
            "strategy_name": ["probability_threshold"],
            "strategy_version": ["v1"],
            "reason": ["LONG test"],
            "model_name": ["logistic_regression"],
            "model_version": ["v1"],
        }
    ).to_csv(signals_file, index=False)

    out_file = tmp_path / "audit_decisions.csv"
    exit_code = run_risk_evaluation(
        signals_file=signals_file,
        candles_file=candles_file,
        initial_capital=100_000.0,
        output_file=out_file,
    )
    assert exit_code == 0
    assert out_file.is_file()
