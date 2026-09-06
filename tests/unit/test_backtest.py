"""Unit tests for the Backtesting Engine, portfolio accounting, and metrics."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from adaptive_trading.backtesting.cli import run_backtest_pipeline
from adaptive_trading.backtesting.config import BacktestConfig
from adaptive_trading.backtesting.engine import BacktestEngine
from adaptive_trading.backtesting.exceptions import (
    BacktestConfigError,
    InsufficientDataError,
)
from adaptive_trading.backtesting.execution import SimulatedExecutionHandler
from adaptive_trading.backtesting.metrics import calculate_performance_metrics
from adaptive_trading.backtesting.models import (
    BacktestFill,
    BacktestOrder,
    BacktestTrade,
    EquityPoint,
    PositionSide,
)
from adaptive_trading.backtesting.portfolio import PortfolioTracker
from adaptive_trading.common.config import TimeFrame
from adaptive_trading.domain.market import Candle
from adaptive_trading.domain.trading import OrderSide
from adaptive_trading.strategy.models import SignalAction, TradingSignal

UTC_TZ = timezone.utc


@pytest.fixture
def sample_candles() -> list[Candle]:
    """Create a deterministic sequence of 6 5-minute market candles."""
    base_time = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    # Price trajectory: 100 -> 105 -> 110 -> 102 -> 108 -> 112
    ohlcv_data = [
        (100.0, 102.0, 99.0, 101.0, 1000),  # T0: open=100.0, close=101.0
        (101.0, 106.0, 100.5, 105.0, 1500),  # T1: open=101.0, close=105.0
        (105.0, 111.0, 104.0, 110.0, 2000),  # T2: open=105.0, close=110.0
        (110.0, 110.5, 101.0, 102.0, 1800),  # T3: open=110.0, close=102.0
        (102.0, 109.0, 101.5, 108.0, 1600),  # T4: open=102.0, close=108.0
        (108.0, 113.0, 107.5, 112.0, 1400),  # T5: open=108.0, close=112.0
    ]

    candles: list[Candle] = []
    for i, (op, hi, lo, cl, vol) in enumerate(ohlcv_data):
        candles.append(
            Candle(
                timestamp=base_time + timedelta(minutes=5 * i),
                symbol="NIFTY",
                timeframe=TimeFrame.FIVE_MINUTES,
                open=op,
                high=hi,
                low=lo,
                close=cl,
                volume=vol,
            )
        )
    return candles


def test_backtest_config_valid_and_invalid() -> None:
    """Test BacktestConfig validation rules."""
    config = BacktestConfig(
        initial_capital=50_000.0,
        commission_bps=2.0,
        slippage_bps=4.0,
        fixed_quantity=10.0,
        execution_timing="NEXT_OPEN",
    )
    assert config.initial_capital == 50_000.0
    assert config.commission_bps == 2.0

    # Negative capital
    with pytest.raises(ValueError):
        BacktestConfig(initial_capital=-100.0)

    # Invalid execution timing
    with pytest.raises(BacktestConfigError, match="Unsupported execution_timing"):
        BacktestConfig(execution_timing="INVALID")


def test_simulated_execution_handler_slippage_and_commission() -> None:
    """Test slippage and commission calculations on market order fills."""
    config = BacktestConfig(
        commission_bps=10.0,  # 0.10% = 0.001
        slippage_bps=20.0,  # 0.20% = 0.002
    )
    handler = SimulatedExecutionHandler(config=config)
    ts = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)

    candle = Candle(
        timestamp=ts,
        symbol="NIFTY",
        timeframe=TimeFrame.FIVE_MINUTES,
        open=100.0,
        high=105.0,
        low=99.0,
        close=104.0,
        volume=1000,
    )

    # 1. BUY Order (price increases by slippage)
    buy_order = BacktestOrder(
        timestamp=ts,
        symbol="NIFTY",
        side=OrderSide.BUY,
        quantity=2.0,
    )
    buy_fill = handler.simulate_fill(buy_order, candle)
    assert buy_fill.price == pytest.approx(100.20)  # 100 * (1 + 0.002)
    assert buy_fill.slippage == pytest.approx(0.20)
    assert buy_fill.commission == pytest.approx(100.20 * 2.0 * 0.001)

    # 2. SELL Order (price decreases by slippage)
    sell_order = BacktestOrder(
        timestamp=ts,
        symbol="NIFTY",
        side=OrderSide.SELL,
        quantity=2.0,
    )
    sell_fill = handler.simulate_fill(sell_order, candle)
    assert sell_fill.price == pytest.approx(99.80)  # 100 * (1 - 0.002)
    assert sell_fill.slippage == pytest.approx(0.20)
    assert sell_fill.commission == pytest.approx(99.80 * 2.0 * 0.001)


def test_portfolio_position_transitions_long_trade() -> None:
    """Test FLAT -> LONG -> FLAT lifecycle with zero fees for clean accounting."""
    config = BacktestConfig(
        initial_capital=100_000.0,
        commission_bps=0.0,
        slippage_bps=0.0,
    )
    portfolio = PortfolioTracker(config=config)
    t1 = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    t2 = datetime(2026, 1, 2, 9, 20, tzinfo=UTC_TZ)

    # 1. Open LONG at 100.0
    entry_fill = BacktestFill(
        order_id="o1",
        timestamp=t1,
        symbol="NIFTY",
        side=OrderSide.BUY,
        quantity=1.0,
        price=100.0,
        commission=0.0,
        slippage=0.0,
    )
    portfolio.open_position(entry_fill, side=PositionSide.LONG)
    assert portfolio.cash == 99_900.0
    assert portfolio.get_position("NIFTY").side == PositionSide.LONG

    # 2. Close LONG at 110.0
    exit_fill = BacktestFill(
        order_id="o2",
        timestamp=t2,
        symbol="NIFTY",
        side=OrderSide.SELL,
        quantity=1.0,
        price=110.0,
        commission=0.0,
        slippage=0.0,
    )
    trade = portfolio.close_position("NIFTY", exit_fill, holding_bars=1)
    assert trade is not None
    assert trade.gross_pnl == 10.0
    assert trade.net_pnl == 10.0
    assert trade.return_pct == 10.0
    assert portfolio.cash == 100_010.0
    assert portfolio.realized_pnl == 10.0
    assert portfolio.get_position("NIFTY").side == PositionSide.FLAT


def test_portfolio_position_transitions_short_trade() -> None:
    """Test FLAT -> SHORT -> FLAT lifecycle."""
    config = BacktestConfig(
        initial_capital=100_000.0,
        commission_bps=0.0,
        slippage_bps=0.0,
    )
    portfolio = PortfolioTracker(config=config)
    t1 = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    t2 = datetime(2026, 1, 2, 9, 20, tzinfo=UTC_TZ)

    # 1. Open SHORT at 110.0
    entry_fill = BacktestFill(
        order_id="o1",
        timestamp=t1,
        symbol="NIFTY",
        side=OrderSide.SELL,
        quantity=1.0,
        price=110.0,
        commission=0.0,
        slippage=0.0,
    )
    portfolio.open_position(entry_fill, side=PositionSide.SHORT)
    assert portfolio.cash == 100_110.0
    assert portfolio.get_position("NIFTY").side == PositionSide.SHORT

    # 2. Close SHORT at 100.0 (Profitable short by 10.0)
    exit_fill = BacktestFill(
        order_id="o2",
        timestamp=t2,
        symbol="NIFTY",
        side=OrderSide.BUY,
        quantity=1.0,
        price=100.0,
        commission=0.0,
        slippage=0.0,
    )
    trade = portfolio.close_position("NIFTY", exit_fill, holding_bars=1)
    assert trade is not None
    assert trade.gross_pnl == 10.0
    assert trade.net_pnl == 10.0
    assert portfolio.cash == 100_010.0
    assert portfolio.realized_pnl == 10.0


def test_end_to_end_backtest_engine_execution(
    sample_candles: list[Candle],
) -> None:
    """Test complete deterministic backtest with signal reversals."""
    # Signals:
    # T0 (09:15) -> LONG  (executes at T1 open=101.0)
    # T2 (09:25) -> SHORT (executes at T3 open=110.0:
    #                      closes LONG at 110, opens SHORT at 110)
    signals = [
        TradingSignal(
            signal_id="s1",
            timestamp=sample_candles[0].timestamp,
            symbol="NIFTY",
            action=SignalAction.LONG,
            confidence=0.75,
            strategy_name="probability_threshold",
            strategy_version="v1",
            reason="LONG signal",
        ),
        TradingSignal(
            signal_id="s2",
            timestamp=sample_candles[2].timestamp,
            symbol="NIFTY",
            action=SignalAction.SHORT,
            confidence=0.70,
            strategy_name="probability_threshold",
            strategy_version="v1",
            reason="SHORT signal",
        ),
    ]

    config = BacktestConfig(
        initial_capital=100_000.0,
        commission_bps=0.0,
        slippage_bps=0.0,
        fixed_quantity=1.0,
    )
    engine = BacktestEngine(config=config)
    result = engine.run(
        candles=sample_candles,
        signals=signals,
        save_artifacts=False,
    )

    assert len(result.trades) == 1
    trade1 = result.trades[0]
    assert trade1.side == PositionSide.LONG
    assert trade1.entry_price == 101.0
    assert trade1.exit_price == 110.0
    assert trade1.net_pnl == 9.0  # 110.0 - 101.0

    # Active SHORT opened at T3 open (110.0) and marked to market at T5 close (112.0)
    # Unrealized PnL = (110.0 - 112.0) = -2.0
    # Final equity = 100_000 + 9.0 (realized) - 2.0 (unrealized) = 100_007.0
    assert result.metrics.final_equity == pytest.approx(100_007.0)
    assert result.metrics.net_pnl == pytest.approx(7.0)
    assert len(result.equity_curve) == len(sample_candles)


def test_missing_subsequent_candle_no_execution(
    sample_candles: list[Candle],
) -> None:
    """Verify signal on the final candle does NOT execute due to no T+1 candle."""
    # Signal only at the last candle
    signals = [
        TradingSignal(
            signal_id="s_last",
            timestamp=sample_candles[-1].timestamp,
            symbol="NIFTY",
            action=SignalAction.LONG,
            confidence=0.80,
            strategy_name="probability_threshold",
            strategy_version="v1",
            reason="Final bar signal",
        )
    ]
    engine = BacktestEngine()
    result = engine.run(
        candles=sample_candles,
        signals=signals,
        save_artifacts=False,
    )
    # No trades executed because no candle after sample_candles[-1]
    assert len(result.trades) == 0
    assert result.metrics.final_equity == 100_000.0


def test_calculate_performance_metrics_drawdown_and_stats() -> None:
    """Test performance metrics and maximum drawdown computation."""
    initial_cap = 100_000.0
    t0 = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)

    trades = [
        BacktestTrade(
            trade_id="t1",
            symbol="NIFTY",
            side=PositionSide.LONG,
            entry_timestamp=t0,
            exit_timestamp=t0 + timedelta(minutes=5),
            entry_price=100.0,
            exit_price=110.0,
            quantity=1.0,
            gross_pnl=10.0,
            commission=1.0,
            slippage_cost=0.5,
            net_pnl=8.5,
            return_pct=8.5,
            holding_bars=1,
        ),
        BacktestTrade(
            trade_id="t2",
            symbol="NIFTY",
            side=PositionSide.SHORT,
            entry_timestamp=t0 + timedelta(minutes=10),
            exit_timestamp=t0 + timedelta(minutes=15),
            entry_price=100.0,
            exit_price=105.0,
            quantity=1.0,
            gross_pnl=-5.0,
            commission=1.0,
            slippage_cost=0.5,
            net_pnl=-6.5,
            return_pct=-6.5,
            holding_bars=1,
        ),
    ]

    # Equity curve with a peak and drawdown
    equity_curve = [
        EquityPoint(
            timestamp=t0,
            cash=100_000.0,
            market_value=0.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            equity=100_000.0,
        ),
        EquityPoint(
            timestamp=t0 + timedelta(minutes=5),
            cash=100_008.5,
            market_value=0.0,
            realized_pnl=8.5,
            unrealized_pnl=0.0,
            equity=100_008.5,  # Peak
        ),
        EquityPoint(
            timestamp=t0 + timedelta(minutes=15),
            cash=100_002.0,
            market_value=0.0,
            realized_pnl=2.0,
            unrealized_pnl=0.0,
            equity=100_002.0,  # Drawdown: 6.5
        ),
    ]

    metrics = calculate_performance_metrics(trades, equity_curve, initial_cap)
    assert metrics.total_trades == 2
    assert metrics.winning_trades == 1
    assert metrics.losing_trades == 1
    assert metrics.win_rate == 50.0
    assert metrics.gross_profit == 8.5
    assert metrics.gross_loss == 6.5
    assert metrics.profit_factor == pytest.approx(8.5 / 6.5, abs=0.01)
    assert metrics.max_drawdown_abs == 6.5
    assert metrics.max_drawdown_pct == pytest.approx(
        (6.5 / 100_008.5) * 100.0, abs=0.01
    )


def test_determinism_across_runs(sample_candles: list[Candle]) -> None:
    """Test running the same backtest produces identical results."""
    signals = [
        TradingSignal(
            signal_id="s1",
            timestamp=sample_candles[0].timestamp,
            symbol="NIFTY",
            action=SignalAction.LONG,
            confidence=0.75,
            strategy_name="probability_threshold",
            strategy_version="v1",
            reason="LONG",
        )
    ]
    engine1 = BacktestEngine()
    engine2 = BacktestEngine()

    res1 = engine1.run(candles=sample_candles, signals=signals, save_artifacts=False)
    res2 = engine2.run(candles=sample_candles, signals=signals, save_artifacts=False)

    assert res1.metrics.net_pnl == res2.metrics.net_pnl
    assert res1.metrics.max_drawdown_abs == res2.metrics.max_drawdown_abs
    assert len(res1.trades) == len(res2.trades)


def test_empty_inputs_raises_error() -> None:
    """Test engine raises InsufficientDataError on empty inputs."""
    engine = BacktestEngine()
    with pytest.raises(InsufficientDataError, match="empty candles"):
        engine.run(candles=[], signals=[])


def test_cli_backtest_execution(tmp_path: Path) -> None:
    """Test CLI backtest command execution and artifact generation."""
    ts0 = datetime(2026, 1, 2, 9, 15, tzinfo=UTC_TZ)
    ts1 = datetime(2026, 1, 2, 9, 20, tzinfo=UTC_TZ)

    # 1. Write candles CSV
    candles_file = tmp_path / "candles.csv"
    candles_df = pd.DataFrame(
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
    )
    candles_df.to_csv(candles_file, index=False)

    # 2. Write signals CSV
    signals_file = tmp_path / "signals.csv"
    signals_df = pd.DataFrame(
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
    )
    signals_df.to_csv(signals_file, index=False)

    out_dir = tmp_path / "bt_out"
    exit_code = run_backtest_pipeline(
        candles_file=candles_file,
        signals_file=signals_file,
        initial_capital=100_000.0,
        output_dir=out_dir,
        save_artifacts=True,
    )
    assert exit_code == 0
    subdirs = [p for p in out_dir.iterdir() if p.is_dir()]
    assert len(subdirs) == 1
    run_dir = subdirs[0]
    assert (run_dir / "metadata.json").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "trades.csv").is_file()
    assert (run_dir / "equity_curve.csv").is_file()
