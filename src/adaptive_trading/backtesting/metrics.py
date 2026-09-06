"""Performance metrics calculation, trade statistics, and maximum drawdown."""

import logging
from collections.abc import Sequence

from adaptive_trading.backtesting.models import (
    BacktestTrade,
    EquityPoint,
    PerformanceMetrics,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PerformanceMetrics",
    "calculate_performance_metrics",
]


def calculate_performance_metrics(
    trades: Sequence[BacktestTrade],
    equity_curve: Sequence[EquityPoint],
    initial_capital: float,
) -> PerformanceMetrics:
    """Calculate comprehensive performance metrics from trades and equity history.

    Args:
        trades: List of completed round-trip BacktestTrade objects.
        equity_curve: Chronological sequence of EquityPoint snapshots.
        initial_capital: Starting simulation cash capital.

    Returns:
        PerformanceMetrics: Comprehensive summary metrics.
    """
    final_equity = equity_curve[-1].equity if equity_curve else initial_capital
    net_pnl = final_equity - initial_capital
    total_return_pct = (
        (net_pnl / initial_capital * 100.0) if initial_capital > 0 else 0.0
    )

    # 1. Maximum Drawdown Calculation
    peak = initial_capital
    max_dd_abs = 0.0
    max_dd_pct = 0.0

    for pt in equity_curve:
        if pt.equity > peak:
            peak = pt.equity
        dd_abs = peak - pt.equity
        dd_pct = (dd_abs / peak * 100.0) if peak > 0 else 0.0

        if dd_abs > max_dd_abs:
            max_dd_abs = dd_abs
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    # 2. Trade Statistics
    total_trades = len(trades)
    winning_trades = [t for t in trades if t.net_pnl > 0]
    losing_trades = [t for t in trades if t.net_pnl < 0]
    scratch_trades = [t for t in trades if t.net_pnl == 0]

    gross_profit = sum(t.net_pnl for t in winning_trades)
    gross_loss = abs(sum(t.net_pnl for t in losing_trades))

    if gross_loss > 0:
        profit_factor = round(gross_profit / gross_loss, 4)
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = None

    win_rate = (
        round((len(winning_trades) / total_trades) * 100.0, 2)
        if total_trades > 0
        else 0.0
    )
    avg_trade_pnl = (
        round(sum(t.net_pnl for t in trades) / total_trades, 4)
        if total_trades > 0
        else 0.0
    )
    avg_winner = round(gross_profit / len(winning_trades), 4) if winning_trades else 0.0
    avg_loser = round(-gross_loss / len(losing_trades), 4) if losing_trades else 0.0
    largest_winner = (
        round(max(t.net_pnl for t in trades), 4) if total_trades > 0 else 0.0
    )
    largest_loser = (
        round(min(t.net_pnl for t in trades), 4) if total_trades > 0 else 0.0
    )

    return PerformanceMetrics(
        initial_capital=round(initial_capital, 2),
        final_equity=round(final_equity, 2),
        net_pnl=round(net_pnl, 2),
        total_return_pct=round(total_return_pct, 4),
        total_trades=total_trades,
        winning_trades=len(winning_trades),
        losing_trades=len(losing_trades),
        scratch_trades=len(scratch_trades),
        win_rate=win_rate,
        gross_profit=round(gross_profit, 2),
        gross_loss=round(gross_loss, 2),
        profit_factor=profit_factor,
        average_trade_pnl=avg_trade_pnl,
        average_winner=avg_winner,
        average_loser=avg_loser,
        largest_winner=largest_winner,
        largest_loser=largest_loser,
        max_drawdown_abs=round(max_dd_abs, 2),
        max_drawdown_pct=round(max_dd_pct, 4),
    )
