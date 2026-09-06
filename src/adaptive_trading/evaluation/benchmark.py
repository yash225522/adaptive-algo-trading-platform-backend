"""Benchmark evaluation and comparison models."""

from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
import logging

from pydantic import BaseModel, ConfigDict, Field

from adaptive_trading.domain.market import Candle

logger = logging.getLogger(__name__)


class BenchmarkType(StrEnum):
    """Type of baseline benchmark strategy."""

    BUY_AND_HOLD = "BUY_AND_HOLD"


class BenchmarkResult(BaseModel):
    """Performance metrics of a passive benchmark across a defined time period."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    benchmark_type: BenchmarkType = Field(default=BenchmarkType.BUY_AND_HOLD)
    start_price: float = Field(gt=0, description="Opening price at start of period")
    end_price: float = Field(gt=0, description="Closing price at end of period")
    benchmark_return_pct: float = Field(description="Total percentage return")
    benchmark_max_drawdown_pct: float = Field(
        ge=0.0, description="Peak-to-trough max drawdown percentage"
    )
    start_time: datetime | None = None
    end_time: datetime | None = None


class BenchmarkComparison(BaseModel):
    """Comparative analysis between active trading strategy and baseline benchmark."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy_return_pct: float = Field(
        description="Active strategy total return (%)"
    )
    benchmark_return_pct: float = Field(
        description="Passive benchmark return (%)"
    )
    absolute_difference_pct: float = Field(
        description="Strategy return minus benchmark return (%)"
    )
    relative_difference_pct: float | None = Field(
        default=None, description="Percentage outperformance relative to benchmark"
    )
    drawdown_comparison_pct: float = Field(
        description="Strategy Max Drawdown minus Benchmark Max Drawdown (%)"
    )
    outperformed_benchmark: bool = Field(
        description="Whether active strategy exceeded benchmark total return"
    )
    disclaimer: str = Field(
        default=(
            "Outperforming a passive buy-and-hold benchmark over historical "
            "periods does not guarantee future risk-adjusted profitability."
        )
    )


class BenchmarkEvaluator:
    """Evaluates passive buy-and-hold benchmark metrics across market candles."""

    def evaluate_buy_and_hold(
        self, candles: Sequence[Candle]
    ) -> BenchmarkResult:
        """Compute Buy & Hold return and max drawdown over candle sequence.

        Args:
            candles: Sequence of historical market candles.

        Returns:
            BenchmarkResult: Computed benchmark return and drawdown.
        """
        if not candles:
            return BenchmarkResult(
                start_price=1.0,
                end_price=1.0,
                benchmark_return_pct=0.0,
                benchmark_max_drawdown_pct=0.0,
            )

        start_price = float(candles[0].open)
        end_price = float(candles[-1].close)

        # Total Buy & Hold return
        ret_pct = (
            ((end_price - start_price) / start_price * 100.0)
            if start_price > 0
            else 0.0
        )

        # Max Drawdown across candle lows and closes
        peak = start_price
        max_dd_pct = 0.0

        for c in candles:
            if c.high > peak:
                peak = float(c.high)
            trough = float(c.low)
            dd_pct = ((peak - trough) / peak * 100.0) if peak > 0 else 0.0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        return BenchmarkResult(
            benchmark_type=BenchmarkType.BUY_AND_HOLD,
            start_price=round(start_price, 4),
            end_price=round(end_price, 4),
            benchmark_return_pct=round(ret_pct, 4),
            benchmark_max_drawdown_pct=round(max_dd_pct, 4),
            start_time=candles[0].timestamp,
            end_time=candles[-1].timestamp,
        )

    def compare(
        self,
        strategy_return_pct: float,
        strategy_max_drawdown_pct: float,
        candles: Sequence[Candle],
    ) -> BenchmarkComparison:
        """Compare strategy out-of-sample performance with Buy & Hold benchmark."""
        bench = self.evaluate_buy_and_hold(candles)
        abs_diff = strategy_return_pct - bench.benchmark_return_pct

        rel_diff = None
        if abs(bench.benchmark_return_pct) > 1e-6:
            rel_diff = round(
                (abs_diff / abs(bench.benchmark_return_pct)) * 100.0, 2
            )

        dd_diff = strategy_max_drawdown_pct - bench.benchmark_max_drawdown_pct
        outperformed = strategy_return_pct > bench.benchmark_return_pct

        return BenchmarkComparison(
            strategy_return_pct=round(strategy_return_pct, 4),
            benchmark_return_pct=round(bench.benchmark_return_pct, 4),
            absolute_difference_pct=round(abs_diff, 4),
            relative_difference_pct=rel_diff,
            drawdown_comparison_pct=round(dd_diff, 4),
            outperformed_benchmark=outperformed,
        )

