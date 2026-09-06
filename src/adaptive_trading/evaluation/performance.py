"""Out-of-sample performance aggregation and statistical compounding across windows."""

from collections.abc import Sequence
import logging
import statistics

from adaptive_trading.evaluation.config import EvaluationConfig
from adaptive_trading.evaluation.models import (
    SampleAdequacy,
    WalkForwardSummary,
    WindowEvaluationResult,
    WindowEvaluationStatus,
)

logger = logging.getLogger(__name__)


class PerformanceAggregator:
    """Aggregates out-of-sample metrics across walk-forward test windows."""

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self.config = config or EvaluationConfig()

    def aggregate(
        self, window_results: Sequence[WindowEvaluationResult]
    ) -> WalkForwardSummary:
        """Calculate compounded total return and aggregate statistics across windows.

        Note:
            We do NOT calculate a naive average of final equity figures.
            Compounded out-of-sample return across K independent test windows is:
            R_total = (Product_{i=1..K} (1 + R_i / 100) - 1) * 100%

        Args:
            window_results: List of per-window evaluation results.

        Returns:
            WalkForwardSummary: Aggregated out-of-sample performance summary.
        """
        total_windows = len(window_results)
        successful = [
            w
            for w in window_results
            if w.status == WindowEvaluationStatus.SUCCESS
        ]
        failed = [
            w
            for w in window_results
            if w.status == WindowEvaluationStatus.FAILED
        ]

        if not successful:
            return WalkForwardSummary(
                number_of_windows=total_windows,
                successful_windows=0,
                failed_windows=len(failed),
                total_return_pct=0.0,
                average_window_return=0.0,
                median_window_return=0.0,
                average_drawdown=0.0,
                worst_drawdown=0.0,
                total_trades=0,
                average_trades_per_window=0.0,
                overall_win_rate=0.0,
                sample_adequacy=SampleAdequacy.LOW_SAMPLE,
            )

        returns = [w.total_return_pct for w in successful]
        drawdowns = [w.max_drawdown_pct for w in successful]
        trades = [w.trade_count for w in successful]
        winning_trades = sum(w.winning_trades for w in successful)
        total_trades = sum(trades)

        # 1. Compounded Out-of-Sample Return Calculation
        compounded_factor = 1.0
        for r in returns:
            compounded_factor *= 1.0 + (r / 100.0)
        total_return_pct = (compounded_factor - 1.0) * 100.0

        avg_return = statistics.mean(returns) if returns else 0.0
        median_return = statistics.median(returns) if returns else 0.0
        avg_drawdown = statistics.mean(drawdowns) if drawdowns else 0.0
        worst_drawdown = max(drawdowns) if drawdowns else 0.0
        avg_trades = (total_trades / len(successful)) if successful else 0.0
        overall_win_rate = (
            (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0
        )

        # 2. Sample Adequacy Evaluation
        is_adequate = (
            len(successful) >= self.config.min_windows_for_adequacy
            and total_trades >= self.config.min_trades_for_adequacy
        )
        sample_adequacy = (
            SampleAdequacy.ADEQUATE_SAMPLE
            if is_adequate
            else SampleAdequacy.LOW_SAMPLE
        )

        return WalkForwardSummary(
            number_of_windows=total_windows,
            successful_windows=len(successful),
            failed_windows=len(failed),
            total_return_pct=round(total_return_pct, 4),
            average_window_return=round(avg_return, 4),
            median_window_return=round(median_return, 4),
            average_drawdown=round(avg_drawdown, 4),
            worst_drawdown=round(worst_drawdown, 4),
            total_trades=total_trades,
            average_trades_per_window=round(avg_trades, 2),
            overall_win_rate=round(overall_win_rate, 2),
            sample_adequacy=sample_adequacy,
        )

