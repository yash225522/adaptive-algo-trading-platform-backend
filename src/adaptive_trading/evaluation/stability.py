"""Performance stability analysis and consistency classification.

Evaluates out-of-sample performance dispersion across evaluation windows.
"""

import logging
import statistics
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from adaptive_trading.evaluation.models import (
    StabilityClassification,
    WindowEvaluationResult,
    WindowEvaluationStatus,
)

logger = logging.getLogger(__name__)


class StabilityReport(BaseModel):
    """Statistical stability metrics measuring performance consistency."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    positive_windows: int = Field(ge=0, description="Count of profitable test windows")
    negative_windows: int = Field(ge=0, description="Count of losing test windows")
    zero_windows: int = Field(
        ge=0, description="Count of scratch/zero return test windows"
    )
    win_window_ratio: float = Field(
        description="Percentage of positive test windows (%)"
    )
    best_window_return: float = Field(
        description="Maximum single-window percentage return"
    )
    worst_window_return: float = Field(
        description="Minimum single-window percentage return"
    )
    return_variance: float = Field(
        ge=0.0, description="Variance of window percentage returns"
    )
    return_std_dev: float = Field(
        ge=0.0, description="Standard deviation of window percentage returns"
    )
    drawdown_variance: float = Field(
        ge=0.0, description="Variance of window maximum drawdowns"
    )
    worst_drawdown: float = Field(
        ge=0.0, description="Maximum observed drawdown across all windows"
    )
    classification: StabilityClassification = Field(
        description="Consistency rating: STABLE, MODERATE, or UNSTABLE"
    )
    classification_reason: str = Field(
        description="Explanation for the assigned stability rating"
    )
    disclaimer: str = Field(
        default=(
            "Stability classification refers strictly to the empirical consistency "
            "of observed evaluation results across windows. It does not imply that a "
            "strategy is safe, profitable, or guaranteed to perform similarly live."
        )
    )


class PerformanceStabilityAnalyzer:
    """Evaluates the variance and consistency of out-of-sample window results."""

    def analyze(
        self, window_results: Sequence[WindowEvaluationResult]
    ) -> StabilityReport:
        """Analyze window return dispersion and assign stability classification.

        Args:
            window_results: Collection of evaluated WalkForwardWindow results.

        Returns:
            StabilityReport: Statistical dispersion report and classification.
        """
        successful = [
            w for w in window_results if w.status == WindowEvaluationStatus.SUCCESS
        ]

        if not successful:
            return StabilityReport(
                positive_windows=0,
                negative_windows=0,
                zero_windows=0,
                win_window_ratio=0.0,
                best_window_return=0.0,
                worst_window_return=0.0,
                return_variance=0.0,
                return_std_dev=0.0,
                drawdown_variance=0.0,
                worst_drawdown=0.0,
                classification=StabilityClassification.UNSTABLE,
                classification_reason="Zero successful windows completed",
            )

        returns = [w.total_return_pct for w in successful]
        drawdowns = [w.max_drawdown_pct for w in successful]

        pos_count = sum(1 for r in returns if r > 0.0)
        neg_count = sum(1 for r in returns if r < 0.0)
        zero_count = sum(1 for r in returns if r == 0.0)
        win_ratio = (pos_count / len(successful)) * 100.0

        best_ret = max(returns)
        worst_ret = min(returns)
        worst_dd = max(drawdowns) if drawdowns else 0.0

        ret_var = statistics.variance(returns) if len(returns) > 1 else 0.0
        ret_std = statistics.stdev(returns) if len(returns) > 1 else 0.0
        dd_var = statistics.variance(drawdowns) if len(drawdowns) > 1 else 0.0

        # Classification rules
        if len(successful) < 2:
            classification = StabilityClassification.MODERATE
            reason = "Single window evaluated; variance cannot be estimated"
        elif win_ratio >= 60.0 and ret_std < 10.0 and worst_dd < 15.0:
            classification = StabilityClassification.STABLE
            reason = (
                f"High win-window ratio ({win_ratio:.1f}%), low return volatility "
                f"(std={ret_std:.2f}%), and controlled drawdown (max={worst_dd:.2f}%)"
            )
        elif win_ratio >= 40.0 and worst_dd < 25.0:
            classification = StabilityClassification.MODERATE
            reason = (
                f"Moderate win-window ratio ({win_ratio:.1f}%) and moderate "
                f"dispersion (std={ret_std:.2f}%, max_dd={worst_dd:.2f}%)"
            )
        else:
            classification = StabilityClassification.UNSTABLE
            reason = (
                f"Low win-window ratio ({win_ratio:.1f}%) or high drawdown/dispersion "
                f"(max_dd={worst_dd:.2f}%, std={ret_std:.2f}%)"
            )

        return StabilityReport(
            positive_windows=pos_count,
            negative_windows=neg_count,
            zero_windows=zero_count,
            win_window_ratio=round(win_ratio, 2),
            best_window_return=round(best_ret, 4),
            worst_window_return=round(worst_ret, 4),
            return_variance=round(ret_var, 4),
            return_std_dev=round(ret_std, 4),
            drawdown_variance=round(dd_var, 4),
            worst_drawdown=round(worst_dd, 4),
            classification=classification,
            classification_reason=reason,
        )
