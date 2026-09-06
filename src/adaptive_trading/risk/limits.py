"""Modular risk check definitions enforcing position sizing and portfolio limits."""

import logging

from adaptive_trading.portfolio.models import PortfolioState, PositionSide
from adaptive_trading.risk.config import RiskConfig
from adaptive_trading.risk.models import RiskCheckResult
from adaptive_trading.strategy.models import SignalAction, TradingSignal

logger = logging.getLogger(__name__)


def check_signal_validity(
    signal: TradingSignal,
    current_price: float,
) -> RiskCheckResult:
    """Verify that signal has an actionable direction and current price is positive."""
    if signal.action == SignalAction.NO_TRADE:
        return RiskCheckResult(
            name="signal_validity",
            passed=False,
            reason="NO_TRADE action generates no market order",
            metric_value=0.0,
            limit_value=0.0,
        )

    if current_price <= 0:
        return RiskCheckResult(
            name="signal_validity",
            passed=False,
            reason=f"Invalid non-positive market price: {current_price}",
            metric_value=current_price,
            limit_value=0.0,
        )

    return RiskCheckResult(
        name="signal_validity",
        passed=True,
        reason=f"Valid {signal.action.value} signal with price {current_price:.2f}",
        metric_value=current_price,
        limit_value=0.0,
    )


def check_daily_loss_limit(
    portfolio_state: PortfolioState,
    config: RiskConfig,
    is_risk_increasing: bool,
) -> RiskCheckResult:
    """Enforce maximum daily loss limit on risk-increasing trades."""
    if not config.enable_daily_loss_check or not config.max_daily_loss_pct:
        return RiskCheckResult(
            name="daily_loss_limit",
            passed=True,
            reason="Daily loss check disabled",
        )

    if not is_risk_increasing:
        return RiskCheckResult(
            name="daily_loss_limit",
            passed=True,
            reason="Risk-reducing or closing trade bypasses daily loss limit",
        )

    start_eq = portfolio_state.daily_start_equity
    if start_eq <= 0:
        return RiskCheckResult(
            name="daily_loss_limit",
            passed=False,
            reason="Invalid non-positive daily start equity",
            metric_value=start_eq,
        )

    daily_loss_pct = (start_eq - portfolio_state.equity) / start_eq
    limit_pct = config.max_daily_loss_pct

    if daily_loss_pct >= limit_pct:
        return RiskCheckResult(
            name="daily_loss_limit",
            passed=False,
            reason=(
                f"Daily loss ({daily_loss_pct * 100:.2f}%) exceeds "
                f"maximum limit ({limit_pct * 100:.2f}%)"
            ),
            metric_value=round(daily_loss_pct * 100, 2),
            limit_value=round(limit_pct * 100, 2),
        )

    return RiskCheckResult(
        name="daily_loss_limit",
        passed=True,
        reason=(
            f"Daily loss ({daily_loss_pct * 100:.2f}%) within "
            f"limit ({limit_pct * 100:.2f}%)"
        ),
        metric_value=round(daily_loss_pct * 100, 2),
        limit_value=round(limit_pct * 100, 2),
    )


def check_drawdown_limit(
    portfolio_state: PortfolioState,
    config: RiskConfig,
    is_risk_increasing: bool,
) -> RiskCheckResult:
    """Enforce maximum portfolio drawdown guard on risk-increasing trades."""
    if not config.enable_drawdown_check or not config.max_drawdown_pct:
        return RiskCheckResult(
            name="max_drawdown",
            passed=True,
            reason="Drawdown limit check disabled",
        )

    if not is_risk_increasing:
        return RiskCheckResult(
            name="max_drawdown",
            passed=True,
            reason="Risk-reducing or closing trade bypasses drawdown limit",
        )

    peak = portfolio_state.peak_equity
    if peak <= 0:
        return RiskCheckResult(
            name="max_drawdown",
            passed=False,
            reason="Invalid non-positive peak equity",
            metric_value=peak,
        )

    drawdown_pct = (peak - portfolio_state.equity) / peak
    limit_pct = config.max_drawdown_pct

    if drawdown_pct >= limit_pct:
        return RiskCheckResult(
            name="max_drawdown",
            passed=False,
            reason=(
                f"Drawdown ({drawdown_pct * 100:.2f}%) exceeds "
                f"maximum threshold ({limit_pct * 100:.2f}%)"
            ),
            metric_value=round(drawdown_pct * 100, 2),
            limit_value=round(limit_pct * 100, 2),
        )

    return RiskCheckResult(
        name="max_drawdown",
        passed=True,
        reason=(
            f"Drawdown ({drawdown_pct * 100:.2f}%) within "
            f"threshold ({limit_pct * 100:.2f}%)"
        ),
        metric_value=round(drawdown_pct * 100, 2),
        limit_value=round(limit_pct * 100, 2),
    )


def check_open_positions_limit(
    portfolio_state: PortfolioState,
    symbol: str,
    config: RiskConfig,
    is_risk_increasing: bool,
) -> RiskCheckResult:
    """Enforce maximum concurrent open positions count."""
    if not config.enable_open_position_check:
        return RiskCheckResult(
            name="open_positions_limit",
            passed=True,
            reason="Open positions check disabled",
        )

    if not is_risk_increasing:
        return RiskCheckResult(
            name="open_positions_limit",
            passed=True,
            reason="Risk-reducing or closing trade bypasses open positions limit",
        )

    current_pos = portfolio_state.positions.get(symbol)
    is_already_open = (
        current_pos is not None
        and current_pos.side != PositionSide.FLAT
        and current_pos.quantity > 0
    )

    open_count = sum(
        1
        for p in portfolio_state.positions.values()
        if p.side != PositionSide.FLAT and p.quantity > 0
    )

    # If opening a brand new symbol and limit is already reached
    if not is_already_open and open_count >= config.max_open_positions:
        return RiskCheckResult(
            name="open_positions_limit",
            passed=False,
            reason=(
                f"Active positions count ({open_count}) reached maximum limit "
                f"({config.max_open_positions})"
            ),
            metric_value=float(open_count),
            limit_value=float(config.max_open_positions),
        )

    return RiskCheckResult(
        name="open_positions_limit",
        passed=True,
        reason=(
            f"Active positions ({open_count}) within limit "
            f"({config.max_open_positions})"
        ),
        metric_value=float(open_count),
        limit_value=float(config.max_open_positions),
    )


def check_position_size_limit(
    quantity: float,
    current_price: float,
    portfolio_state: PortfolioState,
    config: RiskConfig,
    is_risk_increasing: bool,
) -> tuple[RiskCheckResult, float]:
    """Validate position value bounds and optionally reduce quantity to limit."""
    if not config.enable_position_limit_check or not is_risk_increasing:
        return (
            RiskCheckResult(
                name="position_size_limit",
                passed=True,
                reason="Position limit check bypassed or disabled",
            ),
            quantity,
        )

    equity = max(portfolio_state.equity, 0.0)
    proposed_value = quantity * current_price

    # Compute maximum allowed value
    limits: list[float] = []
    if config.max_position_value is not None:
        limits.append(config.max_position_value)
    if config.max_position_pct is not None:
        limits.append(config.max_position_pct * equity)

    if not limits:
        return (
            RiskCheckResult(
                name="position_size_limit",
                passed=True,
                reason="No position value limits configured",
            ),
            quantity,
        )

    max_allowed_val = min(limits)

    if proposed_value > max_allowed_val:
        if config.allow_quantity_reduction:
            reduced_qty = max_allowed_val / current_price if current_price > 0 else 0.0
            return (
                RiskCheckResult(
                    name="position_size_limit",
                    passed=True,
                    reason=(
                        f"Position value ({proposed_value:.2f}) capped to max "
                        f"({max_allowed_val:.2f}); quantity reduced from "
                        f"{quantity} to {reduced_qty:.4f}"
                    ),
                    metric_value=round(proposed_value, 2),
                    limit_value=round(max_allowed_val, 2),
                ),
                reduced_qty,
            )
        else:
            return (
                RiskCheckResult(
                    name="position_size_limit",
                    passed=False,
                    reason=(
                        f"Position value ({proposed_value:.2f}) exceeds max "
                        f"allowed ({max_allowed_val:.2f})"
                    ),
                    metric_value=round(proposed_value, 2),
                    limit_value=round(max_allowed_val, 2),
                ),
                0.0,
            )

    return (
        RiskCheckResult(
            name="position_size_limit",
            passed=True,
            reason=(
                f"Position value ({proposed_value:.2f}) within limit "
                f"({max_allowed_val:.2f})"
            ),
            metric_value=round(proposed_value, 2),
            limit_value=round(max_allowed_val, 2),
        ),
        quantity,
    )


def check_portfolio_exposure_limit(
    quantity: float,
    current_price: float,
    portfolio_state: PortfolioState,
    config: RiskConfig,
    is_risk_increasing: bool,
) -> tuple[RiskCheckResult, float]:
    """Validate total gross portfolio exposure and optionally reduce quantity."""
    if not config.enable_exposure_limit_check or not is_risk_increasing:
        return (
            RiskCheckResult(
                name="portfolio_exposure_limit",
                passed=True,
                reason="Exposure limit check bypassed or disabled",
            ),
            quantity,
        )

    if config.max_portfolio_exposure_pct is None:
        return (
            RiskCheckResult(
                name="portfolio_exposure_limit",
                passed=True,
                reason="No exposure limit configured",
            ),
            quantity,
        )

    equity = max(portfolio_state.equity, 0.0)
    max_exposure = config.max_portfolio_exposure_pct * equity
    current_exp = portfolio_state.portfolio_exposure
    proposed_add = quantity * current_price
    proposed_total = current_exp + proposed_add

    if proposed_total > max_exposure:
        if config.allow_quantity_reduction:
            available_exp = max(0.0, max_exposure - current_exp)
            reduced_qty = available_exp / current_price if current_price > 0 else 0.0
            if reduced_qty > 0:
                return (
                    RiskCheckResult(
                        name="portfolio_exposure_limit",
                        passed=True,
                        reason=(
                            f"Total exposure ({proposed_total:.2f}) exceeds "
                            f"max ({max_exposure:.2f}); quantity reduced to "
                            f"{reduced_qty:.4f}"
                        ),
                        metric_value=round(proposed_total, 2),
                        limit_value=round(max_exposure, 2),
                    ),
                    reduced_qty,
                )
            else:
                return (
                    RiskCheckResult(
                        name="portfolio_exposure_limit",
                        passed=False,
                        reason=(
                            f"Portfolio exposure ({proposed_total:.2f}) exceeds max "
                            f"limit ({max_exposure:.2f}) and no capacity remains"
                        ),
                        metric_value=round(proposed_total, 2),
                        limit_value=round(max_exposure, 2),
                    ),
                    0.0,
                )
        else:
            return (
                RiskCheckResult(
                    name="portfolio_exposure_limit",
                    passed=False,
                    reason=(
                        f"Total exposure ({proposed_total:.2f}) exceeds max "
                        f"limit ({max_exposure:.2f})"
                    ),
                    metric_value=round(proposed_total, 2),
                    limit_value=round(max_exposure, 2),
                ),
                0.0,
            )

    return (
        RiskCheckResult(
            name="portfolio_exposure_limit",
            passed=True,
            reason=(
                f"Total exposure ({proposed_total:.2f}) within limit "
                f"({max_exposure:.2f})"
            ),
            metric_value=round(proposed_total, 2),
            limit_value=round(max_exposure, 2),
        ),
        quantity,
    )
