"""Deterministic risk management engine evaluating signals against portfolio limits."""

import logging
from datetime import datetime

from adaptive_trading.portfolio.models import PortfolioState, PositionSide
from adaptive_trading.risk.config import RiskConfig
from adaptive_trading.risk.limits import (
    check_daily_loss_limit,
    check_drawdown_limit,
    check_open_positions_limit,
    check_portfolio_exposure_limit,
    check_position_size_limit,
    check_signal_validity,
)
from adaptive_trading.risk.models import RiskCheckResult, RiskDecision
from adaptive_trading.risk.position_sizer import BasePositionSizer, FixedPositionSizer
from adaptive_trading.strategy.models import SignalAction, TradingSignal

logger = logging.getLogger(__name__)


class RiskEngine:
    """Evaluates strategy signals, applies position sizing, and enforces risk limits."""

    def __init__(
        self,
        config: RiskConfig | None = None,
        position_sizer: BasePositionSizer | None = None,
    ) -> None:
        self.config = config or RiskConfig()
        self.position_sizer = position_sizer or FixedPositionSizer()

    def evaluate(
        self,
        signal: TradingSignal,
        portfolio_state: PortfolioState,
        current_price: float,
        timestamp: datetime | None = None,
    ) -> RiskDecision:
        """Evaluate a trading signal against portfolio state and return RiskDecision.

        Strict deterministic order:
        1. Signal validity
        2. Existing portfolio state & action categorization
        3. Position sizing initial baseline
        4. Daily loss limit
        5. Maximum drawdown
        6. Open positions limit
        7. Position size limit
        8. Portfolio exposure limit
        9. Final approval

        Args:
            signal: Trading signal to evaluate.
            portfolio_state: Current authoritative portfolio state.
            current_price: Current market price of the symbol.
            timestamp: Evaluation timestamp (defaults to signal timestamp).

        Returns:
            RiskDecision: Complete structured risk evaluation result.
        """
        eval_ts = timestamp or signal.timestamp
        checks: list[RiskCheckResult] = []

        # 1. Signal Validity Check
        validity_check = check_signal_validity(signal, current_price)
        checks.append(validity_check)
        if not validity_check.passed:
            return RiskDecision(
                timestamp=eval_ts,
                symbol=signal.symbol,
                action=signal.action,
                requested_quantity=0.0,
                approved_quantity=0.0,
                approved=False,
                reason=validity_check.reason or "Invalid signal",
                risk_checks=checks,
            )

        # 2. Existing Portfolio State & Action Categorization
        current_pos = portfolio_state.positions.get(signal.symbol)
        pos_side = current_pos.side if current_pos else PositionSide.FLAT

        # Duplicate directional check
        if (signal.action == SignalAction.LONG and pos_side == PositionSide.LONG) or (
            signal.action == SignalAction.SHORT and pos_side == PositionSide.SHORT
        ):
            reason = (
                f"Position on {signal.symbol} is already "
                f"{pos_side.value}; maintaining position"
            )
            dup_check = RiskCheckResult(
                name="duplicate_direction",
                passed=False,
                reason=reason,
            )
            checks.append(dup_check)
            return RiskDecision(
                timestamp=eval_ts,
                symbol=signal.symbol,
                action=signal.action,
                requested_quantity=0.0,
                approved_quantity=0.0,
                approved=False,
                reason=reason,
                risk_checks=checks,
            )

        # Determine if this trade increases portfolio risk:
        # A trade from FLAT creates new risk exposure (is_risk_increasing = True).
        # An opposing signal exits/reduces exposure (is_risk_increasing = False).
        is_risk_increasing = pos_side == PositionSide.FLAT

        # 3. Position Sizing
        initial_quantity = self.position_sizer.calculate_quantity(
            signal=signal,
            portfolio_state=portfolio_state,
            current_price=current_price,
            config=self.config,
        )

        current_qty = initial_quantity

        # 4. Daily Loss Limit Check
        daily_loss_check = check_daily_loss_limit(
            portfolio_state=portfolio_state,
            config=self.config,
            is_risk_increasing=is_risk_increasing,
        )
        checks.append(daily_loss_check)
        if not daily_loss_check.passed:
            return RiskDecision(
                timestamp=eval_ts,
                symbol=signal.symbol,
                action=signal.action,
                requested_quantity=initial_quantity,
                approved_quantity=0.0,
                approved=False,
                reason=daily_loss_check.reason or "Daily loss limit exceeded",
                risk_checks=checks,
            )

        # 5. Maximum Drawdown Check
        drawdown_check = check_drawdown_limit(
            portfolio_state=portfolio_state,
            config=self.config,
            is_risk_increasing=is_risk_increasing,
        )
        checks.append(drawdown_check)
        if not drawdown_check.passed:
            return RiskDecision(
                timestamp=eval_ts,
                symbol=signal.symbol,
                action=signal.action,
                requested_quantity=initial_quantity,
                approved_quantity=0.0,
                approved=False,
                reason=drawdown_check.reason or "Maximum drawdown limit exceeded",
                risk_checks=checks,
            )

        # 6. Open Positions Limit Check
        open_pos_check = check_open_positions_limit(
            portfolio_state=portfolio_state,
            symbol=signal.symbol,
            config=self.config,
            is_risk_increasing=is_risk_increasing,
        )
        checks.append(open_pos_check)
        if not open_pos_check.passed:
            return RiskDecision(
                timestamp=eval_ts,
                symbol=signal.symbol,
                action=signal.action,
                requested_quantity=initial_quantity,
                approved_quantity=0.0,
                approved=False,
                reason=open_pos_check.reason or "Max open positions limit exceeded",
                risk_checks=checks,
            )

        # 7. Position Size Limit Check
        pos_size_check, current_qty = check_position_size_limit(
            quantity=current_qty,
            current_price=current_price,
            portfolio_state=portfolio_state,
            config=self.config,
            is_risk_increasing=is_risk_increasing,
        )
        checks.append(pos_size_check)
        if not pos_size_check.passed or current_qty <= 0:
            return RiskDecision(
                timestamp=eval_ts,
                symbol=signal.symbol,
                action=signal.action,
                requested_quantity=initial_quantity,
                approved_quantity=0.0,
                approved=False,
                reason=pos_size_check.reason or "Position size limit exceeded",
                risk_checks=checks,
            )

        # 8. Portfolio Exposure Limit Check
        exp_check, current_qty = check_portfolio_exposure_limit(
            quantity=current_qty,
            current_price=current_price,
            portfolio_state=portfolio_state,
            config=self.config,
            is_risk_increasing=is_risk_increasing,
        )
        checks.append(exp_check)
        if not exp_check.passed or current_qty <= 0:
            return RiskDecision(
                timestamp=eval_ts,
                symbol=signal.symbol,
                action=signal.action,
                requested_quantity=initial_quantity,
                approved_quantity=0.0,
                approved=False,
                reason=exp_check.reason or "Portfolio exposure limit exceeded",
                risk_checks=checks,
            )

        # 9. Final Approval
        approved_qty = round(current_qty, 4)
        decision_reason = (
            f"Approved {signal.action.value} trade for {approved_qty} "
            f"unit(s) on {signal.symbol}"
        )
        if approved_qty < initial_quantity:
            decision_reason += f" (reduced from {initial_quantity} to respect limits)"

        logger.debug(
            "Risk decision for %s on %s: Approved=%s, Qty=%.2f",
            signal.action.value,
            signal.symbol,
            True,
            approved_qty,
        )

        return RiskDecision(
            timestamp=eval_ts,
            symbol=signal.symbol,
            action=signal.action,
            requested_quantity=initial_quantity,
            approved_quantity=approved_qty,
            approved=True,
            reason=decision_reason,
            risk_checks=checks,
        )
