"""Execution service orchestrating RiskDecision submission to the Broker interface."""

import logging

from adaptive_trading.execution.angelone_broker import AngelOneBroker
from adaptive_trading.execution.broker import Broker
from adaptive_trading.execution.config import ExecutionConfig, ExecutionMode
from adaptive_trading.execution.models import (
    Order,
    OrderRequest,
    OrderSide,
    OrderType,
    ProductType,
)
from adaptive_trading.execution.paper_broker import PaperBroker
from adaptive_trading.risk.models import RiskDecision
from adaptive_trading.strategy.models import SignalAction

logger = logging.getLogger(__name__)


class ExecutionService:
    """Orchestrates order creation from risk decisions to the Broker interface."""

    def __init__(
        self,
        broker: Broker | None = None,
        config: ExecutionConfig | None = None,
    ) -> None:
        self.config = config or ExecutionConfig()
        if broker is not None:
            self.broker = broker
        else:
            if self.config.mode == ExecutionMode.PAPER:
                self.broker = PaperBroker(config=self.config)
            elif self.config.mode == ExecutionMode.LIVE:
                self.broker = AngelOneBroker(live_enabled=False)
            else:
                self.broker = PaperBroker(config=self.config)

    def execute_risk_decision(
        self,
        decision: RiskDecision,
        exchange: str | None = None,
        product_type: ProductType = ProductType.INTRADAY,
        current_price: float | None = None,
    ) -> Order | None:
        """Submit an approved RiskDecision to the active broker.

        Args:
            decision: RiskDecision produced by RiskEngine.
            exchange: Target exchange (defaults to config default_exchange).
            product_type: Product classification (INTRADAY/DELIVERY).
            current_price: Optional reference market price.

        Returns:
            Order | None: Resulting Order if approved and submitted, else None.
        """
        if not decision.approved or decision.approved_quantity <= 0:
            logger.debug(
                "ExecutionService skipping rejected/zero-quantity decision %s: %s",
                decision.decision_id,
                decision.reason,
            )
            return None

        if decision.action == SignalAction.LONG:
            side = OrderSide.BUY
        elif decision.action == SignalAction.SHORT:
            side = OrderSide.SELL
        else:
            logger.debug("ExecutionService ignoring action %s", decision.action)
            return None

        target_exchange = exchange or self.config.default_exchange

        request = OrderRequest(
            client_order_id=decision.decision_id,
            symbol=decision.symbol,
            exchange=target_exchange,
            side=side,
            quantity=decision.approved_quantity,
            order_type=OrderType.MARKET,
            price=current_price,
            product_type=product_type,
            timestamp=decision.timestamp,
        )

        logger.info(
            "ExecutionService submitting %s %s qty=%.2f on %s (Decision ID: %s)",
            side.value,
            decision.symbol,
            decision.approved_quantity,
            target_exchange,
            decision.decision_id,
        )
        return self.broker.submit_order(request)
