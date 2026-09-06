"""Angel One Broker adapter boundary establishing live order payload translation."""

import logging
from typing import Any

from adaptive_trading.execution.broker import Broker
from adaptive_trading.execution.exceptions import LiveTradingNotEnabledError
from adaptive_trading.execution.models import (
    Order,
    OrderRequest,
    OrderSide,
    OrderType,
    PaperAccount,
    PaperPosition,
    ProductType,
)

logger = logging.getLogger(__name__)


class AngelOneBroker(Broker):
    """Angel One SmartAPI broker adapter boundary.

    Safety Notice: Live order execution is explicitly disabled.
    Any attempt to submit or cancel real orders raises LiveTradingNotEnabledError.
    """

    def __init__(self, live_enabled: bool = False) -> None:
        self.live_enabled = live_enabled

    def build_order_payload(
        self,
        request: OrderRequest,
        symbol_token: str = "0",
    ) -> dict[str, Any]:
        """Translate standardized OrderRequest into Angel One SmartAPI payload.

        Args:
            request: Standardized OrderRequest.
            symbol_token: Resolved Angel One instrument token.

        Returns:
            dict[str, Any]: Formatted SmartAPI order dictionary.
        """
        txn_type = "BUY" if request.side == OrderSide.BUY else "SELL"
        ord_type = "MARKET" if request.order_type == OrderType.MARKET else "LIMIT"
        prod_type = (
            "INTRADAY" if request.product_type == ProductType.INTRADAY else "DELIVERY"
        )

        return {
            "variety": "NORMAL",
            "tradingsymbol": request.symbol,
            "symboltoken": symbol_token,
            "transactiontype": txn_type,
            "exchange": request.exchange,
            "ordertype": ord_type,
            "producttype": prod_type,
            "duration": "DAY",
            "price": str(request.price) if request.price is not None else "0",
            "quantity": str(int(request.quantity)),
        }

    def submit_order(self, request: OrderRequest) -> Order:
        """Attempting to submit live order raises LiveTradingNotEnabledError."""
        raise LiveTradingNotEnabledError(
            "Live trading execution with Angel One is disabled in this environment. "
            "Step 17 supports Paper Trading only."
        )

    def cancel_order(self, order_id: str) -> Order:
        """Attempting to cancel live order raises LiveTradingNotEnabledError."""
        raise LiveTradingNotEnabledError(
            "Live order cancellation is disabled in this environment."
        )

    def get_order(self, order_id: str) -> Order | None:
        """Query live order."""
        raise LiveTradingNotEnabledError(
            "Live order query is disabled in this environment."
        )

    def get_order_by_client_id(self, client_order_id: str) -> Order | None:
        """Query live order by client order ID."""
        raise LiveTradingNotEnabledError(
            "Live order query is disabled in this environment."
        )

    def get_positions(self) -> list[PaperPosition]:
        """Query live positions."""
        raise LiveTradingNotEnabledError(
            "Live positions query is disabled in this environment."
        )

    def get_account(self) -> PaperAccount:
        """Query live account funds."""
        raise LiveTradingNotEnabledError(
            "Live account query is disabled in this environment."
        )
