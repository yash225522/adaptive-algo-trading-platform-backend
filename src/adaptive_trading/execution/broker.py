"""Abstract broker interface defining uniform execution operations."""

from abc import ABC, abstractmethod

from adaptive_trading.execution.models import (
    Order,
    OrderRequest,
    PaperAccount,
    PaperPosition,
)


class Broker(ABC):
    """Abstract interface governing order submission and account queries."""

    @abstractmethod
    def submit_order(self, request: OrderRequest) -> Order:
        """Submit a new order request to the broker.

        Args:
            request: Standardized OrderRequest specification.

        Returns:
            Order: Current state of the order after broker acceptance/rejection.
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> Order:
        """Request cancellation of an existing order.

        Args:
            order_id: Unique broker order identifier.

        Returns:
            Order: Updated state of the cancelled order.
        """

    @abstractmethod
    def get_order(self, order_id: str) -> Order | None:
        """Retrieve order details by broker order identifier.

        Args:
            order_id: Broker order identifier.

        Returns:
            Order | None: Order record if found, else None.
        """

    @abstractmethod
    def get_order_by_client_id(self, client_order_id: str) -> Order | None:
        """Retrieve order details by client idempotency identifier.

        Args:
            client_order_id: Client-assigned unique order key.

        Returns:
            Order | None: Order record if found, else None.
        """

    @abstractmethod
    def get_positions(self) -> list[PaperPosition]:
        """Query all currently tracked asset positions.

        Returns:
            list[PaperPosition]: Active and closed position records.
        """

    @abstractmethod
    def get_account(self) -> PaperAccount:
        """Query the current state of the trading account.

        Returns:
            PaperAccount: Current cash, equity, buying power, and P&L.
        """
