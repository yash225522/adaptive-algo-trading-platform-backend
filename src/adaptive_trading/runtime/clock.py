"""Deterministic simulated clock providing event-time semantics for trading."""

from datetime import datetime, timezone

from adaptive_trading.runtime.exceptions import InvalidEventOrderError


class TradingClock:
    """Simulated event-time clock driven strictly by market event timestamps."""

    def __init__(self, initial_time: datetime | None = None) -> None:
        if initial_time is not None:
            if (
                initial_time.tzinfo is None
                or initial_time.tzinfo.utcoffset(initial_time) is None
            ):
                raise ValueError("Initial clock timestamp must be timezone-aware")
            self._current_time: datetime = initial_time
        else:
            self._current_time = datetime.min.replace(tzinfo=timezone.utc)

    def now(self) -> datetime:
        """Return the current simulated trading timestamp."""
        return self._current_time

    def advance_to(self, timestamp: datetime) -> None:
        """Advance the simulated clock forward to a new event timestamp.

        Args:
            timestamp: Next event timestamp (timezone-aware and >= current).

        Raises:
            ValueError: If timestamp lacks timezone information.
            InvalidEventOrderError: If timestamp is earlier than current clock time.
        """
        if timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp) is None:
            raise ValueError("Event timestamp must be timezone-aware")

        if timestamp < self._current_time:
            raise InvalidEventOrderError(
                f"Cannot move clock backwards from {self._current_time} to {timestamp}"
            )

        self._current_time = timestamp
