"""Authoritative portfolio manager tracking equity, positions, and exposure."""

import logging
from datetime import date, datetime, timezone

from adaptive_trading.portfolio.models import PortfolioState, Position, PositionSide

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Maintains single authoritative portfolio state and accounts for all positions."""

    def __init__(self, initial_capital: float = 100_000.0) -> None:
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.realized_pnl = 0.0
        self.peak_equity = initial_capital
        self.daily_start_equity = initial_capital
        self.current_date: date | None = None
        self.positions: dict[str, Position] = {}
        self._last_state_timestamp: datetime = datetime.now(timezone.utc)
        self._entry_commissions: dict[str, float] = {}

    def get_position(self, symbol: str) -> Position:
        """Retrieve active position for a symbol or return a FLAT default."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def get_open_positions_count(self) -> int:
        """Return the number of currently active non-flat positions."""
        return sum(
            1
            for pos in self.positions.values()
            if pos.side != PositionSide.FLAT and pos.quantity > 0
        )

    def get_total_exposure(self) -> float:
        """Calculate total gross portfolio exposure (long + short market values)."""
        return sum(
            pos.market_value
            for pos in self.positions.values()
            if pos.side != PositionSide.FLAT and pos.quantity > 0
        )

    def mark_to_market(
        self,
        timestamp: datetime,
        current_prices: dict[str, float],
    ) -> PortfolioState:
        """Revalue open positions and update daily baselines.

        Args:
            timestamp: Current candle timestamp.
            current_prices: Mapping of symbol to current closing price.

        Returns:
            PortfolioState: Authoritative state snapshot.
        """
        self._last_state_timestamp = timestamp
        candle_date = timestamp.date()

        total_unrealized = 0.0
        total_exposure = 0.0

        for symbol, pos in self.positions.items():
            if pos.side == PositionSide.FLAT or pos.quantity == 0:
                continue

            current_price = current_prices.get(symbol, pos.entry_price)

            if pos.side == PositionSide.LONG:
                unrealized = (current_price - pos.entry_price) * pos.quantity
                mkt_val = current_price * pos.quantity
            elif pos.side == PositionSide.SHORT:
                unrealized = (pos.entry_price - current_price) * pos.quantity
                mkt_val = current_price * pos.quantity
            else:
                unrealized = 0.0
                mkt_val = 0.0

            self.positions[symbol] = pos.model_copy(
                update={
                    "unrealized_pnl": round(unrealized, 6),
                    "market_value": round(mkt_val, 6),
                }
            )
            total_unrealized += unrealized
            total_exposure += mkt_val

        current_equity = round(
            self.initial_capital + self.realized_pnl + total_unrealized, 6
        )

        # Handle calendar day rollover for daily loss calculation
        if self.current_date is None or candle_date != self.current_date:
            self.current_date = candle_date
            self.daily_start_equity = current_equity

        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        state = PortfolioState(
            timestamp=timestamp,
            cash=round(self.cash, 4),
            equity=current_equity,
            realized_pnl=round(self.realized_pnl, 4),
            unrealized_pnl=round(total_unrealized, 4),
            portfolio_exposure=round(total_exposure, 4),
            peak_equity=round(self.peak_equity, 4),
            daily_start_equity=round(self.daily_start_equity, 4),
            current_date=self.current_date,
            positions=dict(self.positions),
        )
        return state

    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        quantity: float,
        price: float,
        timestamp: datetime,
        commission: float = 0.0,
    ) -> None:
        """Open a new LONG or SHORT position."""
        if side == PositionSide.LONG:
            self.cash -= (price * quantity) + commission
        elif side == PositionSide.SHORT:
            self.cash += (price * quantity) - commission

        self.positions[symbol] = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=price,
            entry_timestamp=timestamp,
            unrealized_pnl=0.0,
            realized_pnl=self.realized_pnl,
            market_value=round(price * quantity, 6),
        )
        self._entry_commissions[symbol] = commission

        logger.debug(
            "Portfolio opened %s on %s: qty=%.2f @ %.2f",
            side.value,
            symbol,
            quantity,
            price,
        )

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_timestamp: datetime,
        exit_commission: float = 0.0,
    ) -> tuple[float, float, PositionSide]:
        """Close an open position and return (gross_pnl, net_pnl, closed_side)."""
        pos = self.get_position(symbol)
        if pos.side == PositionSide.FLAT or pos.quantity == 0:
            return 0.0, 0.0, PositionSide.FLAT

        entry_comm = self._entry_commissions.get(symbol, 0.0)
        total_comm = entry_comm + exit_commission

        if pos.side == PositionSide.LONG:
            gross_pnl = (exit_price - pos.entry_price) * pos.quantity
            net_pnl = gross_pnl - total_comm
            self.cash += (exit_price * pos.quantity) - exit_commission
            closed_side = PositionSide.LONG
        elif pos.side == PositionSide.SHORT:
            gross_pnl = (pos.entry_price - exit_price) * pos.quantity
            net_pnl = gross_pnl - total_comm
            self.cash -= (exit_price * pos.quantity) + exit_commission
            closed_side = PositionSide.SHORT
        else:
            return 0.0, 0.0, PositionSide.FLAT

        self.realized_pnl += net_pnl
        self.positions[symbol] = Position(
            symbol=symbol,
            side=PositionSide.FLAT,
            realized_pnl=self.realized_pnl,
        )
        self._entry_commissions.pop(symbol, None)

        logger.debug(
            "Portfolio closed %s on %s: PnL=%.2f (Net=%.2f)",
            closed_side.value,
            symbol,
            gross_pnl,
            net_pnl,
        )
        return gross_pnl, net_pnl, closed_side

    def get_state(self, timestamp: datetime | None = None) -> PortfolioState:
        """Get the current snapshot of the portfolio."""
        ts = timestamp or self._last_state_timestamp
        total_unrealized = sum(
            pos.unrealized_pnl
            for pos in self.positions.values()
            if pos.side != PositionSide.FLAT
        )
        total_exposure = self.get_total_exposure()
        current_equity = self.initial_capital + self.realized_pnl + total_unrealized

        return PortfolioState(
            timestamp=ts,
            cash=round(self.cash, 4),
            equity=round(current_equity, 4),
            realized_pnl=round(self.realized_pnl, 4),
            unrealized_pnl=round(total_unrealized, 4),
            portfolio_exposure=round(total_exposure, 4),
            peak_equity=round(self.peak_equity, 4),
            daily_start_equity=round(self.daily_start_equity, 4),
            current_date=self.current_date,
            positions=dict(self.positions),
        )
