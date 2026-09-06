"""Instrument resolution and symbol token lookup engine."""

import logging

from sqlalchemy.orm import Session

from adaptive_trading.database.session import get_db_session
from adaptive_trading.instruments.exceptions import (
    InstrumentNotFoundError,
    InstrumentResolutionError,
)
from adaptive_trading.instruments.models import (
    Instrument,
    InstrumentQuery,
    InstrumentType,
)
from adaptive_trading.instruments.repository import InstrumentRepository

logger = logging.getLogger(__name__)

KNOWN_INDICES: dict[str, str] = {
    "NIFTY": "Nifty 50",
    "BANKNIFTY": "Nifty Bank",
    "FINNIFTY": "Nifty Fin Services",
    "MIDCPNIFTY": "NIFTY MID SELECT",
    "SENSEX": "Sensex",
}


class InstrumentResolver:
    """Resolves human-readable queries to unambiguous Angel One instruments."""

    def __init__(self, repository: InstrumentRepository | None = None) -> None:
        self.repository = repository or InstrumentRepository()

    def resolve(
        self,
        query: InstrumentQuery | str,
        exchange: str = "NSE",
        session: Session | None = None,
    ) -> Instrument:
        """Resolve a query to a single matching Instrument.

        Args:
            query: InstrumentQuery instance or ticker symbol string.
            exchange: Default exchange if query is a string.
            session: Optional database session.

        Returns:
            Instrument: Exactly one resolved instrument with valid symbol_token.

        Raises:
            InstrumentNotFoundError: If no matching instrument is found.
            InstrumentResolutionError: If multiple ambiguous instruments match.
        """
        if isinstance(query, str):
            sym_upper = query.strip().upper()
            inst_type = InstrumentType.INDEX if sym_upper in KNOWN_INDICES else None
            query_obj = InstrumentQuery(
                symbol=sym_upper,
                exchange=exchange.strip().upper(),
                instrument_type=inst_type,
            )
        else:
            query_obj = query

        if session is not None:
            return self._resolve_with_session(query_obj, session)

        with get_db_session() as new_session:
            return self._resolve_with_session(query_obj, new_session)

    def _resolve_with_session(
        self,
        query: InstrumentQuery,
        session: Session,
    ) -> Instrument:
        """Internal resolution executing within an active database session."""
        matches = self.repository.find(query, session)

        # 1. No matches found
        if not matches:
            raise InstrumentNotFoundError(
                f"No instrument found matching symbol='{query.symbol}' "
                f"on exchange='{query.exchange}' "
                f"(type={query.instrument_type}, expiry={query.expiry})"
            )

        # 2. Exactly one match
        if len(matches) == 1:
            return matches[0]

        # 3. Multiple matches: Attempt smart disambiguation
        sym_upper = query.symbol.upper()
        if sym_upper in KNOWN_INDICES:
            target_tsym = KNOWN_INDICES[sym_upper].upper()
            idx_matches = [
                m
                for m in matches
                if m.trading_symbol.upper() == target_tsym
                or m.instrument_type == InstrumentType.INDEX
            ]
            if len(idx_matches) == 1:
                return idx_matches[0]

        # Check for exact equity match on cash exchange (e.g. SBIN-EQ on NSE)
        eq_matches = [
            m
            for m in matches
            if m.instrument_type == InstrumentType.EQUITY
            and (
                m.trading_symbol.upper() == f"{sym_upper}-EQ"
                or m.symbol.upper() == sym_upper
            )
        ]
        if len(eq_matches) == 1 and query.instrument_type in (
            None,
            InstrumentType.EQUITY,
        ):
            return eq_matches[0]

        # 4. Truly ambiguous matches: raise detailed error
        types = sorted({m.instrument_type.value for m in matches})
        expiries = sorted({str(m.expiry) for m in matches if m.expiry is not None})
        exp_str = f", expiries={expiries[:3]}..." if expiries else ""

        raise InstrumentResolutionError(
            f"Ambiguous query for symbol='{query.symbol}' on {query.exchange}: "
            f"matched {len(matches)} instruments across types {types}{exp_str}. "
            "Please specify additional criteria "
            "(instrument_type, expiry, strike, option_type)."
        )
