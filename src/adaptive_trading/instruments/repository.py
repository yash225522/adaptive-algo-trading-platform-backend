"""Repository for instrument persistence, query filtering, and token retrieval."""

import logging

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from adaptive_trading.database.models.instrument import InstrumentModel
from adaptive_trading.instruments.models import (
    ImportStats,
    Instrument,
    InstrumentQuery,
    InstrumentType,
    OptionType,
)

logger = logging.getLogger(__name__)


class InstrumentRepository:
    """Handles database storage, bulk upsert, and querying of instruments."""

    def _to_domain(self, model: InstrumentModel) -> Instrument:
        """Convert an InstrumentModel database entity to domain Instrument."""
        return Instrument(
            symbol_token=model.symbol_token,
            exchange=model.exchange,
            trading_symbol=model.trading_symbol,
            symbol=model.symbol,
            name=model.name,
            instrument_type=InstrumentType(model.instrument_type),
            expiry=model.expiry,
            strike=model.strike,
            option_type=OptionType(model.option_type) if model.option_type else None,
            lot_size=model.lot_size,
            tick_size=model.tick_size,
        )

    def upsert_batch(
        self,
        instruments: list[Instrument],
        session: Session,
    ) -> ImportStats:
        """Upsert a list of Instrument domain models into the database.

        Args:
            instruments: List of validated Instrument domain models.
            session: SQLAlchemy session.

        Returns:
            ImportStats: Summary of inserted, updated, and unchanged records.
        """
        stats = ImportStats(records_read=len(instruments))
        if not instruments:
            return stats

        # Load existing tokens for this batch's (exchange, symbol_token) pairs
        keys = {(inst.exchange, inst.symbol_token) for inst in instruments}
        exchanges = {k[0] for k in keys}
        tokens = {k[1] for k in keys}

        stmt = select(InstrumentModel).where(
            InstrumentModel.exchange.in_(exchanges),
            InstrumentModel.symbol_token.in_(tokens),
        )
        existing_models = {
            (m.exchange, m.symbol_token): m for m in session.scalars(stmt).all()
        }

        for inst in instruments:
            key = (inst.exchange, inst.symbol_token)
            existing = existing_models.get(key)
            opt_val = inst.option_type.value if inst.option_type else None

            if existing is None:
                new_model = InstrumentModel(
                    symbol_token=inst.symbol_token,
                    exchange=inst.exchange,
                    trading_symbol=inst.trading_symbol,
                    symbol=inst.symbol,
                    name=inst.name,
                    instrument_type=inst.instrument_type.value,
                    expiry=inst.expiry,
                    strike=inst.strike,
                    option_type=opt_val,
                    lot_size=inst.lot_size,
                    tick_size=inst.tick_size,
                )
                session.add(new_model)
                stats.records_inserted += 1
            else:
                # Check if attributes changed
                changed = (
                    existing.trading_symbol != inst.trading_symbol
                    or existing.symbol != inst.symbol
                    or existing.name != inst.name
                    or existing.instrument_type != inst.instrument_type.value
                    or existing.expiry != inst.expiry
                    or existing.strike != inst.strike
                    or existing.option_type != opt_val
                    or existing.lot_size != inst.lot_size
                    or existing.tick_size != inst.tick_size
                )
                if changed:
                    existing.trading_symbol = inst.trading_symbol
                    existing.symbol = inst.symbol
                    existing.name = inst.name
                    existing.instrument_type = inst.instrument_type.value
                    existing.expiry = inst.expiry
                    existing.strike = inst.strike
                    existing.option_type = opt_val
                    existing.lot_size = inst.lot_size
                    existing.tick_size = inst.tick_size
                    stats.records_updated += 1
                else:
                    stats.records_unchanged += 1

        session.flush()
        logger.info(
            "Upsert complete: %d inserted, %d updated, %d unchanged",
            stats.records_inserted,
            stats.records_updated,
            stats.records_unchanged,
        )
        return stats

    def find(
        self,
        query: InstrumentQuery,
        session: Session,
    ) -> list[Instrument]:
        """Find instruments matching structured query criteria.

        Args:
            query: InstrumentQuery specification.
            session: SQLAlchemy session.

        Returns:
            list[Instrument]: Matching instruments.
        """
        symbol_upper = query.symbol.strip().upper()
        exchange_upper = query.exchange.strip().upper()

        stmt = select(InstrumentModel).where(
            InstrumentModel.exchange == exchange_upper,
            (
                (InstrumentModel.symbol == symbol_upper)
                | (InstrumentModel.trading_symbol == symbol_upper)
            ),
        )

        if query.instrument_type is not None:
            stmt = stmt.where(
                InstrumentModel.instrument_type == query.instrument_type.value
            )

        if query.expiry is not None:
            stmt = stmt.where(InstrumentModel.expiry == query.expiry)

        if query.strike is not None:
            stmt = stmt.where(InstrumentModel.strike == query.strike)

        if query.option_type is not None:
            stmt = stmt.where(InstrumentModel.option_type == query.option_type.value)

        models = session.scalars(stmt).all()
        return [self._to_domain(m) for m in models]

    def find_by_token(
        self,
        exchange: str,
        symbol_token: str,
        session: Session,
    ) -> Instrument | None:
        """Retrieve instrument by unique exchange and symbol token."""
        stmt = select(InstrumentModel).where(
            InstrumentModel.exchange == exchange.strip().upper(),
            InstrumentModel.symbol_token == symbol_token.strip(),
        )
        model = session.scalar(stmt)
        return self._to_domain(model) if model else None

    def count(self, session: Session) -> int:
        """Return total number of instruments in database."""
        stmt = select(func.count(InstrumentModel.id))
        res = session.scalar(stmt)
        return int(res or 0)
