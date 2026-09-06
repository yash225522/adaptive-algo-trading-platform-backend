"""Angel One Scrip Master JSON fetcher, parser, normalizer, and database importer."""

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import requests
from sqlalchemy.orm import Session

from adaptive_trading.database.session import get_db_session
from adaptive_trading.instruments.exceptions import InstrumentImportError
from adaptive_trading.instruments.models import (
    ImportStats,
    Instrument,
    InstrumentType,
    OptionType,
)
from adaptive_trading.instruments.repository import InstrumentRepository

logger = logging.getLogger(__name__)

DEFAULT_SCRIP_MASTER_URL: str = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"


class AngelOneInstrumentImporter:
    """Synchronizes local PostgreSQL database with Angel One Scrip Master JSON."""

    def __init__(
        self,
        repository: InstrumentRepository | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        self.repository = repository or InstrumentRepository()
        self.timeout_seconds = timeout_seconds

    def parse_expiry_date(self, raw_expiry: str | None) -> date | None:
        """Parse raw expiry date string into python date."""
        if not raw_expiry or not str(raw_expiry).strip():
            return None

        clean_exp = str(raw_expiry).strip().upper()
        # Common formats in Angel One: '29JAN2026', '29-JAN-2026', '2026-01-29'
        for fmt in ("%d%b%Y", "%d-%b-%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(clean_exp, fmt).date()
            except ValueError:
                continue

        logger.debug("Unable to parse expiry date: '%s'", raw_expiry)
        return None

    def infer_instrument_type(
        self,
        raw_type: str,
        trading_symbol: str,
        exchange: str,
    ) -> InstrumentType:
        """Infer standardized InstrumentType from raw Angel One metadata."""
        inst_type_clean = (raw_type or "").strip().upper()
        tsym_upper = (trading_symbol or "").strip().upper()

        if inst_type_clean == "AMXIDX" or "INDEX" in inst_type_clean:
            return InstrumentType.INDEX
        if inst_type_clean in ("FUTIDX", "FUTSTK", "FUTCOM", "FUTCUR"):
            return InstrumentType.FUTURES
        if inst_type_clean in ("OPTIDX", "OPTSTK", "OPTCOM", "OPTCUR"):
            return InstrumentType.OPTIONS
        if inst_type_clean in ("EQ", ""):
            if tsym_upper.endswith("-EQ") or exchange in ("NSE", "BSE"):
                return InstrumentType.EQUITY
            return InstrumentType.OTHER

        return InstrumentType.OTHER

    def infer_option_type(self, trading_symbol: str) -> OptionType | None:
        """Infer Call (CE) or Put (PE) from trading symbol."""
        tsym = trading_symbol.strip().upper()
        if tsym.endswith("CE"):
            return OptionType.CE
        if tsym.endswith("PE"):
            return OptionType.PE
        return None

    def parse_item(self, raw_item: dict[str, Any]) -> Instrument | None:
        """Normalize and validate a raw Angel One JSON item into an Instrument."""
        token = str(raw_item.get("token", "")).strip()
        exchange = str(raw_item.get("exch_seg", "")).strip().upper()
        trading_symbol = str(raw_item.get("symbol", "")).strip()
        raw_name = str(raw_item.get("name", "")).strip()

        if not token or not exchange or not trading_symbol:
            return None

        # Normalized underlying symbol
        symbol = raw_name.upper() if raw_name else trading_symbol.upper()

        raw_inst_type = str(raw_item.get("instrumenttype", ""))
        instrument_type = self.infer_instrument_type(
            raw_inst_type, trading_symbol, exchange
        )

        expiry_date = self.parse_expiry_date(raw_item.get("expiry"))

        # Strike price parsing
        strike_val: float | None = None
        raw_strike = raw_item.get("strike")
        if raw_strike is not None:
            try:
                s_float = float(raw_strike)
                if s_float > 0.0:
                    if s_float >= 10000.0 and instrument_type == InstrumentType.OPTIONS:
                        strike_val = s_float / 100.0
                    else:
                        strike_val = s_float
            except (ValueError, TypeError):
                strike_val = None

        option_type = (
            self.infer_option_type(trading_symbol)
            if instrument_type == InstrumentType.OPTIONS
            else None
        )

        # Lot size & tick size
        lot_size = 1
        try:
            lot_size = max(1, int(float(raw_item.get("lotsize", 1))))
        except (ValueError, TypeError):
            lot_size = 1

        tick_size = 0.05
        try:
            raw_tick = float(raw_item.get("tick_size", 0.05))
            if raw_tick > 0.0:
                tick_size = raw_tick / 100.0 if raw_tick >= 1.0 else raw_tick
        except (ValueError, TypeError):
            tick_size = 0.05

        return Instrument(
            symbol_token=token,
            exchange=exchange,
            trading_symbol=trading_symbol,
            symbol=symbol,
            name=raw_name,
            instrument_type=instrument_type,
            expiry=expiry_date,
            strike=strike_val,
            option_type=option_type,
            lot_size=lot_size,
            tick_size=tick_size,
        )

    def fetch_raw_data(
        self,
        source: str | Path | list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch raw JSON instruments list from URL, local file, or direct list."""
        if isinstance(source, list):
            return source

        src_str = str(source) if source is not None else DEFAULT_SCRIP_MASTER_URL

        # Local file path
        if Path(src_str).is_file():
            try:
                with open(src_str, encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    raise InstrumentImportError(f"Expected JSON array in {src_str}")
                return data
            except Exception as exc:
                raise InstrumentImportError(
                    f"Failed to read local instrument file: {exc}"
                ) from exc

        # HTTP Download
        try:
            logger.info("Downloading Angel One scrip master from %s", src_str)
            response = requests.get(src_str, timeout=self.timeout_seconds)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list):
                raise InstrumentImportError(f"Expected JSON list from {src_str}")
            return data
        except Exception as exc:
            raise InstrumentImportError(
                f"Failed to download Angel One scrip master: {exc}"
            ) from exc

    def import_instruments(
        self,
        source: str | Path | list[dict[str, Any]] | None = None,
        session: Session | None = None,
    ) -> ImportStats:
        """Download or parse instrument master and upsert into database."""
        raw_items = self.fetch_raw_data(source)
        logger.info("Loaded %d raw instrument records", len(raw_items))

        valid_instruments: list[Instrument] = []
        rejected_count = 0

        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                rejected_count += 1
                continue

            inst = self.parse_item(raw_item)
            if inst is not None:
                valid_instruments.append(inst)
            else:
                rejected_count += 1

        if session is not None:
            stats = self.repository.upsert_batch(valid_instruments, session)
            stats.records_rejected = rejected_count
            return stats

        with get_db_session() as new_session:
            stats = self.repository.upsert_batch(valid_instruments, new_session)
            stats.records_rejected = rejected_count
            return stats
