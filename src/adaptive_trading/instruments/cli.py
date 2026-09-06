"""CLI commands for instrument master synchronization and symbol token resolution."""

import argparse
import logging
import sys
from datetime import datetime

from adaptive_trading.instruments.exceptions import (
    InstrumentError,
    InstrumentNotFoundError,
    InstrumentResolutionError,
)
from adaptive_trading.instruments.importer import AngelOneInstrumentImporter
from adaptive_trading.instruments.models import (
    InstrumentQuery,
    InstrumentType,
    OptionType,
)
from adaptive_trading.instruments.resolver import InstrumentResolver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_refresh(source: str | None) -> int:
    """Execute instrument master refresh command."""
    print("=" * 60)
    print("        ANGEL ONE INSTRUMENT MASTER REFRESH                 ")
    print("=" * 60)

    try:
        importer = AngelOneInstrumentImporter()
        stats = importer.import_instruments(source=source)

        print("Instrument refresh completed successfully.")
        print("-" * 60)
        print(f"Records read      : {stats.records_read}")
        print(f"Records inserted  : {stats.records_inserted}")
        print(f"Records updated   : {stats.records_updated}")
        print(f"Records unchanged : {stats.records_unchanged}")
        print(f"Records rejected  : {stats.records_rejected}")
        print("=" * 60)
        return 0

    except InstrumentError as exc:
        print(f"REFRESH ERROR: {exc}")
        print("=" * 60)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error during instrument refresh")
        print(f"UNEXPECTED ERROR: {exc}")
        print("=" * 60)
        return 1


def run_resolve(
    symbol: str,
    exchange: str,
    inst_type_str: str | None,
    expiry_str: str | None,
    strike: float | None,
    opt_type_str: str | None,
) -> int:
    """Execute instrument resolution command."""
    print("=" * 60)
    print("         INSTRUMENT SYMBOL TOKEN RESOLUTION                 ")
    print("=" * 60)

    try:
        inst_type = InstrumentType(inst_type_str) if inst_type_str else None
        opt_type = OptionType(opt_type_str) if opt_type_str else None
        expiry_date = (
            datetime.strptime(expiry_str, "%Y-%m-%d").date() if expiry_str else None
        )

        query = InstrumentQuery(
            symbol=symbol,
            exchange=exchange,
            instrument_type=inst_type,
            expiry=expiry_date,
            strike=strike,
            option_type=opt_type,
        )

        resolver = InstrumentResolver()
        inst = resolver.resolve(query)

        print("Instrument resolved successfully.")
        print("-" * 60)
        print(f"Symbol            : {inst.symbol}")
        print(f"Trading Symbol    : {inst.trading_symbol}")
        print(f"Exchange          : {inst.exchange}")
        print(f"Symbol Token      : {inst.symbol_token}")
        print(f"Instrument Type   : {inst.instrument_type.value}")
        if inst.expiry:
            print(f"Expiry Date       : {inst.expiry}")
        if inst.strike:
            print(f"Strike Price      : {inst.strike}")
        if inst.option_type:
            print(f"Option Type       : {inst.option_type.value}")
        print(f"Lot Size          : {inst.lot_size}")
        print(f"Tick Size         : {inst.tick_size}")
        print("=" * 60)
        return 0

    except InstrumentNotFoundError as exc:
        print(f"NOT FOUND: {exc}")
        print("=" * 60)
        return 1
    except InstrumentResolutionError as exc:
        print(f"AMBIGUOUS QUERY: {exc}")
        print("=" * 60)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error during instrument resolution")
        print(f"UNEXPECTED ERROR: {exc}")
        print("=" * 60)
        return 1


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Angel One Instrument Master and Symbol Token Manager"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Subcommand: refresh
    refresh_parser = subparsers.add_parser(
        "refresh", help="Refresh instrument master from Angel One"
    )
    refresh_parser.add_argument(
        "--source",
        default=None,
        help="Optional URL or local JSON file path for scrip master",
    )

    # Subcommand: resolve
    resolve_parser = subparsers.add_parser(
        "resolve", help="Resolve instrument symbol to token"
    )
    resolve_parser.add_argument(
        "--symbol", required=True, help="Ticker symbol (e.g. NIFTY, SBIN)"
    )
    resolve_parser.add_argument(
        "--exchange", default="NSE", help="Exchange segment (default: NSE)"
    )
    resolve_parser.add_argument(
        "--type",
        dest="inst_type",
        choices=[t.value for t in InstrumentType],
        default=None,
        help="Instrument type filter",
    )
    resolve_parser.add_argument(
        "--expiry", default=None, help="Contract expiry date (YYYY-MM-DD)"
    )
    resolve_parser.add_argument(
        "--strike", type=float, default=None, help="Option strike price"
    )
    resolve_parser.add_argument(
        "--option-type",
        dest="opt_type",
        choices=[o.value for o in OptionType],
        default=None,
        help="Option contract type (CE/PE)",
    )

    args = parser.parse_args()

    if args.command == "refresh":
        sys.exit(run_refresh(source=args.source))
    elif args.command == "resolve":
        sys.exit(
            run_resolve(
                symbol=args.symbol,
                exchange=args.exchange,
                inst_type_str=args.inst_type,
                expiry_str=args.expiry,
                strike=args.strike,
                opt_type_str=args.opt_type,
            )
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
