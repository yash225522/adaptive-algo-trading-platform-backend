"""CLI commands for testing Angel One authentication and historical data downloading."""

import argparse
import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from adaptive_trading.common.config import get_settings
from adaptive_trading.integrations.angel_one.client import AngelOneClient
from adaptive_trading.integrations.angel_one.config import AngelOneSettings
from adaptive_trading.integrations.angel_one.exceptions import (
    AngelOneAuthenticationError,
    AngelOneConfigurationError,
    AngelOneHistoricalDataError,
)
from adaptive_trading.integrations.angel_one.models import (
    AngelOneInterval,
    HistoricalDataRequest,
    HistoricalDataResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
INDIA_TZ = ZoneInfo("Asia/Kolkata")


def run_auth_smoke_test() -> int:
    """Execute Angel One authentication smoke test."""
    print("=" * 60)
    print("      ANGEL ONE SMARTAPI AUTHENTICATION SMOKE TEST          ")
    print("=" * 60)

    try:
        app_settings = get_settings()
        angel_settings = AngelOneSettings.from_app_settings(app_settings)
        client = AngelOneClient(settings=angel_settings)

        session = client.authenticate()

        print(f"Client Code       : {session.client_code}")
        print(f"Authenticated At  : {session.authenticated_at.isoformat()}")
        print("REST Authorization: Active (JWT Token generated)")
        print("WebSocket Stream  : Active (Feed Token generated)")
        print("-" * 60)
        print("Angel One authentication successful.")
        print("Session established successfully.")
        print("=" * 60)
        return 0

    except AngelOneConfigurationError as exc:
        print("-" * 60)
        print(f"CONFIGURATION ERROR: {exc}")
        print("Please check your .env file for ANGELONE_* credentials.")
        print("=" * 60)
        return 1

    except AngelOneAuthenticationError as exc:
        print("-" * 60)
        print(f"AUTHENTICATION FAILED: {exc}")
        print("Please verify your API key, password/MPIN, and TOTP secret.")
        print("=" * 60)
        return 1

    except Exception as exc:
        logger.exception("Unexpected error during authentication smoke test")
        print(f"UNEXPECTED ERROR: {exc}")
        return 1


def parse_cli_datetime(dt_str: str) -> datetime:
    """Parse user-provided datetime string to timezone-aware datetime."""
    try:
        dt = datetime.fromisoformat(dt_str)
    except ValueError:
        try:
            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
        except ValueError:
            dt = datetime.strptime(dt_str, "%Y-%m-%d")

    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        dt = dt.replace(tzinfo=INDIA_TZ)
    return dt


def run_historical_smoke_test(
    exchange: str,
    symbol: str,
    symbol_token: str,
    interval_str: str,
    from_str: str,
    to_str: str,
) -> int:
    """Execute Angel One historical data download smoke test."""
    print("=" * 60)
    print("      ANGEL ONE SMARTAPI HISTORICAL DATA DOWNLOAD           ")
    print("=" * 60)

    try:
        from_dt = parse_cli_datetime(from_str)
        to_dt = parse_cli_datetime(to_str)
        interval = AngelOneInterval(interval_str)

        request = HistoricalDataRequest(
            exchange=exchange,
            symbol=symbol,
            symbol_token=symbol_token,
            interval=interval,
            from_datetime=from_dt,
            to_datetime=to_dt,
        )

        app_settings = get_settings()
        angel_settings = AngelOneSettings.from_app_settings(app_settings)
        client = AngelOneClient(settings=angel_settings)

        result: HistoricalDataResult = client.get_historical_candles(request)

        print(f"Symbol            : {result.symbol}")
        print(f"Exchange          : {result.exchange}")
        print(f"Symbol Token      : {result.symbol_token}")
        print(f"Interval          : {result.interval.value}")
        print(
            f"Requested Window  : {result.from_datetime.isoformat()} -> "
            f"{result.to_datetime.isoformat()}"
        )
        print(f"Candles Received  : {result.candle_count}")
        print("-" * 60)

        if result.candles:
            first_c = result.candles[0]
            last_c = result.candles[-1]
            print(
                f"First Candle: {first_c.timestamp.isoformat()} | "
                f"Open: {first_c.open} Close: {first_c.close}"
            )
            print(
                f"Last Candle : {last_c.timestamp.isoformat()} | "
                f"Open: {last_c.open} Close: {last_c.close}"
            )
        else:
            print("No candles returned in requested window.")

        if result.quality_report:
            print("-" * 60)
            print(
                f"Data Quality: {result.quality_report.rows_valid} valid, "
                f"{result.quality_report.rows_rejected} rejected, "
                f"{len(result.quality_report.issues)} issues"
            )

        print("-" * 60)
        print("Angel One historical data request successful.")
        print("=" * 60)
        return 0

    except AngelOneConfigurationError as exc:
        print("-" * 60)
        print(f"CONFIGURATION ERROR: {exc}")
        print("=" * 60)
        return 1

    except AngelOneHistoricalDataError as exc:
        print("-" * 60)
        print(f"HISTORICAL REQUEST FAILED: {exc}")
        print("=" * 60)
        return 1

    except Exception as exc:
        logger.exception("Unexpected error during historical smoke test")
        print(f"UNEXPECTED ERROR: {exc}")
        return 1


def main() -> None:
    """CLI entrypoint with subcommands."""
    parser = argparse.ArgumentParser(
        description="Angel One SmartAPI Authentication & Historical Data CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Subcommand: auth
    subparsers.add_parser("auth", help="Run authentication smoke test")

    # Subcommand: historical
    hist_parser = subparsers.add_parser(
        "historical", help="Download historical candles"
    )
    hist_parser.add_argument(
        "--exchange", default="NSE", help="Exchange segment (default: NSE)"
    )
    hist_parser.add_argument(
        "--symbol", default="NIFTY", help="Ticker symbol (default: NIFTY)"
    )
    hist_parser.add_argument(
        "--symbol-token",
        required=True,
        help="Instrument symbol token (e.g. 99926000)",
    )
    hist_parser.add_argument(
        "--interval",
        default="FIVE_MINUTE",
        choices=[i.value for i in AngelOneInterval],
        help="Candle aggregation interval (default: FIVE_MINUTE)",
    )
    hist_parser.add_argument(
        "--from",
        dest="from_date",
        required=True,
        help="Start datetime (e.g. 2026-01-02 09:15)",
    )
    hist_parser.add_argument(
        "--to",
        dest="to_date",
        required=True,
        help="End datetime (e.g. 2026-01-02 15:30)",
    )

    args = parser.parse_args()

    if args.command == "historical":
        exit_code = run_historical_smoke_test(
            exchange=args.exchange,
            symbol=args.symbol,
            symbol_token=args.symbol_token,
            interval_str=args.interval,
            from_str=args.from_date,
            to_str=args.to_date,
        )
    else:
        # Default to auth smoke test
        exit_code = run_auth_smoke_test()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
