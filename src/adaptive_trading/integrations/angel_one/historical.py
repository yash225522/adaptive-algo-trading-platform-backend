"""Angel One SmartAPI historical market data downloader service."""

import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from adaptive_trading.common.config import TimeFrame
from adaptive_trading.data.models import DataQualityReport
from adaptive_trading.data.normalizers.market_data import MarketDataNormalizer
from adaptive_trading.data.quality.checker import MarketDataQualityChecker
from adaptive_trading.integrations.angel_one.client import AngelOneClient
from adaptive_trading.integrations.angel_one.exceptions import (
    AngelOneHistoricalDataError,
    AngelOneSessionError,
)
from adaptive_trading.integrations.angel_one.models import (
    ANGELONE_TO_TIMEFRAME,
    HistoricalDataRequest,
    HistoricalDataResult,
)

logger = logging.getLogger(__name__)
INDIA_TZ = ZoneInfo("Asia/Kolkata")


def format_datetime_for_smartapi(
    dt: datetime, default_tz: ZoneInfo | None = None
) -> str:
    """Format timezone-aware datetime into SmartAPI 'YYYY-MM-DD HH:MM' format.

    Converts any timezone-aware datetime into Asia/Kolkata market timezone.
    """
    target_tz = default_tz or INDIA_TZ
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        localized_dt = dt.replace(tzinfo=target_tz)
    else:
        localized_dt = dt.astimezone(target_tz)

    return localized_dt.strftime("%Y-%m-%d %H:%M")


class AngelOneHistoricalService:
    """Retrieves, validates, and normalizes historical candle data via SmartAPI."""

    def __init__(
        self,
        client: AngelOneClient | None = None,
        normalizer: MarketDataNormalizer | None = None,
        quality_checker: MarketDataQualityChecker | None = None,
        resolver: Any | None = None,
    ) -> None:
        self.client = client or AngelOneClient()
        self.normalizer = normalizer or MarketDataNormalizer(
            default_timezone="Asia/Kolkata"
        )
        self.quality_checker = quality_checker or MarketDataQualityChecker()
        self.resolver = resolver

    def get_candles(
        self,
        request: HistoricalDataRequest,
    ) -> HistoricalDataResult:
        """Download historical candles and convert them into domain Candle models.

        Args:
            request: Validated HistoricalDataRequest parameters.

        Returns:
            HistoricalDataResult: Domain candles with quality report.

        Raises:
            AngelOneHistoricalDataError: On API gateway, network, or format errors.
        """
        # 1. Resolve symbol token if not explicitly provided
        if request.symbol_token:
            resolved_token = request.symbol_token
        else:
            try:
                if self.resolver is None:
                    from adaptive_trading.instruments.resolver import (
                        InstrumentResolver,
                    )

                    resolver = InstrumentResolver()
                else:
                    resolver = self.resolver

                instrument = resolver.resolve(
                    query=request.symbol, exchange=request.exchange
                )
                resolved_token = instrument.symbol_token
                logger.info(
                    "Resolved '%s' on %s to symbol_token '%s'",
                    request.symbol,
                    request.exchange,
                    resolved_token,
                )
            except Exception as exc:
                raise AngelOneHistoricalDataError(
                    f"Failed to resolve symbol token for '{request.symbol}': {exc}"
                ) from exc

        # 2. Ensure client is authenticated
        if not self.client.is_authenticated():
            logger.info("Authenticating Angel One client prior to historical request")
            self.client.authenticate()

        smart_connect = self.client.smart_connect
        if smart_connect is None:
            raise AngelOneSessionError("SmartConnect client instance is not available")

        # 3. Build SmartAPI historic parameters payload
        from_str = format_datetime_for_smartapi(request.from_datetime)
        to_str = format_datetime_for_smartapi(request.to_datetime)

        interval_str = (
            request.interval.value
            if hasattr(request.interval, "value")
            else str(request.interval)
        )

        historic_param = {
            "exchange": request.exchange,
            "symboltoken": resolved_token,
            "interval": interval_str,
            "fromdate": from_str,
            "todate": to_str,
        }

        logger.info(
            "Requesting historical data for %s (%s:%s, %s, %s -> %s)",
            request.symbol,
            request.exchange,
            resolved_token,
            interval_str,
            from_str,
            to_str,
        )

        # 4. Call getCandleData via SmartConnect
        try:
            response = smart_connect.getCandleData(historic_param)
        except Exception as exc:
            logger.error("Network or transport error during getCandleData")
            raise AngelOneHistoricalDataError(
                f"SmartAPI historical request network error: {exc}"
            ) from exc

        # 5. Validate SmartAPI response
        if not response or not isinstance(response, dict):
            raise AngelOneHistoricalDataError(
                "Invalid response format received from SmartAPI gateway"
            )

        if not response.get("status"):
            msg = response.get("message", "Unknown SmartAPI error")
            err_code = response.get("errorcode", "UNKNOWN")
            raise AngelOneHistoricalDataError(
                f"SmartAPI historical request rejected: {msg} (code: {err_code})"
            )

        raw_data = response.get("data")
        if raw_data is None or raw_data == []:
            logger.info(
                "No candle data returned for %s in requested window",
                request.symbol,
            )
            return HistoricalDataResult(
                symbol=request.symbol,
                symbol_token=resolved_token,
                exchange=request.exchange,
                interval=request.interval,
                from_datetime=request.from_datetime,
                to_datetime=request.to_datetime,
                candles=[],
                quality_report=DataQualityReport(
                    rows_checked=0, rows_valid=0, rows_rejected=0
                ),
            )

        if not isinstance(raw_data, list):
            raise AngelOneHistoricalDataError(
                "Malformed SmartAPI response: 'data' payload is not a list"
            )

        # 6. Map raw arrays [timestamp, open, high, low, close, volume] to dicts
        timeframe_enum = ANGELONE_TO_TIMEFRAME.get(
            request.interval, TimeFrame.FIVE_MINUTES
        )
        raw_records: list[tuple[int, dict[str, Any]]] = []

        for idx, row in enumerate(raw_data, start=1):
            if not isinstance(row, list) or len(row) < 6:
                logger.warning("Skipping malformed row at index %d: %s", idx, row)
                continue

            record_dict = {
                "timestamp": row[0],
                "symbol": request.symbol,
                "timeframe": timeframe_enum.value,
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5],
                "open_interest": row[6] if len(row) > 6 else None,
            }
            raw_records.append((idx, record_dict))

        # 7. Apply Step 6 Normalization & Data Quality Validation
        normalized_records, norm_issues = self.normalizer.normalize_batch(raw_records)
        valid_candles, quality_report = self.quality_checker.check_batch(
            normalized_records
        )

        # Merge any normalization issues into quality report
        for issue in norm_issues:
            quality_report.add_issue(issue)

        logger.info(
            "Historical fetch completed for %s: %d received, %d valid candles",
            request.symbol,
            len(raw_data),
            len(valid_candles),
        )

        return HistoricalDataResult(
            symbol=request.symbol,
            symbol_token=resolved_token,
            exchange=request.exchange,
            interval=request.interval,
            from_datetime=request.from_datetime,
            to_datetime=request.to_datetime,
            candles=valid_candles,
            quality_report=quality_report,
        )
