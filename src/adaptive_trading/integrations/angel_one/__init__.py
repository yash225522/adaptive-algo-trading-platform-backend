"""Angel One SmartAPI integration package."""

from adaptive_trading.integrations.angel_one.auth import (
    AngelOneAuthenticator,
    AngelOneSession,
)
from adaptive_trading.integrations.angel_one.client import AngelOneClient
from adaptive_trading.integrations.angel_one.config import AngelOneSettings
from adaptive_trading.integrations.angel_one.exceptions import (
    AngelOneAuthenticationError,
    AngelOneConfigurationError,
    AngelOneError,
    AngelOneHistoricalDataError,
    AngelOneSessionError,
)
from adaptive_trading.integrations.angel_one.historical import (
    AngelOneHistoricalService,
    format_datetime_for_smartapi,
)
from adaptive_trading.integrations.angel_one.models import (
    ANGELONE_TO_TIMEFRAME,
    TIMEFRAME_TO_ANGELONE,
    AngelOneInterval,
    HistoricalDataRequest,
    HistoricalDataResult,
)

__all__ = [
    "ANGELONE_TO_TIMEFRAME",
    "TIMEFRAME_TO_ANGELONE",
    "AngelOneAuthenticationError",
    "AngelOneAuthenticator",
    "AngelOneClient",
    "AngelOneConfigurationError",
    "AngelOneError",
    "AngelOneHistoricalDataError",
    "AngelOneHistoricalService",
    "AngelOneInterval",
    "AngelOneSession",
    "AngelOneSessionError",
    "AngelOneSettings",
    "HistoricalDataRequest",
    "HistoricalDataResult",
    "format_datetime_for_smartapi",
]
