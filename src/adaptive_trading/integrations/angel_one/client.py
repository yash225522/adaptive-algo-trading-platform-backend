"""Application-level Angel One SmartAPI client wrapper."""

import logging
from typing import TYPE_CHECKING

from SmartApi import SmartConnect

from adaptive_trading.common.config import get_settings
from adaptive_trading.integrations.angel_one.auth import (
    AngelOneAuthenticator,
    AngelOneSession,
)
from adaptive_trading.integrations.angel_one.config import AngelOneSettings
from adaptive_trading.integrations.angel_one.exceptions import (
    AngelOneSessionError,
)

if TYPE_CHECKING:
    from adaptive_trading.integrations.angel_one.models import (
        HistoricalDataRequest,
        HistoricalDataResult,
    )

logger = logging.getLogger(__name__)


class AngelOneClient:
    """Unified application interface for Angel One SmartAPI operations."""

    def __init__(
        self,
        settings: AngelOneSettings | None = None,
        authenticator: AngelOneAuthenticator | None = None,
    ) -> None:
        if settings is None:
            app_settings = get_settings()
            self.settings = AngelOneSettings.from_app_settings(app_settings)
        else:
            self.settings = settings

        self.authenticator = authenticator or AngelOneAuthenticator(
            settings=self.settings
        )
        self.session: AngelOneSession | None = None
        self.smart_connect: SmartConnect | None = None

    def authenticate(self) -> AngelOneSession:
        """Authenticate with Angel One SmartAPI and establish active session.

        Returns:
            AngelOneSession: Authenticated session object.
        """
        self.smart_connect = SmartConnect(api_key=self.settings.api_key)
        self.session = self.authenticator.authenticate(
            settings=self.settings,
            smart_connect=self.smart_connect,
        )
        return self.session

    def is_authenticated(self) -> bool:
        """Check whether the client holds an active session."""
        return self.session is not None

    def get_active_session(self) -> AngelOneSession:
        """Retrieve active session or raise error if unauthenticated.

        Returns:
            AngelOneSession: Active session.

        Raises:
            AngelOneSessionError: If client is not authenticated.
        """
        if self.session is None:
            raise AngelOneSessionError(
                "Angel One client is not authenticated. Call authenticate() first."
            )
        return self.session

    def clear_session(self) -> None:
        """Clear cached authentication session."""
        self.session = None
        self.smart_connect = None
        logger.info("Angel One session cleared")

    def get_historical_candles(
        self,
        request: "HistoricalDataRequest",
    ) -> "HistoricalDataResult":
        """Download and validate historical market candles.

        Args:
            request: HistoricalDataRequest parameter specification.

        Returns:
            HistoricalDataResult: Validated candles and quality report.
        """
        from adaptive_trading.integrations.angel_one.historical import (
            AngelOneHistoricalService,
        )

        service = AngelOneHistoricalService(client=self)
        return service.get_candles(request)
