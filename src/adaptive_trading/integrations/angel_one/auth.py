"""Angel One SmartAPI authentication service and session representation."""

import logging
from datetime import datetime, timezone
from typing import Any

import pyotp
from pydantic import BaseModel, ConfigDict, Field
from SmartApi import SmartConnect

from adaptive_trading.integrations.angel_one.config import AngelOneSettings
from adaptive_trading.integrations.angel_one.exceptions import (
    AngelOneAuthenticationError,
    AngelOneConfigurationError,
)

logger = logging.getLogger(__name__)


class AngelOneSession(BaseModel):
    """Encapsulates active Angel One SmartAPI authorization and feed tokens."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    jwt_token: str = Field(description="JWT token for REST API request authorization")
    refresh_token: str = Field(
        description="Token used to refresh expired REST sessions"
    )
    feed_token: str = Field(description="Token used for WebSocket live data streaming")
    client_code: str = Field(description="Angel One Client Code")
    authenticated_at: datetime = Field(
        description="Timestamp when session was generated"
    )

    def __repr__(self) -> str:
        """Mask sensitive token contents in representations."""
        return (
            f"AngelOneSession(client_code='{self.client_code}', "
            f"authenticated_at='{self.authenticated_at.isoformat()}', "
            "jwt_token='***', feed_token='***', refresh_token='***')"
        )


class AngelOneAuthenticator:
    """Handles secure TOTP generation and session creation with Angel One SmartAPI."""

    def __init__(self, settings: AngelOneSettings | None = None) -> None:
        self.settings = settings

    def generate_totp(self, totp_secret: str) -> str:
        """Generate a valid 6-digit Time-based One-Time Password (TOTP).

        Args:
            totp_secret: Base32 encoded TOTP secret key.

        Returns:
            str: 6-digit TOTP string.
        """
        clean_secret = totp_secret.strip().replace(" ", "")
        if not clean_secret:
            raise AngelOneConfigurationError("TOTP secret cannot be empty")

        try:
            totp = pyotp.TOTP(clean_secret)
            otp_code = str(totp.now())
            if len(otp_code) != 6 or not otp_code.isdigit():
                raise ValueError("Generated OTP does not match 6-digit format")
            return otp_code
        except Exception as e:
            raise AngelOneConfigurationError(
                f"Failed to generate TOTP from secret: {e}"
            ) from e

    def authenticate(
        self,
        settings: AngelOneSettings | None = None,
        smart_connect: Any = None,
    ) -> AngelOneSession:
        """Perform full authentication flow against Angel One SmartAPI.

        Args:
            settings: Optional settings override.
            smart_connect: Optional pre-configured SmartConnect instance for testing.

        Returns:
            AngelOneSession: Validated authenticated session with JWT and Feed tokens.
        """
        cfg = settings or self.settings
        if cfg is None:
            raise AngelOneConfigurationError(
                "AngelOneSettings must be provided to authenticate"
            )

        # 1. Validate credentials presence
        cfg.validate_credentials()

        logger.info(
            "Starting Angel One SmartAPI authentication for client %s", cfg.client_code
        )

        # 2. Generate TOTP
        otp = self.generate_totp(cfg.totp_secret)

        # 3. Create SmartConnect client if not provided
        client = smart_connect or SmartConnect(api_key=cfg.api_key)

        # 4. Generate Session via SmartAPI
        try:
            response = client.generateSession(
                clientCode=cfg.client_code,
                password=cfg.password,
                totp=otp,
            )
        except Exception as exc:
            logger.error("SmartAPI network request failed during authentication")
            raise AngelOneAuthenticationError(
                f"Angel One authentication network error: {exc}"
            ) from exc

        # 5. Validate response structure
        if not response or not isinstance(response, dict):
            raise AngelOneAuthenticationError(
                "Invalid response format received from SmartAPI gateway"
            )

        if not response.get("status"):
            msg = response.get("message", "Unknown authentication error")
            err_code = response.get("errorcode", "UNKNOWN")
            logger.warning("Angel One authentication failed (errorcode: %s)", err_code)
            raise AngelOneAuthenticationError(
                f"SmartAPI authentication rejected: {msg} (code: {err_code})"
            )

        data = response.get("data")
        if not data or not isinstance(data, dict):
            raise AngelOneAuthenticationError(
                "Malformed SmartAPI response: 'data' block is missing"
            )

        jwt_token = data.get("jwtToken")
        refresh_token = data.get("refreshToken")
        feed_token = data.get("feedToken")

        if not jwt_token or not refresh_token or not feed_token:
            raise AngelOneAuthenticationError(
                "Malformed SmartAPI response: required tokens missing"
            )

        session = AngelOneSession(
            jwt_token=str(jwt_token),
            refresh_token=str(refresh_token),
            feed_token=str(feed_token),
            client_code=cfg.client_code,
            authenticated_at=datetime.now(timezone.utc),
        )

        logger.info(
            "Angel One authentication successful; session created for %s",
            cfg.client_code,
        )
        return session
