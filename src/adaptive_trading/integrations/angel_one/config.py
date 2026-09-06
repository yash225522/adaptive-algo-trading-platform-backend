"""Strongly-typed configuration model for Angel One SmartAPI."""

from pydantic import BaseModel, ConfigDict, Field

from adaptive_trading.common.config import Settings
from adaptive_trading.integrations.angel_one.exceptions import (
    AngelOneConfigurationError,
)


class AngelOneSettings(BaseModel):
    """Encapsulates and validates Angel One SmartAPI credentials and endpoints."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    api_key: str = Field(default="", description="Angel One SmartAPI Application Key")
    client_code: str = Field(default="", description="Angel One Client ID / Username")
    password: str = Field(default="", description="Angel One Password or MPIN")
    totp_secret: str = Field(default="", description="Base32 TOTP secret key for 2FA")
    root_url: str = Field(
        default="https://apiconnect.angelone.in",
        description="SmartAPI root gateway endpoint",
    )

    def validate_credentials(self) -> None:
        """Verify that all required authentication credentials are provided.

        Raises:
            AngelOneConfigurationError: If any credential is missing or empty.
        """
        missing = []
        if not self.api_key.strip():
            missing.append("api_key (ANGELONE_API_KEY)")
        if not self.client_code.strip():
            missing.append("client_code (ANGELONE_CLIENT_CODE)")
        if not self.password.strip():
            missing.append("password (ANGELONE_PASSWORD)")
        if not self.totp_secret.strip():
            missing.append("totp_secret (ANGELONE_TOTP_SECRET)")

        if missing:
            raise AngelOneConfigurationError(
                f"Missing required Angel One credentials: {', '.join(missing)}"
            )

    @classmethod
    def from_app_settings(cls, settings: Settings) -> "AngelOneSettings":
        """Create AngelOneSettings from the centralized application Settings."""
        return cls(
            api_key=settings.angelone_api_key,
            client_code=settings.angelone_client_code,
            password=settings.angelone_password,
            totp_secret=settings.angelone_totp_secret,
        )
