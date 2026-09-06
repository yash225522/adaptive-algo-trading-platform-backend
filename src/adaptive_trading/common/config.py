"""Typed application configuration management using Pydantic Settings."""

from enum import StrEnum
from functools import lru_cache
from zoneinfo import ZoneInfo

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Supported runtime environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(StrEnum):
    """Supported logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TimeFrame(StrEnum):
    """Supported market data timeframes."""

    ONE_MINUTE = "1m"
    THREE_MINUTES = "3m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    ONE_DAY = "1d"


class Settings(BaseSettings):
    """Application settings with environment variable loading and validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = Field(
        default="adaptive-algo-trading-platform",
        min_length=1,
        description="Application name",
    )
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Runtime environment",
    )
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level",
    )
    timezone: str = Field(
        default="Asia/Kolkata",
        min_length=1,
        description="Application timezone string (IANA format)",
    )
    market_symbol: str = Field(
        default="NIFTY",
        min_length=1,
        description="Default market symbol",
    )
    timeframe: TimeFrame = Field(
        default=TimeFrame.FIVE_MINUTES,
        description="Default trading timeframe",
    )
    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/trading_db",
        min_length=1,
        description="Database connection URL placeholder",
    )

    # Angel One SmartAPI Settings
    angelone_api_key: str = Field(
        default="",
        description="Angel One SmartAPI API Key",
    )
    angelone_client_code: str = Field(
        default="",
        description="Angel One Client Code / User ID",
    )
    angelone_password: str = Field(
        default="",
        description="Angel One Account Password or MPIN",
    )
    angelone_totp_secret: str = Field(
        default="",
        description="Angel One TOTP Secret Key for 2FA",
    )

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        """Validate that the timezone is a valid IANA timezone."""
        try:
            ZoneInfo(v)
        except Exception as exc:
            raise ValueError(f"Invalid timezone identifier: {v}") from exc
        return v

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, v: str | LogLevel) -> str | LogLevel:
        """Normalize string log level to uppercase."""
        if isinstance(v, str):
            return v.upper()
        return v


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of application settings."""
    return Settings()
