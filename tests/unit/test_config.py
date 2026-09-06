"""Unit tests for the application configuration system."""

import pytest
from pydantic import ValidationError

from adaptive_trading.common.config import (
    Environment,
    LogLevel,
    Settings,
    TimeFrame,
    get_settings,
)


def test_default_settings() -> None:
    """Test that default configuration loads with expected sensible defaults."""
    settings = Settings()

    assert settings.app_name == "adaptive-algo-trading-platform"
    assert settings.environment == Environment.DEVELOPMENT
    assert settings.log_level == LogLevel.INFO
    assert settings.timezone == "Asia/Kolkata"
    assert settings.market_symbol == "NIFTY"
    assert settings.timeframe == TimeFrame.FIVE_MINUTES
    assert "postgresql+psycopg://" in settings.database_url


def test_get_settings_caching() -> None:
    """Test that get_settings() returns a cached Settings instance."""
    settings_1 = get_settings()
    settings_2 = get_settings()
    assert settings_1 is settings_2


def test_environment_variable_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that environment variables override default settings."""
    monkeypatch.setenv("APP_NAME", "custom-trading-platform")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("TIMEZONE", "UTC")
    monkeypatch.setenv("MARKET_SYMBOL", "BANKNIFTY")
    monkeypatch.setenv("TIMEFRAME", "15m")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://custom:5432/custom_db")

    settings = Settings()

    assert settings.app_name == "custom-trading-platform"
    assert settings.environment == Environment.PRODUCTION
    assert settings.log_level == LogLevel.DEBUG
    assert settings.timezone == "UTC"
    assert settings.market_symbol == "BANKNIFTY"
    assert settings.timeframe == TimeFrame.FIFTEEN_MINUTES
    assert settings.database_url == "postgresql+psycopg://custom:5432/custom_db"


def test_case_insensitive_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that log_level string is normalized regardless of case."""
    monkeypatch.setenv("LOG_LEVEL", "warning")
    settings = Settings()
    assert settings.log_level == LogLevel.WARNING


def test_invalid_environment_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that an unsupported environment value raises a validation error."""
    monkeypatch.setenv("ENVIRONMENT", "invalid_env")
    with pytest.raises(ValidationError):
        Settings()


def test_invalid_log_level_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that an invalid log level raises a validation error."""
    monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
    with pytest.raises(ValidationError):
        Settings()


def test_invalid_timeframe_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that an unsupported timeframe raises a validation error."""
    monkeypatch.setenv("TIMEFRAME", "2m")
    with pytest.raises(ValidationError):
        Settings()


def test_invalid_timezone_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that an invalid IANA timezone string raises a validation error."""
    monkeypatch.setenv("TIMEZONE", "Invalid/Timezone")
    with pytest.raises(ValidationError):
        Settings()


def test_empty_app_name_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that empty string for app_name is rejected."""
    monkeypatch.setenv("APP_NAME", "")
    with pytest.raises(ValidationError):
        Settings()


def test_empty_market_symbol_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that empty string for market_symbol is rejected."""
    monkeypatch.setenv("MARKET_SYMBOL", "")
    with pytest.raises(ValidationError):
        Settings()


def test_empty_database_url_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that empty string for database_url is rejected."""
    monkeypatch.setenv("DATABASE_URL", "")
    with pytest.raises(ValidationError):
        Settings()
