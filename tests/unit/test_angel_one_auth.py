"""Unit tests for Angel One SmartAPI configuration, TOTP, and authentication."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pyotp
import pytest

from adaptive_trading.common.config import Settings
from adaptive_trading.integrations.angel_one.auth import (
    AngelOneAuthenticator,
    AngelOneSession,
)
from adaptive_trading.integrations.angel_one.cli import run_auth_smoke_test
from adaptive_trading.integrations.angel_one.client import AngelOneClient
from adaptive_trading.integrations.angel_one.config import AngelOneSettings
from adaptive_trading.integrations.angel_one.exceptions import (
    AngelOneAuthenticationError,
    AngelOneConfigurationError,
    AngelOneSessionError,
)

TEST_TOTP_SECRET = "JBSWY3DPEHPK3PXP"  # Standard RFC test secret


def test_angel_one_settings_validation_missing_fields() -> None:
    """Verify AngelOneSettings raises error when credentials are empty."""
    settings = AngelOneSettings(
        api_key="",
        client_code="A1234",
        password="password",
        totp_secret=TEST_TOTP_SECRET,
    )
    with pytest.raises(
        AngelOneConfigurationError, match="Missing required Angel One credentials"
    ):
        settings.validate_credentials()


def test_angel_one_settings_from_app_settings() -> None:
    """Verify loading AngelOneSettings from centralized app Settings."""
    app_settings = Settings(
        angelone_api_key="key123",
        angelone_client_code="code456",
        angelone_password="pass789",
        angelone_totp_secret=TEST_TOTP_SECRET,
    )
    angel_settings = AngelOneSettings.from_app_settings(app_settings)
    assert angel_settings.api_key == "key123"
    assert angel_settings.client_code == "code456"
    assert angel_settings.password == "pass789"
    assert angel_settings.totp_secret == TEST_TOTP_SECRET


def test_totp_generation_with_known_secret() -> None:
    """Verify TOTP generation produces a valid 6-digit numeric string."""
    authenticator = AngelOneAuthenticator()
    otp = authenticator.generate_totp(TEST_TOTP_SECRET)

    assert len(otp) == 6
    assert otp.isdigit()

    # Validate against direct pyotp computation
    expected_otp = pyotp.TOTP(TEST_TOTP_SECRET).now()
    assert otp == expected_otp


def test_totp_generation_empty_or_invalid_secret() -> None:
    """Verify TOTP generation handles empty or malformed secrets."""
    authenticator = AngelOneAuthenticator()

    with pytest.raises(AngelOneConfigurationError, match="cannot be empty"):
        authenticator.generate_totp("")

    with pytest.raises(AngelOneConfigurationError, match="Failed to generate TOTP"):
        authenticator.generate_totp("INVALID!@#$%")


def test_authentication_success_creates_session() -> None:
    """Verify successful authentication creates a valid AngelOneSession."""
    mock_smart_connect = MagicMock()
    mock_smart_connect.generateSession.return_value = {
        "status": True,
        "message": "SUCCESS",
        "errorcode": "",
        "data": {
            "jwtToken": "jwt_token_sample_abc123",
            "refreshToken": "refresh_token_sample_xyz456",
            "feedToken": "feed_token_sample_789",
        },
    }

    settings = AngelOneSettings(
        api_key="test_api_key",
        client_code="TEST001",
        password="test_password",
        totp_secret=TEST_TOTP_SECRET,
    )

    authenticator = AngelOneAuthenticator(settings=settings)
    session = authenticator.authenticate(
        settings=settings,
        smart_connect=mock_smart_connect,
    )

    assert isinstance(session, AngelOneSession)
    assert session.client_code == "TEST001"
    assert session.jwt_token == "jwt_token_sample_abc123"
    assert session.refresh_token == "refresh_token_sample_xyz456"
    assert session.feed_token == "feed_token_sample_789"
    assert session.authenticated_at is not None

    # Verify generateSession was called with correct arguments
    mock_smart_connect.generateSession.assert_called_once()
    call_kwargs = mock_smart_connect.generateSession.call_args.kwargs
    assert call_kwargs["clientCode"] == "TEST001"
    assert call_kwargs["password"] == "test_password"
    assert len(call_kwargs["totp"]) == 6


def test_authentication_failure_status_false() -> None:
    """Verify SmartAPI failure response raises AngelOneAuthenticationError."""
    mock_smart_connect = MagicMock()
    mock_smart_connect.generateSession.return_value = {
        "status": False,
        "message": "Invalid password or OTP",
        "errorcode": "AB1001",
        "data": None,
    }

    settings = AngelOneSettings(
        api_key="test_api_key",
        client_code="TEST001",
        password="test_password",
        totp_secret=TEST_TOTP_SECRET,
    )

    authenticator = AngelOneAuthenticator()
    with pytest.raises(AngelOneAuthenticationError, match="Invalid password or OTP"):
        authenticator.authenticate(settings=settings, smart_connect=mock_smart_connect)


def test_authentication_malformed_response_missing_tokens() -> None:
    """Verify malformed response missing tokens raises AngelOneAuthenticationError."""
    mock_smart_connect = MagicMock()
    mock_smart_connect.generateSession.return_value = {
        "status": True,
        "data": {
            "jwtToken": "sample_jwt",
            # Missing refreshToken and feedToken
        },
    }

    settings = AngelOneSettings(
        api_key="test_api_key",
        client_code="TEST001",
        password="test_password",
        totp_secret=TEST_TOTP_SECRET,
    )

    authenticator = AngelOneAuthenticator()
    with pytest.raises(AngelOneAuthenticationError, match="required tokens"):
        authenticator.authenticate(settings=settings, smart_connect=mock_smart_connect)


def test_authentication_network_exception_wrapped() -> None:
    """Verify network or gateway exceptions are cleanly wrapped."""
    mock_smart_connect = MagicMock()
    mock_smart_connect.generateSession.side_effect = ConnectionError(
        "Connection refused"
    )

    settings = AngelOneSettings(
        api_key="test_api_key",
        client_code="TEST001",
        password="test_password",
        totp_secret=TEST_TOTP_SECRET,
    )

    authenticator = AngelOneAuthenticator()
    with pytest.raises(AngelOneAuthenticationError, match="Connection refused"):
        authenticator.authenticate(settings=settings, smart_connect=mock_smart_connect)


def test_angel_one_client_lifecycle() -> None:
    """Test client authenticate, is_authenticated, and clear_session."""
    mock_authenticator = MagicMock()
    mock_session = MagicMock(spec=AngelOneSession)
    mock_authenticator.authenticate.return_value = mock_session

    settings = AngelOneSettings(
        api_key="key",
        client_code="code",
        password="pass",
        totp_secret=TEST_TOTP_SECRET,
    )

    client = AngelOneClient(settings=settings, authenticator=mock_authenticator)
    assert not client.is_authenticated()

    with pytest.raises(AngelOneSessionError, match="not authenticated"):
        client.get_active_session()

    # Authenticate
    session = client.authenticate()
    assert session == mock_session
    assert client.is_authenticated()
    assert client.get_active_session() == mock_session

    # Clear
    client.clear_session()
    assert not client.is_authenticated()


def test_session_repr_masks_tokens() -> None:
    """Verify session __repr__ does not reveal actual JWT or feed tokens."""
    session = AngelOneSession(
        jwt_token="SECRET_JWT_KEY_12345",
        refresh_token="SECRET_REFRESH_KEY_67890",
        feed_token="SECRET_FEED_KEY_112233",
        client_code="USER999",
        authenticated_at=MagicMock(),
    )
    repr_str = repr(session)
    assert "SECRET_JWT_KEY_12345" not in repr_str
    assert "SECRET_REFRESH_KEY_67890" not in repr_str
    assert "SECRET_FEED_KEY_112233" not in repr_str
    assert "jwt_token='***'" in repr_str
    assert "feed_token='***'" in repr_str


def test_cli_smoke_test_with_mocked_auth() -> None:
    """Verify CLI smoke test returns exit code 0 when authentication succeeds."""
    session = AngelOneSession(
        jwt_token="mock_jwt",
        refresh_token="mock_refresh",
        feed_token="mock_feed",
        client_code="USER123",
        authenticated_at=datetime(2026, 8, 30, 14, 50, tzinfo=timezone.utc),
    )

    with patch(
        "adaptive_trading.integrations.angel_one.cli.AngelOneClient"
    ) as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.authenticate.return_value = session

        exit_code = run_auth_smoke_test()
        assert exit_code == 0
