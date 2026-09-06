"""Structured application logging, JSON formatting, and credential redaction."""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from adaptive_trading.observability.config import ObservabilityConfig

SENSITIVE_KEY_PATTERNS = [
    r"password",
    r"secret",
    r"jwt",
    r"token",
    r"api_key",
    r"apikey",
    r"auth",
    r"client_code",
    r"pin",
    r"totp",
]

SENSITIVE_STRING_PATTERNS = [
    # JWT tokens (three dot-separated base64url segments starting with eyJ)
    (
        re.compile(
            r"eyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]+",
            re.IGNORECASE,
        ),
        r"[REDACTED_JWT]",
    ),
    # Bearer tokens
    (
        re.compile(r"(?i)bearer\s+([A-Za-z0-9\-\._~\+\/=]+)", re.IGNORECASE),
        r"Bearer [REDACTED]",
    ),
    # Key-value pairs (e.g. password=xyz, api_key: "abc")
    (
        re.compile(
            r"(?i)(password|secret|jwt|token|api_key|apikey|auth_token|client_secret|totp_secret|pin)"
            r"([\s:=]+[\"']?)([^\s,;'\"\}]+)([\"']?)",
            re.IGNORECASE,
        ),
        r"\1\2[REDACTED]\4",
    ),
]


def redact_text(text: str) -> str:
    """Mask sensitive credentials and tokens within a string."""
    if not text:
        return text
    result = text
    for pattern, replacement in SENSITIVE_STRING_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def redact_data(data: Any) -> Any:
    """Recursively scrub sensitive keys and token values from nested data structures."""
    if isinstance(data, dict):
        scrubbed = {}
        for k, v in data.items():
            k_lower = str(k).lower()
            if any(re.search(pat, k_lower) for pat in SENSITIVE_KEY_PATTERNS):
                scrubbed[k] = "[REDACTED]"
            else:
                scrubbed[k] = redact_data(v)
        return scrubbed
    if isinstance(data, (list, tuple, set)):
        return [redact_data(item) for item in data]
    if isinstance(data, str):
        return redact_text(data)
    return data


class SensitiveDataFilter(logging.Filter):
    """Logging filter that scrubs sensitive credentials from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_text(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = redact_data(record.args)
            elif isinstance(record.args, tuple):
                record.args = tuple(redact_data(arg) for arg in record.args)
        return True


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as compact, machine-readable JSON strings."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Inject context attributes if present on record
        context_fields = [
            "component",
            "run_id",
            "correlation_id",
            "event_id",
            "symbol",
            "status",
        ]
        for field in context_fields:
            val = getattr(record, field, None)
            if val is not None:
                log_entry[field] = val

        # Include exception trace if present
        if record.exc_info and record.exc_text:
            log_entry["exception"] = record.exc_text
        elif record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(redact_data(log_entry))


def setup_logging(config: ObservabilityConfig | None = None) -> None:
    """Configure system-wide logging with formatters and redaction filters."""
    cfg = config or ObservabilityConfig()
    level = getattr(logging, cfg.log_level, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove pre-existing handlers to prevent duplicated output
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)

    if cfg.redact_sensitive_data:
        stream_handler.addFilter(SensitiveDataFilter())

    if cfg.structured_logging:
        stream_handler.setFormatter(StructuredJsonFormatter())
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        stream_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
