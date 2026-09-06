"""Cryptographic deterministic fingerprint generation.

Generates SHA-256 digests for datasets, models, and configurations.
"""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from adaptive_trading.observability.logging import redact_data


def compute_string_fingerprint(text: str) -> str:
    """Compute deterministic SHA-256 hex fingerprint for a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_dict_fingerprint(
    data: dict[str, Any], exclude_keys: Sequence[str] | None = None
) -> str:
    """Compute canonical SHA-256 fingerprint for a configuration dictionary.

    Keys are sorted alphabetically and scrubbed of volatile/sensitive data to ensure
    identical logical configurations yield identical hashes regardless of dict order.
    """
    default_excluded = [
        "created_at",
        "started_at",
        "ended_at",
        "experiment_id",
    ]
    excluded = set(exclude_keys or default_excluded)

    # Scrub sensitive data and excluded keys
    cleaned: dict[str, Any] = {}
    for k, v in data.items():
        if k not in excluded:
            cleaned[k] = v

    scrubbed = redact_data(cleaned)
    canonical_json = json.dumps(
        scrubbed, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def compute_file_fingerprint(path: Path | str, chunk_size: int = 65536) -> str:
    """Compute deterministic SHA-256 fingerprint of a binary file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for fingerprinting: {p}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dataset_fingerprint(candles: Sequence[Any]) -> str:
    """Compute deterministic fingerprint over a sequence of candles.

    Encodes timestamps, symbol, and OHLCV values in strict chronological order.
    """
    if not candles:
        return compute_string_fingerprint("EMPTY_DATASET")

    lines: list[str] = []
    for c in candles:
        ts_str = (
            c.timestamp.isoformat()
            if hasattr(c, "timestamp")
            else str(getattr(c, "timestamp", ""))
        )
        sym = str(getattr(c, "symbol", "UNKNOWN"))
        o = f"{float(getattr(c, 'open', 0.0)):.4f}"
        h = f"{float(getattr(c, 'high', 0.0)):.4f}"
        low_val = f"{float(getattr(c, 'low', 0.0)):.4f}"
        cl = f"{float(getattr(c, 'close', 0.0)):.4f}"
        vol = f"{float(getattr(c, 'volume', 0.0)):.2f}"
        lines.append(f"{ts_str}|{sym}|{o}|{h}|{low_val}|{cl}|{vol}")

    canonical_text = "\n".join(lines)
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
