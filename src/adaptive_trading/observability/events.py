"""Event filtering, querying, and audit trail retrieval utilities."""

import json
import logging
from collections.abc import Sequence
from pathlib import Path

from adaptive_trading.runtime.events import EventRecord, EventType

logger = logging.getLogger(__name__)


def filter_events(
    events: Sequence[EventRecord],
    event_type: EventType | str | None = None,
    symbol: str | None = None,
    correlation_id: str | None = None,
) -> list[EventRecord]:
    """Filter a sequence of EventRecord objects by criteria.

    Args:
        events: Input sequence of EventRecords.
        event_type: Optional EventType or string type filter.
        symbol: Optional asset ticker symbol filter.
        correlation_id: Optional correlation ID filter.

    Returns:
        list[EventRecord]: Matching EventRecords.
    """
    filtered = list(events)

    if event_type is not None:
        type_val = (
            event_type.value if isinstance(event_type, EventType) else str(event_type)
        )
        filtered = [
            e
            for e in filtered
            if (
                e.event_type.value
                if isinstance(e.event_type, EventType)
                else str(e.event_type)
            )
            == type_val
        ]

    if symbol is not None:
        filtered = [e for e in filtered if e.symbol == symbol]

    if correlation_id is not None:
        filtered = [e for e in filtered if e.correlation_id == correlation_id]

    return filtered


def load_run_events(
    run_dir_or_id: Path | str,
    base_dir: Path | str = "artifacts/runtime",
) -> list[EventRecord]:
    """Load persisted EventRecord stream from a run directory.

    Args:
        run_dir_or_id: Directory path or run ID.
        base_dir: Base directory if a run ID string is supplied.

    Returns:
        list[EventRecord]: Parsed list of EventRecords.
    """
    path = Path(run_dir_or_id)
    if not path.is_dir():
        path = Path(base_dir) / str(run_dir_or_id)

    events_file = path / "events.jsonl"
    if not events_file.is_file():
        return []

    records: list[EventRecord] = []
    with open(events_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                data = json.loads(line_str)
                records.append(EventRecord(**data))

    return records
