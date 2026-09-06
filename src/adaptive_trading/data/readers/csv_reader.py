"""CSV reader for market data files with configurable column mappings."""

import csv
from pathlib import Path
from typing import Any

DEFAULT_COLUMN_MAPPING: dict[str, str] = {
    "timestamp": "timestamp",
    "symbol": "symbol",
    "timeframe": "timeframe",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "open_interest": "open_interest",
}

REQUIRED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "symbol",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


class CSVMarketDataReader:
    """Reads raw CSV market data files and produces normalized record dicts."""

    def __init__(
        self,
        column_mapping: dict[str, str] | None = None,
        encoding: str = "utf-8",
    ) -> None:
        self.column_mapping = column_mapping or DEFAULT_COLUMN_MAPPING
        self._field_to_csv = {v: k for k, v in self.column_mapping.items()}
        self.encoding = encoding

    def read_records(self, file_path: str | Path) -> list[tuple[int, dict[str, Any]]]:
        """Read CSV file and return list of (row_number, record_dict).

        Row numbers are 1-indexed (starting at 2 for the first data line).
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Market data file not found: {path}")

        records: list[tuple[int, dict[str, Any]]] = []

        with path.open(mode="r", encoding=self.encoding, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                return records

            # Check that required columns exist in the header
            header_set = {h.strip() for h in reader.fieldnames if h}
            for required_field in REQUIRED_COLUMNS:
                expected_csv_col = self.column_mapping.get(
                    required_field, required_field
                )
                if expected_csv_col not in header_set:
                    avail = sorted(header_set)
                    raise ValueError(
                        f"Missing required CSV column '{expected_csv_col}' "
                        f"(mapped from '{required_field}'). Available: {avail}"
                    )

            # Read data rows
            for row_idx, raw_row in enumerate(reader, start=2):
                normalized: dict[str, Any] = {}
                for target_field, csv_col in self.column_mapping.items():
                    raw_val = raw_row.get(csv_col)
                    normalized[target_field] = (
                        raw_val.strip() if isinstance(raw_val, str) else raw_val
                    )
                records.append((row_idx, normalized))

        return records
