"""Unit tests for CSVMarketDataReader."""

from pathlib import Path

import pytest

from adaptive_trading.data.readers.csv_reader import CSVMarketDataReader


def test_read_valid_csv(tmp_path: Path) -> None:
    """Test reading records from a valid CSV file."""
    csv_file = tmp_path / "valid.csv"
    csv_file.write_text(
        "timestamp,symbol,timeframe,open,high,low,close,volume,open_interest\n"
        "2026-01-02T09:15:00+05:30,NIFTY,5m,26000.0,26050.0,25980.0,26030.0,1000,50000\n"
        "2026-01-02T09:20:00+05:30,NIFTY,5m,26030.0,26070.0,26020.0,26050.0,1200,51000\n",
        encoding="utf-8",
    )

    reader = CSVMarketDataReader()
    records = reader.read_records(csv_file)

    assert len(records) == 2
    row_num, row_data = records[0]
    assert row_num == 2
    assert row_data["symbol"] == "NIFTY"
    assert row_data["open"] == "26000.0"
    assert row_data["close"] == "26030.0"


def test_missing_required_column_raises_error(tmp_path: Path) -> None:
    """Test that missing required columns raise a descriptive ValueError."""
    csv_file = tmp_path / "missing_col.csv"
    csv_file.write_text(
        "timestamp,symbol,open,high,low,close,volume\n"  # missing timeframe
        "2026-01-02T09:15:00+05:30,NIFTY,26000.0,26050.0,25980.0,26030.0,1000\n",
        encoding="utf-8",
    )

    reader = CSVMarketDataReader()
    with pytest.raises(ValueError, match="Missing required CSV column 'timeframe'"):
        reader.read_records(csv_file)


def test_custom_column_mapping(tmp_path: Path) -> None:
    """Test reading with custom provider column header names."""
    csv_file = tmp_path / "custom.csv"
    csv_file.write_text(
        "Date,Ticker,Interval,OpenPrice,HighPrice,LowPrice,ClosePrice,Vol\n"
        "2026-01-02T09:15:00+05:30,BANKNIFTY,5m,51000,51100,50950,51050,5000\n",
        encoding="utf-8",
    )

    custom_mapping = {
        "timestamp": "Date",
        "symbol": "Ticker",
        "timeframe": "Interval",
        "open": "OpenPrice",
        "high": "HighPrice",
        "low": "LowPrice",
        "close": "ClosePrice",
        "volume": "Vol",
    }

    reader = CSVMarketDataReader(column_mapping=custom_mapping)
    records = reader.read_records(csv_file)

    assert len(records) == 1
    _, row_data = records[0]
    assert row_data["symbol"] == "BANKNIFTY"
    assert row_data["open"] == "51000"


def test_file_not_found() -> None:
    """Test that non-existent file raises FileNotFoundError."""
    reader = CSVMarketDataReader()
    with pytest.raises(FileNotFoundError):
        reader.read_records("non_existent_file.csv")
