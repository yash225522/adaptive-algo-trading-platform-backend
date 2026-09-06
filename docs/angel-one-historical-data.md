# Angel One Historical Market Data Downloader Guide

This document describes the design, configuration, request lifecycle, response mapping, data quality validation, and CLI usage for downloading historical market candles from Angel One's SmartAPI.

---

## 1. Purpose of the Historical Downloader

The historical market data downloader retrieves OHLCV candle bars from Angel One via the SmartAPI `getCandleData` endpoint and transforms them into the platform's standardized, validated domain models (`Candle`).

```text
Angel One SmartAPI Gateway (https://apiconnect.angelone.in)
        ↓
AngelOneAuthenticator (Step 10 Session / JWT Authorization)
        ↓
HistoricalDataRequest (Exchange, Token, Interval, From, To)
        ↓
SmartConnect.getCandleData(historicParam)
        ↓
Response Mapper (OHLCV Array -> Record Dict)
        ↓
MarketDataNormalizer & MarketDataQualityChecker (Step 6 Quality Layer)
        ↓
HistoricalDataResult (Candle[] + DataQualityReport)
```

---

## 2. Authentication Prerequisite

All requests to the historical API require an active REST JWT token generated during authentication:
- Handled automatically by `AngelOneClient` / `AngelOneHistoricalService`.
- If unauthenticated, the client authenticates using local credentials configured in `.env`.

---

## 3. Request Parameters

Requests are structured via `HistoricalDataRequest`:

| Parameter | Type | SmartAPI Field | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `exchange` | `str` | `exchange` | Exchange segment | `"NSE"`, `"BSE"`, `"NFO"` |
| `symbol` | `str` | N/A | Human-readable ticker symbol | `"NIFTY"`, `"RELIANCE"` |
| `symbol_token` | `str` | `symboltoken` | Angel One instrument token | `"99926000"`, `"3045"` |
| `interval` | `AngelOneInterval` | `interval` | Candle resolution | `AngelOneInterval.FIVE_MINUTE` |
| `from_datetime` | `datetime` | `fromdate` | Start timestamp | `2026-01-02T09:15:00+05:30` |
| `to_datetime` | `datetime` | `todate` | End timestamp | `2026-01-02T15:30:00+05:30` |

---

## 4. Timeframe & Interval Mappings

The system maintains a clean bidirectional mapping between application domain timeframes (`TimeFrame`) and SmartAPI intervals (`AngelOneInterval`):

| Domain Timeframe (`TimeFrame`) | Angel One Interval (`AngelOneInterval`) | SmartAPI String |
| :--- | :--- | :--- |
| `TimeFrame.ONE_MINUTE` (`"1m"`) | `AngelOneInterval.ONE_MINUTE` | `"ONE_MINUTE"` |
| `TimeFrame.THREE_MINUTES` (`"3m"`) | `AngelOneInterval.THREE_MINUTE` | `"THREE_MINUTE"` |
| `TimeFrame.FIVE_MINUTES` (`"5m"`) | `AngelOneInterval.FIVE_MINUTE` | `"FIVE_MINUTE"` |
| `TimeFrame.FIFTEEN_MINUTES` (`"15m"`) | `AngelOneInterval.FIFTEEN_MINUTE` | `"FIFTEEN_MINUTE"` |
| `TimeFrame.THIRTY_MINUTES` (`"30m"`) | `AngelOneInterval.THIRTY_MINUTE` | `"THIRTY_MINUTE"` |
| `TimeFrame.ONE_HOUR` (`"1h"`) | `AngelOneInterval.ONE_HOUR` | `"ONE_HOUR"` |
| `TimeFrame.ONE_DAY` (`"1d"`) | `AngelOneInterval.ONE_DAY` | `"ONE_DAY"` |

---

## 5. Date & Time Handling

Angel One requires `fromdate` and `todate` formatted strictly as `YYYY-MM-DD HH:MM`:
- The helper `format_datetime_for_smartapi()` converts any timezone-aware datetime into the Indian market timezone (`Asia/Kolkata`) prior to string formatting.
- Naive datetimes are not silently guessed; they are explicitly localized or rejected.

---

## 6. Response Mapping to Domain Models

SmartAPI returns historical candles as an array of numerical/string arrays:

```json
{
  "status": true,
  "message": "SUCCESS",
  "errorcode": "",
  "data": [
    ["2026-01-02T09:15:00+05:30", 26000.0, 26050.0, 25980.0, 26030.0, 120000],
    ["2026-01-02T09:20:00+05:30", 26030.0, 26070.0, 26020.0, 26050.0, 95000]
  ]
}
```

The response mapper converts these raw arrays into `Candle` objects:
- `data[i][0]` $\rightarrow$ `timestamp` (ISO-8601 string parsed to timezone-aware `datetime`)
- `data[i][1]` $\rightarrow$ `open` (float)
- `data[i][2]` $\rightarrow$ `high` (float)
- `data[i][3]` $\rightarrow$ `low` (float)
- `data[i][4]` $\rightarrow$ `close` (float)
- `data[i][5]` $\rightarrow$ `volume` (float)
- `data[i][6]` $\rightarrow$ `open_interest` (`None` if absent or not provided)

---

## 7. Integration with Data Quality Layer (Step 6)

Every downloaded candle batch passes through the Step 6 data quality engine:
1. **Normalization**: Canonical symbol uppercase, lowercase timeframe, timezone validation.
2. **Chronological Sorting**: Enforces strict timestamp order.
3. **Deduplication**: Identifies and flags repeated timestamps.
4. **Heuristics**: Checks for zero-volume warnings, stale flat OHLC bars, and invalid price relationships ($H \ge O, C, L$ and $L \le O, C, H$).

---

## 8. CLI Historical Download Usage

You can test historical downloading directly from the command line:

```bash
# Download 5-minute historical candles for NIFTY
python -m adaptive_trading.integrations.angel_one.cli historical \
    --exchange NSE \
    --symbol NIFTY \
    --symbol-token 99926000 \
    --interval FIVE_MINUTE \
    --from "2026-01-02 09:15" \
    --to "2026-01-02 15:30"
```

### Sample CLI Output:
```text
============================================================
      ANGEL ONE SMARTAPI HISTORICAL DATA DOWNLOAD           
============================================================
Symbol            : NIFTY
Exchange          : NSE
Symbol Token      : 99926000
Interval          : FIVE_MINUTE
Requested Window  : 2026-01-02T09:15:00+05:30 -> 2026-01-02T15:30:00+05:30
Candles Received  : 75
------------------------------------------------------------
First Candle (Open): 2026-01-02T09:15:00+05:30 | Open: 26000.0 Close: 26030.0
Last Candle (Close): 2026-01-02T15:25:00+05:30 | Open: 26180.0 Close: 26170.0
------------------------------------------------------------
Data Quality Check: 75 valid, 0 rejected, 0 issues
------------------------------------------------------------
Angel One historical data request successful.
============================================================
```

---

## 9. Security & Error Handling

- **Zero Secret Exposure**: Tokens and API keys are never printed in logs or exceptions.
- **Graceful Failure**: Clean exceptions (`AngelOneHistoricalDataError`, `AngelOneSessionError`) for API errors, timeouts, or malformed responses.
- **No Database Coupling**: The historical downloader returns pure in-memory `Candle` lists without side-effects on PostgreSQL.

---

## 10. What Is Intentionally Deferred

- Instrument master / automatic symbol token lookup (Step 12+).
- Live WebSocket tick streaming.
- Order placement and execution.

