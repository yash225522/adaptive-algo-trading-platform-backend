# Database Architecture & PostgreSQL Guide

This document describes the database persistence layer, configuration, schema, migrations, and testing practices for `adaptive-algo-trading-platform`.

---

## 1. Why PostgreSQL?

PostgreSQL is selected as the primary relational database for the platform due to:
- **ACID Compliance & Reliability**: Critical for accurate market candle storage, execution auditability, and trade accounting.
- **Time-Series Capabilities**: High-performance querying over timestamp ranges with composite b-tree indexes.
- **Strict Integrity Constraints**: Database-level unique constraints and relational integrity prevent corrupt or duplicated market data.
- **Ecosystem & Tooling**: Seamless integration with SQLAlchemy 2.0 and Alembic for declarative schema management and reproducible migrations.

---

## 2. Domain Models vs. Database Models

The architecture enforces a strict boundary between Domain Models and Database Models:

```
┌──────────────────────────────────────────────┐
│        Pydantic Domain Models (Step 3)       │
│  • Pure Python dataclasses & contracts       │
│  • Fast in-memory validation & typing        │
│  • Free from ORM/database dependencies       │
└──────────────────────────────────────────────┘
                       ▲
                       │ (Explicit mapping / conversion)
                       ▼
┌──────────────────────────────────────────────┐
│       SQLAlchemy ORM Models (Step 4)         │
│  • Database table mapping & persistence      │
│  • Foreign keys, indexes, & table constraints│
│  • Managed lifecycle via SessionFactory      │
└──────────────────────────────────────────────┘
```

- **Domain Models** (`adaptive_trading.domain`) represent core business logic and component interfaces.
- **Database Models** (`adaptive_trading.database.models`) represent storage schema and table structures.

---

## 3. Database Configuration

Database parameters are loaded dynamically via `adaptive_trading.common.get_settings()` from environment variables or a local `.env` file:

```env
POSTGRES_DB=trading_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/trading_db
```

---

## 4. Starting PostgreSQL Locally

A `docker-compose.yml` file is provided for local development:

```bash
# Start PostgreSQL in background
docker compose up -d

# Check service status
docker compose ps

# Stop PostgreSQL
docker compose down
```

---

## 5. Running Alembic Migrations

Migrations are managed with Alembic:

```bash
# Apply migrations to head
alembic upgrade head

# Roll back the most recent migration
alembic downgrade -1

# Create a new auto-generated migration
alembic revision --autogenerate -m "description_of_change"
```

---

## 6. Running Database Tests

Automated tests run against isolated in-memory test engines or dedicated test databases without requiring a running production database:

```bash
pytest tests/unit/test_database.py -v
```

---

## 7. Current Tables & Schema

### `market_candles`
Stores validated historical and streaming OHLCV market candles.

| Column | Type | Nullable | Description |
| :--- | :--- | :---: | :--- |
| `id` | `INTEGER` (PK) | No | Auto-incrementing primary key |
| `timestamp` | `TIMESTAMP WITH TIME ZONE` | No | Timestamp of the candle interval |
| `symbol` | `VARCHAR(32)` | No | Market ticker symbol (e.g. `NIFTY`) |
| `timeframe` | `VARCHAR(16)` | No | Candle interval (e.g. `5m`, `1h`) |
| `open` | `FLOAT` | No | Open price |
| `high` | `FLOAT` | No | High price |
| `low` | `FLOAT` | No | Low price |
| `close` | `FLOAT` | No | Close price |
| `volume` | `FLOAT` | No | Traded volume |
| `open_interest` | `FLOAT` | Yes | Optional open interest |
| `created_at` | `TIMESTAMP WITH TIME ZONE` | No | Row insertion timestamp |

**Constraints & Indexes**:
- Unique constraint: `uq_market_candles_ts_sym_tf (timestamp, symbol, timeframe)`
- Composite index: `ix_market_candles_sym_tf_ts (symbol, timeframe, timestamp)`
- Index: `ix_market_candles_sym_ts (symbol, timestamp)`

---

### `models`
Tracks machine learning model versions and lifecycle states.

| Column | Type | Nullable | Description |
| :--- | :--- | :---: | :--- |
| `id` | `INTEGER` (PK) | No | Primary key |
| `model_version` | `VARCHAR(64)` | No | Unique version identifier |
| `model_type` | `VARCHAR(64)` | No | Architecture / model type |
| `status` | `VARCHAR(32)` | No | Status (`active`, `inactive`, `training`) |
| `created_at` | `TIMESTAMP WITH TIME ZONE` | No | Creation timestamp |

---

### `runs`
Tracks operational executions across data ingestion, training, backtesting, and paper trading.

| Column | Type | Nullable | Description |
| :--- | :--- | :---: | :--- |
| `id` | `INTEGER` (PK) | No | Primary key |
| `run_type` | `VARCHAR(64)` | No | Execution category |
| `status` | `VARCHAR(32)` | No | Execution status (`running`, `completed`, `failed`) |
| `started_at` | `TIMESTAMP WITH TIME ZONE` | No | Start timestamp |
| `completed_at` | `TIMESTAMP WITH TIME ZONE` | Yes | Completion timestamp |
| `created_at` | `TIMESTAMP WITH TIME ZONE` | No | Record creation timestamp |

