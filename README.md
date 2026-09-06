# adaptive-algo-trading-platform

An algorithmic trading platform designed for adaptive market strategies.

## Project Status

Current project status: Early Development

## Technology

- Python 3.12+

## Installation

1. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   # Linux/macOS
   source .venv/bin/activate
   ```

2. Install the package in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Configuration

The application uses an environment-based configuration system powered by Pydantic Settings.

### Setting up Environment Variables

1. Copy the example configuration template to create your local `.env` file:
   ```bash
   # Windows (PowerShell)
   Copy-Item .env.example .env
   # Linux/macOS
   cp .env.example .env
   ```

2. Customize values in `.env` as needed. The following settings are supported:
   - `APP_NAME`: Name of the application (default: `adaptive-algo-trading-platform`).
   - `ENVIRONMENT`: Runtime environment (`development`, `staging`, `production`, `testing`).
   - `LOG_LEVEL`: Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
   - `TIMEZONE`: IANA timezone string (default: `Asia/Kolkata`).
   - `MARKET_SYMBOL`: Default market symbol (default: `NIFTY`).
   - `TIMEFRAME`: Market data timeframe (`1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `1d`).
   - `DATABASE_URL`: Connection string placeholder for the database.

### Loading Configuration

Configuration is loaded automatically through `adaptive_trading.common.get_settings` or by instantiating `adaptive_trading.common.Settings`. Values are read from environment variables or a local `.env` file, falling back to sensible development defaults.

```python
from adaptive_trading.common import get_settings

settings = get_settings()
print(settings.app_name)
print(settings.environment)
```

## Running Tests

Run the test suite using pytest:
```bash
pytest
```
