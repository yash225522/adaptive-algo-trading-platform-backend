"""Replay runner consuming historical data and managing artifact output."""

import json
import logging
import uuid
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from adaptive_trading.domain.market import Candle
from adaptive_trading.runtime.config import RuntimeConfig
from adaptive_trading.runtime.event_loop import EventLoop
from adaptive_trading.runtime.events import MarketEvent

logger = logging.getLogger(__name__)


class MarketDataProvider(Protocol):
    """Protocol for streaming or iterating over market data events."""

    def stream(self) -> Iterable[MarketEvent]:
        """Yield sequential MarketEvent instances."""
        ...


class HistoricalDataProvider:
    """Provides market events from an in-memory Candle list or historical CSV file."""

    def __init__(
        self,
        candles: Sequence[Candle] | None = None,
        file_path: Path | str | None = None,
        sort: bool = True,
    ) -> None:
        self.sort = sort
        self._events: list[MarketEvent] = []

        if candles is not None:
            self._events = [MarketEvent.from_candle(c) for c in candles]
        elif file_path is not None:
            self._load_from_csv(Path(file_path))

        if self.sort and self._events:
            self._events.sort(key=lambda x: x.timestamp)

    def _load_from_csv(self, path: Path) -> None:
        """Load candles from CSV file."""
        if not path.is_file():
            raise FileNotFoundError(f"Historical data file not found: {path}")

        df = pd.read_csv(path)
        events: list[MarketEvent] = []
        for _, row in df.iterrows():
            ts = (
                pd.to_datetime(row["timestamp"]).to_pydatetime()
                if "timestamp" in row
                else datetime.now(timezone.utc)
            )
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            sym = str(row.get("symbol", "NIFTY"))
            tf = str(row.get("timeframe", "5m"))
            oi_val = None
            if "open_interest" in row and pd.notna(row["open_interest"]):
                oi_val = float(row["open_interest"])
            elif "oi" in row and pd.notna(row["oi"]):
                oi_val = float(row["oi"])

            ev = MarketEvent(
                timestamp=ts,
                symbol=sym,
                timeframe=tf,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0.0)),
                open_interest=oi_val,
            )
            events.append(ev)
        self._events = events

    def stream(self) -> Iterable[MarketEvent]:
        """Stream market events sequentially."""
        yield from self._events


class ReplayRunner:
    """Drives historical event replay through an EventLoop and saves run artifacts."""

    def __init__(
        self,
        event_loop: EventLoop | None = None,
        config: RuntimeConfig | None = None,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.event_loop = event_loop or EventLoop(config=self.config)

    def run(
        self,
        provider: MarketDataProvider,
        artifacts_base_dir: Path | str = "artifacts/runtime",
        run_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Execute chronological replay over all market events from provider.

        Args:
            provider: MarketDataProvider yielding MarketEvents.
            artifacts_base_dir: Base directory for storing run outputs.
            run_id: Optional unique run identifier.
            extra_metadata: Optional supplementary run metadata.

        Returns:
            Path: Output artifact directory for the completed run.
        """
        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        active_run_id = run_id or f"run_{now_str}_{uuid.uuid4().hex[:6]}"
        out_dir = Path(artifacts_base_dir) / active_run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        started_at = datetime.now(timezone.utc)
        first_ts: datetime | None = None
        last_ts: datetime | None = None

        logger.info("Starting historical replay run %s", active_run_id)

        for event in provider.stream():
            if first_ts is None:
                first_ts = event.timestamp
            last_ts = event.timestamp

            self.event_loop.process_market_event(event)

        ended_at = datetime.now(timezone.utc)

        # 1. Save events.jsonl
        events_file = out_dir / "events.jsonl"
        with open(events_file, "w", encoding="utf-8") as f:
            for ev in self.event_loop.events:
                f.write(json.dumps(ev.model_dump(mode="json")) + "\n")

        # 2. Save runtime_stats.json
        stats_file = out_dir / "runtime_stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(self.event_loop.state.stats.model_dump(mode="json"), f, indent=2)

        # 3. Save final_state.json
        account = self.event_loop.execution_service.broker.get_account()
        final_state = {
            "timestamp": last_ts.isoformat() if last_ts else None,
            "cash": account.cash,
            "equity": account.equity,
            "realized_pnl": account.realized_pnl,
            "unrealized_pnl": account.unrealized_pnl,
            "positions": {
                k: v.model_dump(mode="json") for k, v in account.positions.items()
            },
        }
        with open(out_dir / "final_state.json", "w", encoding="utf-8") as f:
            json.dump(final_state, f, indent=2)

        # 4. Save checkpoint.json
        last_ev_id = (
            self.event_loop.events[-1].event_id if self.event_loop.events else ""
        )
        checkpoint = {
            "timestamp": last_ts.isoformat() if last_ts else None,
            "last_event_id": last_ev_id,
            "state_version": "v1",
            "stats": self.event_loop.state.stats.model_dump(mode="json"),
            "account_summary": final_state,
        }
        with open(out_dir / "checkpoint.json", "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)

        # 5. Save metadata.json
        metadata = {
            "run_id": active_run_id,
            "mode": self.config.mode.value,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "data_start": first_ts.isoformat() if first_ts else None,
            "data_end": last_ts.isoformat() if last_ts else None,
            "runtime_config": self.config.model_dump(mode="json"),
            "extra_metadata": extra_metadata or {},
        }
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Replay completed successfully. Artifacts saved to %s", out_dir)
        return out_dir
