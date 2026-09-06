"""Deterministic, event-driven orchestrator connecting market events to execution."""

import logging
import uuid
from collections import defaultdict
from typing import Protocol

import numpy as np

from adaptive_trading.domain.market import Candle
from adaptive_trading.domain.prediction import FeatureVector
from adaptive_trading.execution.config import ExecutionConfig
from adaptive_trading.execution.models import OrderStatus
from adaptive_trading.execution.paper_broker import (
    PaperBroker,
    StaticMarketDataProvider,
)
from adaptive_trading.execution.service import ExecutionService
from adaptive_trading.features.pipeline import FeaturePipeline
from adaptive_trading.ml.models import BaseMLModel
from adaptive_trading.ml.preprocessing import MLPreprocessor
from adaptive_trading.portfolio.models import PositionSide
from adaptive_trading.portfolio.state import PortfolioManager
from adaptive_trading.risk.config import RiskConfig
from adaptive_trading.risk.engine import RiskEngine
from adaptive_trading.runtime.clock import TradingClock
from adaptive_trading.runtime.config import RuntimeConfig
from adaptive_trading.runtime.events import EventRecord, EventType, MarketEvent
from adaptive_trading.runtime.exceptions import EventProcessingError
from adaptive_trading.runtime.state import RuntimeState
from adaptive_trading.strategy.engine import StrategyEngine
from adaptive_trading.strategy.models import SignalAction, StrategyPrediction

logger = logging.getLogger(__name__)


class PredictionService(Protocol):
    """Protocol for generating predictions from calculated FeatureVectors."""

    def predict(self, feature_vector: FeatureVector) -> StrategyPrediction | None:
        """Generate prediction for a given feature vector."""
        ...


class MLPredictionService:
    """Prediction service wrapping a trained BaseMLModel and MLPreprocessor."""

    def __init__(
        self,
        model: BaseMLModel,
        preprocessor: MLPreprocessor,
        model_version: str = "v1",
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.model_version = model_version

    def predict(self, feature_vector: FeatureVector) -> StrategyPrediction | None:
        """Run preprocessing and model inference on feature vector."""
        try:
            row_features = [
                float(feature_vector.features.get(col, 0.0))
                for col in self.preprocessor.feature_names
            ]
            X_raw = np.array([row_features], dtype=np.float64)
            if self.preprocessor.scaler is not None:
                X_scaled = self.preprocessor.scaler.transform(X_raw)
            else:
                X_scaled = X_raw

            proba = self.model.predict_proba(X_scaled)
            # Binary classifier: proba shape (1, 2) or (1, 1)
            prob_up = float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])
            pred_class = "UP" if prob_up >= 0.5 else "DOWN"

            return StrategyPrediction(
                timestamp=feature_vector.timestamp,
                symbol=feature_vector.symbol,
                predicted_class=pred_class,
                probability_up=prob_up,
                model_name=self.model.model_name,
                model_version=self.model_version,
            )
        except Exception as exc:
            logger.warning(
                "Prediction failed for %s at %s: %s",
                feature_vector.symbol,
                feature_vector.timestamp,
                exc,
            )
            return None


class SimplePredictionService:
    """Deterministic prediction service for testing and baseline simulation."""

    def __init__(
        self,
        fixed_probability_up: float = 0.70,
        model_name: str = "simple_rule",
        model_version: str = "v1",
    ) -> None:
        self.fixed_probability_up = fixed_probability_up
        self.model_name = model_name
        self.model_version = model_version

    def predict(self, feature_vector: FeatureVector) -> StrategyPrediction | None:
        """Return deterministic prediction."""
        pred_class = "UP" if self.fixed_probability_up >= 0.5 else "DOWN"
        return StrategyPrediction(
            timestamp=feature_vector.timestamp,
            symbol=feature_vector.symbol,
            predicted_class=pred_class,
            probability_up=self.fixed_probability_up,
            model_name=self.model_name,
            model_version=self.model_version,
        )


class EventLoop:
    """Orchestrates end-to-end event flow across strategy, risk, and execution."""

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        feature_pipeline: FeaturePipeline | None = None,
        prediction_service: PredictionService | None = None,
        strategy_engine: StrategyEngine | None = None,
        risk_engine: RiskEngine | None = None,
        execution_service: ExecutionService | None = None,
        portfolio_manager: PortfolioManager | None = None,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.feature_pipeline = feature_pipeline or FeaturePipeline()
        self.prediction_service = prediction_service
        self.strategy_engine = strategy_engine or StrategyEngine()
        self.risk_engine = risk_engine or RiskEngine(
            config=RiskConfig(fixed_quantity=1.0)
        )
        self.portfolio_manager = portfolio_manager or PortfolioManager(
            initial_capital=self.config.initial_cash
        )

        if execution_service is not None:
            self.execution_service = execution_service
        else:
            market_provider = StaticMarketDataProvider()
            paper_broker = PaperBroker(
                config=ExecutionConfig(
                    initial_cash=self.config.initial_cash,
                    commission_bps=self.config.commission_bps,
                    slippage_bps=self.config.slippage_bps,
                    default_exchange=self.config.default_exchange,
                ),
                market_data_provider=market_provider,
            )
            self.execution_service = ExecutionService(broker=paper_broker)

        self.state = RuntimeState()
        self._rolling_candles: dict[str, list[Candle]] = defaultdict(list)
        self.events: list[EventRecord] = []

    @property
    def clock(self) -> TradingClock:
        """Access the simulated trading clock."""
        return self.state.clock

    def process_market_event(self, event: MarketEvent) -> list[EventRecord]:
        """Process a single market event through the orchestration pipeline.

        Args:
            event: Next chronological MarketEvent.

        Returns:
            list[EventRecord]: List of generated lifecycle event records.

        Raises:
            EventProcessingError: If fail_fast is enabled and an error occurs.
        """
        records: list[EventRecord] = []
        correlation_id = uuid.uuid4().hex

        try:
            # 1. Clock Advance
            self.clock.advance_to(event.timestamp)
            self.state.current_timestamp = event.timestamp

            # 2. Deduplication Check
            if self.config.deduplicate_events and self.state.is_duplicate(
                event.symbol, event.timestamp
            ):
                logger.debug(
                    "Duplicate event (%s, %s) dropped by deduplication policy",
                    event.symbol,
                    event.timestamp,
                )
                return records

            # 3. Record MARKET_DATA Event
            mkt_record = EventRecord(
                correlation_id=correlation_id,
                event_type=EventType.MARKET_DATA,
                timestamp=event.timestamp,
                symbol=event.symbol,
                payload=event.model_dump(mode="json"),
            )
            records.append(mkt_record)
            self.events.append(mkt_record)
            self.state.stats.market_events_processed += 1
            self.state.last_market_price[event.symbol] = event.close

            # Update market data provider price for paper execution
            broker = self.execution_service.broker
            if isinstance(broker, PaperBroker) and isinstance(
                broker.market_data_provider, StaticMarketDataProvider
            ):
                broker.market_data_provider.set_price(event.symbol, event.close)

            # 4. Mark-to-Market Portfolio
            portfolio_state = self.portfolio_manager.mark_to_market(
                timestamp=event.timestamp,
                current_prices=self.state.last_market_price,
            )

            # 5. Feature State & Warmup
            candle = event.to_candle()
            self._rolling_candles[event.symbol].append(candle)

            if len(self._rolling_candles[event.symbol]) < self.config.warmup_period:
                logger.debug(
                    "Warm-up in progress for %s: %d/%d candles",
                    event.symbol,
                    len(self._rolling_candles[event.symbol]),
                    self.config.warmup_period,
                )
                return records

            # Keep rolling window bounded
            max_window = max(self.config.warmup_period * 3, 100)
            window_candles = self._rolling_candles[event.symbol][-max_window:]

            fvs = self.feature_pipeline.generate_feature_vectors(
                window_candles, drop_warmup=True, validate=False
            )
            if not fvs:
                fvs = self.feature_pipeline.generate_feature_vectors(
                    window_candles, drop_warmup=False, validate=False
                )
            if not fvs:
                return records
            latest_fv = fvs[-1]

            # 6. ML Prediction
            if self.prediction_service is None:
                return records

            prediction = self.prediction_service.predict(latest_fv)
            if prediction is None:
                return records

            self.state.stats.predictions_generated += 1
            self.state.last_prediction = prediction
            pred_record = EventRecord(
                correlation_id=correlation_id,
                event_type=EventType.PREDICTION,
                timestamp=event.timestamp,
                symbol=event.symbol,
                payload=prediction.model_dump(mode="json"),
            )
            records.append(pred_record)
            self.events.append(pred_record)

            # 7. Strategy Signal Generation
            signal = self.strategy_engine.generate_signal(prediction)
            self.state.stats.signals_generated += 1
            self.state.last_signal = signal
            sig_record = EventRecord(
                correlation_id=correlation_id,
                event_type=EventType.SIGNAL,
                timestamp=event.timestamp,
                symbol=event.symbol,
                payload=signal.model_dump(mode="json"),
            )
            records.append(sig_record)
            self.events.append(sig_record)

            if signal.action == SignalAction.NO_TRADE:
                return records

            # 8. Risk Engine Evaluation
            decision = self.risk_engine.evaluate(
                signal=signal,
                portfolio_state=portfolio_state,
                current_price=event.close,
            )
            self.state.last_risk_decision = decision
            risk_record = EventRecord(
                correlation_id=correlation_id,
                event_type=EventType.RISK_DECISION,
                timestamp=event.timestamp,
                symbol=event.symbol,
                payload=decision.model_dump(mode="json"),
            )
            records.append(risk_record)
            self.events.append(risk_record)

            if not decision.approved or decision.approved_quantity <= 0:
                self.state.stats.signals_rejected += 1
                return records

            # 9. Execution Service & Paper Fill
            order = self.execution_service.execute_risk_decision(
                decision=decision,
                current_price=event.close,
            )
            if order is None:
                return records

            self.state.stats.orders_submitted += 1
            self.state.last_order = order
            order_record = EventRecord(
                correlation_id=correlation_id,
                event_type=EventType.ORDER,
                timestamp=event.timestamp,
                symbol=event.symbol,
                payload=order.model_dump(mode="json"),
            )
            records.append(order_record)
            self.events.append(order_record)

            if order.status == OrderStatus.FILLED:
                self.state.stats.orders_filled += 1
                fills = getattr(self.execution_service.broker, "fills", [])
                fill = fills[-1] if fills else None
                if fill:
                    self.state.last_fill = fill
                    fill_record = EventRecord(
                        correlation_id=correlation_id,
                        event_type=EventType.FILL,
                        timestamp=event.timestamp,
                        symbol=event.symbol,
                        payload=fill.model_dump(mode="json"),
                    )
                    records.append(fill_record)
                    self.events.append(fill_record)

                # Synchronize authoritative PortfolioManager
                fill_price = order.average_fill_price or event.close
                fill_comm = fill.commission if fill else 0.0
                curr_pos = self.portfolio_manager.get_position(event.symbol)

                if decision.action == SignalAction.LONG:
                    if curr_pos.side == PositionSide.SHORT:
                        self.portfolio_manager.close_position(
                            symbol=event.symbol,
                            exit_price=fill_price,
                            exit_timestamp=event.timestamp,
                            exit_commission=fill_comm,
                        )
                    else:
                        self.portfolio_manager.open_position(
                            symbol=event.symbol,
                            side=PositionSide.LONG,
                            quantity=order.filled_quantity,
                            price=fill_price,
                            timestamp=event.timestamp,
                            commission=fill_comm,
                        )
                elif decision.action == SignalAction.SHORT:
                    if curr_pos.side == PositionSide.LONG:
                        self.portfolio_manager.close_position(
                            symbol=event.symbol,
                            exit_price=fill_price,
                            exit_timestamp=event.timestamp,
                            exit_commission=fill_comm,
                        )
                    else:
                        self.portfolio_manager.open_position(
                            symbol=event.symbol,
                            side=PositionSide.SHORT,
                            quantity=order.filled_quantity,
                            price=fill_price,
                            timestamp=event.timestamp,
                            commission=fill_comm,
                        )

                updated_pstate = self.portfolio_manager.mark_to_market(
                    timestamp=event.timestamp,
                    current_prices=self.state.last_market_price,
                )
                port_record = EventRecord(
                    correlation_id=correlation_id,
                    event_type=EventType.PORTFOLIO_UPDATE,
                    timestamp=event.timestamp,
                    symbol=event.symbol,
                    payload=updated_pstate.model_dump(mode="json"),
                )
                records.append(port_record)
                self.events.append(port_record)

            elif order.status == OrderStatus.REJECTED:
                self.state.stats.orders_rejected += 1

            return records

        except Exception as exc:
            self.state.stats.errors += 1
            err_record = EventRecord(
                correlation_id=correlation_id,
                event_type=EventType.ERROR,
                timestamp=event.timestamp,
                symbol=event.symbol,
                payload={
                    "error_type": exc.__class__.__name__,
                    "message": str(exc),
                },
            )
            records.append(err_record)
            self.events.append(err_record)
            logger.exception("Error processing event at %s: %s", event.timestamp, exc)

            if self.config.fail_fast:
                raise EventProcessingError(
                    f"Fail-fast aborted on {exc.__class__.__name__}: {exc}"
                ) from exc

            return records
