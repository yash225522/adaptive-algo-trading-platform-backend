"""Lightweight metrics primitives, registries, and runtime telemetry."""

import math
from collections import defaultdict
from typing import Any


def _format_labels_key(labels: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    """Convert a dictionary of labels to a sorted, hashable tuple key."""
    if not labels:
        return ()
    return tuple(sorted(labels.items()))


def _labels_key_to_str(key: tuple[tuple[str, str], ...]) -> str:
    """Format a hashable labels key tuple into a readable string."""
    if not key:
        return "default"
    return ",".join(f"{k}={v}" for k, v in key)


class Counter:
    """Monotonically increasing cumulative metric counter."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._values: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)

    def increment(
        self, amount: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        """Increment counter by specified positive amount."""
        if amount < 0:
            raise ValueError(
                f"Counter increment amount must be non-negative (got {amount})"
            )
        key = _format_labels_key(labels)
        self._values[key] += amount

    def get_value(self, labels: dict[str, str] | None = None) -> float:
        """Get value for a specific label set."""
        key = _format_labels_key(labels)
        return self._values.get(key, 0.0)

    def total_value(self) -> float:
        """Sum of all label dimensions for this counter."""
        return sum(self._values.values())

    def reset(self) -> None:
        """Clear all counter values."""
        self._values.clear()

    def snapshot(self) -> dict[str, Any]:
        """Produce dictionary snapshot of counter values."""
        return {
            "name": self.name,
            "description": self.description,
            "type": "counter",
            "total": self.total_value(),
            "values": {_labels_key_to_str(k): v for k, v in self._values.items()},
        }


class Gauge:
    """Metric representing a single instantaneous numerical value."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._values: dict[tuple[tuple[str, str], ...], float] = {}

    def set(self, value: float, labels: dict[str, str] | None = None) -> None:
        """Set gauge to a given numerical value."""
        key = _format_labels_key(labels)
        self._values[key] = float(value)

    def get_value(self, labels: dict[str, str] | None = None) -> float:
        """Get current gauge value for a label set."""
        key = _format_labels_key(labels)
        return self._values.get(key, 0.0)

    def reset(self) -> None:
        """Clear all gauge values."""
        self._values.clear()

    def snapshot(self) -> dict[str, Any]:
        """Produce dictionary snapshot of gauge values."""
        return {
            "name": self.name,
            "description": self.description,
            "type": "gauge",
            "values": {_labels_key_to_str(k): v for k, v in self._values.items()},
        }


class Histogram:
    """Metric recording numerical distributions and calculating summary statistics."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._samples: dict[tuple[tuple[str, str], ...], list[float]] = defaultdict(
            list
        )

    def record(self, value: float, labels: dict[str, str] | None = None) -> None:
        """Record a single observation value."""
        key = _format_labels_key(labels)
        self._samples[key].append(float(value))

    def get_stats(self, labels: dict[str, str] | None = None) -> dict[str, float]:
        """Calculate count, sum, min, max, mean, p50, p95, p99 statistics."""
        key = _format_labels_key(labels)
        samples = self._samples.get(key, [])
        if not samples:
            return {
                "count": 0.0,
                "sum": 0.0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        sorted_s = sorted(samples)
        n = len(sorted_s)
        total = sum(sorted_s)

        def percentile(p: float) -> float:
            k = (n - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return sorted_s[int(k)]
            d0 = sorted_s[int(f)] * (c - k)
            d1 = sorted_s[int(c)] * (k - f)
            return d0 + d1

        return {
            "count": float(n),
            "sum": total,
            "min": sorted_s[0],
            "max": sorted_s[-1],
            "mean": total / n,
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
        }

    def reset(self) -> None:
        """Clear all recorded histogram samples."""
        self._samples.clear()

    def snapshot(self) -> dict[str, Any]:
        """Produce dictionary snapshot of histogram distribution statistics."""
        return {
            "name": self.name,
            "description": self.description,
            "type": "histogram",
            "stats": {
                _labels_key_to_str(k): self.get_stats(dict(k) if k else None)
                for k in self._samples
            },
        }


class MetricsRegistry:
    """Central container managing registered metric instances."""

    def __init__(self) -> None:
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}

    def counter(self, name: str, description: str = "") -> Counter:
        """Get or register a Counter metric."""
        if name not in self._counters:
            self._counters[name] = Counter(name=name, description=description)
        return self._counters[name]

    def gauge(self, name: str, description: str = "") -> Gauge:
        """Get or register a Gauge metric."""
        if name not in self._gauges:
            self._gauges[name] = Gauge(name=name, description=description)
        return self._gauges[name]

    def histogram(self, name: str, description: str = "") -> Histogram:
        """Get or register a Histogram metric."""
        if name not in self._histograms:
            self._histograms[name] = Histogram(name=name, description=description)
        return self._histograms[name]

    def reset(self) -> None:
        """Reset all registered metrics to zero."""
        for c in self._counters.values():
            c.reset()
        for g in self._gauges.values():
            g.reset()
        for h in self._histograms.values():
            h.reset()

    def snapshot(self) -> dict[str, Any]:
        """Generate comprehensive metrics snapshot."""
        return {
            "counters": {k: v.snapshot() for k, v in self._counters.items()},
            "gauges": {k: v.snapshot() for k, v in self._gauges.items()},
            "histograms": {k: v.snapshot() for k, v in self._histograms.items()},
        }


# Global metrics registry instance
global_metrics = MetricsRegistry()
