"""Per-stage latency accounting with percentile reporting (P50/P95/P99/max).

Design goals: zero allocation on the hot path after warm-up (fixed-capacity
ring buffers), ``perf_counter_ns`` resolution, and a single object that both
CPU stages and externally timed GPU stages (CUDA events) can feed through
:meth:`StageTimer.record`.

Example::

    timer = StageTimer(capacity=4096)
    with timer.stage("rectify"):
        out = rectifier.rectify(frame)
    timer.record("detector", gpu_ms)          # measured elsewhere (CUDA events)
    print(timer.format_table())
    timer.to_json("latency.json")
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_PERCENTILES: tuple[float, float, float] = (50.0, 95.0, 99.0)


@dataclass(frozen=True)
class StageStats:
    """Summary statistics of one stage, all latencies in milliseconds."""

    name: str
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


class _RingBuffer:
    """Fixed-capacity float64 ring buffer; keeps the most recent ``capacity`` samples."""

    __slots__ = ("_buf", "_n", "_pos")

    def __init__(self, capacity: int) -> None:
        self._buf: NDArray[np.float64] = np.empty(capacity, dtype=np.float64)
        self._pos = 0
        self._n = 0

    def push(self, value: float) -> None:
        self._buf[self._pos] = value
        self._pos = (self._pos + 1) % self._buf.shape[0]
        self._n = min(self._n + 1, self._buf.shape[0])

    def values(self) -> NDArray[np.float64]:
        if self._n < self._buf.shape[0]:
            return self._buf[: self._n].copy()
        return np.roll(self._buf, -self._pos)

    def __len__(self) -> int:
        return self._n


class StageTimer:
    """Collects per-stage latencies and reports percentiles.

    Args:
        capacity: Samples kept per stage (oldest are overwritten). 4096 frames
            at 60 Hz is ~68 s of history, enough for P99 with <1 % quantile error.
    """

    def __init__(self, capacity: int = 4096) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = capacity
        self._stages: dict[str, _RingBuffer] = {}
        self._total: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, name: str, latency_ms: float) -> None:
        """Append one latency sample (milliseconds) to ``name``."""
        buf = self._stages.get(name)
        if buf is None:
            buf = _RingBuffer(self._capacity)
            self._stages[name] = buf
            self._total[name] = 0
        buf.push(float(latency_ms))
        self._total[name] += 1

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time the enclosed block with ``perf_counter_ns`` and record it under ``name``."""
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            self.record(name, (time.perf_counter_ns() - t0) * 1e-6)

    def reset(self) -> None:
        self._stages.clear()
        self._total.clear()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def stage_names(self) -> list[str]:
        return list(self._stages)

    def samples(self, name: str) -> NDArray[np.float64]:
        """Retained samples of ``name`` in chronological order (ms)."""
        return self._stages[name].values()

    def total_count(self, name: str) -> int:
        """Samples ever recorded for ``name`` (including those evicted from the ring)."""
        return self._total[name]

    def stats(self, name: str) -> StageStats:
        vals = self.samples(name)
        if vals.size == 0:
            return StageStats(name, 0, float("nan"), float("nan"), float("nan"), float("nan"), 0.0)
        p50, p95, p99 = (float(x) for x in np.percentile(vals, _PERCENTILES))
        return StageStats(
            name=name,
            count=int(vals.size),
            mean_ms=float(vals.mean()),
            p50_ms=p50,
            p95_ms=p95,
            p99_ms=p99,
            max_ms=float(vals.max()),
        )

    def report(self) -> dict[str, StageStats]:
        return {name: self.stats(name) for name in self._stages}

    def format_table(self) -> str:
        """Fixed-width table suitable for a terminal or a Markdown code block."""
        header = f"{'stage':<20}{'n':>7}{'mean':>9}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}  [ms]"
        rows = [header, "-" * len(header)]
        for s in self.report().values():
            rows.append(
                f"{s.name:<20}{s.count:>7}{s.mean_ms:>9.3f}{s.p50_ms:>9.3f}"
                f"{s.p95_ms:>9.3f}{s.p99_ms:>9.3f}{s.max_ms:>9.3f}"
            )
        return "\n".join(rows)

    def to_json(self, path: Path | str) -> None:
        data = {name: asdict(s) for name, s in self.report().items()}
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def to_csv(self, path: Path | str) -> None:
        stats = list(self.report().values())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(StageStats.__dataclass_fields__))
            writer.writeheader()
            for s in stats:
                writer.writerow(asdict(s))


def percentiles_ms(values: NDArray[np.float64] | list[float]) -> tuple[float, float, float]:
    """Convenience: (P50, P95, P99) of a latency sample in the same unit as the input."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot compute percentiles of an empty sample.")
    p50, p95, p99 = (float(x) for x in np.percentile(arr, _PERCENTILES))
    return p50, p95, p99
