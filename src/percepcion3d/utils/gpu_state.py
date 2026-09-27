"""Background GPU clock / temperature / power / throttle-reason sampler (NVML).

Latency percentiles on a laptop GPU are only interpretable next to the clock
state they were measured under: a P95 regression can be thermal or power
throttling rather than the code. ``GpuStateSampler`` polls NVML from a daemon
thread and summarises what fraction of the run the GPU spent clock-limited by
each reason, so a benchmark JSON carries its own thermal context.

``query_fn`` is injectable; tests run on CPU-only CI with a fake.

Example::

    with GpuStateSampler(interval_s=0.2) as gpu:
        run_benchmark()
    print(gpu.summary().format())
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from types import TracebackType
from typing import Any

import numpy as np

# NVML ``nvmlClocksThrottleReason*`` bit masks (stable across driver versions).
REASON_BITS: dict[str, int] = {
    "gpu_idle": 0x001,
    "applications_clocks": 0x002,
    "sw_power_cap": 0x004,
    "hw_slowdown": 0x008,
    "sync_boost": 0x010,
    "sw_thermal": 0x020,
    "hw_thermal": 0x040,
    "hw_power_brake": 0x080,
    "display_clocks": 0x100,
}
THERMAL_MASK = REASON_BITS["sw_thermal"] | REASON_BITS["hw_thermal"] | REASON_BITS["hw_slowdown"]
POWER_MASK = REASON_BITS["sw_power_cap"] | REASON_BITS["hw_power_brake"]


@dataclass(frozen=True)
class GpuState:
    sm_clock_mhz: float
    temperature_c: float
    power_w: float
    reasons: int | None
    """Throttle-reason bit mask, ``None`` if the driver does not report it."""


QueryFn = Callable[[], GpuState]


@dataclass(frozen=True)
class GpuStateSummary:
    samples: int
    sm_clock_mhz_min: float
    sm_clock_mhz_p50: float
    sm_clock_mhz_max: float
    temperature_c_max: float
    power_w_p50: float
    power_w_max: float
    thermal_frac: float
    power_frac: float
    reason_frac: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format(self) -> str:
        if self.samples == 0:
            return "GPU state: no samples"
        active = ", ".join(f"{k} {v:.0%}" for k, v in self.reason_frac.items() if v > 0.0)
        return (
            f"GPU state ({self.samples} samples): SM clock min/p50/max "
            f"{self.sm_clock_mhz_min:.0f}/{self.sm_clock_mhz_p50:.0f}/{self.sm_clock_mhz_max:.0f} MHz"
            f" | T max {self.temperature_c_max:.0f} °C | power p50/max "
            f"{self.power_w_p50:.0f}/{self.power_w_max:.0f} W | throttled: thermal "
            f"{self.thermal_frac:.0%}, power {self.power_frac:.0%}"
            + (f" | reasons: {active}" if active else "")
        )


def summarize(states: list[GpuState]) -> GpuStateSummary:
    nan = float("nan")
    if not states:
        return GpuStateSummary(0, nan, nan, nan, nan, nan, nan, nan, nan, {})
    clk = np.array([s.sm_clock_mhz for s in states], dtype=np.float64)
    temp = np.array([s.temperature_c for s in states], dtype=np.float64)
    pwr = np.array([s.power_w for s in states], dtype=np.float64)
    masks = [s.reasons for s in states if s.reasons is not None]

    def frac(bits: int) -> float:
        return sum(1 for m in masks if m & bits) / len(masks) if masks else nan

    return GpuStateSummary(
        samples=len(states),
        sm_clock_mhz_min=float(clk.min()),
        sm_clock_mhz_p50=float(np.median(clk)),
        sm_clock_mhz_max=float(clk.max()),
        temperature_c_max=float(temp.max()),
        power_w_p50=float(np.median(pwr)),
        power_w_max=float(pwr.max()),
        thermal_frac=frac(THERMAL_MASK),
        power_frac=frac(POWER_MASK),
        reason_frac={k: frac(b) for k, b in REASON_BITS.items()} if masks else {},
    )


def nvml_query_fn(device_index: int = 0) -> QueryFn:
    """NVML-backed query for ``device_index``.

    Raises:
        RuntimeError: if ``nvidia-ml-py`` is missing or NVML cannot initialise.
    """
    try:
        import pynvml
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "nvidia-ml-py is required for NVML sampling: uv pip install -e '.[runtime]'"
        ) from exc
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    except pynvml.NVMLError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(f"NVML initialisation failed: {exc}") from exc

    def _q() -> GpuState:  # pragma: no cover - needs a GPU
        def read(fn: Callable[[], float]) -> float:
            try:
                return float(fn())
            except pynvml.NVMLError:
                return float("nan")

        try:
            reasons: int | None = int(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle))
        except pynvml.NVMLError:
            reasons = None
        return GpuState(
            sm_clock_mhz=read(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)),
            temperature_c=read(
                lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            ),
            power_w=read(lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0),
            reasons=reasons,
        )

    return _q


class GpuStateSampler:
    """Poll ``query_fn`` from a daemon thread and keep every sample.

    Args:
        interval_s: Sampling period (default 5 Hz; NVML calls cost ~0.1–1 ms).
        query_fn: Injected query; ``None`` selects NVML on ``start``.
    """

    def __init__(self, interval_s: float = 0.2, query_fn: QueryFn | None = None) -> None:
        if interval_s <= 0.0:
            raise ValueError("interval_s must be > 0")
        self._interval_s = interval_s
        self._query_fn = query_fn
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._states: list[GpuState] = []

    def _run(self, fn: QueryFn) -> None:
        next_t = time.perf_counter()
        while not self._stop.is_set():
            s = fn()
            with self._lock:
                self._states.append(s)
            next_t += self._interval_s
            remaining = next_t - time.perf_counter()
            if remaining > 0.0:
                self._stop.wait(remaining)
            else:
                next_t = time.perf_counter()

    def start(self) -> GpuStateSampler:
        if self._thread is not None:
            raise RuntimeError("GpuStateSampler already started")
        fn = self._query_fn if self._query_fn is not None else nvml_query_fn()
        self._query_fn = fn
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(fn,), name="gpu-state", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> GpuStateSummary:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None
        return self.summary()

    @property
    def states(self) -> list[GpuState]:
        with self._lock:
            return list(self._states)

    def summary(self) -> GpuStateSummary:
        return summarize(self.states)

    def __enter__(self) -> GpuStateSampler:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
