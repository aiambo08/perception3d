"""Background VRAM sampler (NVML) reporting per-process and whole-GPU peaks.

The sampler runs a daemon thread that polls a ``sample_fn`` at a fixed rate
and keeps running peaks. ``sample_fn`` defaults to NVML via ``nvidia-ml-py``
(``runtime`` extra); tests inject a fake function so the logic is exercised
on CPU-only CI.

Example::

    with VramSampler(interval_s=0.1) as vram:
        run_pipeline()
    print(vram.peak)   # VramPeak(process_mb=812.5, gpu_mb=2310.0, samples=143)
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType

SampleFn = Callable[[], tuple[float, float]]
"""Returns ``(process_used_mb, gpu_used_mb)``."""


@dataclass(frozen=True)
class VramPeak:
    process_mb: float
    gpu_mb: float
    samples: int


class DeviceDeltaSampleFn:
    """Per-process VRAM from NVML, falling back to *device used − baseline*.

    Under Windows WDDM (native and WSL2) NVML reports no per-process memory
    (``usedGpuMemory`` is ``None``). The fallback subtracts the device-wide usage
    captured at construction, so build this **before** creating any CUDA
    context: the estimate then includes this process's context and engines,
    plus whatever other processes allocate meanwhile (an upper bound).

    Args:
        device_used_mb: Device-wide used memory.
        process_used_mb: This process's memory, or ``None`` when unavailable.
    """

    def __init__(
        self,
        device_used_mb: Callable[[], float],
        process_used_mb: Callable[[], float | None],
    ) -> None:
        self._device_used_mb = device_used_mb
        self._process_used_mb = process_used_mb
        self.baseline_gpu_mb = device_used_mb()
        self.process_source = "nvml_process"

    def __call__(self) -> tuple[float, float]:
        gpu_mb = self._device_used_mb()
        proc = self._process_used_mb()
        if proc is not None:
            self.process_source = "nvml_process"
            return proc, gpu_mb
        self.process_source = "device_delta"
        return max(0.0, gpu_mb - self.baseline_gpu_mb), gpu_mb


def nvml_sample_fn(device_index: int = 0, pid: int | None = None) -> DeviceDeltaSampleFn:
    """Build a sampler backed by NVML for ``device_index`` and ``pid`` (default: this process).

    See :class:`DeviceDeltaSampleFn` for the fallback used when NVML has no
    per-process accounting (WDDM); check ``.process_source`` after sampling.

    Raises:
        RuntimeError: if ``nvidia-ml-py`` is not installed or NVML cannot initialise
            (no driver / no GPU).
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

    target_pid = os.getpid() if pid is None else pid

    def _device() -> float:  # pragma: no cover - needs a GPU
        return float(pynvml.nvmlDeviceGetMemoryInfo(handle).used) / 2**20

    def _process() -> float | None:  # pragma: no cover - needs a GPU
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        except pynvml.NVMLError:
            return None
        for p in procs:
            if p.pid == target_pid and p.usedGpuMemory is not None:
                return float(p.usedGpuMemory) / 2**20
        return None

    return DeviceDeltaSampleFn(_device, _process)


class VramSampler:
    """Poll ``sample_fn`` from a daemon thread and track peak usage.

    Args:
        interval_s: Sampling period (default 10 Hz).
        sample_fn: Injected sampler; ``None`` selects NVML on first ``start``.
    """

    def __init__(self, interval_s: float = 0.1, sample_fn: SampleFn | None = None) -> None:
        if interval_s <= 0.0:
            raise ValueError("interval_s must be > 0")
        self._interval_s = interval_s
        self._sample_fn = sample_fn
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._peak_process_mb = 0.0
        self._peak_gpu_mb = 0.0
        self._samples = 0

    # ------------------------------------------------------------------

    def _run(self, fn: SampleFn) -> None:
        next_t = time.perf_counter()
        while not self._stop.is_set():
            process_mb, gpu_mb = fn()
            with self._lock:
                self._peak_process_mb = max(self._peak_process_mb, process_mb)
                self._peak_gpu_mb = max(self._peak_gpu_mb, gpu_mb)
                self._samples += 1
            next_t += self._interval_s
            remaining = next_t - time.perf_counter()
            if remaining > 0.0:
                self._stop.wait(remaining)
            else:
                next_t = time.perf_counter()

    def start(self) -> VramSampler:
        if self._thread is not None:
            raise RuntimeError("VramSampler already started")
        fn = self._sample_fn if self._sample_fn is not None else nvml_sample_fn()
        self._sample_fn = fn
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(fn,), name="vram-sampler", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> VramPeak:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None
        return self.peak

    def sample_once(self) -> VramPeak:
        """Take a single synchronous sample (no thread), updating the peaks."""
        fn = self._sample_fn if self._sample_fn is not None else nvml_sample_fn()
        self._sample_fn = fn
        process_mb, gpu_mb = fn()
        with self._lock:
            self._peak_process_mb = max(self._peak_process_mb, process_mb)
            self._peak_gpu_mb = max(self._peak_gpu_mb, gpu_mb)
            self._samples += 1
        return self.peak

    @property
    def peak(self) -> VramPeak:
        with self._lock:
            return VramPeak(self._peak_process_mb, self._peak_gpu_mb, self._samples)

    def __enter__(self) -> VramSampler:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
