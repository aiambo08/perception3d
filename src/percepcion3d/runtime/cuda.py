"""Thin wrapper over the ``cuda-python`` runtime API used by the pipeline.

Everything the runtime needs from CUDA is here: streams with priority, events
(timing and completion), pinned host buffers with a NumPy view, device buffers
and asynchronous H2D/D2H copies. :class:`TrtEngine` builds on it; the dual-rate
loop only sees the stream/event handles through the engines.

``cudart`` is injected so the wrapper is testable with a fake module; the real
one is imported lazily (``cuda.bindings.runtime`` on cuda-python ≥ 12.6, else
``cuda.cudart``), keeping the CPU core importable without a GPU.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


def cuda_check(result: Any) -> Any:
    """Unpack ``(err, *values)`` tuples returned by cuda-python's runtime bindings."""
    if isinstance(result, tuple):
        err, *values = result
    else:
        err, values = result, []
    if int(err) != 0:
        raise RuntimeError(f"CUDA runtime error {err!r}")
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


def import_cudart() -> Any:
    try:
        from cuda.bindings import runtime as cudart  # cuda-python >= 12.6
    except ImportError:  # pragma: no cover - depends on installed version
        from cuda import cudart
    return cudart


@dataclass
class PinnedBuffer:
    """Page-locked host memory with a typed NumPy view (``host``) and a device mirror."""

    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    nbytes: int
    device_ptr: int
    host_ptr: int
    host: NDArray[Any] = field(repr=False)


class CudaStream:
    def __init__(self, rt: CudaRuntime, handle: Any, priority: int) -> None:
        self.rt = rt
        self.handle = handle
        self.priority = priority

    def __int__(self) -> int:
        return int(self.handle)

    def synchronize(self) -> None:
        cuda_check(self.rt.cudart.cudaStreamSynchronize(self.handle))

    def wait_event(self, event: CudaEvent) -> None:
        """Make later work on this stream wait for ``event`` (cross-stream dependency)."""
        cuda_check(self.rt.cudart.cudaStreamWaitEvent(self.handle, event.handle, 0))

    def destroy(self) -> None:
        self.rt.cudart.cudaStreamDestroy(self.handle)


class CudaEvent:
    def __init__(self, rt: CudaRuntime, handle: Any, timing: bool) -> None:
        self.rt = rt
        self.handle = handle
        self.timing = timing

    def record(self, stream: CudaStream) -> None:
        cuda_check(self.rt.cudart.cudaEventRecord(self.handle, stream.handle))

    def query(self) -> bool:
        """``True`` once all work recorded before the event has completed (never blocks)."""
        cudart = self.rt.cudart
        (err,) = cudart.cudaEventQuery(self.handle)
        if err == cudart.cudaError_t.cudaSuccess:
            return True
        if err == cudart.cudaError_t.cudaErrorNotReady:
            return False
        raise RuntimeError(f"cudaEventQuery failed: {err}")

    def synchronize(self) -> None:
        cuda_check(self.rt.cudart.cudaEventSynchronize(self.handle))

    def elapsed_ms(self, start: CudaEvent) -> float:
        return float(cuda_check(self.rt.cudart.cudaEventElapsedTime(start.handle, self.handle)))

    def destroy(self) -> None:
        self.rt.cudart.cudaEventDestroy(self.handle)


class CudaRuntime:
    """Device selection plus factories for streams, events and buffers."""

    def __init__(self, device_index: int = 0, cudart: Any | None = None) -> None:
        self.cudart = cudart if cudart is not None else import_cudart()
        self.device_index = device_index
        cuda_check(self.cudart.cudaSetDevice(device_index))
        least, greatest = cuda_check(self.cudart.cudaDeviceGetStreamPriorityRange())
        self.least_priority = int(least)
        self.greatest_priority = int(greatest)

    def create_stream(self, high_priority: bool) -> CudaStream:
        """Non-blocking stream at the highest (detector) or lowest (depth) priority."""
        priority = self.greatest_priority if high_priority else self.least_priority
        cudart = self.cudart
        handle = cuda_check(
            cudart.cudaStreamCreateWithPriority(cudart.cudaStreamNonBlocking, priority)
        )
        return CudaStream(self, handle, priority)

    def create_event(self, timing: bool) -> CudaEvent:
        cudart = self.cudart
        flags = cudart.cudaEventDefault if timing else cudart.cudaEventDisableTiming
        return CudaEvent(self, cuda_check(cudart.cudaEventCreateWithFlags(flags)), timing)

    def alloc_pinned(self, shape: tuple[int, ...], dtype: np.dtype[Any]) -> PinnedBuffer:
        """Device buffer plus page-locked host mirror; the view aliases the pinned memory."""
        cudart = self.cudart
        nbytes = int(np.prod(shape)) * dtype.itemsize
        device_ptr = int(cuda_check(cudart.cudaMalloc(nbytes)))
        host_ptr = int(cuda_check(cudart.cudaHostAlloc(nbytes, cudart.cudaHostAllocDefault)))
        buf = (ctypes.c_uint8 * nbytes).from_address(host_ptr)
        host = np.frombuffer(memoryview(buf), dtype=np.uint8).view(dtype).reshape(shape)
        return PinnedBuffer(shape, dtype, nbytes, device_ptr, host_ptr, host)

    def free_pinned(self, buf: PinnedBuffer) -> None:
        self.cudart.cudaFree(buf.device_ptr)
        self.cudart.cudaFreeHost(buf.host_ptr)

    def copy_h2d_async(self, buf: PinnedBuffer, stream: CudaStream) -> None:
        cudart = self.cudart
        cuda_check(
            cudart.cudaMemcpyAsync(
                buf.device_ptr,
                buf.host_ptr,
                buf.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                stream.handle,
            )
        )

    def copy_d2h_async(self, buf: PinnedBuffer, stream: CudaStream) -> None:
        cudart = self.cudart
        cuda_check(
            cudart.cudaMemcpyAsync(
                buf.host_ptr,
                buf.device_ptr,
                buf.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream.handle,
            )
        )
