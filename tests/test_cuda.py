"""F7 — cuda-python wrapper against an in-process fake runtime (no GPU)."""

from __future__ import annotations

import ctypes
import enum
import importlib.util
import sys
from typing import Any

import numpy as np
import pytest

from percepcion3d.runtime.cuda import CudaRuntime, cuda_check


class _Err(enum.IntEnum):
    cudaSuccess = 0
    cudaErrorNotReady = 600
    cudaErrorInvalidValue = 1


class _Kind(enum.IntEnum):
    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2


class FakeCudart:
    """Mimics the ``(err, *values)`` calling convention of ``cuda.bindings.runtime``.

    Streams and events are integers; each stream keeps an ordered log of the work
    enqueued on it and events complete when :meth:`tick` is called.
    """

    cudaError_t = _Err
    cudaMemcpyKind = _Kind
    cudaStreamNonBlocking = 1
    cudaEventDefault = 0
    cudaEventDisableTiming = 2
    cudaHostAllocDefault = 0

    def __init__(self) -> None:
        self.device: int | None = None
        self.streams: dict[int, tuple[int, list[str]]] = {}
        self.events: dict[int, tuple[int, bool]] = {}
        self.pending_events: set[int] = set()
        self.host_allocs: dict[int, Any] = {}
        self.device_allocs: dict[int, bytearray] = {}
        self.freed: list[str] = []
        self.copies: list[tuple[str, int, int, int]] = []
        self._next = 100

    def _new(self) -> int:
        self._next += 1
        return self._next

    def cudaSetDevice(self, idx: int) -> tuple[_Err]:
        self.device = idx
        return (_Err.cudaSuccess,)

    def cudaDeviceGetStreamPriorityRange(self) -> tuple[_Err, int, int]:
        return (_Err.cudaSuccess, 0, -5)

    def cudaStreamCreateWithPriority(self, flags: int, priority: int) -> tuple[_Err, int]:
        assert flags == self.cudaStreamNonBlocking
        h = self._new()
        self.streams[h] = (priority, [])
        return (_Err.cudaSuccess, h)

    def cudaStreamSynchronize(self, h: int) -> tuple[_Err]:
        self.streams[h][1].append("sync")
        return (_Err.cudaSuccess,)

    def cudaStreamWaitEvent(self, h: int, ev: int, flags: int) -> tuple[_Err]:
        self.streams[h][1].append(f"wait:{ev}")
        return (_Err.cudaSuccess,)

    def cudaStreamDestroy(self, h: int) -> tuple[_Err]:
        self.freed.append(f"stream:{h}")
        return (_Err.cudaSuccess,)

    def cudaEventCreateWithFlags(self, flags: int) -> tuple[_Err, int]:
        h = self._new()
        self.events[h] = (flags, flags & self.cudaEventDisableTiming == 0)
        return (_Err.cudaSuccess, h)

    def cudaEventRecord(self, ev: int, stream: int) -> tuple[_Err]:
        self.streams[stream][1].append(f"event:{ev}")
        self.pending_events.add(ev)
        return (_Err.cudaSuccess,)

    def cudaEventQuery(self, ev: int) -> tuple[_Err]:
        return (_Err.cudaErrorNotReady if ev in self.pending_events else _Err.cudaSuccess,)

    def cudaEventSynchronize(self, ev: int) -> tuple[_Err]:
        self.pending_events.discard(ev)
        return (_Err.cudaSuccess,)

    def cudaEventElapsedTime(self, a: int, b: int) -> tuple[_Err, float]:
        if not (self.events[a][1] and self.events[b][1]):
            return (_Err.cudaErrorInvalidValue, 0.0)
        return (_Err.cudaSuccess, 4.25)

    def cudaEventDestroy(self, ev: int) -> tuple[_Err]:
        self.freed.append(f"event:{ev}")
        return (_Err.cudaSuccess,)

    def cudaMalloc(self, nbytes: int) -> tuple[_Err, int]:
        h = self._new()
        self.device_allocs[h] = bytearray(nbytes)
        return (_Err.cudaSuccess, h)

    def cudaHostAlloc(self, nbytes: int, flags: int) -> tuple[_Err, int]:
        buf = (ctypes.c_uint8 * nbytes)()
        addr = ctypes.addressof(buf)
        self.host_allocs[addr] = buf  # keep alive
        return (_Err.cudaSuccess, addr)

    def cudaFree(self, ptr: int) -> tuple[_Err]:
        self.freed.append(f"dev:{ptr}")
        return (_Err.cudaSuccess,)

    def cudaFreeHost(self, ptr: int) -> tuple[_Err]:
        self.freed.append(f"host:{ptr}")
        return (_Err.cudaSuccess,)

    def cudaMemcpyAsync(
        self, dst: int, src: int, nbytes: int, kind: _Kind, stream: int
    ) -> tuple[_Err]:
        self.streams[stream][1].append(f"copy:{kind.name}")
        self.copies.append((kind.name, dst, src, nbytes))
        if kind is _Kind.cudaMemcpyHostToDevice:
            self.device_allocs[dst][:] = ctypes.string_at(src, nbytes)
        else:
            ctypes.memmove(dst, bytes(self.device_allocs[src]), nbytes)
        return (_Err.cudaSuccess,)

    def tick(self) -> None:
        """Complete every recorded event (the GPU caught up)."""
        self.pending_events.clear()


def test_cuda_check_unpacks_and_raises() -> None:
    assert cuda_check((_Err.cudaSuccess,)) is None
    assert cuda_check((_Err.cudaSuccess, 7)) == 7
    assert cuda_check((_Err.cudaSuccess, 1, 2)) == (1, 2)
    assert cuda_check(_Err.cudaSuccess) is None
    with pytest.raises(RuntimeError, match="CUDA runtime error"):
        cuda_check((_Err.cudaErrorInvalidValue, 0))


def test_runtime_streams_priorities_events_and_query() -> None:
    fake = FakeCudart()
    rt = CudaRuntime(device_index=1, cudart=fake)
    assert fake.device == 1 and (rt.least_priority, rt.greatest_priority) == (0, -5)
    hi, lo = rt.create_stream(high_priority=True), rt.create_stream(high_priority=False)
    assert hi.priority == -5 and lo.priority == 0 and int(hi) != int(lo)
    ev_t, ev_n = rt.create_event(timing=True), rt.create_event(timing=False)
    assert ev_t.timing and not ev_n.timing
    ev_n.record(hi)
    assert not ev_n.query()  # never blocks; GPU not done yet
    fake.tick()
    assert ev_n.query()
    lo.wait_event(ev_n)  # cross-stream dependency: depth waits for detector's event
    assert fake.streams[lo.handle][1] == [f"wait:{ev_n.handle}"]
    ev_t.record(hi)
    ev_t.synchronize()
    assert ev_t.query()
    with pytest.raises(RuntimeError):
        ev_n.elapsed_ms(ev_t)  # timing disabled → runtime error surfaced, not swallowed
    ev_t2 = rt.create_event(timing=True)
    assert ev_t2.elapsed_ms(ev_t) == pytest.approx(4.25)
    hi.synchronize()
    assert fake.streams[hi.handle][1][-1] == "sync"
    for obj in (hi, lo, ev_t, ev_n, ev_t2):
        obj.destroy()
    assert len([f for f in fake.freed if f.startswith("stream:")]) == 2
    assert len([f for f in fake.freed if f.startswith("event:")]) == 3


def test_pinned_buffer_view_aliases_host_memory_and_copies_round_trip() -> None:
    fake = FakeCudart()
    rt = CudaRuntime(cudart=fake)
    stream = rt.create_stream(high_priority=False)
    buf = rt.alloc_pinned((2, 3), np.dtype(np.float32))
    assert buf.nbytes == 24 and buf.host.shape == (2, 3) and buf.host.dtype == np.float32
    assert buf.host.ctypes.data == buf.host_ptr  # the NumPy view IS the pinned block
    buf.host[...] = np.arange(6, dtype=np.float32).reshape(2, 3)
    rt.copy_h2d_async(buf, stream)
    assert np.frombuffer(fake.device_allocs[buf.device_ptr], np.float32).tolist() == list(range(6))
    buf.host[...] = 0.0
    rt.copy_d2h_async(buf, stream)
    assert buf.host.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    assert [c[0] for c in fake.copies] == ["cudaMemcpyHostToDevice", "cudaMemcpyDeviceToHost"]
    assert fake.streams[stream.handle][1] == [
        "copy:cudaMemcpyHostToDevice",
        "copy:cudaMemcpyDeviceToHost",
    ]
    rt.free_pinned(buf)
    assert f"dev:{buf.device_ptr}" in fake.freed and f"host:{buf.host_ptr}" in fake.freed


def test_runtime_surfaces_device_selection_errors() -> None:
    class Broken(FakeCudart):
        def cudaSetDevice(self, idx: int) -> tuple[_Err]:
            return (_Err.cudaErrorInvalidValue,)

    with pytest.raises(RuntimeError):
        CudaRuntime(device_index=3, cudart=Broken())


def test_cuda_module_import_is_lazy() -> None:
    import percepcion3d.runtime.cuda as mod

    if importlib.util.find_spec("cuda") is None:
        assert "cuda.bindings.runtime" not in sys.modules and "cuda.cudart" not in sys.modules
        with pytest.raises(ImportError):
            mod.import_cudart()
    else:  # pragma: no cover - only with the runtime extra installed
        assert mod.import_cudart() is not None
