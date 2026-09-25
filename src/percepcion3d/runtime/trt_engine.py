"""Shared TensorRT 10 backend (``cuda-python``) behind a CPU-testable protocol.

Layering:

    Detector / DepthEstimator ──► EngineBackend (Protocol) ──► TrtEngine
                                                            └─► any fake in tests

``TrtEngine`` owns one set of preallocated device/pinned buffers, a CUDA
stream (optionally high priority), start/stop events for GPU timing and an
optional CUDA Graph of the ``execute_async_v3`` call. ``infer_async`` returns
a handle; ``wait()`` synchronises on a per-inference event only (never
``cudaDeviceSynchronize``), so two engines on different streams overlap on the
GPU: the detector on the high-priority stream, depth on the default one (R3).

TensorRT and ``cuda-python`` are imported lazily inside ``TrtEngine`` so the
CPU core stays importable without a GPU.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

# ─── Backend protocol ─────────────────────────────────────────────────────────


@runtime_checkable
class InferenceHandle(Protocol):
    """Result of one asynchronous inference."""

    def wait(self) -> dict[str, NDArray[Any]]:
        """Block until outputs are on the host; returns ``{name: array}`` (batch dim kept)."""
        ...

    def ready(self) -> bool:
        """Non-blocking: ``True`` once the completion event has fired (``wait`` won't block)."""
        ...

    @property
    def gpu_ms(self) -> float | None:
        """Device time between the start/stop events, if the backend measures it."""
        ...


@runtime_checkable
class EngineBackend(Protocol):
    @property
    def input_name(self) -> str: ...

    @property
    def input_shape(self) -> tuple[int, ...]:
        """``(1, H, W, 3)`` for a uint8 NHWC engine."""
        ...

    @property
    def output_names(self) -> tuple[str, ...]: ...

    def infer_async(self, host_input: NDArray[Any]) -> InferenceHandle: ...

    def close(self) -> None: ...


# ─── TensorRT backend ─────────────────────────────────────────────────────────


def _cuda_check(result: Any) -> Any:
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


def _import_cudart() -> Any:
    try:
        from cuda.bindings import runtime as cudart  # cuda-python >= 12.6
    except ImportError:  # pragma: no cover - depends on installed version
        from cuda import cudart
    return cudart


@dataclass
class _Tensor:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    nbytes: int
    device_ptr: int
    host_ptr: int
    host: NDArray[Any] = field(repr=False)


class _TrtHandle:
    def __init__(self, engine: TrtEngine) -> None:
        self._engine = engine
        self._done = False
        self._gpu_ms: float | None = None
        self._outputs: dict[str, NDArray[Any]] = {}

    def wait(self) -> dict[str, NDArray[Any]]:
        if not self._done:
            self._gpu_ms, self._outputs = self._engine._finish()
            self._done = True
        return self._outputs

    def ready(self) -> bool:
        return self._done or self._engine._done_event_fired()

    @property
    def gpu_ms(self) -> float | None:
        return self._gpu_ms


class TrtEngine:
    """TensorRT 10 engine with preallocated buffers, timing events and optional CUDA Graph.

    Requires the ``runtime`` extra (``tensorrt-cu12``, ``cuda-python``). Only one
    inference may be in flight; calling :meth:`infer_async` again first waits
    for the previous one.
    """

    def __init__(
        self,
        engine_path: Path | str,
        use_cuda_graph: bool = True,
        high_priority_stream: bool = True,
        device_index: int = 0,
    ) -> None:
        import tensorrt as trt

        self._trt = trt
        self._cudart = cudart = _import_cudart()
        self._pending: _TrtHandle | None = None

        _cuda_check(cudart.cudaSetDevice(device_index))
        self._logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self._logger, "")
        runtime = trt.Runtime(self._logger)
        blob = Path(engine_path).read_bytes()
        self._engine = runtime.deserialize_cuda_engine(blob)
        if self._engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine {engine_path}")
        self._context = self._engine.create_execution_context()

        least, greatest = _cuda_check(cudart.cudaDeviceGetStreamPriorityRange())
        priority = greatest if high_priority_stream else least
        self._stream = _cuda_check(
            cudart.cudaStreamCreateWithPriority(cudart.cudaStreamNonBlocking, priority)
        )
        self._ev_start = _cuda_check(cudart.cudaEventCreateWithFlags(cudart.cudaEventDefault))
        self._ev_stop = _cuda_check(cudart.cudaEventCreateWithFlags(cudart.cudaEventDefault))
        self._ev_done = _cuda_check(cudart.cudaEventCreateWithFlags(cudart.cudaEventDisableTiming))

        self._inputs: list[_Tensor] = []
        self._outputs: list[_Tensor] = []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = tuple(int(d) for d in self._engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise RuntimeError(
                    f"tensor {name} has dynamic shape {shape}; export static engines"
                )
            dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(name)))
            tensor = self._alloc(name, shape, dtype)
            self._context.set_tensor_address(name, tensor.device_ptr)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._inputs.append(tensor)
            else:
                self._outputs.append(tensor)
        if len(self._inputs) != 1:
            raise RuntimeError(
                f"expected exactly one input tensor, got {[t.name for t in self._inputs]}"
            )

        self._graph_exec: Any | None = None
        if use_cuda_graph:
            self._capture_graph()

    # -- allocation ---------------------------------------------------------

    def _alloc(self, name: str, shape: tuple[int, ...], dtype: np.dtype[Any]) -> _Tensor:
        cudart = self._cudart
        nbytes = int(np.prod(shape)) * dtype.itemsize
        device_ptr = int(_cuda_check(cudart.cudaMalloc(nbytes)))
        host_ptr = int(_cuda_check(cudart.cudaHostAlloc(nbytes, cudart.cudaHostAllocDefault)))
        buf = (ctypes.c_uint8 * nbytes).from_address(host_ptr)
        host = np.frombuffer(memoryview(buf), dtype=np.uint8).view(dtype).reshape(shape)
        return _Tensor(name, shape, dtype, nbytes, device_ptr, host_ptr, host)

    def _capture_graph(self) -> None:
        cudart = self._cudart
        # Warm-up enqueue outside capture (TensorRT requirement).
        if not self._context.execute_async_v3(int(self._stream)):
            raise RuntimeError("execute_async_v3 failed during warm-up")
        _cuda_check(cudart.cudaStreamSynchronize(self._stream))
        _cuda_check(
            cudart.cudaStreamBeginCapture(
                self._stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal
            )
        )
        ok = self._context.execute_async_v3(int(self._stream))
        graph = _cuda_check(cudart.cudaStreamEndCapture(self._stream))
        if not ok:
            raise RuntimeError("execute_async_v3 failed during graph capture")
        self._graph_exec = _cuda_check(cudart.cudaGraphInstantiate(graph, 0))
        _cuda_check(cudart.cudaGraphDestroy(graph))

    # -- EngineBackend ------------------------------------------------------

    @property
    def input_name(self) -> str:
        return self._inputs[0].name

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._inputs[0].shape

    @property
    def input_dtype(self) -> np.dtype[Any]:
        return self._inputs[0].dtype

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self._outputs)

    @property
    def stream(self) -> Any:
        return self._stream

    def infer_async(self, host_input: NDArray[Any]) -> InferenceHandle:
        if self._pending is not None:
            self._pending.wait()
        inp = self._inputs[0]
        src = np.ascontiguousarray(host_input, dtype=inp.dtype)
        if src.shape != inp.shape:
            raise ValueError(f"input shape {src.shape} != engine input {inp.shape}")
        inp.host[...] = src  # into pinned memory
        cudart = self._cudart
        _cuda_check(
            cudart.cudaMemcpyAsync(
                inp.device_ptr,
                inp.host_ptr,
                inp.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self._stream,
            )
        )
        _cuda_check(cudart.cudaEventRecord(self._ev_start, self._stream))
        if self._graph_exec is not None:
            _cuda_check(cudart.cudaGraphLaunch(self._graph_exec, self._stream))
        elif not self._context.execute_async_v3(int(self._stream)):
            raise RuntimeError("execute_async_v3 failed")
        _cuda_check(cudart.cudaEventRecord(self._ev_stop, self._stream))
        for out in self._outputs:
            _cuda_check(
                cudart.cudaMemcpyAsync(
                    out.host_ptr,
                    out.device_ptr,
                    out.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self._stream,
                )
            )
        _cuda_check(cudart.cudaEventRecord(self._ev_done, self._stream))
        self._pending = _TrtHandle(self)
        return self._pending

    def _done_event_fired(self) -> bool:
        cudart = self._cudart
        (err,) = cudart.cudaEventQuery(self._ev_done)
        if err == cudart.cudaError_t.cudaSuccess:
            return True
        if err == cudart.cudaError_t.cudaErrorNotReady:
            return False
        raise RuntimeError(f"cudaEventQuery failed: {err}")

    def _finish(self) -> tuple[float, dict[str, NDArray[Any]]]:
        cudart = self._cudart
        _cuda_check(cudart.cudaEventSynchronize(self._ev_done))
        gpu_ms = float(_cuda_check(cudart.cudaEventElapsedTime(self._ev_start, self._ev_stop)))
        self._pending = None
        return gpu_ms, {t.name: t.host.copy() for t in self._outputs}

    def close(self) -> None:
        cudart = self._cudart
        if self._pending is not None:
            self._pending.wait()
        if self._graph_exec is not None:
            cudart.cudaGraphExecDestroy(self._graph_exec)
            self._graph_exec = None
        for t in self._inputs + self._outputs:
            cudart.cudaFree(t.device_ptr)
            cudart.cudaFreeHost(t.host_ptr)
        self._inputs.clear()
        self._outputs.clear()
        cudart.cudaEventDestroy(self._ev_start)
        cudart.cudaEventDestroy(self._ev_stop)
        cudart.cudaEventDestroy(self._ev_done)
        cudart.cudaStreamDestroy(self._stream)

    def __enter__(self) -> TrtEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
