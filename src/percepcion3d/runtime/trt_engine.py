"""Shared TensorRT 10 backend (``cuda-python``) behind a CPU-testable protocol.

Layering:

    Detector / DepthEstimator ──► EngineBackend (Protocol) ──► TrtEngine
                                                            └─► any fake in tests

``TrtEngine`` owns one set of preallocated device/pinned buffers, a CUDA
stream (optionally high priority), start/stop events for GPU timing and an
optional CUDA Graph of the ``execute_async_v3`` call — all through
:mod:`percepcion3d.runtime.cuda`. ``infer_async`` returns a handle; ``wait()``
synchronises on a per-inference event only (never ``cudaDeviceSynchronize``),
so two engines on different streams overlap on the GPU: the detector on the
high-priority stream, depth on the low-priority one (R3).

TensorRT and ``cuda-python`` are imported lazily inside ``TrtEngine`` so the
CPU core stays importable without a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from percepcion3d.runtime.cuda import CudaRuntime, CudaStream, PinnedBuffer, cuda_check

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


class _Tensor:
    def __init__(self, name: str, buf: PinnedBuffer) -> None:
        self.name = name
        self.buf = buf

    @property
    def shape(self) -> tuple[int, ...]:
        return self.buf.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.buf.dtype

    @property
    def host(self) -> NDArray[Any]:
        return self.buf.host

    @property
    def device_ptr(self) -> int:
        return self.buf.device_ptr


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
        self._rt = CudaRuntime(device_index)
        self._pending: _TrtHandle | None = None

        self._logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self._logger, "")
        runtime = trt.Runtime(self._logger)
        blob = Path(engine_path).read_bytes()
        self._engine = runtime.deserialize_cuda_engine(blob)
        if self._engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine {engine_path}")
        self._context = self._engine.create_execution_context()

        self._stream = self._rt.create_stream(high_priority=high_priority_stream)
        self._ev_start = self._rt.create_event(timing=True)
        self._ev_stop = self._rt.create_event(timing=True)
        self._ev_done = self._rt.create_event(timing=False)

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
            tensor = _Tensor(name, self._rt.alloc_pinned(shape, dtype))
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

    # -- CUDA Graph -----------------------------------------------------------

    def _capture_graph(self) -> None:
        cudart = self._rt.cudart
        stream = self._stream.handle
        # Warm-up enqueue outside capture (TensorRT requirement).
        if not self._context.execute_async_v3(int(self._stream)):
            raise RuntimeError("execute_async_v3 failed during warm-up")
        self._stream.synchronize()
        cuda_check(
            cudart.cudaStreamBeginCapture(
                stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal
            )
        )
        ok = self._context.execute_async_v3(int(self._stream))
        graph = cuda_check(cudart.cudaStreamEndCapture(stream))
        if not ok:
            raise RuntimeError("execute_async_v3 failed during graph capture")
        self._graph_exec = cuda_check(cudart.cudaGraphInstantiate(graph, 0))
        cuda_check(cudart.cudaGraphDestroy(graph))

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
    def stream(self) -> CudaStream:
        return self._stream

    def infer_async(self, host_input: NDArray[Any]) -> InferenceHandle:
        if self._pending is not None:
            self._pending.wait()
        inp = self._inputs[0]
        src = np.ascontiguousarray(host_input, dtype=inp.dtype)
        if src.shape != inp.shape:
            raise ValueError(f"input shape {src.shape} != engine input {inp.shape}")
        inp.host[...] = src  # into pinned memory
        rt, stream = self._rt, self._stream
        rt.copy_h2d_async(inp.buf, stream)
        self._ev_start.record(stream)
        if self._graph_exec is not None:
            cuda_check(rt.cudart.cudaGraphLaunch(self._graph_exec, stream.handle))
        elif not self._context.execute_async_v3(int(stream)):
            raise RuntimeError("execute_async_v3 failed")
        self._ev_stop.record(stream)
        for out in self._outputs:
            rt.copy_d2h_async(out.buf, stream)
        self._ev_done.record(stream)
        self._pending = _TrtHandle(self)
        return self._pending

    def _done_event_fired(self) -> bool:
        return self._ev_done.query()

    def _finish(self) -> tuple[float, dict[str, NDArray[Any]]]:
        self._ev_done.synchronize()
        gpu_ms = self._ev_stop.elapsed_ms(self._ev_start)
        self._pending = None
        return gpu_ms, {t.name: t.host.copy() for t in self._outputs}

    def close(self) -> None:
        if self._pending is not None:
            self._pending.wait()
        if self._graph_exec is not None:
            self._rt.cudart.cudaGraphExecDestroy(self._graph_exec)
            self._graph_exec = None
        for t in self._inputs + self._outputs:
            self._rt.free_pinned(t.buf)
        self._inputs.clear()
        self._outputs.clear()
        self._ev_start.destroy()
        self._ev_stop.destroy()
        self._ev_done.destroy()
        self._stream.destroy()

    def __enter__(self) -> TrtEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
