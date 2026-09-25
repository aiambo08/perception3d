"""2-D detector on a TensorRT engine with NMS inside the graph.

Layering (so that everything except the CUDA calls is CPU-testable):

    Detector ──► EngineBackend (Protocol) ──► TrtEngine (TensorRT 10 + cuda-python)
       │                                  └─► any fake in tests
       └── Letterboxer (CPU resize into a preallocated uint8 canvas)

``TrtEngine`` owns one set of preallocated device/pinned buffers, a CUDA
stream (optionally high priority), start/stop events for GPU timing and an
optional CUDA Graph of the ``execute_async_v3`` call. ``infer_async`` returns
a handle; ``wait()`` synchronises on a per-inference event only (never
``cudaDeviceSynchronize``), so a depth engine on another stream keeps running.

The engine is expected to have been exported by ``scripts/export_detector.py``:
input ``images_u8 [1,H,W,3] uint8`` (BGR), outputs ``num_dets``, ``det_boxes``
(xyxy in canvas pixels), ``det_scores``, ``det_classes``. ``Detector`` maps the
boxes back to frame coordinates through the exact letterbox inverse.
"""

from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from percepcion3d.detection.letterbox import Letterboxer, LetterboxParams
from percepcion3d.utils.profiling import StageTimer

NMS_OUTPUTS: tuple[str, str, str, str] = ("num_dets", "det_boxes", "det_scores", "det_classes")


# ─── Backend protocol ─────────────────────────────────────────────────────────


@runtime_checkable
class InferenceHandle(Protocol):
    """Result of one asynchronous inference."""

    def wait(self) -> dict[str, NDArray[Any]]:
        """Block until outputs are on the host; returns ``{name: array}`` (batch dim kept)."""
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


# ─── Detections ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Detections:
    """Boxes in **frame** pixel coordinates (xyxy), one row per detection."""

    boxes: NDArray[np.float32]
    scores: NDArray[np.float32]
    classes: NDArray[np.int32]
    letterbox: LetterboxParams
    gpu_ms: float | None = None

    def __len__(self) -> int:
        return int(self.boxes.shape[0])

    def as_array(self) -> NDArray[np.float32]:
        """``[N,6]`` = ``(x1, y1, x2, y2, score, cls)``."""
        return np.concatenate(
            [self.boxes, self.scores[:, None], self.classes[:, None].astype(np.float32)], axis=1
        )

    def filter(self, keep: NDArray[np.bool_]) -> Detections:
        return Detections(
            self.boxes[keep], self.scores[keep], self.classes[keep], self.letterbox, self.gpu_ms
        )

    @staticmethod
    def empty(letterbox: LetterboxParams) -> Detections:
        return Detections(
            np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32), letterbox
        )


@dataclass(frozen=True)
class DetectorConfig:
    score_threshold: float = 0.25
    min_box_px: float = 2.0
    keep_classes: frozenset[int] | None = None
    class_names: tuple[str, ...] = ()
    fill_value: int = 114

    def name_of(self, cls: int) -> str:
        return self.class_names[cls] if 0 <= cls < len(self.class_names) else str(cls)


def postprocess_nms_outputs(
    outputs: dict[str, NDArray[Any]],
    params: LetterboxParams,
    cfg: DetectorConfig,
    gpu_ms: float | None = None,
) -> Detections:
    """Turn raw ``EfficientNMS_TRT`` outputs into frame-coordinate :class:`Detections`."""
    n = int(np.asarray(outputs[NMS_OUTPUTS[0]]).reshape(-1)[0])
    if n <= 0:
        return Detections.empty(params)
    boxes_net = np.asarray(outputs[NMS_OUTPUTS[1]]).reshape(-1, 4)[:n]
    scores = np.asarray(outputs[NMS_OUTPUTS[2]]).reshape(-1)[:n].astype(np.float32)
    classes = np.asarray(outputs[NMS_OUTPUTS[3]]).reshape(-1)[:n].astype(np.int32)

    boxes = params.clip_to_frame(params.to_frame(boxes_net))
    wh = boxes[:, 2:] - boxes[:, :2]
    keep = (
        (scores >= cfg.score_threshold)
        & (wh[:, 0] >= cfg.min_box_px)
        & (wh[:, 1] >= cfg.min_box_px)
    )
    if cfg.keep_classes is not None:
        keep &= np.isin(classes, np.fromiter(cfg.keep_classes, dtype=np.int32))
    return Detections(boxes[keep].astype(np.float32), scores[keep], classes[keep], params, gpu_ms)


# ─── Detector ─────────────────────────────────────────────────────────────────


class DetectionsHandle:
    """Pending detection; :meth:`wait` finishes the GPU work and post-processes."""

    def __init__(
        self,
        handle: InferenceHandle,
        params: LetterboxParams,
        cfg: DetectorConfig,
        timer: StageTimer | None,
    ) -> None:
        self._handle = handle
        self._params = params
        self._cfg = cfg
        self._timer = timer
        self._result: Detections | None = None

    def wait(self) -> Detections:
        if self._result is None:
            outputs = self._handle.wait()
            gpu_ms = self._handle.gpu_ms
            t0 = time.perf_counter_ns()
            self._result = postprocess_nms_outputs(outputs, self._params, self._cfg, gpu_ms)
            if self._timer is not None:
                self._timer.record("det.postprocess", (time.perf_counter_ns() - t0) * 1e-6)
                if gpu_ms is not None:
                    self._timer.record("det.gpu", gpu_ms)
        return self._result


class Detector:
    """Letterbox → engine → boxes in frame coordinates."""

    def __init__(
        self,
        engine: EngineBackend,
        cfg: DetectorConfig | None = None,
        timer: StageTimer | None = None,
    ) -> None:
        cfg = cfg if cfg is not None else DetectorConfig()
        shape = engine.input_shape
        if len(shape) != 4 or shape[0] != 1 or shape[3] != 3:
            raise ValueError(f"expected engine input (1,H,W,3) uint8, got {shape}")
        missing = [n for n in NMS_OUTPUTS if n not in engine.output_names]
        if missing:
            raise ValueError(f"engine lacks EfficientNMS outputs {missing}: {engine.output_names}")
        self.engine = engine
        self.cfg = cfg
        self.timer = timer
        self.input_hw: tuple[int, int] = (int(shape[1]), int(shape[2]))
        self._letterbox = Letterboxer(self.input_hw, fill=cfg.fill_value)

    def infer_async(self, frame: NDArray[np.uint8]) -> DetectionsHandle:
        t0 = time.perf_counter_ns()
        canvas, params = self._letterbox.apply(frame)
        handle = self.engine.infer_async(canvas[None])
        if self.timer is not None:
            self.timer.record("det.preprocess", (time.perf_counter_ns() - t0) * 1e-6)
        return DetectionsHandle(handle, params, self.cfg, self.timer)

    def infer(self, frame: NDArray[np.uint8]) -> Detections:
        return self.infer_async(frame).wait()

    def close(self) -> None:
        self.engine.close()


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
