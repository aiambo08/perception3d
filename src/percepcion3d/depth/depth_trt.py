"""Relative depth (Depth Anything V2) on a TensorRT engine.

    DepthEstimator ──► EngineBackend (Protocol) ──► TrtEngine (runtime/trt_engine.py)
         │                                      └─► any fake in tests
         └── DepthResizer (CPU stretch-resize into a preallocated uint8 canvas)

The engine is expected to have been exported by ``scripts/export_depth.py``:
input ``images_u8 [1,H,W,3] uint8`` (BGR; ImageNet normalisation lives in the
graph), single output ``[1,H,W]`` (or ``[1,1,H,W]``) float. For the *relative*
checkpoints the output is an affine-invariant inverse depth,

    d̂(u,v) = s · 1/Z(u,v) + t        (s > 0, t unknown and frame-dependent),

which is why :class:`DepthMap` calls it ``disparity`` and never a distance:
recovering ``(s, t)`` is the job of the metric-fusion stage (F4). The
*metric outdoor* checkpoints output depth in metres directly
(``kind="metric_depth"``); the map keeps that distinction in ``kind`` so
downstream code can call :meth:`DepthMap.inverse_depth` uniformly.

``infer_async`` returns immediately after enqueuing on the engine's stream;
``DepthHandle.wait`` blocks only on that inference's completion event, so a
detector engine on a higher-priority stream keeps running (R3).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from percepcion3d.depth.preprocess import DepthResize, DepthResizer
from percepcion3d.runtime.buffer import FrameStamped
from percepcion3d.runtime.trt_engine import EngineBackend, InferenceHandle
from percepcion3d.utils.profiling import StageTimer

DepthKind = Literal["relative_disparity", "metric_depth"]


@dataclass(frozen=True)
class DepthConfig:
    """Runtime options of :class:`DepthEstimator`."""

    kind: DepthKind = "relative_disparity"
    output_name: str | None = None
    """Engine output to read; ``None`` requires a single-output engine."""
    store_fp16: bool = True
    """Keep the map as float16 (halves host memory; ViT outputs have ~3 significant digits)."""


@dataclass(frozen=True)
class DepthMap:
    """One depth inference at **network** resolution, tagged with its frame metadata."""

    frame_id: int
    t_capture_ns: int
    values: NDArray[np.floating[Any]]
    """``[h, w]`` map; relative disparity or metres depending on ``kind``."""
    kind: DepthKind
    resize: DepthResize
    gpu_ms: float | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.values.shape[0]), int(self.values.shape[1])

    @property
    def disparity(self) -> NDArray[np.floating[Any]]:
        """Alias of ``values`` for relative models (raises for metric ones)."""
        if self.kind != "relative_disparity":
            raise AttributeError("metric depth map has no disparity; use inverse_depth()")
        return self.values

    def inverse_depth(self) -> NDArray[np.float32]:
        """``1/Z`` up to an affine transform: identity for relative, ``1/depth`` for metric."""
        v = self.values.astype(np.float32, copy=False)
        if self.kind == "metric_depth":
            with np.errstate(divide="ignore"):
                return np.where(v > 0, 1.0 / v, np.float32(0.0)).astype(np.float32)
        return v

    def sample(
        self, u_frame: NDArray[np.floating[Any]], v_frame: NDArray[np.floating[Any]]
    ) -> NDArray[np.float32]:
        """Nearest-neighbour lookup at frame pixel coordinates (no map resize)."""
        row, col = self.resize.frame_to_index(u_frame, v_frame)
        return self.values[row, col].astype(np.float32)

    def to_frame_resolution(self, interpolation: int = cv2.INTER_LINEAR) -> NDArray[np.float32]:
        """Upsample the map to the source frame size (float32; costs ~1 ms at 1242×375)."""
        src = self.values.astype(np.float32, copy=False)
        out = cv2.resize(src, (self.resize.src_w, self.resize.src_h), interpolation=interpolation)
        return np.asarray(out, dtype=np.float32)


def extract_depth_output(
    outputs: dict[str, NDArray[Any]],
    output_name: str | None,
    expected_hw: tuple[int, int],
) -> NDArray[np.float32]:
    """Pick and squeeze the depth tensor to ``[h, w]`` float32, validating its shape."""
    if output_name is None:
        if len(outputs) != 1:
            raise ValueError(
                f"engine has {len(outputs)} outputs {sorted(outputs)}; set DepthConfig.output_name"
            )
        (arr,) = outputs.values()
    else:
        if output_name not in outputs:
            raise KeyError(f"output {output_name!r} not in {sorted(outputs)}")
        arr = outputs[output_name]
    h, w = expected_hw
    if arr.shape[-2:] != (h, w) or int(np.prod(arr.shape[:-2], dtype=np.int64)) != 1:
        raise ValueError(f"depth output shape {arr.shape} incompatible with [1,{h},{w}]")
    return np.ascontiguousarray(arr.reshape(h, w), dtype=np.float32)


class DepthHandle:
    """Pending depth inference; ``wait()`` returns the :class:`DepthMap`."""

    def __init__(
        self,
        inner: InferenceHandle,
        frame_id: int,
        t_capture_ns: int,
        resize: DepthResize,
        cfg: DepthConfig,
        timer: StageTimer | None,
    ) -> None:
        self._inner = inner
        self._frame_id = frame_id
        self._t_capture_ns = t_capture_ns
        self._resize = resize
        self._cfg = cfg
        self._timer = timer
        self._result: DepthMap | None = None

    @property
    def frame_id(self) -> int:
        return self._frame_id

    def ready(self) -> bool:
        return self._result is not None or self._inner.ready()

    def wait(self) -> DepthMap:
        if self._result is not None:
            return self._result
        outputs = self._inner.wait()
        gpu_ms = self._inner.gpu_ms
        t0 = time.perf_counter()
        vals = extract_depth_output(
            outputs, self._cfg.output_name, (self._resize.dst_h, self._resize.dst_w)
        )
        stored: NDArray[np.floating[Any]] = (
            vals.astype(np.float16) if self._cfg.store_fp16 else vals
        )
        self._result = DepthMap(
            frame_id=self._frame_id,
            t_capture_ns=self._t_capture_ns,
            values=stored,
            kind=self._cfg.kind,
            resize=self._resize,
            gpu_ms=gpu_ms,
        )
        if self._timer is not None:
            self._timer.record("depth.postprocess", (time.perf_counter() - t0) * 1e3)
            if gpu_ms is not None:
                self._timer.record("depth.gpu", gpu_ms)
        return self._result


class DepthEstimator:
    """Resize → async engine call → :class:`DepthMap` with frame metadata."""

    def __init__(
        self,
        engine: EngineBackend,
        cfg: DepthConfig | None = None,
        timer: StageTimer | None = None,
    ) -> None:
        self.engine = engine
        self.cfg = cfg or DepthConfig()
        self.timer = timer
        shape = tuple(engine.input_shape)
        if len(shape) != 4 or shape[0] != 1 or shape[3] != 3:
            raise ValueError(f"expected engine input [1,H,W,3] uint8, got {shape}")
        if self.cfg.output_name is not None and self.cfg.output_name not in engine.output_names:
            raise ValueError(
                f"output {self.cfg.output_name!r} not among engine outputs {engine.output_names}"
            )
        self.input_hw: tuple[int, int] = (int(shape[1]), int(shape[2]))
        self._resizer = DepthResizer(self.input_hw)

    def infer_async(
        self, frame: NDArray[np.uint8], frame_id: int = 0, t_capture_ns: int = 0
    ) -> DepthHandle:
        t0 = time.perf_counter()
        canvas, resize = self._resizer.apply(frame)
        if self.timer is not None:
            self.timer.record("depth.preprocess", (time.perf_counter() - t0) * 1e3)
        inner = self.engine.infer_async(canvas[None])
        return DepthHandle(inner, frame_id, t_capture_ns, resize, self.cfg, self.timer)

    def infer_stamped(self, fs: FrameStamped) -> DepthHandle:
        return self.infer_async(fs.img, fs.frame_id, fs.t_capture_ns)

    def infer(self, frame: NDArray[np.uint8], frame_id: int = 0, t_capture_ns: int = 0) -> DepthMap:
        return self.infer_async(frame, frame_id, t_capture_ns).wait()

    def close(self) -> None:
        self.engine.close()
