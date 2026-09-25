"""2-D detector on a TensorRT engine with NMS inside the graph.

Layering (so that everything except the CUDA calls is CPU-testable):

    Detector ──► EngineBackend (Protocol) ──► TrtEngine (runtime/trt_engine.py)
       │                                  └─► any fake in tests
       └── Letterboxer (CPU resize into a preallocated uint8 canvas)

The engine is expected to have been exported by ``scripts/export_detector.py``:
input ``images_u8 [1,H,W,3] uint8`` (BGR), outputs ``num_dets``, ``det_boxes``
(xyxy in canvas pixels), ``det_scores``, ``det_classes``. ``Detector`` maps the
boxes back to frame coordinates through the exact letterbox inverse.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from percepcion3d.detection.letterbox import Letterboxer, LetterboxParams
from percepcion3d.runtime.trt_engine import EngineBackend, InferenceHandle, TrtEngine
from percepcion3d.utils.profiling import StageTimer

__all__ = [
    "NMS_OUTPUTS",
    "Detections",
    "DetectionsHandle",
    "Detector",
    "DetectorConfig",
    "EngineBackend",
    "InferenceHandle",
    "TrtEngine",
    "postprocess_nms_outputs",
]

NMS_OUTPUTS: tuple[str, str, str, str] = ("num_dets", "det_boxes", "det_scores", "det_classes")


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

    def ready(self) -> bool:
        return self._result is not None or self._handle.ready()


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
