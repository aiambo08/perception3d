"""Detector + depth on two CUDA streams: the R3 contention loop.

Per frame ``k`` (paced at ``pace_hz`` if given, else as fast as the detector)::

    if depth not in flight and k % depth_every == 0:
        h_depth = depth.infer_async(frame_k)      # low-priority stream, enqueued FIRST
    h_det = det.infer_async(frame_k)              # high-priority stream
    dets_k = h_det.wait()                         # only the detector's completion event
    if h_depth.ready():                           # cudaEventQuery, never blocks
        depth_map = h_depth.wait()                # lags its frame by (k - frame_id)

Enqueuing depth *before* the detector is deliberate: it is the worst case for
the detector's tail (its kernels queue behind ViT kernels already running),
which is exactly what the DoD ``P99(det) ≤ 8 ms under contention`` has to
hold against. Depth is never waited for inside the frame loop, so the
detector cadence is preserved and the measured depth rate/lag is what the
dual-rate pipeline (F7) will actually get.

The loop is written against the ``infer_async``/``ready``/``wait``
interfaces only, so it runs unchanged with fake engines in tests.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from percepcion3d.depth.depth_trt import DepthEstimator, DepthHandle, DepthMap
from percepcion3d.detection.detector_trt import Detections, Detector
from percepcion3d.utils.profiling import StageTimer


@dataclass
class ContentionResult:
    frames: int = 0
    depth_maps: int = 0
    depth_skipped: int = 0
    """Depth slots (``k % depth_every == 0``) skipped because the previous map was in flight."""
    lag_frames: list[int] = field(default_factory=list)

    @property
    def depth_rate(self) -> float:
        return self.depth_maps / self.frames if self.frames else 0.0

    @property
    def max_lag(self) -> int:
        return max(self.lag_frames) if self.lag_frames else 0


def run_contention(
    det: Detector | None,
    depth: DepthEstimator | None,
    frames: Iterable[NDArray[np.uint8]],
    timer: StageTimer,
    depth_every: int = 1,
    pace_hz: float | None = None,
    on_result: Callable[[int, Detections | None, DepthMap | None], None] | None = None,
) -> ContentionResult:
    """Run the dual-stream loop; records ``det_e2e``/``depth_e2e`` wall times in ``timer``.

    Either engine may be ``None`` (isolated baselines of the matrix). With
    ``det=None`` each depth inference is waited for synchronously.
    """
    if det is None and depth is None:
        raise ValueError("at least one of det/depth is required")
    if depth_every < 1:
        raise ValueError("depth_every must be >= 1")
    res = ContentionResult()
    pending: tuple[DepthHandle, float] | None = None
    period = 1.0 / pace_hz if pace_hz else None
    t_next = time.perf_counter()

    def _consume(k: int, entry: tuple[DepthHandle, float]) -> DepthMap:
        h, t_start = entry
        m = h.wait()
        timer.record("depth_e2e", (time.perf_counter() - t_start) * 1e3)
        res.depth_maps += 1
        res.lag_frames.append(k - h.frame_id)
        return m

    for k, frame in enumerate(frames):
        if period is not None:
            now = time.perf_counter()
            if now < t_next:
                time.sleep(t_next - now)
            t_next = max(t_next + period, now)
        res.frames += 1
        depth_now: DepthMap | None = None
        if depth is not None and k % depth_every == 0:
            if pending is None:
                pending = (depth.infer_async(frame, frame_id=k), time.perf_counter())
            else:
                res.depth_skipped += 1
        dets: Detections | None = None
        if det is not None:
            t0 = time.perf_counter()
            dets = det.infer_async(frame).wait()
            timer.record("det_e2e", (time.perf_counter() - t0) * 1e3)
        if pending is not None and (det is None or pending[0].ready()):
            depth_now = _consume(k, pending)
            pending = None
        if on_result is not None:
            on_result(k, dets, depth_now)
    if pending is not None:
        last = _consume(res.frames - 1, pending)
        if on_result is not None:
            on_result(res.frames - 1, None, last)
    return res
