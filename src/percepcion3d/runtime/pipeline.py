"""Dual-rate perception loop (F7): one thread, two GPU streams, one ``frame_id`` end to end.

Per captured frame ``k`` (:meth:`Pipeline.process`)::

    h_boxes = boxes.submit(frame_k)                                               # high-prio stream
    if depth idle and cadence.due(k):   h_depth = depth.infer_stamped(frame_k)   # low-prio stream
    boxes_k = h_boxes.wait()
    if h_depth.ready():                 map_j = h_depth.wait()   (j ≤ k; never waited for)
    net_boxes = history.match(boxes_k, j)          # boxes of frame j for sampling map_j (R6)
    meas  = fusion.process(k, boxes_k, map_j, net_boxes, age)
    tracks = tracker.step(...);  alerts = safety.step(...)
    sink.log(FrameResult)                          # asynchronous, never blocks the loop

The reactive path (boxes → fusion → tracks → alerts) runs every frame; the depth
map is refreshed at the rate the GPU sustains, governed by :class:`DepthCadence`
(``depth_every_n_frames`` adapted from the measured depth turnaround). Every
result carries the capture ``frame_id``/timestamp, the ``frame_id`` of the map
it used and the map's wall-clock age, so latency and staleness are measurable
per frame (:class:`PipelineStats`).

Two clocks: ``FrameStamped.t_capture_ns`` is the *scene* clock (dataset
timestamps when replaying, the capture clock live) and drives the KF/TTC;
``t_ingress_ns`` is the local monotonic clock at the moment the frame entered
the loop and measures capture → alert latency and map age. Live they coincide.

Frame delivery is separate from processing: :func:`iter_paced` replays a source
in the loop thread at a target rate with latest-wins skipping, and
:func:`iter_slot` + :func:`start_pump` do the same through a capture thread and
a :class:`LatestFrameSlot` (the variant the plan asks to measure if the P99
fails on CPU time).
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from percepcion3d.depth.depth_trt import DepthMap
from percepcion3d.depth.fusion import Measurement3D, MetricFusionStage
from percepcion3d.detection.detector_trt import DetectionsHandle, Detector, DetectorConfig
from percepcion3d.eval.detection import match_greedy
from percepcion3d.runtime.buffer import FrameStamped, LatestFrameSlot
from percepcion3d.runtime.playback import _sleep_precise
from percepcion3d.safety.gates import AlertLevel, SafetyConfig
from percepcion3d.safety.kinematics import kinematics_from_track
from percepcion3d.safety.ttc import Alert, SafetyState, initial_state, max_level, step
from percepcion3d.tracking.ego_motion import EgoMotionProvider
from percepcion3d.tracking.kalman_filter import EgoDelta
from percepcion3d.tracking.tracker3d import Track3D, Tracker3D, measurement_from_fusion
from percepcion3d.utils.profiling import StageTimer, percentiles_ms

# ─── Stage protocols ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Boxes2D:
    boxes: NDArray[np.float64]
    """``[n, 4]`` xyxy in frame pixels."""
    scores: NDArray[np.float64]
    classes: list[str]

    @staticmethod
    def empty() -> Boxes2D:
        return Boxes2D(np.zeros((0, 4)), np.zeros(0), [])

    def __len__(self) -> int:
        return int(self.boxes.shape[0])


class BoxHandle(Protocol):
    def wait(self) -> Boxes2D: ...


class BoxStage(Protocol):
    """2D detections for a frame; ``submit`` enqueues, ``wait`` finishes."""

    def submit(self, fs: FrameStamped) -> BoxHandle: ...


class DepthHandleLike(Protocol):
    @property
    def frame_id(self) -> int: ...

    def ready(self) -> bool: ...

    def wait(self) -> DepthMap: ...


class DepthStage(Protocol):
    """Anything with :meth:`DepthEstimator.infer_stamped`."""

    def infer_stamped(self, fs: FrameStamped) -> DepthHandleLike: ...


class _ReadyBoxes:
    def __init__(self, b: Boxes2D) -> None:
        self._b = b

    def wait(self) -> Boxes2D:
        return self._b


class CallableBoxStage:
    """Synchronous boxes (GT labels, cached detections, tests)."""

    def __init__(self, fn: Callable[[FrameStamped], Boxes2D]) -> None:
        self._fn = fn

    def submit(self, fs: FrameStamped) -> BoxHandle:
        return _ReadyBoxes(self._fn(fs))


class _DetectorHandle:
    def __init__(self, handle: DetectionsHandle, cfg: DetectorConfig) -> None:
        self._h = handle
        self._cfg = cfg

    def wait(self) -> Boxes2D:
        d = self._h.wait()
        return Boxes2D(
            d.boxes.astype(np.float64),
            d.scores.astype(np.float64),
            [self._cfg.name_of(int(c)) for c in d.classes],
        )


class DetectorBoxStage:
    """:class:`Detector` on the high-priority stream, class ids mapped to names."""

    def __init__(self, det: Detector) -> None:
        self.det = det

    def submit(self, fs: FrameStamped) -> BoxHandle:
        return _DetectorHandle(self.det.infer_async(fs.img), self.det.cfg)


# ─── Depth cadence ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DepthCadenceConfig:
    n_min: int = 1
    n_max: int = 8
    n_init: int = 2
    margin: float = 1.1
    """Enqueue period = ``ceil(margin · turnaround / frame_period)`` frames."""
    ema: float = 0.2
    """Weight of the newest turnaround in the running estimate."""


class DepthCadence:
    """Adaptive ``depth_every_n_frames``.

    Depth is enqueued when ``k − k_last ≥ n``. ``n`` follows the measured turnaround
    (enqueue → map consumed) so that the next map is normally ready by the time the
    next slot comes: too small an ``n`` only produces skipped slots (the previous
    map still in flight) and GPU contention with the detector; too large wastes
    depth rate. ``n`` grows by one on every skipped slot and shrinks by at most one
    per completed map, so it settles from above.
    """

    def __init__(self, cfg: DepthCadenceConfig, frame_period_ms: float) -> None:
        if cfg.n_min < 1 or cfg.n_max < cfg.n_min or not (cfg.n_min <= cfg.n_init <= cfg.n_max):
            raise ValueError("need 1 <= n_min <= n_init <= n_max")
        if frame_period_ms <= 0.0:
            raise ValueError("frame_period_ms must be positive")
        self.cfg = cfg
        self.frame_period_ms = frame_period_ms
        self.n = cfg.n_init
        self.turnaround_ms: float | None = None
        self._k_last: int | None = None
        self.enqueued = 0
        self.skipped = 0

    def due(self, k: int) -> bool:
        return self._k_last is None or k - self._k_last >= self.n

    def on_enqueue(self, k: int) -> None:
        self._k_last = k
        self.enqueued += 1

    def on_skip(self) -> None:
        self.skipped += 1
        self.n = min(self.cfg.n_max, self.n + 1)

    def on_complete(self, turnaround_ms: float) -> None:
        c = self.cfg
        if self.turnaround_ms is None:
            self.turnaround_ms = turnaround_ms
        else:
            self.turnaround_ms += c.ema * (turnaround_ms - self.turnaround_ms)
        target = int(math.ceil(c.margin * self.turnaround_ms / self.frame_period_ms - 1e-6))
        target = min(c.n_max, max(c.n_min, target))
        self.n = target if target >= self.n else self.n - 1


# ─── Box history (R6) ─────────────────────────────────────────────────────────


class BoxHistory:
    """Boxes and ingress time of the last ``capacity`` frames, keyed by ``frame_id``.

    :meth:`net_boxes` returns, for every current box, the box of the same object in
    frame ``j`` (greedy IoU), falling back to the current box when unmatched — so a
    stale depth map is read where the object *was* when the map was rendered.
    """

    def __init__(self, capacity: int = 64, iou_threshold: float = 0.3) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.iou_threshold = iou_threshold
        self._items: OrderedDict[int, tuple[Boxes2D, int]] = OrderedDict()

    def push(self, frame_id: int, boxes: Boxes2D, t_ingress_ns: int) -> None:
        self._items[frame_id] = (boxes, t_ingress_ns)
        self._items.move_to_end(frame_id)
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)

    def get(self, frame_id: int) -> tuple[Boxes2D, int] | None:
        return self._items.get(frame_id)

    def net_boxes(self, current: Boxes2D, frame_id: int) -> tuple[NDArray[np.float64] | None, int]:
        """``(boxes_to_sample, n_matched)``; ``None`` when frame ``j`` is unknown/current."""
        old = self._items.get(frame_id)
        if old is None or len(current) == 0 or len(old[0]) == 0:
            return None, 0
        assign, _ = match_greedy(current.boxes, current.scores, old[0].boxes, self.iou_threshold)
        out = current.boxes.copy()
        hit = assign >= 0
        out[hit] = old[0].boxes[assign[hit]]
        return out, int(hit.sum())

    def __len__(self) -> int:
        return len(self._items)


# ─── Results ──────────────────────────────────────────────────────────────────


@dataclass
class FrameResult:
    frame_id: int
    t_capture_ns: int
    t_ingress_ns: int
    boxes: Boxes2D
    measurements: list[Measurement3D]
    tracks: list[Track3D]
    alerts: list[Alert]
    level: AlertLevel
    depth_frame_id: int | None
    """Map used by the fusion this frame (``None`` = geometry + height prior only)."""
    depth_lag_frames: int
    depth_age_ms: float
    """Wall-clock age of the map when fused (``nan`` without a map)."""
    depth_new: DepthMap | None
    """Set on the frame that consumed a fresh map, for subsampled telemetry."""
    net_boxes_matched: int
    timings_ms: dict[str, float] = field(default_factory=dict)
    """``boxes``, ``depth_wait``, ``fusion``, ``tracker``, ``safety``, ``e2e_alert``."""


class TelemetrySink(Protocol):
    def log(self, result: FrameResult) -> None: ...

    def close(self) -> None: ...


@dataclass
class PipelineStats:
    frames: int = 0
    dropped: int = 0
    """Frames skipped by latest-wins delivery (pacer or slot), before processing."""
    elapsed_s: float = 0.0
    depth_maps: int = 0
    depth_enqueued: int = 0
    depth_skipped: int = 0
    depth_n_final: int = 0
    e2e_alert_ms: list[float] = field(default_factory=list)
    loop_ms: list[float] = field(default_factory=list)
    depth_age_ms: list[float] = field(default_factory=list)
    depth_lag_frames: list[int] = field(default_factory=list)
    depth_turnaround_ms: list[float] = field(default_factory=list)
    telemetry_ms: list[float] = field(default_factory=list)
    boxes_ms_depth: list[float] = field(default_factory=list)
    """``boxes`` time of frames that enqueued a depth inference."""
    boxes_ms_no_depth: list[float] = field(default_factory=list)
    e2e_alert_ms_depth: list[float] = field(default_factory=list)
    e2e_alert_ms_no_depth: list[float] = field(default_factory=list)
    alerts_by_level: dict[str, int] = field(default_factory=dict)
    series: list[dict[str, float]] = field(default_factory=list)
    """One entry per ``series_period_s`` of loop time (see :meth:`Pipeline.run`)."""

    @property
    def input_hz(self) -> float:
        n = self.frames + self.dropped
        return n / self.elapsed_s if self.elapsed_s > 0 else float("nan")

    @property
    def processed_hz(self) -> float:
        return self.frames / self.elapsed_s if self.elapsed_s > 0 else float("nan")

    @property
    def depth_hz(self) -> float:
        return self.depth_maps / self.elapsed_s if self.elapsed_s > 0 else float("nan")

    @property
    def drop_frac(self) -> float:
        n = self.frames + self.dropped
        return self.dropped / n if n else 0.0

    def to_dict(self) -> dict[str, Any]:
        def pct(v: Sequence[float]) -> dict[str, float]:
            if not v:
                return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan"), "n": 0}
            p50, p95, p99 = percentiles_ms(list(v))
            return {"p50": p50, "p95": p95, "p99": p99, "n": len(v)}

        return {
            "frames": self.frames,
            "dropped": self.dropped,
            "drop_frac": self.drop_frac,
            "elapsed_s": self.elapsed_s,
            "input_hz": self.input_hz,
            "processed_hz": self.processed_hz,
            "depth": {
                "maps": self.depth_maps,
                "enqueued": self.depth_enqueued,
                "skipped": self.depth_skipped,
                "hz": self.depth_hz,
                "n_final": self.depth_n_final,
                "age_ms": pct(self.depth_age_ms),
                "lag_frames": pct([float(v) for v in self.depth_lag_frames]),
                "turnaround_ms": pct(self.depth_turnaround_ms),
            },
            "e2e_alert_ms": pct(self.e2e_alert_ms),
            "loop_ms": pct(self.loop_ms),
            "telemetry_ms": pct(self.telemetry_ms),
            "by_depth_enqueue": {
                "boxes_ms": {
                    "depth": pct(self.boxes_ms_depth),
                    "no_depth": pct(self.boxes_ms_no_depth),
                },
                "e2e_alert_ms": {
                    "depth": pct(self.e2e_alert_ms_depth),
                    "no_depth": pct(self.e2e_alert_ms_no_depth),
                },
            },
            "alerts_by_level": dict(self.alerts_by_level),
            "series": list(self.series),
        }


# ─── Pipeline ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineConfig:
    frame_hz: float = 60.0
    """Nominal input rate; sets the frame period the depth cadence reasons in."""
    cadence: DepthCadenceConfig = field(default_factory=DepthCadenceConfig)
    box_history: int = 64
    net_box_iou: float = 0.3
    ego_front_m: float = 1.5
    depth_first: bool = False
    """Enqueue depth before the detector (the R3 worst case for the detector's tail).
    Default: detector first, so its kernels are not queued behind a running depth pass."""
    series_period_s: float = 1.0


class Pipeline:
    def __init__(
        self,
        boxes: BoxStage,
        fusion: MetricFusionStage,
        tracker: Tracker3D,
        safety: SafetyConfig,
        depth: DepthStage | None = None,
        sink: TelemetrySink | None = None,
        cfg: PipelineConfig | None = None,
        timer: StageTimer | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.cfg = cfg if cfg is not None else PipelineConfig()
        self.boxes = boxes
        self.depth = depth
        self.fusion = fusion
        self.tracker = tracker
        self.safety_cfg = safety
        self.sink = sink
        self.timer = timer if timer is not None else StageTimer()
        self.clock_ns = clock_ns
        self.cadence = DepthCadence(self.cfg.cadence, 1e3 / self.cfg.frame_hz)
        self.history = BoxHistory(self.cfg.box_history, self.cfg.net_box_iou)
        self.stats = PipelineStats()
        self.safety_state: SafetyState = initial_state()
        self.newest_map: DepthMap | None = None
        self._pending: tuple[DepthHandleLike, int] | None = None
        self._last_frame_id: int | None = None
        self._series = _SeriesWindow(self.cfg.series_period_s)

    # -- one frame -------------------------------------------------------------

    def _poll_depth(self) -> DepthMap | None:
        if self._pending is None or not self._pending[0].ready():
            return None
        h, t_enq = self._pending
        m = h.wait()
        self._pending = None
        turnaround = (self.clock_ns() - t_enq) * 1e-6
        self.cadence.on_complete(turnaround)
        self.stats.depth_turnaround_ms.append(turnaround)
        self.stats.depth_maps += 1
        self.timer.record("depth.turnaround", turnaround)
        self.newest_map = m
        return m

    def _maybe_enqueue_depth(self, fs: FrameStamped) -> bool:
        enqueued = False
        if self.depth is not None and self.cadence.due(fs.frame_id):
            if self._pending is None:
                self._pending = (self.depth.infer_stamped(fs), self.clock_ns())
                self.cadence.on_enqueue(fs.frame_id)
                enqueued = True
            else:
                self.cadence.on_skip()
        self.stats.depth_enqueued = self.cadence.enqueued
        self.stats.depth_skipped = self.cadence.skipped
        return enqueued

    def process(self, fs: FrameStamped, t_ingress_ns: int | None = None) -> FrameResult:
        """Run the reactive path for ``fs``; returns everything stamped with ``fs.frame_id``."""
        if self._last_frame_id is not None and fs.frame_id <= self._last_frame_id:
            raise ValueError(f"frame_id must increase: {fs.frame_id} after {self._last_frame_id}")
        self._last_frame_id = fs.frame_id
        k = fs.frame_id
        t_in = self.clock_ns() if t_ingress_ns is None else t_ingress_ns
        t0 = self.clock_ns()
        timings: dict[str, float] = {}

        depth_now = False
        if self.cfg.depth_first:
            depth_now = self._maybe_enqueue_depth(fs)
        t = self.clock_ns()
        h_boxes = self.boxes.submit(fs)
        if not self.cfg.depth_first:
            depth_now = self._maybe_enqueue_depth(fs)
        b = h_boxes.wait()
        timings["boxes"] = (self.clock_ns() - t) * 1e-6

        t = self.clock_ns()
        new_map = self._poll_depth()
        timings["depth_wait"] = (self.clock_ns() - t) * 1e-6

        self.history.push(k, b, t_in)
        m = self.newest_map
        net_boxes: NDArray[np.float64] | None = None
        matched = 0
        age_ms = float("nan")
        lag = 0
        if m is not None:
            lag = k - m.frame_id
            old = self.history.get(m.frame_id)
            age_ms = (t_in - old[1]) * 1e-6 if old is not None else float("nan")
            if lag > 0:
                net_boxes, matched = self.history.net_boxes(b, m.frame_id)

        t = self.clock_ns()
        out = self.fusion.process(k, fs.t_capture_ns, b.boxes, b.classes, m, net_boxes)
        meas = [measurement_from_fusion(x) for x in out.measurements]
        timings["fusion"] = (self.clock_ns() - t) * 1e-6

        t = self.clock_ns()
        tracks = self.tracker.step(fs.t_capture_ns, b.boxes, b.scores, b.classes, meas)
        timings["tracker"] = (self.clock_ns() - t) * 1e-6

        t = self.clock_ns()
        kin = [kinematics_from_track(tr, self.cfg.ego_front_m) for tr in tracks]
        self.safety_state, alerts = step(self.safety_state, kin, self.safety_cfg, fs.t_capture_ns)
        level = max_level(alerts)
        t_alert = self.clock_ns()
        timings["safety"] = (t_alert - t) * 1e-6
        timings["e2e_alert"] = (t_alert - t_in) * 1e-6
        timings["loop"] = (t_alert - t0) * 1e-6

        res = FrameResult(
            frame_id=k,
            t_capture_ns=fs.t_capture_ns,
            t_ingress_ns=t_in,
            boxes=b,
            measurements=out.measurements,
            tracks=tracks,
            alerts=alerts,
            level=level,
            depth_frame_id=None if m is None else m.frame_id,
            depth_lag_frames=lag,
            depth_age_ms=age_ms,
            depth_new=new_map,
            net_boxes_matched=matched,
            timings_ms=timings,
        )
        if self.sink is not None:
            t = self.clock_ns()
            self.sink.log(res)
            tel = (self.clock_ns() - t) * 1e-6
            timings["telemetry"] = tel
            self.stats.telemetry_ms.append(tel)
            self.timer.record("telemetry", tel)

        st = self.stats
        st.frames += 1
        st.e2e_alert_ms.append(timings["e2e_alert"])
        st.loop_ms.append(timings["loop"])
        if depth_now:
            st.boxes_ms_depth.append(timings["boxes"])
            st.e2e_alert_ms_depth.append(timings["e2e_alert"])
        else:
            st.boxes_ms_no_depth.append(timings["boxes"])
            st.e2e_alert_ms_no_depth.append(timings["e2e_alert"])
        self._series.add(t_alert, timings["e2e_alert"], timings["boxes"], new_map is not None)
        if m is not None:
            st.depth_age_ms.append(age_ms)
            st.depth_lag_frames.append(lag)
        name = level.name
        st.alerts_by_level[name] = st.alerts_by_level.get(name, 0) + 1
        for key in ("boxes", "depth_wait", "fusion", "tracker", "safety", "e2e_alert", "loop"):
            self.timer.record(key, timings[key])
        return res

    # -- whole run -----------------------------------------------------------------

    def run(
        self,
        frames: Iterable[tuple[FrameStamped, int]],
        on_result: Callable[[FrameResult], None] | None = None,
        dropped: Callable[[], int] | None = None,
    ) -> PipelineStats:
        """Process ``(frame, t_ingress_ns)`` pairs until exhausted; closes the sink.

        Also fills :attr:`PipelineStats.series`: per ``series_period_s`` window of
        loop time, frames, dropped frames, depth maps and e2e/boxes percentiles.
        """
        t_start = self.clock_ns()
        self._series.start(t_start, dropped)
        try:
            for fs, t_in in frames:
                res = self.process(fs, t_in)
                if on_result is not None:
                    on_result(res)
        finally:
            self._series.flush()
            self.stats.series = self._series.rows
            if self._pending is not None:
                self._pending[0].wait()
                self._pending = None
            self.stats.elapsed_s = (self.clock_ns() - t_start) * 1e-9
            self.stats.depth_n_final = self.cadence.n
            if dropped is not None:
                self.stats.dropped = dropped()
            if self.sink is not None:
                self.sink.close()
        return self.stats


class _SeriesWindow:
    def __init__(self, period_s: float) -> None:
        if period_s <= 0:
            raise ValueError("series_period_s must be > 0")
        self.period_ns = int(round(period_s * 1e9))
        self.rows: list[dict[str, float]] = []
        self._t0: int | None = None
        self._dropped: Callable[[], int] | None = None
        self._reset(0, 0)

    def _reset(self, idx: int, dropped_base: int) -> None:
        self._idx = idx
        self._drop_base = dropped_base
        self._e2e: list[float] = []
        self._boxes: list[float] = []
        self._maps = 0

    def start(self, t_ns: int, dropped: Callable[[], int] | None) -> None:
        self._t0 = t_ns
        self._dropped = dropped
        self.rows = []
        self._reset(0, dropped() if dropped is not None else 0)

    def add(self, t_ns: int, e2e_ms: float, boxes_ms: float, new_map: bool) -> None:
        if self._t0 is None:
            return
        idx = (t_ns - self._t0) // self.period_ns
        if idx != self._idx:
            self._emit()
            self._reset(idx, self._drop_now())
        self._e2e.append(e2e_ms)
        self._boxes.append(boxes_ms)
        self._maps += int(new_map)

    def flush(self) -> None:
        if self._t0 is not None:
            self._emit()
            self._t0 = None

    def _drop_now(self) -> int:
        return self._dropped() if self._dropped is not None else 0

    def _emit(self) -> None:
        if not self._e2e:
            return
        e50, _, e99 = percentiles_ms(self._e2e)
        _, _, b99 = percentiles_ms(self._boxes)
        self.rows.append(
            {
                "t_s": self._idx * self.period_ns * 1e-9,
                "frames": float(len(self._e2e)),
                "dropped": float(self._drop_now() - self._drop_base),
                "depth_maps": float(self._maps),
                "e2e_p50_ms": e50,
                "e2e_p99_ms": e99,
                "e2e_max_ms": max(self._e2e),
                "boxes_p99_ms": b99,
            }
        )


# ─── Frame delivery ────────────────────────────────────────────────────────────


class FramePacer:
    """Replay in the loop thread at ``hz`` with latest-wins skipping.

    When processing falls behind by a whole period the intermediate frames are
    dropped (counted in :attr:`dropped`) so the loop always sees the frame a
    camera would have delivered *now*. ``hz=None`` replays as fast as possible.
    """

    def __init__(
        self,
        source: Iterable[FrameStamped],
        hz: float | None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = _sleep_precise,
        max_frames: int | None = None,
        duration_s: float | None = None,
    ) -> None:
        self.source = source
        self.hz = hz
        self.clock_ns = clock_ns
        self.sleep = sleep
        self.max_frames = max_frames
        self.duration_s = duration_s
        self.dropped = 0
        self.delivered = 0

    def __iter__(self) -> Iterator[tuple[FrameStamped, int]]:
        period_ns = int(round(1e9 / self.hz)) if self.hz else 0
        t_start = self.clock_ns()
        idx = 0
        for fs in self.source:
            if (
                self.duration_s is not None
                and (self.clock_ns() - t_start) * 1e-9 >= self.duration_s
            ):
                break
            if self.max_frames is not None and self.delivered >= self.max_frames:
                break
            if period_ns:
                due_at = t_start + idx * period_ns
                idx += 1
                now = self.clock_ns()
                if now < due_at:
                    self.sleep((due_at - now) * 1e-9)
                elif now - due_at >= period_ns:
                    # A newer frame is already available: this one is stale.
                    self.dropped += 1
                    continue
                t_in = due_at
            else:
                t_in = self.clock_ns()
            self.delivered += 1
            yield fs, t_in


def iter_paced(
    source: Iterable[FrameStamped],
    hz: float | None,
    max_frames: int | None = None,
    duration_s: float | None = None,
) -> FramePacer:
    return FramePacer(source, hz, max_frames=max_frames, duration_s=duration_s)


@dataclass(frozen=True)
class IngressFrame:
    frame: FrameStamped
    t_ingress_ns: int


def start_pump(
    source: Iterable[FrameStamped],
    slot: LatestFrameSlot[IngressFrame],
    hz: float | None,
    stop: threading.Event,
    max_frames: int | None = None,
) -> threading.Thread:
    """Capture thread: push ``source`` into ``slot`` at ``hz`` (drops counted by the slot)."""

    def run() -> None:
        period_s = 1.0 / hz if hz else 0.0
        t_next = time.perf_counter()
        n = 0
        try:
            for fs in source:
                if stop.is_set():
                    break
                if period_s > 0.0:
                    remaining = t_next - time.perf_counter()
                    if remaining > 0.0:
                        _sleep_precise(remaining)
                    t_next = max(t_next + period_s, time.perf_counter() - period_s)
                slot.put(IngressFrame(fs, time.monotonic_ns()))
                n += 1
                if max_frames is not None and n >= max_frames:
                    break
        finally:
            slot.close()

    th = threading.Thread(target=run, name="capture", daemon=True)
    th.start()
    return th


def iter_slot(
    slot: LatestFrameSlot[IngressFrame],
    stop: threading.Event | None = None,
    duration_s: float | None = None,
    timeout_s: float = 1.0,
) -> Iterator[tuple[FrameStamped, int]]:
    """Consume the newest frame from a capture thread until the slot closes."""
    t_start = time.monotonic()
    while not slot.closed or slot.peek() is not None:
        if stop is not None and stop.is_set():
            return
        if duration_s is not None and time.monotonic() - t_start >= duration_s:
            if stop is not None:
                stop.set()
            return
        item = slot.get(timeout=timeout_s)
        if item is None:
            if slot.closed:
                return
            continue
        yield item.frame, item.t_ingress_ns


class LoopedSource:
    """Repeat a finite source ``repeats`` times with increasing ``frame_id``/timestamps."""

    def __init__(self, source: Iterable[FrameStamped], repeats: int, period_ns: int) -> None:
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        self.source = source
        self.repeats = repeats
        self.period_ns = period_ns

    def __iter__(self) -> Iterator[FrameStamped]:
        id_off = 0
        t_off = 0
        prev_last_t: int | None = None
        for _ in range(self.repeats):
            first = True
            last_id = -1
            last_t = 0
            for fs in self.source:
                if first:
                    first = False
                    if prev_last_t is not None:
                        t_off = prev_last_t + self.period_ns - fs.t_capture_ns
                last_t = fs.t_capture_ns + t_off
                last_id = fs.frame_id + id_off
                yield FrameStamped(last_id, last_t, fs.img)
            id_off = last_id + 1
            prev_last_t = last_t


class LoopedEgoMotion:
    """Ego-motion for a :class:`LoopedSource`: folds looped timestamps back onto the original span."""

    def __init__(self, inner: EgoMotionProvider, t_first_ns: int, t_last_ns: int, period_ns: int):
        if t_last_ns < t_first_ns or period_ns <= 0:
            raise ValueError("need t_last >= t_first and period > 0")
        self.inner = inner
        self.t_first_ns = t_first_ns
        self.span_ns = t_last_ns - t_first_ns + period_ns

    def _fold(self, t_ns: int) -> int:
        return self.t_first_ns + (t_ns - self.t_first_ns) % self.span_ns

    def delta(self, t0_ns: int, t1_ns: int) -> EgoDelta:
        t0 = self._fold(t0_ns)
        return self.inner.delta(t0, t0 + (t1_ns - t0_ns))


class PreloadedSource:
    """Frames decoded once and served from memory (a camera does not pay image decoding)."""

    def __init__(self, source: Iterable[FrameStamped], max_frames: int | None = None) -> None:
        self.frames: list[FrameStamped] = []
        for fs in source:
            if max_frames is not None and len(self.frames) >= max_frames:
                break
            self.frames.append(fs)

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[FrameStamped]:
        return iter(self.frames)
