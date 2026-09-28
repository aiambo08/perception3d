"""Telemetry that never blocks the loop (R9).

The loop thread calls :meth:`AsyncSink.log` with a :class:`FrameResult`; that
builds a small :class:`TelemetryRecord` (boxes, track states, alerts, per-stage
latencies, the map age, and — every ``depth_every`` maps — a subsampled depth
map) and appends it to a bounded deque. A worker thread hands records to a
:class:`TelemetryBackend`: :class:`RerunBackend` (external viewer, lazy import
of ``rerun-sdk``), :class:`JsonlBackend` (file) or :class:`RecordingBackend`
(tests). When the backend is slower than the loop — or the viewer is gone —
the oldest records are dropped (:attr:`AsyncSink.dropped`); the cost seen by
the loop is only the record build, measured as the ``telemetry`` stage.

Timelines: ``capture`` (``t_capture_ns``, scene clock), ``frame`` (sequence).
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from percepcion3d.runtime.pipeline import FrameResult
from percepcion3d.safety.gates import AlertLevel

# ─── Records ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrackSample:
    track_id: int
    cls: str
    x_m: float
    z_m: float
    vx_mps: float
    vz_mps: float
    sigma_z_m: float
    level: str


@dataclass(frozen=True)
class AlertSample:
    track_id: int
    level: str
    ttc_low_s: float
    d_cpa_m: float
    reason: str


@dataclass(frozen=True)
class TelemetryRecord:
    frame_id: int
    t_capture_ns: int
    t_ingress_ns: int
    level: str
    timings_ms: dict[str, float]
    depth_frame_id: int | None
    depth_lag_frames: int
    depth_age_ms: float
    boxes: NDArray[np.float32]
    scores: NDArray[np.float32]
    classes: tuple[str, ...]
    tracks: tuple[TrackSample, ...]
    alerts: tuple[AlertSample, ...]
    depth: NDArray[np.float32] | None = None
    """Subsampled inverse-depth map of ``depth_frame_id`` (only every ``depth_every`` maps)."""
    depth_stride: int = 1

    def to_json_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "frame_id": self.frame_id,
            "t_capture_ns": self.t_capture_ns,
            "t_ingress_ns": self.t_ingress_ns,
            "level": self.level,
            "timings_ms": self.timings_ms,
            "depth_frame_id": self.depth_frame_id,
            "depth_lag_frames": self.depth_lag_frames,
            "depth_age_ms": self.depth_age_ms,
            "boxes": self.boxes.tolist(),
            "scores": self.scores.tolist(),
            "classes": list(self.classes),
            "tracks": [asdict(t) for t in self.tracks],
            "alerts": [asdict(a) for a in self.alerts],
            "depth_shape": None if self.depth is None else list(self.depth.shape),
        }
        return d


def build_record(res: FrameResult, depth_stride: int, with_depth: bool) -> TelemetryRecord:
    """Cheap snapshot of a result (called in the loop thread)."""
    level_by_track = {a.track_id: a.level.name for a in res.alerts}
    tracks = tuple(
        TrackSample(
            tr.track_id,
            tr.cls,
            float(tr.position_xz[0]),
            float(tr.position_xz[1]),
            float(tr.velocity_rel_xz[0]),
            float(tr.velocity_rel_xz[1]),
            float(np.sqrt(max(tr.cov[1, 1], 0.0))),
            level_by_track.get(tr.track_id, AlertLevel.NONE.name),
        )
        for tr in res.tracks
    )
    alerts = tuple(
        AlertSample(a.track_id, a.level.name, a.ttc_low_s, a.d_cpa_m, a.reason)
        for a in res.alerts
        if a.level != AlertLevel.NONE
    )
    depth: NDArray[np.float32] | None = None
    if with_depth and res.depth_new is not None:
        s = max(1, depth_stride)
        depth = np.ascontiguousarray(res.depth_new.inverse_depth()[::s, ::s], dtype=np.float32)
    return TelemetryRecord(
        frame_id=res.frame_id,
        t_capture_ns=res.t_capture_ns,
        t_ingress_ns=res.t_ingress_ns,
        level=res.level.name,
        timings_ms=dict(res.timings_ms),
        depth_frame_id=res.depth_frame_id,
        depth_lag_frames=res.depth_lag_frames,
        depth_age_ms=res.depth_age_ms,
        boxes=np.asarray(res.boxes.boxes, dtype=np.float32),
        scores=np.asarray(res.boxes.scores, dtype=np.float32),
        classes=tuple(res.boxes.classes),
        tracks=tracks,
        alerts=alerts,
        depth=depth,
        depth_stride=depth_stride,
    )


# ─── Backends ─────────────────────────────────────────────────────────────────


class TelemetryBackend(Protocol):
    def emit(self, rec: TelemetryRecord) -> None: ...

    def close(self) -> None: ...


class NullBackend:
    def emit(self, rec: TelemetryRecord) -> None:
        return None

    def close(self) -> None:
        return None


class RecordingBackend:
    """Keeps every record (tests); optional per-record delay to emulate a slow viewer."""

    def __init__(self, delay_s: float = 0.0) -> None:
        self.records: list[TelemetryRecord] = []
        self.delay_s = delay_s
        self.closed = False

    def emit(self, rec: TelemetryRecord) -> None:
        if self.delay_s > 0.0:
            time.sleep(self.delay_s)
        self.records.append(rec)

    def close(self) -> None:
        self.closed = True


class JsonlBackend:
    """One JSON object per record (no depth maps)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")

    def emit(self, rec: TelemetryRecord) -> None:
        self._fh.write(json.dumps(rec.to_json_dict()) + "\n")

    def close(self) -> None:
        self._fh.close()


_LEVEL_COLORS: dict[str, tuple[int, int, int]] = {
    "NONE": (80, 200, 120),
    "CAUTION": (255, 200, 0),
    "WARNING": (255, 120, 0),
    "CRITICAL": (255, 40, 40),
}


class RerunBackend:
    """Logs to a Rerun viewer (``spawn``/``connect``) or an ``.rrd`` file.

    Imports ``rerun`` lazily (``runtime`` extra). Entities: ``camera/boxes``,
    ``camera/depth``, ``world/tracks``, ``world/velocity``, ``latency/<stage>``,
    ``depth/age_ms``, ``depth/lag_frames``, ``alerts``.
    """

    def __init__(
        self,
        app_id: str = "percepcion3d",
        spawn: bool = False,
        connect: str | None = None,
        save: Path | str | None = None,
        camera_hw: tuple[int, int] | None = None,
    ) -> None:
        import rerun as rr

        self._rr = rr
        rr.init(app_id, spawn=spawn)
        if connect is not None:
            self._connect(connect)
        if save is not None:
            rr.save(str(save))
        self._new_time_api = hasattr(rr, "set_time")
        self._scalar = rr.Scalars if hasattr(rr, "Scalars") else rr.Scalar
        rr.log("world", rr.ViewCoordinates.RDF, static=True)
        if camera_hw is not None:
            rr.log(
                "camera",
                rr.Pinhole(
                    resolution=[camera_hw[1], camera_hw[0]],
                    focal_length=float(max(camera_hw)),
                ),
                static=True,
            )

    def _connect(self, addr: str) -> None:
        rr = self._rr
        if hasattr(rr, "connect_grpc"):
            rr.connect_grpc(addr if "://" in addr else f"rerun+http://{addr}/proxy")
        else:
            rr.connect(addr)

    def _set_times(self, rec: TelemetryRecord) -> None:
        rr = self._rr
        if self._new_time_api:
            rr.set_time("capture", duration=rec.t_capture_ns * 1e-9)
            rr.set_time("frame", sequence=rec.frame_id)
        else:
            rr.set_time_nanos("capture", rec.t_capture_ns)
            rr.set_time_sequence("frame", rec.frame_id)

    def emit(self, rec: TelemetryRecord) -> None:
        rr = self._rr
        self._set_times(rec)
        if rec.boxes.shape[0]:
            rr.log(
                "camera/boxes",
                rr.Boxes2D(
                    array=rec.boxes,
                    array_format=rr.Box2DFormat.XYXY,
                    labels=list(rec.classes),
                ),
            )
        else:
            rr.log("camera/boxes", rr.Clear(recursive=False))
        if rec.tracks:
            pos = np.array([[t.x_m, 0.0, t.z_m] for t in rec.tracks], dtype=np.float32)
            vel = np.array([[t.vx_mps, 0.0, t.vz_mps] for t in rec.tracks], dtype=np.float32)
            colors = [_LEVEL_COLORS.get(t.level, _LEVEL_COLORS["NONE"]) for t in rec.tracks]
            labels = [f"{t.track_id}:{t.cls} {t.z_m:.1f}m {t.vz_mps:+.1f}m/s" for t in rec.tracks]
            rr.log("world/tracks", rr.Points3D(pos, colors=colors, labels=labels, radii=0.3))
            rr.log("world/velocity", rr.Arrows3D(origins=pos, vectors=vel, colors=colors))
        else:
            rr.log("world/tracks", rr.Clear(recursive=False))
            rr.log("world/velocity", rr.Clear(recursive=False))
        for name, ms in rec.timings_ms.items():
            rr.log(f"latency/{name}", self._scalar(ms))
        if rec.depth_frame_id is not None:
            rr.log("depth/age_ms", self._scalar(rec.depth_age_ms))
            rr.log("depth/lag_frames", self._scalar(float(rec.depth_lag_frames)))
        rr.log("alerts/level", self._scalar(float(AlertLevel[rec.level].value)))
        for a in rec.alerts:
            rr.log(
                "alerts/log",
                rr.TextLog(
                    f"frame {rec.frame_id} track {a.track_id} {a.level} "
                    f"ttc={a.ttc_low_s:.2f}s d_cpa={a.d_cpa_m:.2f}m ({a.reason})",
                    level=a.level,
                ),
            )
        if rec.depth is not None:
            rr.log("camera/depth", rr.Image(_to_u8(rec.depth)))

    def close(self) -> None:
        self._rr.disconnect()


def _to_u8(inv: NDArray[np.float32]) -> NDArray[np.uint8]:
    finite = inv[np.isfinite(inv)]
    if finite.size == 0:
        return np.zeros(inv.shape, dtype=np.uint8)
    lo, hi = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
    if hi <= lo:
        return np.zeros(inv.shape, dtype=np.uint8)
    out = np.clip((inv - lo) / (hi - lo), 0.0, 1.0)
    return (np.nan_to_num(out) * 255.0).astype(np.uint8)


# ─── Async sink ───────────────────────────────────────────────────────────────


class AsyncSink:
    """Bounded queue + worker thread in front of a :class:`TelemetryBackend`.

    ``log`` never blocks and never raises; when the queue is full the oldest record
    is dropped. ``depth_every`` controls how many fresh maps pass between two
    logged (``depth_stride``-subsampled) maps.
    """

    def __init__(
        self,
        backend: TelemetryBackend,
        maxlen: int = 64,
        depth_every: int = 10,
        depth_stride: int = 4,
        build: Callable[[FrameResult, int, bool], TelemetryRecord] = build_record,
        start: bool = True,
    ) -> None:
        if maxlen < 1 or depth_every < 1 or depth_stride < 1:
            raise ValueError("maxlen, depth_every and depth_stride must be >= 1")
        self.backend = backend
        self.maxlen = maxlen
        self.depth_every = depth_every
        self.depth_stride = depth_stride
        self._build = build
        self._q: deque[TelemetryRecord] = deque()
        self._cond = threading.Condition()
        self._closed = False
        self.dropped = 0
        self.logged = 0
        self.emitted = 0
        self.errors = 0
        self.last_error: str | None = None
        self._maps_seen = 0
        self._thread = threading.Thread(target=self._run, name="telemetry", daemon=True)
        if start:
            self._thread.start()

    # loop thread ----------------------------------------------------------------

    def log(self, result: FrameResult) -> None:
        with_depth = False
        if result.depth_new is not None:
            with_depth = self._maps_seen % self.depth_every == 0
            self._maps_seen += 1
        rec = self._build(result, self.depth_stride, with_depth)
        with self._cond:
            if self._closed:
                return
            if len(self._q) >= self.maxlen:
                self._q.popleft()
                self.dropped += 1
            self._q.append(rec)
            self.logged += 1
            self._cond.notify()

    def close(self, timeout_s: float = 2.0) -> None:
        """Stop accepting records, drain what is queued (bounded by ``timeout_s``), close."""
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._cond.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout_s)
        self.backend.close()

    @property
    def pending(self) -> int:
        with self._cond:
            return len(self._q)

    # worker thread --------------------------------------------------------------

    def _run(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._q or self._closed)
                if not self._q:
                    return  # closed and drained
                rec = self._q.popleft()
            try:
                self.backend.emit(rec)
                self.emitted += 1
            except Exception as exc:  # a dead viewer must not kill the loop
                self.errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"


@dataclass
class SinkReport:
    logged: int = 0
    emitted: int = 0
    dropped: int = 0
    errors: int = 0
    last_error: str | None = None
    backend: str = "none"

    @staticmethod
    def from_sink(sink: AsyncSink | None) -> SinkReport:
        if sink is None:
            return SinkReport()
        return SinkReport(
            sink.logged,
            sink.emitted,
            sink.dropped,
            sink.errors,
            sink.last_error,
            type(sink.backend).__name__,
        )


__all__ = [
    "AlertSample",
    "AsyncSink",
    "JsonlBackend",
    "NullBackend",
    "RecordingBackend",
    "RerunBackend",
    "SinkReport",
    "TelemetryBackend",
    "TelemetryRecord",
    "TrackSample",
    "build_record",
]
