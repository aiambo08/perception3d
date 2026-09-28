"""F7 — asynchronous telemetry: bounded queue, drop policy, dead viewer, lazy Rerun."""

from __future__ import annotations

import json
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from percepcion3d.depth.depth_trt import DepthMap
from percepcion3d.depth.preprocess import DepthResize
from percepcion3d.runtime.pipeline import Boxes2D, FrameResult
from percepcion3d.safety.gates import AlertLevel
from percepcion3d.safety.ttc import Alert
from percepcion3d.telemetry.rerun_sink import (
    AsyncSink,
    JsonlBackend,
    NullBackend,
    RecordingBackend,
    RerunBackend,
    SinkReport,
    TelemetryRecord,
    build_record,
)
from percepcion3d.tracking.tracker3d import MotionState, Track3D

MS = 1_000_000


def _track(tid: int, z: float, vz: float) -> Track3D:
    return Track3D(
        track_id=tid,
        cls="car",
        t_ns=0,
        box=np.array([100.0, 100.0, 200.0, 180.0]),
        det_index=0,
        position_xz=np.array([0.5, z]),
        velocity_rel_xz=np.array([0.0, vz]),
        velocity_abs_xz=np.array([0.0, vz]),
        cov=np.diag([0.04, 0.09, 0.1, 0.1]),
        cov_vel_rel=0.1 * np.eye(2),
        age_s=0.5,
        n_updates=5,
        time_since_update_s=0.0,
        motion=MotionState.UNKNOWN,
        last_nis=0.0,
    )


def _result(k: int, with_map: bool = False, level: AlertLevel = AlertLevel.NONE) -> FrameResult:
    dm = None
    if with_map:
        rs = DepthResize(src_h=8, src_w=8, dst_h=8, dst_w=8)
        dm = DepthMap(k, k * 33 * MS, np.full((8, 8), 0.5, np.float32), "relative_disparity", rs)
    alerts = []
    if level is not AlertLevel.NONE:
        alerts.append(Alert(1, level, ttc_low_s=1.2, d_cpa_m=0.3, reason="path"))
    return FrameResult(
        frame_id=k,
        t_capture_ns=k * 33 * MS,
        t_ingress_ns=10_000 * MS + k * 33 * MS,
        boxes=Boxes2D(np.array([[100.0, 100.0, 200.0, 180.0]]), np.array([0.9]), ["car"]),
        measurements=[],
        tracks=[_track(1, 20.0, -5.0)],
        alerts=alerts,
        level=level,
        depth_frame_id=k - 1 if k else None,
        depth_lag_frames=1 if k else 0,
        depth_age_ms=16.5 if k else float("nan"),
        depth_new=dm,
        net_boxes_matched=0,
        timings_ms={"boxes": 4.0, "fusion": 1.5, "e2e_alert": 7.0},
    )


# ─── Records ──────────────────────────────────────────────────────────────────


def test_build_record_snapshots_tracks_alerts_and_subsampled_depth() -> None:
    rec = build_record(_result(3, with_map=True, level=AlertLevel.WARNING), 2, True)
    assert rec.frame_id == 3 and rec.level == "WARNING" and rec.depth_lag_frames == 1
    assert rec.boxes.dtype == np.float32 and rec.boxes.shape == (1, 4) and rec.classes == ("car",)
    (t,) = rec.tracks
    assert (t.track_id, t.z_m, t.vz_mps, t.level) == (1, 20.0, -5.0, "WARNING")
    assert t.sigma_z_m == pytest.approx(0.3)
    (a,) = rec.alerts
    assert (a.track_id, a.level, a.ttc_low_s, a.reason) == (1, "WARNING", 1.2, "path")
    assert rec.depth is not None and rec.depth.shape == (4, 4) and rec.depth_stride == 2
    no_depth = build_record(_result(3, with_map=True), 2, False)
    assert no_depth.depth is None and no_depth.alerts == ()
    d = rec.to_json_dict()
    assert d["depth_shape"] == [4, 4] and json.dumps(d)  # serialisable


# ─── AsyncSink ────────────────────────────────────────────────────────────────


def test_sink_delivers_every_record_in_order_and_closes_backend() -> None:
    be = RecordingBackend()
    sink = AsyncSink(be, maxlen=16, depth_every=2, depth_stride=2)
    for k in range(10):
        sink.log(_result(k, with_map=k % 3 == 0))  # maps at k = 0, 3, 6, 9
    sink.close()
    assert be.closed and sink.pending == 0
    assert [r.frame_id for r in be.records] == list(range(10))
    assert sink.logged == sink.emitted == 10 and sink.dropped == 0 and sink.errors == 0
    # depth_every=2 → every other fresh map is attached: maps 0 and 6.
    assert [r.frame_id for r in be.records if r.depth is not None] == [0, 6]
    rep = SinkReport.from_sink(sink)
    assert rep.backend == "RecordingBackend" and rep.emitted == 10
    assert SinkReport.from_sink(None).backend == "none"


def test_sink_drops_oldest_when_backend_is_slow_and_never_blocks_the_loop() -> None:
    be = RecordingBackend(delay_s=0.02)
    sink = AsyncSink(be, maxlen=4)
    t0 = time.perf_counter()
    for k in range(40):
        sink.log(_result(k))
    log_cost = time.perf_counter() - t0
    assert log_cost < 0.2  # 40 records; a blocking sink would take 40 · 20 ms = 0.8 s
    assert sink.pending <= 4 and sink.dropped >= 40 - 4 - 2
    sink.close(timeout_s=1.0)
    ids = [r.frame_id for r in be.records]
    assert ids == sorted(ids) and ids[-1] == 39  # latest survives, oldest dropped
    assert sink.logged == 40 and sink.emitted + sink.dropped == 40


def test_sink_survives_a_dead_viewer() -> None:
    class Dead:
        def __init__(self) -> None:
            self.n = 0

        def emit(self, rec: TelemetryRecord) -> None:
            self.n += 1
            raise ConnectionError("viewer gone")

        def close(self) -> None:
            return None

    be = Dead()
    sink = AsyncSink(be, maxlen=8)
    for k in range(5):
        sink.log(_result(k))
    sink.close()
    assert be.n == 5 and sink.errors == 5 and sink.emitted == 0
    assert sink.last_error is not None and "viewer gone" in sink.last_error


def test_sink_close_is_idempotent_and_rejects_late_logs() -> None:
    be = RecordingBackend()
    sink = AsyncSink(be)
    sink.log(_result(0))
    sink.close()
    sink.close()
    sink.log(_result(1))  # ignored after close
    assert [r.frame_id for r in be.records] == [0] and sink.logged == 1


def test_sink_close_bounded_by_timeout_when_backend_hangs() -> None:
    release = threading.Event()

    class Hanging:
        def emit(self, rec: TelemetryRecord) -> None:
            release.wait(5.0)

        def close(self) -> None:
            return None

    sink = AsyncSink(Hanging(), maxlen=2)
    sink.log(_result(0))
    t0 = time.perf_counter()
    sink.close(timeout_s=0.1)
    assert time.perf_counter() - t0 < 1.0
    release.set()


def test_sink_validates_parameters_and_null_backend() -> None:
    with pytest.raises(ValueError):
        AsyncSink(NullBackend(), maxlen=0)
    with pytest.raises(ValueError):
        AsyncSink(NullBackend(), depth_every=0)
    sink = AsyncSink(NullBackend())
    sink.log(_result(0))
    sink.close()
    assert sink.emitted == 1


def test_jsonl_backend_writes_one_object_per_record(tmp_path: Path) -> None:
    path = tmp_path / "out" / "tel.jsonl"
    sink = AsyncSink(JsonlBackend(path), depth_every=1)
    sink.log(_result(0, with_map=True, level=AlertLevel.CRITICAL))
    sink.log(_result(1))
    sink.close()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    d0, d1 = json.loads(lines[0]), json.loads(lines[1])
    assert d0["frame_id"] == 0 and d0["alerts"][0]["level"] == "CRITICAL"
    assert d0["depth_shape"] == [2, 2] and d1["depth_shape"] is None
    assert d1["depth_lag_frames"] == 1 and d1["timings_ms"]["e2e_alert"] == 7.0


# ─── RerunBackend with a fake ``rerun`` module ────────────────────────────────


class _FakeRerun(types.ModuleType):
    """Records every ``rr.log`` call; archetypes are recorded by name only."""

    def __init__(self) -> None:
        super().__init__("rerun")
        self.calls: list[tuple[str, Any]] = []
        self.inits: list[tuple[str, bool]] = []
        self.saved: list[str] = []
        self.connected: list[str] = []
        self.times: list[tuple[str, float | int]] = []
        self.disconnected = False

    # module API ---------------------------------------------------------------
    def init(self, app_id: str, spawn: bool = False) -> None:
        self.inits.append((app_id, spawn))

    def save(self, path: str) -> None:
        self.saved.append(path)

    def connect_grpc(self, url: str) -> None:
        self.connected.append(url)

    def disconnect(self) -> None:
        self.disconnected = True

    def set_time(
        self, name: str, duration: float | None = None, sequence: int | None = None
    ) -> None:
        self.times.append((name, duration if duration is not None else int(sequence or 0)))

    def log(self, entity: str, payload: Any, static: bool = False) -> None:
        self.calls.append((entity, payload))

    # archetypes ----------------------------------------------------------------
    class ViewCoordinates:
        RDF = "RDF"

    class Box2DFormat:
        XYXY = "XYXY"

    @staticmethod
    def Pinhole(**kw: Any) -> tuple[str, dict[str, Any]]:
        return ("Pinhole", kw)

    @staticmethod
    def Boxes2D(**kw: Any) -> tuple[str, dict[str, Any]]:
        return ("Boxes2D", kw)

    @staticmethod
    def Clear(recursive: bool) -> tuple[str, bool]:
        return ("Clear", recursive)

    @staticmethod
    def Points3D(pos: Any, **kw: Any) -> tuple[str, Any]:
        return ("Points3D", pos)

    @staticmethod
    def Arrows3D(**kw: Any) -> tuple[str, dict[str, Any]]:
        return ("Arrows3D", kw)

    @staticmethod
    def Scalars(v: float) -> tuple[str, float]:
        return ("Scalars", v)

    @staticmethod
    def TextLog(text: str, level: str) -> tuple[str, str]:
        return ("TextLog", text)

    @staticmethod
    def Image(img: Any) -> tuple[str, Any]:
        return ("Image", img)


def test_rerun_import_is_lazy() -> None:
    assert "rerun" not in sys.modules or sys.modules["rerun"].__class__ is _FakeRerun


def test_rerun_backend_logs_expected_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRerun()
    monkeypatch.setitem(sys.modules, "rerun", fake)
    be = RerunBackend(spawn=False, connect="127.0.0.1:9876", save="x.rrd", camera_hw=(375, 1242))
    assert fake.inits == [("percepcion3d", False)] and fake.saved == ["x.rrd"]
    assert fake.connected == ["rerun+http://127.0.0.1:9876/proxy"]
    assert [c[0] for c in fake.calls] == ["world", "camera"]
    fake.calls.clear()
    rec = build_record(_result(4, with_map=True, level=AlertLevel.CRITICAL), 2, True)
    be.emit(rec)
    ents = [c[0] for c in fake.calls]
    assert ents[:3] == ["camera/boxes", "world/tracks", "world/velocity"]
    assert {"latency/boxes", "latency/fusion", "latency/e2e_alert"} <= set(ents)
    assert {
        "depth/age_ms",
        "depth/lag_frames",
        "alerts/level",
        "alerts/log",
        "camera/depth",
    } <= set(ents)
    assert fake.times == [("capture", pytest.approx(4 * 0.033)), ("frame", 4)]
    img = dict(fake.calls)["camera/depth"]
    assert img[0] == "Image" and img[1].dtype == np.uint8 and img[1].shape == (4, 4)
    level = dict(fake.calls)["alerts/level"]
    assert level == ("Scalars", float(AlertLevel.CRITICAL.value))
    # An empty frame clears the entities instead of leaving stale boxes on screen.
    fake.calls.clear()
    empty = _result(5)
    empty.boxes = Boxes2D.empty()
    empty.tracks = []
    be.emit(build_record(empty, 1, False))
    assert dict(fake.calls)["camera/boxes"] == ("Clear", False)
    assert dict(fake.calls)["world/tracks"] == ("Clear", False)
    be.close()
    assert fake.disconnected
