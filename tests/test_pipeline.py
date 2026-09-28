"""F7 — dual-rate loop with fake stages (CPU only): frame_id, cadence, map age, delivery."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from percepcion3d.camera.calibration import CameraIntrinsics, ExtrinsicMountConfig
from percepcion3d.camera.geometry import PinholeGeometry
from percepcion3d.depth.depth_trt import DepthMap
from percepcion3d.depth.fusion import MetricFusionStage, load_fusion_config, load_solver_configs
from percepcion3d.runtime.buffer import FrameStamped, LatestFrameSlot
from percepcion3d.runtime.pipeline import (
    Boxes2D,
    BoxHistory,
    CallableBoxStage,
    DepthCadence,
    DepthCadenceConfig,
    FramePacer,
    FrameResult,
    IngressFrame,
    LoopedEgoMotion,
    LoopedSource,
    Pipeline,
    PipelineConfig,
    PreloadedSource,
    iter_slot,
    start_pump,
)
from percepcion3d.safety.gates import AlertLevel, load_safety_config
from percepcion3d.sim.synthetic import SyntheticNoise, SyntheticObject, SyntheticScene
from percepcion3d.tracking.ego_motion import ConstantEgoMotion
from percepcion3d.tracking.kalman_filter import EgoDelta
from percepcion3d.tracking.tracker3d import Tracker3D, load_tracker_config

ROOT = Path(__file__).resolve().parents[1]
INTR = CameraIntrinsics(fx=721.5377, fy=721.5377, cx=609.5593, cy=172.854, width=1242, height=375)
EXTR = ExtrinsicMountConfig(camera_height_m=1.65, pitch_rad=float(np.deg2rad(2.5)))
GEO = PinholeGeometry(INTR, EXTR)
NET_HW = (140, 462)
MS = 1_000_000


# ─── Fakes ────────────────────────────────────────────────────────────────────


class FakeClock:
    """Manual monotonic clock; ``advance`` is the only way time moves."""

    def __init__(self) -> None:
        self.t_ns = 1_000_000_000

    def __call__(self) -> int:
        return self.t_ns

    def advance(self, ms: float) -> None:
        self.t_ns += int(round(ms * MS))


class FakeDepthHandle:
    def __init__(self, stage: FakeDepthStage, fs: FrameStamped, ready_at_ns: int) -> None:
        self.stage = stage
        self.fs = fs
        self.ready_at_ns = ready_at_ns
        self.waited = False

    @property
    def frame_id(self) -> int:
        return self.fs.frame_id

    def ready(self) -> bool:
        return self.stage.clock() >= self.ready_at_ns

    def wait(self) -> DepthMap:
        if self.stage.clock() < self.ready_at_ns:
            self.stage.clock.t_ns = self.ready_at_ns  # a real wait would block until then
        self.waited = True
        return self.stage.render(self.fs)


class FakeDepthStage:
    """Renders the synthetic scene's map for ``frame_id`` after ``latency_ms`` of fake time."""

    def __init__(self, scene: SyntheticScene, clock: FakeClock, latency_ms: float) -> None:
        self.scene = scene
        self.clock = clock
        self.latency_ms = latency_ms
        self.submitted: list[int] = []
        self.handles: list[FakeDepthHandle] = []

    def infer_stamped(self, fs: FrameStamped) -> FakeDepthHandle:
        self.submitted.append(fs.frame_id)
        h = FakeDepthHandle(self, fs, self.clock() + int(self.latency_ms * MS))
        self.handles.append(h)
        return h

    def render(self, fs: FrameStamped) -> DepthMap:
        f = self.scene.frame(fs.frame_id)
        inv, rs = self.scene.dense_inv_depth_map(f, NET_HW, pixel_noise_rel=0.01)
        return DepthMap(
            fs.frame_id, fs.t_capture_ns, inv.astype(np.float32), "relative_disparity", rs
        )


class RecordingSink:
    def __init__(self) -> None:
        self.results: list[FrameResult] = []
        self.closed = False

    def log(self, result: FrameResult) -> None:
        self.results.append(result)

    def close(self) -> None:
        self.closed = True


def _scene(objs: list[SyntheticObject] | None = None, fps: float = 30.0) -> SyntheticScene:
    if objs is None:
        objs = [
            SyntheticObject(1, "car", x0_m=0.5, z0_m=12.0, vz_mps=1.0, height_m=1.52),
            SyntheticObject(2, "car", x0_m=-3.5, z0_m=30.0, vz_mps=-2.0, height_m=1.45),
        ]
    scene = SyntheticScene(GEO, objs, noise=SyntheticNoise(box_px=0.5, inv_depth_rel=0.01), fps=fps)
    scene.disparity_scale, scene.disparity_shift = 3.0, 0.2
    return scene


def _scene_boxes(scene: SyntheticScene) -> CallableBoxStage:
    def fn(fs: FrameStamped) -> Boxes2D:
        f = scene.frame(fs.frame_id)
        return Boxes2D(
            np.asarray(f.boxes, dtype=np.float64), np.ones(len(f.classes)), list(f.classes)
        )

    return CallableBoxStage(fn)


def _frames(scene: SyntheticScene, n: int) -> list[FrameStamped]:
    img = np.zeros((INTR.height, INTR.width, 3), dtype=np.uint8)
    return [FrameStamped(k, scene.frame(k).t_ns, img) for k in range(n)]


def _pipeline(
    scene: SyntheticScene,
    clock: FakeClock,
    depth: FakeDepthStage | None,
    sink: RecordingSink | None = None,
    cfg: PipelineConfig | None = None,
    ego_mps: float = 0.0,
) -> Pipeline:
    fusion_cfg = load_fusion_config(ROOT / "configs" / "fusion.yaml")
    road, aff, pit = load_solver_configs(ROOT / "configs" / "fusion.yaml")
    stage = MetricFusionStage(INTR, GEO, fusion_cfg, road, aff, pit)
    trk = Tracker3D(
        load_tracker_config(ROOT / "configs" / "tracking.yaml"), ConstantEgoMotion(ego_mps)
    )
    safety = load_safety_config(ROOT / "configs" / "safety.yaml")
    return Pipeline(_scene_boxes(scene), stage, trk, safety, depth, sink, cfg, clock_ns=clock)


def _feed(
    pipe: Pipeline, frames: list[FrameStamped], clock: FakeClock, loop_ms: float
) -> list[FrameResult]:
    out = []
    for fs in frames:
        out.append(pipe.process(fs, clock()))
        clock.advance(loop_ms)
    return out


# ─── DepthCadence ─────────────────────────────────────────────────────────────


def test_cadence_due_enqueue_skip_and_adaptation() -> None:
    c = DepthCadence(
        DepthCadenceConfig(n_min=1, n_max=8, n_init=2, margin=1.1), frame_period_ms=16.667
    )
    assert c.due(0)
    c.on_enqueue(0)
    assert not c.due(1) and c.due(2)
    c.on_skip()  # previous map still in flight → n grows
    assert c.n == 3
    for _ in range(10):
        c.on_skip()
    assert c.n == 8
    # Turnaround of 30 ms at 60 Hz → ceil(1.1·30/16.667) = 2; n shrinks one per completion.
    for expected in (7, 6, 5, 4, 3, 2, 2):
        c.on_complete(30.0)
        assert c.n == expected
    c.on_complete(200.0)  # slow map: EMA → 64 ms → ceil(1.1·64/16.667) = 5, jumps up at once
    assert c.n == 5 and c.turnaround_ms == pytest.approx(64.0)
    c.on_complete(5.0)  # EMA → 52.2 ms → target 4 < n → shrinks by one only
    assert c.n == 4


def test_cadence_rejects_bad_config() -> None:
    with pytest.raises(ValueError):
        DepthCadence(DepthCadenceConfig(n_min=0), 16.0)
    with pytest.raises(ValueError):
        DepthCadence(DepthCadenceConfig(n_min=2, n_init=1), 16.0)
    with pytest.raises(ValueError):
        DepthCadence(DepthCadenceConfig(), 0.0)


# ─── BoxHistory ───────────────────────────────────────────────────────────────


def _b(*rows: list[float]) -> Boxes2D:
    a = np.asarray(rows, dtype=np.float64).reshape(-1, 4)
    return Boxes2D(a, np.ones(a.shape[0]), ["car"] * a.shape[0])


def test_box_history_ring_and_matching() -> None:
    h = BoxHistory(capacity=3)
    for k in range(5):
        h.push(k, _b([100 + 10 * k, 100, 200 + 10 * k, 180]), k * 16 * MS)
    assert len(h) == 3 and h.get(1) is None and h.get(2) is not None
    got = h.get(2)
    assert got is not None and got[1] == 2 * 16 * MS
    # Current frame 4, map from frame 2: the box of frame 4 is replaced by its frame-2 twin.
    cur = _b([140, 100, 240, 180], [900, 50, 950, 90])
    out, n = h.net_boxes(cur, 2)
    assert out is not None and n == 1
    assert out[0].tolist() == [120, 100, 220, 180] and out[1].tolist() == [900, 50, 950, 90]
    assert h.net_boxes(cur, 1) == (None, 0)  # frame evicted
    assert h.net_boxes(Boxes2D.empty(), 2) == (None, 0)
    with pytest.raises(ValueError):
        BoxHistory(capacity=0)


# ─── Pipeline with fakes ─────────────────────────────────────────────────────


def test_frame_id_and_timestamps_propagate_end_to_end() -> None:
    scene = _scene()
    clock = FakeClock()
    depth = FakeDepthStage(scene, clock, latency_ms=20.0)
    sink = RecordingSink()
    pipe = _pipeline(scene, clock, depth, sink, PipelineConfig(frame_hz=30.0))
    frames = _frames(scene, 12)
    res = _feed(pipe, frames, clock, loop_ms=33.0)
    for fs, r in zip(frames, res, strict=True):
        assert r.frame_id == fs.frame_id and r.t_capture_ns == fs.t_capture_ns
        for m in r.measurements:
            assert m.frame_id == fs.frame_id and m.t_capture_ns == fs.t_capture_ns
        for tr in r.tracks:
            assert tr.t_ns == fs.t_capture_ns
        if r.depth_frame_id is not None:
            assert r.depth_frame_id <= r.frame_id
            assert r.depth_lag_frames == r.frame_id - r.depth_frame_id
        assert set(r.timings_ms) >= {
            "boxes",
            "depth_wait",
            "fusion",
            "tracker",
            "safety",
            "e2e_alert",
            "loop",
            "telemetry",
        }
    assert [r.frame_id for r in sink.results] == list(range(12))
    with pytest.raises(ValueError):
        pipe.process(frames[3], clock())


def test_depth_never_blocks_and_map_age_lag_are_measured() -> None:
    scene = _scene()
    clock = FakeClock()
    depth = FakeDepthStage(scene, clock, latency_ms=50.0)  # 1.5 frames at 30 Hz
    pipe = _pipeline(
        scene, clock, depth, cfg=PipelineConfig(frame_hz=30.0, cadence=DepthCadenceConfig(n_init=1))
    )
    frames = _frames(scene, 20)
    res = _feed(pipe, frames, clock, loop_ms=33.0)
    # The loop never waited for the GPU: every consumed handle was ready when polled.
    assert all(r.timings_ms["depth_wait"] == 0.0 for r in res)
    consumed = [r for r in res if r.depth_new is not None]
    assert consumed and all(
        r.depth_new is not None and r.depth_new.frame_id < r.frame_id for r in consumed
    )
    # Age = ingress(now) − ingress(map frame) ≈ lag · 33 ms with the fake clock.
    with_map = [r for r in res if r.depth_frame_id is not None]
    assert with_map
    for r in with_map:
        assert r.depth_lag_frames >= 1
        assert r.depth_age_ms == pytest.approx(r.depth_lag_frames * 33.0, abs=1e-6)
    assert res[0].depth_frame_id is None and np.isnan(res[0].depth_age_ms)
    # A slot came due while the previous map was in flight → skipped and n adapted upward.
    assert pipe.cadence.skipped >= 1 and pipe.cadence.n >= 2
    st = pipe.stats
    assert st.depth_maps == len(consumed) and st.depth_enqueued == len(depth.submitted)
    assert len(st.depth_age_ms) == len(with_map) and len(st.depth_turnaround_ms) == st.depth_maps


def test_cadence_adapts_to_depth_turnaround() -> None:
    scene = _scene()
    clock = FakeClock()
    depth = FakeDepthStage(scene, clock, latency_ms=70.0)  # ≈ 4.2 frames at 60 Hz
    cfg = PipelineConfig(
        frame_hz=60.0, cadence=DepthCadenceConfig(n_init=1, n_min=1, n_max=8, margin=1.0)
    )
    pipe = _pipeline(scene, clock, depth, cfg=cfg)
    _feed(pipe, _frames(scene, 60), clock, loop_ms=1000 / 60)
    # Turnaround ≈ 5 frames (ready at 70 ms, polled at the next 16.7 ms tick) → n settles at 5.
    assert pipe.cadence.n == 5
    gaps = np.diff(depth.submitted[-5:])
    assert gaps.min() >= 5
    assert pipe.stats.depth_maps >= 8


def test_stale_map_uses_boxes_of_the_map_frame_and_inflates_sigma() -> None:
    scene = _scene([SyntheticObject(1, "car", x0_m=0.0, z0_m=25.0, vz_mps=-12.0, height_m=1.52)])
    clock_f, clock_s = FakeClock(), FakeClock()
    fresh = FakeDepthStage(scene, clock_f, latency_ms=0.0)
    stale = FakeDepthStage(scene, clock_s, latency_ms=100.0)  # 3 frames old at 30 Hz
    cfg_fresh = PipelineConfig(frame_hz=30.0, cadence=DepthCadenceConfig(n_init=1, n_max=1))
    cfg_stale = PipelineConfig(
        frame_hz=30.0, cadence=DepthCadenceConfig(n_init=4, n_min=4, n_max=8)
    )
    p_fresh = _pipeline(scene, clock_f, fresh, cfg=cfg_fresh)
    p_stale = _pipeline(scene, clock_s, stale, cfg=cfg_stale)
    frames = _frames(scene, 24)
    r_fresh = _feed(p_fresh, frames, clock_f, loop_ms=33.0)
    r_stale = _feed(p_stale, frames, clock_s, loop_ms=33.0)
    assert all(r.depth_lag_frames == 0 for r in r_fresh[1:])
    lagged = [r for r in r_stale if r.depth_lag_frames >= 3 and r.measurements]
    assert lagged and all(r.net_boxes_matched == 1 for r in lagged)
    # With the map 3 frames old the net cue must be less trusted than with a fresh one.
    sig_fresh = np.median([r.measurements[0].sigma_net_m for r in r_fresh[8:] if r.measurements])
    sig_stale = np.median([r.measurements[0].sigma_net_m for r in lagged[2:]])
    assert sig_stale > sig_fresh * 1.5
    # ...and still centred on the truth (boxes of frame j read map j).
    for r in lagged[2:]:
        zt = scene.frame(r.frame_id).gt_z_front_m[0]
        assert r.measurements[0].z_cam_m == pytest.approx(zt, rel=0.12)


def test_alerts_are_deterministic_and_fire_for_head_on_target() -> None:
    scene = _scene([SyntheticObject(1, "car", x0_m=0.0, z0_m=40.0, vz_mps=-15.0, height_m=1.52)])

    def run() -> list[FrameResult]:
        clock = FakeClock()
        pipe = _pipeline(scene, clock, FakeDepthStage(scene, clock, 20.0), RecordingSink())
        return _feed(pipe, _frames(scene, 60), clock, loop_ms=33.0)

    a, b = run(), run()
    levels_a = [r.level for r in a]
    assert levels_a == [r.level for r in b]
    assert (
        levels_a[0] is AlertLevel.NONE
        and max(levels_a, key=lambda lv: lv.value) is AlertLevel.CRITICAL
    )
    first = next(i for i, lv in enumerate(levels_a) if lv is not AlertLevel.NONE)
    assert first > 0
    critical = [r for r in a if r.level is AlertLevel.CRITICAL]
    assert all(
        al.ttc_low_s < 3.0 for r in critical for al in r.alerts if al.level is AlertLevel.CRITICAL
    )


def test_run_without_depth_closes_sink_and_counts_drops() -> None:
    scene = _scene()
    clock = FakeClock()
    sink = RecordingSink()
    pipe = _pipeline(scene, clock, None, sink)
    frames = _frames(scene, 10)

    def gen() -> Iterator[tuple[FrameStamped, int]]:
        for fs in frames:
            yield fs, clock()
            clock.advance(16.0)

    st = pipe.run(gen(), dropped=lambda: 3)
    assert (
        sink.closed
        and st.frames == 10
        and st.dropped == 3
        and st.drop_frac == pytest.approx(3 / 13)
    )
    assert st.depth_maps == 0 and all(r.depth_frame_id is None for r in sink.results)
    assert st.elapsed_s == pytest.approx(0.16)
    d = st.to_dict()
    assert (
        d["e2e_alert_ms"]["n"] == 10
        and d["depth"]["age_ms"]["n"] == 0
        and d["input_hz"] == pytest.approx(13 / 0.16)
    )


def test_run_waits_for_in_flight_depth_on_exit() -> None:
    scene = _scene()
    clock = FakeClock()
    depth = FakeDepthStage(scene, clock, latency_ms=500.0)
    pipe = _pipeline(scene, clock, depth)
    pipe.run((fs, clock()) for fs in _frames(scene, 3))
    assert depth.handles and depth.handles[-1].waited
    assert pipe._pending is None


# ─── Frame delivery ────────────────────────────────────────────────────────────


def test_pacer_drops_stale_frames_when_the_loop_is_slow() -> None:
    clock = FakeClock()
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        clock.advance(s * 1e3)

    frames = _frames(_scene(), 12)
    pacer = FramePacer(frames, hz=60.0, clock_ns=clock, sleep=sleep)
    got = []
    for i, (fs, _t_in) in enumerate(pacer):
        got.append(fs.frame_id)
        clock.advance(40.0 if i < 3 else 5.0)  # 3 slow iterations (2.4 periods each)
    assert pacer.dropped > 0 and pacer.delivered == len(got)
    assert pacer.dropped + pacer.delivered == 12
    assert got == sorted(got) and got[0] == 0
    assert all(s > 0 for s in slept)


def test_pacer_max_frames_duration_and_unpaced() -> None:
    clock = FakeClock()
    frames = _frames(_scene(), 10)
    assert [fs.frame_id for fs, _ in FramePacer(frames, None, clock_ns=clock, max_frames=4)] == [
        0,
        1,
        2,
        3,
    ]

    def sleep(s: float) -> None:
        clock.advance(s * 1e3)

    p = FramePacer(frames, hz=100.0, clock_ns=clock, sleep=sleep, duration_s=0.045)
    ids = [fs.frame_id for fs, _ in p]
    assert 4 <= len(ids) <= 6 and p.dropped == 0


def test_capture_thread_slot_latest_wins_and_clean_shutdown() -> None:
    frames = _frames(_scene(), 40)
    slot: LatestFrameSlot[IngressFrame] = LatestFrameSlot()
    stop = threading.Event()
    th = start_pump(frames, slot, hz=2000.0, stop=stop)
    got = []
    for fs, t_in in iter_slot(slot, stop, timeout_s=0.2):
        got.append(fs.frame_id)
        assert t_in > 0
        time.sleep(0.005)  # slow consumer → producer overwrites the slot
    th.join(timeout=2.0)
    assert not th.is_alive() and slot.closed
    assert got == sorted(got) and got[-1] == 39
    assert slot.dropped + len(got) == 40 and slot.dropped > 0


def test_iter_slot_duration_sets_stop_for_the_pump() -> None:
    frames = _frames(_scene(), 5)
    slot: LatestFrameSlot[IngressFrame] = LatestFrameSlot()
    stop = threading.Event()
    th = start_pump(LoopedSource(frames, repeats=10_000, period_ns=MS), slot, hz=200.0, stop=stop)
    n = sum(1 for _ in iter_slot(slot, stop, duration_s=0.1, timeout_s=0.05))
    th.join(timeout=2.0)
    assert n > 0 and stop.is_set() and not th.is_alive()


def test_looped_source_ids_and_timestamps_keep_increasing() -> None:
    frames = _frames(_scene(fps=10.0), 3)  # t = 0, 100, 200 ms
    looped = list(LoopedSource(frames, repeats=3, period_ns=100 * MS))
    assert [f.frame_id for f in looped] == list(range(9))
    t = [f.t_capture_ns for f in looped]
    assert t == [k * 100 * MS for k in range(9)]
    assert looped[4].img is frames[1].img
    with pytest.raises(ValueError):
        LoopedSource(frames, repeats=0, period_ns=1)


def test_looped_ego_motion_folds_time_onto_the_first_pass() -> None:
    class Probe:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def delta(self, t0_ns: int, t1_ns: int) -> EgoDelta:
            self.calls.append((t0_ns, t1_ns))
            return ConstantEgoMotion(0.0).delta(t0_ns, t1_ns)

    probe = Probe()
    ego = LoopedEgoMotion(probe, t_first_ns=0, t_last_ns=200 * MS, period_ns=100 * MS)
    ego.delta(650 * MS, 700 * MS)  # third pass: 650 → 50 ms into the original span
    assert probe.calls == [(50 * MS, 100 * MS)]
    with pytest.raises(ValueError):
        LoopedEgoMotion(probe, 10, 0, 1)


def test_preloaded_source_decodes_once() -> None:
    n = 0

    def gen() -> Iterator[FrameStamped]:
        nonlocal n
        for fs in _frames(_scene(), 6):
            n += 1
            yield fs

    pre = PreloadedSource(gen(), max_frames=4)
    assert len(pre) == 4 and n == 5  # stops after the first frame past the limit
    assert [f.frame_id for f in pre] == [0, 1, 2, 3] == [f.frame_id for f in pre]


# ─── Enqueue order, per-enqueue split, time series ───────────────────────────


class _OrderedBoxes:
    def __init__(self, inner: CallableBoxStage, log: list[str]) -> None:
        self.inner = inner
        self.log = log

    def submit(self, fs: FrameStamped) -> _OrderedHandle:
        self.log.append(f"det.submit {fs.frame_id}")
        return _OrderedHandle(self.inner.submit(fs).wait(), self.log, fs.frame_id)


class _OrderedHandle:
    def __init__(self, b: Boxes2D, log: list[str], k: int) -> None:
        self.b = b
        self.log = log
        self.k = k

    def wait(self) -> Boxes2D:
        self.log.append(f"det.wait {self.k}")
        return self.b


class _OrderedDepth(FakeDepthStage):
    def __init__(self, scene: SyntheticScene, clock: FakeClock, log: list[str]) -> None:
        super().__init__(scene, clock, latency_ms=0.0)
        self.log = log

    def infer_stamped(self, fs: FrameStamped) -> FakeDepthHandle:
        self.log.append(f"depth {fs.frame_id}")
        return super().infer_stamped(fs)


@pytest.mark.parametrize("depth_first", [False, True])
def test_enqueue_order_detector_first_by_default(depth_first: bool) -> None:
    scene = _scene()
    clock = FakeClock()
    log: list[str] = []
    cfg = PipelineConfig(
        frame_hz=30.0,
        cadence=DepthCadenceConfig(n_init=2, n_min=2, n_max=2),
        depth_first=depth_first,
    )
    pipe = _pipeline(scene, clock, _OrderedDepth(scene, clock, log), cfg=cfg)
    pipe.boxes = _OrderedBoxes(_scene_boxes(scene), log)
    _feed(pipe, _frames(scene, 3), clock, loop_ms=33.0)
    if depth_first:
        assert log[:3] == ["depth 0", "det.submit 0", "det.wait 0"]
    else:
        assert log[:3] == ["det.submit 0", "depth 0", "det.wait 0"]
    assert log[3:5] == ["det.submit 1", "det.wait 1"]  # frame 1: no depth due (n = 2)
    assert "depth 2" in log


def test_stats_split_by_depth_enqueue() -> None:
    scene = _scene()
    clock = FakeClock()
    cfg = PipelineConfig(frame_hz=30.0, cadence=DepthCadenceConfig(n_init=3, n_min=3, n_max=3))
    depth = FakeDepthStage(scene, clock, latency_ms=10.0)
    pipe = _pipeline(scene, clock, depth, cfg=cfg)
    _feed(pipe, _frames(scene, 12), clock, loop_ms=33.0)
    st = pipe.stats
    assert len(st.boxes_ms_depth) == len(depth.submitted) == 4
    assert len(st.boxes_ms_no_depth) == 8
    assert len(st.e2e_alert_ms_depth) + len(st.e2e_alert_ms_no_depth) == st.frames
    d = st.to_dict()["by_depth_enqueue"]
    assert d["boxes_ms"]["depth"]["n"] == 4 and d["e2e_alert_ms"]["no_depth"]["n"] == 8


def test_run_series_one_row_per_window_with_drops() -> None:
    scene = _scene()
    clock = FakeClock()
    cfg = PipelineConfig(frame_hz=10.0, cadence=DepthCadenceConfig(n_init=1), series_period_s=1.0)
    pipe = _pipeline(scene, clock, FakeDepthStage(scene, clock, latency_ms=0.0), cfg=cfg)
    drops = [0]

    def gen() -> Iterator[tuple[FrameStamped, int]]:
        for fs in _frames(scene, 25):
            yield fs, clock()
            clock.advance(100.0)
            if fs.frame_id % 5 == 4:
                drops[0] += 1

    st = pipe.run(gen(), dropped=lambda: drops[0])
    rows = st.series
    assert [r["t_s"] for r in rows] == [0.0, 1.0, 2.0]
    assert [r["frames"] for r in rows] == [10.0, 10.0, 5.0]
    assert sum(r["frames"] for r in rows) == st.frames
    assert sum(r["dropped"] for r in rows) == st.dropped == 5
    assert all(r["e2e_max_ms"] >= r["e2e_p99_ms"] >= r["e2e_p50_ms"] for r in rows)
    assert rows[1]["depth_maps"] >= 1
    assert st.to_dict()["series"] == rows
    with pytest.raises(ValueError):
        _pipeline(scene, clock, None, cfg=PipelineConfig(series_period_s=0.0))
