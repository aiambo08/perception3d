"""F5: ByteTrack 2D, CV Kalman with variable Δt, ego-motion providers and Tracker3D (CPU)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from percepcion3d.camera.calibration import CameraIntrinsics, ExtrinsicMountConfig
from percepcion3d.camera.geometry import PinholeGeometry
from percepcion3d.depth.fusion import MetricFusionStage, load_fusion_config
from percepcion3d.eval.kitti import KittiTrackLabel
from percepcion3d.eval.tracking_kitti import gt_kinematics, run_tracking_kitti
from percepcion3d.eval.tracking_synthetic import (
    TrackingScenario,
    cpu_benchmark,
    run_tracking_scenario,
    tracking_dod,
)
from percepcion3d.io.sources import OxtsRecord
from percepcion3d.runtime.buffer import FrameStamped
from percepcion3d.sim.synthetic import SyntheticNoise, SyntheticObject, SyntheticScene
from percepcion3d.tracking.byte_tracker import ByteTrackConfig, ByteTracker, TrackState
from percepcion3d.tracking.ego_motion import ConstantEgoMotion, OxtsEgoMotion, ZeroEgoMotion
from percepcion3d.tracking.kalman_filter import (
    CvKalman,
    cv_transition,
    cwna_process_noise,
    predict_batch,
    rotation_2d,
    update_batch,
)
from percepcion3d.tracking.tracker3d import (
    MotionState,
    PositionMeasurement,
    Tracker3D,
    Tracker3DConfig,
    load_tracker_config,
    position_covariance,
)
from tests.test_fusion_kitti import EXTR as EXTR_F4
from tests.test_fusion_kitti import INTR as INTR_F4
from tests.test_fusion_kitti import _synthetic_labels

ROOT = Path(__file__).resolve().parents[1]
INTR = CameraIntrinsics(fx=721.5377, fy=721.5377, cx=609.5593, cy=172.854, width=1242, height=375)
EXTR = ExtrinsicMountConfig(camera_height_m=1.65, pitch_rad=0.0)
MS = 1_000_000


def _oxts(vf: float = 0.0, vl: float = 0.0, wu: float = 0.0) -> OxtsRecord:
    vals = [0.0] * 23
    vals[8], vals[9], vals[22] = vf, vl, wu
    return OxtsRecord(*vals)


# ----------------------------------------------------------------------------
# Kalman filter
# ----------------------------------------------------------------------------


def test_cv_transition_and_cwna_noise_closed_form() -> None:
    f = cv_transition(0.1)
    assert f @ np.array([1.0, 2.0, 3.0, -4.0]) == pytest.approx([1.3, 1.6, 3.0, -4.0])
    q = cwna_process_noise(0.1, 2.0)
    assert q[0, 0] == pytest.approx(2.0 * 1e-3 / 3) and q[0, 2] == pytest.approx(2.0 * 0.005)
    assert q[2, 2] == pytest.approx(0.2) and q[0, 1] == 0.0
    assert np.all(np.linalg.eigvalsh(q) > 0.0)


def test_predict_batch_matches_per_filter_propagation() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 4))
    a = rng.normal(size=(5, 4, 4))
    p = a @ np.transpose(a, (0, 2, 1)) + np.eye(4)
    dt = rng.uniform(0.01, 0.05, 5)
    q = rng.uniform(0.5, 2.0, 5)
    xb, pb = predict_batch(x, p, dt, q)
    for i in range(5):
        f = cv_transition(float(dt[i]))
        assert xb[i] == pytest.approx(f @ x[i])
        assert pb[i] == pytest.approx(f @ p[i] @ f.T + cwna_process_noise(float(dt[i]), q[i]))
    with pytest.raises(ValueError):
        predict_batch(x, p, -dt, q)


def test_predict_with_ego_keeps_static_world_point_fixed() -> None:
    ego = ConstantEgoMotion(v_forward_mps=10.0, yaw_rate_rps=0.3).delta(0, 100 * MS)
    kf = CvKalman(np.array([2.0, 20.0]), 0.01 * np.eye(2), 0, q=1.0, sigma_v0_mps=1.0)
    kf.predict(100 * MS, ego)
    expected = rotation_2d(ego.yaw_rad).T @ (np.array([2.0, 20.0]) - np.array(ego.translation_xz))
    assert kf.position == pytest.approx(expected)
    assert kf.t_ns == 100 * MS and np.allclose(kf.P, kf.P.T)


def test_update_reduces_uncertainty_and_gates_outliers() -> None:
    kf = CvKalman(np.array([0.0, 10.0]), np.eye(2), 0, q=1.0, sigma_v0_mps=5.0)
    kf.predict(100 * MS)
    tr0 = np.trace(kf.P)
    ok, nis = kf.update(np.array([0.1, 10.1]), 0.25 * np.eye(2), gate_chi2=13.8)
    assert ok and nis < 1.0 and np.trace(kf.P) < tr0
    x_before = kf.x.copy()
    ok, nis = kf.update(np.array([30.0, 60.0]), 0.25 * np.eye(2), gate_chi2=13.8)
    assert not ok and nis > 13.8 and kf.x == pytest.approx(x_before)


def test_lagged_measurement_uses_state_at_capture_time() -> None:
    x = np.array([[0.0, 10.0, 0.0, 5.0]])
    p = np.diag([0.01, 0.01, 1.0, 1.0])[None]
    z = np.array([[0.0, 9.5]])  # position 0.1 s ago under V_Z = 5 m/s
    _, _, ok, nis_lag = update_batch(x, p, z, 0.01 * np.eye(2)[None], np.ones(1), np.array([0.1]))
    _, _, _, nis_now = update_batch(x, p, z, 0.01 * np.eye(2)[None], np.ones(1))
    assert ok[0] and nis_lag[0] < 0.1 < nis_now[0]


def test_filter_converges_with_random_dt() -> None:
    rng = np.random.default_rng(1)
    t, pos = 0, np.array([1.0, 15.0])
    v = np.array([0.5, -3.0])
    kf = CvKalman(pos, 0.04 * np.eye(2), 0, q=0.5, sigma_v0_mps=10.0)
    for _ in range(100):
        dt_ns = int(rng.uniform(10, 50) * MS)
        t += dt_ns
        pos = pos + v * dt_ns * 1e-9
        kf.predict(t)
        kf.update(pos + rng.normal(0.0, 0.2, 2), 0.04 * np.eye(2))
    assert kf.velocity == pytest.approx(v, abs=0.5)


# ----------------------------------------------------------------------------
# Ego-motion
# ----------------------------------------------------------------------------


def test_zero_and_constant_ego_motion() -> None:
    z = ZeroEgoMotion().delta(0, 50 * MS)
    assert z.dt_s == pytest.approx(0.05) and not z.absolute and z.translation_xz == (0.0, 0.0)
    c = ConstantEgoMotion(12.0, sigma_velocity_mps=0.1).delta(0, 100 * MS)
    assert c.absolute and c.translation_xz == pytest.approx((0.0, 1.2))
    assert c.velocity_xz == pytest.approx((0.0, 12.0)) and c.sigma_translation_m == pytest.approx(
        0.01
    )
    turn = ConstantEgoMotion(10.0, yaw_rate_rps=0.5).delta(0, 200 * MS)
    assert turn.yaw_rad == pytest.approx(0.1)
    assert math.hypot(*turn.translation_xz) == pytest.approx(2.0)
    assert turn.translation_xz[0] < 0.0  # left turn: chord bends to −X


def test_oxts_ego_motion_axes_interpolation_and_lever_arm() -> None:
    t = [0, 100 * MS]
    ego = OxtsEgoMotion(t, [_oxts(10.0, 1.0), _oxts(12.0, 1.0)])
    d = ego.delta(0, 50 * MS)
    assert d.absolute and d.velocity_xz == pytest.approx((-1.0, 11.0))
    assert d.translation_xz == pytest.approx((-0.05, 0.525))
    arm = OxtsEgoMotion(t, [_oxts(10.0, wu=0.2)] * 2, lever_arm_fwd_m=1.0).delta(0, 10 * MS)
    assert arm.velocity_xz[0] == pytest.approx(-0.2)
    with pytest.raises(ValueError):
        OxtsEgoMotion([0], [])


# ----------------------------------------------------------------------------
# ByteTrack
# ----------------------------------------------------------------------------


def _box(cx: float, cy: float = 200.0, w: float = 60.0, h: float = 40.0) -> list[float]:
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def test_bytetrack_ids_stable_and_low_score_second_stage() -> None:
    bt = ByteTracker(ByteTrackConfig())
    ids: list[set[int]] = []
    for k in range(6):
        boxes = np.array([_box(100 + 5 * k), _box(600 - 5 * k)])
        scores = np.array([0.9, 0.9 if k != 4 else 0.3])  # second object dips to low score
        out = bt.update(k * 33 * MS, boxes, scores, ["car", "car"])
        ids.append({t.track_id for t in out if t.state is TrackState.CONFIRMED})
    assert ids[-1] == ids[2] and len(ids[-1]) == 2
    assert all(t.det_index >= 0 for t in bt.tracks)


def test_bytetrack_lifecycle_tentative_lost_and_deleted() -> None:
    bt = ByteTracker(ByteTrackConfig(max_lost_s=0.1))
    bt.update(0, np.array([_box(300)]), np.array([0.9]), ["car"])
    assert bt.update(33 * MS, np.zeros((0, 4)), np.zeros(0), []) == []  # tentative dropped
    for k in range(3):
        bt.update((100 + 33 * k) * MS, np.array([_box(300)]), np.array([0.9]), ["car"])
    (tr,) = bt.tracks
    assert tr.state is TrackState.CONFIRMED
    out = bt.update(200 * MS, np.zeros((0, 4)), np.zeros(0), [])
    assert out[0].state is TrackState.LOST and out[0].det_index == -1
    assert bt.update(400 * MS, np.zeros((0, 4)), np.zeros(0), []) == []


def test_bytetrack_class_groups_block_cross_class_matches() -> None:
    bt = ByteTracker(ByteTrackConfig(min_hits=1))

    def matched(boxes: list[list[float]], cls: str, t_ms: int) -> set[int]:
        out = bt.update(t_ms * MS, np.array(boxes), np.array([0.9]), [cls])
        return {t.track_id for t in out if t.det_index >= 0}

    car = matched([_box(300)], "car", 0)
    assert matched([_box(300)], "pedestrian", 33).isdisjoint(car)
    assert matched([_box(302)], "van", 66) == car


# ----------------------------------------------------------------------------
# Tracker3D
# ----------------------------------------------------------------------------


def test_position_covariance_follows_viewing_ray() -> None:
    c = position_covariance(5.0, 20.0, 0.3, 1.0)
    assert c[0, 1] == pytest.approx(0.25) and np.all(np.linalg.eigvalsh(c) > 0.0)
    assert position_covariance(-5.0, 20.0, 0.3, 1.0)[0, 1] < 0.0
    assert position_covariance(1.0, 0.0, 0.1, 1.0)[0, 1] == 0.0


def test_load_tracker_config_from_repo_yaml() -> None:
    cfg = load_tracker_config(ROOT / "configs" / "tracking.yaml")
    assert cfg.byte.high_thresh == 0.5 and cfg.dynamics_for("pedestrian").sigma_v0_mps == 3.0
    assert cfg.dynamics_for("unknown") == cfg.default_dynamics


def _static_step(trk: Tracker3D, k: int, dt_ms: int = 100) -> MotionState:
    t = k * dt_ms * MS
    z = 20.0 - 10.0 * k * dt_ms * 1e-3  # static object, ego at 10 m/s
    m = PositionMeasurement(0.0, z, position_covariance(0.0, z, 0.1, 0.3), t)
    out = trk.step(t, np.array([_box(600, w=60 + 2 * k)]), np.array([0.9]), ["car"], [m])
    return out[0].motion if out else MotionState.UNKNOWN


def test_tracker3d_static_object_with_ego_motion() -> None:
    trk = Tracker3D(Tracker3DConfig(), ConstantEgoMotion(10.0))
    states = [_static_step(trk, k) for k in range(15)]
    assert states[0] is MotionState.UNKNOWN and states[-1] is MotionState.STATIC
    (st,) = trk._state.values()
    assert np.hypot(*st.kf.velocity) < 1.0


def test_tracker3d_zero_ego_reports_relative_velocity_only() -> None:
    trk = Tracker3D(Tracker3DConfig(), ZeroEgoMotion())
    for k in range(15):
        motion = _static_step(trk, k)
    assert motion is MotionState.UNKNOWN
    out = trk.step(15 * 100 * MS, np.array([_box(600, w=90)]), np.array([0.9]), ["car"],
                   [PositionMeasurement(0.0, 18.5, 0.09 * np.eye(2), 15 * 100 * MS)])  # fmt: skip
    assert out[0].velocity_rel_xz[1] == pytest.approx(-10.0, abs=1.5)
    assert out[0].velocity_rel_xz == pytest.approx(out[0].velocity_abs_xz)


def test_tracker3d_requires_one_measurement_per_detection() -> None:
    with pytest.raises(ValueError):
        Tracker3D().step(0, np.array([_box(300)]), np.array([0.9]), ["car"], [])


# ----------------------------------------------------------------------------
# Evaluation harnesses
# ----------------------------------------------------------------------------


def test_synthetic_scenarios_meet_dod_except_cpu_timing() -> None:
    geo = PinholeGeometry(INTR, EXTR)
    res = {
        "nominal": run_tracking_scenario(geo, TrackingScenario("nominal", duration_s=4.0)),
        "dt_jitter_10_50ms": run_tracking_scenario(
            geo, TrackingScenario("dt_jitter_10_50ms", duration_s=4.0, dt_range_s=(0.01, 0.05))
        ),
        "turn_0.2rps": run_tracking_scenario(
            geo, TrackingScenario("turn_0.2rps", duration_s=4.0, ego_yaw_rate_rps=0.2)
        ),
        "straight_zero_ego": run_tracking_scenario(
            geo, TrackingScenario("straight_zero_ego", duration_s=4.0, ego_provider="zero")
        ),
    }
    cpu = cpu_benchmark(geo, n_objects=30, n_frames=20)
    assert cpu.shape == (10,) and np.all(cpu > 0.0)
    checks = tracking_dod(res, cpu)["checks"]
    for k, v in checks.items():
        if k != "cpu_p95_30_tracks_le_1ms":
            assert v, (k, res[k.split("_rmse")[0]].to_dict() if k in res else checks)
    assert res["nominal"].to_dict()["id_switches"] == 0


def test_gt_kinematics_apparent_and_translational_velocity() -> None:
    geo = PinholeGeometry(INTR, EXTR)
    labels = {
        f: [
            KittiTrackLabel(
                f,
                7,
                "Car",
                0.0,
                0,
                0.0,
                (0, 0, 100, 60),
                (1.5, 1.8, 4.2),
                (3.0, 1.65, 20.0 - 1.0 * f),
                math.pi / 2,
            )
        ]  # fmt: skip
        for f in range(5)
    }
    k = gt_kinematics(labels, geo)
    assert set(k) == {(1, 7), (2, 7), (3, 7)}
    assert k[(2, 7)].v_app_xz == pytest.approx((0.0, -10.0))
    assert k[(2, 7)].v_rel_xz is None
    oxts = [_oxts(10.0)] * 5
    ko = gt_kinematics(labels, geo, oxts)[(2, 7)]
    assert ko.v_rel_xz == pytest.approx((0.0, -10.0)) and ko.v_abs_mps == pytest.approx(0.0)
    kt = gt_kinematics(labels, geo, [_oxts(10.0, wu=0.1)] * 5)[(2, 7)]
    assert kt.v_rel_xz == pytest.approx((-0.1 * 18.0, -10.0 + 0.1 * 3.0))


def test_run_tracking_kitti_on_synthetic_sequence() -> None:
    geo = PinholeGeometry(INTR_F4, EXTR_F4)
    scene = SyntheticScene(
        geo,
        [
            SyntheticObject(1, "car", x0_m=0.5, z0_m=12.0, vz_mps=1.0, height_m=1.52),
            SyntheticObject(2, "car", x0_m=-3.5, z0_m=25.0, vz_mps=-3.0, height_m=1.45),
        ],
        noise=SyntheticNoise(box_px=0.3, inv_depth_rel=0.0),
        fps=10.0,
    )
    n = 30
    labels = _synthetic_labels(scene, n)
    frames = [FrameStamped(k, scene.frame(k).t_ns, np.zeros((4, 4, 3), np.uint8)) for k in range(n)]
    stage = MetricFusionStage(INTR_F4, geo, load_fusion_config(ROOT / "configs" / "fusion.yaml"))
    trk = Tracker3D(load_tracker_config(ROOT / "configs" / "tracking.yaml"))
    res = run_tracking_kitti(frames, labels, stage, trk, gt_kinematics(labels, geo))
    d = res.to_dict()
    assert d["velocity_reference"] == "apparent" and d["n_frames"] == n
    assert d["id_switches"] == 0 and d["n_gt_ids"] == 2
    assert d["n_vel_samples"] > 20 and d["rmse_vel_rel_mps"] < 1.0, d
    assert len(d["cpu_ms_p50_p95_p99"]) == 3
