"""Synthetic evaluation of the F5 tracker (DoD of ``docs/01_plan_fases_mvp.md``).

A small world simulator (independent of the tracker's own conventions): the ego
vehicle drives an arc at constant speed ``v`` and yaw rate ``ω``; objects move at
constant over-ground velocity. At each capture time (fixed ``1/fps`` or jittered
``Δt ~ U[lo, hi]``) every visible object yields

* a detection box — projected cuboid + ``box_px`` noise, score 0.9;
* a position measurement of its centre with the converted-measurement
  covariance the F4 fusion reports (``σ_Z = rel_sigma_z·Z``, ``σ_X`` from the
  ray geometry and ``sigma_col_px``), drawn from exactly that covariance.

Ground truth per sample: relative translational velocity
``Rᵀ(ψ)(v_obj − v_ego)``, over-ground velocity ``Rᵀ(ψ)·v_obj`` and position in
the current ego axes. Scored:

* ``RMSE(V_Z rel)`` for tracks older than ``score_after_s`` whose object is at
  ``Z ∈ [10, 20] m`` (the plan's "at 15 m");
* filter consistency: fraction of samples whose 4-dof NEES exceeds the χ² 99.9 %
  bound (18.47) — the divergence check under random ``Δt``;
* static labelling: fraction of samples of static objects (track age ≥ 0.5 s)
  labelled ``STATIC``;
* ID switches of the 2D association.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from percepcion3d.camera.geometry import PinholeGeometry
from percepcion3d.sim.synthetic import cuboid_corners
from percepcion3d.tracking.ego_motion import ConstantEgoMotion, EgoMotionProvider, ZeroEgoMotion
from percepcion3d.tracking.kalman_filter import rotation_2d
from percepcion3d.tracking.tracker3d import (
    MotionState,
    PositionMeasurement,
    Tracker3D,
    Tracker3DConfig,
    position_covariance,
)

NEES_4DOF_999 = 18.47
_DIMS: dict[str, tuple[float, float, float]] = {
    "car": (1.8, 1.5, 4.3),
    "pedestrian": (0.6, 1.7, 0.5),
}


@dataclass(frozen=True)
class SimObject:
    obj_id: int
    cls: str
    x0_m: float
    z0_m: float
    """Initial centre in the ego axes at ``t = 0`` (m)."""
    vx_mps: float = 0.0
    vz_mps: float = 0.0
    """Over-ground velocity in the initial ego axes."""


@dataclass(frozen=True)
class TrackingScenario:
    name: str
    duration_s: float = 6.0
    fps: float = 30.0
    dt_range_s: tuple[float, float] | None = None
    """Jittered capture interval ``U[lo, hi]``; ``None`` → fixed ``1/fps``."""
    ego_speed_mps: float = 12.0
    ego_yaw_rate_rps: float = 0.0
    ego_provider: str = "true"
    """``true`` (exact ego-motion) or ``zero``."""
    rel_sigma_z: float = 0.05
    sigma_col_px: float = 1.5
    box_px: float = 1.5
    score_after_s: float = 1.0
    bin_m: tuple[float, float] = (10.0, 20.0)
    objects: tuple[SimObject, ...] = field(default_factory=lambda: default_objects())


def default_objects() -> tuple[SimObject, ...]:
    """Lead car closing at 3 m/s, oncoming car, crossing pedestrian, parked cars."""
    objs = [
        SimObject(1, "car", 0.0, 15.0, vz_mps=9.0),
        SimObject(2, "car", -3.5, 70.0, vz_mps=-10.0),
        SimObject(3, "pedestrian", 6.0, 22.0, vx_mps=-1.2),
        SimObject(4, "car", 3.6, 45.0, vz_mps=14.0),
    ]
    objs += [SimObject(10 + i, "car", -6.0 if i % 2 else 6.0, 20.0 + 9.0 * i) for i in range(8)]
    return tuple(objs)


@dataclass
class TrackingScenarioResult:
    name: str
    n_samples: int = 0
    vz_err: list[float] = field(default_factory=list)
    vx_err: list[float] = field(default_factory=list)
    nees: list[float] = field(default_factory=list)
    static_samples: int = 0
    static_labelled: int = 0
    moving_samples: int = 0
    moving_as_static: int = 0
    id_switches: int = 0
    step_ms: list[float] = field(default_factory=list)

    @property
    def rmse_vz(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.vz_err)))) if self.vz_err else float("nan")

    @property
    def rmse_vx(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.vx_err)))) if self.vx_err else float("nan")

    @property
    def nees_exceed_frac(self) -> float:
        return float(np.mean(np.asarray(self.nees) > NEES_4DOF_999)) if self.nees else float("nan")

    @property
    def static_frac(self) -> float:
        return self.static_labelled / self.static_samples if self.static_samples else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_scored": len(self.vz_err),
            "rmse_vz_rel_mps": self.rmse_vz,
            "rmse_vx_rel_mps": self.rmse_vx,
            "nees_mean": float(np.mean(self.nees)) if self.nees else float("nan"),
            "nees_exceed_frac": self.nees_exceed_frac,
            "static_frac": self.static_frac,
            "static_samples": self.static_samples,
            "moving_as_static_frac": (
                self.moving_as_static / self.moving_samples if self.moving_samples else float("nan")
            ),
            "id_switches": self.id_switches,
        }


def _ego_pose(t: float, v: float, w: float) -> tuple[NDArray[np.float64], float]:
    """Ego position (initial axes) and heading after ``t`` s of a left-positive arc."""
    psi = w * t
    if abs(w) < 1e-9:
        return np.array([0.0, v * t]), 0.0
    return np.array([v / w * (np.cos(psi) - 1.0), v / w * np.sin(psi)]), psi


def _project_box(
    geo: PinholeGeometry, x: float, z: float, dims: tuple[float, float, float]
) -> NDArray[np.float64] | None:
    w, h, length = dims
    k = geo.intrinsics
    corners = cuboid_corners(x, z, geo.extrinsics.camera_height_m, w, h, length) @ geo.r_cg
    if np.any(corners[:, 2] < 0.5):
        return None
    u = k.fx * corners[:, 0] / corners[:, 2] + k.cx
    v = k.fy * corners[:, 1] / corners[:, 2] + k.cy
    box = np.array([u.min(), v.min(), u.max(), v.max()])
    clip = np.clip(box, [0, 0, 0, 0], [k.width - 1, k.height - 1, k.width - 1, k.height - 1])
    if clip[2] - clip[0] < 4 or clip[3] - clip[1] < 8:
        return None
    return np.asarray(clip, dtype=np.float64)


def _timestamps(sc: TrackingScenario, rng: np.random.Generator) -> NDArray[np.float64]:
    if sc.dt_range_s is None:
        n = int(sc.duration_s * sc.fps)
        return np.asarray(np.arange(n, dtype=np.float64) / float(sc.fps), dtype=np.float64)
    lo, hi = sc.dt_range_s
    dts = rng.uniform(lo, hi, size=int(sc.duration_s / lo) + 1)
    t = np.concatenate([[0.0], np.cumsum(dts)])
    return np.asarray(t[t <= sc.duration_s], dtype=np.float64)


def run_tracking_scenario(
    geo: PinholeGeometry,
    sc: TrackingScenario,
    cfg: Tracker3DConfig | None = None,
    seed: int = 0,
) -> TrackingScenarioResult:
    rng = np.random.default_rng(seed)
    ego: EgoMotionProvider = (
        ConstantEgoMotion(sc.ego_speed_mps, yaw_rate_rps=sc.ego_yaw_rate_rps)
        if sc.ego_provider == "true"
        else ZeroEgoMotion()
    )
    tracker = Tracker3D(cfg, ego)
    res = TrackingScenarioResult(sc.name)
    fx = geo.intrinsics.fx
    last_tid: dict[int, int] = {}
    birth: dict[int, float] = {}
    for t in _timestamps(sc, rng):
        pos_e, psi = _ego_pose(float(t), sc.ego_speed_mps, sc.ego_yaw_rate_rps)
        rt = rotation_2d(psi).T
        v_ego_w = sc.ego_speed_mps * np.array([-np.sin(psi), np.cos(psi)])
        boxes, classes, meas, gts = [], [], [], []
        for o in sc.objects:
            v_w = np.array([o.vx_mps, o.vz_mps])
            p = rt @ (np.array([o.x0_m, o.z0_m]) + v_w * t - pos_e)
            box = _project_box(geo, float(p[0]), float(p[1]), _DIMS[o.cls])
            if box is None:
                continue
            sz = sc.rel_sigma_z * p[1]
            sx = float(np.hypot(abs(p[0]) / p[1] * sz, p[1] * sc.sigma_col_px / fx))
            cov = position_covariance(float(p[0]), float(p[1]), sx, sz)
            zm = rng.multivariate_normal(p, cov)
            t_ns = int(round(t * 1e9))
            boxes.append(box + rng.normal(0.0, sc.box_px, 4))
            classes.append(o.cls)
            meas.append(PositionMeasurement(float(zm[0]), float(zm[1]), cov, t_ns))
            gts.append((o, p, rt @ v_w, rt @ (v_w - v_ego_w)))
        t_ns = int(round(t * 1e9))
        b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        t0 = time.perf_counter()
        tracks = tracker.step(t_ns, b, np.full(len(boxes), 0.9), classes, meas)
        res.step_ms.append((time.perf_counter() - t0) * 1e3)
        for tr in tracks:
            if tr.det_index < 0:
                continue
            o, p, v_abs, v_rel = gts[tr.det_index]
            if o.obj_id in last_tid and last_tid[o.obj_id] != tr.track_id:
                res.id_switches += 1
            last_tid[o.obj_id] = tr.track_id
            birth.setdefault(tr.track_id, float(t))
            res.n_samples += 1
            v_state = v_abs if sc.ego_provider == "true" else v_rel
            e = np.r_[tr.position_xz - p, tr.velocity_abs_xz - v_state]
            res.nees.append(float(e @ np.linalg.solve(tr.cov, e)))
            if o.vx_mps == 0.0 and o.vz_mps == 0.0 and tr.age_s >= 0.5:
                res.static_samples += 1
                res.static_labelled += int(tr.motion is MotionState.STATIC)
            elif np.hypot(o.vx_mps, o.vz_mps) >= 5.0 and tr.age_s >= 1.0:
                res.moving_samples += 1
                res.moving_as_static += int(tr.motion is MotionState.STATIC)
            if tr.age_s >= sc.score_after_s and sc.bin_m[0] <= p[1] <= sc.bin_m[1]:
                dv = tr.velocity_rel_xz - v_rel
                res.vx_err.append(float(dv[0]))
                res.vz_err.append(float(dv[1]))
    return res


def default_tracking_scenarios() -> list[TrackingScenario]:
    return [
        TrackingScenario("nominal"),
        TrackingScenario("dt_jitter_10_50ms", dt_range_s=(0.010, 0.050)),
        TrackingScenario("turn_0.2rps", ego_yaw_rate_rps=0.2),
        TrackingScenario("straight_zero_ego", ego_provider="zero"),
        TrackingScenario("turn_0.2rps_zero_ego", ego_yaw_rate_rps=0.2, ego_provider="zero"),
    ]


def cpu_benchmark(
    geo: PinholeGeometry, n_objects: int = 30, n_frames: int = 300, seed: int = 0
) -> NDArray[np.float64]:
    """``Tracker3D.step`` time (ms) per frame with ``n_objects`` visible, warm-up excluded."""
    cols = np.linspace(-12.0, 12.0, 6)
    rows = np.linspace(12.0, 60.0, (n_objects + 5) // 6)
    objs = tuple(
        SimObject(i, "car", float(x), float(z), vz_mps=12.0 + 0.1 * i)
        for i, (x, z) in enumerate((x, z) for z in rows for x in cols)
    )[:n_objects]
    sc = TrackingScenario("cpu", duration_s=n_frames / 30.0, objects=objs)
    res = run_tracking_scenario(geo, sc, seed=seed)
    return np.asarray(res.step_ms[10:], dtype=np.float64)


def tracking_dod(
    results: dict[str, TrackingScenarioResult], cpu_ms: NDArray[np.float64]
) -> dict[str, Any]:
    nom, jit = results["nominal"], results["dt_jitter_10_50ms"]
    turn = results["turn_0.2rps"]
    zero = results["straight_zero_ego"]
    p95 = float(np.percentile(cpu_ms, 95)) if cpu_ms.size else float("nan")
    checks = {
        "rmse_vz_15m_le_0.5": nom.rmse_vz <= 0.5,
        "jitter_rmse_vz_le_0.5": jit.rmse_vz <= 0.5,
        "jitter_nees_exceed_le_5pct": jit.nees_exceed_frac <= 0.05,
        "static_frac_turn_ge_0.9": turn.static_frac >= 0.9,
        "zero_ego_rel_velocity_invariant": abs(zero.rmse_vz - nom.rmse_vz) <= 0.1,
        "cpu_p95_30_tracks_le_1ms": p95 <= 1.0,
    }
    return {"checks": checks, "cpu_p95_ms": p95, "pass": all(checks.values())}
