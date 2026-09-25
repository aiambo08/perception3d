"""Synthetic evaluation of the F4 metric fusion (DoD of ``docs/01_plan_fases_mvp.md``).

A :class:`~percepcion3d.sim.synthetic.SyntheticScene` renders dense relative
inverse-depth maps and detections with known ground truth; the true camera
pitch is jittered per frame (bumps) and optionally biased (mount error) while
the fusion runs with the *nominal* extrinsics. Errors are reported per cue
(``Z_g``, ``Z_n``, ``Z_h``) and for the fused ``Ẑ`` against the depth of the
near ground contact (``gt_z_front_m``).

Object heights are drawn per object from the class prior ``N(h, σ_h)`` of
``configs/fusion.yaml`` so that ``Z_h`` carries its real-world error (a scene
whose objects all have exactly the prior height makes ``Z_h`` unbeatable and
the comparison meaningless). A scene has only ~10 objects, i.e. ~10 height
draws, so each scenario is repeated over several seeds and the errors pooled:
with a single seed the ranking of cues at P95 is decided by which cars happened
to be drawn close to the prior height.

DoD reading used here (the plan's wording is ambiguous):

* *nominal*: ``P95(|Ẑ−Z|/Z) ≤ min_cue P95`` — the fused estimate is at least as
  good as the best single cue at the 95th percentile (per-sample dominance of
  the best of three cues is impossible by construction and is reported only as
  a fraction).
* *pitch bias 1°*: degradation of the *median* error, because a constant bias
  shifts the centre of the distribution while the P95 tail is dominated by the
  per-frame pitch jitter on far objects. ``Z_g`` must degrade > 100 % when the
  bias is not corrected (online pitch off) and the full system (online pitch
  on) must degrade < 30 %; the fused error with online pitch off is reported
  too so the contribution of the estimator is visible.
* *CPU*: ``P95`` of ``MetricFusionStage.process`` with 20 boxes ≤ 2 ms, measured
  on the frames without a ground-grid rebuild (the rebuild is a ≈ 1 ms
  amortised cost that only fires when the pitch estimate moves ≥ 0.1°); the
  overall P95 including rebuilds is reported alongside.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from percepcion3d.camera.calibration import CameraIntrinsics, ExtrinsicMountConfig
from percepcion3d.camera.geometry import PinholeGeometry
from percepcion3d.depth.depth_trt import DepthMap
from percepcion3d.depth.fusion import FusionConfig, MetricFusionStage
from percepcion3d.depth.ground_solver import (
    AffineFilterConfig,
    PitchFilterConfig,
    RoadSampleConfig,
)
from percepcion3d.sim.synthetic import (
    SyntheticNoise,
    SyntheticObject,
    SyntheticScene,
)

CUES = ("ground", "net", "height", "fused")

#: (class, width_m, length_m, silhouette_frac) of the synthetic object kinds.
_KINDS: dict[str, tuple[float, float, float]] = {
    "car": (1.8, 4.3, 1.0),
    "pedestrian": (0.6, 0.5, 0.4),
    "cyclist": (0.6, 1.7, 0.5),
}


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    pitch_bias_deg: float = 0.0
    """Constant error of the assumed mount pitch (assumed = true + bias)."""
    pitch_jitter_deg: float = 0.5
    """Per-frame 1-σ of the true pitch around its nominal (road bumps, braking)."""
    box_px: float = 1.5
    """Detector jitter on every box edge; equals ``noise.sigma_row_px`` in ``fusion.yaml``
    so the evaluation exercises the noise the fusion's σ models are declared for."""
    inv_depth_rel: float = 0.02
    pixel_noise_rel: float = 0.01
    online_pitch: bool = True
    n_frames: int = 90
    warmup_frames: int = 15
    n_seeds: int = 4
    """Independent scenes (object heights, noise, jitter) pooled into one result."""
    net_hw: tuple[int, int] = (280, 924)
    dense_traffic: bool = False
    """Use the 20-object scene (timing DoD) instead of the default highway scene."""


@dataclass
class ScenarioResult:
    name: str
    n_samples: int
    median: dict[str, float] = field(default_factory=dict)
    p95: dict[str, float] = field(default_factory=dict)
    frac_fused_le_best: float = float("nan")
    pitch_est_deg: float = float("nan")
    pitch_true_deg: float = float("nan")
    timing_ms: tuple[float, float, float] = (float("nan"),) * 3
    """P50/P95/P99 of ``process`` on frames *without* a ground-grid rebuild."""
    timing_all_ms: tuple[float, float, float] = (float("nan"),) * 3
    """Same percentiles over all frames (rebuilds included)."""
    n_boxes_mean: float = float("nan")
    n_grid_builds: int = 0
    n_frames_timed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "median_rel": self.median,
            "p95_rel": self.p95,
            "frac_fused_le_best": self.frac_fused_le_best,
            "pitch_est_deg": self.pitch_est_deg,
            "pitch_true_deg": self.pitch_true_deg,
            "timing_ms_p50_p95_p99": list(self.timing_ms),
            "timing_all_ms_p50_p95_p99": list(self.timing_all_ms),
            "n_boxes_mean": self.n_boxes_mean,
            "n_grid_builds": self.n_grid_builds,
            "n_frames_timed": self.n_frames_timed,
        }


def _object(
    tid: int,
    kind: str,
    x: float,
    z: float,
    vx: float,
    vz: float,
    fusion_cfg: FusionConfig,
    rng: np.random.Generator,
) -> SyntheticObject:
    prior = fusion_cfg.priors[kind]
    width, length, sil = _KINDS[kind]
    height = float(rng.normal(prior.height_m, prior.sigma_height_m))
    return SyntheticObject(
        tid,
        kind,
        x0_m=x,
        z0_m=z,
        vx_mps=vx,
        vz_mps=vz,
        width_m=width,
        height_m=max(height, 0.5 * prior.height_m),
        length_m=length,
        silhouette_frac=sil,
    )


def mixed_scene(
    geometry: PinholeGeometry, fusion_cfg: FusionConfig, seed: int = 0
) -> SyntheticScene:
    """Six cars, two pedestrians and a cyclist over 8–65 m (nominal / pitch DoD).

    Slow relative motion so that every object stays in frame for ~3 s.
    """
    rng = np.random.default_rng([seed, 41])
    spec: list[tuple[str, float, float, float, float]] = [
        ("car", 0.0, 12.0, 0.0, -1.0),
        ("car", 3.5, 20.0, 0.0, 1.0),
        ("car", -3.5, 28.0, 0.0, -1.5),
        ("car", 0.0, 38.0, 0.0, 1.0),
        ("car", 3.5, 50.0, 0.0, -2.0),
        ("car", -3.5, 65.0, 0.0, 1.5),
        ("pedestrian", 6.0, 9.0, -0.5, 0.0),
        ("pedestrian", -6.5, 24.0, 0.4, 0.0),
        ("cyclist", 5.5, 16.0, 0.0, 2.0),
    ]
    objs = [
        _object(i + 1, kind, x, z, vx, vz, fusion_cfg, rng)
        for i, (kind, x, z, vx, vz) in enumerate(spec)
    ]
    return SyntheticScene(geometry, objs, fps=30.0, ego_velocity_mps=(0.0, 0.0), seed=seed)


def dense_traffic_scene(
    geometry: PinholeGeometry, fusion_cfg: FusionConfig, seed: int = 0
) -> SyntheticScene:
    """20 cars in five lanes, 10–70 m, slow relative motion (timing / many-box DoD)."""
    rng = np.random.default_rng([seed, 42])
    objs: list[SyntheticObject] = []
    lanes = (-7.0, -3.5, 0.0, 3.5, 7.0)
    tid = 1
    for i, x in enumerate(lanes):
        for j in range(4):
            z = 10.0 + 15.0 * j + 3.0 * i
            objs.append(_object(tid, "car", x, z, 0.0, -1.0 + 0.5 * j, fusion_cfg, rng))
            tid += 1
    return SyntheticScene(geometry, objs, fps=30.0, ego_velocity_mps=(0.0, 0.0), seed=seed)


def _scene(
    geometry: PinholeGeometry, fusion_cfg: FusionConfig, sc: ScenarioConfig, seed: int
) -> SyntheticScene:
    scene = (
        dense_traffic_scene(geometry, fusion_cfg, seed)
        if sc.dense_traffic
        else mixed_scene(geometry, fusion_cfg, seed)
    )
    scene.noise = SyntheticNoise(box_px=sc.box_px, inv_depth_rel=sc.inv_depth_rel)
    scene.disparity_scale = 3.0
    scene.disparity_shift = 0.2
    return scene


def run_scenario(
    intrinsics: CameraIntrinsics,
    extrinsics_true: ExtrinsicMountConfig,
    fusion_cfg: FusionConfig,
    road_cfg: RoadSampleConfig,
    affine_cfg: AffineFilterConfig,
    pitch_cfg: PitchFilterConfig,
    sc: ScenarioConfig,
    seed: int = 0,
) -> ScenarioResult:
    acc = _Accumulator()
    pitch_est: list[float] = []
    n_builds = 0
    for s in range(sc.n_seeds):
        stage = _run_once(
            intrinsics,
            extrinsics_true,
            fusion_cfg,
            road_cfg,
            affine_cfg,
            pitch_cfg,
            sc,
            seed + s,
            acc,
        )
        pitch_est.append(float(np.rad2deg(stage.pitch.state.pitch_rad)))
        n_builds += stage.solver.grids.n_builds

    res = ScenarioResult(name=sc.name, n_samples=len(acc.errs["fused"]))
    for c in CUES:
        arr = np.asarray(acc.errs[c])
        res.median[c] = float(np.median(arr)) if arr.size else float("nan")
        res.p95[c] = float(np.percentile(arr, 95)) if arr.size else float("nan")
    if acc.triple:
        t = np.asarray(acc.triple)
        res.frac_fused_le_best = float(np.mean(t[:, 3] <= t[:, :3].min(axis=1) + 1e-12))
    res.pitch_est_deg = float(np.mean(pitch_est))
    res.pitch_true_deg = float(np.rad2deg(extrinsics_true.pitch_rad))
    if acc.times:
        arr_t = np.asarray(acc.times)
        steady = arr_t[~np.asarray(acc.rebuilt)]
        if steady.size == 0:
            steady = arr_t
        res.timing_ms = _percentiles(steady)
        res.timing_all_ms = _percentiles(arr_t)
        res.n_boxes_mean = float(np.mean(acc.n_boxes))
        res.n_frames_timed = int(arr_t.size)
    res.n_grid_builds = n_builds
    return res


@dataclass
class _Accumulator:
    errs: dict[str, list[float]] = field(default_factory=lambda: {c: [] for c in CUES})
    triple: list[tuple[float, float, float, float]] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    rebuilt: list[bool] = field(default_factory=list)
    n_boxes: list[int] = field(default_factory=list)


def _run_once(
    intrinsics: CameraIntrinsics,
    extrinsics_true: ExtrinsicMountConfig,
    fusion_cfg: FusionConfig,
    road_cfg: RoadSampleConfig,
    affine_cfg: AffineFilterConfig,
    pitch_cfg: PitchFilterConfig,
    sc: ScenarioConfig,
    seed: int,
    acc: _Accumulator,
) -> MetricFusionStage:
    """One scene / one seed; appends per-sample errors and per-frame timings to ``acc``."""
    rng = np.random.default_rng([seed, 4])
    geo_true_nominal = PinholeGeometry(intrinsics, extrinsics_true)
    base_scene = _scene(geo_true_nominal, fusion_cfg, sc, seed)
    assumed = replace(
        extrinsics_true, pitch_rad=extrinsics_true.pitch_rad + float(np.deg2rad(sc.pitch_bias_deg))
    )
    stage = MetricFusionStage(
        intrinsics,
        PinholeGeometry(intrinsics, assumed),
        fusion_cfg,
        road_cfg,
        affine_cfg,
        pitch_cfg,
        online_pitch=sc.online_pitch,
        seed=seed,
    )
    for k in range(sc.n_frames):
        jitter = (
            float(rng.normal(0.0, np.deg2rad(sc.pitch_jitter_deg))) if sc.pitch_jitter_deg else 0.0
        )
        ext_k = replace(extrinsics_true, pitch_rad=extrinsics_true.pitch_rad + jitter)
        scene_k = replace(base_scene, geometry=PinholeGeometry(intrinsics, ext_k))
        f = scene_k.frame(k)
        inv, rs = scene_k.dense_inv_depth_map(f, sc.net_hw, pixel_noise_rel=sc.pixel_noise_rel)
        dm = DepthMap(f.frame_id, f.t_ns, inv.astype(np.float32), "relative_disparity", rs)
        builds_before = stage.solver.grids.n_builds
        t0 = time.perf_counter()
        out = stage.process(f.frame_id, f.t_ns, f.boxes, f.classes, dm)
        dt_ms = (time.perf_counter() - t0) * 1e3
        if k < sc.warmup_frames:
            continue
        acc.times.append(dt_ms)
        acc.rebuilt.append(stage.solver.grids.n_builds != builds_before)
        acc.n_boxes.append(len(f))
        for m, zt in zip(out.measurements, f.gt_z_front_m, strict=True):
            vals = {
                "ground": m.z_ground_m,
                "net": m.z_net_m,
                "height": m.z_height_m,
                "fused": m.z_cam_m,
            }
            rel = {c: abs(v - zt) / zt for c, v in vals.items()}
            for c, r in rel.items():
                if np.isfinite(r):
                    acc.errs[c].append(float(r))
            if all(np.isfinite(r) for r in rel.values()):
                acc.triple.append((rel["ground"], rel["net"], rel["height"], rel["fused"]))
    return stage


def _percentiles(x: np.ndarray[Any, np.dtype[np.float64]]) -> tuple[float, float, float]:
    p50, p95, p99 = (float(np.percentile(x, q)) for q in (50, 95, 99))
    return p50, p95, p99


def default_scenarios() -> list[ScenarioConfig]:
    return [
        ScenarioConfig("nominal"),
        ScenarioConfig("pitch_bias_1deg", pitch_bias_deg=1.0),
        ScenarioConfig("pitch_bias_1deg_no_online", pitch_bias_deg=1.0, online_pitch=False),
        ScenarioConfig("dense_traffic_timing", dense_traffic=True, n_frames=120, n_seeds=1),
    ]


def dod_summary(results: dict[str, ScenarioResult]) -> dict[str, Any]:
    nom = results["nominal"]
    per_on = results["pitch_bias_1deg"]
    per_off = results["pitch_bias_1deg_no_online"]
    best_cue_p95 = min(nom.p95[c] for c in ("ground", "net", "height"))
    deg_f_on = per_on.median["fused"] / nom.median["fused"] - 1.0
    deg_f_off = per_off.median["fused"] / nom.median["fused"] - 1.0
    deg_g_off = per_off.median["ground"] / nom.median["ground"] - 1.0
    timing = results["dense_traffic_timing"]
    return {
        "nominal_fused_p95_le_best_cue": bool(nom.p95["fused"] <= best_cue_p95),
        "nominal_fused_p95": nom.p95["fused"],
        "nominal_best_cue_p95": best_cue_p95,
        "pitch1deg_fused_median_degradation_online": deg_f_on,
        "pitch1deg_fused_median_degradation_no_online": deg_f_off,
        "pitch1deg_ground_median_degradation_no_online": deg_g_off,
        "pitch1deg_pass": bool(deg_f_on < 0.30 and deg_g_off > 1.0),
        "cpu_p95_ms_20_boxes_steady": timing.timing_ms[1],
        "cpu_p95_ms_20_boxes_all": timing.timing_all_ms[1],
        "cpu_n_boxes_mean": timing.n_boxes_mean,
        "cpu_pass": bool(timing.timing_ms[1] <= 2.0),
    }


def format_results(results: dict[str, ScenarioResult], summary: dict[str, Any]) -> str:
    lines = [f"{'scenario':28s} {'n':>5s} " + " ".join(f"{c + ' P50/P95':>18s}" for c in CUES)]
    for r in results.values():
        cells = " ".join(f"{r.median[c] * 100:7.2f}%/{r.p95[c] * 100:6.2f}%" for c in CUES)
        lines.append(f"{r.name:28s} {r.n_samples:5d} {cells}")
        lines.append(
            f"    pitch est/true {r.pitch_est_deg:.2f}/{r.pitch_true_deg:.2f} deg · "
            f"boxes {r.n_boxes_mean:.1f} · CPU P50/P95/P99 "
            f"{r.timing_ms[0]:.2f}/{r.timing_ms[1]:.2f}/{r.timing_ms[2]:.2f} ms "
            f"(all frames P95 {r.timing_all_ms[1]:.2f} ms, grid builds {r.n_grid_builds}) · "
            f"fused≤best {r.frac_fused_le_best * 100:.0f}%"
        )
    lines.append("DoD:")
    for k, v in summary.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)
