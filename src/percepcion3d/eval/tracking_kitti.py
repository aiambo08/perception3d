"""F5 on KITTI tracking: identity and relative-velocity accuracy of :class:`Tracker3D`.

Pipeline per frame: boxes (labels or detector) → F4 :class:`MetricFusionStage` →
:func:`measurement_from_fusion` → :meth:`Tracker3D.step`.

Ground truth from the 3D labels (bottom centre in the rectified camera frame, rotated
into the ground frame G with the nominal mount pitch):

* **apparent velocity** ``dp/dt`` of the relative position, central difference over
  ``±k`` frames of the same ``track_id``;
* with OXTS, the **translational relative velocity** that the tracker reports,
  ``v_rel = dp/dt + ω·(−Z, X)`` (the rotating-frame term of a yawing camera), and the
  over-ground velocity ``v_rel + v_ego`` used to label GT objects static.

Scored samples: labels of ``score_types`` with ``occluded ≤ 1``, ``truncated ≤ 0.3``,
height ≥ 25 px, whose track is at least ``min_age_s`` old and ``Z ∈ bin_m``.

ID switches are counted per GT id over every matched frame and normalised per 100
matches. The 3D stage never changes 2D identities (see
:mod:`percepcion3d.tracking.tracker3d`), so they only depend on the boxes and the
``byte_track`` thresholds.

Diagnostics: the velocity error is broken down by distance bin and by ego yaw rate
(bias, RMSE, P50/P95 of ``|Δv|``). ``gt_alt`` (the same reference with a wider
difference window) gives the spread of the GT itself, a floor on the attainable RMSE
at that distance.

Turn diagnostics (``turn``, needs OXTS) separate the candidate causes of a velocity
bias while the ego vehicle yaws:

* ``vz_err_vs_omega_x`` / ``vx_err_vs_omega_z``: fits of the error against the
  rotating-frame term ``ω·(−Z, X)``; a slope near ``±1`` means that term is missing
  or has the wrong sign on one side (tracker or GT);
* ``by_turn_direction``: left / straight / right, with the mean position error
  ``ez`` (tracker − GT), its relative value ``ez/Z`` and its rate ``d(ez)/dt`` (central
  difference over the same GT track). A velocity bias equal to ``d(ez)/dt`` comes
  from a range error that drifts during the turn (depth/ground), not from the
  ego-motion compensation; ``vz_err_minus_range_drift`` is what remains.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from percepcion3d.camera.geometry import PinholeGeometry
from percepcion3d.depth.fusion import MetricFusionStage
from percepcion3d.eval.detection import match_greedy
from percepcion3d.eval.fusion_kitti import (
    KITTI_TO_FUSION_CLASS,
    BoxProvider,
    DepthProvider,
    labels_to_boxes,
)
from percepcion3d.eval.kitti import KittiTrackLabel
from percepcion3d.io.sources import OxtsRecord
from percepcion3d.runtime.buffer import FrameStamped
from percepcion3d.tracking.tracker3d import MotionState, Tracker3D, measurement_from_fusion

KITTI_FPS = 10.0


@dataclass(frozen=True)
class GtKinematics:
    x_m: float
    z_m: float
    v_app_xz: tuple[float, float]
    """Apparent velocity ``dp/dt`` in G (m/s)."""
    v_rel_xz: tuple[float, float] | None
    v_abs_mps: float | None
    yaw_rate_rps: float | None = None
    ego_fwd_mps: float | None = None
    a_ref_xz: tuple[float, float] | None = None
    """Acceleration of the velocity reference (``v_rel`` with OXTS, else ``v_app``), by
    central difference of the velocities over the same window (m/s²)."""


def gt_kinematics(
    labels: dict[int, list[KittiTrackLabel]],
    geo: PinholeGeometry,
    oxts: Sequence[OxtsRecord] | None = None,
    half_window: int = 2,
    fps: float = KITTI_FPS,
) -> dict[tuple[int, int], GtKinematics]:
    """Per ``(frame, track_id)`` GT position and velocities (see module docstring)."""
    pos: dict[int, dict[int, NDArray[np.float64]]] = {}
    for f, labs in labels.items():
        for lab in labs:
            if lab.track_id < 0:
                continue
            g = geo.camera_to_ground(np.asarray(lab.location, dtype=np.float64))
            pos.setdefault(lab.track_id, {})[f] = np.array([g[0], g[2]])
    out: dict[tuple[int, int], GtKinematics] = {}
    for tid, by_f in pos.items():
        for f, p in by_f.items():
            lo = next((f - k for k in range(half_window, 0, -1) if f - k in by_f), None)
            hi = next((f + k for k in range(half_window, 0, -1) if f + k in by_f), None)
            if lo is None or hi is None:
                continue
            v = (by_f[hi] - by_f[lo]) * fps / (hi - lo)
            v_rel: tuple[float, float] | None = None
            v_abs: float | None = None
            wu: float | None = None
            vf: float | None = None
            if oxts is not None and f < len(oxts):
                o = oxts[f]
                vr = v + o.wu * np.array([-p[1], p[0]])
                v_rel = (float(vr[0]), float(vr[1]))
                v_abs = float(np.hypot(vr[0] - o.vl, vr[1] + o.vf))
                wu = float(o.wu)
                vf = float(o.vf)
            out[(f, tid)] = GtKinematics(
                float(p[0]), float(p[1]), (float(v[0]), float(v[1])), v_rel, v_abs, wu, vf
            )
    for (f, tid), k in list(out.items()):
        lo = next((f - j for j in range(half_window, 0, -1) if (f - j, tid) in out), None)
        hi = next((f + j for j in range(half_window, 0, -1) if (f + j, tid) in out), None)
        if lo is None or hi is None:
            continue
        r_lo, r_hi = _ref_velocity(out[(lo, tid)]), _ref_velocity(out[(hi, tid)])
        a = (r_hi - r_lo) * fps / (hi - lo)
        out[(f, tid)] = replace(k, a_ref_xz=(float(a[0]), float(a[1])))
    return out


def _ref_velocity(k: GtKinematics) -> NDArray[np.float64]:
    return np.asarray(k.v_rel_xz if k.v_rel_xz is not None else k.v_app_xz, dtype=np.float64)


def _is_scored(lab: KittiTrackLabel, types: Sequence[str]) -> bool:
    return (
        lab.obj_type in types
        and lab.occluded <= 1
        and lab.truncated <= 0.3
        and (lab.bbox[3] - lab.bbox[1]) >= 25.0
    )


DIAG_BINS_M: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0, 60.0)
DIAG_ACCEL_BINS: tuple[float, ...] = (-math.inf, -1.5, -0.5, 0.5, 1.5, math.inf)
DIAG_AGE_BINS_S: tuple[float, ...] = (1.0, 2.0, 4.0, math.inf)
TURN_YAW_RATE_RPS = 0.05
_DRIFT_HALF_WINDOW = 2


def _err_stats(e: NDArray[np.float64]) -> dict[str, Any]:
    if e.shape[0] == 0:
        return {"n": 0}
    n = np.hypot(e[:, 0], e[:, 1])
    return {
        "n": int(e.shape[0]),
        "rmse": float(np.sqrt(np.mean(n**2))),
        "bias_xz": [float(e[:, 0].mean()), float(e[:, 1].mean())],
        "p50": float(np.percentile(n, 50)),
        "p95": float(np.percentile(n, 95)),
    }


def _linfit(x: NDArray[np.float64], y: NDArray[np.float64]) -> dict[str, Any]:
    """``y ≈ intercept + slope·x`` by least squares on finite pairs. For ``dvz`` against
    the GT ``v_z``, a slope ``k − 1`` is the signature of a range scale error ``Z·k``
    (it turns ``dZ/dt`` into ``k·dZ/dt``); a slope against ``v_ego`` alone that of an
    ego-speed error."""
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3 or float(np.ptp(x[m])) == 0.0:
        return {"n": int(m.sum())}
    a = np.stack([np.ones(int(m.sum())), x[m]], axis=1)
    coef, *_ = np.linalg.lstsq(a, y[m], rcond=None)
    r = y[m] - a @ coef
    return {
        "n": int(m.sum()),
        "intercept": float(coef[0]),
        "slope": float(coef[1]),
        "resid_rms": float(np.sqrt(np.mean(r**2))),
    }


@dataclass
class TrackingKittiResult:
    n_frames: int = 0
    vel_err: list[tuple[float, float]] = field(default_factory=list)
    diag: list[tuple[float, ...]] = field(default_factory=list)
    """``(dvx, dvz, Z, |yaw rate|, gt_spread_x, gt_spread_z, ref_vz, ego_fwd, ref_az, age_s)`` per sample,
    any distance (``nan`` where unknown)."""
    turn_diag: list[tuple[float, ...]] = field(default_factory=list)
    """``(signed yaw rate, X, ex, ez, frame, gt_track_id)`` per ``diag`` sample (position
    error = tracker − GT, m)."""
    filter_diag: list[tuple[float, float, float]] = field(default_factory=list)
    """``(NIS of the last update, predicted σ of relative V_Z, n_updates)`` per ``diag``
    sample."""
    id_switches: int = 0
    n_gt_ids: int = 0
    n_matches: int = 0
    static_samples: int = 0
    static_labelled: int = 0
    cpu_ms: list[float] = field(default_factory=list)
    fusion_ms: list[float] = field(default_factory=list)
    tracker_ms: list[float] = field(default_factory=list)
    velocity_reference: str = "apparent"

    def diagnostics(self) -> dict[str, Any]:
        d = np.asarray(self.diag, dtype=np.float64).reshape(-1, 10)
        e, z, w, g = d[:, :2], d[:, 2], d[:, 3], d[:, 4:6]
        by_bin: dict[str, Any] = {}
        for lo, hi in zip(DIAG_BINS_M[:-1], DIAG_BINS_M[1:], strict=True):
            m = (z >= lo) & (z < hi)
            st = _err_stats(e[m])
            gm = g[m & np.all(np.isfinite(g), axis=1)]
            st["gt_spread_rmse"] = (
                float(np.sqrt(np.mean(np.sum(gm**2, axis=1)))) if gm.size else float("nan")
            )
            by_bin[f"{lo:.0f}-{hi:.0f}"] = st
        out: dict[str, Any] = {"by_distance_m": by_bin}
        if np.any(np.isfinite(w)):
            out["by_yaw_rate"] = {
                f"straight_lt_{TURN_YAW_RATE_RPS}": _err_stats(e[w < TURN_YAW_RATE_RPS]),
                f"turning_ge_{TURN_YAW_RATE_RPS}": _err_stats(e[w >= TURN_YAW_RATE_RPS]),
            }
        out["vz_err_vs_ref_vz"] = _linfit(d[:, 6], e[:, 1])
        out["vz_err_vs_ego_fwd"] = _linfit(d[:, 7], e[:, 1])
        ego = d[:, 7][np.isfinite(d[:, 7])]
        out["ego_fwd_mps_mean"] = float(ego.mean()) if ego.size else float("nan")
        # CV filter lag τ: e_vz ≈ −τ·a_z, so the slope against the GT acceleration is −τ.
        out["vz_err_vs_ref_az"] = _linfit(d[:, 8], e[:, 1])
        az = d[:, 8]
        out["by_ref_az_mps2"] = {
            f"{lo:g}_{hi:g}": _err_stats(e[(az >= lo) & (az < hi)])
            for lo, hi in zip(DIAG_ACCEL_BINS[:-1], DIAG_ACCEL_BINS[1:], strict=True)
        }
        age = d[:, 9]
        out["by_track_age_s"] = {
            f"{lo:g}_{hi:g}": _err_stats(e[(age >= lo) & (age < hi)])
            for lo, hi in zip(DIAG_AGE_BINS_S[:-1], DIAG_AGE_BINS_S[1:], strict=True)
        }
        if (
            self.turn_diag
            and len(self.turn_diag) == d.shape[0]
            and np.any(np.isfinite(np.asarray(self.turn_diag, dtype=np.float64)[:, 0]))
        ):
            out["turn"] = self._turn_diagnostics(d)
        if self.filter_diag and len(self.filter_diag) == d.shape[0]:
            out["consistency"] = self._consistency(d)
        return out

    def _consistency(self, d: NDArray[np.float64]) -> dict[str, Any]:
        """Filter self-consistency by track age: with a correct ``R``/``q`` the NIS has
        mean 2 (χ² 2 dof, 5 % above 5.99) and the predicted ``σ_VZ`` matches the actual
        ``V_Z`` RMSE (``ratio`` ≈ 1; > 1 ⇒ over-confident filter)."""
        f = np.asarray(self.filter_diag, dtype=np.float64).reshape(-1, 3)
        evz, age, nis, sig = d[:, 1], d[:, 9], f[:, 0], f[:, 1]

        def block(m: NDArray[np.bool_]) -> dict[str, Any]:
            if not m.any():
                return {"n": 0}
            nm = m & np.isfinite(nis)
            rmse = float(np.sqrt(np.mean(evz[m] ** 2)))
            s = float(np.sqrt(np.mean(sig[m] ** 2)))
            return {
                "n": int(m.sum()),
                "rmse_vz": rmse,
                "sigma_vz_pred_rms": s,
                "ratio": rmse / s if s > 0.0 else float("nan"),
                "n_nis": int(nm.sum()),
                "mean_nis": float(nis[nm].mean()) if nm.any() else float("nan"),
                "frac_nis_gt_5.99": float(np.mean(nis[nm] > 5.99)) if nm.any() else float("nan"),
                "mean_n_updates": float(f[m, 2].mean()),
            }

        out: dict[str, Any] = {"all": block(np.ones(age.shape[0], dtype=bool))}
        out["by_track_age_s"] = {
            f"{lo:g}_{hi:g}": block((age >= lo) & (age < hi))
            for lo, hi in zip(DIAG_AGE_BINS_S[:-1], DIAG_AGE_BINS_S[1:], strict=True)
        }
        return out

    def _turn_diagnostics(self, d: NDArray[np.float64]) -> dict[str, Any]:
        t = np.asarray(self.turn_diag, dtype=np.float64).reshape(-1, 6)
        e, z = d[:, :2], d[:, 2]
        w, x, ez = t[:, 0], t[:, 1], t[:, 3]
        drift = _range_drift_rate(t[:, 4], t[:, 5], ez)
        out: dict[str, Any] = {
            "vz_err_vs_omega_x": _linfit(w * x, e[:, 1]),
            "vx_err_vs_omega_z": _linfit(-w * z, e[:, 0]),
        }
        groups = {
            "left": w >= TURN_YAW_RATE_RPS,
            "straight": np.abs(w) < TURN_YAW_RATE_RPS,
            "right": w <= -TURN_YAW_RATE_RPS,
        }
        by_dir: dict[str, Any] = {}
        for name, m in groups.items():
            st = _err_stats(e[m])
            if st["n"]:
                zm = z[m]
                dm = drift[m]
                fin = np.isfinite(dm)
                st["mean_ez_m"] = float(ez[m].mean())
                st["mean_ez_rel"] = float(np.mean(ez[m] / np.maximum(zm, 1e-3)))
                st["mean_x_m"] = float(x[m].mean())
                st["n_drift"] = int(fin.sum())
                st["mean_dez_dt_mps"] = float(dm[fin].mean()) if fin.any() else float("nan")
                st["vz_err_minus_range_drift"] = (
                    float(np.mean(e[m][fin, 1] - dm[fin])) if fin.any() else float("nan")
                )
            by_dir[name] = st
        out["by_turn_direction"] = by_dir
        return out

    def to_dict(self) -> dict[str, Any]:
        e = np.asarray(self.vel_err, dtype=np.float64).reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.sum(e**2, axis=1)))) if e.size else float("nan")
        rmse_z = float(np.sqrt(np.mean(e[:, 1] ** 2))) if e.size else float("nan")

        def pct(v: list[float]) -> list[float]:
            a = np.asarray(v, dtype=np.float64)
            return [float(x) for x in np.percentile(a, [50, 95, 99])] if a.size else []

        return {
            "n_frames": self.n_frames,
            "velocity_reference": self.velocity_reference,
            "n_vel_samples": int(e.shape[0]),
            "rmse_vel_rel_mps": rmse,
            "rmse_vz_rel_mps": rmse_z,
            "id_switches": self.id_switches,
            "idsw_per_100_matches": (
                100.0 * self.id_switches / self.n_matches if self.n_matches else float("nan")
            ),
            "n_gt_ids": self.n_gt_ids,
            "n_matches": self.n_matches,
            "static_samples": self.static_samples,
            "static_frac": (
                self.static_labelled / self.static_samples if self.static_samples else float("nan")
            ),
            "cpu_ms_p50_p95_p99": pct(self.cpu_ms),
            "fusion_ms_p50_p95_p99": pct(self.fusion_ms),
            "tracker_ms_p50_p95_p99": pct(self.tracker_ms),
            "diagnostics": self.diagnostics(),
        }


def _range_drift_rate(
    frame: NDArray[np.float64],
    tid: NDArray[np.float64],
    ez: NDArray[np.float64],
    fps: float = KITTI_FPS,
    half_window: int = _DRIFT_HALF_WINDOW,
) -> NDArray[np.float64]:
    """``d(ez)/dt`` per sample by central difference over the same GT track (``nan``
    without neighbours on both sides)."""
    idx = {(int(f), int(k)): i for i, (f, k) in enumerate(zip(frame, tid, strict=True))}
    out = np.full(ez.shape[0], np.nan)
    for i, (f, k) in enumerate(zip(frame.astype(int), tid.astype(int), strict=True)):
        lo = next((f - j for j in range(half_window, 0, -1) if (f - j, k) in idx), None)
        hi = next((f + j for j in range(half_window, 0, -1) if (f + j, k) in idx), None)
        if lo is not None and hi is not None:
            out[i] = (ez[idx[(hi, k)]] - ez[idx[(lo, k)]]) * fps / (hi - lo)
    return out


def merge_results(results: Sequence[TrackingKittiResult]) -> TrackingKittiResult:
    """Pool the per-sample diagnostics of several sequences (``diagnostics()`` only).

    Track ids are offset per sequence so the range-drift differences stay within one
    sequence."""
    out = TrackingKittiResult()
    offset = 0.0
    for r in results:
        out.diag.extend(r.diag)
        if len(r.turn_diag) == len(r.diag):
            out.turn_diag.extend((*t[:5], t[5] + offset) for t in r.turn_diag)
            offset += max((t[5] for t in r.turn_diag), default=0.0) + 1.0
        out.vel_err.extend(r.vel_err)
        out.filter_diag.extend(r.filter_diag if len(r.filter_diag) == len(r.diag) else [])
    if len(out.turn_diag) != len(out.diag):
        out.turn_diag = []
    if len(out.filter_diag) != len(out.diag):
        out.filter_diag = []
    return out


def run_tracking_kitti(
    frames: Iterable[FrameStamped],
    labels: dict[int, list[KittiTrackLabel]],
    stage: MetricFusionStage,
    tracker: Tracker3D,
    gt: dict[tuple[int, int], GtKinematics],
    depth_provider: DepthProvider | None = None,
    box_provider: BoxProvider | None = None,
    score_types: Sequence[str] = ("Car",),
    bin_m: tuple[float, float] = (0.0, 30.0),
    min_age_s: float = 1.0,
    static_age_s: float = 0.5,
    static_below_mps: float = 0.5,
    iou_threshold: float = 0.5,
    gt_alt: dict[tuple[int, int], GtKinematics] | None = None,
) -> TrackingKittiResult:
    use_rel = any(k.v_rel_xz is not None for k in gt.values())
    res = TrackingKittiResult(velocity_reference="translational" if use_rel else "apparent")
    last_tid: dict[int, int] = {}
    for fs in frames:
        labs = [lab for lab in labels.get(fs.frame_id, []) if lab.obj_type in KITTI_TO_FUSION_CLASS]
        if box_provider is None:
            boxes, classes = labels_to_boxes(labs)
            scores = np.ones(boxes.shape[0])
            det_to_lab = dict(enumerate(labs))
        else:
            boxes, scores, classes = box_provider(fs)
            gtb = np.array([lab.bbox for lab in labs], dtype=np.float64).reshape(-1, 4)
            assign, _ = match_greedy(boxes, scores, gtb, iou_threshold)
            det_to_lab = {i: labs[j] for i, j in enumerate(assign.tolist()) if j >= 0}
        depth = depth_provider(fs) if depth_provider is not None else None
        t0 = time.perf_counter()
        out = stage.process(fs.frame_id, fs.t_capture_ns, boxes, classes, depth)
        meas = [measurement_from_fusion(m) for m in out.measurements]
        t1 = time.perf_counter()
        tracks = tracker.step(fs.t_capture_ns, boxes, scores, classes, meas)
        t2 = time.perf_counter()
        res.fusion_ms.append((t1 - t0) * 1e3)
        res.tracker_ms.append((t2 - t1) * 1e3)
        res.cpu_ms.append((t2 - t0) * 1e3)
        res.n_frames += 1
        for tr in tracks:
            lab = det_to_lab.get(tr.det_index)
            if lab is None or not _is_scored(lab, score_types):
                continue
            res.n_matches += 1
            if lab.track_id in last_tid and last_tid[lab.track_id] != tr.track_id:
                res.id_switches += 1
            last_tid[lab.track_id] = tr.track_id
            k = gt.get((fs.frame_id, lab.track_id))
            if k is None:
                continue
            if (
                k.v_abs_mps is not None
                and k.v_abs_mps < static_below_mps
                and tr.age_s >= static_age_s
            ):
                res.static_samples += 1
                res.static_labelled += int(tr.motion is MotionState.STATIC)
            ref = k.v_rel_xz if use_rel else k.v_app_xz
            if ref is None or tr.age_s < min_age_s:
                continue
            dv = tr.velocity_rel_xz - np.asarray(ref, dtype=np.float64)
            spread = (float("nan"), float("nan"))
            ka = gt_alt.get((fs.frame_id, lab.track_id)) if gt_alt is not None else None
            ref_a = None if ka is None else (ka.v_rel_xz if use_rel else ka.v_app_xz)
            if ref_a is not None:
                spread = (ref[0] - ref_a[0], ref[1] - ref_a[1])
            w = abs(k.yaw_rate_rps) if k.yaw_rate_rps is not None else float("nan")
            ego_fwd = k.ego_fwd_mps if k.ego_fwd_mps is not None else float("nan")
            az = k.a_ref_xz[1] if k.a_ref_xz is not None else float("nan")
            res.diag.append(
                (
                    float(dv[0]),
                    float(dv[1]),
                    k.z_m,
                    w,
                    *spread,
                    float(ref[1]),
                    ego_fwd,
                    az,
                    tr.age_s,
                )
            )
            res.turn_diag.append(
                (
                    float(k.yaw_rate_rps) if k.yaw_rate_rps is not None else float("nan"),
                    k.x_m,
                    float(tr.position_xz[0]) - k.x_m,
                    float(tr.position_xz[1]) - k.z_m,
                    float(fs.frame_id),
                    float(lab.track_id),
                )
            )
            res.filter_diag.append(
                (tr.last_nis, float(np.sqrt(tr.cov_vel_rel[1, 1])), float(tr.n_updates))
            )
            if bin_m[0] <= k.z_m <= bin_m[1]:
                res.vel_err.append((float(dv[0]), float(dv[1])))
    res.n_gt_ids = len(last_tid)
    return res
