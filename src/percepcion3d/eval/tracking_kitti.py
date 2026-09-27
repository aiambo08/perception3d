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

ID switches are counted per GT id over every matched frame. The 3D stage never
changes 2D identities (see :mod:`percepcion3d.tracking.tracker3d`), so the ByteTrack
reference of the DoD is this same count; the number is reported to compare
configurations (``byte_track`` thresholds, detector vs. GT boxes).
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
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
            if oxts is not None and f < len(oxts):
                o = oxts[f]
                vr = v + o.wu * np.array([-p[1], p[0]])
                v_rel = (float(vr[0]), float(vr[1]))
                v_abs = float(np.hypot(vr[0] - o.vl, vr[1] + o.vf))
            out[(f, tid)] = GtKinematics(
                float(p[0]), float(p[1]), (float(v[0]), float(v[1])), v_rel, v_abs
            )
    return out


def _is_scored(lab: KittiTrackLabel, types: Sequence[str]) -> bool:
    return (
        lab.obj_type in types
        and lab.occluded <= 1
        and lab.truncated <= 0.3
        and (lab.bbox[3] - lab.bbox[1]) >= 25.0
    )


@dataclass
class TrackingKittiResult:
    n_frames: int = 0
    vel_err: list[tuple[float, float]] = field(default_factory=list)
    id_switches: int = 0
    n_gt_ids: int = 0
    n_matches: int = 0
    static_samples: int = 0
    static_labelled: int = 0
    cpu_ms: list[float] = field(default_factory=list)
    velocity_reference: str = "apparent"

    def to_dict(self) -> dict[str, Any]:
        e = np.asarray(self.vel_err, dtype=np.float64).reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.sum(e**2, axis=1)))) if e.size else float("nan")
        rmse_z = float(np.sqrt(np.mean(e[:, 1] ** 2))) if e.size else float("nan")
        cpu = np.asarray(self.cpu_ms, dtype=np.float64)
        return {
            "n_frames": self.n_frames,
            "velocity_reference": self.velocity_reference,
            "n_vel_samples": int(e.shape[0]),
            "rmse_vel_rel_mps": rmse,
            "rmse_vz_rel_mps": rmse_z,
            "id_switches": self.id_switches,
            "n_gt_ids": self.n_gt_ids,
            "n_matches": self.n_matches,
            "static_samples": self.static_samples,
            "static_frac": (
                self.static_labelled / self.static_samples if self.static_samples else float("nan")
            ),
            "cpu_ms_p50_p95_p99": (
                [float(x) for x in np.percentile(cpu, [50, 95, 99])] if cpu.size else []
            ),
        }


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
        tracks = tracker.step(fs.t_capture_ns, boxes, scores, classes, meas)
        res.cpu_ms.append((time.perf_counter() - t0) * 1e3)
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
            if ref is None or tr.age_s < min_age_s or not (bin_m[0] <= k.z_m <= bin_m[1]):
                continue
            dv = tr.velocity_rel_xz - np.asarray(ref, dtype=np.float64)
            res.vel_err.append((float(dv[0]), float(dv[1])))
    res.n_gt_ids = len(last_tid)
    return res
