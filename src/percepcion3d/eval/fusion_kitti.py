"""F4 on KITTI tracking: fused depth vs. 3D labels, AbsRel per range bin.

The tracking benchmark (``training/{image_02,label_02,calib}``) is used because its
labels carry the 3D box centre in the rectified camera frame; the raw drives only
have LiDAR. Two ground truths are derived from each label:

* **near face** — depth of the box face closest to the camera along the optical
  axis, ``z − (l·|cos r_y| + w·|sin r_y|)/2``. This is what the contact point (the
  lowest visible wheel/corner) and the network median over the visible surface
  measure, so it is the primary reference for ``Measurement3D.z_cam_m``.
* **centre** — ``z`` itself, compared with ``z_cam_m + center_offset_m`` (the class
  length prior); it shows what a downstream consumer of object centres gets.

Boxes come either from the labels (``gt`` mode: isolates the metric fusion from
detector recall, all labelled objects are fed so occlusion masking is exercised)
or from the detector matched to the labels at IoU ≥ 0.5. Only non-truncated Cars
(``truncated == 0``, ``occluded ≤ max_occluded``, height ≥ 25 px) are scored, as in
the F4 DoD; other classes are still fed as boxes.

The depth map is produced by an injected callable so the same evaluation runs with a
TensorRT engine on the target GPU, with maps pre-computed to disk, or with no network at
all (geometry + height prior only — the CPU-only baseline this module is tested with).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from percepcion3d.camera.calibration import CameraIntrinsics
from percepcion3d.depth.depth_trt import DepthMap
from percepcion3d.depth.fusion import FusionFlag, Measurement3D, MetricFusionStage
from percepcion3d.eval.detection import match_greedy
from percepcion3d.eval.kitti import (
    DepthBinMetrics,
    KittiTrackLabel,
    depth_metrics_by_bin,
    format_depth_metrics,
    load_kitti_tracking_labels,
)
from percepcion3d.io.sources import ImageSequenceSource
from percepcion3d.runtime.buffer import FrameStamped

#: KITTI type → class name understood by ``configs/fusion.yaml`` (aliases included).
KITTI_TO_FUSION_CLASS: dict[str, str] = {
    "Car": "car",
    "Van": "van",
    "Truck": "truck",
    "Pedestrian": "pedestrian",
    "Cyclist": "cyclist",
    "Person_sitting": "pedestrian",
    "Tram": "bus",
}

#: DoD bins: AbsRel ≤ 10 % at 0–30 m, ≤ 20 % at 30–60 m.
DOD_BINS_M: tuple[float, ...] = (0.0, 30.0, 60.0)
DOD_ABS_REL: tuple[float, ...] = (0.10, 0.20)
KITTI_TRACKING_FPS = 10.0


# ----------------------------------------------------------------------------
# Dataset layout
# ----------------------------------------------------------------------------


def parse_kitti_tracking_calib(path: Path | str, cam_idx: int = 2) -> CameraIntrinsics:
    """``P2: fx 0 cx tx 0 fy cy ty 0 0 1 0`` from a tracking ``calib/<seq>.txt``."""
    key = f"P{cam_idx}"
    with Path(path).open(encoding="utf-8") as fh:
        for raw in fh:
            if not raw.startswith(key + ":"):
                continue
            p = np.array(raw.split(":", 1)[1].split(), dtype=np.float64).reshape(3, 4)
            return CameraIntrinsics(
                fx=float(p[0, 0]),
                fy=float(p[1, 1]),
                cx=float(p[0, 2]),
                cy=float(p[1, 2]),
                width=1242,
                height=375,
            )
    raise KeyError(f"{key} not found in {path}")


@dataclass
class KittiTrackingSequence:
    """``<root>/{image_02/<seq>/*.png, label_02/<seq>.txt, calib/<seq>.txt}``."""

    root: Path
    seq: str
    cam_idx: int = 2
    max_frames: int | None = None

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.img_dir = self.root / f"image_{self.cam_idx:02d}" / self.seq
        self.label_file = self.root / f"label_{self.cam_idx:02d}" / f"{self.seq}.txt"
        self.calib_file = self.root / "calib" / f"{self.seq}.txt"
        for p in (self.img_dir, self.label_file, self.calib_file):
            if not p.exists():
                raise FileNotFoundError(p)

    def intrinsics(self) -> CameraIntrinsics:
        return parse_kitti_tracking_calib(self.calib_file, self.cam_idx)

    def labels(self) -> dict[int, list[KittiTrackLabel]]:
        return load_kitti_tracking_labels(self.label_file, keep_types=None)

    def source(self) -> ImageSequenceSource:
        paths = sorted(self.img_dir.glob("*.png"))
        if self.max_frames is not None:
            paths = paths[: self.max_frames]
        t_ns = [int(round(i * 1e9 / KITTI_TRACKING_FPS)) for i in range(len(paths))]
        return ImageSequenceSource(paths, t_ns, intrinsics=self.intrinsics())


# ----------------------------------------------------------------------------
# Ground truth
# ----------------------------------------------------------------------------


def gt_near_face_depth_m(lab: KittiTrackLabel) -> float:
    """``z`` minus the half-extent of the box along the optical axis.

    KITTI's ``rotation_y`` turns the object length axis from camera ``X`` about ``Y``:
    ``R_y·(1,0,0) = (cos r_y, 0, −sin r_y)``, so a car driving along the road
    (``r_y ≈ ±π/2``) extends ``l/2`` in depth and one seen side-on (``r_y ≈ 0``) ``w/2``.
    """
    _, w, length = lab.dims_hwl
    extent = 0.5 * (length * abs(math.sin(lab.rotation_y)) + w * abs(math.cos(lab.rotation_y)))
    return lab.depth_m - extent


def is_scored(
    lab: KittiTrackLabel,
    types: Sequence[str] = ("Car",),
    max_truncated: float = 0.0,
    max_occluded: int = 1,
    min_height_px: float = 25.0,
) -> bool:
    return (
        lab.obj_type in types
        and lab.truncated <= max_truncated
        and lab.occluded <= max_occluded
        and (lab.bbox[3] - lab.bbox[1]) >= min_height_px
        and lab.depth_m > 0.0
    )


def labels_to_boxes(
    labels: Sequence[KittiTrackLabel],
) -> tuple[NDArray[np.float64], list[str]]:
    """All labels with a known fusion class as detector-style boxes (``DontCare`` dropped)."""
    keep = [lab for lab in labels if lab.obj_type in KITTI_TO_FUSION_CLASS]
    boxes = np.array([lab.bbox for lab in keep], dtype=np.float64).reshape(-1, 4)
    return boxes, [KITTI_TO_FUSION_CLASS[lab.obj_type] for lab in keep]


# ----------------------------------------------------------------------------
# Accumulation + DoD
# ----------------------------------------------------------------------------


@dataclass
class ScoredSample:
    frame: int
    track_id: int
    gt_near_m: float
    gt_center_m: float
    z_cam_m: float
    z_center_m: float
    z_ground_m: float
    z_net_m: float
    z_height_m: float
    sigma_z_m: float
    flags: int


@dataclass
class FusionKittiResult:
    samples: list[ScoredSample] = field(default_factory=list)
    n_frames: int = 0
    n_frames_with_depth: int = 0
    n_boxes_fed: int = 0
    n_scored_labels: int = 0
    pitch_final_deg: float = float("nan")
    affine_s: float = float("nan")
    affine_t: float = float("nan")
    cpu_ms: list[float] = field(default_factory=list)

    def _cols(self) -> dict[str, NDArray[np.float64]]:
        s = self.samples
        return {
            "gt_near": np.array([x.gt_near_m for x in s], dtype=np.float64),
            "gt_center": np.array([x.gt_center_m for x in s], dtype=np.float64),
            "fused": np.array([x.z_cam_m for x in s], dtype=np.float64),
            "fused_center": np.array([x.z_center_m for x in s], dtype=np.float64),
            "ground": np.array([x.z_ground_m for x in s], dtype=np.float64),
            "net": np.array([x.z_net_m for x in s], dtype=np.float64),
            "height": np.array([x.z_height_m for x in s], dtype=np.float64),
        }

    def metrics(self, bins_m: Sequence[float] = DOD_BINS_M) -> dict[str, list[DepthBinMetrics]]:
        c = self._cols()
        if c["gt_near"].shape[0] == 0:
            return {}
        return {
            "fused (near face)": depth_metrics_by_bin(c["fused"], c["gt_near"], bins_m),
            "fused + L/2 (centre)": depth_metrics_by_bin(c["fused_center"], c["gt_center"], bins_m),
            "ground only": depth_metrics_by_bin(c["ground"], c["gt_near"], bins_m),
            "net only": depth_metrics_by_bin(c["net"], c["gt_near"], bins_m),
            "height only": depth_metrics_by_bin(c["height"], c["gt_near"], bins_m),
        }

    def dod(self) -> dict[str, Any]:
        """Coverage-aware DoD: AbsRel of the fused near-face depth per bin, valid fraction."""
        c = self._cols()
        out: dict[str, Any] = {"n_scored": int(c["gt_near"].shape[0])}
        if out["n_scored"] == 0:
            out["pass"] = False
            return out
        rows = depth_metrics_by_bin(c["fused"], c["gt_near"], DOD_BINS_M)
        ok_all = True
        for r, thr in zip(rows, DOD_ABS_REL, strict=True):
            key = f"{r.lo_m:g}_{r.hi_m:g}m"
            in_bin = (c["gt_near"] >= r.lo_m) & (c["gt_near"] < r.hi_m)
            n_bin = int(in_bin.sum())
            valid = float(np.isfinite(c["fused"][in_bin]).mean()) if n_bin else float("nan")
            passed = bool(n_bin > 0 and r.abs_rel <= thr)
            out[f"abs_rel_{key}"] = r.abs_rel
            out[f"n_{key}"] = n_bin
            out[f"valid_frac_{key}"] = valid
            out[f"pass_{key}"] = passed
            ok_all &= passed
        out["pass"] = ok_all
        if self.cpu_ms:
            p = np.percentile(np.asarray(self.cpu_ms), [50, 95, 99])
            out["cpu_ms_p50_p95_p99"] = [float(x) for x in p]
        return out

    def format(self) -> str:
        lines = [
            f"frames {self.n_frames} (with depth {self.n_frames_with_depth}) · boxes fed "
            f"{self.n_boxes_fed} · scored Car labels {self.n_scored_labels} · "
            f"pitch {self.pitch_final_deg:.2f}° · affine s={self.affine_s:.3f} t={self.affine_t:.3f}"
        ]
        for name, rows in self.metrics().items():
            lines.append(f"\n[{name}]")
            lines.append(format_depth_metrics(rows))
        return "\n".join(lines)


DepthProvider = Callable[[FrameStamped], DepthMap | None]
BoxProvider = Callable[[FrameStamped], tuple[NDArray[np.float64], NDArray[np.float64], list[str]]]
"""``frame → (boxes_xyxy, scores, class_names)``; used instead of the labels when given."""


def _score(
    res: FusionKittiResult,
    lab: KittiTrackLabel,
    m: Measurement3D,
) -> None:
    res.samples.append(
        ScoredSample(
            frame=lab.frame,
            track_id=lab.track_id,
            gt_near_m=gt_near_face_depth_m(lab),
            gt_center_m=lab.depth_m,
            z_cam_m=m.z_cam_m,
            z_center_m=m.z_cam_m + m.center_offset_m,
            z_ground_m=m.z_ground_m,
            z_net_m=m.z_net_m,
            z_height_m=m.z_height_m,
            sigma_z_m=m.sigma_z_m,
            flags=int(m.flags.value),
        )
    )


def run_fusion_kitti(
    frames: Iterable[FrameStamped],
    labels: dict[int, list[KittiTrackLabel]],
    stage: MetricFusionStage,
    depth_provider: DepthProvider | None,
    box_provider: BoxProvider | None = None,
    iou_threshold: float = 0.5,
    max_occluded: int = 1,
    timer_ms: Callable[[], float] | None = None,
) -> FusionKittiResult:
    """Feed every frame through ``stage`` and score non-truncated Cars.

    ``timer_ms`` (e.g. ``lambda: time.perf_counter() * 1e3``) measures the
    solver + fusion CPU time per frame; the depth provider is timed separately by its
    own :class:`StageTimer` when it is an engine.
    """
    res = FusionKittiResult()
    for fs in frames:
        labs = labels.get(fs.frame_id, [])
        scored = [lab for lab in labs if is_scored(lab, max_occluded=max_occluded)]
        res.n_scored_labels += len(scored)
        depth = depth_provider(fs) if depth_provider is not None else None
        if depth is not None:
            res.n_frames_with_depth += 1
        if box_provider is None:
            boxes, classes = labels_to_boxes(labs)
            fed = [lab for lab in labs if lab.obj_type in KITTI_TO_FUSION_CLASS]
            score_idx = [
                (i, lab) for i, lab in enumerate(fed) if is_scored(lab, max_occluded=max_occluded)
            ]
        else:
            boxes, scores, classes = box_provider(fs)
            gt_boxes = np.array([lab.bbox for lab in scored], dtype=np.float64).reshape(-1, 4)
            assign, _ = match_greedy(boxes, scores, gt_boxes, iou_threshold)
            score_idx = [(i, scored[j]) for i, j in enumerate(assign.tolist()) if j >= 0]
        t0 = timer_ms() if timer_ms is not None else 0.0
        out = stage.process(fs.frame_id, fs.t_capture_ns, boxes, classes, depth)
        if timer_ms is not None:
            res.cpu_ms.append(timer_ms() - t0)
        res.n_frames += 1
        res.n_boxes_fed += boxes.shape[0]
        for i, lab in score_idx:
            _score(res, lab, out.measurements[i])
        if out.affine is not None:
            res.affine_s, res.affine_t = out.affine.s, out.affine.t
        res.pitch_final_deg = float(np.rad2deg(out.pitch.pitch_rad))
    return res


def flag_histogram(res: FusionKittiResult) -> dict[str, int]:
    """How often each :class:`FusionFlag` was raised on scored samples."""
    out: dict[str, int] = {}
    for f in FusionFlag:
        n = sum(1 for s in res.samples if s.flags & f.value)
        if n:
            out[f.name or str(f.value)] = n
    return out
