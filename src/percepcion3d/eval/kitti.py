"""KITTI ground truth loading and per-distance-bin depth metrics.

Per-object depth GT comes from the KITTI *tracking* labels (3D ``location``
in the camera frame, one line per object per frame), which avoids projecting
LiDAR: the ``z`` of the 3D box centre is the optical depth of the object and
its ``track_id`` gives tracking GT for free.

Label line format (``label_02/XXXX.txt``)::

    frame track_id type truncated occluded alpha
    bbox_left bbox_top bbox_right bbox_bottom
    height width length x y z rotation_y [score]
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

DEFAULT_DISTANCE_BINS_M: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0, 50.0, 80.0)


@dataclass(frozen=True)
class KittiTrackLabel:
    frame: int
    track_id: int
    obj_type: str
    truncated: float
    occluded: int
    alpha: float
    bbox: tuple[float, float, float, float]
    """``(left, top, right, bottom)`` in pixels."""
    dims_hwl: tuple[float, float, float]
    location: tuple[float, float, float]
    """3D box centre ``(x, y, z)`` in the rectified camera frame (metres)."""
    rotation_y: float

    @property
    def depth_m(self) -> float:
        return self.location[2]

    @property
    def bottom_center_camera(self) -> NDArray[np.float64]:
        """Ground-contact point below the box centre (KITTI ``y`` is the box *bottom*)."""
        return np.array(self.location, dtype=np.float64)


def parse_kitti_tracking_line(line: str) -> KittiTrackLabel:
    f = line.split()
    if len(f) < 17:
        raise ValueError(f"KITTI tracking label needs >= 17 fields, got {len(f)}: {line!r}")
    return KittiTrackLabel(
        frame=int(f[0]),
        track_id=int(f[1]),
        obj_type=f[2],
        truncated=float(f[3]),
        occluded=int(float(f[4])),
        alpha=float(f[5]),
        bbox=(float(f[6]), float(f[7]), float(f[8]), float(f[9])),
        dims_hwl=(float(f[10]), float(f[11]), float(f[12])),
        location=(float(f[13]), float(f[14]), float(f[15])),
        rotation_y=float(f[16]),
    )


def load_kitti_tracking_labels(
    path: Path | str,
    keep_types: Iterable[str] | None = ("Car", "Van", "Truck", "Pedestrian", "Cyclist"),
) -> dict[int, list[KittiTrackLabel]]:
    """Load a tracking label file grouped by frame; ``DontCare``/unwanted types dropped."""
    keep = set(keep_types) if keep_types is not None else None
    by_frame: dict[int, list[KittiTrackLabel]] = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            lab = parse_kitti_tracking_line(line)
            if keep is not None and lab.obj_type not in keep:
                continue
            by_frame[lab.frame].append(lab)
    return dict(by_frame)


@dataclass(frozen=True)
class DepthBinMetrics:
    lo_m: float
    hi_m: float
    count: int
    abs_rel: float
    rmse_m: float
    bias_m: float
    """Mean of ``pred - gt`` (positive = over-estimation)."""


def depth_metrics_by_bin(
    pred_m: NDArray[np.float64] | Sequence[float],
    gt_m: NDArray[np.float64] | Sequence[float],
    bins_m: Sequence[float] = DEFAULT_DISTANCE_BINS_M,
) -> list[DepthBinMetrics]:
    """AbsRel, RMSE and bias per GT-distance bin; ``nan`` predictions are excluded.

    Bins are ``[bins[i], bins[i+1])``; samples outside ``[bins[0], bins[-1])`` are ignored.
    """
    pred = np.asarray(pred_m, dtype=np.float64).ravel()
    gt = np.asarray(gt_m, dtype=np.float64).ravel()
    if pred.shape != gt.shape:
        raise ValueError(f"pred {pred.shape} and gt {gt.shape} must have the same shape")
    if len(bins_m) < 2 or np.any(np.diff(np.asarray(bins_m)) <= 0):
        raise ValueError("bins_m must be strictly increasing with >= 2 edges")

    ok = np.isfinite(pred) & np.isfinite(gt) & (gt > 0.0)
    pred, gt = pred[ok], gt[ok]
    out: list[DepthBinMetrics] = []
    for lo, hi in zip(bins_m[:-1], bins_m[1:], strict=True):
        m = (gt >= lo) & (gt < hi)
        n = int(m.sum())
        if n == 0:
            out.append(DepthBinMetrics(lo, hi, 0, float("nan"), float("nan"), float("nan")))
            continue
        err = pred[m] - gt[m]
        out.append(
            DepthBinMetrics(
                lo_m=lo,
                hi_m=hi,
                count=n,
                abs_rel=float(np.mean(np.abs(err) / gt[m])),
                rmse_m=float(np.sqrt(np.mean(err**2))),
                bias_m=float(np.mean(err)),
            )
        )
    return out


def format_depth_metrics(rows: Sequence[DepthBinMetrics]) -> str:
    header = f"{'bin [m]':<14}{'n':>7}{'AbsRel':>9}{'RMSE[m]':>10}{'bias[m]':>10}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{f'{r.lo_m:g}-{r.hi_m:g}':<14}{r.count:>7}{r.abs_rel:>9.3f}{r.rmse_m:>10.3f}{r.bias_m:>10.3f}"
        )
    return "\n".join(lines)
