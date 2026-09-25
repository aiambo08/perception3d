"""Export/health check of a relative depth engine against LiDAR (F3 DoD).

The relative model outputs ``d̂ = s·(1/Z) + t`` with unknown ``(s, t)``, so an
absolute metric is meaningless here; what *is* invariant to the affine
ambiguity is the **rank** of the pixels: Spearman's ρ between ``d̂`` and
``1/Z_lidar`` must stay ≥ 0.95 on road pixels (smooth surface, dense LiDAR,
no thin structures) if the ONNX surgery, FP16 build and resize are healthy. A
drop below that threshold means a broken export (wrong channel order or
normalisation, FP16 overflow in an attention block), **not** a metric error.

As a secondary diagnostic, :func:`affine_abs_rel` fits ``(s, t)`` robustly
(Theil–Sen on a subsample) and reports AbsRel of the recovered depth; this is a
lower bound on what the F4 metric fusion can achieve on that frame.

Everything here is CPU/NumPy; the caller provides the sampled predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from percepcion3d.depth.depth_trt import DepthMap
from percepcion3d.eval.lidar import ProjectedLidar


def road_mask(
    image_hw: tuple[int, int],
    horizon_v: float,
    boxes_xyxy: NDArray[np.floating[Any]] | None = None,
    horizon_margin_px: float = 20.0,
    bottom_margin_px: float = 0.0,
) -> NDArray[np.bool_]:
    """Pixels below the horizon (+margin), above the bottom margin, outside detections."""
    h, w = image_hw
    mask = np.zeros((h, w), dtype=bool)
    v0 = int(np.clip(np.ceil(horizon_v + horizon_margin_px), 0, h))
    v1 = int(np.clip(np.floor(h - bottom_margin_px), 0, h))
    mask[v0:v1, :] = True
    if boxes_xyxy is not None:
        for x0, y0, x1, y1 in np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4):
            c0, c1 = int(np.clip(np.floor(x0), 0, w)), int(np.clip(np.ceil(x1), 0, w))
            r0, r1 = int(np.clip(np.floor(y0), 0, h)), int(np.clip(np.ceil(y1), 0, h))
            mask[r0:r1, c0:c1] = False
    return mask


@dataclass(frozen=True)
class DepthSanity:
    n: int
    spearman: float
    """ρ between predicted values and ``1/Z``; NaN when ``n < 3``."""
    affine_scale: float
    affine_offset: float
    abs_rel: float
    """AbsRel of ``1/((d̂ − t)/s)`` vs LiDAR after the robust affine fit; NaN if degenerate."""


def spearman_vs_inverse_depth(
    pred: NDArray[np.floating[Any]], z_m: NDArray[np.floating[Any]]
) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    inv = 1.0 / np.asarray(z_m, dtype=np.float64)
    if pred.shape[0] < 3 or np.ptp(pred) == 0.0 or np.ptp(inv) == 0.0:
        return float("nan")
    rho = stats.spearmanr(pred, inv).statistic
    return float(rho)


def affine_abs_rel(
    pred: NDArray[np.floating[Any]],
    z_m: NDArray[np.floating[Any]],
    max_points: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Theil–Sen fit of ``pred ≈ s·(1/z) + t``; returns ``(s, t, AbsRel)``."""
    pred = np.asarray(pred, dtype=np.float64)
    z = np.asarray(z_m, dtype=np.float64)
    inv = 1.0 / z
    n = pred.shape[0]
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    if n > max_points:
        idx = np.random.default_rng(seed).choice(n, size=max_points, replace=False)
        fit_x, fit_y = inv[idx], pred[idx]
    else:
        fit_x, fit_y = inv, pred
    slope, intercept = stats.theilslopes(fit_y, fit_x)[:2]
    s, t = float(slope), float(intercept)
    if not np.isfinite(s) or s <= 0:
        return s, t, float("nan")
    inv_hat = (pred - t) / s
    valid = inv_hat > 1e-6
    if not np.any(valid):
        return s, t, float("nan")
    z_hat = 1.0 / inv_hat[valid]
    abs_rel = float(np.mean(np.abs(z_hat - z[valid]) / z[valid]))
    return s, t, abs_rel


def evaluate_depth_map(
    depth: DepthMap,
    lidar: ProjectedLidar,
    mask: NDArray[np.bool_] | None = None,
    z_max_m: float = 80.0,
) -> DepthSanity:
    """Sample ``depth`` at the LiDAR pixels (frame coords) and compare ranks with ``1/Z``."""
    keep = (lidar.z > 0) & (lidar.z <= z_max_m)
    if mask is not None:
        if mask.shape != (depth.resize.src_h, depth.resize.src_w):
            raise ValueError(
                f"mask {mask.shape} must match the frame {(depth.resize.src_h, depth.resize.src_w)}"
            )
        row = np.clip(np.floor(lidar.v).astype(np.intp), 0, mask.shape[0] - 1)
        col = np.clip(np.floor(lidar.u).astype(np.intp), 0, mask.shape[1] - 1)
        keep &= mask[row, col]
    sel = lidar.select(keep)
    pred = depth.inverse_depth()[depth.resize.frame_to_index(sel.u, sel.v)].astype(np.float64)
    finite = np.isfinite(pred)
    pred, z = pred[finite], sel.z[finite]
    rho = spearman_vs_inverse_depth(pred, z)
    s, t, abs_rel = affine_abs_rel(pred, z)
    return DepthSanity(
        n=int(pred.shape[0]), spearman=rho, affine_scale=s, affine_offset=t, abs_rel=abs_rel
    )
