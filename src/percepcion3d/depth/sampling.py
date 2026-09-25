"""Robust per-box statistics of the inverse-depth map (F4).

A detection box contains the object *and* background (road in front, sky,
buildings behind). The mean is useless; the median survives up to 50 %
contamination, and the MAD tells how contaminated the box is. For *thin*
classes (pedestrian, pole, cyclist) the object can be the **minority** of the
pixels, so the median lands on the background: the box is split in two
clusters (1-D Otsu on the sorted values) and, when the split is significant,
the near cluster (higher inverse depth) is taken for thin classes and the
majority cluster otherwise.

Everything works at network resolution through :class:`DepthResize`, on an
inner ROI (borders trimmed) to avoid the box-edge mix of object/background.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from percepcion3d.depth.ground_solver import AffineState
from percepcion3d.depth.preprocess import DepthResize

_MAD_TO_SIGMA = 1.4826
#: Standard error of the median relative to the standard error of the mean (Gaussian).
_MEDIAN_EFFICIENCY = 1.2533


@dataclass(frozen=True)
class BoxSampleConfig:
    shrink: float = 0.2
    """Fraction of the box width/height trimmed on each side before sampling."""
    max_pixels: int = 1024
    """Stride-subsample above this count (keeps the cost bounded per box)."""
    min_pixels: int = 9
    split_k: float = 4.0
    """Clusters are 'bimodal' when their medians differ by more than ``split_k·σ_pooled``."""
    min_cluster_frac: float = 0.15
    """Smaller clusters are treated as outlier tails, not as a second mode."""
    split_min_spread: float = 0.05
    """Skip the (costlier) bimodality search when the 10–90 % range is below this
    fraction of ``|median|`` (a unimodal 1 % Gaussian spans ≈ 2.6 %)."""
    exclude_dilate_px: float = 3.0
    """Frame-space dilation of ``exclude_boxes`` (detector jitter of the occluder's edges)."""


@dataclass(frozen=True)
class BoxSample:
    value: float
    """Chosen inverse-depth estimate (map units)."""
    sigma: float
    """Standard error of ``value`` (median of the chosen cluster)."""
    median: float
    """Median of the whole ROI."""
    mad: float
    n: int
    n_cluster: int
    bimodal: bool
    cluster_lo: float
    cluster_hi: float

    @property
    def contamination(self) -> float:
        """``MAD / |median|`` — a cheap 'how mixed is this box' indicator."""
        return self.mad / abs(self.median) if self.median != 0.0 else float("inf")


def inner_roi(
    box_xyxy: NDArray[np.floating[Any]], shrink: float, image_hw: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Shrink ``box`` by ``shrink`` of its size on every side and clip to the image."""
    x0, y0, x1, y1 = (float(v) for v in np.asarray(box_xyxy, dtype=np.float64).ravel()[:4])
    w, h = x1 - x0, y1 - y0
    x0s, x1s = x0 + shrink * w, x1 - shrink * w
    y0s, y1s = y0 + shrink * h, y1 - shrink * h
    img_h, img_w = image_hw
    return (
        min(max(x0s, 0.0), img_w - 1.0),
        min(max(y0s, 0.0), img_h - 1.0),
        min(max(x1s, 0.0), img_w - 1.0),
        min(max(y1s, 0.0), img_h - 1.0),
    )


def otsu_split(sorted_values: NDArray[np.float64]) -> int:
    """Index ``k`` maximising the between-class variance of ``v[:k] | v[k:]`` (``1 ≤ k < n``)."""
    v = sorted_values
    n = v.shape[0]
    if n < 2:
        return 1
    csum = np.cumsum(v)
    total = csum[-1]
    k = np.arange(1, n, dtype=np.float64)
    m0 = csum[:-1] / k
    m1 = (total - csum[:-1]) / (n - k)
    between = k * (n - k) * (m0 - m1) ** 2
    return int(np.argmax(between)) + 1


def _sorted_median(v: NDArray[np.float64]) -> float:
    n = v.shape[0]
    return float(v[n // 2]) if n % 2 else float(0.5 * (v[n // 2 - 1] + v[n // 2]))


def _mad(v: NDArray[np.float64], centre: float) -> float:
    d = np.abs(v - centre)
    n = d.shape[0]
    if n % 2:
        return float(np.partition(d, n // 2)[n // 2])
    p = np.partition(d, (n // 2 - 1, n // 2))
    return float(0.5 * (p[n // 2 - 1] + p[n // 2]))


def _median_sigma(v: NDArray[np.float64], mad: float) -> float:
    n = v.shape[0]
    sigma = _MAD_TO_SIGMA * mad
    if sigma <= 0.0:
        sigma = float(np.std(v))
    return float(_MEDIAN_EFFICIENCY * sigma / np.sqrt(max(n, 1)))


def sample_values(
    values: NDArray[np.floating[Any]], cfg: BoxSampleConfig, thin: bool = False
) -> BoxSample | None:
    """Robust statistics of a flat sample (``None`` when too few finite values)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.shape[0] > cfg.max_pixels:
        step = int(np.ceil(v.shape[0] / cfg.max_pixels))
        v = v[::step]
    n = int(v.shape[0])
    if n < cfg.min_pixels:
        return None
    v = np.sort(v)
    median = _sorted_median(v)
    mad = _mad(v, median)

    bimodal = False
    mad_lo = mad_hi = 0.0
    m_lo = m_hi = median
    lo = hi = v
    spread = float(v[(9 * n) // 10] - v[n // 10])
    if spread >= cfg.split_min_spread * abs(median):
        k = otsu_split(v)
        lo, hi = v[:k], v[k:]
        m_lo, m_hi = _sorted_median(lo), _sorted_median(hi)
    frac = min(lo.shape[0], hi.shape[0]) / n
    if lo is not hi and frac >= cfg.min_cluster_frac and lo.shape[0] >= 2 and hi.shape[0] >= 2:
        mad_lo, mad_hi = _mad(lo, m_lo), _mad(hi, m_hi)
        var_pooled = (
            (_MAD_TO_SIGMA * mad_lo) ** 2 * lo.shape[0]
            + (_MAD_TO_SIGMA * mad_hi) ** 2 * hi.shape[0]
        ) / n
        sigma_pooled = float(np.sqrt(var_pooled))
        if sigma_pooled <= 0.0:
            sigma_pooled = float(np.std(v)) * 0.5
        bimodal = (m_hi - m_lo) > cfg.split_k * sigma_pooled

    if bimodal:
        take_hi = thin or hi.shape[0] >= lo.shape[0]
        chosen, centre, mad_c = (hi, m_hi, mad_hi) if take_hi else (lo, m_lo, mad_lo)
    else:
        chosen, centre, mad_c = v, median, mad
    return BoxSample(
        value=centre,
        sigma=_median_sigma(chosen, mad_c),
        median=median,
        mad=mad,
        n=n,
        n_cluster=int(chosen.shape[0]),
        bimodal=bimodal,
        cluster_lo=m_lo,
        cluster_hi=m_hi,
    )


def sample_box(
    inv_depth_map: NDArray[np.floating[Any]],
    resize: DepthResize,
    box_xyxy_frame: NDArray[np.floating[Any]],
    cfg: BoxSampleConfig,
    thin: bool = False,
    exclude_boxes: NDArray[np.floating[Any]] | None = None,
) -> BoxSample | None:
    """Statistics of the map inside the inner ROI of a frame-space box.

    Pixels covered by any of ``exclude_boxes`` (frame space; e.g. nearer
    detections overlapping this one), dilated by ``cfg.exclude_dilate_px``,
    are dropped before the statistics.
    """
    x0, y0, x1, y1 = inner_roi(box_xyxy_frame, cfg.shrink, (resize.src_h, resize.src_w))
    r0, r1 = _row_span(resize, y0, y1)
    c0, c1 = _col_span(resize, x0, x1)
    if r1 <= r0 or c1 <= c0:
        return None
    roi = inv_depth_map[r0:r1, c0:c1]
    if exclude_boxes is not None:
        roi = np.array(roi, dtype=np.float64, copy=True)
        d = cfg.exclude_dilate_px
        for ex in np.asarray(exclude_boxes, dtype=np.float64).reshape(-1, 4):
            er0, er1 = _row_span(resize, float(ex[1]) - d, float(ex[3]) + d)
            ec0, ec1 = _col_span(resize, float(ex[0]) - d, float(ex[2]) + d)
            rr0, rr1 = max(er0 - r0, 0), min(er1 - r0, r1 - r0)
            cc0, cc1 = max(ec0 - c0, 0), min(ec1 - c0, c1 - c0)
            if rr1 > rr0 and cc1 > cc0:
                roi[rr0:rr1, cc0:cc1] = np.nan
    # Regular 2-D stride before flattening: the copy/finite/sort work is then bounded
    # by ``max_pixels`` instead of the ROI area (large near boxes are 10³–10⁴ px).
    if roi.size > cfg.max_pixels:
        step = int(math.ceil(math.sqrt(roi.size / cfg.max_pixels)))
        roi = roi[::step, ::step]
    return sample_values(roi, cfg, thin=thin)


def _row_span(resize: DepthResize, v0: float, v1: float) -> tuple[int, int]:
    """Scalar twin of ``DepthResize.frame_to_index`` for rows: ``[r0, r1)`` half-open."""
    hi = resize.dst_h - 1
    r0 = min(max(int(math.floor((v0 + 0.5) * resize.scale_y)), 0), hi)
    r1 = min(max(int(math.floor((v1 + 0.5) * resize.scale_y)), 0), hi) + 1
    return r0, r1


def _col_span(resize: DepthResize, u0: float, u1: float) -> tuple[int, int]:
    hi = resize.dst_w - 1
    c0 = min(max(int(math.floor((u0 + 0.5) * resize.scale_x)), 0), hi)
    c1 = min(max(int(math.floor((u1 + 0.5) * resize.scale_x)), 0), hi) + 1
    return c0, c1


@dataclass(frozen=True)
class NetDepth:
    z_m: float
    sigma_m: float
    """Total 1-σ: sample noise + ``(s, t)`` covariance + network relative error."""
    sigma_fit_m: float
    """Part of ``sigma_m`` due to the ``(s, t)`` uncertainty (pitch-correlated, see fusion)."""


@dataclass(frozen=True)
class NetDepthBatch:
    z_m: NDArray[np.float64]
    """``nan`` where the sample was missing or the affine map puts it at/beyond infinity."""
    sigma_m: NDArray[np.float64]
    sigma_fit_m: NDArray[np.float64]
    inv_z: NDArray[np.float64]
    """``ρ = (d̂ − t) / s`` [1/m]; finite wherever the sample was (may be ``≤ 0``)."""
    sigma_inv: NDArray[np.float64]
    """1-σ of ``ρ`` (sample + fit + network terms); the fusion works in this space."""
    sigma_inv_fit: NDArray[np.float64]


def disparity_to_depth_batch(
    values: NDArray[np.floating[Any]],
    sigmas: NDArray[np.floating[Any]],
    state: AffineState,
    net_rel_sigma: float,
) -> NetDepthBatch:
    """``ρ = (d̂ − t) / s``, ``Z = 1/ρ`` with first-order error propagation, vectorised.

    In inverse depth the map is affine, so the propagation is exact:
    ``∂ρ/∂d̂ = 1/s``, ``∂ρ/∂s = −ρ/s``, ``∂ρ/∂t = −1/s``; the network's own
    relative error ``net_rel_sigma`` (on ``1/Z``) is added in quadrature. The
    depth-space σ is the delta method ``σ_Z = σ_ρ / ρ²`` (``∂Z/∂d̂ = −Z²/s``,
    ``∂Z/∂s = Z/s``, ``∂Z/∂t = Z²/s``).
    """
    d = np.asarray(values, dtype=np.float64)
    sd = np.asarray(sigmas, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = (d - state.t) / state.s
        var_sample = (sd / state.s) ** 2
        jac = np.stack([-inv / state.s, np.full(inv.shape, -1.0 / state.s)], axis=-1)
        var_fit = np.einsum("...i,ij,...j->...", jac, state.cov, jac)
        var_net = (net_rel_sigma * inv) ** 2
        sigma_inv = np.sqrt(var_sample + var_fit + var_net)
        sigma_inv_fit = np.sqrt(var_fit)
        ok = np.isfinite(inv) & (inv > 0.0)
        z = np.where(ok, 1.0 / inv, np.nan)
        z2 = z * z
    return NetDepthBatch(
        z_m=z,
        sigma_m=sigma_inv * z2,
        sigma_fit_m=sigma_inv_fit * z2,
        inv_z=inv,
        sigma_inv=sigma_inv,
        sigma_inv_fit=sigma_inv_fit,
    )


def disparity_to_depth(
    sample: BoxSample, state: AffineState, net_rel_sigma: float
) -> NetDepth | None:
    """Scalar :func:`disparity_to_depth_batch`; ``None`` when ``d̂ ≤ t``."""
    nd = disparity_to_depth_batch(
        np.array([sample.value]), np.array([sample.sigma]), state, net_rel_sigma
    )
    if not np.isfinite(nd.z_m[0]):
        return None
    return NetDepth(
        z_m=float(nd.z_m[0]), sigma_m=float(nd.sigma_m[0]), sigma_fit_m=float(nd.sigma_fit_m[0])
    )
