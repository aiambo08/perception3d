"""Scale recovery of relative depth from the road (F4, CPU/NumPy).

Relative models output ``d̂ = s·(1/Z) + t`` with ``(s, t)`` unknown and
frame-dependent. Every road pixel below the horizon has a *geometric* depth
``Z_c(u, v)`` from the pinhole/ground-plane model, so thousands of pairs
``(d̂, 1/Z_c)`` per frame pin ``(s, t)`` down without touching the objects
that will be measured with them (R1 mitigation 1). The fit is robust (median
of random pair slopes → inlier re-weighting), then smoothed by a 2-state
random-walk Kalman filter with a χ² gate (mitigation 2).

Observability caveat (proved by ``tests/test_fusion.py``)
----------------------------------------------------------------
With ``roll = 0`` the inverse ground depth is **exactly affine in the row**::

    1/Z_c(v) = (sin θ + cos θ · y) / h,      y = (v − c_y) / f_y

so if the assumed pitch θ' is wrong, ``d̂`` is *still* exactly affine in
``1/Z_c(θ')``: the fit stays perfect and ``(s, t)`` silently absorb the pitch
error. Deprojecting the road with that ``(s, t)`` and fitting a plane
(:func:`fit_plane`) returns θ' — the assumption — not the true pitch. Pitch is
therefore **unobservable from road pixels + an affine-invariant depth map**;
it only becomes observable through a second metric cue: the metric model
(``t ≡ 0``) or objects of known height (:class:`PitchEstimator`, which solves
``Z_c(v_contact; θ) = f_y·H / h_px`` per detection and filters the median).

All arrays are at **network** resolution; :class:`GroundGrid` caches the
geometry per (resize, extrinsics) so the per-frame cost is the fit itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from percepcion3d.camera.calibration import CameraIntrinsics, ExtrinsicMountConfig
from percepcion3d.camera.geometry import PinholeGeometry
from percepcion3d.depth.preprocess import DepthResize

_MAD_TO_SIGMA = 1.4826
_MIN_FIT_POINTS = 12
#: Same grazing-angle guard as :mod:`percepcion3d.camera.geometry`.
_ANGLE_GUARD_RAD = 1e-4
#: 99 % χ² quantiles for 2 and 1 degrees of freedom.
CHI2_99_2DOF = 9.210
CHI2_99_1DOF = 6.635


# ----------------------------------------------------------------------------
# Geometry cache at network resolution
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundGrid:
    """Per-canvas-pixel ray directions and geometric inverse ground depth."""

    resize: DepthResize
    x_norm: NDArray[np.float64]
    """``[h, w]`` normalised ray x ``(u − c_x)/f_x`` at each canvas pixel centre."""
    y_norm: NDArray[np.float64]
    inv_z_ground: NDArray[np.float64]
    """``[h, w]`` ``1/Z_c`` of the ground hit; ``nan`` at/above the horizon."""
    horizon_row: float
    """Canvas row of the horizon at the principal column."""

    @staticmethod
    def build(
        geometry: PinholeGeometry, resize: DepthResize, base: GridRays | None = None
    ) -> GroundGrid:
        if base is None:
            base = GridRays.build(geometry.intrinsics, resize)
        # 1/Z_c = g_y / h with g_y the downward component of R_cg·(x, y, 1): closed form,
        # no finite differences (the σ model is evaluated per box, not per pixel).
        # g_y is separable in (row, col), so it is one outer sum, not four array passes.
        r1 = geometry.r_cg[1]
        inv_h = 1.0 / geometry.extrinsics.camera_height_m
        inv_z = np.add.outer(
            (r1[1] * base.y_norm_row + r1[2]) * inv_h, (r1[0] * base.x_norm_col) * inv_h
        )
        inv_z[inv_z <= base.guard * inv_h] = np.nan
        horizon_row = (geometry.get_horizon_v() + 0.5) * resize.scale_y - 0.5
        return GroundGrid(
            resize=resize,
            x_norm=base.x_norm,
            y_norm=base.y_norm,
            inv_z_ground=np.asarray(inv_z, dtype=np.float64),
            horizon_row=float(horizon_row),
        )

    def deproject(self, inv_depth: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """``[h, w, 3]`` camera-frame points from a metric inverse depth map."""
        with np.errstate(divide="ignore", invalid="ignore"):
            z = 1.0 / np.asarray(inv_depth, dtype=np.float64)
        return np.stack([self.x_norm * z, self.y_norm * z, z], axis=-1)


@dataclass(frozen=True)
class GridRays:
    """Intrinsics-only part of :class:`GroundGrid` (independent of pitch/roll/height)."""

    x_norm: NDArray[np.float64]
    y_norm: NDArray[np.float64]
    x_norm_col: NDArray[np.float64]
    """``[w]`` normalised ray x per canvas column (``x_norm`` is its row-broadcast)."""
    y_norm_row: NDArray[np.float64]
    guard: NDArray[np.float64]
    """``[h, w]`` grazing-angle guard ``ε·‖(x, y, 1)‖`` on ``g_y``."""

    @staticmethod
    def build(k: CameraIntrinsics, resize: DepthResize) -> GridRays:
        h, w = resize.dst_h, resize.dst_w
        cols = (np.arange(w, dtype=np.float64) + 0.5) / resize.scale_x - 0.5
        rows = (np.arange(h, dtype=np.float64) + 0.5) / resize.scale_y - 0.5
        x_col = (cols - k.cx) / k.fx
        y_row = (rows - k.cy) / k.fy
        uu, vv = np.meshgrid(x_col, y_row)
        return GridRays(
            x_norm=np.ascontiguousarray(uu),
            y_norm=np.ascontiguousarray(vv),
            x_norm_col=x_col,
            y_norm_row=y_row,
            guard=_ANGLE_GUARD_RAD * np.sqrt(uu * uu + vv * vv + 1.0),
        )


class GroundGridCache:
    """Rebuilds :class:`GroundGrid` only when the resize or the extrinsics change.

    Pitch/roll are quantised to ``angle_quantum_rad`` so an online pitch
    estimate drifting by micro-radians every frame does not trigger a rebuild
    — the grid is reused until the change is material (±0.05° of stale pitch in
    the road ``1/Z_c`` targets is ≈ 1 % at 20 m, below the network noise). The
    intrinsics-only rays (:class:`GridRays`) are cached separately, so a pitch
    rebuild costs two FMAs per pixel (≈ 1 ms at 924×280), not the full meshgrid.
    """

    def __init__(self, angle_quantum_rad: float = float(np.deg2rad(0.1))) -> None:
        self.angle_quantum_rad = angle_quantum_rad
        self._key: tuple[DepthResize, float, int, int] | None = None
        self._grid: GroundGrid | None = None
        self._rays_key: tuple[DepthResize, float, float, float, float] | None = None
        self._rays: GridRays | None = None
        self.n_builds = 0

    def rays(self, k: CameraIntrinsics, resize: DepthResize) -> GridRays:
        key = (resize, k.fx, k.fy, k.cx, k.cy)
        if self._rays is None or key != self._rays_key:
            self._rays = GridRays.build(k, resize)
            self._rays_key = key
        return self._rays

    def get(self, geometry: PinholeGeometry, resize: DepthResize) -> GroundGrid:
        ext = geometry.extrinsics
        q = self.angle_quantum_rad
        key = (
            resize,
            ext.camera_height_m,
            int(round(ext.pitch_rad / q)),
            int(round(ext.roll_rad / q)),
        )
        if self._grid is None or key != self._key:
            self._grid = GroundGrid.build(geometry, resize, self.rays(geometry.intrinsics, resize))
            self._key = key
            self.n_builds += 1
        return self._grid


@dataclass(frozen=True)
class ContactDepth:
    """Ground depth of one contact pixel with its analytic sensitivities."""

    z_m: float
    dz_dv: float
    """∂Z_c/∂v (m/px) — row jitter sensitivity."""
    dz_dpitch: float
    """∂Z_c/∂θ (m/rad) — shared with every quantity calibrated on the ground."""


def contact_depth(geometry: PinholeGeometry, u: float, v: float) -> ContactDepth | None:
    """Closed-form ``Z_c = h / g_y`` at pixel ``(u, v)`` with ∂/∂v and ∂/∂θ; ``None`` above horizon.

    ``g_y = R_cg[1]·(x, y, 1)`` and ``R_cg = P(θ)·R(φ)``, so ``∂R_cg[1]/∂θ`` is the
    derivative of the second row of the pitch matrix ``(0, −sinθ, cosθ)`` times ``R(φ)``.
    Scalar math only — this runs once per detection per frame.
    """
    k = geometry.intrinsics
    ext = geometry.extrinsics
    x = (u - k.cx) / k.fx
    y = (v - k.cy) / k.fy
    r1 = geometry.r_cg[1]
    gy = float(r1[0] * x + r1[1] * y + r1[2])
    if gy / math.sqrt(x * x + y * y + 1.0) <= _ANGLE_GUARD_RAD:
        return None
    ct, st = math.cos(ext.pitch_rad), math.sin(ext.pitch_rad)
    cr, sr = math.cos(ext.roll_rad), math.sin(ext.roll_rad)
    # d(row 1 of P)/dθ = (0, −sinθ, cosθ); times R(φ) = [[cr,−sr,0],[sr,cr,0],[0,0,1]].
    dr1 = (-st * sr, -st * cr, ct)
    dgy_dtheta = dr1[0] * x + dr1[1] * y + dr1[2]
    h = ext.camera_height_m
    z = h / gy
    return ContactDepth(
        z_m=z,
        dz_dv=-h / (gy * gy) * float(r1[1]) / k.fy,
        dz_dpitch=-h / (gy * gy) * dgy_dtheta,
    )


def contact_depth_batch(
    geometry: PinholeGeometry,
    u: NDArray[np.float64],
    v: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Vectorised :func:`contact_depth`: ``(z, ∂z/∂v, ∂z/∂θ)``, ``nan`` above the horizon."""
    k = geometry.intrinsics
    ext = geometry.extrinsics
    x = (np.asarray(u, dtype=np.float64) - k.cx) / k.fx
    y = (np.asarray(v, dtype=np.float64) - k.cy) / k.fy
    r1 = geometry.r_cg[1]
    gy = r1[0] * x + r1[1] * y + r1[2]
    valid = gy / np.sqrt(x * x + y * y + 1.0) > _ANGLE_GUARD_RAD
    ct, st = math.cos(ext.pitch_rad), math.sin(ext.pitch_rad)
    cr, sr = math.cos(ext.roll_rad), math.sin(ext.roll_rad)
    dgy_dtheta = (-st * sr) * x + (-st * cr) * y + ct
    h = ext.camera_height_m
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_gy2 = np.where(valid, h / (gy * gy), np.nan)
        z = np.where(valid, h / gy, np.nan)
    return z, -inv_gy2 * float(r1[1]) / k.fy, -inv_gy2 * dgy_dtheta


# ----------------------------------------------------------------------------
# Road pixel selection
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class RoadSampleConfig:
    max_points: int = 2000
    horizon_margin_px: float = 20.0
    """Frame rows below the horizon that are skipped (geometry σ diverges there)."""
    z_max_m: float = 60.0
    z_min_m: float = 3.0
    lateral_band: float = 0.6
    """Fraction of the image width kept around the principal column (0 < band ≤ 1)."""
    box_dilate_px: float = 4.0
    """Frame pixels added around each detection before masking it out."""
    oversample: float = 1.5
    """Draw ``oversample·max_points`` static candidates before removing boxes/NaNs."""


@dataclass(frozen=True)
class RoadSamples:
    d_hat: NDArray[np.float64]
    inv_z: NDArray[np.float64]
    n_candidates: int

    def __len__(self) -> int:
        return int(self.d_hat.shape[0])


def road_static_mask(grid: GroundGrid, cfg: RoadSampleConfig) -> NDArray[np.bool_]:
    """Canvas pixels usable as road regardless of detections: valid ground, in range, in band."""
    rs = grid.resize
    h, w = rs.dst_h, rs.dst_w
    row0 = int(np.clip(np.ceil(grid.horizon_row + cfg.horizon_margin_px * rs.scale_y), 0, h))
    half = 0.5 * cfg.lateral_band * w
    c0 = int(np.clip(np.floor(0.5 * w - half), 0, w))
    c1 = int(np.clip(np.ceil(0.5 * w + half), 0, w))
    mask = np.zeros((h, w), dtype=np.bool_)
    sub = grid.inv_z_ground[row0:, c0:c1]
    with np.errstate(invalid="ignore"):
        mask[row0:, c0:c1] = (sub >= 1.0 / cfg.z_max_m) & (sub <= 1.0 / cfg.z_min_m)
    return mask


def sample_road(
    inv_depth_map: NDArray[np.floating[Any]],
    grid: GroundGrid,
    static_idx: NDArray[np.intp],
    boxes_frame: NDArray[np.floating[Any]] | None,
    cfg: RoadSampleConfig,
    rng: np.random.Generator,
) -> RoadSamples:
    """Random subsample (≤ ``max_points``) of road pixels as ``(d̂, 1/Z_c)`` pairs.

    ``static_idx`` are the flat canvas indices of :func:`road_static_mask`
    (cached per grid); only the drawn candidates are checked against the
    detections and for NaNs, so the cost is O(max_points), not O(pixels).
    """
    if inv_depth_map.shape != grid.inv_z_ground.shape:
        raise ValueError(
            f"depth map {inv_depth_map.shape} does not match the grid {grid.inv_z_ground.shape}"
        )
    n_static = int(static_idx.shape[0])
    if n_static == 0:
        empty = np.empty(0, dtype=np.float64)
        return RoadSamples(d_hat=empty, inv_z=empty, n_candidates=0)
    n_draw = int(min(n_static, np.ceil(cfg.max_points * cfg.oversample)))
    idx = static_idx[rng.integers(0, n_static, size=n_draw)] if n_draw < n_static else static_idx
    flat = np.asarray(inv_depth_map).ravel()
    d_hat = flat[idx].astype(np.float64)
    keep = np.isfinite(d_hat)
    if boxes_frame is not None:
        bx = np.asarray(boxes_frame, dtype=np.float64).reshape(-1, 4)
        if bx.shape[0]:
            # Rasterise the dilated boxes once (a few slice writes) instead of testing
            # every candidate against every box.
            rs = grid.resize
            h, w = rs.dst_h, rs.dst_w
            c0 = np.clip(np.floor((bx[:, 0] - cfg.box_dilate_px) * rs.scale_x), 0, w).astype(int)
            c1 = np.clip(np.ceil((bx[:, 2] + 1.0 + cfg.box_dilate_px) * rs.scale_x), 0, w).astype(
                int
            )
            r0 = np.clip(np.floor((bx[:, 1] - cfg.box_dilate_px) * rs.scale_y), 0, h).astype(int)
            r1 = np.clip(np.ceil((bx[:, 3] + 1.0 + cfg.box_dilate_px) * rs.scale_y), 0, h).astype(
                int
            )
            occupied = np.zeros((h, w), dtype=np.bool_)
            for a, b, c, d in zip(r0, r1, c0, c1, strict=True):
                occupied[a:b, c:d] = True
            keep &= ~occupied.ravel()[idx]
    sel = np.flatnonzero(keep)[: cfg.max_points]
    return RoadSamples(
        d_hat=d_hat[sel], inv_z=grid.inv_z_ground.ravel()[idx[sel]], n_candidates=n_static
    )


# ----------------------------------------------------------------------------
# Robust affine fit  d̂ ≈ s·x + t
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class AffineFit:
    s: float
    t: float
    cov: NDArray[np.float64]
    """2×2 covariance of ``(s, t)`` from the inlier least squares."""
    n_inliers: int
    n_total: int
    sigma_resid: float
    """Robust 1-σ of the residual ``d̂ − (s·x + t)`` (MAD-based)."""

    @property
    def inlier_ratio(self) -> float:
        return self.n_inliers / self.n_total if self.n_total else 0.0


def robust_affine_fit(
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    rng: np.random.Generator,
    n_pairs: int = 2000,
    inlier_k: float = 3.0,
    min_dx: float = 1e-6,
) -> AffineFit | None:
    """Median-of-random-pair-slopes seed + one inlier-weighted LS refinement.

    Same ~29 % breakdown point as Theil–Sen in practice but O(n_pairs) instead
    of O(n²): ~0.2 ms for 2 000 points, which is what the ≤ 2 ms/frame budget
    of F4 needs. Returns ``None`` when the data are degenerate (too few points,
    no spread in ``x``, non-positive slope).
    """
    xs = np.asarray(x, dtype=np.float64).ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    n = int(xs.shape[0])
    if n < _MIN_FIT_POINTS or ys.shape[0] != n:
        return None
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    dx = xs[i] - xs[j]
    ok = np.abs(dx) > min_dx
    if ok.sum() < _MIN_FIT_POINTS:
        return None
    slopes = (ys[i][ok] - ys[j][ok]) / dx[ok]
    s0 = float(np.median(slopes))
    t0 = float(np.median(ys - s0 * xs))
    resid = ys - (s0 * xs + t0)
    mad = float(np.median(np.abs(resid - np.median(resid))))
    sigma_r = _MAD_TO_SIGMA * mad
    if sigma_r <= 0.0:
        sigma_r = max(float(np.std(resid)), 1e-12)
    inliers = np.abs(resid) <= inlier_k * sigma_r
    n_in = int(inliers.sum())
    if n_in < _MIN_FIT_POINTS:
        return None
    x_in, y_in = xs[inliers], ys[inliers]
    sx, sy = float(x_in.sum()), float(y_in.sum())
    sxx, sxy = float(x_in @ x_in), float(x_in @ y_in)
    det = n_in * sxx - sx * sx
    if det <= 1e-12 * max(n_in * sxx, 1e-300):
        return None
    s = (n_in * sxy - sx * sy) / det
    t = (sy - s * sx) / n_in
    if not np.isfinite(s) or s <= 0.0:
        return None
    r_in = y_in - (s * x_in + t)
    dof = max(n_in - 2, 1)
    var = float(r_in @ r_in) / dof
    # (AᵀA)⁻¹ for A = [x, 1] in closed form.
    cov = var / det * np.array([[n_in, -sx], [-sx, sxx]], dtype=np.float64)
    return AffineFit(
        s=s,
        t=t,
        cov=np.asarray(cov, dtype=np.float64),
        n_inliers=n_in,
        n_total=n,
        sigma_resid=max(float(np.sqrt(var)), sigma_r),
    )


# ----------------------------------------------------------------------------
# Temporal filter on (s, t)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class AffineFilterConfig:
    q_s_rel_per_s: float = 0.05
    """Random-walk σ of ``s`` per √second, relative to ``|s|``."""
    q_t_rel_per_s: float = 0.05
    """Random-walk σ of ``t`` per √second, relative to ``|s|`` (same units as d̂)."""
    r_floor_rel: float = 0.005
    """Floor on the measurement σ of ``s`` and ``t`` (relative to ``|s|``) — fits with
    thousands of pixels are optimistically confident because residuals are correlated."""
    chi2_gate: float = CHI2_99_2DOF
    reset_after_rejects: int = 15
    min_inlier_ratio: float = 0.5


@dataclass(frozen=True)
class AffineState:
    s: float
    t: float
    cov: NDArray[np.float64]
    n_updates: int
    reject_streak: int
    t_ns: int

    @property
    def sigma_s(self) -> float:
        return float(np.sqrt(self.cov[0, 0]))

    @property
    def sigma_t(self) -> float:
        return float(np.sqrt(self.cov[1, 1]))

    def inverse_depth(self, d_hat: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """``(d̂ − t)/s``: metric ``1/Z`` (may be ≤ 0 for pixels beyond the fit's reach)."""
        return (np.asarray(d_hat, dtype=np.float64) - self.t) / self.s


class AffineKalman:
    """Random-walk Kalman filter on ``(s, t)`` with a χ² gate and self-reset."""

    def __init__(self, cfg: AffineFilterConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else AffineFilterConfig()
        self._x: NDArray[np.float64] | None = None
        self._p: NDArray[np.float64] = np.eye(2)
        self._t_ns = 0
        self._n = 0
        self._streak = 0
        self.n_rejected_total = 0

    @property
    def state(self) -> AffineState | None:
        if self._x is None:
            return None
        return AffineState(
            s=float(self._x[0]),
            t=float(self._x[1]),
            cov=self._p.copy(),
            n_updates=self._n,
            reject_streak=self._streak,
            t_ns=self._t_ns,
        )

    def _measurement_cov(self, fit: AffineFit) -> NDArray[np.float64]:
        floor = (self.cfg.r_floor_rel * abs(fit.s)) ** 2
        return np.asarray(fit.cov + np.eye(2) * floor, dtype=np.float64)

    def _reset(self, fit: AffineFit, t_ns: int) -> None:
        self._x = np.array([fit.s, fit.t], dtype=np.float64)
        self._p = self._measurement_cov(fit)
        self._t_ns = t_ns
        self._n = 1
        self._streak = 0

    def update(self, fit: AffineFit | None, t_ns: int) -> AffineState | None:
        """Predict to ``t_ns`` and fuse ``fit`` (or only predict when it is ``None``)."""
        if self._x is None:
            if fit is not None and fit.inlier_ratio >= self.cfg.min_inlier_ratio:
                self._reset(fit, t_ns)
            return self.state
        dt = max((t_ns - self._t_ns) * 1e-9, 0.0)
        s_abs = abs(float(self._x[0]))
        q = np.diag(
            [
                (self.cfg.q_s_rel_per_s * s_abs) ** 2 * dt,
                (self.cfg.q_t_rel_per_s * s_abs) ** 2 * dt,
            ]
        )
        self._p = self._p + q
        self._t_ns = t_ns
        if fit is None or fit.inlier_ratio < self.cfg.min_inlier_ratio:
            return self.state
        z = np.array([fit.s, fit.t], dtype=np.float64)
        r = self._measurement_cov(fit)
        nu = z - self._x
        s_mat = self._p + r
        chi2 = float(nu @ np.linalg.solve(s_mat, nu))
        if chi2 > self.cfg.chi2_gate:
            self._streak += 1
            self.n_rejected_total += 1
            if self._streak >= self.cfg.reset_after_rejects:
                self._reset(fit, t_ns)
            return self.state
        k = np.linalg.solve(s_mat.T, self._p.T).T
        self._x = self._x + k @ nu
        self._p = (np.eye(2) - k) @ self._p
        self._p = 0.5 * (self._p + self._p.T)
        self._n += 1
        self._streak = 0
        return self.state


# ----------------------------------------------------------------------------
# Online pitch from objects of known height
# ----------------------------------------------------------------------------


def pitch_from_contact(
    y_norm: NDArray[np.floating[Any]],
    z_m: NDArray[np.floating[Any]],
    camera_height_m: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Pitch θ such that the ground ray through row ``y`` has optical depth ``Z``.

    From ``1/Z_c = (sin θ + y cos θ)/h``: ``sin θ + y cos θ = h/Z`` ⇒
    ``θ = asin(h / (Z·√(1+y²))) − atan(y)``. Invalid when ``h/Z > √(1+y²)`` (the
    point would be closer than the camera height allows).
    """
    y = np.asarray(y_norm, dtype=np.float64)
    z = np.asarray(z_m, dtype=np.float64)
    norm = np.sqrt(1.0 + y * y)
    with np.errstate(invalid="ignore", divide="ignore"):
        arg = camera_height_m / (z * norm)
        valid = np.isfinite(arg) & (z > 0) & (np.abs(arg) < 1.0)
        theta = np.where(valid, np.arcsin(np.clip(arg, -1.0, 1.0)) - np.arctan(y), np.nan)
    return np.asarray(theta, dtype=np.float64), np.asarray(valid, dtype=np.bool_)


def pitch_sigma_from_depth_sigma(
    y_norm: NDArray[np.floating[Any]],
    z_m: NDArray[np.floating[Any]],
    sigma_z_m: NDArray[np.floating[Any]],
    theta_rad: NDArray[np.floating[Any]],
    camera_height_m: float,
) -> NDArray[np.float64]:
    """First-order ``σ_θ`` induced by ``σ_Z`` at a fixed row (``dθ/dZ = −h/(Z²(cos θ − y sin θ))``)."""
    y = np.asarray(y_norm, dtype=np.float64)
    z = np.asarray(z_m, dtype=np.float64)
    th = np.asarray(theta_rad, dtype=np.float64)
    denom = z * z * np.abs(np.cos(th) - y * np.sin(th))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = camera_height_m / denom * np.asarray(sigma_z_m, dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


@dataclass(frozen=True)
class PitchFilterConfig:
    sigma0_rad: float = float(np.deg2rad(1.0))
    """Prior σ around the nominal mount pitch."""
    q_rad_per_sqrt_s: float = float(np.deg2rad(0.3))
    """Random-walk σ per √second (braking/load changes are slow; bumps are outliers)."""
    chi2_gate: float = CHI2_99_1DOF
    min_objects: int = 1
    max_step_rad: float = float(np.deg2rad(3.0))
    """Measurements farther than this from the nominal pitch are discarded outright."""


@dataclass(frozen=True)
class PitchState:
    pitch_rad: float
    sigma_rad: float
    n_updates: int
    n_rejected: int


class PitchEstimator:
    """1-D Kalman filter on the mount pitch fed by the robust median of per-object pitches."""

    def __init__(self, nominal: ExtrinsicMountConfig, cfg: PitchFilterConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else PitchFilterConfig()
        self.nominal = nominal
        self._x = float(nominal.pitch_rad)
        self._p = float(self.cfg.sigma0_rad**2)
        self._t_ns: int | None = None
        self._n = 0
        self._rej = 0

    @property
    def state(self) -> PitchState:
        return PitchState(self._x, float(np.sqrt(self._p)), self._n, self._rej)

    def extrinsics(self) -> ExtrinsicMountConfig:
        return ExtrinsicMountConfig(
            camera_height_m=self.nominal.camera_height_m,
            pitch_rad=self._x,
            roll_rad=self.nominal.roll_rad,
        )

    def update(
        self,
        theta_rad: NDArray[np.floating[Any]],
        sigma_rad: NDArray[np.floating[Any]],
        t_ns: int,
    ) -> PitchState:
        """Fuse one frame of per-object pitch measurements (``nan`` entries are skipped)."""
        if self._t_ns is not None:
            dt = max((t_ns - self._t_ns) * 1e-9, 0.0)
            self._p += self.cfg.q_rad_per_sqrt_s**2 * dt
        self._t_ns = t_ns
        th = np.asarray(theta_rad, dtype=np.float64).ravel()
        sg = np.asarray(sigma_rad, dtype=np.float64).ravel()
        ok = np.isfinite(th) & np.isfinite(sg) & (sg > 0)
        ok &= np.abs(th - self.nominal.pitch_rad) <= self.cfg.max_step_rad
        th, sg = th[ok], sg[ok]
        n = int(th.shape[0])
        if n < self.cfg.min_objects:
            return self.state
        if n == 1:
            z, r = float(th[0]), float(sg[0] ** 2)
        else:
            w = 1.0 / sg**2
            order = np.argsort(th)
            cw = np.cumsum(w[order])
            z = float(th[order][np.searchsorted(cw, 0.5 * cw[-1])])
            r = float(np.pi / 2.0 / w.sum())
        nu = z - self._x
        s = self._p + r
        if nu * nu / s > self.cfg.chi2_gate:
            self._rej += 1
            return self.state
        k = self._p / s
        self._x += k * nu
        self._p *= 1.0 - k
        self._n += 1
        return self.state


# ----------------------------------------------------------------------------
# Plane fit (metric depth only — see module docstring)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class PlaneFit:
    normal: NDArray[np.float64]
    """Unit normal in camera coordinates, oriented like the ground-frame +Y (down)."""
    height_m: float
    """Signed distance from the camera centre to the plane along ``normal``."""
    pitch_rad: float
    roll_rad: float
    rms_m: float
    n: int


def pitch_roll_from_normal(normal: NDArray[np.floating[Any]]) -> tuple[float, float]:
    """Invert ``n = (cos θ sin ρ, cos θ cos ρ, sin θ)`` (row 1 of ``R_cg``)."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    pitch = float(np.arcsin(np.clip(n[2], -1.0, 1.0)))
    roll = float(np.arctan2(n[0], n[1]))
    return pitch, roll


def fit_plane(points_xyz: NDArray[np.floating[Any]]) -> PlaneFit | None:
    """Total-least-squares plane through ``[N,3]`` camera points (needs ``N ≥ 3``)."""
    p = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    p = p[np.all(np.isfinite(p), axis=1)]
    n_pts = int(p.shape[0])
    if n_pts < 3:
        return None
    centroid = p.mean(axis=0)
    q = p - centroid
    _, sv, vt = np.linalg.svd(q, full_matrices=False)
    normal = vt[2]
    if normal[1] < 0.0:
        normal = -normal
    height = float(normal @ centroid)
    rms = float(sv[2] / np.sqrt(n_pts)) if sv.shape[0] == 3 else 0.0
    pitch, roll = pitch_roll_from_normal(normal)
    return PlaneFit(
        normal=np.asarray(normal, dtype=np.float64),
        height_m=height,
        pitch_rad=pitch,
        roll_rad=roll,
        rms_m=rms,
        n=n_pts,
    )


# ----------------------------------------------------------------------------
# Per-frame orchestration
# ----------------------------------------------------------------------------


@dataclass
class GroundSolverReport:
    fit: AffineFit | None
    state: AffineState | None
    n_candidates: int
    accepted: bool


@dataclass
class GroundSolver:
    """Road sampling + robust fit + temporal filter, one call per depth map."""

    sample_cfg: RoadSampleConfig = field(default_factory=RoadSampleConfig)
    filter_cfg: AffineFilterConfig = field(default_factory=AffineFilterConfig)
    seed: int = 0

    def __post_init__(self) -> None:
        self.grids = GroundGridCache()
        self.kalman = AffineKalman(self.filter_cfg)
        self.last_samples: RoadSamples | None = None
        self._static_for: GroundGrid | None = None
        self._static_idx: NDArray[np.intp] = np.empty(0, dtype=np.intp)

    def static_candidates(self, grid: GroundGrid) -> NDArray[np.intp]:
        if grid is not self._static_for:
            self._static_idx = np.flatnonzero(road_static_mask(grid, self.sample_cfg))
            self._static_for = grid
        return self._static_idx

    @property
    def state(self) -> AffineState | None:
        return self.kalman.state

    def update(
        self,
        inv_depth_map: NDArray[np.floating[Any]],
        resize: DepthResize,
        geometry: PinholeGeometry,
        boxes_frame: NDArray[np.floating[Any]] | None,
        frame_id: int,
        t_ns: int,
    ) -> GroundSolverReport:
        grid = self.grids.get(geometry, resize)
        rng = np.random.default_rng([self.seed, frame_id])
        static_idx = self.static_candidates(grid)
        samples = sample_road(inv_depth_map, grid, static_idx, boxes_frame, self.sample_cfg, rng)
        self.last_samples = samples
        fit = robust_affine_fit(samples.inv_z, samples.d_hat, rng)
        n_before = self.kalman.n_rejected_total
        state = self.kalman.update(fit, t_ns)
        accepted = fit is not None and self.kalman.n_rejected_total == n_before
        return GroundSolverReport(
            fit=fit, state=state, n_candidates=samples.n_candidates, accepted=accepted
        )
