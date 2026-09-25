"""Metric fusion: three depth cues per detection → one ``Measurement3D`` (F4).

For a box of class ``c`` with bottom row ``v_b`` and pixel height ``h_px``::

    Z_g  = ground-plane depth of the contact ray through (u_b, v_b)      [geometry]
    Z_n  = s / (d̂_box − t), d̂_box the robust box statistic of the map    [depth net]
    Z_h  = f_y · H_c / h_px                                              [height prior]

Each comes with a first-order variance, and each has a *gate* that turns it
off (contact row truncated by the image border, class that does not touch the
ground, no depth map yet, box beyond the affine fit's reach, ...).

The cues are fused in **inverse depth** ``ρ = 1/Z``, not in ``Z``: there the
error models are (close to) linear — ``ρ_g = (sinθ + y cosθ)/h`` is linear in the
row and has a pitch sensitivity ``∂ρ_g/∂θ ≈ 1/h`` independent of range, ``ρ_n``
is affine in the network output by construction, and ``ρ_h ∝ h_px``. In ``Z`` the
same errors are heavy-tailed towards the far side (``Z = 1/ρ`` explodes as the
contact row nears the horizon), so a Gaussian BLUE in ``Z`` under-weights exactly
the bad far-field ground samples. Depth and its σ are recovered at the end
(``Z = 1/ρ̂``, ``σ_Z = σ_ρ/ρ̂²``).

The estimators are **not independent**: ``ρ_g`` and ``ρ_n`` both inherit the
mount-pitch error — ``ρ_n`` through the ``(s, t)`` fitted on the road with the
same pitch (see :mod:`percepcion3d.depth.ground_solver`). A plain
inverse-variance average would double-count them, so the fusion is the BLUE
under ``Σ = D + c cᵀ`` with ``D`` the independent variances and ``c_i = (∂ρ_i/∂θ)·σ_θ``
the shared pitch sensitivity (``c_h = 0``: the height prior is pitch-free,
which is exactly why it rescues the far field when the pitch is off)::

    w = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1),   ρ̂ = wᵀ ρ,   σ_ρ̂² = 1 / (1ᵀ Σ⁻¹ 1)

A χ² consistency test on the residuals flags disagreeing cues and inflates the
fused variance instead of silently averaging a wrong estimator.

``Z`` here is the optical depth of the **nearest ground contact** (what the
box bottom sees); ``Measurement3D.center_offset_m`` (half the class length)
converts to the object centre that KITTI labels and the tracker use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Flag, auto
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from percepcion3d.camera.calibration import CameraIntrinsics
from percepcion3d.camera.geometry import PinholeGeometry
from percepcion3d.depth.depth_trt import DepthMap
from percepcion3d.depth.ground_solver import (
    CHI2_99_1DOF,
    AffineFilterConfig,
    AffineState,
    GroundSolver,
    GroundSolverReport,
    PitchEstimator,
    PitchFilterConfig,
    PitchState,
    RoadSampleConfig,
    contact_depth_batch,
    pitch_from_contact,
    pitch_sigma_from_depth_sigma,
)
from percepcion3d.depth.preprocess import DepthResize
from percepcion3d.depth.sampling import (
    BoxSampleConfig,
    disparity_to_depth_batch,
    sample_box,
)
from percepcion3d.utils.profiling import StageTimer

#: 99 % χ² quantile for 2 dof (three consistent estimators → 2 residual dof).
_CHI2_99_2DOF = 9.210


class FusionFlag(Flag):
    NONE = 0
    NO_PRIOR = auto()
    GROUND_TRUNCATED = auto()
    GROUND_ABOVE_HORIZON = auto()
    GROUND_OFF = auto()
    NET_UNAVAILABLE = auto()
    NET_NO_SAMPLE = auto()
    NET_BEYOND = auto()
    NET_BIMODAL = auto()
    HEIGHT_TRUNCATED = auto()
    HEIGHT_TOO_SMALL = auto()
    INCONSISTENT = auto()
    GROUND_REJECTED = auto()
    NET_REJECTED = auto()
    HEIGHT_REJECTED = auto()
    OUT_OF_RANGE = auto()
    NO_ESTIMATE = auto()


@dataclass(frozen=True)
class ClassPrior:
    height_m: float
    sigma_height_m: float
    length_m: float
    thin: bool = False
    touches_ground: bool = True


@dataclass(frozen=True)
class FusionConfig:
    priors: dict[str, ClassPrior]
    default_prior: ClassPrior | None = None
    sigma_pitch_rad: float = float(np.deg2rad(0.5))
    sigma_row_px: float = 1.5
    """1-σ of the box bottom row (detector jitter + contact-point ambiguity)."""
    sigma_col_px: float = 1.5
    net_rel_sigma: float = 0.05
    """Relative error of the network's inverse depth after affine correction (``k``)."""
    z_min_m: float = 1.0
    z_max_m: float = 80.0
    """Operating range: the fused estimate is flagged ``OUT_OF_RANGE`` outside it."""
    range_slack: float = 1.5
    """Individual cues are only dropped beyond ``[z_min / slack, z_max · slack]``, so a
    cue that overshoots the operating range by its own σ keeps voting instead of
    leaving the fusion to whichever cue happened to land inside."""
    min_box_height_px: float = 12.0
    border_margin_px: float = 2.0
    """A box edge closer than this to the image border counts as truncated."""
    chi2_consistency: float = CHI2_99_1DOF
    use_ground: bool = True
    use_net: bool = True
    use_height: bool = True
    sampling: BoxSampleConfig = field(default_factory=BoxSampleConfig)

    def prior_for(self, cls: str) -> ClassPrior | None:
        return self.priors.get(cls, self.default_prior)


@dataclass(frozen=True)
class Measurement3D:
    det_index: int
    cls: str
    frame_id: int
    t_capture_ns: int
    z_cam_m: float
    """Fused optical depth of the nearest ground contact (``nan`` when nothing survived)."""
    sigma_z_m: float
    x_lat_m: float
    """Lateral offset in the ground frame (right positive)."""
    z_fwd_m: float
    """Forward planar distance in the ground frame."""
    sigma_x_m: float
    center_offset_m: float
    """Add to ``z_fwd_m`` to get the object centre (half the class length)."""
    z_ground_m: float
    z_net_m: float
    z_height_m: float
    sigma_ground_m: float
    sigma_net_m: float
    sigma_height_m: float
    weights: NDArray[np.float64]
    """BLUE weights ``(w_g, w_n, w_h)`` on the inverse-depth cues (zeros for gated cues)."""
    chi2: float
    flags: FusionFlag
    pitch_meas_rad: float = float("nan")
    """Pitch implied by ``Z_h`` at the contact row (``nan`` when unavailable)."""
    pitch_meas_sigma_rad: float = float("nan")

    @property
    def valid(self) -> bool:
        return bool(np.isfinite(self.z_cam_m))

    @property
    def z_center_fwd_m(self) -> float:
        return self.z_fwd_m + self.center_offset_m


def load_fusion_config(path: Path | str) -> FusionConfig:
    with Path(path).open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    priors: dict[str, ClassPrior] = {}
    for name, p in (raw.get("classes") or {}).items():
        prior = ClassPrior(
            height_m=float(p["height_m"]),
            sigma_height_m=float(p["sigma_height_m"]),
            length_m=float(p.get("length_m", 0.0)),
            thin=bool(p.get("thin", False)),
            touches_ground=bool(p.get("touches_ground", True)),
        )
        priors[str(name)] = prior
        for alias in p.get("aliases", []) or []:
            priors[str(alias)] = prior
    noise = raw.get("noise") or {}
    gates = raw.get("gates") or {}
    samp = raw.get("sampling") or {}
    sampling = BoxSampleConfig(
        shrink=float(samp.get("shrink", 0.2)),
        max_pixels=int(samp.get("max_pixels", 1024)),
        min_pixels=int(samp.get("min_pixels", 9)),
        split_k=float(samp.get("split_k", 4.0)),
        min_cluster_frac=float(samp.get("min_cluster_frac", 0.15)),
        split_min_spread=float(samp.get("split_min_spread", 0.05)),
        exclude_dilate_px=float(samp.get("exclude_dilate_px", 3.0)),
    )
    return FusionConfig(
        priors=priors,
        default_prior=None,
        sigma_pitch_rad=float(np.deg2rad(float(noise.get("sigma_pitch_deg", 0.5)))),
        sigma_row_px=float(noise.get("sigma_row_px", 1.5)),
        sigma_col_px=float(noise.get("sigma_col_px", 1.5)),
        net_rel_sigma=float(noise.get("net_rel_sigma", 0.05)),
        z_min_m=float(gates.get("z_min_m", 1.0)),
        z_max_m=float(gates.get("z_max_m", 80.0)),
        range_slack=float(gates.get("range_slack", 1.5)),
        min_box_height_px=float(gates.get("min_box_height_px", 12.0)),
        border_margin_px=float(gates.get("border_margin_px", 2.0)),
        chi2_consistency=float(gates.get("chi2_consistency", CHI2_99_1DOF)),
        sampling=sampling,
    )


def load_solver_configs(
    path: Path | str,
) -> tuple[RoadSampleConfig, AffineFilterConfig, PitchFilterConfig]:
    """``road:``, ``affine_filter:`` and ``pitch_filter:`` sections of ``fusion.yaml``."""
    with Path(path).open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    road = raw.get("road") or {}
    aff = raw.get("affine_filter") or {}
    pit = raw.get("pitch_filter") or {}
    road_cfg = RoadSampleConfig(
        max_points=int(road.get("max_points", 2000)),
        horizon_margin_px=float(road.get("horizon_margin_px", 20.0)),
        z_max_m=float(road.get("z_max_m", 60.0)),
        z_min_m=float(road.get("z_min_m", 3.0)),
        lateral_band=float(road.get("lateral_band", 0.6)),
        box_dilate_px=float(road.get("box_dilate_px", 4.0)),
    )
    aff_cfg = AffineFilterConfig(
        q_s_rel_per_s=float(aff.get("q_s_rel_per_s", 0.05)),
        q_t_rel_per_s=float(aff.get("q_t_rel_per_s", 0.05)),
        r_floor_rel=float(aff.get("r_floor_rel", 0.005)),
        reset_after_rejects=int(aff.get("reset_after_rejects", 15)),
        min_inlier_ratio=float(aff.get("min_inlier_ratio", 0.5)),
    )
    pit_cfg = PitchFilterConfig(
        sigma0_rad=float(np.deg2rad(float(pit.get("sigma0_deg", 1.0)))),
        q_rad_per_sqrt_s=float(np.deg2rad(float(pit.get("q_deg_per_sqrt_s", 0.3)))),
        min_objects=int(pit.get("min_objects", 1)),
        max_step_rad=float(np.deg2rad(float(pit.get("max_step_deg", 3.0)))),
    )
    return road_cfg, aff_cfg, pit_cfg


# ----------------------------------------------------------------------------
# Fusion core
# ----------------------------------------------------------------------------


def fuse_correlated(
    z: NDArray[np.float64],
    sigma_indep: NDArray[np.float64],
    c_pitch: NDArray[np.float64],
) -> tuple[float, float, NDArray[np.float64], float]:
    """BLUE of ``z`` under ``Σ = diag(σ²) + c cᵀ``; returns ``(ẑ, σ_ẑ, w, χ²)``."""
    m = z.shape[0]
    sigma_mat = np.diag(sigma_indep**2) + np.outer(c_pitch, c_pitch)
    ones = np.ones(m)
    sinv_1 = np.linalg.solve(sigma_mat, ones)
    denom = float(ones @ sinv_1)
    w = sinv_1 / denom
    z_hat = float(w @ z)
    var = 1.0 / denom
    r = z - z_hat
    chi2 = float(r @ np.linalg.solve(sigma_mat, r)) if m > 1 else 0.0
    return z_hat, float(np.sqrt(var)), np.asarray(w, dtype=np.float64), chi2


def occluding_boxes(
    boxes: NDArray[np.float64], i: int, margin_px: float = 2.0
) -> NDArray[np.float64] | None:
    """Boxes overlapping box ``i`` whose bottom edge is lower in the image.

    For objects standing on the ground a lower contact row means a nearer
    object, so those pixels inside box ``i`` belong to an occluder, not to it.
    """
    if boxes.shape[0] < 2:
        return None
    x0, y0, x1, y1 = boxes[i]
    others = np.delete(boxes, i, axis=0)
    overlap = (others[:, 0] < x1) & (others[:, 2] > x0) & (others[:, 1] < y1) & (others[:, 3] > y0)
    nearer = others[:, 3] > y1 + margin_px
    sel = others[overlap & nearer]
    return sel if sel.shape[0] else None


def occlusion_matrix(boxes: NDArray[np.float64], margin_px: float = 2.0) -> NDArray[np.bool_]:
    """``occ[i, j]`` ⇔ box ``j`` overlaps box ``i`` and is nearer (see :func:`occluding_boxes`)."""
    n = boxes.shape[0]
    if n < 2:
        return np.zeros((n, n), dtype=bool)
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    overlap = (
        (x0[None, :] < x1[:, None])
        & (x1[None, :] > x0[:, None])
        & (y0[None, :] < y1[:, None])
        & (y1[None, :] > y0[:, None])
    )
    nearer = y1[None, :] > y1[:, None] + margin_px
    occ = overlap & nearer
    np.fill_diagonal(occ, False)
    return occ


def _row_for_depth(geometry: PinholeGeometry, z_m: NDArray[np.float64]) -> NDArray[np.float64]:
    """Row at the principal column whose ground contact has optical depth ``z_m`` (roll ≈ 0)."""
    k = geometry.intrinsics
    ext = geometry.extrinsics
    # 1/Z = (sinθ + cosθ·y)/h  →  y = (h/Z − sinθ)/cosθ.
    y = (ext.camera_height_m / z_m - np.sin(ext.pitch_rad)) / np.cos(ext.pitch_rad)
    return np.asarray(k.cy + y * k.fy, dtype=np.float64)


#: σ assigned to a gated cue inside the batched BLUE. Cues are in inverse depth
#: (ρ ≤ 1.5 m⁻¹, σ_ρ ~ 1e-4..1e-1), so 1e4 makes the weight negligible (< 1e-10 relative)
#: while keeping the covariance condition number ≤ 1e16.
_INACTIVE_SIGMA = 1e4

_CUE_REJECTED = (FusionFlag.GROUND_REJECTED, FusionFlag.NET_REJECTED, FusionFlag.HEIGHT_REJECTED)


def fuse_correlated_batch(
    z: NDArray[np.float64],
    sigma_indep: NDArray[np.float64],
    c_pitch: NDArray[np.float64],
    active: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Row-wise :func:`fuse_correlated` over ``[N, m]`` cues; inactive cues get ``σ = 1e4``.

    Returns ``(ẑ, σ_ẑ, w, χ²)`` with ``ẑ = nan`` where nothing is active, ``χ² = 0``
    with fewer than two active cues and ``w = 0`` on inactive cues.
    """
    n, m = z.shape
    sig = np.where(active, sigma_indep, _INACTIVE_SIGMA)
    zz = np.where(active, z, 0.0)
    cc = np.where(active, c_pitch, 0.0)
    s_mat = np.zeros((n, m, m), dtype=np.float64)
    idx = np.arange(m)
    s_mat[:, idx, idx] = sig * sig
    s_mat += cc[:, :, None] * cc[:, None, :]
    ones = np.ones((n, m, 1), dtype=np.float64)
    sinv_1 = np.linalg.solve(s_mat, ones)[:, :, 0]
    denom = sinv_1.sum(axis=1)
    w = sinv_1 / denom[:, None]
    z_hat = (w * zz).sum(axis=1)
    r = np.where(active, zz - z_hat[:, None], 0.0)
    chi2 = np.einsum("ij,ij->i", r, np.linalg.solve(s_mat, r[:, :, None])[:, :, 0])
    n_act = active.sum(axis=1)
    z_hat = np.where(n_act > 0, z_hat, np.nan)
    sigma_hat = np.where(n_act > 0, np.sqrt(1.0 / denom), np.nan)
    chi2 = np.where(n_act > 1, chi2, 0.0)
    return z_hat, sigma_hat, np.where(active, w, 0.0), chi2


class MetricFuser:
    """Batched fusion of all boxes of a frame; geometry and affine state are passed per call."""

    def __init__(self, cfg: FusionConfig) -> None:
        self.cfg = cfg
        self._none_prior = ClassPrior(height_m=1.0, sigma_height_m=1e6, length_m=0.0)

    def _priors(
        self, classes: list[str]
    ) -> tuple[list[ClassPrior], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        priors = [self.cfg.prior_for(c) for c in classes]
        no_prior = np.array([p is None for p in priors], dtype=bool)
        full = [p if p is not None else self._none_prior for p in priors]
        return (
            full,
            np.array([p.height_m for p in full]),
            np.array([p.sigma_height_m for p in full]),
            no_prior,
        )

    def _net_cue(
        self,
        b: NDArray[np.float64],
        priors: list[ClassPrior],
        geometry: PinholeGeometry,
        inv_map: NDArray[np.floating[Any]] | None,
        resize: DepthResize | None,
        affine: AffineState | None,
        flags: list[FusionFlag],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """``(ρ_n, σ_ρn, c_ρn)`` per box in inverse depth (``nan`` where gated); updates ``flags``."""
        n = b.shape[0]
        rho = np.full(n, np.nan)
        sig = np.full(n, np.nan)
        c = np.zeros(n)
        if not self.cfg.use_net or inv_map is None or resize is None or affine is None:
            for i in range(n):
                flags[i] |= FusionFlag.NET_UNAVAILABLE
            return rho, sig, c
        occ = occlusion_matrix(b, self.cfg.border_margin_px)
        vals = np.full(n, np.nan)
        sig_d = np.full(n, np.nan)
        for i in range(n):
            excl = b[occ[i]] if occ[i].any() else None
            sample = sample_box(
                inv_map, resize, b[i], self.cfg.sampling, thin=priors[i].thin, exclude_boxes=excl
            )
            if sample is None:
                flags[i] |= FusionFlag.NET_NO_SAMPLE
                continue
            if sample.bimodal:
                flags[i] |= FusionFlag.NET_BIMODAL
            vals[i], sig_d[i] = sample.value, sample.sigma
        nd = disparity_to_depth_batch(vals, sig_d, affine, self.cfg.net_rel_sigma)
        beyond = np.isfinite(vals) & ~np.isfinite(nd.z_m)
        for i in np.flatnonzero(beyond):
            flags[i] |= FusionFlag.NET_BEYOND
        ok = np.isfinite(nd.z_m)
        rho[ok], sig[ok] = nd.inv_z[ok], nd.sigma_inv[ok]
        # (s, t) were fitted on the road under the current pitch, so ρ_n carries the pitch
        # sensitivity of a ground contact at that depth: ∂ρ/∂θ = −(∂Z/∂θ)/Z² ≈ 1/h.
        if ok.any():
            k = geometry.intrinsics
            z = nd.z_m[ok]
            rows = _row_for_depth(geometry, z)
            _, _, dz_dth = contact_depth_batch(geometry, np.full(rows.shape, k.cx), rows)
            drho_dth = np.where(
                np.isfinite(dz_dth), -dz_dth / (z * z), 1.0 / geometry.extrinsics.camera_height_m
            )
            c[ok] = drho_dth * self.cfg.sigma_pitch_rad
        return rho, sig, c

    def fuse(
        self,
        boxes: NDArray[np.floating[Any]],
        classes: list[str],
        geometry: PinholeGeometry,
        inv_map: NDArray[np.floating[Any]] | None,
        resize: DepthResize | None,
        affine: AffineState | None,
        frame_id: int,
        t_ns: int,
    ) -> list[Measurement3D]:
        b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        n = b.shape[0]
        if len(classes) != n:
            raise ValueError(f"{n} boxes but {len(classes)} class names")
        if n == 0:
            return []
        cfg = self.cfg
        k = geometry.intrinsics
        h_cam = geometry.extrinsics.camera_height_m
        m_px = cfg.border_margin_px
        flags = [FusionFlag.NONE] * n
        priors, h_prior, sig_h_prior, no_prior = self._priors(classes)
        touches = np.array([p.touches_ground for p in priors], dtype=bool)
        for i in np.flatnonzero(no_prior):
            flags[i] |= FusionFlag.NO_PRIOR

        x0, y0, x1, y1 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        u_b, v_b = 0.5 * (x0 + x1), y1
        h_px = y1 - y0

        # -- ground cue (in inverse depth ρ = 1/Z: ∂ρ/∂x = −(∂Z/∂x)/Z²) ------------
        z_g, dz_dv, dz_dth = contact_depth_batch(geometry, u_b, v_b)
        g_off = ~(cfg.use_ground & touches)
        g_trunc = ~g_off & (v_b >= k.height - 1.0 - m_px)
        g_horizon = ~g_off & ~g_trunc & ~np.isfinite(z_g)
        g_on = ~g_off & ~g_trunc & ~g_horizon
        with np.errstate(divide="ignore", invalid="ignore"):
            rho_g = np.where(g_on, 1.0 / z_g, np.nan)
            inv_z2 = np.where(g_on, 1.0 / (z_g * z_g), 0.0)
        sig_g = np.abs(dz_dv) * inv_z2 * cfg.sigma_row_px
        c_g = np.where(g_on, -dz_dth * inv_z2 * cfg.sigma_pitch_rad, 0.0)
        for i in np.flatnonzero(g_off):
            flags[i] |= FusionFlag.GROUND_OFF
        for i in np.flatnonzero(g_trunc):
            flags[i] |= FusionFlag.GROUND_TRUNCATED
        for i in np.flatnonzero(g_horizon):
            flags[i] |= FusionFlag.GROUND_ABOVE_HORIZON

        # -- net cue ----------------------------------------------------------
        rho_n, sig_n, c_n = self._net_cue(b, priors, geometry, inv_map, resize, affine, flags)

        # -- height cue -------------------------------------------------------
        h_trunc = (y0 <= m_px) | (y1 >= k.height - 1.0 - m_px)
        h_small = ~h_trunc & (h_px < cfg.min_box_height_px)
        h_on = cfg.use_height & ~no_prior & ~h_trunc & ~h_small
        with np.errstate(divide="ignore", invalid="ignore"):
            rho_h = np.where(h_on, h_px / (k.fy * h_prior), np.nan)
            rel_h = sig_h_prior / h_prior
            rel_px = np.sqrt(2.0) * cfg.sigma_row_px / h_px
            sig_h = rho_h * np.sqrt(rel_h**2 + rel_px**2)
        if cfg.use_height:
            for i in np.flatnonzero(h_trunc):
                flags[i] |= FusionFlag.HEIGHT_TRUNCATED
            for i in np.flatnonzero(h_small):
                flags[i] |= FusionFlag.HEIGHT_TOO_SMALL

        # -- gates + BLUE in inverse depth ------------------------------------
        rho_all = np.stack([rho_g, rho_n, rho_h], axis=1)
        sig_all = np.stack([sig_g, sig_n, sig_h], axis=1)
        c_all = np.stack([c_g, c_n, np.zeros(n)], axis=1)
        ok = np.isfinite(rho_all) & np.isfinite(sig_all) & (sig_all > 0.0)
        with np.errstate(invalid="ignore"):
            plausible = (rho_all >= 1.0 / (cfg.z_max_m * cfg.range_slack)) & (
                rho_all <= cfg.range_slack / cfg.z_min_m
            )
        implausible = ok & ~plausible
        active = ok & plausible
        rho_all = np.where(implausible, np.nan, rho_all)
        sig_all = np.where(implausible, np.nan, sig_all)
        rho_hat, sigma_rho, w, chi2 = fuse_correlated_batch(rho_all, sig_all, c_all, active)
        n_act = active.sum(axis=1)
        for i in np.flatnonzero(n_act == 0):
            flags[i] |= FusionFlag.NO_ESTIMATE
        gate = np.where(n_act == 3, _CHI2_99_2DOF, cfg.chi2_consistency)
        for i in np.flatnonzero((n_act >= 2) & (chi2 > gate)):
            rho_hat[i], sigma_rho[i], w[i], chi2[i], flags[i] = self._arbitrate(
                rho_all[i],
                sig_all[i],
                c_all[i],
                active[i],
                rho_hat[i],
                sigma_rho[i],
                w[i],
                chi2[i],
                flags[i],
            )
        # Back to depth: Z = 1/ρ, σ_Z = σ_ρ/ρ² (per cue and fused). A BLUE with negative
        # weights (strong pitch correlation) can in principle land at ρ̂ ≤ 0: no depth.
        with np.errstate(divide="ignore", invalid="ignore"):
            degenerate = (n_act > 0) & ~(rho_hat > 0.0)
            rho_hat = np.where(degenerate, np.nan, rho_hat)
            z_hat = 1.0 / rho_hat
            sigma_hat = sigma_rho * z_hat * z_hat
            z_all = 1.0 / rho_all
            sig_z_all = sig_all * z_all * z_all
            c_z_all = np.abs(c_all) * z_all * z_all
            fused_out = (n_act > 0) & ((z_hat < cfg.z_min_m) | (z_hat > cfg.z_max_m))
        for i in np.flatnonzero(fused_out | implausible.any(axis=1)):
            flags[i] |= FusionFlag.OUT_OF_RANGE
        for i in np.flatnonzero(degenerate):
            flags[i] |= FusionFlag.NO_ESTIMATE

        # -- lateral position along the contact ray, in the ground frame ------------
        with np.errstate(invalid="ignore"):
            p_c = np.stack(
                [(u_b - k.cx) * z_hat / k.fx, (v_b - k.cy) * z_hat / k.fy, z_hat], axis=1
            )
            p_g = geometry.camera_to_ground(p_c)
            sigma_x = np.hypot(
                np.abs(p_c[:, 0]) / z_hat * sigma_hat, z_hat * cfg.sigma_col_px / k.fx
            )

        # -- pitch implied by Z_h at the contact row -----------------------------
        pitch_meas = np.full(n, np.nan)
        pitch_sig = np.full(n, np.nan)
        p_sel = active[:, 2] & ~g_trunc & touches
        if p_sel.any():
            y_n = (v_b[p_sel] - k.cy) / k.fy
            th, th_ok = pitch_from_contact(y_n, z_all[p_sel, 2], h_cam)
            sg = pitch_sigma_from_depth_sigma(y_n, z_all[p_sel, 2], sig_z_all[p_sel, 2], th, h_cam)
            idx = np.flatnonzero(p_sel)[th_ok]
            pitch_meas[idx], pitch_sig[idx] = th[th_ok], sg[th_ok]

        sig_g_tot = np.where(active[:, 0], np.hypot(sig_z_all[:, 0], c_z_all[:, 0]), np.nan)
        sig_n_tot = np.where(active[:, 1], np.hypot(sig_z_all[:, 1], c_z_all[:, 1]), np.nan)
        return [
            Measurement3D(
                det_index=i,
                cls=classes[i],
                frame_id=frame_id,
                t_capture_ns=t_ns,
                z_cam_m=float(z_hat[i]),
                sigma_z_m=float(sigma_hat[i]),
                x_lat_m=float(p_g[i, 0]),
                z_fwd_m=float(p_g[i, 2]),
                sigma_x_m=float(sigma_x[i]),
                center_offset_m=0.5 * priors[i].length_m,
                z_ground_m=float(z_all[i, 0]),
                z_net_m=float(z_all[i, 1]),
                z_height_m=float(z_all[i, 2]),
                sigma_ground_m=float(sig_g_tot[i]),
                sigma_net_m=float(sig_n_tot[i]),
                sigma_height_m=float(sig_z_all[i, 2]),
                weights=w[i],
                chi2=float(chi2[i]),
                flags=flags[i],
                pitch_meas_rad=float(pitch_meas[i]),
                pitch_meas_sigma_rad=float(pitch_sig[i]),
            )
            for i in range(n)
        ]

    def _arbitrate(
        self,
        z: NDArray[np.float64],
        sig: NDArray[np.float64],
        c: NDArray[np.float64],
        active: NDArray[np.bool_],
        z_hat: float,
        sigma_hat: float,
        w: NDArray[np.float64],
        chi2: float,
        flags: FusionFlag,
    ) -> tuple[float, float, NDArray[np.float64], float, FusionFlag]:
        """Failed χ² test: with three cues drop the odd one out; with two, inflate σ."""
        flags |= FusionFlag.INCONSISTENT
        idx = np.flatnonzero(active)
        if idx.shape[0] == 3:
            best: tuple[float, int, tuple[float, float, NDArray[np.float64], float]] | None = None
            for drop in idx:
                pair = idx[idx != drop]
                res = fuse_correlated(z[pair], sig[pair], c[pair])
                if best is None or res[3] < best[0]:
                    best = (res[3], int(drop), res)
            assert best is not None
            if best[0] <= self.cfg.chi2_consistency:
                drop = best[1]
                w_out = np.zeros(3)
                w_out[idx[idx != drop]] = best[2][2]
                return best[2][0], best[2][1], w_out, best[2][3], flags | _CUE_REJECTED[drop]
        # Two cues disagreeing (or no consistent pair): nobody to arbitrate → keep the
        # BLUE but inflate its σ so downstream gating sees the disagreement.
        return z_hat, sigma_hat * float(np.sqrt(chi2 / (idx.shape[0] - 1))), w, chi2, flags


# ----------------------------------------------------------------------------
# Per-frame stage: pitch → affine solve → boxes → pitch feedback
# ----------------------------------------------------------------------------


@dataclass
class FusionOutput:
    frame_id: int
    t_ns: int
    measurements: list[Measurement3D]
    affine: AffineState | None
    pitch: PitchState
    solver: GroundSolverReport | None
    depth_lag_frames: int
    """``frame_id − depth.frame_id`` of the map used (0 when synchronous)."""


class MetricFusionStage:
    """Wires :class:`GroundSolver`, :class:`MetricFuser` and :class:`PitchEstimator`.

    Call :meth:`process` once per detector frame; pass the newest
    :class:`DepthMap` (possibly from an earlier frame — its lag is reported) or
    ``None`` to run on geometry + height prior only.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        nominal_geometry: PinholeGeometry,
        fusion_cfg: FusionConfig,
        road_cfg: RoadSampleConfig | None = None,
        affine_cfg: AffineFilterConfig | None = None,
        pitch_cfg: PitchFilterConfig | None = None,
        online_pitch: bool = True,
        timer: StageTimer | None = None,
        seed: int = 0,
    ) -> None:
        self.intrinsics = intrinsics
        self.nominal = nominal_geometry
        self.fuser = MetricFuser(fusion_cfg)
        self.solver = GroundSolver(
            sample_cfg=road_cfg if road_cfg is not None else RoadSampleConfig(),
            filter_cfg=affine_cfg if affine_cfg is not None else AffineFilterConfig(),
            seed=seed,
        )
        self.pitch = PitchEstimator(nominal_geometry.extrinsics, pitch_cfg)
        self.online_pitch = online_pitch
        self.timer = timer

    def geometry(self) -> PinholeGeometry:
        """Effective geometry: nominal intrinsics, filtered pitch, nominal noise model."""
        if not self.online_pitch:
            return self.nominal
        return PinholeGeometry(self.intrinsics, self.pitch.extrinsics(), self.nominal.noise)

    def process(
        self,
        frame_id: int,
        t_ns: int,
        boxes: NDArray[np.floating[Any]],
        classes: list[str],
        depth: DepthMap | None,
    ) -> FusionOutput:
        geo = self.geometry()
        b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        report: GroundSolverReport | None = None
        inv_map: NDArray[np.float32] | None = None
        resize: DepthResize | None = None
        lag = 0
        if depth is not None:
            inv_map = depth.inverse_depth()
            resize = depth.resize
            lag = frame_id - depth.frame_id
            if self.timer is not None:
                with self.timer.stage("fusion.solver"):
                    report = self.solver.update(inv_map, resize, geo, b, frame_id, t_ns)
            else:
                report = self.solver.update(inv_map, resize, geo, b, frame_id, t_ns)
        affine = self.solver.state
        if self.timer is not None:
            with self.timer.stage("fusion.boxes"):
                meas = self.fuser.fuse(b, classes, geo, inv_map, resize, affine, frame_id, t_ns)
        else:
            meas = self.fuser.fuse(b, classes, geo, inv_map, resize, affine, frame_id, t_ns)
        if self.online_pitch:
            th = np.array([m.pitch_meas_rad for m in meas], dtype=np.float64)
            sg = np.array([m.pitch_meas_sigma_rad for m in meas], dtype=np.float64)
            self.pitch.update(th, sg, t_ns)
        return FusionOutput(
            frame_id=frame_id,
            t_ns=t_ns,
            measurements=meas,
            affine=affine,
            pitch=self.pitch.state,
            solver=report,
            depth_lag_frames=lag,
        )
