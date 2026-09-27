"""Relative kinematics for collision checks: CPA and conservative TTC with σ propagation.

Inputs are in the ego ground frame ``(X right, Z forward)`` with the origin at the
ego front centre (callers shift the tracker's camera-origin positions by
``ego_front_m``), relative position ``p`` and relative velocity ``v = V_obj − v_ego``::

    t_cpa = −(p·v)/‖v‖²      (clamped to 0 when receding or ‖v‖ ≈ 0)
    d_cpa = ‖p + v·t_cpa‖

At the CPA ``r = p + v·t_cpa ⟂ v``, so the Jacobians simplify to ``∂d/∂p = r̂`` and
``∂d/∂v = t_cpa·r̂``; ``σ_d`` uses the full ``[p, v]`` covariance including the
cross block. The longitudinal lower bound on the time to contact is::

    TTC_low = max(gap − k·σ_Z, 0) / (−V_Z + k·σ_VZ)     (∞ if the denominator ≤ 0)

where ``gap = Z − L_obj/2``. All arithmetic is scalar ``math`` (a few µs per track).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from percepcion3d.tracking.tracker3d import Track3D

_EPS_V2 = 1e-6


@dataclass(frozen=True)
class RelativeKinematics:
    track_id: int
    cls: str
    t_ns: int
    p_xz: tuple[float, float]
    """Object centre relative to the ego front centre (m)."""
    v_xz: tuple[float, float]
    """``V_obj − v_ego`` (m/s)."""
    cov_p: tuple[float, float, float]
    """``(σ_XX, σ_XZ, σ_ZZ)`` of the position (m²)."""
    cov_v: tuple[float, float, float]
    """``(σ_VxVx, σ_VxVz, σ_VzVz)`` of the relative velocity (m²/s²)."""
    cov_pv: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    """``(X·Vx, X·Vz, Z·Vx, Z·Vz)`` cross covariance."""
    n_updates: int = 1_000_000
    time_since_update_s: float = 0.0


def kinematics_from_track(tr: Track3D, ego_front_m: float) -> RelativeKinematics:
    """Adapter from :class:`Track3D` (camera-origin centre) to the ego-front frame."""
    c: NDArray[np.float64] = tr.cov
    cv: NDArray[np.float64] = tr.cov_vel_rel
    return RelativeKinematics(
        track_id=tr.track_id,
        cls=tr.cls,
        t_ns=tr.t_ns,
        p_xz=(float(tr.position_xz[0]), float(tr.position_xz[1]) - ego_front_m),
        v_xz=(float(tr.velocity_rel_xz[0]), float(tr.velocity_rel_xz[1])),
        cov_p=(float(c[0, 0]), float(c[0, 1]), float(c[1, 1])),
        cov_v=(float(cv[0, 0]), float(cv[0, 1]), float(cv[1, 1])),
        cov_pv=(float(c[0, 2]), float(c[0, 3]), float(c[1, 2]), float(c[1, 3])),
        n_updates=tr.n_updates,
        time_since_update_s=tr.time_since_update_s,
    )


@dataclass(frozen=True)
class CpaResult:
    t_cpa_s: float
    d_cpa_m: float
    sigma_d_cpa_m: float
    closing: bool
    """``p·v < 0``: the range is decreasing now."""


def cpa(k: RelativeKinematics) -> CpaResult:
    """Closest point of approach of the object centre to the ego front centre."""
    px, pz = k.p_xz
    vx, vz = k.v_xz
    pv = px * vx + pz * vz
    v2 = vx * vx + vz * vz
    closing = pv < 0.0
    if v2 < _EPS_V2 or not closing:
        t = 0.0
        rx, rz = px, pz
    else:
        t = -pv / v2
        rx, rz = px + vx * t, pz + vz * t
    d = math.hypot(rx, rz)
    if d < 1e-9:
        return CpaResult(t, d, 0.0, closing)
    ux, uz = rx / d, rz / d
    # J = [ux, uz, t·ux, t·uz] over [X, Z, Vx, Vz]
    sxx, sxz, szz = k.cov_p
    vxx, vxz, vzz = k.cov_v
    cxa, cxb, cza, czb = k.cov_pv
    var = (
        ux * ux * sxx + 2.0 * ux * uz * sxz + uz * uz * szz
        + t * t * (ux * ux * vxx + 2.0 * ux * uz * vxz + uz * uz * vzz)
        + 2.0 * t * (ux * ux * cxa + ux * uz * (cxb + cza) + uz * uz * czb)
    )  # fmt: skip
    return CpaResult(t, d, math.sqrt(max(var, 0.0)), closing)


def ttc_low_s(k: RelativeKinematics, half_length_m: float, k_sigma: float) -> tuple[float, float]:
    """``(TTC_low, gap_low)``: conservative longitudinal time to contact and gap (m)."""
    gap_low = k.p_xz[1] - half_length_m - k_sigma * math.sqrt(max(k.cov_p[2], 0.0))
    closing_hi = -k.v_xz[1] + k_sigma * math.sqrt(max(k.cov_v[2], 0.0))
    if closing_hi <= 1e-6:
        return math.inf, gap_low
    return max(gap_low, 0.0) / closing_hi, gap_low
