"""F6: CPA / TTC_low closed forms and σ propagation (CPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from percepcion3d.safety.kinematics import (
    RelativeKinematics,
    cpa,
    kinematics_from_track,
    ttc_low_s,
)
from percepcion3d.tracking.tracker3d import MotionState, Track3D


def _kin(
    p: tuple[float, float],
    v: tuple[float, float],
    sp: float = 0.0,
    sv: float = 0.0,
    cov_pv: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> RelativeKinematics:
    return RelativeKinematics(
        1, "car", 0, p, v, (sp * sp, 0.0, sp * sp), (sv * sv, 0.0, sv * sv), cov_pv
    )


def test_cpa_closed_form_lateral_pass() -> None:
    # Object 3 m to the left, 20 m ahead, closing straight along -Z at 10 m/s.
    c = cpa(_kin((-3.0, 20.0), (0.0, -10.0)))
    assert c.closing
    assert c.t_cpa_s == pytest.approx(2.0)
    assert c.d_cpa_m == pytest.approx(3.0)


def test_cpa_head_on_is_zero_distance() -> None:
    c = cpa(_kin((0.0, 30.0), (0.0, -15.0)))
    assert c.t_cpa_s == pytest.approx(2.0)
    assert c.d_cpa_m == pytest.approx(0.0)


def test_cpa_receding_or_stationary_clamps_to_now() -> None:
    rec = cpa(_kin((2.0, 10.0), (0.0, 5.0)))
    assert not rec.closing and rec.t_cpa_s == 0.0
    assert rec.d_cpa_m == pytest.approx(math.hypot(2.0, 10.0))
    still = cpa(_kin((0.0, 4.0), (0.0, 0.0)))
    assert still.t_cpa_s == 0.0 and still.d_cpa_m == pytest.approx(4.0)


def test_cpa_sigma_matches_monte_carlo() -> None:
    rng = np.random.default_rng(0)
    p, v, sp, sv = (-3.0, 20.0), (0.4, -10.0), 0.3, 0.2
    k = _kin(p, v, sp, sv)
    c = cpa(k)
    n = 20000
    ps = np.array(p) + sp * rng.standard_normal((n, 2))
    vs = np.array(v) + sv * rng.standard_normal((n, 2))
    t = -np.einsum("ij,ij->i", ps, vs) / np.einsum("ij,ij->i", vs, vs)
    d = np.linalg.norm(ps + vs * t[:, None], axis=1)
    assert c.sigma_d_cpa_m == pytest.approx(float(d.std()), rel=0.1)
    # Position-velocity cross terms change σ_d: positive X·Vx correlation with t>0 adds.
    k2 = _kin(p, v, sp, sv, cov_pv=(0.5 * sp * sv, 0.0, 0.0, 0.5 * sp * sv))
    assert cpa(k2).sigma_d_cpa_m != pytest.approx(c.sigma_d_cpa_m)


def test_ttc_low_is_conservative_and_inf_when_not_closing() -> None:
    k = _kin((0.0, 22.15), (0.0, -10.0), sp=1.0, sv=0.5)
    ttc, gap = ttc_low_s(k, half_length_m=2.15, k_sigma=1.0)
    assert gap == pytest.approx(22.15 - 2.15 - 1.0)
    assert ttc == pytest.approx(19.0 / 10.5)
    assert ttc < (22.15 - 2.15) / 10.0
    ttc_inf, _ = ttc_low_s(_kin((0.0, 10.0), (0.0, 0.2), sv=0.1), 2.0, 1.0)
    assert ttc_inf == math.inf
    ttc0, _ = ttc_low_s(_kin((0.0, 1.0), (0.0, -5.0)), 2.0, 1.0)
    assert ttc0 == 0.0


def test_kinematics_from_track_shifts_to_ego_front_and_copies_covariances() -> None:
    cov = np.diag([0.1, 0.4, 0.05, 0.09])
    cov[0, 2] = cov[2, 0] = 0.01
    tr = Track3D(
        track_id=7,
        cls="truck",
        t_ns=5,
        box=np.zeros(4),
        det_index=0,
        position_xz=np.array([1.0, 20.0]),
        velocity_rel_xz=np.array([0.2, -8.0]),
        velocity_abs_xz=np.array([0.2, 4.0]),
        cov=cov,
        cov_vel_rel=cov[2:, 2:] + 0.04 * np.eye(2),
        age_s=1.0,
        n_updates=12,
        time_since_update_s=0.03,
        motion=MotionState.MOVING,
        last_nis=1.0,
    )
    k = kinematics_from_track(tr, ego_front_m=1.5)
    assert k.track_id == 7 and k.cls == "truck" and k.t_ns == 5
    assert k.p_xz == (1.0, 18.5) and k.v_xz == (0.2, -8.0)
    assert k.cov_p == (0.1, 0.0, 0.4)
    assert k.cov_v == pytest.approx((0.09, 0.0, 0.13))
    assert k.cov_pv == (0.01, 0.0, 0.0, 0.0)
    assert k.n_updates == 12 and k.time_since_update_s == 0.03
