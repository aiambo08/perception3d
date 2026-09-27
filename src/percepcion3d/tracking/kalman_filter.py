"""Constant-velocity Kalman filter in the ground plane with variable ``Δt`` and ego-motion.

State ``x = [X, Z, V_X, V_Z]`` in the ground frame G of the *current* camera pose
(X right, Z forward, see :mod:`percepcion3d.camera.geometry`):

* ``(X, Z)`` — object position relative to the camera;
* ``(V_X, V_Z)`` — object velocity **over ground**, expressed in the current
  camera axes. With :class:`~percepcion3d.tracking.ego_motion.ZeroEgoMotion`
  the camera is treated as fixed, so this velocity degenerates to the relative
  velocity and the filter is the plain relative-motion CV model.

Keeping the over-ground velocity in the state (instead of the relative one)
makes the CV model valid while the ego vehicle turns: a static object has
``V = 0`` whatever the ego yaw rate, whereas its *relative* velocity rotates
with the camera and would look like an acceleration. The relative velocity
needed by TTC is derived as ``V − v_ego`` (:meth:`CvKalman.relative_velocity`).

Prediction from ``t_{k-1}`` to ``t_k`` (ego rotated by ``ψ`` left-positive and
translated by ``t`` expressed in the previous axes)::

    p⁻ = Rᵀ(ψ) · (p + V·Δt − t)        V⁻ = Rᵀ(ψ) · V
    P⁻ = A P Aᵀ + Q_c(Δt) + diag(σ_t², σ_t², 0, 0),   A = blkdiag(Rᵀ, Rᵀ) · F(Δt)

``Q_c`` is the *continuous* white-noise-acceleration model
``q · [[Δt³/3, Δt²/2], [Δt²/2, Δt]]`` per axis, whose integrated noise does not
depend on how an interval is split into steps (the discrete ``Δt⁴/4`` model
does), which matters with the jittered timestamps of the dual-rate pipeline.

Measurements are positions ``(X, Z)`` with a full 2×2 covariance (the
converted-measurement design I of ``docs/00_analisis_critico.md`` §3.4). A
measurement ``lag`` seconds older than the state (depth map from an earlier
frame, R6) is applied by retrodiction: ``z = p − V·lag + w`` with
``H = [I, −lag·I]`` and ``R ← R + q·lag³/3·I``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_EYE2 = np.eye(2)


def rotation_2d(yaw_rad: float) -> NDArray[np.float64]:
    """Axes of a frame rotated ``yaw_rad`` to the left, as columns in ``(X, Z)`` coordinates.

    ``p_new = rotation_2d(ψ).T @ (p_old − t)`` re-expresses a point after the
    camera turned left by ``ψ``: an object straight ahead drifts to the right.
    """
    c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def cv_transition(dt_s: float) -> NDArray[np.float64]:
    f = np.eye(4)
    f[0, 2] = f[1, 3] = dt_s
    return f


def cwna_process_noise(dt_s: float, q: float) -> NDArray[np.float64]:
    """Continuous white-noise acceleration ``Q(Δt)`` for ``[X, Z, V_X, V_Z]`` (``q`` in m²/s³)."""
    d = float(dt_s)
    a, b, c = q * d**3 / 3.0, q * d**2 / 2.0, q * d
    return np.array(
        [[a, 0.0, b, 0.0], [0.0, a, 0.0, b], [b, 0.0, c, 0.0], [0.0, b, 0.0, c]],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class EgoDelta:
    """Camera motion between two capture timestamps, in the ground plane."""

    dt_s: float
    translation_xz: tuple[float, float] = (0.0, 0.0)
    """Displacement of the camera expressed in the axes of the *earlier* pose (m)."""
    yaw_rad: float = 0.0
    """Rotation of the camera about the up axis, left positive."""
    velocity_xz: tuple[float, float] = (0.0, 0.0)
    """Camera velocity over ground at the later timestamp, in its own axes (m/s)."""
    sigma_translation_m: float = 0.0
    sigma_velocity_mps: float = 0.0
    absolute: bool = False
    """True when the provider measures real ego-motion (over-ground velocities are meaningful)."""


def cwna_process_noise_batch(
    dt_s: NDArray[np.float64], q: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``(N, 4, 4)`` stack of :func:`cwna_process_noise`."""
    d = np.asarray(dt_s, dtype=np.float64)
    qq = np.asarray(q, dtype=np.float64)
    out = np.zeros((d.size, 4, 4))
    a, b, c = qq * d**3 / 3.0, qq * d**2 / 2.0, qq * d
    out[:, 0, 0] = out[:, 1, 1] = a
    out[:, 0, 2] = out[:, 2, 0] = out[:, 1, 3] = out[:, 3, 1] = b
    out[:, 2, 2] = out[:, 3, 3] = c
    return out


def predict_batch(
    x: NDArray[np.float64],
    p: NDArray[np.float64],
    dt_s: NDArray[np.float64],
    q: NDArray[np.float64],
    ego: EgoDelta | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Predict ``N`` filters ``x (N, 4)``, ``P (N, 4, 4)`` by their own ``Δt`` under one ego delta."""
    d = np.asarray(dt_s, dtype=np.float64)
    if np.any(d < 0.0):
        raise ValueError(f"predict backwards in time (min Δt {d.min() * 1e3:.2f} ms)")
    f = np.broadcast_to(np.eye(4), (d.size, 4, 4)).copy()
    f[:, 0, 2] = f[:, 1, 3] = d
    x_new = np.einsum("nij,nj->ni", f, x)
    if ego is not None and (ego.yaw_rad != 0.0 or ego.translation_xz != (0.0, 0.0)):
        rt = rotation_2d(ego.yaw_rad).T
        blk = np.zeros((4, 4))
        blk[:2, :2] = blk[2:, 2:] = rt
        f = blk @ f
        x_new = x_new @ blk.T
        x_new[:, :2] -= rt @ np.asarray(ego.translation_xz, dtype=np.float64)
    p_new = f @ p @ np.transpose(f, (0, 2, 1)) + cwna_process_noise_batch(d, q)
    if ego is not None and ego.sigma_translation_m > 0.0:
        p_new[:, 0, 0] += ego.sigma_translation_m**2
        p_new[:, 1, 1] += ego.sigma_translation_m**2
    return x_new, 0.5 * (p_new + np.transpose(p_new, (0, 2, 1)))


def update_batch(
    x: NDArray[np.float64],
    p: NDArray[np.float64],
    z: NDArray[np.float64],
    r: NDArray[np.float64],
    q: NDArray[np.float64],
    lag_s: NDArray[np.float64] | None = None,
    gate_chi2: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
    """Joseph-form position update of ``N`` filters; rows with ``NIS > gate_chi2`` are left
    untouched. Returns ``(x, P, accepted, NIS)``."""
    n = x.shape[0]
    lag = np.zeros(n) if lag_s is None else np.asarray(lag_s, dtype=np.float64)
    h = np.zeros((n, 2, 4))
    h[:, 0, 0] = h[:, 1, 1] = 1.0
    h[:, 0, 2] = h[:, 1, 3] = -lag
    rr = np.asarray(r, dtype=np.float64) + (np.asarray(q) * lag**3 / 3.0)[:, None, None] * _EYE2
    ht = np.transpose(h, (0, 2, 1))
    nu = np.asarray(z, dtype=np.float64) - np.einsum("nij,nj->ni", h, x)
    ph = p @ ht
    s = h @ ph + rr
    det = s[:, 0, 0] * s[:, 1, 1] - s[:, 0, 1] * s[:, 1, 0]
    s_inv = np.empty_like(s)
    s_inv[:, 0, 0] = s[:, 1, 1] / det
    s_inv[:, 1, 1] = s[:, 0, 0] / det
    s_inv[:, 0, 1] = -s[:, 0, 1] / det
    s_inv[:, 1, 0] = -s[:, 1, 0] / det
    nis = np.einsum("ni,nij,nj->n", nu, s_inv, nu)
    ok = np.ones(n, dtype=bool) if gate_chi2 is None else nis <= gate_chi2
    k = ph @ s_inv
    x_new = x + np.einsum("nij,nj->ni", k, nu)
    ikh = np.eye(4) - k @ h
    p_new = ikh @ p @ np.transpose(ikh, (0, 2, 1)) + k @ rr @ np.transpose(k, (0, 2, 1))
    p_new = 0.5 * (p_new + np.transpose(p_new, (0, 2, 1)))
    x_out = np.where(ok[:, None], x_new, x)
    p_out = np.where(ok[:, None, None], p_new, p)
    return x_out, p_out, ok, nis


class CvKalman:
    """One track's CV filter; see the module docstring for the frame conventions."""

    __slots__ = ("P", "q", "t_ns", "x")

    def __init__(
        self,
        z_xz: NDArray[np.float64],
        r_xz: NDArray[np.float64],
        t_ns: int,
        q: float,
        sigma_v0_mps: float,
        v0_xz: NDArray[np.float64] | None = None,
    ) -> None:
        self.x = np.zeros(4)
        self.x[:2] = z_xz
        if v0_xz is not None:
            self.x[2:] = v0_xz
        self.P = np.zeros((4, 4))
        self.P[:2, :2] = r_xz
        self.P[2, 2] = self.P[3, 3] = sigma_v0_mps**2
        self.q = float(q)
        self.t_ns = int(t_ns)

    @property
    def position(self) -> NDArray[np.float64]:
        return self.x[:2]

    @property
    def velocity(self) -> NDArray[np.float64]:
        """Over-ground velocity (relative when the ego provider is not absolute)."""
        return self.x[2:]

    def relative_velocity(self, ego: EgoDelta) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``V − v_ego`` and its covariance (ego velocity noise added isotropically)."""
        v = self.x[2:] - np.asarray(ego.velocity_xz, dtype=np.float64)
        cov = self.P[2:, 2:] + ego.sigma_velocity_mps**2 * _EYE2
        return v, cov

    def predict(self, t_ns: int, ego: EgoDelta | None = None) -> None:
        dt = np.array([(int(t_ns) - self.t_ns) * 1e-9])
        x, p = predict_batch(self.x[None], self.P[None], dt, np.array([self.q]), ego)
        self.x, self.P = x[0], p[0]
        self.t_ns = int(t_ns)

    def update(
        self,
        z_xz: NDArray[np.float64],
        r_xz: NDArray[np.float64],
        lag_s: float = 0.0,
        gate_chi2: float | None = None,
    ) -> tuple[bool, float]:
        """Joseph-form update; returns ``(accepted, NIS)`` (rejected when ``NIS > gate_chi2``)."""
        x, p, ok, nis = update_batch(
            self.x[None],
            self.P[None],
            np.asarray(z_xz, dtype=np.float64)[None],
            np.asarray(r_xz, dtype=np.float64)[None],
            np.array([self.q]),
            np.array([lag_s]),
            gate_chi2,
        )
        self.x, self.P = x[0], p[0]
        return bool(ok[0]), float(nis[0])
