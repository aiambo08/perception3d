"""3D tracking: 2D identities from :class:`ByteTracker`, metric state from :class:`CvKalman`.

Per frame (:meth:`Tracker3D.step`):

1. ego-motion ``Δ`` since the previous frame from the injected provider;
2. 2D association (ByteTrack) → which detection belongs to which track id;
3. every 3D filter of a live 2D track is predicted to ``t`` with ``Δ``;
4. matched tracks with a valid :class:`PositionMeasurement` are updated
   (χ² gate on the NIS; after ``max_rejects`` consecutive rejections the filter is
   re-initialised from the measurement, which handles a wrong initial association or
   a depth jump the gate cannot bridge);
5. motion label ``STATIC`` / ``MOVING`` with hysteresis on the over-ground
   speed, only when the ego provider is absolute.

Identity is owned by the 2D tracker: the 3D stage never merges or splits tracks,
so its ID-switch count equals ByteTrack's by construction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from percepcion3d.depth.fusion import Measurement3D
from percepcion3d.tracking.byte_tracker import ByteTrackConfig, ByteTracker, TrackState
from percepcion3d.tracking.ego_motion import EgoMotionProvider, ZeroEgoMotion
from percepcion3d.tracking.kalman_filter import CvKalman, EgoDelta, predict_batch, update_batch


@dataclass(frozen=True)
class PositionMeasurement:
    """Object centre in the ground frame with its 2×2 covariance."""

    x_m: float
    z_m: float
    cov: NDArray[np.float64]
    t_ns: int

    @property
    def xz(self) -> NDArray[np.float64]:
        return np.array([self.x_m, self.z_m], dtype=np.float64)


def position_covariance(
    x_m: float, z_m: float, sigma_x_m: float, sigma_z_m: float
) -> NDArray[np.float64]:
    """Covariance of a point placed along a viewing ray.

    ``X = a·Z + ε_col`` with ``a = X/Z``: the depth error moves the point along the
    ray, so ``cov(X, Z) = a·σ_Z²``; ``σ_X²`` already contains ``a²·σ_Z²`` (fusion),
    the remainder is the column noise. The correlation is clipped to keep ``R`` SPD.
    """
    a = x_m / z_m if z_m > 0.0 else 0.0
    vz = sigma_z_m**2
    vx = max(sigma_x_m**2, a * a * vz + 1e-6)
    c = float(np.clip(a * vz, -0.999 * np.sqrt(vx * vz), 0.999 * np.sqrt(vx * vz)))
    return np.array([[vx, c], [c, vz]], dtype=np.float64)


def measurement_from_fusion(m: Measurement3D) -> PositionMeasurement | None:
    """Object centre (contact + half the class length) from an F4 measurement."""
    if not m.valid or not np.isfinite(m.sigma_z_m) or not np.isfinite(m.x_lat_m):
        return None
    z = m.z_center_fwd_m
    return PositionMeasurement(
        m.x_lat_m, z, position_covariance(m.x_lat_m, z, m.sigma_x_m, m.sigma_z_m), m.t_capture_ns
    )


class MotionState(Enum):
    UNKNOWN = "unknown"
    STATIC = "static"
    MOVING = "moving"


@dataclass(frozen=True)
class ClassDynamics:
    q: float
    """CWNA spectral density (m²/s³)."""
    sigma_v0_mps: float
    """Prior σ of the over-ground velocity at track birth."""


@dataclass(frozen=True)
class Tracker3DConfig:
    dynamics: Mapping[str, ClassDynamics] = field(
        default_factory=lambda: {
            "vehicle": ClassDynamics(q=1.0, sigma_v0_mps=15.0),
            "person": ClassDynamics(q=0.5, sigma_v0_mps=3.0),
            "cycle": ClassDynamics(q=1.0, sigma_v0_mps=8.0),
        }
    )
    default_dynamics: ClassDynamics = ClassDynamics(q=1.0, sigma_v0_mps=15.0)
    gate_chi2: float = 13.82
    """NIS gate (χ² 2 dof, 99.9 %)."""
    robust_chi2: float | None = 5.99
    """Above this NIS the measurement is down-weighted (``R·NIS/robust_chi2``); ``None``
    disables it (plain gated KF)."""
    max_rejects: int = 3
    static_below_mps: float = 1.0
    moving_above_mps: float = 2.0
    static_chi2: float = 5.99
    """Over-ground velocity consistent with zero (χ² 2 dof, 95 %) → static candidate."""
    moving_chi2: float = 13.82
    """Moving needs ``speed > moving_above_mps`` *and* zero rejected at 99.9 %."""
    motion_hold_s: float = 0.3
    """A new motion label needs its condition to hold this long (hysteresis in time)."""
    min_motion_age_s: float = 0.3
    byte: ByteTrackConfig = field(default_factory=ByteTrackConfig)

    def dynamics_for(self, cls: str) -> ClassDynamics:
        return self.dynamics.get(self.byte.group(cls), self.default_dynamics)


def load_tracker_config(path: Path | str) -> Tracker3DConfig:
    with Path(path).open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    bt: dict[str, Any] = dict(raw.get("byte_track", {}))
    groups = bt.pop("class_groups", None)
    byte = ByteTrackConfig(**bt)
    if groups:
        byte = replace(byte, class_groups=dict(groups))
    dyn_raw: dict[str, dict[str, float]] = raw.get("dynamics", {})
    dyn = {k: ClassDynamics(**v) for k, v in dyn_raw.items()}
    default = dyn.pop("default", ClassDynamics(q=1.0, sigma_v0_mps=15.0))
    return Tracker3DConfig(
        dynamics=dyn, default_dynamics=default, byte=byte, **dict(raw.get("filter", {}))
    )


@dataclass(slots=True)
class Track3D:
    track_id: int
    cls: str
    t_ns: int
    box: NDArray[np.float64]
    det_index: int
    """Detection matched this frame (``-1`` while coasting)."""
    position_xz: NDArray[np.float64]
    """Object centre ``(X, Z)`` in the current ground frame (m)."""
    velocity_rel_xz: NDArray[np.float64]
    """``V_obj − v_ego`` in the current axes (m/s) — the quantity TTC needs."""
    velocity_abs_xz: NDArray[np.float64]
    """Over-ground velocity; equals ``velocity_rel_xz`` with a non-absolute ego provider."""
    cov: NDArray[np.float64]
    """4×4 state covariance ``[X, Z, V_X, V_Z]`` (over-ground velocity block)."""
    cov_vel_rel: NDArray[np.float64]
    age_s: float
    n_updates: int
    time_since_update_s: float
    motion: MotionState
    last_nis: float


@dataclass
class _Track3DState:
    kf: CvKalman
    t_birth_ns: int
    t_update_ns: int
    n_updates: int = 1
    rejects: int = 0
    last_nis: float = float("nan")
    motion: MotionState = MotionState.UNKNOWN
    pending: MotionState = MotionState.UNKNOWN
    t_pending_ns: int = 0


class Tracker3D:
    def __init__(
        self, cfg: Tracker3DConfig | None = None, ego: EgoMotionProvider | None = None
    ) -> None:
        self.cfg = cfg if cfg is not None else Tracker3DConfig()
        self.ego: EgoMotionProvider = ego if ego is not None else ZeroEgoMotion()
        self.byte = ByteTracker(self.cfg.byte)
        self._state: dict[int, _Track3DState] = {}
        self._t_ns: int | None = None
        self.last_ego = EgoDelta(dt_s=0.0)

    def _init(self, m: PositionMeasurement, cls: str, t_ns: int) -> _Track3DState:
        d = self.cfg.dynamics_for(cls)
        kf = CvKalman(m.xz, m.cov, m.t_ns, d.q, d.sigma_v0_mps)
        if m.t_ns < t_ns:
            kf.predict(t_ns, None)
        return _Track3DState(kf, t_ns, t_ns)

    def _motion(self, st: _Track3DState, t_ns: int) -> None:
        cfg = self.cfg
        if not self.last_ego.absolute or (t_ns - st.t_birth_ns) * 1e-9 < cfg.min_motion_age_s:
            st.motion = st.pending = MotionState.UNKNOWN
            return
        vx, vz = float(st.kf.x[2]), float(st.kf.x[3])
        pxx, pxz, pzz = float(st.kf.P[2, 2]), float(st.kf.P[2, 3]), float(st.kf.P[3, 3])
        d2 = (pzz * vx * vx - 2.0 * pxz * vx * vz + pxx * vz * vz) / (pxx * pzz - pxz * pxz)
        speed = float(np.hypot(vx, vz))
        if speed < cfg.static_below_mps or d2 < cfg.static_chi2:
            cand = MotionState.STATIC
        elif speed > cfg.moving_above_mps and d2 > cfg.moving_chi2:
            cand = MotionState.MOVING
        else:
            cand = st.motion
        if cand is st.motion:
            st.pending = cand
            return
        if cand is not st.pending:
            st.pending, st.t_pending_ns = cand, t_ns
        if st.motion is MotionState.UNKNOWN or (t_ns - st.t_pending_ns) * 1e-9 >= cfg.motion_hold_s:
            st.motion = cand

    def step(
        self,
        t_ns: int,
        boxes: NDArray[np.floating[Any]],
        scores: NDArray[np.floating[Any]],
        classes: Sequence[str],
        measurements: Sequence[PositionMeasurement | None],
    ) -> list[Track3D]:
        """Advance to ``t_ns``; ``measurements[i]`` is the metric position of detection ``i``."""
        if len(measurements) != len(classes):
            raise ValueError("one measurement (or None) per detection")
        ego = self.ego.delta(t_ns if self._t_ns is None else self._t_ns, t_ns)
        self.last_ego = ego
        self._t_ns = t_ns

        tracks2d = self.byte.update(t_ns, boxes, scores, classes)
        alive = {t.track_id for t in tracks2d}
        for tid in [k for k in self._state if k not in alive]:
            del self._state[tid]

        live = [(tr, self._state[tr.track_id]) for tr in tracks2d if tr.track_id in self._state]
        if live:
            kfs = [st.kf for _, st in live]
            x, p = predict_batch(
                np.stack([k.x for k in kfs]),
                np.stack([k.P for k in kfs]),
                np.array([(t_ns - k.t_ns) * 1e-9 for k in kfs]),
                np.array([k.q for k in kfs]),
                ego,
            )
            for i, k in enumerate(kfs):
                k.x, k.P, k.t_ns = x[i], p[i], t_ns

        upd = [
            (st, m)
            for tr, st in live
            if tr.det_index >= 0 and (m := measurements[tr.det_index]) is not None
        ]
        if upd:
            kfs = [st.kf for st, _ in upd]
            x, p, ok, nis = update_batch(
                np.stack([k.x for k in kfs]),
                np.stack([k.P for k in kfs]),
                np.stack([m.xz for _, m in upd]),
                np.stack([m.cov for _, m in upd]),
                np.array([k.q for k in kfs]),
                np.array([max(0.0, (t_ns - m.t_ns) * 1e-9) for _, m in upd]),
                self.cfg.gate_chi2,
                self.cfg.robust_chi2,
            )
            for i, (su, _) in enumerate(upd):
                su.kf.x, su.kf.P, su.last_nis = x[i], p[i], float(nis[i])
                if ok[i]:
                    su.rejects = 0
                    su.n_updates += 1
                    su.t_update_ns = t_ns
                else:
                    su.rejects += 1

        v_ego = np.asarray(ego.velocity_xz, dtype=np.float64)
        var_ego = ego.sigma_velocity_mps**2 * np.eye(2)
        out: list[Track3D] = []
        for tr in tracks2d:
            st: _Track3DState | None = self._state.get(tr.track_id)
            m = measurements[tr.det_index] if tr.det_index >= 0 else None
            if m is not None and (st is None or st.rejects >= self.cfg.max_rejects):
                st = self._init(m, tr.cls, t_ns)
                self._state[tr.track_id] = st
            if st is None or tr.state is TrackState.TENTATIVE:
                continue
            self._motion(st, t_ns)
            kx, kp = st.kf.x, st.kf.P
            box = tr.box
            out.append(
                Track3D(
                    track_id=tr.track_id,
                    cls=tr.cls,
                    t_ns=t_ns,
                    box=box,
                    det_index=tr.det_index,
                    position_xz=kx[:2].copy(),
                    velocity_rel_xz=kx[2:] - v_ego,
                    velocity_abs_xz=kx[2:].copy(),
                    cov=kp.copy(),
                    cov_vel_rel=kp[2:, 2:] + var_ego,
                    age_s=(t_ns - st.t_birth_ns) * 1e-9,
                    n_updates=st.n_updates,
                    time_since_update_s=(t_ns - st.t_update_ns) * 1e-9,
                    motion=st.motion,
                    last_nis=st.last_nis,
                )
            )
        return out
