"""Ego-motion providers: camera motion between two capture timestamps.

Every provider returns an :class:`~percepcion3d.tracking.kalman_filter.EgoDelta`
in the ground frame G (X right, Z forward, yaw left-positive). Implementations:

* :class:`ZeroEgoMotion` — camera assumed fixed; tracks then carry *relative*
  velocities (enough for TTC, not for static/moving classification).
* :class:`ConstantEgoMotion` — fixed forward/lateral speed and yaw rate
  (tests, simulations, a vehicle with only a speedometer).
* :class:`OxtsEgoMotion` — KITTI OXTS/INS records (reference ego-motion for
  validation); velocities and yaw rate interpolated linearly between records.

The translation over ``Δt`` is the mean velocity rotated by half the yaw
increment (mid-point rule of a constant-turn-rate arc), expressed in the axes
of the earlier pose.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import astuple
from pathlib import Path
from typing import Protocol

import numpy as np

from percepcion3d.io.sources import OxtsRecord, parse_oxts_line
from percepcion3d.tracking.kalman_filter import EgoDelta, rotation_2d


class EgoMotionProvider(Protocol):
    def delta(self, t0_ns: int, t1_ns: int) -> EgoDelta: ...


def _arc_delta(
    dt: float,
    v0: tuple[float, float],
    v1: tuple[float, float],
    w0: float,
    w1: float,
    sigma_v: float,
) -> EgoDelta:
    yaw = 0.5 * (w0 + w1) * dt
    v_mean = 0.5 * (np.asarray(v0, dtype=np.float64) + np.asarray(v1, dtype=np.float64))
    t = rotation_2d(0.5 * yaw) @ v_mean * dt
    return EgoDelta(
        dt_s=dt,
        translation_xz=(float(t[0]), float(t[1])),
        yaw_rad=float(yaw),
        velocity_xz=(float(v1[0]), float(v1[1])),
        sigma_translation_m=sigma_v * dt,
        sigma_velocity_mps=sigma_v,
        absolute=True,
    )


class ZeroEgoMotion:
    def delta(self, t0_ns: int, t1_ns: int) -> EgoDelta:
        return EgoDelta(dt_s=(t1_ns - t0_ns) * 1e-9)


class ConstantEgoMotion:
    def __init__(
        self,
        v_forward_mps: float,
        v_right_mps: float = 0.0,
        yaw_rate_rps: float = 0.0,
        sigma_velocity_mps: float = 0.0,
    ) -> None:
        self.v = (float(v_right_mps), float(v_forward_mps))
        self.w = float(yaw_rate_rps)
        self.sigma_v = float(sigma_velocity_mps)

    def delta(self, t0_ns: int, t1_ns: int) -> EgoDelta:
        return _arc_delta((t1_ns - t0_ns) * 1e-9, self.v, self.v, self.w, self.w, self.sigma_v)


class OxtsEgoMotion:
    """Ego-motion from OXTS records stamped with the camera timestamps.

    OXTS velocities are in the vehicle frame (``vf`` forward, ``vl`` left, ``wu``
    about up); in G that is ``v = (−vl, vf)`` and yaw rate ``wu``. The camera sits
    ``lever_arm_fwd_m`` ahead of the IMU, which adds ``(−wu·L, 0)`` to its velocity.
    """

    def __init__(
        self,
        t_ns: Sequence[int],
        records: Sequence[OxtsRecord],
        sigma_velocity_mps: float = 0.05,
        lever_arm_fwd_m: float = 0.0,
    ) -> None:
        if len(t_ns) != len(records) or not records:
            raise ValueError("need one OXTS record per timestamp")
        self._t = np.asarray(t_ns, dtype=np.float64)
        wu = np.array([r.wu for r in records], dtype=np.float64)
        self._vx = np.array([-r.vl for r in records], dtype=np.float64) - wu * lever_arm_fwd_m
        self._vz = np.array([r.vf for r in records], dtype=np.float64)
        self._wu = wu
        self.sigma_v = float(sigma_velocity_mps)

    def _at(self, t_ns: int) -> tuple[tuple[float, float], float]:
        t = float(t_ns)
        vx = float(np.interp(t, self._t, self._vx))
        vz = float(np.interp(t, self._t, self._vz))
        return (vx, vz), float(np.interp(t, self._t, self._wu))

    def delta(self, t0_ns: int, t1_ns: int) -> EgoDelta:
        v0, w0 = self._at(t0_ns)
        v1, w1 = self._at(t1_ns)
        return _arc_delta((t1_ns - t0_ns) * 1e-9, v0, v1, w0, w1, self.sigma_v)


_OXTS_ANGLE_FIELDS = (3, 4, 5)  # roll, pitch, yaw


def resample_oxts(
    records: Sequence[OxtsRecord], t_ns: Sequence[int], offset_ns: int
) -> list[OxtsRecord]:
    """Records at ``t_ns[i] + offset_ns``, linearly interpolated (clamped at the ends).

    Models an OXTS clock ``offset_ns`` ahead of the camera: frame ``i`` gets the INS
    state measured ``offset_ns`` later. Angles are unwrapped before interpolating and wrapped back to ``(−π, π]``.
    """
    if len(t_ns) != len(records) or not records:
        raise ValueError("need one OXTS record per timestamp")
    t = np.asarray(t_ns, dtype=np.float64)
    vals = np.array([astuple(r) for r in records], dtype=np.float64)
    for j in _OXTS_ANGLE_FIELDS:
        vals[:, j] = np.unwrap(vals[:, j])
    tq = t + float(offset_ns)
    cols = [np.interp(tq, t, vals[:, j]) for j in range(vals.shape[1])]
    for j in _OXTS_ANGLE_FIELDS:
        cols[j] = np.angle(np.exp(1j * cols[j]))
    return [OxtsRecord(*(float(c[i]) for c in cols)) for i in range(len(records))]


def load_oxts_file(path: Path | str) -> list[OxtsRecord]:
    """One record per non-empty line (KITTI tracking ``oxts/<seq>.txt``)."""
    with open(path, encoding="utf-8") as fh:
        return [parse_oxts_line(line) for line in fh if line.strip()]
