"""Synthetic battery for the F6 alert machine (DoD of ``docs/01_plan_fases_mvp.md``).

Kinematic truth (ego front frame, constant relative velocity per object) plus Gaussian
measurement noise with the covariance the tracker would report
(``σ_Z = rel_sigma_z·Z``, ``σ_X = rel_sigma_z·|X| + Z·σ_col/f`` from ``X = Z·(u−c_x)/f``,
``σ_V`` fixed) is fed straight to
:func:`percepcion3d.safety.ttc.step`; the tracker itself is covered by F5. Scored per
scenario: the level time series, its transitions per second, the lead time of the
first ``CRITICAL`` before contact and the maximum level reached.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from percepcion3d.safety.gates import AlertLevel, SafetyConfig
from percepcion3d.safety.kinematics import RelativeKinematics
from percepcion3d.safety.ttc import SafetyState, initial_state, max_level, step


@dataclass(frozen=True)
class SafetyObject:
    obj_id: int
    cls: str
    x0_m: float
    z0_m: float
    """Initial centre relative to the ego front (m)."""
    vx_mps: float = 0.0
    vz_mps: float = 0.0
    """Relative velocity ``V_obj − v_ego`` (m/s)."""
    visible: tuple[tuple[float, float], ...] | None = None
    """Visibility windows ``[t0, t1)``; ``None`` → always visible."""

    def at(self, t: float) -> tuple[float, float]:
        return self.x0_m + self.vx_mps * t, self.z0_m + self.vz_mps * t

    def is_visible(self, t: float) -> bool:
        return self.visible is None or any(a <= t < b for a, b in self.visible)


@dataclass(frozen=True)
class SafetyScenario:
    name: str
    objects: tuple[SafetyObject, ...]
    duration_s: float = 6.0
    fps: float = 30.0
    rel_sigma_z: float = 0.05
    sigma_col_px: float = 1.5
    focal_px: float = 721.5
    sigma_v_mps: float = 0.5
    contact_s: float | None = None
    """Truth time of contact (front bumper reaches the object's near face); ``None`` if none."""
    max_level: AlertLevel = AlertLevel.CRITICAL
    """DoD: highest level allowed."""
    min_level: AlertLevel = AlertLevel.NONE
    """DoD: level that must be reached at some point."""
    critical_lead_s: float | None = None
    """DoD: ``CRITICAL`` at least this long before ``contact_s``."""
    max_transitions_per_s: float | None = None


@dataclass
class SafetyScenarioResult:
    name: str
    levels: list[int] = field(default_factory=list)
    t_s: list[float] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    first_critical_s: float | None = None
    contact_s: float | None = None
    step_us: list[float] = field(default_factory=list)

    @property
    def peak(self) -> AlertLevel:
        return AlertLevel(max(self.levels, default=0))

    @property
    def transitions(self) -> int:
        return (
            int(np.count_nonzero(np.diff(np.asarray(self.levels, dtype=int)))) if self.levels else 0
        )

    @property
    def transitions_per_s(self) -> float:
        span = (self.t_s[-1] - self.t_s[0]) if len(self.t_s) > 1 else 0.0
        return self.transitions / span if span > 0 else 0.0

    @property
    def critical_lead_s(self) -> float | None:
        if self.first_critical_s is None or self.contact_s is None:
            return None
        return self.contact_s - self.first_critical_s

    def digest(self) -> str:
        """Hash of the level sequence (determinism check)."""
        return hashlib.sha256(bytes(self.levels)).hexdigest()

    def summary(self) -> dict[str, Any]:
        return {
            "peak": self.peak.name,
            "first_critical_s": self.first_critical_s,
            "critical_lead_s": self.critical_lead_s,
            "transitions": self.transitions,
            "transitions_per_s": self.transitions_per_s,
            "n_frames": len(self.levels),
            "digest": self.digest(),
        }


def _measure(
    obj: SafetyObject, t: float, t_ns: int, sc: SafetyScenario, rng: np.random.Generator | None
) -> RelativeKinematics:
    x, z = obj.at(t)
    sz = max(sc.rel_sigma_z * abs(z), 0.05)
    sx = max(sc.rel_sigma_z * abs(x) + abs(z) * sc.sigma_col_px / sc.focal_px, 0.05)
    sv = sc.sigma_v_mps
    vx, vz = obj.vx_mps, obj.vz_mps
    if rng is not None:
        x += sx * rng.standard_normal()
        z += sz * rng.standard_normal()
        vx += sv * rng.standard_normal()
        vz += sv * rng.standard_normal()
    return RelativeKinematics(
        track_id=obj.obj_id,
        cls=obj.cls,
        t_ns=t_ns,
        p_xz=(x, z),
        v_xz=(vx, vz),
        cov_p=(sx * sx, 0.0, sz * sz),
        cov_v=(sv * sv, 0.0, sv * sv),
    )


def run_safety_scenario(
    sc: SafetyScenario, cfg: SafetyConfig, seed: int | None = 0
) -> SafetyScenarioResult:
    """``seed=None`` → noiseless truth."""
    rng = np.random.default_rng(seed) if seed is not None else None
    res = SafetyScenarioResult(sc.name, contact_s=sc.contact_s)
    state: SafetyState = initial_state()
    n = int(round(sc.duration_s * sc.fps))
    for i in range(n):
        t = i / sc.fps
        t_ns = int(round(t * 1e9))
        kins = [_measure(o, t, t_ns, sc, rng) for o in sc.objects if o.is_visible(t)]
        t0 = time.perf_counter()
        state, alerts = step(state, kins, cfg, t_ns)
        res.step_us.append((time.perf_counter() - t0) * 1e6)
        lv = max_level(alerts)
        res.levels.append(int(lv))
        res.t_s.append(t)
        res.reasons.append(alerts[0].reason if alerts else "")
        if lv is AlertLevel.CRITICAL and res.first_critical_s is None:
            res.first_critical_s = t
    return res


def default_safety_scenarios(cfg: SafetyConfig) -> list[SafetyScenario]:
    car_half_len = 0.5 * cfg.shape("car").length_m
    v_ego = 15.0
    z0 = 60.0
    contact = (z0 - car_half_len) / v_ego
    lateral = 0.5 * (cfg.ego_width_m + cfg.shape("car").width_m) + 1.5
    return [
        SafetyScenario(
            "head_on",
            (SafetyObject(1, "car", 0.0, z0, 0.0, -v_ego),),
            duration_s=contact,
            contact_s=contact,
            min_level=AlertLevel.CRITICAL,
            critical_lead_s=1.0,
        ),
        SafetyScenario(
            "static_near_ego_stopped",
            (SafetyObject(1, "car", 0.0, 2.0 + car_half_len),),
            duration_s=2.0,
            min_level=AlertLevel.WARNING,
        ),
        SafetyScenario(
            "lateral_overtake_1p5m",
            (SafetyObject(1, "car", -lateral, 40.0, 0.0, -8.0),),
            duration_s=6.0,
            max_level=AlertLevel.CAUTION,
        ),
        SafetyScenario(
            "cut_in_intersects",
            (SafetyObject(1, "car", -4.5, 30.0, 1.6, -10.0),),
            duration_s=2.6,
            min_level=AlertLevel.WARNING,
        ),
        SafetyScenario(
            "crossing_no_intersect",
            (SafetyObject(1, "person", -12.0, 25.0, 1.5, -10.0),),
            duration_s=3.0,
            max_level=AlertLevel.CAUTION,
        ),
        SafetyScenario(
            "intermittent_track",
            (
                SafetyObject(
                    1,
                    "car",
                    0.0,
                    45.0,
                    0.0,
                    -8.0,
                    visible=tuple((s, s + 0.7) for s in np.arange(0.0, 5.0, 1.0)),
                ),
            ),
            duration_s=5.0,
            max_transitions_per_s=1.0,
        ),
    ]


def safety_dod(
    results: dict[str, SafetyScenarioResult], scenarios: list[SafetyScenario]
) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for sc in scenarios:
        r = results[sc.name]
        ok = r.peak <= sc.max_level and r.peak >= sc.min_level
        if sc.critical_lead_s is not None:
            lead = r.critical_lead_s
            ok = ok and lead is not None and lead >= sc.critical_lead_s
        if sc.max_transitions_per_s is not None:
            ok = ok and r.transitions_per_s <= sc.max_transitions_per_s
        out[sc.name] = ok
    return out


def is_deterministic(sc: SafetyScenario, cfg: SafetyConfig, seed: int = 0) -> bool:
    return (
        run_safety_scenario(sc, cfg, seed).digest() == run_safety_scenario(sc, cfg, seed).digest()
    )


def cpu_p95_us(results: dict[str, SafetyScenarioResult]) -> float:
    xs = [u for r in results.values() for u in r.step_us]
    return float(np.percentile(xs, 95)) if xs else math.nan
