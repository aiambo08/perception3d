"""Corridor, proximity and trajectory gates → candidate alert level per track.

Two gates, one per failure mode of a plain ``−Z/Ż``:

* **proximity** (velocity-free): the object is in the ego corridor now
  (``|X| − k·σ_X < w_ego/2 + w_obj/2 + margin``) and its conservative gap is below
  the level's distance threshold — covers a static object ahead with the ego stopped;
* **trajectory**: ``TTC_low`` only counts when the object is *on path* (in the corridor
  now, or closing with ``d_cpa + k_path·σ_d`` inside the corridor half-width). An object
  off path is "passing" and can at most raise ``CAUTION`` (closing, ``t_cpa`` below the
  caution time and ``d_cpa_low`` within ``passing_margin_m`` of the corridor).

``enter`` uses the entry thresholds; ``hold`` the exit ones (time × ``exit_time_factor``,
distance × ``exit_dist_factor``), which is the hysteresis in value that
:mod:`percepcion3d.safety.ttc` pairs with hysteresis in time.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import yaml

from percepcion3d.safety.kinematics import RelativeKinematics, cpa, ttc_low_s


class AlertLevel(IntEnum):
    NONE = 0
    CAUTION = 1
    WARNING = 2
    CRITICAL = 3


LEVELS_DESC: tuple[AlertLevel, ...] = (AlertLevel.CRITICAL, AlertLevel.WARNING, AlertLevel.CAUTION)


@dataclass(frozen=True)
class ClassShape:
    width_m: float
    length_m: float


@dataclass(frozen=True)
class SafetyConfig:
    ego_width_m: float = 1.8
    ego_front_m: float = 1.5
    """Camera → front bumper along Z (tracker positions are camera-origin)."""
    lateral_margin_m: float = 0.5
    passing_margin_m: float = 1.5
    k_sigma: float = 1.5
    """Conservative bounds on gap, closing speed and lateral offset."""
    k_sigma_path: float = 1.0
    """Trajectory gate needs ``d_cpa + k·σ_d < half`` (confidently on path). ``σ_d`` grows
    with ``t_cpa·σ_V``, so a *lower* bound would turn a car passing 1.5 m alongside into
    an on-path target; objects in the corridor now are on path regardless."""
    ttc_enter_s: Mapping[AlertLevel, float] = field(
        default_factory=lambda: {
            AlertLevel.CAUTION: 4.0,
            AlertLevel.WARNING: 2.5,
            AlertLevel.CRITICAL: 1.2,
        }
    )
    dist_enter_m: Mapping[AlertLevel, float] = field(
        default_factory=lambda: {
            AlertLevel.CAUTION: 12.0,
            AlertLevel.WARNING: 6.0,
            AlertLevel.CRITICAL: 3.0,
        }
    )
    enter_frames: Mapping[AlertLevel, int] = field(
        default_factory=lambda: {
            AlertLevel.CAUTION: 3,
            AlertLevel.WARNING: 2,
            AlertLevel.CRITICAL: 1,
        }
    )
    path_frames: int = 3
    """Minimum consecutive frames for a target that is on path only through its CPA (not in
    the corridor now): the CPA extrapolates velocity, so a single-frame outlier must not
    trigger the 1-frame ``CRITICAL`` entry."""
    exit_time_factor: float = 1.3
    exit_dist_factor: float = 1.2
    exit_dwell_s: float = 0.5
    """The hold level must stay below the current one this long before lowering."""
    min_dwell_s: float = 0.5
    """No level is lowered before it has been held this long."""
    lost_decay_s: float = 0.5
    """Lost track: one level down per this interval, never up."""
    forget_s: float = 2.0
    min_updates: int = 3
    """Unconfirmed tracks (fewer updates, coasting, or wide covariance) never raise."""
    max_coast_s: float = 0.3
    max_pos_sigma_m: float = 3.0
    classes: Mapping[str, ClassShape] = field(
        default_factory=lambda: {
            "car": ClassShape(1.8, 4.3),
            "truck": ClassShape(2.5, 8.0),
            "bus": ClassShape(2.5, 11.0),
            "person": ClassShape(0.6, 0.5),
            "bicycle": ClassShape(0.6, 1.7),
            "motorcycle": ClassShape(0.8, 2.1),
        }
    )
    default_shape: ClassShape = ClassShape(1.8, 4.3)

    def shape(self, cls: str) -> ClassShape:
        return self.classes.get(cls, self.default_shape)


def _levels(raw: Mapping[str, Any], cast: type[float] | type[int]) -> dict[AlertLevel, Any]:
    return {AlertLevel[k.upper()]: cast(v) for k, v in raw.items()}


def load_safety_config(path: Path | str) -> SafetyConfig:
    with Path(path).open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    d = SafetyConfig()
    ego = raw.get("ego") or {}
    lv = raw.get("levels") or {}
    hy = raw.get("hysteresis") or {}
    cf = raw.get("confirmation") or {}
    cls_raw: dict[str, dict[str, float]] = raw.get("classes") or {}
    shapes = {k: ClassShape(float(v["width_m"]), float(v["length_m"])) for k, v in cls_raw.items()}
    default = shapes.pop("default", d.default_shape)
    return SafetyConfig(
        ego_width_m=float(ego.get("width_m", d.ego_width_m)),
        ego_front_m=float(ego.get("front_m", d.ego_front_m)),
        lateral_margin_m=float(ego.get("lateral_margin_m", d.lateral_margin_m)),
        passing_margin_m=float(ego.get("passing_margin_m", d.passing_margin_m)),
        k_sigma=float(raw.get("k_sigma", d.k_sigma)),
        k_sigma_path=float(raw.get("k_sigma_path", d.k_sigma_path)),
        ttc_enter_s=_levels(lv["ttc_enter_s"], float) if "ttc_enter_s" in lv else d.ttc_enter_s,
        dist_enter_m=_levels(lv["dist_enter_m"], float) if "dist_enter_m" in lv else d.dist_enter_m,
        enter_frames=_levels(lv["enter_frames"], int) if "enter_frames" in lv else d.enter_frames,
        path_frames=int(lv.get("path_frames", d.path_frames)),
        exit_time_factor=float(hy.get("exit_time_factor", d.exit_time_factor)),
        exit_dist_factor=float(hy.get("exit_dist_factor", d.exit_dist_factor)),
        exit_dwell_s=float(hy.get("exit_dwell_s", d.exit_dwell_s)),
        min_dwell_s=float(hy.get("min_dwell_s", d.min_dwell_s)),
        lost_decay_s=float(hy.get("lost_decay_s", d.lost_decay_s)),
        forget_s=float(hy.get("forget_s", d.forget_s)),
        min_updates=int(cf.get("min_updates", d.min_updates)),
        max_coast_s=float(cf.get("max_coast_s", d.max_coast_s)),
        max_pos_sigma_m=float(cf.get("max_pos_sigma_m", d.max_pos_sigma_m)),
        classes=shapes or d.classes,
        default_shape=default,
    )


@dataclass(frozen=True)
class GateResult:
    ttc_low_s: float
    """``∞`` when off path."""
    t_cpa_s: float
    d_cpa_m: float
    gap_low_m: float
    in_corridor: bool
    on_path: bool
    confirmed: bool
    enter: AlertLevel
    hold: AlertLevel
    reason: str


def is_confirmed(k: RelativeKinematics, cfg: SafetyConfig) -> bool:
    return (
        k.n_updates >= cfg.min_updates
        and k.time_since_update_s <= cfg.max_coast_s
        and k.cov_p[0] + k.cov_p[2] <= cfg.max_pos_sigma_m**2
    )


def evaluate_gates(k: RelativeKinematics, cfg: SafetyConfig) -> GateResult:
    shp = cfg.shape(k.cls)
    half = 0.5 * (cfg.ego_width_m + shp.width_m) + cfg.lateral_margin_m
    ks = cfg.k_sigma
    c = cpa(k)
    x_low = abs(k.p_xz[0]) - ks * math.sqrt(max(k.cov_p[0], 0.0))
    ttc, gap_low = ttc_low_s(k, 0.5 * shp.length_m, ks)
    in_corr = x_low < half and gap_low > -shp.length_m
    d_path = c.d_cpa_m + cfg.k_sigma_path * c.sigma_d_cpa_m
    on_path = in_corr or (c.closing and c.t_cpa_s > 0.0 and d_path < half)
    if not on_path:
        ttc = math.inf

    def level(ft: float, fd: float) -> tuple[AlertLevel, str]:
        if on_path:
            for lv in LEVELS_DESC:
                if ttc < cfg.ttc_enter_s[lv] * ft:
                    return lv, "ttc"
                if in_corr and gap_low < cfg.dist_enter_m[lv] * fd:
                    return lv, "proximity"
            return AlertLevel.NONE, "clear"
        if (
            c.closing
            and c.t_cpa_s < cfg.ttc_enter_s[AlertLevel.CAUTION] * ft
            and c.d_cpa_m < half + cfg.passing_margin_m * fd
        ):
            return AlertLevel.CAUTION, "passing"
        return AlertLevel.NONE, "clear"

    enter, reason = level(1.0, 1.0)
    hold, hold_reason = level(cfg.exit_time_factor, cfg.exit_dist_factor)
    return GateResult(
        ttc_low_s=ttc,
        t_cpa_s=c.t_cpa_s,
        d_cpa_m=c.d_cpa_m,
        gap_low_m=gap_low,
        in_corridor=in_corr,
        on_path=on_path,
        confirmed=is_confirmed(k, cfg),
        enter=enter,
        hold=hold,
        reason=reason if enter > AlertLevel.NONE else hold_reason,
    )
