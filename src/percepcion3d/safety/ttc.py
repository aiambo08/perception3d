"""Alert state machine per track: pure ``step(state, kinematics, cfg, t_ns) → (state, alerts)``.

No wall clock, no randomness: the output is a deterministic function of the input
sequence (timestamps included), so a scenario replays bit-for-bit.

Per track and frame, with :func:`evaluate_gates` giving ``enter`` (entry thresholds)
and ``hold`` (exit thresholds = entry × factor):

* **raise** to ``enter`` once it has been the candidate for ``enter_frames[enter]``
  consecutive frames (``CRITICAL``: 1 frame in the corridor, ``path_frames`` when on path
  only through the CPA). An unconfirmed track (few updates, coasting, wide covariance)
  never raises;
* **lower** to ``hold`` only after the current level has been held ``min_dwell_s`` and
  ``hold < level`` has persisted ``exit_dwell_s`` — hysteresis in value *and* time;
* **lost** track (absent from the input): one level down per ``lost_decay_s``, never up,
  forgotten after ``forget_s``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from percepcion3d.safety.gates import AlertLevel, GateResult, SafetyConfig, evaluate_gates
from percepcion3d.safety.kinematics import RelativeKinematics


@dataclass(frozen=True)
class TrackAlertState:
    level: AlertLevel = AlertLevel.NONE
    t_level_ns: int = 0
    """When ``level`` was last set."""
    candidate: AlertLevel = AlertLevel.NONE
    candidate_frames: int = 0
    t_below_ns: int | None = None
    """Since when ``hold < level`` (``None`` while the level is supported)."""
    t_seen_ns: int = 0
    lost_level: AlertLevel = AlertLevel.NONE
    """Level at the moment the track was last seen (decay anchor)."""


@dataclass(frozen=True)
class SafetyState:
    tracks: Mapping[int, TrackAlertState]
    t_ns: int = 0


@dataclass(frozen=True)
class Alert:
    track_id: int
    level: AlertLevel
    ttc_low_s: float
    d_cpa_m: float
    reason: str
    """``ttc`` / ``proximity`` / ``passing`` / ``lost`` / ``hold``."""


def initial_state() -> SafetyState:
    return SafetyState(tracks={})


def _step_track(
    st: TrackAlertState, g: GateResult, cfg: SafetyConfig, t_ns: int
) -> tuple[TrackAlertState, str]:
    level = st.level
    enter = g.enter if g.confirmed else min(g.enter, level)
    candidate, frames = st.candidate, st.candidate_frames
    t_level, t_below = st.t_level_ns, st.t_below_ns
    reason = "hold"
    if enter > level:
        frames = frames + 1 if candidate == enter else 1
        candidate = enter
        needed = (
            cfg.enter_frames[enter]
            if g.in_corridor
            else max(cfg.enter_frames[enter], cfg.path_frames)
        )
        if frames >= needed:
            level, t_level, t_below = enter, t_ns, None
            candidate, frames = AlertLevel.NONE, 0
            reason = g.reason
    else:
        candidate, frames = AlertLevel.NONE, 0
    if g.hold < level:
        if t_below is None:
            t_below = t_ns
        held = (t_ns - t_level) * 1e-9 >= cfg.min_dwell_s
        below = (t_ns - t_below) * 1e-9 >= cfg.exit_dwell_s
        if held and below:
            level, t_level, t_below = g.hold, t_ns, None
            reason = g.reason
    else:
        t_below = None
    return (
        TrackAlertState(
            level=level,
            t_level_ns=t_level,
            candidate=candidate,
            candidate_frames=frames,
            t_below_ns=t_below,
            t_seen_ns=t_ns,
            lost_level=level,
        ),
        reason,
    )


def _decay(st: TrackAlertState, cfg: SafetyConfig, t_ns: int) -> TrackAlertState | None:
    lost_s = (t_ns - st.t_seen_ns) * 1e-9
    if lost_s > cfg.forget_s:
        return None
    steps = int(lost_s // cfg.lost_decay_s) if cfg.lost_decay_s > 0 else 0
    level = AlertLevel(max(int(st.lost_level) - steps, 0))
    if level == st.level:
        return st
    return replace(st, level=level, t_level_ns=t_ns, candidate=AlertLevel.NONE, candidate_frames=0)


def step(
    state: SafetyState,
    kinematics: Sequence[RelativeKinematics],
    cfg: SafetyConfig,
    t_ns: int,
) -> tuple[SafetyState, list[Alert]]:
    """Advance every track to ``t_ns``; alerts (level > NONE) are sorted by track id."""
    tracks = dict(state.tracks)
    alerts: list[Alert] = []
    seen: set[int] = set()
    for k in kinematics:
        seen.add(k.track_id)
        g = evaluate_gates(k, cfg)
        st, reason = _step_track(tracks.get(k.track_id, TrackAlertState()), g, cfg, t_ns)
        tracks[k.track_id] = st
        if st.level > AlertLevel.NONE:
            alerts.append(Alert(k.track_id, st.level, g.ttc_low_s, g.d_cpa_m, reason))
    for tid in [t for t in tracks if t not in seen]:
        decayed = _decay(tracks[tid], cfg, t_ns)
        if decayed is None:
            del tracks[tid]
            continue
        tracks[tid] = decayed
        if decayed.level > AlertLevel.NONE:
            alerts.append(Alert(tid, decayed.level, float("inf"), float("nan"), "lost"))
    alerts.sort(key=lambda a: a.track_id)
    return SafetyState(tracks=tracks, t_ns=t_ns), alerts


def max_level(alerts: Sequence[Alert]) -> AlertLevel:
    return max((a.level for a in alerts), default=AlertLevel.NONE)
