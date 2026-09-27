"""F6: gates, alert state machine (hysteresis, dwell, lost decay), synthetic battery (CPU)."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from percepcion3d.eval.safety_synthetic import (
    SafetyObject,
    SafetyScenario,
    default_safety_scenarios,
    is_deterministic,
    run_safety_scenario,
    safety_dod,
)
from percepcion3d.safety.gates import (
    AlertLevel,
    SafetyConfig,
    evaluate_gates,
    load_safety_config,
)
from percepcion3d.safety.kinematics import RelativeKinematics
from percepcion3d.safety.ttc import SafetyState, initial_state, max_level, step

ROOT = Path(__file__).resolve().parents[1]
CFG = load_safety_config(ROOT / "configs" / "safety.yaml")
FPS = 30.0


def _kin(
    p: tuple[float, float],
    v: tuple[float, float],
    t_ns: int = 0,
    sp: float = 0.1,
    sv: float = 0.1,
    cls: str = "car",
    n_updates: int = 100,
    coast_s: float = 0.0,
    tid: int = 1,
) -> RelativeKinematics:
    return RelativeKinematics(
        tid,
        cls,
        t_ns,
        p,
        v,
        (sp * sp, 0.0, sp * sp),
        (sv * sv, 0.0, sv * sv),
        n_updates=n_updates,
        time_since_update_s=coast_s,
    )


def _run(
    kins_per_frame: list[list[RelativeKinematics]], cfg: SafetyConfig = CFG
) -> tuple[list[AlertLevel], SafetyState]:
    state = initial_state()
    levels = []
    for i, kins in enumerate(kins_per_frame):
        t_ns = int(round(i / FPS * 1e9))
        state, alerts = step(state, [replace(k, t_ns=t_ns) for k in kins], cfg, t_ns)
        levels.append(max_level(alerts))
    return levels, state


def test_config_loads_and_matches_defaults_shape() -> None:
    assert CFG.ttc_enter_s[AlertLevel.CAUTION] == 4.0
    assert CFG.enter_frames[AlertLevel.CRITICAL] == 1
    assert CFG.shape("bus").length_m == 11.0
    assert CFG.shape("unknown") == CFG.default_shape
    assert SafetyConfig().ttc_enter_s == CFG.ttc_enter_s


def test_gates_static_object_in_corridor_uses_proximity_not_velocity() -> None:
    # Ego stopped, car 1.35 m of gap ahead: no closing speed → TTC = ∞ but proximity fires.
    g = evaluate_gates(_kin((0.0, 1.35 + 2.15), (0.0, 0.0), sv=0.0), CFG)
    assert g.in_corridor and g.on_path
    assert g.ttc_low_s == math.inf
    assert g.enter is AlertLevel.CRITICAL and g.reason == "proximity"
    # With velocity uncertainty the bound is finite but proximity still decides.
    g2 = evaluate_gates(_kin((0.0, 1.35 + 2.15), (0.0, 0.0), sv=0.1), CFG)
    assert g2.ttc_low_s == pytest.approx(1.2 / 0.15) and g2.reason == "proximity"


def test_gates_lateral_passer_is_at_most_caution() -> None:
    half = 0.5 * (CFG.ego_width_m + CFG.shape("car").width_m) + CFG.lateral_margin_m
    g = evaluate_gates(_kin((-(half + 1.0), 20.0), (0.0, -8.0)), CFG)
    assert not g.in_corridor and not g.on_path
    assert g.ttc_low_s == math.inf
    assert g.enter is AlertLevel.CAUTION and g.reason == "passing"
    far = evaluate_gates(_kin((-(half + 5.0), 20.0), (0.0, -8.0)), CFG)
    assert far.enter is AlertLevel.NONE


def test_gates_hold_uses_wider_thresholds() -> None:
    # TTC_low ≈ 3.0 s: below the 4 s CAUTION entry; WARNING entry is 2.5 s but its exit is
    # 2.5 × 1.3 = 3.25 s, so `hold` already says WARNING while `enter` says CAUTION.
    k = _kin((0.0, 30.0 + 2.15), (0.0, -10.0), sp=0.0, sv=0.0)
    g = evaluate_gates(k, CFG)
    assert g.ttc_low_s == pytest.approx(3.0)
    assert g.enter is AlertLevel.CAUTION and g.hold is AlertLevel.WARNING


def test_gates_unconfirmed_track_flagged() -> None:
    assert not evaluate_gates(_kin((0.0, 5.0), (0.0, -5.0), n_updates=1), CFG).confirmed
    assert not evaluate_gates(_kin((0.0, 5.0), (0.0, -5.0), coast_s=1.0), CFG).confirmed
    assert not evaluate_gates(_kin((0.0, 5.0), (0.0, -5.0), sp=5.0), CFG).confirmed
    assert evaluate_gates(_kin((0.0, 5.0), (0.0, -5.0)), CFG).confirmed


def test_raise_needs_consecutive_frames_and_critical_is_immediate_in_corridor() -> None:
    caution = _kin((0.0, 40.0), (0.0, -11.0), sp=0.0, sv=0.0)  # TTC ≈ 3.4 s
    levels, _ = _run([[caution]] * 5)
    n = CFG.enter_frames[AlertLevel.CAUTION]
    assert levels[: n - 1] == [AlertLevel.NONE] * (n - 1)
    assert levels[n - 1 :] == [AlertLevel.CAUTION] * (5 - n + 1)
    critical = _kin((0.0, 5.0), (0.0, -10.0), sp=0.0, sv=0.0)
    levels, _ = _run([[critical]])
    assert levels == [AlertLevel.CRITICAL]


def test_interrupted_candidate_restarts_the_frame_count() -> None:
    caution = _kin((0.0, 40.0), (0.0, -11.0), sp=0.0, sv=0.0)
    clear = _kin((0.0, 80.0), (0.0, -1.0), sp=0.0, sv=0.0)
    levels, _ = _run([[caution], [caution], [clear], [caution], [caution], [caution]])
    assert levels[:5] == [AlertLevel.NONE] * 5 and levels[5] is AlertLevel.CAUTION


def test_cpa_only_target_needs_path_frames_before_critical() -> None:
    # Outside the corridor now, CPA inside it, TTC_low < 1.2 s → CRITICAL via the trajectory
    # gate only after `path_frames` consecutive frames.
    k = _kin((-3.5, 12.0), (3.0, -10.0), sp=0.0, sv=0.0)
    g = evaluate_gates(k, CFG)
    assert g.on_path and not g.in_corridor and g.enter is AlertLevel.CRITICAL
    levels, _ = _run([[k]] * (CFG.path_frames + 1))
    assert levels[: CFG.path_frames - 1] == [AlertLevel.NONE] * (CFG.path_frames - 1)
    assert levels[CFG.path_frames - 1] is AlertLevel.CRITICAL


def test_unconfirmed_track_never_raises() -> None:
    rng = np.random.default_rng(1)
    frames = []
    for _ in range(60):
        frames.append(
            [
                _kin(
                    (float(rng.uniform(-3, 3)), float(rng.uniform(0.5, 30))),
                    (float(rng.uniform(-2, 2)), float(rng.uniform(-20, 0))),
                    n_updates=int(rng.integers(0, CFG.min_updates)),
                )
            ]
        )
    levels, _ = _run(frames)
    assert set(levels) == {AlertLevel.NONE}


def test_level_never_drops_before_min_dwell_and_needs_exit_dwell() -> None:
    critical = _kin((0.0, 5.0), (0.0, -10.0), sp=0.0, sv=0.0)
    clear = _kin((0.0, 90.0), (0.0, -1.0), sp=0.0, sv=0.0)
    n_dwell = int(math.ceil(CFG.min_dwell_s * FPS))
    n_exit = int(math.ceil(CFG.exit_dwell_s * FPS))
    levels, _ = _run([[critical]] + [[clear]] * (n_dwell + n_exit + 5))
    # frame 0 sets CRITICAL; the exit condition starts at frame 1.
    first_drop = next(i for i, lv in enumerate(levels) if lv is not AlertLevel.CRITICAL)
    assert first_drop >= max(n_dwell, n_exit)
    assert first_drop <= max(n_dwell, n_exit) + 2
    assert levels[first_drop] is AlertLevel.NONE


def test_lowering_goes_to_hold_level_with_hysteresis_band() -> None:
    critical = _kin((0.0, 5.0), (0.0, -10.0), sp=0.0, sv=0.0)
    # TTC_low = 3.0 s: `enter` CAUTION, `hold` WARNING (exit band of WARNING is 3.25 s).
    band = _kin((0.0, 32.15), (0.0, -10.0), sp=0.0, sv=0.0)
    levels, _ = _run([[critical]] + [[band]] * 60)
    assert AlertLevel.WARNING in levels
    assert AlertLevel.CAUTION not in levels
    assert levels[-1] is AlertLevel.WARNING


def test_lost_track_decays_one_level_per_interval_and_is_forgotten() -> None:
    critical = _kin((0.0, 5.0), (0.0, -10.0), sp=0.0, sv=0.0)
    state, alerts = step(initial_state(), [critical], CFG, 0)
    assert max_level(alerts) is AlertLevel.CRITICAL
    ns = int(CFG.lost_decay_s * 1e9)
    seq = []
    for k in (0.5, 1.0, 2.0, 3.0):
        state, alerts = step(state, [], CFG, int(k * ns))
        seq.append((max_level(alerts), [a.reason for a in alerts]))
    assert seq[0] == (AlertLevel.CRITICAL, ["lost"])
    assert seq[1] == (AlertLevel.WARNING, ["lost"])
    assert seq[2] == (AlertLevel.CAUTION, ["lost"])
    assert seq[3] == (AlertLevel.NONE, [])
    state, _ = step(state, [], CFG, int((CFG.forget_s + 0.1) * 1e9))
    assert 1 not in state.tracks


def test_lost_track_never_raises_and_reappearance_keeps_level() -> None:
    warning = _kin((0.0, 20.0 + 2.15), (0.0, -10.0), sp=0.0, sv=0.0)  # TTC 2.0 s
    state = initial_state()
    for i in range(3):
        state, alerts = step(state, [replace(warning, t_ns=i * 33_000_000)], CFG, i * 33_000_000)
    assert max_level(alerts) is AlertLevel.WARNING
    state, alerts = step(state, [], CFG, 200_000_000)
    assert max_level(alerts) is AlertLevel.WARNING and alerts[0].reason == "lost"
    state, alerts = step(state, [replace(warning, t_ns=233_000_000)], CFG, 233_000_000)
    assert max_level(alerts) is AlertLevel.WARNING and alerts[0].reason == "hold"


def test_step_is_pure_and_alerts_sorted_by_track_id() -> None:
    k2 = _kin((0.0, 5.0), (0.0, -10.0), sp=0.0, sv=0.0, tid=2)
    k9 = _kin((0.0, 6.0), (0.0, -10.0), sp=0.0, sv=0.0, tid=9)
    s0 = initial_state()
    s1, alerts = step(s0, [k9, k2], CFG, 0)
    assert [a.track_id for a in alerts] == [2, 9]
    assert s0.tracks == {} and set(s1.tracks) == {2, 9}
    s1b, alerts_b = step(s0, [k9, k2], CFG, 0)
    assert s1b == s1 and alerts_b == alerts


def test_synthetic_battery_passes_dod_and_is_deterministic() -> None:
    scenarios = default_safety_scenarios(CFG)
    names = {sc.name for sc in scenarios}
    assert {
        "head_on",
        "static_near_ego_stopped",
        "lateral_overtake_1p5m",
        "cut_in_intersects",
        "crossing_no_intersect",
        "intermittent_track",
    } <= names
    for seed in range(3):
        res = {sc.name: run_safety_scenario(sc, CFG, seed) for sc in scenarios}
        dod = safety_dod(res, scenarios)
        assert all(dod.values()), dod
        lead = res["head_on"].critical_lead_s
        assert lead is not None and lead >= 1.0
        assert res["lateral_overtake_1p5m"].peak <= AlertLevel.CAUTION
        assert res["static_near_ego_stopped"].peak >= AlertLevel.WARNING
        assert res["intermittent_track"].transitions_per_s <= 1.0
    assert all(is_deterministic(sc, CFG) for sc in scenarios)


def test_synthetic_scenario_noiseless_head_on_lead_matches_threshold() -> None:
    cfg = CFG
    sc = SafetyScenario(
        "hd",
        (SafetyObject(1, "car", 0.0, 60.0, 0.0, -15.0),),
        duration_s=3.8,
        contact_s=(60.0 - 2.15) / 15.0,
    )
    r = run_safety_scenario(sc, cfg, seed=None)
    lead = r.critical_lead_s
    assert lead is not None
    # Noiseless values but reported σ (σ_Z = 0.05·Z, σ_V = 0.5): CRITICAL when
    # (Z − L/2 − kσ_Z)/(15 + kσ_V) < 1.2 s, i.e. earlier than the nominal 1.2 s.
    k, v, half_len = cfg.k_sigma, 15.0, 2.15
    z_crit = (1.2 * (v + k * sc.sigma_v_mps) + half_len) / (1.0 - k * sc.rel_sigma_z)
    expected = sc.contact_s - (60.0 - z_crit) / v if sc.contact_s is not None else None
    assert expected is not None and expected > cfg.ttc_enter_s[AlertLevel.CRITICAL]
    assert lead == pytest.approx(expected, abs=1.5 / FPS)
