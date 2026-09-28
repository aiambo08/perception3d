"""F5 on KITTI tracking: ID switches, relative-velocity RMSE (0–30 m), static labelling.

DoD (docs/01_plan_fases_mvp.md): 3 sequences; RMSE of the relative velocity ≤ 1.0 m/s at
0–30 m; with ``--ego oxts`` ≥ 90 % of GT-static objects labelled static after 0.5 s.

Ego-motion: ``--ego oxts`` reads ``<root>/oxts/<seq>.txt`` (``data_tracking_oxts.zip``);
``--ego zero`` tracks relative motion only (velocity compared with the apparent GT
velocity). Depth/boxes as in ``scripts/eval_fusion_kitti.py``.

Examples::

    uv run python scripts/eval_tracking_kitti.py --root $KT --seqs 0000 0001 0020 --ego oxts
    uv run python scripts/eval_tracking_kitti.py --root $KT --seqs 0000 0001 0020 --ego oxts \
        --engine models/depth_924x280_fp16.engine --boxes detector --json reports/f5_kitti.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from percepcion3d.camera.calibration import load_camera_config_yaml  # noqa: E402
from percepcion3d.camera.geometry import PinholeGeometry  # noqa: E402
from percepcion3d.depth.fusion import (  # noqa: E402
    MetricFusionStage,
    load_fusion_config,
    load_solver_configs,
)
from percepcion3d.eval.fusion_kitti import DepthProvider, KittiTrackingSequence  # noqa: E402
from percepcion3d.eval.providers import (  # noqa: E402
    detector_box_provider,
    engine_depth_provider,
    npy_depth_provider,
)
from percepcion3d.eval.tracking_kitti import (  # noqa: E402
    TrackingKittiResult,
    gt_kinematics,
    merge_results,
    run_tracking_kitti,
)
from percepcion3d.tracking.ego_motion import (  # noqa: E402
    EgoMotionProvider,
    OxtsEgoMotion,
    ZeroEgoMotion,
    load_oxts_file,
    resample_oxts,
)
from percepcion3d.tracking.kalman_filter import RangeBias  # noqa: E402
from percepcion3d.tracking.tracker3d import Tracker3D, load_tracker_config  # noqa: E402
from percepcion3d.utils.profiling import StageTimer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _p95(p50_p95_p99: list[float]) -> float:
    return p50_p95_p99[1] if p50_p95_p99 else float("nan")


def _print_consistency(c: dict[str, Any] | None) -> None:
    if not c:
        return
    rows = {"all": c["all"], **{f"age {k} s": v for k, v in c["by_track_age_s"].items()}}
    for name, st in rows.items():
        if st["n"]:
            print(
                f"    consistency {name:>11}: n={st['n']:5d} RMSE V_Z {st['rmse_vz']:.2f} · "
                f"σ pred {st['sigma_vz_pred_rms']:.2f} (ratio {st['ratio']:.2f}) · NIS mean "
                f"{st['mean_nis']:.2f} (>5.99: {100 * st['frac_nis_gt_5.99']:.0f} %)"
            )


def _print_range_error(r: dict[str, Any] | None) -> None:
    if not r or r.get("n", 0) < 3:
        return
    ac = " ".join(f"{k}f={v['rho']:.2f}" for k, v in r.get("autocorr_by_lag_frames", {}).items())
    print(
        f"    range error: n={r['n']} mean {100 * r['mean_rel']:+.1f} % rms "
        f"{100 * r['rms_rel']:.1f} % · frame share {r.get('frame_share', float('nan')):.2f} · "
        f"autocorr {ac}"
    )
    be = r.get("bias_estimate")
    if be:
        print(
            f"    bias estimate: corr with error {be['corr_with_error']:.2f} · "
            f"rms(error − b) {100 * be['rms_error_minus_bias']:.1f} %"
        )


def _print_turn(turn: dict[str, Any] | None) -> None:
    if not turn:
        return
    for name in ("vz_err_vs_omega_x", "vx_err_vs_omega_z"):
        fit = turn[name]
        if "slope" in fit:
            print(
                f"    {name}: slope {fit['slope']:+.3f} intercept {fit['intercept']:+.2f} "
                f"resid {fit['resid_rms']:.2f} (n={fit['n']})"
            )
    for name, st in turn["by_turn_direction"].items():
        if st["n"]:
            print(
                f"    turn {name:>8}: n={st['n']:4d} RMSE {st['rmse']:.2f} "
                f"bias ({st['bias_xz'][0]:+.2f}, {st['bias_xz'][1]:+.2f}) · ez "
                f"{st['mean_ez_m']:+.2f} m ({100 * st['mean_ez_rel']:+.1f} %) · d(ez)/dt "
                f"{st['mean_dez_dt_mps']:+.2f} → vz bias − drift "
                f"{st['vz_err_minus_range_drift']:+.2f}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, required=True, help="KITTI tracking training/ folder")
    ap.add_argument("--seqs", nargs="+", default=["0000", "0001", "0020"])
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--ego", choices=("oxts", "zero"), default="oxts")
    ap.add_argument(
        "--lever-arm", type=float, default=1.08, help="Camera ahead of the IMU along X_fwd (m)"
    )
    ap.add_argument("--camera", type=Path, default=ROOT / "configs" / "camera_kitti.yaml")
    ap.add_argument("--fusion", type=Path, default=ROOT / "configs" / "fusion.yaml")
    ap.add_argument("--tracking", type=Path, default=ROOT / "configs" / "tracking_kitti.yaml")
    ap.add_argument("--models", type=Path, default=ROOT / "configs" / "models.yaml")
    ap.add_argument("--engine", type=Path, default=None, help="TensorRT depth engine")
    ap.add_argument("--depth-dir", type=Path, default=None, help="Pre-computed <frame>.npy maps")
    ap.add_argument("--boxes", choices=("gt", "detector"), default="gt")
    ap.add_argument("--det-engine", type=Path, default=None)
    ap.add_argument(
        "--q-vehicle",
        type=float,
        default=None,
        help="Override dynamics.vehicle.q of --tracking (CWNA m²/s³)",
    )
    ap.add_argument(
        "--arbitration",
        choices=("select", "inflate"),
        default=None,
        help="Override gates.arbitration of --fusion",
    )
    ap.add_argument(
        "--r-scale",
        type=float,
        default=None,
        help="Override filter.r_scale of --tracking (R ← r_scale·R of F4)",
    )
    ap.add_argument(
        "--robust-chi2",
        type=float,
        default=None,
        help="Override filter.robust_chi2 of --tracking (0 disables the robust update)",
    )
    ap.add_argument(
        "--gt-alt-window",
        type=int,
        default=5,
        help="Half window (frames) of the second GT velocity used to estimate its spread",
    )
    ap.add_argument(
        "--pitch-q",
        type=float,
        default=None,
        help="Override pitch_filter.q_deg_per_sqrt_s of --fusion (online pitch agility)",
    )
    ap.add_argument(
        "--sigma-pitch-deg",
        type=float,
        default=None,
        help="Override noise.sigma_pitch_deg of --fusion (BLUE weight of the ground cue)",
    )
    ap.add_argument(
        "--pitch-filter",
        choices=("robust", "legacy"),
        default="robust",
        help="legacy: formal median variance + hard χ² gate, no reset (pre-fix A/B)",
    )
    ap.add_argument(
        "--nominal-pitch-deg",
        type=float,
        default=None,
        help="Override extrinsics.pitch_deg of --camera (2.5 reproduces the old KITTI nominal)",
    )
    ap.add_argument(
        "--oxts-offset-ms",
        type=float,
        default=0.0,
        help="OXTS clock ahead of the camera (ms): frame i uses the INS state at t_i + offset "
        "(ego-motion and GT rotation term alike)",
    )
    ap.add_argument(
        "--range-bias",
        type=float,
        nargs=2,
        metavar=("SIGMA", "TAU_S"),
        default=None,
        help="Per-track Gauss-Markov range-scale bias state (σ as a fraction of range, "
        "correlation time in s); 0 0 disables it",
    )
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    if args.engine is not None and args.depth_dir is not None:
        ap.error("--engine and --depth-dir are mutually exclusive")

    _, extr = load_camera_config_yaml(args.camera)
    if args.nominal_pitch_deg is not None:
        extr = replace(extr, pitch_rad=float(np.deg2rad(args.nominal_pitch_deg)))
    fusion_cfg = load_fusion_config(args.fusion)
    if args.arbitration is not None:
        fusion_cfg = replace(fusion_cfg, arbitration=args.arbitration)
    if args.sigma_pitch_deg is not None:
        fusion_cfg = replace(fusion_cfg, sigma_pitch_rad=float(np.deg2rad(args.sigma_pitch_deg)))
    road_cfg, aff_cfg, pit_cfg = load_solver_configs(args.fusion)
    if args.pitch_q is not None:
        pit_cfg = replace(pit_cfg, q_rad_per_sqrt_s=float(np.deg2rad(args.pitch_q)))
    if args.pitch_filter == "legacy":
        pit_cfg = pit_cfg.legacy()
    trk_cfg = load_tracker_config(args.tracking)
    if args.robust_chi2 is not None:
        trk_cfg = replace(trk_cfg, robust_chi2=args.robust_chi2 if args.robust_chi2 > 0 else None)
    if args.r_scale is not None:
        trk_cfg = replace(trk_cfg, r_scale=args.r_scale)
    if args.range_bias is not None:
        sb, tau = args.range_bias
        trk_cfg = replace(trk_cfg, range_bias=RangeBias(sb, tau) if sb > 0.0 else None)
    if args.q_vehicle is not None:
        veh = replace(trk_cfg.dynamics_for("car"), q=args.q_vehicle)
        trk_cfg = replace(trk_cfg, dynamics={**trk_cfg.dynamics, "vehicle": veh})
    timer = StageTimer(capacity=8192)
    box_provider = (
        detector_box_provider(args.models, args.det_engine, timer)
        if args.boxes == "detector"
        else None
    )
    engine_depth: DepthProvider | None = (
        engine_depth_provider(args.engine, args.models, timer) if args.engine else None
    )

    per_seq: dict[str, dict[str, Any]] = {}
    results: list[TrackingKittiResult] = []
    ok = True
    for seq_id in args.seqs:
        seq = KittiTrackingSequence(args.root, seq_id, max_frames=args.frames)
        intr = seq.intrinsics()
        geo = PinholeGeometry(intr, extr)
        stage = MetricFusionStage(intr, geo, fusion_cfg, road_cfg, aff_cfg, pit_cfg)
        src = seq.source()
        ego: EgoMotionProvider = ZeroEgoMotion()
        oxts = None
        if args.ego == "oxts":
            oxts_file = args.root / "oxts" / f"{seq_id}.txt"
            if not oxts_file.is_file():
                ap.error(f"{oxts_file} not found (download data_tracking_oxts.zip or --ego zero)")
            oxts_raw = load_oxts_file(oxts_file)
            t_ns = src.timestamps_ns
            n = min(len(t_ns), len(oxts_raw))
            oxts = oxts_raw[:n]
            if args.oxts_offset_ms != 0.0:
                oxts = resample_oxts(oxts, t_ns[:n], round(args.oxts_offset_ms * 1e6))
            ego = OxtsEgoMotion(t_ns[:n], oxts, lever_arm_fwd_m=args.lever_arm)
        depth = engine_depth
        if args.depth_dir is not None:
            depth = npy_depth_provider(args.depth_dir / seq_id)
        labels = seq.labels()
        res = run_tracking_kitti(
            src,
            labels,
            stage,
            Tracker3D(trk_cfg, ego),
            gt_kinematics(labels, geo, oxts),
            depth,
            box_provider,
            gt_alt=gt_kinematics(labels, geo, oxts, half_window=args.gt_alt_window),
        )
        d = res.to_dict()
        per_seq[seq_id] = d
        results.append(res)
        print(
            f"{seq_id}: vel RMSE {d['rmse_vel_rel_mps']:.3f} m/s (vz {d['rmse_vz_rel_mps']:.3f}, "
            f"n={d['n_vel_samples']}, ref {d['velocity_reference']}) · IDSW {d['id_switches']} "
            f"/ {d['n_gt_ids']} ids ({d['idsw_per_100_matches']:.2f}/100 matches) · "
            f"static {d['static_frac']:.3f} (n={d['static_samples']}) · tracker P95 "
            f"{_p95(d['tracker_ms_p50_p95_p99']):.2f} ms, fusion P95 "
            f"{_p95(d['fusion_ms_p50_p95_p99']):.2f} ms"
        )
        for name, st in d["diagnostics"]["by_distance_m"].items():
            if st["n"]:
                print(
                    f"    Z {name:>5} m: n={st['n']:4d} RMSE {st['rmse']:.2f} P50 {st['p50']:.2f} "
                    f"P95 {st['p95']:.2f} bias ({st['bias_xz'][0]:+.2f}, {st['bias_xz'][1]:+.2f})"
                    f" · GT spread {st['gt_spread_rmse']:.2f}"
                )
        for name in ("vz_err_vs_ref_vz", "vz_err_vs_ego_fwd", "vz_err_vs_ref_az"):
            fit = d["diagnostics"][name]
            if "slope" in fit:
                print(
                    f"    {name}: slope {fit['slope']:+.3f} intercept {fit['intercept']:+.2f} "
                    f"resid {fit['resid_rms']:.2f} (n={fit['n']})"
                )
        for group in ("by_ref_az_mps2", "by_track_age_s"):
            for name, st in d["diagnostics"][group].items():
                if st["n"]:
                    print(
                        f"    {group} {name:>9}: n={st['n']:4d} RMSE {st['rmse']:.2f} "
                        f"bias ({st['bias_xz'][0]:+.2f}, {st['bias_xz'][1]:+.2f})"
                    )
        for name, st in d["diagnostics"].get("by_yaw_rate", {}).items():
            if st["n"]:
                print(
                    f"    {name:>20}: n={st['n']:4d} RMSE {st['rmse']:.2f} "
                    f"bias ({st['bias_xz'][0]:+.2f}, {st['bias_xz'][1]:+.2f})"
                )
        _print_turn(d["diagnostics"].get("turn"))
        _print_consistency(d["diagnostics"].get("consistency"))
        _print_range_error(d["diagnostics"].get("range_error"))
        ok &= d["rmse_vel_rel_mps"] <= 1.0
        if args.ego == "oxts":
            ok &= d["static_frac"] >= 0.9

    pooled = merge_results(results).diagnostics()
    if len(results) > 1:
        print(f"\npooled ({len(results)} seqs):")
        for name, st in pooled.get("by_yaw_rate", {}).items():
            if st["n"]:
                print(
                    f"    {name:>20}: n={st['n']:4d} RMSE {st['rmse']:.2f} "
                    f"bias ({st['bias_xz'][0]:+.2f}, {st['bias_xz'][1]:+.2f})"
                )
        _print_turn(pooled.get("turn"))
        _print_consistency(pooled.get("consistency"))
        _print_range_error(pooled.get("range_error"))

    depth_mode = (
        "engine" if args.engine else ("npy" if args.depth_dir else "none (geometry+height)")
    )
    print(
        f"\ndepth: {depth_mode} · boxes: {args.boxes} · ego: {args.ego} · DoD {'PASS' if ok else 'FAIL'}"
    )
    if timer.stage_names:
        print(timer.format_table())
    if args.json is not None:
        payload = {
            "ego": args.ego,
            "depth": depth_mode,
            "boxes": args.boxes,
            "robust_chi2": trk_cfg.robust_chi2,
            "q_vehicle": trk_cfg.dynamics_for("car").q,
            "r_scale": trk_cfg.r_scale,
            "range_bias": None
            if trk_cfg.range_bias is None
            else {"sigma": trk_cfg.range_bias.sigma, "tau_s": trk_cfg.range_bias.tau_s},
            "arbitration": fusion_cfg.arbitration,
            "sigma_pitch_deg": float(np.rad2deg(fusion_cfg.sigma_pitch_rad)),
            "pitch_q_deg_per_sqrt_s": float(np.rad2deg(pit_cfg.q_rad_per_sqrt_s)),
            "pitch_filter": args.pitch_filter,
            "nominal_pitch_deg": float(np.rad2deg(extr.pitch_rad)),
            "lever_arm_m": args.lever_arm,
            "oxts_offset_ms": args.oxts_offset_ms,
            "per_seq": per_seq,
            "pooled_diagnostics": pooled,
            "pass": ok,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
