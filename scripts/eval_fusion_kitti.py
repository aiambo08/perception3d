"""F4 DoD on KITTI tracking: AbsRel of the fused depth for non-truncated Cars.

DoD: AbsRel ≤ 10 % at 0–30 m and ≤ 20 % at 30–60 m (near-face depth vs. 3D labels).

Depth source (pick one):

* ``--engine PATH``   TensorRT depth engine → runs on the target GPU (``runtime`` extra).
* ``--depth-dir DIR`` pre-computed relative inverse-depth maps ``DIR/<frame:06d>.npy``
  (float, canvas resolution) — e.g. dumped once on the GPU, evaluated anywhere.
* neither             geometry + height prior only (CPU baseline; no network cue).

Boxes: ``--boxes gt`` (default, isolates the fusion) or ``--boxes detector`` (matched to
the labels at IoU ≥ 0.5; needs ``--det-engine`` or ``configs/models.yaml``).

Examples::

    uv run python scripts/eval_fusion_kitti.py --root /data/kitti_tracking/training --seq 0000
    uv run python scripts/eval_fusion_kitti.py --root ... --seq 0001 --engine models/depth_924x280.engine \
        --json out/f4_kitti_0001.json
    uv run python scripts/eval_fusion_kitti.py --root ... --seq 0000 --depth-dir out/depth_0000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
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
from percepcion3d.eval.fusion_kitti import (  # noqa: E402
    DepthProvider,
    KittiTrackingSequence,
    flag_histogram,
    run_fusion_kitti,
)
from percepcion3d.eval.providers import (  # noqa: E402
    detector_box_provider,
    engine_depth_provider,
    npy_depth_provider,
)
from percepcion3d.utils.profiling import StageTimer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, required=True, help="KITTI tracking training/ folder")
    ap.add_argument("--seq", default="0000")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--camera", type=Path, default=ROOT / "configs" / "camera_kitti.yaml")
    ap.add_argument("--fusion", type=Path, default=ROOT / "configs" / "fusion.yaml")
    ap.add_argument("--models", type=Path, default=ROOT / "configs" / "models.yaml")
    ap.add_argument("--engine", type=Path, default=None, help="TensorRT depth engine")
    ap.add_argument("--depth-dir", type=Path, default=None, help="Pre-computed <frame>.npy maps")
    ap.add_argument("--boxes", choices=("gt", "detector"), default="gt")
    ap.add_argument("--det-engine", type=Path, default=None)
    ap.add_argument("--max-occluded", type=int, default=1)
    ap.add_argument("--no-online-pitch", action="store_true")
    ap.add_argument(
        "--arbitration",
        choices=("select", "inflate"),
        default=None,
        help="Override gates.arbitration of --fusion",
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
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    if args.engine is not None and args.depth_dir is not None:
        ap.error("--engine and --depth-dir are mutually exclusive")

    seq = KittiTrackingSequence(args.root, args.seq, max_frames=args.frames)
    intr = seq.intrinsics()
    _, extr = load_camera_config_yaml(args.camera)
    if args.nominal_pitch_deg is not None:
        extr = replace(extr, pitch_rad=float(np.deg2rad(args.nominal_pitch_deg)))
    geo = PinholeGeometry(intr, extr)
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
    timer = StageTimer(capacity=8192)
    stage = MetricFusionStage(
        intr, geo, fusion_cfg, road_cfg, aff_cfg, pit_cfg,
        online_pitch=not args.no_online_pitch, timer=timer,
    )  # fmt: skip

    depth_provider: DepthProvider | None = None
    if args.engine is not None:
        depth_provider = engine_depth_provider(args.engine, args.models, timer)
    elif args.depth_dir is not None:
        depth_provider = npy_depth_provider(args.depth_dir)
    box_provider = (
        detector_box_provider(args.models, args.det_engine, timer)
        if args.boxes == "detector"
        else None
    )

    res = run_fusion_kitti(
        seq.source(),
        seq.labels(),
        stage,
        depth_provider,
        box_provider,
        max_occluded=args.max_occluded,
        timer_ms=lambda: time.perf_counter() * 1e3,
    )
    depth_mode = (
        "engine" if args.engine else ("npy" if args.depth_dir else "none (geometry+height)")
    )
    print(f"KITTI tracking {args.seq} · depth: {depth_mode} · boxes: {args.boxes}")
    print(res.format())
    if timer.stage_names:
        print("\nstage timing [ms]:")
        print(timer.format_table())
    print("\nflags on scored samples:", flag_histogram(res))
    dod = res.dod()
    stab = res.stability()
    print(
        f"\nstability: {stab['n_pairs']} consecutive pairs · dominant-cue switch "
        f"{stab['switch_frac']:.3f} · |ΔZ err| {stab['abs_dz_err_m']} m "
        f"(switch {stab['abs_dz_err_m_switch']}, no switch {stab['abs_dz_err_m_no_switch']})"
    )
    print("\npitch (filtered vs. height-implied):", res.pitch_summary())
    print("\nDoD:")
    for k, v in dod.items():
        print(f"  {k}: {v}")
    if depth_provider is None:
        print(
            "  (no network cue: this is the geometry + height-prior baseline, not the F4 DoD run)"
        )
    if args.json is not None:
        payload: dict[str, Any] = {
            "seq": args.seq,
            "depth": depth_mode,
            "boxes": args.boxes,
            "arbitration": fusion_cfg.arbitration,
            "dod": dod,
            "stability": stab,
            "sigma_pitch_deg": float(np.rad2deg(fusion_cfg.sigma_pitch_rad)),
            "pitch_q_deg_per_sqrt_s": float(np.rad2deg(pit_cfg.q_rad_per_sqrt_s)),
            "pitch_filter": args.pitch_filter,
            "nominal_pitch_deg": float(np.rad2deg(extr.pitch_rad)),
            "pitch": res.pitch_summary(),
            "pitch_series": [asdict(p) for p in res.pitch_series],
            "flags": flag_histogram(res),
            "stages": {n: asdict(s) for n, s in timer.report().items()},
            "samples": [s.__dict__ for s in res.samples],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0 if dod.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
