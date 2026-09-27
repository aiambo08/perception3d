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
from pathlib import Path
from typing import Any

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
from percepcion3d.eval.tracking_kitti import gt_kinematics, run_tracking_kitti  # noqa: E402
from percepcion3d.tracking.ego_motion import (  # noqa: E402
    EgoMotionProvider,
    OxtsEgoMotion,
    ZeroEgoMotion,
    load_oxts_file,
)
from percepcion3d.tracking.tracker3d import Tracker3D, load_tracker_config  # noqa: E402
from percepcion3d.utils.profiling import StageTimer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


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
    ap.add_argument("--tracking", type=Path, default=ROOT / "configs" / "tracking.yaml")
    ap.add_argument("--models", type=Path, default=ROOT / "configs" / "models.yaml")
    ap.add_argument("--engine", type=Path, default=None, help="TensorRT depth engine")
    ap.add_argument("--depth-dir", type=Path, default=None, help="Pre-computed <frame>.npy maps")
    ap.add_argument("--boxes", choices=("gt", "detector"), default="gt")
    ap.add_argument("--det-engine", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    if args.engine is not None and args.depth_dir is not None:
        ap.error("--engine and --depth-dir are mutually exclusive")

    _, extr = load_camera_config_yaml(args.camera)
    fusion_cfg = load_fusion_config(args.fusion)
    road_cfg, aff_cfg, pit_cfg = load_solver_configs(args.fusion)
    trk_cfg = load_tracker_config(args.tracking)
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
            oxts = load_oxts_file(oxts_file)
            t_ns = src.timestamps_ns
            n = min(len(t_ns), len(oxts))
            ego = OxtsEgoMotion(t_ns[:n], oxts[:n], lever_arm_fwd_m=args.lever_arm)
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
        )
        d = res.to_dict()
        per_seq[seq_id] = d
        print(
            f"{seq_id}: vel RMSE {d['rmse_vel_rel_mps']:.3f} m/s (vz {d['rmse_vz_rel_mps']:.3f}, "
            f"n={d['n_vel_samples']}, ref {d['velocity_reference']}) · IDSW {d['id_switches']} "
            f"/ {d['n_gt_ids']} ids · static {d['static_frac']:.3f} (n={d['static_samples']})"
        )
        ok &= d["rmse_vel_rel_mps"] <= 1.0
        if args.ego == "oxts":
            ok &= d["static_frac"] >= 0.9

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
            "per_seq": per_seq,
            "pass": ok,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
