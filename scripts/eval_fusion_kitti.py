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
from dataclasses import asdict
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
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    if args.engine is not None and args.depth_dir is not None:
        ap.error("--engine and --depth-dir are mutually exclusive")

    seq = KittiTrackingSequence(args.root, args.seq, max_frames=args.frames)
    intr = seq.intrinsics()
    _, extr = load_camera_config_yaml(args.camera)
    geo = PinholeGeometry(intr, extr)
    fusion_cfg = load_fusion_config(args.fusion)
    road_cfg, aff_cfg, pit_cfg = load_solver_configs(args.fusion)
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
            "dod": dod,
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
