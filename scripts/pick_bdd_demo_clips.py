"""Suggest BDD100K clips for README demo (night / rain / highway).

Uses Detection 2020 attributes (weather, scene, timeofday) on val keyframes.
Video filenames share the image stem (``.mov`` under ``videos/100k/``).

Example::

    uv run python scripts/pick_bdd_demo_clips.py --root ~/datasets/bdd100k
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from percepcion3d.eval.bdd import load_det_frame_meta, pick_demo_candidates  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument(
        "--labels",
        type=Path,
        help="default: <root>/labels/det_20/det_val.json",
    )
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    labels = args.labels or (args.root / "labels" / "det_20" / "det_val.json")
    meta = load_det_frame_meta(labels)
    buckets = pick_demo_candidates(meta, limit_per_bucket=args.limit)
    print(f"from {labels} ({len(meta)} images)")
    for kind, stems in buckets.items():
        print(f"\n# {kind}")
        for s in stems:
            print(f"  {s}  →  videos/100k/{s}.mov   (o .mp4 según el ZIP)")
    print(
        "\nLazo F7 (P99 captura→alerta @ 720p):\n"
        "  HZ=30 DURATION=40 bash scripts/run_pipeline.sh video <clip.mov> \\\n"
        "      --camera configs/camera_bdd.yaml"
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(buckets, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
