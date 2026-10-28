"""Build a short 720p30 MP4 from BDD100K val keyframes for F7 demo / P99.

Picks night / rain / highway frames via Detection attributes when available,
then fills with more val images up to ``--frames``.

Example::

    uv run python scripts/make_bdd_demo_video.py --root ~/datasets/bdd100k \\
        --out data/outputs/bdd_demo_720p30.mp4 --frames 150
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from percepcion3d.eval.bdd import load_det_frame_meta, pick_demo_candidates, resolve_image_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/outputs/bdd_demo_720p30.mp4"))
    ap.add_argument("--frames", type=int, default=150, help="distinct keyframes to use")
    ap.add_argument(
        "--hold",
        type=int,
        default=1,
        help="repeat each keyframe N times (simulates temporal continuity)",
    )
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    images_dir = args.root / "images" / "100k" / "val"
    labels = args.root / "labels" / "det_20" / "det_val.json"
    meta = load_det_frame_meta(labels) if labels.is_file() else {}
    buckets = pick_demo_candidates(meta, limit_per_bucket=25) if meta else {}

    stems: list[str] = []
    for kind in ("night", "rain", "highway"):
        stems.extend(buckets.get(kind, []))
    # Dedup preserving order, then fill from sorted val names.
    seen: set[str] = set()
    ordered: list[str] = []
    for s in stems:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    for name in sorted(p.stem for p in images_dir.glob("*.jpg")):
        if name not in seen:
            ordered.append(name)
            seen.add(name)
        if len(ordered) >= args.frames:
            break
    ordered = ordered[: args.frames]
    if not ordered:
        raise SystemExit(f"no images under {images_dir}")

    paths: list[Path] = []
    for stem in ordered:
        p = resolve_image_path(images_dir, f"{stem}.jpg")
        if p is not None:
            paths.append(p)
    if not paths:
        raise SystemExit("resolved 0 image paths")

    first = cv2.imread(str(paths[0]))
    if first is None:
        raise SystemExit(f"cannot read {paths[0]}")
    h, w = first.shape[:2]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, args.fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"VideoWriter failed for {args.out}")
    hold = max(1, int(args.hold))
    written = 0
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h))
        for _ in range(hold):
            writer.write(img)
            written += 1
    writer.release()

    summary = {
        "out": str(args.out),
        "n_keyframes": len(paths),
        "hold": hold,
        "n_frames": written,
        "fps": args.fps,
        "wh": [w, h],
        "buckets_used": {k: len(v) for k, v in buckets.items()},
        "first_stems": ordered[:12],
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out} ({written} frames = {len(paths)}×{hold} @ {args.fps} fps, {w}x{h})")


if __name__ == "__main__":
    main()
