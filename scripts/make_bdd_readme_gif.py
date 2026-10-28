"""Build docs/assets/demo.gif from real BDD100K val keyframes + detector boxes.

Picks night / rain / highway stems, letterboxes through the TensorRT detector,
draws boxes, and writes a GIF via ffmpeg (same palette path as make_readme_assets).

Example (WSL + runtime)::

    uv run --extra runtime python scripts/make_bdd_readme_gif.py \\
        --root ~/datasets/bdd100k \\
        --engine models/detector_672x384_fp16.engine \\
        --config configs/models_bdd.yaml \\
        --out docs/assets
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from percepcion3d.detection.config import load_detector_config  # noqa: E402
from percepcion3d.detection.detector_trt import Detector, TrtEngine  # noqa: E402
from percepcion3d.eval.bdd import (  # noqa: E402
    load_det_frame_meta,
    pick_demo_candidates,
    resolve_image_path,
)

ROOT = Path(__file__).resolve().parents[1]
KEEP = {0, 1, 2, 3, 5, 7}  # person, bicycle, car, motorcycle, bus, truck
COLORS = {
    0: (90, 60, 170),
    1: (40, 180, 120),
    2: (170, 120, 60),
    3: (60, 160, 220),
    5: (30, 140, 255),
    7: (50, 50, 200),
}


def _ensure_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    # imageio-ffmpeg ships a static binary (no apt/sudo).
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "ffmpeg not found; install system ffmpeg or `uv pip install imageio-ffmpeg`"
        ) from exc


def write_gif(frames: list[NDArray[np.uint8]], path: Path, fps: int, width: int) -> None:
    ffmpeg = _ensure_ffmpeg()
    with tempfile.TemporaryDirectory() as tmp:
        for i, fr in enumerate(frames):
            h, w = fr.shape[:2]
            scaled: NDArray[np.uint8] = fr
            if w != width:
                nh = max(1, int(round(h * width / w)))
                scaled = np.asarray(
                    cv2.resize(fr, (width, nh), interpolation=cv2.INTER_AREA), dtype=np.uint8
                )
            cv2.imwrite(str(Path(tmp) / f"{i:04d}.png"), scaled)
        vf = (
            f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
            "[a]palettegen=max_colors=128:stats_mode=diff[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=4"
        )
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-i",
                str(Path(tmp) / "%04d.png"),
                "-vf",
                vf,
                "-loop",
                "0",
                str(path),
            ],
            check=True,
        )


def _draw(
    img: NDArray[np.uint8],
    boxes: NDArray[np.float32],
    scores: NDArray[np.float32],
    classes: NDArray[np.int32],
    name_of: Callable[[int], str],
    tag: str,
) -> NDArray[np.uint8]:
    out = img.copy()
    for i in range(len(boxes)):
        cls = int(classes[i])
        if cls not in KEEP:
            continue
        x1, y1, x2, y2 = (int(v) for v in boxes[i])
        color = COLORS.get(cls, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{name_of(cls)} {float(scores[i]):.2f}"
        cv2.putText(out, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(out, tag, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--engine", type=Path, default=ROOT / "models" / "detector_672x384_fp16.engine")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "models_bdd.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "assets")
    ap.add_argument("--per-bucket", type=int, default=8)
    ap.add_argument("--hold", type=int, default=4, help="repeat each keyframe for GIF continuity")
    ap.add_argument("--gif-fps", type=int, default=12)
    ap.add_argument("--gif-width", type=int, default=900)
    args = ap.parse_args()

    images_dir = args.root / "images" / "100k" / "val"
    labels = args.root / "labels" / "det_20" / "det_val.json"
    meta = load_det_frame_meta(labels)
    buckets = pick_demo_candidates(meta, limit_per_bucket=args.per_bucket)

    model_cfg = load_detector_config(args.config)
    runtime_cfg = model_cfg.runtime_config()
    det = Detector(TrtEngine(str(args.engine)), runtime_cfg)

    frames: list[NDArray[np.uint8]] = []
    for kind in ("night", "rain", "highway"):
        for stem in buckets.get(kind, []):
            path = resolve_image_path(images_dir, f"{stem}.jpg")
            if path is None:
                continue
            raw = cv2.imread(str(path))
            if raw is None:
                continue
            img = np.asarray(raw, dtype=np.uint8)
            dets = det.infer(img)
            annotated = _draw(img, dets.boxes, dets.scores, dets.classes, runtime_cfg.name_of, kind)
            for _ in range(max(1, args.hold)):
                frames.append(annotated)
    if not frames:
        raise SystemExit("no frames rendered — check BDD root / labels attributes")

    args.out.mkdir(parents=True, exist_ok=True)
    gif = args.out / "demo.gif"
    write_gif(frames, gif, args.gif_fps, args.gif_width)
    still = args.out / "demo_frame.png"
    cv2.imwrite(str(still), frames[len(frames) // 2])
    print(f"wrote {gif} ({gif.stat().st_size / 1024:.0f} KiB, {len(frames)} frames)")
    print(f"wrote {still}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
