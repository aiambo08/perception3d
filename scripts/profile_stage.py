"""Profile one CPU stage (or the replay path) and print P50/P95/P99 latencies.

Examples::

    # Rectification + ground geometry on synthetic frames (no dataset needed)
    uv run python scripts/profile_stage.py --stage rectify --frames 300
    uv run python scripts/profile_stage.py --stage ground_hits --frames 300

    # Replay a KITTI drive through LatestFrameSlot at 60 Hz and report jitter
    uv run python scripts/profile_stage.py --stage playback --kitti /data/kitti/2011_09_26/2011_09_26_drive_0001_sync --hz 60

    # Any stage, JSON/CSV report
    uv run python scripts/profile_stage.py --stage rectify --json out/rectify.json --csv out/rectify.csv
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from percepcion3d.camera.calibration import (  # noqa: E402
    CameraIntrinsics,
    ExtrinsicMountConfig,
    load_camera_config_yaml,
)
from percepcion3d.camera.geometry import PinholeGeometry  # noqa: E402
from percepcion3d.camera.undistort import ImageRectifier  # noqa: E402
from percepcion3d.io.sources import FrameSource, KittiSequenceSource, VideoFileSource  # noqa: E402
from percepcion3d.runtime.buffer import FrameStamped, LatestFrameSlot  # noqa: E402
from percepcion3d.runtime.playback import play_into_slot  # noqa: E402
from percepcion3d.utils.profiling import StageTimer  # noqa: E402
from percepcion3d.utils.vram import VramSampler  # noqa: E402

STAGES = ("rectify", "ground_hits", "playback")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "camera_kitti.yaml"


def _synthetic_frames(intr: CameraIntrinsics, n: int, seed: int = 0) -> list[NDArray[np.uint8]]:
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, 256, size=(intr.height, intr.width, 3), dtype=np.uint8) for _ in range(n)
    ]


def _open_source(args: argparse.Namespace) -> FrameSource | None:
    if args.kitti:
        return KittiSequenceSource(args.kitti, max_frames=args.frames)
    if args.video:
        return VideoFileSource(args.video)
    return None


def profile_rectify(
    timer: StageTimer, intr: CameraIntrinsics, frames: list[NDArray[np.uint8]], warmup: int
) -> None:
    with timer.stage("rectifier_init"):
        rect = ImageRectifier(intr)
    for i, img in enumerate(frames):
        if i < warmup:
            rect.rectify(img)
            continue
        with timer.stage("rectify"):
            rect.rectify(img)


def profile_ground_hits(
    timer: StageTimer, geo: PinholeGeometry, n_iter: int, n_points: int, warmup: int
) -> None:
    rng = np.random.default_rng(0)
    intr = geo.intrinsics
    v_h = geo.get_horizon_v()
    for i in range(n_iter + warmup):
        u = rng.uniform(0.0, intr.width - 1.0, size=n_points)
        v = rng.uniform(v_h + 1.0, intr.height - 1.0, size=n_points)
        if i < warmup:
            geo.ground_hits(u, v)
            continue
        with timer.stage(f"ground_hits[{n_points}]"):
            geo.ground_hits(u, v)


def profile_playback(
    args: argparse.Namespace, intr: CameraIntrinsics, frames: list[NDArray[np.uint8]]
) -> None:
    source = _open_source(args)
    iterable = (
        source if source is not None else [FrameStamped(i, 0, img) for i, img in enumerate(frames)]
    )
    slot: LatestFrameSlot[FrameStamped] = LatestFrameSlot()
    consumed = 0
    stop = threading.Event()

    def consumer() -> None:
        nonlocal consumed
        while not stop.is_set():
            if slot.get(timeout=0.05) is not None:
                consumed += 1

    th = threading.Thread(target=consumer, daemon=True)
    th.start()
    try:
        stats = play_into_slot(iterable, slot, target_hz=args.hz, max_frames=args.frames)
    finally:
        stop.set()
        th.join()
        if source is not None:
            source.close()

    print(
        f"playback: {stats.frames} frames in {stats.elapsed_s:.2f} s → {stats.achieved_hz:.1f} Hz "
        f"(target {stats.target_hz}); consumed {consumed}, dropped {stats.dropped}"
    )
    print(
        f"period  P50/P95/P99 = {stats.period_p50_ms:.3f} / {stats.period_p95_ms:.3f} / "
        f"{stats.period_p99_ms:.3f} ms"
    )
    print(
        f"jitter  P50/P95/P99 = {stats.jitter_p50_ms:.3f} / {stats.jitter_p95_ms:.3f} / "
        f"{stats.jitter_p99_ms:.3f} ms"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="camera YAML")
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--points", type=int, default=20000, help="pixels per ground_hits call")
    p.add_argument("--hz", type=float, default=60.0, help="playback target rate (0 = unpaced)")
    p.add_argument("--kitti", type=Path, default=None, help="KITTI *_drive_XXXX_sync folder")
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--vram", action="store_true", help="sample NVML while profiling (needs GPU)")
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args(argv)

    intr: CameraIntrinsics
    extr: ExtrinsicMountConfig
    intr, extr = load_camera_config_yaml(args.config)
    geo = PinholeGeometry(intr, extr)
    timer = StageTimer(capacity=max(args.frames, 16))

    vram = VramSampler(interval_s=0.1) if args.vram else None
    if vram is not None:
        vram.start()

    frames = _synthetic_frames(intr, min(args.frames + args.warmup, 64))
    frames = (frames * (args.frames // len(frames) + 2))[: args.frames + args.warmup]

    if args.stage == "rectify":
        profile_rectify(timer, intr, frames, args.warmup)
    elif args.stage == "ground_hits":
        profile_ground_hits(timer, geo, args.frames, args.points, args.warmup)
    else:
        profile_playback(args, intr, frames)

    if vram is not None:
        peak = vram.stop()
        print(
            f"VRAM peak: process {peak.process_mb:.0f} MB, GPU {peak.gpu_mb:.0f} MB ({peak.samples} samples)"
        )

    if timer.stage_names:
        print(timer.format_table())
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        timer.to_json(args.json)
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        timer.to_csv(args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
