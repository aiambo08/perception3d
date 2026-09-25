"""Benchmark the TensorRT depth engine: alone, and in contention with the detector (R3).

Runs on the target GPU (``runtime`` extra). Modes:

* ``depth``  — depth engine only, every configured size (or ``--size``): P50/P95/P99
  of ``depth.gpu`` (CUDA events) and ``depth_e2e`` (resize + H2D + engine + D2H), VRAM.
* ``matrix`` — for one size: {detector solo, depth solo, both}. In ``both`` the
  detector runs on the high-priority stream and depth on the low-priority one,
  depth enqueued first; the DoD is ``P99(det.gpu) ≤ 8 ms`` there.
* ``--kitti-drive`` adds the export sanity check: Spearman ρ between the predicted
  disparity and ``1/Z`` of the projected Velodyne returns on road pixels (below
  the horizon, outside the detections when the detector runs), DoD ρ ≥ 0.95.

Examples::

    uv run python scripts/bench_depth.py --mode depth
    uv run python scripts/bench_depth.py --mode matrix --size 924x280 --pace-hz 60
    uv run python scripts/bench_depth.py --mode matrix --kitti-drive \
        /data/kitti_raw/2011_09_26/2011_09_26_drive_0005_sync --frames 150 --json out/f3.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from percepcion3d.depth.config import DepthModelConfig, load_depth_config  # noqa: E402
from percepcion3d.depth.depth_trt import DepthEstimator, DepthMap  # noqa: E402
from percepcion3d.detection.config import load_detector_config  # noqa: E402
from percepcion3d.detection.detector_trt import Detections, Detector  # noqa: E402
from percepcion3d.eval.depth_sanity import DepthSanity, evaluate_depth_map, road_mask  # noqa: E402
from percepcion3d.eval.lidar import (  # noqa: E402
    LidarProjector,
    ProjectedLidar,
    load_velodyne_bin,
    nearest_per_pixel,
)
from percepcion3d.io.sources import ImageSequenceSource, KittiSequenceSource  # noqa: E402
from percepcion3d.runtime.contention import ContentionResult, run_contention  # noqa: E402
from percepcion3d.runtime.trt_engine import TrtEngine  # noqa: E402
from percepcion3d.utils.profiling import StageTimer  # noqa: E402
from percepcion3d.utils.vram import VramSampler, nvml_sample_fn  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"


def parse_size(text: str) -> tuple[int, int]:
    w, h = (int(t) for t in text.lower().split("x"))
    return h, w


@dataclass
class Scene:
    frames: list[NDArray[np.uint8]]
    lidar: list[ProjectedLidar] | None = None
    horizon_v: float | None = None


def _load_scene(args: argparse.Namespace) -> Scene:
    if args.kitti_drive:
        src = KittiSequenceSource(args.kitti_drive, max_frames=args.frames)
        frames = [f.img for f in src]
        drive = Path(args.kitti_drive)
        hw = (frames[0].shape[0], frames[0].shape[1])
        proj = LidarProjector.from_kitti_raw(drive.parent, cam_idx=2, image_hw=hw)
        velo_dir = drive / "velodyne_points" / "data"
        lidar: list[ProjectedLidar] = []
        for i in range(len(frames)):
            pts = load_velodyne_bin(velo_dir / f"{i:010d}.bin")
            lidar.append(nearest_per_pixel(proj.project(pts), hw))
        horizon = src.intrinsics.cy if src.intrinsics is not None else hw[0] * 0.5
        return Scene(frames, lidar, float(horizon))
    if args.kitti:
        paths = sorted(Path(args.kitti).glob("*.png"))[: args.frames]
        seq = ImageSequenceSource(paths, [i * 100_000_000 for i in range(len(paths))])
        return Scene([f.img for f in seq])
    rng = np.random.default_rng(0)
    return Scene(
        [rng.integers(0, 256, size=(375, 1242, 3), dtype=np.uint8) for _ in range(args.frames)]
    )


class SanityAccumulator:
    """Collects Spearman/AbsRel per depth map produced during a run."""

    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self.results: list[DepthSanity] = []
        self._boxes: dict[int, NDArray[np.float32]] = {}

    def __call__(self, k: int, dets: Detections | None, depth: DepthMap | None) -> None:
        if dets is not None:
            self._boxes[k] = dets.boxes
        if depth is None or self.scene.lidar is None or self.scene.horizon_v is None:
            return
        lidar = self.scene.lidar[depth.frame_id]
        mask = road_mask(
            (depth.resize.src_h, depth.resize.src_w),
            self.scene.horizon_v,
            self._boxes.get(depth.frame_id),
        )
        self.results.append(evaluate_depth_map(depth, lidar, mask))

    def summary(self) -> dict[str, float]:
        if not self.results:
            return {}
        rho = np.array([r.spearman for r in self.results], dtype=np.float64)
        ar = np.array([r.abs_rel for r in self.results], dtype=np.float64)
        return {
            "maps": float(len(self.results)),
            "spearman_median": float(np.nanmedian(rho)),
            "spearman_p05": float(np.nanpercentile(rho, 5)),
            "spearman_min": float(np.nanmin(rho)),
            "abs_rel_affine_median": float(np.nanmedian(ar)),
            "points_median": float(np.median([r.n for r in self.results])),
        }


def _run_mode(
    mode: str,
    scene: Scene,
    depth_cfg: DepthModelConfig,
    hw: tuple[int, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    det_cfg = load_detector_config(args.config)
    timer = StageTimer(capacity=max(4096, len(scene.frames)))
    sanity = SanityAccumulator(scene)
    det: Detector | None = None
    depth: DepthEstimator | None = None
    with VramSampler(interval_s=0.05, sample_fn=nvml_sample_fn()) as vram:
        base = vram.sample_once()
        try:
            if mode in ("det", "both"):
                det = Detector(
                    TrtEngine(
                        args.det_engine or det_cfg.engine,
                        use_cuda_graph=not args.no_cuda_graph,
                        high_priority_stream=True,
                    ),
                    det_cfg.runtime_config(),
                    timer=timer,
                )
            if mode in ("depth", "both"):
                depth = DepthEstimator(
                    TrtEngine(
                        args.engine or depth_cfg.engine_path(hw),
                        use_cuda_graph=not args.no_cuda_graph,
                        high_priority_stream=False,
                    ),
                    depth_cfg.runtime_config(),
                    timer=timer,
                )
            after_load = vram.sample_once()
            warm = scene.frames[: max(1, min(args.warmup, len(scene.frames)))]
            run_contention(det, depth, warm, StageTimer(), depth_every=1)
            timer.reset()
            res: ContentionResult = run_contention(
                det,
                depth,
                scene.frames,
                timer,
                depth_every=args.depth_every,
                pace_hz=args.pace_hz,
                on_result=sanity,
            )
        finally:
            if det is not None:
                det.close()
            if depth is not None:
                depth.close()
        peak = vram.stop()

    print(f"\n=== mode={mode} size={hw[1]}x{hw[0]} frames={res.frames} ===")
    print(timer.format_table())
    print(
        f"VRAM process: baseline {base.process_mb:.0f} → loaded {after_load.process_mb:.0f} → "
        f"peak {peak.process_mb:.0f} MB (GPU total peak {peak.gpu_mb:.0f} MB)"
    )
    if depth is not None:
        print(
            f"depth maps: {res.depth_maps} ({res.depth_rate:.2f}/frame), "
            f"skipped slots {res.depth_skipped}, max lag {res.max_lag} frames"
        )
    summary = sanity.summary()
    if summary:
        print(
            f"LiDAR sanity (road): Spearman median {summary['spearman_median']:.3f} "
            f"p05 {summary['spearman_p05']:.3f} min {summary['spearman_min']:.3f} | "
            f"AbsRel(affine) median {summary['abs_rel_affine_median']:.3f} "
            f"| pts/frame {summary['points_median']:.0f}"
        )
    return {
        "mode": mode,
        "size_hw": list(hw),
        "frames": res.frames,
        "stages": {n: asdict(s) for n, s in timer.report().items()},
        "vram_mb": {
            "baseline": base.process_mb,
            "loaded": after_load.process_mb,
            "peak": peak.process_mb,
            "gpu_peak": peak.gpu_mb,
        },
        "depth_maps": res.depth_maps,
        "depth_skipped": res.depth_skipped,
        "max_lag_frames": res.max_lag,
        "lidar_sanity": summary,
    }


def _dod_report(results: list[dict[str, Any]], dod: dict[str, float]) -> None:
    def stage(r: dict[str, Any], name: str, key: str) -> float | None:
        s = r["stages"].get(name)
        return None if s is None else float(s[key])

    print("\n=== DoD F3 ===")
    for r in results:
        if r["mode"] == "depth":
            p95 = stage(r, "depth.gpu", "p95_ms")
            if p95 is not None:
                ok = p95 <= dod["depth_p95_ms"]
                print(
                    f"depth solo P95 {p95:.2f} ms ≤ {dod['depth_p95_ms']} → {'OK' if ok else 'FAIL'}"
                )
            rho = r["lidar_sanity"].get("spearman_median")
            if rho is not None:
                ok = rho >= dod["depth_spearman_road"]
                print(
                    f"Spearman(road) median {rho:.3f} ≥ {dod['depth_spearman_road']} → {'OK' if ok else 'FAIL'}"
                )
        if r["mode"] == "both":
            p99 = stage(r, "det.gpu", "p99_ms")
            if p99 is not None:
                ok = p99 <= dod["detector_p99_contended_ms"]
                print(
                    f"detector P99 under contention {p99:.2f} ms ≤ "
                    f"{dod['detector_p99_contended_ms']} → {'OK' if ok else 'FAIL'}"
                )
            peak = float(r["vram_mb"]["peak"])
            ok = peak <= dod["total_vram_mb"]
            print(
                f"VRAM both engines peak {peak:.0f} MB ≤ {dod['total_vram_mb']} → {'OK' if ok else 'FAIL'}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--mode", choices=("depth", "matrix"), default="depth")
    ap.add_argument("--size", type=parse_size, help="WxH; default: all depth.input_sizes")
    ap.add_argument("--engine", type=Path, help="override the depth engine path (single size)")
    ap.add_argument("--det-engine", type=Path, help="override detector.engine")
    ap.add_argument("--kitti-drive", type=Path, help="KITTI raw *_sync drive (images+velodyne)")
    ap.add_argument("--kitti", type=Path, help="directory of PNG frames (no LiDAR)")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--depth-every", type=int, default=1, help="launch depth every N frames")
    ap.add_argument("--pace-hz", type=float, help="pace the frame loop (e.g. 60)")
    ap.add_argument("--no-cuda-graph", action="store_true")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    depth_cfg = load_depth_config(args.config)
    with args.config.open(encoding="utf-8") as fh:
        dod_raw: dict[str, Any] = yaml.safe_load(fh).get("dod") or {}
    dod = {k: float(v) for k, v in dod_raw.items()}
    scene = _load_scene(args)
    if not scene.frames:
        raise SystemExit("no frames")
    sizes = [args.size] if args.size else list(depth_cfg.input_sizes)
    if args.engine is not None and len(sizes) != 1:
        raise SystemExit("--engine requires --size")

    results: list[dict[str, Any]] = []
    if args.mode == "depth":
        for hw in sizes:
            results.append(_run_mode("depth", scene, depth_cfg, hw, args))
    else:
        hw = sizes[0]
        for mode in ("det", "depth", "both"):
            results.append(_run_mode(mode, scene, depth_cfg, hw, args))
    _dod_report(results, dod)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
