"""Benchmark the TensorRT detector in isolation: P50/P95/P99 latency + VRAM.

Runs on the target GPU (``runtime`` extra). Reports two latencies per frame:

* ``detector_gpu``  — CUDA-event time inside TrtEngine (H2D → engine → D2H).
* ``detector_e2e``  — wall time of ``Detector.infer`` including letterbox and
  the frame-coordinate post-filter (what the pipeline actually pays).

Examples::

    # Synthetic frames at KITTI resolution
    uv run python scripts/bench_detector.py --engine models/detector_1024x320_fp16.engine

    # Real KITTI frames + recall against tracking labels (export validation)
    uv run python scripts/bench_detector.py --engine models/detector_1024x320_fp16.engine \
        --kitti /data/kitti_tracking/training/image_02/0000 \
        --labels /data/kitti_tracking/training/label_02/0000.txt --frames 200

    uv run python scripts/bench_detector.py --engine ... --no-cuda-graph --json out/det.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from percepcion3d.detection.config import DetectorModelConfig, load_detector_config  # noqa: E402
from percepcion3d.detection.detector_trt import Detections, Detector, TrtEngine  # noqa: E402
from percepcion3d.eval.detection import (  # noqa: E402
    GtBox,
    PredBox,
    format_recall,
    recall_by_type,
)
from percepcion3d.eval.kitti import load_kitti_tracking_labels  # noqa: E402
from percepcion3d.io.sources import ImageSequenceSource, VideoFileSource  # noqa: E402
from percepcion3d.utils.profiling import StageTimer  # noqa: E402
from percepcion3d.utils.vram import VramSampler, nvml_sample_fn  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"


def _frames(args: argparse.Namespace) -> list[NDArray[np.uint8]]:
    if args.kitti:
        paths = sorted(Path(args.kitti).glob("*.png"))[: args.frames]
        src = ImageSequenceSource(paths, [i * 100_000_000 for i in range(len(paths))])
        return [f.img for f in src]
    if args.video:
        vs = VideoFileSource(args.video)
        out: list[NDArray[np.uint8]] = []
        for f in vs:
            out.append(f.img)
            if len(out) >= args.frames:
                break
        vs.close()
        return out
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, size=(375, 1242, 3), dtype=np.uint8) for _ in range(args.frames)]


def _to_preds(frame_idx: int, det: Detections, cfg: DetectorModelConfig) -> list[PredBox]:
    names = cfg.class_names
    return [
        PredBox(
            frame_idx,
            names[int(c)] if int(c) < len(names) else str(int(c)),
            float(s),
            (float(b[0]), float(b[1]), float(b[2]), float(b[3])),
        )
        for b, s, c in zip(det.boxes, det.scores, det.classes, strict=True)
    ]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--engine", type=Path, help="override detector.engine from the config")
    ap.add_argument("--kitti", type=Path, help="KITTI image_02/<seq> directory")
    ap.add_argument("--labels", type=Path, help="KITTI tracking label_02/<seq>.txt for recall")
    ap.add_argument("--video", type=Path)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--no-cuda-graph", action="store_true")
    ap.add_argument("--no-priority", action="store_true", help="default-priority CUDA stream")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()

    cfg = load_detector_config(args.config)
    engine_path = args.engine or cfg.engine
    frames = _frames(args)
    if not frames:
        raise SystemExit("no frames")
    print(f"engine={engine_path} frames={len(frames)} shape={frames[0].shape}")

    timer = StageTimer(capacity=max(4096, len(frames)))
    preds: list[PredBox] = []
    with VramSampler(interval_s=0.05, sample_fn=nvml_sample_fn()) as vram:
        base = vram.sample_once()
        engine = TrtEngine(
            engine_path,
            use_cuda_graph=not args.no_cuda_graph,
            high_priority_stream=not args.no_priority,
        )
        det = Detector(engine, cfg.runtime_config(), timer=timer)
        after_load = vram.sample_once()
        try:
            for _ in range(min(args.warmup, len(frames))):
                det.infer(frames[0])
            timer.reset()
            for i, frame in enumerate(frames):
                t0 = time.perf_counter()
                d = det.infer(frame)
                timer.record("detector_e2e", (time.perf_counter() - t0) * 1e3)
                if args.labels:
                    preds.extend(_to_preds(i, d, cfg))
        finally:
            det.close()
        peak = vram.stop()

    print(timer.format_table())
    print(
        f"VRAM process: baseline {base.process_mb:.0f} MB → after load "
        f"{after_load.process_mb:.0f} MB → peak {peak.process_mb:.0f} MB "
        f"(DoD ≤ 900 MB incl. contexto)"
    )
    p95 = timer.stats("detector_e2e").p95_ms
    print(f"detector_e2e P95 = {p95:.2f} ms (DoD ≤ 6 ms aislado)")

    if args.labels:
        labels = load_kitti_tracking_labels(args.labels)
        gts = [
            GtBox(lab.frame, lab.obj_type, lab.bbox)
            for frame_idx, labs in labels.items()
            if frame_idx < len(frames)
            for lab in labs
        ]
        rows = recall_by_type(preds, gts, type_to_classes=cfg.kitti_to_coco)
        print(format_recall(rows))

    if args.json:
        timer.to_json(args.json)
    if args.csv:
        timer.to_csv(args.csv)


if __name__ == "__main__":
    main()
