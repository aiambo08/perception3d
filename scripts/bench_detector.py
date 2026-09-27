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

    # Recall pooled over several sequences (KITTI "moderate" filter, 95 % Wilson CI)
    uv run python scripts/bench_detector.py --engine models/detector_1024x320_fp16.engine \
        --kitti-root /data/kitti_tracking/training --seqs 0000 0001 0020 --frames 200

    uv run python scripts/bench_detector.py --engine ... --no-cuda-graph --json out/det.json

Recall counts GT boxes of the chosen ``--difficulty`` (default ``moderate``:
height ≥ 25 px, occluded ≤ 1, truncated ≤ 0.3); harder boxes are ignored, not
missed. The DoD verdict is PASS/FAIL only when the 95 % interval clears the
threshold, INCONCLUSIVE otherwise. The JSON also records the GPU clock /
temperature / throttle state during the run and how process VRAM was obtained
(``nvml_process``, or ``device_delta`` under WDDM where NVML has no per-process
accounting).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from percepcion3d.detection.config import DetectorModelConfig, load_detector_config  # noqa: E402
from percepcion3d.detection.detector_trt import Detections, Detector, TrtEngine  # noqa: E402
from percepcion3d.eval.detection import (  # noqa: E402
    HEIGHT_ONLY,
    KITTI_EASY,
    KITTI_MODERATE,
    ClassRecall,
    GtBox,
    PredBox,
    format_recall,
    merge_recall,
    recall_by_type,
)
from percepcion3d.eval.kitti import load_kitti_tracking_labels  # noqa: E402
from percepcion3d.io.sources import ImageSequenceSource, VideoFileSource  # noqa: E402
from percepcion3d.utils.gpu_state import GpuStateSampler, nvml_query_fn  # noqa: E402
from percepcion3d.utils.profiling import StageTimer  # noqa: E402
from percepcion3d.utils.vram import VramSampler, nvml_sample_fn  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"
GT_FILTERS = {"moderate": KITTI_MODERATE, "easy": KITTI_EASY, "all": HEIGHT_ONLY}


def _png_frames(directory: Path, n: int) -> list[NDArray[np.uint8]]:
    paths = sorted(directory.glob("*.png"))[:n]
    src = ImageSequenceSource(paths, [i * 100_000_000 for i in range(len(paths))])
    return [f.img for f in src]


def _groups(args: argparse.Namespace) -> list[tuple[str, Path | None, Path | None]]:
    """``(name, image_dir, labels)`` per evaluated sequence; image_dir ``None`` = video/synthetic."""
    if args.kitti_root is not None:
        if not args.seqs:
            raise SystemExit("--kitti-root requires --seqs")
        root = Path(args.kitti_root)
        return [
            (seq, root / "image_02" / seq, root / "label_02" / f"{seq}.txt") for seq in args.seqs
        ]
    if args.kitti:
        return [(Path(args.kitti).name, Path(args.kitti), args.labels)]
    return [("video" if args.video else "synthetic", None, None)]


def _frames(args: argparse.Namespace, image_dir: Path | None) -> list[NDArray[np.uint8]]:
    if image_dir is not None:
        return _png_frames(image_dir, args.frames)
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


def _recall_rows(
    preds: list[PredBox],
    labels_path: Path,
    n_frames: int,
    args: argparse.Namespace,
    cfg: DetectorModelConfig,
) -> dict[str, ClassRecall]:
    labels = load_kitti_tracking_labels(labels_path)
    gts = [
        GtBox(lab.frame, lab.obj_type, lab.bbox, lab.truncated, lab.occluded)
        for frame_idx, labs in labels.items()
        if frame_idx < n_frames
        for lab in labs
    ]
    return recall_by_type(
        preds, gts, type_to_classes=cfg.kitti_to_coco, gt_filter=GT_FILTERS[args.difficulty]
    )


def _recall_json(rows: dict[str, ClassRecall]) -> dict[str, Any]:
    return {
        t: {**asdict(r), "recall": r.recall, "recall_ci95": list(r.recall_ci95)}
        for t, r in rows.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--engine", type=Path, help="override detector.engine from the config")
    ap.add_argument("--kitti", type=Path, help="KITTI image_02/<seq> directory")
    ap.add_argument("--labels", type=Path, help="KITTI tracking label_02/<seq>.txt for recall")
    ap.add_argument("--kitti-root", type=Path, help="KITTI tracking training/ root (with --seqs)")
    ap.add_argument("--seqs", nargs="+", help="sequences under --kitti-root, e.g. 0000 0001")
    ap.add_argument("--difficulty", choices=tuple(GT_FILTERS), default="moderate")
    ap.add_argument("--video", type=Path)
    ap.add_argument("--frames", type=int, default=300, help="frames per sequence")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--no-cuda-graph", action="store_true")
    ap.add_argument("--no-priority", action="store_true", help="default-priority CUDA stream")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()

    cfg = load_detector_config(args.config)
    with args.config.open(encoding="utf-8") as fh:
        dod = {k: float(v) for k, v in (yaml.safe_load(fh).get("dod") or {}).items()}
    engine_path = args.engine or cfg.engine
    groups = _groups(args)
    print(f"engine={engine_path} sequences={[g[0] for g in groups]} difficulty={args.difficulty}")

    timer = StageTimer(capacity=max(4096, args.frames * len(groups)))
    per_seq: dict[str, dict[str, ClassRecall]] = {}
    n_frames_total = 0
    vram_fn = nvml_sample_fn()
    with (
        VramSampler(interval_s=0.05, sample_fn=vram_fn) as vram,
        GpuStateSampler(interval_s=0.2, query_fn=nvml_query_fn()) as gpu,
    ):
        base = vram.sample_once()
        engine = TrtEngine(
            engine_path,
            use_cuda_graph=not args.no_cuda_graph,
            high_priority_stream=not args.no_priority,
        )
        det = Detector(engine, cfg.runtime_config(), timer=timer)
        after_load = vram.sample_once()
        try:
            warmed = False
            for name, image_dir, labels_path in groups:
                frames = _frames(args, image_dir)
                if not frames:
                    raise SystemExit(f"no frames for {name}")
                if not warmed:
                    for _ in range(min(args.warmup, len(frames))):
                        det.infer(frames[0])
                    timer.reset()
                    warmed = True
                preds: list[PredBox] = []
                for i, frame in enumerate(frames):
                    t0 = time.perf_counter()
                    d = det.infer(frame)
                    timer.record("detector_e2e", (time.perf_counter() - t0) * 1e3)
                    if labels_path is not None:
                        preds.extend(_to_preds(i, d, cfg))
                n_frames_total += len(frames)
                if labels_path is not None:
                    per_seq[name] = _recall_rows(preds, labels_path, len(frames), args, cfg)
        finally:
            det.close()
        peak = vram.stop()
        gpu_summary = gpu.stop()

    print(f"frames={n_frames_total} shape={frames[0].shape}")
    print(timer.format_table())
    print(
        f"VRAM process [{vram_fn.process_source}]: baseline {base.process_mb:.0f} MB → after load "
        f"{after_load.process_mb:.0f} MB → peak {peak.process_mb:.0f} MB "
        f"(GPU total peak {peak.gpu_mb:.0f} MB; DoD ≤ {dod.get('detector_vram_mb', 900):.0f} MB "
        f"incl. contexto)"
    )
    print(gpu_summary.format())
    p95 = timer.stats("detector_e2e").p95_ms
    print(f"detector_e2e P95 = {p95:.2f} ms (DoD ≤ {dod.get('detector_p95_ms', 6.0)} ms aislado)")

    pooled = merge_recall(per_seq.values()) if per_seq else {}
    for name, rows in per_seq.items():
        print(f"\n--- recall seq {name} ({args.difficulty}) ---")
        print(format_recall(rows))
    if len(per_seq) > 1:
        print(f"\n--- recall pooled over {len(per_seq)} sequences ---")
        print(format_recall(pooled))
    verdicts: dict[str, str] = {}
    for kitti_type, key in (("Car", "recall_car"), ("Pedestrian", "recall_pedestrian")):
        if kitti_type in pooled and key in dod:
            r = pooled[kitti_type]
            lo, hi = r.recall_ci95
            verdicts[kitti_type] = r.verdict(dod[key])
            print(
                f"recall {kitti_type} {r.recall:.3f} CI95 [{lo:.3f}, {hi:.3f}] (n={r.n_gt}) "
                f"vs DoD ≥ {dod[key]} → {verdicts[kitti_type]}"
            )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "engine": str(engine_path),
            "frames": n_frames_total,
            "stages": {n: asdict(s) for n, s in timer.report().items()},
            "vram_mb": {
                "baseline": base.process_mb,
                "loaded": after_load.process_mb,
                "peak": peak.process_mb,
                "gpu_peak": peak.gpu_mb,
                "process_source": vram_fn.process_source,
            },
            "gpu_state": gpu_summary.to_dict(),
            "recall": {
                "difficulty": args.difficulty,
                "per_seq": {k: _recall_json(v) for k, v in per_seq.items()},
                "pooled": _recall_json(pooled),
                "verdicts": verdicts,
            },
        }
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    if args.csv:
        timer.to_csv(args.csv)


if __name__ == "__main__":
    main()
