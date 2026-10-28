"""F2 detector recall on BDD100K Detection 2020 val (real dashcam, beyond KITTI).

Runs the TensorRT detector over the ~10K val keyframes, matches greedily by IoU
against ``det_val.json``, and reports per-BDD-class recall with a 95 % Wilson CI
(same metric stack as ``bench_detector.py`` on KITTI).

BDD → COCO hit map (default): car/bus/truck/train/bicycle/motorcycle direct;
pedestrian and rider → person. Pass ``--with-signs`` to also score traffic light
and traffic sign (sign ≈ COCO ``stop sign`` only).

Layout expected (see ``scripts/download_bdd100k.sh eval``)::

    <root>/images/100k/val/*.jpg
    <root>/labels/det_20/det_val.json

Examples::

    uv run python scripts/eval_detector_bdd.py --root ~/datasets/bdd100k \\
        --engine models/detector_1024x320_fp16.engine --json reports/f2_bdd_val.json

    # Latency-only subset (first 500 images with GT)
    uv run python scripts/eval_detector_bdd.py --root ~/datasets/bdd100k --max-images 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from percepcion3d.detection.config import DetectorModelConfig, load_detector_config  # noqa: E402
from percepcion3d.detection.detector_trt import Detections, Detector, TrtEngine  # noqa: E402
from percepcion3d.eval.bdd import (  # noqa: E402
    BDD_TO_COCO_WITH_SIGNS,
    DEFAULT_BDD_TO_COCO,
    flatten_det_gts,
    load_det_labels,
    resolve_image_path,
    summary_counts,
)
from percepcion3d.eval.detection import (  # noqa: E402
    HEIGHT_ONLY,
    ClassRecall,
    PredBox,
    format_recall,
    recall_by_type,
)
from percepcion3d.utils.bench_history import archive_report  # noqa: E402
from percepcion3d.utils.gpu_state import GpuStateSampler, nvml_query_fn  # noqa: E402
from percepcion3d.utils.profiling import StageTimer  # noqa: E402
from percepcion3d.utils.vram import VramSampler, nvml_sample_fn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "models.yaml"

# BDD has no KITTI truncation/occlusion; score all boxes above a min height.
GT_FILTERS = {"height": HEIGHT_ONLY}


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


def _read_bgr(path: Path) -> NDArray[np.uint8]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return np.asarray(img, dtype=np.uint8)


def _recall_json(rows: dict[str, ClassRecall]) -> dict[str, Any]:
    return {
        t: {**asdict(r), "recall": r.recall, "recall_ci95": list(r.recall_ci95)}
        for t, r in rows.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, required=True, help="bdd100k/ root")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--engine", type=Path, help="override detector.engine")
    ap.add_argument(
        "--labels",
        type=Path,
        help="default: <root>/labels/det_20/det_val.json",
    )
    ap.add_argument(
        "--images",
        type=Path,
        help="default: <root>/images/100k/val",
    )
    ap.add_argument("--with-signs", action="store_true", help="also score traffic light/sign")
    ap.add_argument("--max-images", type=int, default=0, help="0 = all labelled images found")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--no-cuda-graph", action="store_true")
    ap.add_argument("--no-priority", action="store_true")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--archive", action="store_true")
    args = ap.parse_args()

    images_dir = args.images or (args.root / "images" / "100k" / "val")
    labels_path = args.labels or (args.root / "labels" / "det_20" / "det_val.json")
    if not labels_path.is_file():
        raise SystemExit(f"labels not found: {labels_path}")
    if not images_dir.is_dir():
        raise SystemExit(f"images dir not found: {images_dir}")

    type_map = BDD_TO_COCO_WITH_SIGNS if args.with_signs else DEFAULT_BDD_TO_COCO
    by_name = load_det_labels(labels_path, keep_categories=tuple(type_map.keys()))
    stats = summary_counts(by_name)
    print(f"labels={labels_path} images={images_dir}")
    print(
        f"labelled images={stats['n_images']} boxes={stats['n_boxes']} "
        f"per_cat={stats['per_category']}"
    )

    names: list[str] = []
    paths: list[Path] = []
    for name in sorted(by_name):
        p = resolve_image_path(images_dir, name)
        if p is None:
            continue
        names.append(name)
        paths.append(p)
        if args.max_images and len(names) >= args.max_images:
            break
    if not names:
        raise SystemExit(
            f"no images from {labels_path} found under {images_dir}; "
            "run scripts/download_bdd100k.sh eval"
        )
    missing = stats["n_images"] - len(names) if not args.max_images else 0
    if missing:
        print(f"warning: {missing} labelled images missing on disk (skipped)")

    gts, _ = flatten_det_gts(by_name, names)
    cfg = load_detector_config(args.config)
    with args.config.open(encoding="utf-8") as fh:
        dod = {k: float(v) for k, v in (yaml.safe_load(fh).get("dod") or {}).items()}
    engine_path = args.engine or cfg.engine
    print(f"engine={engine_path} n_eval={len(names)}")

    timer = StageTimer(capacity=max(4096, len(names)))
    preds: list[PredBox] = []
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
            first = _read_bgr(paths[0])
            for _ in range(min(args.warmup, len(paths))):
                det.infer(first)
            timer.reset()
            for i, path in enumerate(paths):
                frame = _read_bgr(path)
                t0 = time.perf_counter()
                d = det.infer(frame)
                timer.record("detector_e2e", (time.perf_counter() - t0) * 1e3)
                preds.extend(_to_preds(i, d, cfg))
        finally:
            det.close()
        peak = vram.stop()
        gpu_summary = gpu.stop()

    print(f"frames={len(paths)} shape={first.shape}")
    print(timer.format_table())
    print(
        f"VRAM process [{vram_fn.process_source}]: baseline {base.process_mb:.0f} MB → after load "
        f"{after_load.process_mb:.0f} MB → peak {peak.process_mb:.0f} MB "
        f"(GPU total peak {peak.gpu_mb:.0f} MB)"
    )
    print(gpu_summary.format())
    p95 = timer.stats("detector_e2e").p95_ms
    print(f"detector_e2e P95 = {p95:.2f} ms (KITTI DoD ≤ {dod.get('detector_p95_ms', 6.0)} ms)")

    rows = recall_by_type(preds, gts, type_to_classes=type_map, gt_filter=GT_FILTERS["height"])
    print("\n--- recall BDD val (height ≥ 25 px) ---")
    print(format_recall(rows))

    # Informational verdicts vs KITTI DoD thresholds where categories align.
    verdicts: dict[str, str] = {}
    for bdd_type, key in (("car", "recall_car"), ("pedestrian", "recall_pedestrian")):
        if bdd_type in rows and key in dod:
            r = rows[bdd_type]
            lo, hi = r.recall_ci95
            verdicts[bdd_type] = r.verdict(dod[key])
            print(
                f"recall {bdd_type} {r.recall:.3f} CI95 [{lo:.3f}, {hi:.3f}] (n={r.n_gt}) "
                f"vs KITTI DoD ≥ {dod[key]} → {verdicts[bdd_type]} (informational)"
            )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "dataset": "bdd100k_det20_val",
            "engine": str(engine_path),
            "frames": len(paths),
            "labels": str(labels_path),
            "images": str(images_dir),
            "bdd_to_coco": {k: list(v) for k, v in type_map.items()},
            "label_stats": stats,
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
                "gt_filter": "height_only",
                "pooled": _recall_json(rows),
                "verdicts_vs_kitti_dod": verdicts,
            },
        }
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
        if args.archive:
            print(f"archived {archive_report(out, 'detector_bdd')}")


if __name__ == "__main__":
    main()
