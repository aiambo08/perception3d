"""F5 · 2D tracking on BDD100K MOT 2020 val (ByteTrack IDs vs Scalabel GT).

Measures identity quality of the *2D* association only (ID switches / match
rate). Metric 3D stays qualitative on BDD — no LiDAR / calibrated depth GT.

Layout (after ``scripts/download_bdd100k.sh mot`` + MOT 2020 Images)::

    <root>/images/track/val/<video>/*.jpg
    <root>/labels/box_track_20/val/<video>.json

Examples::

    uv run python scripts/eval_tracking_bdd.py --root ~/datasets/bdd100k \\
        --engine models/detector_1024x320_fp16.engine --max-videos 20 \\
        --json reports/f5_bdd_mot.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from percepcion3d.detection.config import load_detector_config  # noqa: E402
from percepcion3d.detection.detector_trt import Detector, TrtEngine  # noqa: E402
from percepcion3d.eval.bdd import (  # noqa: E402
    DEFAULT_BDD_TO_COCO,
    list_mot_val_videos,
    load_mot_sequence,
)
from percepcion3d.eval.detection import iou_xyxy  # noqa: E402
from percepcion3d.tracking.byte_tracker import (  # noqa: E402
    ByteTrackConfig,
    ByteTracker,
    TrackState,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "models.yaml"
FPS_MOT = 5.0  # MOT 2020 is resampled to 5 Hz


@dataclass
class Mot2dResult:
    video: str
    n_frames: int = 0
    n_gt: int = 0
    n_matches: int = 0
    id_switches: int = 0
    n_fp: int = 0
    n_fn: int = 0

    @property
    def idsw_rate(self) -> float:
        return self.id_switches / self.n_matches if self.n_matches else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "idsw_rate": self.idsw_rate,
            "recall": self.n_matches / self.n_gt if self.n_gt else float("nan"),
        }


@dataclass
class PooledMot2d:
    videos: list[Mot2dResult] = field(default_factory=list)

    def merge(self) -> Mot2dResult:
        acc = Mot2dResult(video="pooled")
        for r in self.videos:
            acc.n_frames += r.n_frames
            acc.n_gt += r.n_gt
            acc.n_matches += r.n_matches
            acc.id_switches += r.id_switches
            acc.n_fp += r.n_fp
            acc.n_fn += r.n_fn
        return acc


def _read_bgr(path: Path) -> NDArray[np.uint8]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return np.asarray(img, dtype=np.uint8)


def _coco_to_bdd_group(coco: str) -> str:
    """Map detector COCO name to a BDD category group for GT matching."""
    if coco == "person":
        return "person"  # matches pedestrian + rider GT
    if coco in {"car", "bus", "truck", "train"}:
        return "vehicle"
    if coco in {"bicycle", "motorcycle"}:
        return "cycle"
    return coco


def _gt_group(bdd_cat: str) -> str:
    if bdd_cat in {"pedestrian", "rider"}:
        return "person"
    if bdd_cat in {"car", "bus", "truck", "train"}:
        return "vehicle"
    if bdd_cat in {"bicycle", "motorcycle"}:
        return "cycle"
    return bdd_cat


def _score_video(
    video: str,
    frames_dir: Path,
    labels_json: Path,
    det: Detector,
    class_names: tuple[str, ...],
    iou_thr: float,
) -> Mot2dResult:
    seq = load_mot_sequence(labels_json)
    tracker = ByteTracker(ByteTrackConfig())
    res = Mot2dResult(video=video)
    # gt_id → last matched track_id
    last_tid: dict[int, int] = {}
    period_ns = int(round(1e9 / FPS_MOT))

    for fr in seq:
        # Image name may be full or just index; try several layouts.
        candidates = [
            frames_dir / fr.name,
            frames_dir / Path(fr.name).name,
            frames_dir / f"{fr.frame_idx:07d}.jpg",
        ]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            continue
        img = _read_bgr(path)
        t_ns = fr.frame_idx * period_ns
        d = det.infer(img)
        boxes = np.asarray(d.boxes, dtype=np.float64)
        scores = np.asarray(d.scores, dtype=np.float64)
        classes = [
            class_names[int(c)] if int(c) < len(class_names) else str(int(c)) for c in d.classes
        ]
        tracks = [
            t
            for t in tracker.update(t_ns, boxes, scores, classes)
            if t.state is TrackState.CONFIRMED
        ]

        # Restrict GT to categories we can hit.
        keep = [
            (g, tid)
            for g, tid in zip(fr.boxes, fr.track_ids, strict=True)
            if g.kitti_type in DEFAULT_BDD_TO_COCO
        ]
        res.n_frames += 1
        res.n_gt += len(keep)
        if not keep:
            res.n_fp += len(tracks)
            continue
        if not tracks:
            res.n_fn += len(keep)
            continue

        gt_boxes = np.array([g.box for g, _ in keep], dtype=np.float64)
        pred_boxes = np.array([t.box for t in tracks], dtype=np.float64)
        iou = iou_xyxy(pred_boxes, gt_boxes)
        # Greedy match by IoU, same class group.
        order = np.argsort(-np.array([t.score for t in tracks], dtype=np.float64), kind="stable")
        gt_matched = np.zeros(len(keep), dtype=bool)
        for pi in order:
            tr = tracks[int(pi)]
            pg = _coco_to_bdd_group(tr.cls)
            row = iou[int(pi)].copy()
            for gj, (g, _) in enumerate(keep):
                if gt_matched[gj] or _gt_group(g.kitti_type) != pg:
                    row[gj] = -1.0
            j = int(np.argmax(row))
            if row[j] < iou_thr:
                res.n_fp += 1
                continue
            gt_matched[j] = True
            res.n_matches += 1
            gt_id = keep[j][1]
            if gt_id in last_tid and last_tid[gt_id] != tr.track_id:
                res.id_switches += 1
            last_tid[gt_id] = tr.track_id
        res.n_fn += int((~gt_matched).sum())
    return res


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--engine", type=Path)
    ap.add_argument("--labels-dir", type=Path, help="default: <root>/labels/box_track_20/val")
    ap.add_argument("--images-dir", type=Path, help="default: <root>/images/track/val")
    ap.add_argument("--max-videos", type=int, default=0, help="0 = all")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    labels_dir = args.labels_dir or (args.root / "labels" / "box_track_20" / "val")
    images_dir = args.images_dir or (args.root / "images" / "track" / "val")
    videos = list_mot_val_videos(labels_dir)
    if not videos:
        raise SystemExit(f"no MOT json under {labels_dir}")
    if args.max_videos:
        videos = videos[: args.max_videos]

    cfg = load_detector_config(args.config)
    engine_path = args.engine or cfg.engine
    engine = TrtEngine(engine_path)
    det = Detector(engine, cfg.runtime_config())
    pooled = PooledMot2d()
    try:
        for jp in videos:
            vid = jp.stem
            frames = images_dir / vid
            if not frames.is_dir():
                print(f"skip {vid}: missing frames dir {frames}")
                continue
            r = _score_video(vid, frames, jp, det, cfg.class_names, args.iou)
            pooled.videos.append(r)
            print(
                f"{vid}: frames={r.n_frames} gt={r.n_gt} matches={r.n_matches} "
                f"IDSW={r.id_switches} ({100 * r.idsw_rate:.1f} %) "
                f"FP={r.n_fp} FN={r.n_fn}"
            )
    finally:
        det.close()

    if not pooled.videos:
        raise SystemExit("no videos scored — check MOT images layout")
    m = pooled.merge()
    print(
        f"\npooled over {len(pooled.videos)} videos: matches={m.n_matches} "
        f"IDSW={m.id_switches} ({100 * m.idsw_rate:.2f} %) "
        f"recall={m.n_matches / m.n_gt if m.n_gt else float('nan'):.3f}"
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "dataset": "bdd100k_mot2020_val",
            "engine": str(engine_path),
            "iou": args.iou,
            "per_video": [r.to_dict() for r in pooled.videos],
            "pooled": m.to_dict(),
        }
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
