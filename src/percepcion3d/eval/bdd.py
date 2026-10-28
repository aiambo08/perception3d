"""BDD100K Detection 2020 / MOT 2020 loaders (Scalabel JSON).

Detection layout (after ``scripts/download_bdd100k.sh eval``)::

    <root>/
      images/100k/val/*.jpg
      labels/det_20/det_val.json

MOT 2020 (optional tracking eval)::

    <root>/
      images/track/val/<video>/*.jpg
      labels/box_track_20/val/<video>.json

BDD categories → COCO names that count as a hit for export validation (F2).
``rider`` maps to ``person`` (YOLO has no rider class). ``traffic sign`` is
omitted from the default map: COCO only has ``stop sign``, not a general sign.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from percepcion3d.eval.detection import GtBox

# BDD category → COCO class names that count as a hit.
DEFAULT_BDD_TO_COCO: dict[str, tuple[str, ...]] = {
    "car": ("car",),
    "bus": ("bus",),
    "truck": ("truck",),
    "train": ("train",),
    "bicycle": ("bicycle",),
    "motorcycle": ("motorcycle",),
    "pedestrian": ("person",),
    "rider": ("person",),
}

# Optional extras (COCO has traffic light; traffic sign ≈ stop sign only).
BDD_TO_COCO_WITH_SIGNS: dict[str, tuple[str, ...]] = {
    **DEFAULT_BDD_TO_COCO,
    "traffic light": ("traffic light",),
    "traffic sign": ("stop sign",),
}


@dataclass(frozen=True)
class BddFrameMeta:
    """Per-image attributes from Detection 2020 (weather / scene / time of day)."""

    name: str
    weather: str = ""
    scene: str = ""
    timeofday: str = ""


@dataclass(frozen=True)
class BddMotFrame:
    """One frame of a MOT 2020 sequence (5 Hz resampled clips)."""

    frame_idx: int
    name: str
    boxes: tuple[GtBox, ...]
    track_ids: tuple[int, ...]


def load_det_labels(
    path: Path | str,
    *,
    keep_categories: Sequence[str] | None = None,
) -> dict[str, list[GtBox]]:
    """Load Detection 2020 Scalabel JSON → ``image_name → [GtBox]``.

    ``GtBox.kitti_type`` holds the BDD category string. ``frame`` is always 0
    (one keyframe per clip); callers that pool many images should re-index.
    Boxes without ``box2d`` (rare) are skipped. ``truncated`` / ``occluded``
    stay at defaults (BDD det has no KITTI-style difficulty fields).
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON list of frames, got {type(raw).__name__}")
    keep = set(keep_categories) if keep_categories is not None else None
    out: dict[str, list[GtBox]] = {}
    for item in raw:
        name = str(item.get("name") or "")
        if not name:
            continue
        boxes: list[GtBox] = []
        for lab in item.get("labels") or ():
            cat = str(lab.get("category") or "")
            if keep is not None and cat not in keep:
                continue
            box2d = lab.get("box2d")
            if not isinstance(box2d, dict):
                continue
            try:
                x1, y1 = float(box2d["x1"]), float(box2d["y1"])
                x2, y2 = float(box2d["x2"]), float(box2d["y2"])
            except (KeyError, TypeError, ValueError):
                continue
            boxes.append(GtBox(0, cat, (x1, y1, x2, y2)))
        out[name] = boxes
    return out


def load_det_frame_meta(path: Path | str) -> dict[str, BddFrameMeta]:
    """Image name → weather / scene / timeofday (for picking demo clips)."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, BddFrameMeta] = {}
    for item in raw:
        name = str(item.get("name") or "")
        if not name:
            continue
        attrs = item.get("attributes") or {}
        out[name] = BddFrameMeta(
            name=name,
            weather=str(attrs.get("weather") or ""),
            scene=str(attrs.get("scene") or ""),
            timeofday=str(attrs.get("timeofday") or ""),
        )
    return out


def resolve_image_path(images_dir: Path | str, name: str) -> Path | None:
    """Locate ``name`` under ``images_dir`` (exact path or recursive by stem)."""
    images_dir = Path(images_dir)
    direct = images_dir / name
    if direct.is_file():
        return direct
    # Some layouts nest train/val; accept a stem search one level deep.
    stem = Path(name).stem
    for cand in images_dir.rglob(f"{stem}.*"):
        if cand.suffix.lower() in {".jpg", ".jpeg", ".png"} and cand.is_file():
            return cand
    return None


def load_mot_sequence(labels_json: Path | str) -> list[BddMotFrame]:
    """Load one MOT 2020 Scalabel video JSON (sorted by ``frameIndex``)."""
    path = Path(labels_json)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON list of frames")
    frames: list[BddMotFrame] = []
    for item in raw:
        idx = int(item.get("frameIndex", len(frames)))
        name = str(item.get("name") or f"{idx:07d}.jpg")
        boxes: list[GtBox] = []
        tids: list[int] = []
        for lab in item.get("labels") or ():
            cat = str(lab.get("category") or "")
            box2d = lab.get("box2d")
            if not isinstance(box2d, dict):
                continue
            try:
                x1, y1 = float(box2d["x1"]), float(box2d["y1"])
                x2, y2 = float(box2d["x2"]), float(box2d["y2"])
                tid = int(str(lab.get("id")))
            except (KeyError, TypeError, ValueError):
                continue
            boxes.append(GtBox(idx, cat, (x1, y1, x2, y2)))
            tids.append(tid)
        frames.append(BddMotFrame(idx, name, tuple(boxes), tuple(tids)))
    frames.sort(key=lambda f: f.frame_idx)
    return frames


def list_mot_val_videos(labels_dir: Path | str) -> list[Path]:
    """Sorted list of ``*.json`` under MOT ``box_track_20/val``."""
    d = Path(labels_dir)
    return sorted(d.glob("*.json"))


def pick_demo_candidates(
    meta: Mapping[str, BddFrameMeta],
    *,
    night: bool = True,
    rainy: bool = True,
    highway: bool = True,
    limit_per_bucket: int = 5,
) -> dict[str, list[str]]:
    """Suggest image stems for README demo clips (night / rain / highway)."""
    buckets: dict[str, list[str]] = {"night": [], "rain": [], "highway": []}
    for name, m in meta.items():
        stem = Path(name).stem
        tod = m.timeofday.lower()
        weather = m.weather.lower()
        scene = m.scene.lower()
        if night and "night" in tod and len(buckets["night"]) < limit_per_bucket:
            buckets["night"].append(stem)
        if rainy and "rain" in weather and len(buckets["rain"]) < limit_per_bucket:
            buckets["rain"].append(stem)
        if highway and "highway" in scene and len(buckets["highway"]) < limit_per_bucket:
            buckets["highway"].append(stem)
        if all(len(v) >= limit_per_bucket for v in buckets.values()):
            break
    return buckets


def flatten_det_gts(
    by_name: Mapping[str, Sequence[GtBox]],
    names: Iterable[str],
) -> tuple[list[GtBox], dict[str, int]]:
    """Re-index GT boxes so each image gets a unique ``frame`` index.

    Returns ``(gts, name_to_frame)``.
    """
    gts: list[GtBox] = []
    name_to_frame: dict[str, int] = {}
    for i, name in enumerate(names):
        name_to_frame[name] = i
        for g in by_name.get(name, ()):
            gts.append(GtBox(i, g.kitti_type, g.box, g.truncated, g.occluded))
    return gts, name_to_frame


def summary_counts(by_name: Mapping[str, Sequence[GtBox]]) -> dict[str, Any]:
    """Quick sanity stats for a det split."""
    n_img = len(by_name)
    n_box = sum(len(v) for v in by_name.values())
    per_cat: dict[str, int] = {}
    for boxes in by_name.values():
        for g in boxes:
            per_cat[g.kitti_type] = per_cat.get(g.kitti_type, 0) + 1
    return {"n_images": n_img, "n_boxes": n_box, "per_category": dict(sorted(per_cat.items()))}
