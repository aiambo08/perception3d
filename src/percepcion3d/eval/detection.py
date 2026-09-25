"""Detection recall/precision against KITTI-style boxes (export validation, F2 DoD).

This measures whether the *exported engine* still finds the objects — not the
model's quality — so the metric is deliberately simple: greedy one-to-one
matching by IoU (predictions sorted by score), per KITTI type, with a mapping
from the detector's class names (COCO) to KITTI types.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

# KITTI type → COCO class names that count as a hit for it.
DEFAULT_KITTI_TO_COCO: dict[str, tuple[str, ...]] = {
    "Car": ("car", "truck"),
    "Van": ("car", "truck", "bus"),
    "Truck": ("truck", "bus"),
    "Pedestrian": ("person",),
    "Cyclist": ("bicycle", "motorcycle", "person"),
}


def iou_xyxy(a: NDArray[np.floating[Any]], b: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """Pairwise IoU between ``a [N,4]`` and ``b [M,4]`` xyxy boxes → ``[N,M]``."""
    a2 = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b2 = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    lt = np.maximum(a2[:, None, :2], b2[None, :, :2])
    rb = np.minimum(a2[:, None, 2:], b2[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.prod(np.clip(a2[:, 2:] - a2[:, :2], 0.0, None), axis=1)
    area_b = np.prod(np.clip(b2[:, 2:] - b2[:, :2], 0.0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0.0, inter / union, 0.0)
    return np.asarray(iou, dtype=np.float64)


def match_greedy(
    pred_boxes: NDArray[np.floating[Any]],
    pred_scores: NDArray[np.floating[Any]],
    gt_boxes: NDArray[np.floating[Any]],
    iou_threshold: float = 0.5,
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Greedy one-to-one matching, highest-score prediction first.

    Returns ``(gt_index_per_pred, gt_matched)``; unmatched predictions get ``-1``.
    """
    p = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    g = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    assign = np.full(p.shape[0], -1, dtype=np.int64)
    matched = np.zeros(g.shape[0], dtype=bool)
    if p.shape[0] == 0 or g.shape[0] == 0:
        return assign, matched
    iou = iou_xyxy(p, g)
    order = np.argsort(-np.asarray(pred_scores, dtype=np.float64).reshape(-1), kind="stable")
    for i in order:
        row = np.where(matched, -1.0, iou[i])
        j = int(np.argmax(row))
        if row[j] >= iou_threshold:
            assign[i] = j
            matched[j] = True
    return assign, matched


@dataclass(frozen=True)
class GtBox:
    frame: int
    kitti_type: str
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class PredBox:
    frame: int
    class_name: str
    score: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class ClassRecall:
    kitti_type: str
    n_gt: int
    n_matched: int
    n_pred: int

    @property
    def recall(self) -> float:
        return self.n_matched / self.n_gt if self.n_gt else float("nan")

    @property
    def precision(self) -> float:
        return self.n_matched / self.n_pred if self.n_pred else float("nan")


def recall_by_type(
    preds: Iterable[PredBox],
    gts: Iterable[GtBox],
    iou_threshold: float = 0.5,
    type_to_classes: Mapping[str, Sequence[str]] = DEFAULT_KITTI_TO_COCO,
    min_gt_height_px: float = 25.0,
) -> dict[str, ClassRecall]:
    """Per-KITTI-type recall at ``iou_threshold`` over all frames.

    GT boxes shorter than ``min_gt_height_px`` are ignored (KITTI "moderate"
    difficulty uses 25 px), as are GT types absent from ``type_to_classes``.
    Predictions whose class maps to several KITTI types are matched against
    each of them independently, so precision is only indicative.
    """
    gt_by_frame: dict[int, list[GtBox]] = {}
    for g in gts:
        if g.kitti_type in type_to_classes and (g.box[3] - g.box[1]) >= min_gt_height_px:
            gt_by_frame.setdefault(g.frame, []).append(g)
    pred_by_frame: dict[int, list[PredBox]] = {}
    for p in preds:
        pred_by_frame.setdefault(p.frame, []).append(p)

    out: dict[str, ClassRecall] = {}
    for kitti_type, coco_names in type_to_classes.items():
        names = set(coco_names)
        n_gt = n_matched = n_pred = 0
        for frame in sorted(set(gt_by_frame) | set(pred_by_frame)):
            g_list = [g for g in gt_by_frame.get(frame, []) if g.kitti_type == kitti_type]
            p_list = [p for p in pred_by_frame.get(frame, []) if p.class_name in names]
            n_gt += len(g_list)
            n_pred += len(p_list)
            if not g_list or not p_list:
                continue
            _, matched = match_greedy(
                np.array([p.box for p in p_list]),
                np.array([p.score for p in p_list]),
                np.array([g.box for g in g_list]),
                iou_threshold,
            )
            n_matched += int(matched.sum())
        out[kitti_type] = ClassRecall(kitti_type, n_gt, n_matched, n_pred)
    return out


def format_recall(rows: Mapping[str, ClassRecall]) -> str:
    lines = [f"{'type':<12}{'n_gt':>7}{'matched':>9}{'n_pred':>8}{'recall':>8}{'prec':>7}"]
    for r in rows.values():
        lines.append(
            f"{r.kitti_type:<12}{r.n_gt:>7d}{r.n_matched:>9d}{r.n_pred:>8d}"
            f"{r.recall:>8.3f}{r.precision:>7.3f}"
        )
    return "\n".join(lines)
