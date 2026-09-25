"""Typed view of the ``detector:`` section of ``configs/models.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from percepcion3d.detection.detector_trt import DetectorConfig


@dataclass(frozen=True)
class NmsSettings:
    score_threshold: float = 0.25
    iou_threshold: float = 0.6
    max_output_boxes: int = 100
    class_agnostic: bool = False


@dataclass(frozen=True)
class DetectorModelConfig:
    weights: str
    input_hw: tuple[int, int]
    onnx: Path
    engine: Path
    precision: str = "fp16"
    nms: NmsSettings = field(default_factory=NmsSettings)
    min_box_px: float = 2.0
    keep_class_ids: tuple[int, ...] = ()
    class_names: tuple[str, ...] = ()
    kitti_to_coco: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        h, w = self.input_hw
        if h <= 0 or w <= 0 or h % 32 or w % 32:
            raise ValueError(f"input_hw must be positive multiples of 32, got {self.input_hw}")
        if self.precision not in ("fp16", "int8", "fp32"):
            raise ValueError(f"unknown precision {self.precision!r}")

    def runtime_config(self) -> DetectorConfig:
        return DetectorConfig(
            score_threshold=self.nms.score_threshold,
            min_box_px=self.min_box_px,
            keep_classes=frozenset(self.keep_class_ids) if self.keep_class_ids else None,
            class_names=self.class_names,
        )


def load_detector_config(path: Path | str, root: Path | None = None) -> DetectorModelConfig:
    """Parse ``detector:`` from a models YAML; relative paths resolve against ``root``."""
    path = Path(path)
    root = root if root is not None else path.resolve().parents[1]
    with path.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    det = data.get("detector")
    if not isinstance(det, dict):
        raise ValueError(f"{path}: missing 'detector' section")
    for key in ("weights", "input_hw", "onnx", "engine"):
        if key not in det:
            raise ValueError(f"{path}: detector.{key} is required")
    nms_raw: dict[str, Any] = det.get("nms") or {}
    hw = det["input_hw"]
    if not (isinstance(hw, list) and len(hw) == 2):
        raise ValueError(f"{path}: detector.input_hw must be [H, W]")
    return DetectorModelConfig(
        weights=str(det["weights"]),
        input_hw=(int(hw[0]), int(hw[1])),
        onnx=_resolve(root, det["onnx"]),
        engine=_resolve(root, det["engine"]),
        precision=str(det.get("precision", "fp16")),
        nms=NmsSettings(
            score_threshold=float(nms_raw.get("score_threshold", 0.25)),
            iou_threshold=float(nms_raw.get("iou_threshold", 0.6)),
            max_output_boxes=int(nms_raw.get("max_output_boxes", 100)),
            class_agnostic=bool(nms_raw.get("class_agnostic", False)),
        ),
        min_box_px=float(det.get("min_box_px", 2.0)),
        keep_class_ids=tuple(int(c) for c in det.get("keep_class_ids") or ()),
        class_names=tuple(str(n) for n in det.get("class_names") or ()),
        kitti_to_coco={
            str(k): tuple(str(n) for n in v) for k, v in (det.get("kitti_to_coco") or {}).items()
        },
    )


def _resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p
