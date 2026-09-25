"""Typed view of the ``depth:`` section of ``configs/models.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from percepcion3d.depth.depth_trt import DepthConfig, DepthKind

VIT_PATCH = 14

VARIANT_KIND: dict[str, DepthKind] = {
    "relative": "relative_disparity",
    "metric_outdoor": "metric_depth",
    "metric_indoor": "metric_depth",
}


@dataclass(frozen=True)
class DepthModelConfig:
    weights: str
    input_sizes: tuple[tuple[int, int], ...]
    """Candidate ``(H, W)`` engine sizes, all multiples of 14; the first is the default."""
    onnx: str
    engine: str
    """Path templates with ``{h}``/``{w}`` placeholders, e.g. ``models/depth_{w}x{h}.onnx``."""
    variant: str = "relative"
    precision: str = "fp16"
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    store_fp16: bool = True
    output_name: str | None = None
    root: Path = field(default_factory=Path)

    def __post_init__(self) -> None:
        if not self.input_sizes:
            raise ValueError("depth.input_sizes must list at least one [H, W]")
        for h, w in self.input_sizes:
            if h <= 0 or w <= 0 or h % VIT_PATCH or w % VIT_PATCH:
                raise ValueError(
                    f"depth input sizes must be positive multiples of 14, got {(h, w)}"
                )
        if self.variant not in VARIANT_KIND:
            raise ValueError(
                f"unknown depth variant {self.variant!r}; one of {sorted(VARIANT_KIND)}"
            )
        if self.precision not in ("fp16", "int8", "fp32"):
            raise ValueError(f"unknown precision {self.precision!r}")
        if len(self.mean) != 3 or len(self.std) != 3 or any(s <= 0 for s in self.std):
            raise ValueError("mean/std must have 3 entries and std must be positive")

    @property
    def kind(self) -> DepthKind:
        return VARIANT_KIND[self.variant]

    @property
    def default_hw(self) -> tuple[int, int]:
        return self.input_sizes[0]

    def onnx_path(self, hw: tuple[int, int] | None = None) -> Path:
        return self._resolve(self.onnx, hw or self.default_hw)

    def engine_path(self, hw: tuple[int, int] | None = None) -> Path:
        return self._resolve(self.engine, hw or self.default_hw)

    def runtime_config(self) -> DepthConfig:
        return DepthConfig(kind=self.kind, output_name=self.output_name, store_fp16=self.store_fp16)

    def _resolve(self, template: str, hw: tuple[int, int]) -> Path:
        p = Path(template.format(h=hw[0], w=hw[1]))
        return p if p.is_absolute() else self.root / p


def load_depth_config(path: Path | str, root: Path | None = None) -> DepthModelConfig:
    """Parse ``depth:`` from a models YAML; relative paths resolve against ``root``."""
    path = Path(path)
    root = root if root is not None else path.resolve().parents[1]
    with path.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    dep = data.get("depth")
    if not isinstance(dep, dict):
        raise ValueError(f"{path}: missing 'depth' section")
    for key in ("weights", "input_sizes", "onnx", "engine"):
        if key not in dep:
            raise ValueError(f"{path}: depth.{key} is required")
    sizes_raw = dep["input_sizes"]
    if not (isinstance(sizes_raw, list) and sizes_raw):
        raise ValueError(f"{path}: depth.input_sizes must be a non-empty list of [H, W]")
    sizes: list[tuple[int, int]] = []
    for hw in sizes_raw:
        if not (isinstance(hw, list) and len(hw) == 2):
            raise ValueError(f"{path}: depth.input_sizes entries must be [H, W], got {hw!r}")
        sizes.append((int(hw[0]), int(hw[1])))
    norm: dict[str, Any] = dep.get("normalize") or {}
    mean = tuple(float(x) for x in norm.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(float(x) for x in norm.get("std", (0.229, 0.224, 0.225)))
    if len(mean) != 3 or len(std) != 3:
        raise ValueError(f"{path}: depth.normalize.mean/std must have 3 entries")
    output_name = dep.get("output_name")
    return DepthModelConfig(
        weights=str(dep["weights"]),
        input_sizes=tuple(sizes),
        onnx=str(dep["onnx"]),
        engine=str(dep["engine"]),
        variant=str(dep.get("variant", "relative")),
        precision=str(dep.get("precision", "fp16")),
        mean=(mean[0], mean[1], mean[2]),
        std=(std[0], std[1], std[2]),
        store_fp16=bool(dep.get("store_fp16", True)),
        output_name=str(output_name) if output_name is not None else None,
        root=root,
    )
