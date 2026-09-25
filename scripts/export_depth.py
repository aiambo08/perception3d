"""Export Depth Anything V2-Small to ONNX graphs ready for TensorRT (one per input size).

Pipeline (``export`` extra: torch, transformers, onnx)::

    HF checkpoint ──torch.onnx──► raw ONNX  pixel_values [1,3,H,W] f32 → depth [1,H,W]
                  ──surgery────► images_u8 [1,H,W,3] uint8 (BGR) + RGB swap + /255
                                 + ImageNet (x−mean)/std folded into Mul+Add

Then build the engines on the target GPU::

    uv run python scripts/export_depth.py --config configs/models.yaml
    for s in 924x280 840x252 1064x322; do
        bash scripts/export_trt.sh models/depth_$s.onnx models/depth_${s}_fp16.engine
    done

``--raw-onnx`` skips torch and applies the surgery to an existing raw export
(its input size must match ``--size``). The script prints the raw output
tensor shape: for the relative checkpoints it is ``[1,H,W]`` inverse depth,
for the metric ones ``[1,H,W]`` metres — ``DepthConfig.kind`` in the runtime
follows ``depth.variant`` from the config, so keep them consistent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import onnx  # noqa: E402

from percepcion3d.depth.config import DepthModelConfig, load_depth_config  # noqa: E402
from percepcion3d.detection.onnx_surgery import (  # noqa: E402
    input_hw,
    output_dims,
    prepend_uint8_preprocess,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"


def export_raw_onnx(weights: str, hw: tuple[int, int], out: Path, opset: int) -> Path:
    import torch
    from transformers import DepthAnythingForDepthEstimation

    class _Wrapper(torch.nn.Module):  # type: ignore[misc]  # torch is untyped here
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return self.inner(pixel_values=pixel_values).predicted_depth

    model = DepthAnythingForDepthEstimation.from_pretrained(weights).eval()
    wrapper = _Wrapper(model).eval()
    dummy = torch.zeros(1, 3, hw[0], hw[1], dtype=torch.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, object] = dict(
        input_names=["pixel_values"],
        output_names=["depth"],
        opset_version=opset,
        dynamic_axes=None,
        do_constant_folding=True,
    )
    with torch.no_grad():
        try:
            # torch >= 2.5: keep the TorchScript exporter (static shapes, TensorRT-proven path)
            torch.onnx.export(wrapper, (dummy,), str(out), dynamo=False, **kwargs)
        except TypeError:
            torch.onnx.export(wrapper, (dummy,), str(out), **kwargs)
    return out


def parse_size(text: str) -> tuple[int, int]:
    """``WxH`` (as in the file names) → ``(H, W)``."""
    w, h = (int(t) for t in text.lower().split("x"))
    return h, w


def export_one(cfg: DepthModelConfig, hw: tuple[int, int], args: argparse.Namespace) -> Path:
    out = cfg.onnx_path(hw)
    raw_override: Path | None = args.raw_onnx
    raw_path = raw_override if raw_override is not None else out.with_name(out.stem + "_raw.onnx")
    if raw_override is None:
        export_raw_onnx(cfg.weights, hw, raw_path, args.opset)

    raw = onnx.load(str(raw_path))
    raw_hw = input_hw(raw)
    if raw_hw != hw:
        raise SystemExit(f"raw export input {raw_hw} != requested size {hw}")
    name, dims = output_dims(raw)
    print(f"[{hw[1]}x{hw[0]}] raw output {name!r} dims={dims} (variant={cfg.variant})")
    if dims[-2:] != list(hw):
        raise SystemExit(f"raw output spatial dims {dims[-2:]} != input {list(hw)}")

    model = prepend_uint8_preprocess(raw, bgr_to_rgb=not args.keep_bgr, mean=cfg.mean, std=cfg.std)
    onnx.checker.check_model(model)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(
        "  inputs :",
        [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in model.graph.input],
    )
    print(
        "  outputs:",
        [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in model.graph.output],
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument(
        "--size",
        type=parse_size,
        help="single WxH to export (default: every entry of depth.input_sizes)",
    )
    ap.add_argument("--raw-onnx", type=Path, help="skip torch; use this raw [1,3,H,W] export")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--keep-bgr", action="store_true", help="do not swap BGR→RGB in-graph")
    args = ap.parse_args()

    cfg = load_depth_config(args.config)
    sizes = [args.size] if args.size else list(cfg.input_sizes)
    if args.raw_onnx is not None and len(sizes) != 1:
        raise SystemExit("--raw-onnx requires --size")
    for hw in sizes:
        export_one(cfg, hw, args)


if __name__ == "__main__":
    main()
