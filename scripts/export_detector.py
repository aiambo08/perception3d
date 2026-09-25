"""Export a YOLO(n|s) checkpoint to an ONNX graph ready for TensorRT.

Pipeline (``export`` extra: ultralytics, onnx, onnxslim)::

    YOLO .pt ──ultralytics──► raw ONNX  [1,3,H,W] f32 → [1,84,N]
             ──surgery─────► + uint8 NHWC preprocess in-graph
                             + EfficientNMS_TRT (num_dets, det_boxes, det_scores, det_classes)

Then build the engine on the target GPU::

    uv run python scripts/export_detector.py --config configs/models.yaml
    bash scripts/export_trt.sh models/detector_1024x320.onnx models/detector_1024x320_fp16.engine

Nothing here needs a GPU; ultralytics exports on CPU. ``--raw-onnx`` skips
ultralytics and applies the surgery to an existing raw export.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import onnx  # noqa: E402

from percepcion3d.detection.config import load_detector_config  # noqa: E402
from percepcion3d.detection.onnx_surgery import (  # noqa: E402
    NmsConfig,
    append_efficient_nms,
    head_layout,
    input_hw,
    prepend_uint8_preprocess,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"


def export_raw_onnx(weights: str, hw: tuple[int, int], out: Path, opset: int) -> Path:
    from ultralytics import YOLO

    model = YOLO(weights)
    # imgsz=[H, W]; dynamic=False → static shapes required by the surgery/engine.
    produced = model.export(
        format="onnx", imgsz=list(hw), opset=opset, simplify=True, dynamic=False, half=False
    )
    src = Path(str(produced))
    out.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != out.resolve():
        src.replace(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument(
        "--raw-onnx", type=Path, help="skip ultralytics; use this raw [1,3,H,W]->[1,4+C,N] export"
    )
    ap.add_argument("--out", type=Path, help="override detector.onnx from the config")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--keep-bgr", action="store_true", help="do not swap BGR→RGB in-graph")
    args = ap.parse_args()

    cfg = load_detector_config(args.config)
    out = args.out or cfg.onnx
    raw_path = args.raw_onnx or out.with_name(out.stem + "_raw.onnx")
    if args.raw_onnx is None:
        export_raw_onnx(cfg.weights, cfg.input_hw, raw_path, args.opset)

    raw = onnx.load(str(raw_path))
    b, ch, n = head_layout(raw)
    hw = input_hw(raw)
    if hw != cfg.input_hw:
        raise SystemExit(f"raw export input {hw} != config input_hw {cfg.input_hw}")
    print(f"raw head: batch={b} channels={ch} (4+{ch - 4} classes) anchors={n} input={hw}")

    model = prepend_uint8_preprocess(raw, bgr_to_rgb=not args.keep_bgr)
    onnx.checker.check_model(model)  # the NMS plugin op is unknown to the ONNX checker
    model = append_efficient_nms(
        model,
        NmsConfig(
            score_threshold=cfg.nms.score_threshold,
            iou_threshold=cfg.nms.iou_threshold,
            max_output_boxes=cfg.nms.max_output_boxes,
            class_agnostic=cfg.nms.class_agnostic,
        ),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(
        "inputs :",
        [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in model.graph.input],
    )
    print(
        "outputs:",
        [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in model.graph.output],
    )


if __name__ == "__main__":
    main()
