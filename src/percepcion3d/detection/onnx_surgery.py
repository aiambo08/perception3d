"""ONNX graph surgery for the detector/depth exports (requires the ``export`` extra).

Two independent transformations on a single-input/single-output ONNX graph:

1. :func:`prepend_uint8_preprocess` — replace the ``float32 [1,3,H,W]`` input
   by ``uint8 [1,H,W,3]`` (BGR, as delivered by OpenCV / V4L2) and put
   ``Cast → Transpose(NHWC→NCHW) → channel reverse (BGR→RGB) → /255`` inside
   the graph, optionally followed by per-channel ``(x - mean) / std`` (folded
   into one ``Mul`` + one ``Add``, as needed by ImageNet-normalised backbones
   such as Depth Anything). The host uploads 1 byte/pixel and does no
   ``astype``/``transpose``.

2. :func:`append_efficient_nms` — replace the raw head output
   ``output0 [1, 4+C, N]`` (cx, cy, w, h, class scores) by the TensorRT
   ``EfficientNMS_TRT`` plugin, so the engine returns
   ``num_dets [1,1] int32``, ``det_boxes [1,K,4]`` (xyxy, canvas px),
   ``det_scores [1,K]`` and ``det_classes [1,K] int32``.

Both are pure ``onnx`` protobuf edits; nothing here imports TensorRT. The
resulting graph runs in ``onnxruntime`` only up to the NMS node (the plugin is
TensorRT-only), which is what the tests exercise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

U8_INPUT_NAME = "images_u8"
NMS_OUTPUT_NAMES: tuple[str, str, str, str] = (
    "num_dets",
    "det_boxes",
    "det_scores",
    "det_classes",
)


@dataclass(frozen=True)
class NmsConfig:
    """``EfficientNMS_TRT`` attributes (see TensorRT plugin docs)."""

    score_threshold: float = 0.25
    iou_threshold: float = 0.6
    max_output_boxes: int = 100
    class_agnostic: bool = False


def _single_io(model: onnx.ModelProto) -> tuple[onnx.ValueInfoProto, onnx.ValueInfoProto]:
    g = model.graph
    initializer_names = {t.name for t in g.initializer}
    inputs = [i for i in g.input if i.name not in initializer_names]
    if len(inputs) != 1 or len(g.output) != 1:
        raise ValueError(
            f"expected a single-input/single-output graph, got {len(inputs)} inputs "
            f"and {len(g.output)} outputs"
        )
    return inputs[0], g.output[0]


def _static_dims(vi: onnx.ValueInfoProto) -> list[int]:
    dims = vi.type.tensor_type.shape.dim
    out: list[int] = []
    for d in dims:
        if not d.HasField("dim_value"):
            raise ValueError(
                f"tensor '{vi.name}' has a dynamic dimension; export with static shapes"
            )
        out.append(int(d.dim_value))
    return out


def _default_opset(model: onnx.ModelProto) -> int:
    for op in model.opset_import:
        if op.domain in ("", "ai.onnx"):
            return int(op.version)
    return 0


IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def prepend_uint8_preprocess(
    model: onnx.ModelProto,
    input_name: str = U8_INPUT_NAME,
    bgr_to_rgb: bool = True,
    scale: float = 1.0 / 255.0,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> onnx.ModelProto:
    """Return a copy of ``model`` whose input is ``uint8 [N,H,W,3]`` (NHWC).

    With ``mean``/``std`` (RGB order, in ``[0,1]`` units) the graph computes
    ``(x·scale − mean) / std`` as ``x·(scale/std) + (−mean/std)`` per channel.
    """
    if (mean is None) != (std is None):
        raise ValueError("mean and std must be given together")
    model = onnx.ModelProto.FromString(model.SerializeToString())
    g = model.graph
    old_in, _ = _single_io(model)
    n, c, h, w = _static_dims(old_in)
    if c != 3:
        raise ValueError(f"expected 3 input channels, got {c}")
    if old_in.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise ValueError("expected a float32 model input")

    new_in = helper.make_tensor_value_info(input_name, TensorProto.UINT8, [n, h, w, c])
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Cast", [input_name], ["pre/f32"], to=TensorProto.FLOAT, name="pre/cast"),
        helper.make_node(
            "Transpose", ["pre/f32"], ["pre/nchw"], perm=[0, 3, 1, 2], name="pre/nchw"
        ),
    ]
    last = "pre/nchw"
    if bgr_to_rgb:
        g.initializer.append(
            numpy_helper.from_array(np.array([2, 1, 0], dtype=np.int64), "pre/rgb_idx")
        )
        nodes.append(
            helper.make_node("Gather", [last, "pre/rgb_idx"], ["pre/rgb"], axis=1, name="pre/rgb")
        )
        last = "pre/rgb"
    if mean is None or std is None:
        g.initializer.append(
            numpy_helper.from_array(np.array(scale, dtype=np.float32), "pre/scale")
        )
        nodes.append(
            helper.make_node("Mul", [last, "pre/scale"], [old_in.name], name="pre/scale_mul")
        )
    else:
        std_arr = np.asarray(std, dtype=np.float64)
        if std_arr.shape != (3,) or np.any(std_arr <= 0):
            raise ValueError(f"std must be 3 positive values, got {std}")
        mul = (scale / std_arr).astype(np.float32).reshape(1, 3, 1, 1)
        add = (-np.asarray(mean, dtype=np.float64) / std_arr).astype(np.float32).reshape(1, 3, 1, 1)
        g.initializer.append(numpy_helper.from_array(mul, "pre/scale"))
        g.initializer.append(numpy_helper.from_array(add, "pre/bias"))
        nodes.append(
            helper.make_node("Mul", [last, "pre/scale"], ["pre/scaled"], name="pre/scale_mul")
        )
        nodes.append(
            helper.make_node("Add", ["pre/scaled", "pre/bias"], [old_in.name], name="pre/bias_add")
        )

    g.input.remove(old_in)
    g.input.insert(0, new_in)
    # Keep the old input tensor documented as an intermediate value.
    g.value_info.append(old_in)
    for node in reversed(nodes):
        g.node.insert(0, node)
    return model


def append_efficient_nms(
    model: onnx.ModelProto,
    cfg: NmsConfig | None = None,
    output_names: tuple[str, str, str, str] = NMS_OUTPUT_NAMES,
) -> onnx.ModelProto:
    """Return a copy of ``model`` with ``EfficientNMS_TRT`` consuming the YOLO head.

    Expects the head output ``[B, 4+C, N]`` with ``(cx, cy, w, h)`` first
    (Ultralytics YOLOv8/YOLO11 layout, no objectness). ``box_coding=1`` tells
    the plugin the boxes are centre-size, so no Slice/Sub/Add is needed.
    """
    cfg = cfg if cfg is not None else NmsConfig()
    model = onnx.ModelProto.FromString(model.SerializeToString())
    g = model.graph
    _, old_out = _single_io(model)
    b, ch, n = _static_dims(old_out)
    num_classes = ch - 4
    if num_classes <= 0:
        raise ValueError(f"head output has {ch} channels; expected 4 + num_classes")

    k = cfg.max_output_boxes
    if _default_opset(model) >= 13:
        g.initializer.append(
            numpy_helper.from_array(np.array([4, num_classes], dtype=np.int64), "nms/split")
        )
        split_node = helper.make_node(
            "Split",
            ["nms/pred_bnc", "nms/split"],
            ["nms/boxes_cxcywh", "nms/scores"],
            axis=2,
            name="nms/split_node",
        )
    else:
        split_node = helper.make_node(
            "Split",
            ["nms/pred_bnc"],
            ["nms/boxes_cxcywh", "nms/scores"],
            axis=2,
            split=[4, num_classes],
            name="nms/split_node",
        )
    nodes = [
        helper.make_node(
            "Transpose", [old_out.name], ["nms/pred_bnc"], perm=[0, 2, 1], name="nms/transpose"
        ),
        split_node,
        helper.make_node(
            "EfficientNMS_TRT",
            ["nms/boxes_cxcywh", "nms/scores"],
            list(output_names),
            name="nms/efficient_nms",
            domain="",
            background_class=-1,
            box_coding=1,
            iou_threshold=float(cfg.iou_threshold),
            max_output_boxes=int(k),
            plugin_version="1",
            score_activation=0,
            score_threshold=float(cfg.score_threshold),
            class_agnostic=int(cfg.class_agnostic),
        ),
    ]
    g.node.extend(nodes)

    g.output.remove(old_out)
    g.value_info.extend(
        [
            old_out,
            helper.make_tensor_value_info("nms/pred_bnc", TensorProto.FLOAT, [b, n, ch]),
            helper.make_tensor_value_info("nms/boxes_cxcywh", TensorProto.FLOAT, [b, n, 4]),
            helper.make_tensor_value_info("nms/scores", TensorProto.FLOAT, [b, n, num_classes]),
        ]
    )
    g.output.extend(
        [
            helper.make_tensor_value_info(output_names[0], TensorProto.INT32, [b, 1]),
            helper.make_tensor_value_info(output_names[1], TensorProto.FLOAT, [b, k, 4]),
            helper.make_tensor_value_info(output_names[2], TensorProto.FLOAT, [b, k]),
            helper.make_tensor_value_info(output_names[3], TensorProto.INT32, [b, k]),
        ]
    )
    return model


def head_layout(model: onnx.ModelProto) -> tuple[int, int, int]:
    """Return ``(batch, 4 + num_classes, num_anchors)`` of a raw YOLO head output."""
    _, out = _single_io(model)
    b, ch, n = _static_dims(out)
    return b, ch, n


def output_dims(model: onnx.ModelProto) -> tuple[str, list[int]]:
    """``(name, static dims)`` of the single graph output."""
    _, out = _single_io(model)
    return out.name, _static_dims(out)


def input_hw(model: onnx.ModelProto) -> tuple[int, int]:
    """Spatial size of the (single) model input, for either NCHW float or NHWC uint8."""
    inp, _ = _single_io(model)
    dims = _static_dims(inp)
    if inp.type.tensor_type.elem_type == TensorProto.UINT8:
        return dims[1], dims[2]
    return dims[2], dims[3]
