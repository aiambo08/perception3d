"""ONNX surgery tests on tiny synthetic YOLO-shaped graphs, executed with onnxruntime."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import onnx  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from percepcion3d.detection.onnx_surgery import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    NMS_OUTPUT_NAMES,
    U8_INPUT_NAME,
    NmsConfig,
    append_efficient_nms,
    head_layout,
    input_hw,
    output_dims,
    prepend_uint8_preprocess,
)

H, W = 4, 8
N_CLASSES = 4  # head channels = 4 + 4 = 8, anchors = 3*4*8/8 = 12
IR = 9  # onnxruntime accepts IR <= 10; ultralytics exports IR 8-9


def _identity_model() -> onnx.ModelProto:
    """float [1,3,H,W] → identical output (lets us read back the preprocessed tensor)."""
    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, H, W])
    y = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 3, H, W])
    node = helper.make_node("Identity", ["images"], ["output0"])
    graph = helper.make_graph([node], "ident", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=IR)


def _yolo_like_model(opset: int = 17) -> onnx.ModelProto:
    """float [1,3,H,W] → Reshape → head [1, 4+C, N] (YOLOv8/11 raw layout)."""
    n_anchors = 3 * H * W // (4 + N_CLASSES)
    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, H, W])
    y = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 4 + N_CLASSES, n_anchors])
    shape = numpy_helper.from_array(
        np.array([1, 4 + N_CLASSES, n_anchors], dtype=np.int64), "shape"
    )
    node = helper.make_node("Reshape", ["images", "shape"], ["output0"])
    graph = helper.make_graph([node], "yolo_like", [x], [y], initializer=[shape])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=IR)


def _run(model: onnx.ModelProto, feeds: dict[str, NDArray[Any]]) -> list[NDArray[Any]]:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return [np.asarray(o) for o in sess.run(None, feeds)]


def test_preprocess_matches_numpy_reference() -> None:
    model = prepend_uint8_preprocess(_identity_model())
    onnx.checker.check_model(model, full_check=True)

    assert [i.name for i in model.graph.input] == [U8_INPUT_NAME]
    inp = model.graph.input[0]
    assert inp.type.tensor_type.elem_type == TensorProto.UINT8
    assert [d.dim_value for d in inp.type.tensor_type.shape.dim] == [1, H, W, 3]
    assert input_hw(model) == (H, W)

    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 256, size=(1, H, W, 3), dtype=np.uint8)
    (out,) = _run(model, {U8_INPUT_NAME: bgr})
    ref = bgr[..., ::-1].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    assert out.shape == (1, 3, H, W) and out.dtype == np.float32
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-7)


def test_preprocess_keep_bgr_and_custom_scale() -> None:
    model = prepend_uint8_preprocess(_identity_model(), bgr_to_rgb=False, scale=1.0)
    bgr = np.arange(3 * H * W, dtype=np.uint8).reshape(1, H, W, 3)
    (out,) = _run(model, {U8_INPUT_NAME: bgr})
    np.testing.assert_array_equal(out, bgr.transpose(0, 3, 1, 2).astype(np.float32))
    assert not any(n.op_type == "Gather" for n in model.graph.node)


def test_preprocess_does_not_mutate_input_model() -> None:
    raw = _identity_model()
    before = raw.SerializeToString()
    prepend_uint8_preprocess(raw)
    assert raw.SerializeToString() == before


def test_preprocess_rejects_dynamic_or_non_float_inputs() -> None:
    dyn = _identity_model()
    dyn.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
    with pytest.raises(ValueError, match="dynamic"):
        prepend_uint8_preprocess(dyn)
    u8 = _identity_model()
    u8.graph.input[0].type.tensor_type.elem_type = TensorProto.UINT8
    with pytest.raises(ValueError, match="float32"):
        prepend_uint8_preprocess(u8)
    gray = _identity_model()
    gray.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 1
    with pytest.raises(ValueError, match="3 input channels"):
        prepend_uint8_preprocess(gray)


def test_preprocess_with_imagenet_mean_std_matches_torchvision_style_reference() -> None:
    """Depth Anything path: (x/255 - mean) / std per RGB channel, folded into Mul+Add."""
    model = prepend_uint8_preprocess(_identity_model(), mean=IMAGENET_MEAN, std=IMAGENET_STD)
    onnx.checker.check_model(model, full_check=True)
    assert input_hw(model) == (H, W)
    assert output_dims(model) == ("output0", [1, 3, H, W])

    rng = np.random.default_rng(3)
    bgr = rng.integers(0, 256, size=(1, H, W, 3), dtype=np.uint8)
    (out,) = _run(model, {U8_INPUT_NAME: bgr})
    rgb = bgr[..., ::-1].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(IMAGENET_STD, np.float32).reshape(1, 3, 1, 1)
    np.testing.assert_allclose(out, (rgb - mean) / std, rtol=1e-5, atol=1e-5)
    # channel order matters: swapping the constants must change the result
    swapped = prepend_uint8_preprocess(
        _identity_model(), mean=IMAGENET_MEAN[::-1], std=IMAGENET_STD[::-1]
    )
    (out_sw,) = _run(swapped, {U8_INPUT_NAME: bgr})
    assert not np.allclose(out, out_sw)


def test_preprocess_mean_std_validation() -> None:
    with pytest.raises(ValueError, match="together"):
        prepend_uint8_preprocess(_identity_model(), mean=IMAGENET_MEAN)
    with pytest.raises(ValueError, match="std"):
        prepend_uint8_preprocess(_identity_model(), mean=IMAGENET_MEAN, std=(0.0, 1.0, 1.0))


@pytest.mark.parametrize("opset", [11, 17])
def test_append_efficient_nms_graph_structure(opset: int) -> None:
    raw = _yolo_like_model(opset)
    b, ch, n = head_layout(raw)
    assert (b, ch, n) == (1, 4 + N_CLASSES, 12)

    cfg = NmsConfig(score_threshold=0.3, iou_threshold=0.5, max_output_boxes=7, class_agnostic=True)
    model = append_efficient_nms(raw, cfg)

    outs = {o.name: o for o in model.graph.output}
    assert tuple(outs) == NMS_OUTPUT_NAMES
    dims = {k: [d.dim_value for d in v.type.tensor_type.shape.dim] for k, v in outs.items()}
    assert dims == {
        "num_dets": [1, 1],
        "det_boxes": [1, 7, 4],
        "det_scores": [1, 7],
        "det_classes": [1, 7],
    }
    assert outs["num_dets"].type.tensor_type.elem_type == TensorProto.INT32
    assert outs["det_classes"].type.tensor_type.elem_type == TensorProto.INT32
    assert outs["det_boxes"].type.tensor_type.elem_type == TensorProto.FLOAT

    nms = [nd for nd in model.graph.node if nd.op_type == "EfficientNMS_TRT"]
    assert len(nms) == 1
    attrs = {a.name: helper.get_attribute_value(a) for a in nms[0].attribute}
    assert attrs["box_coding"] == 1  # cx,cy,w,h straight from the YOLO head
    assert attrs["background_class"] == -1
    assert attrs["max_output_boxes"] == 7
    assert attrs["class_agnostic"] == 1
    assert attrs["score_threshold"] == pytest.approx(0.3)
    assert attrs["iou_threshold"] == pytest.approx(0.5)
    assert attrs["plugin_version"] == b"1"
    assert list(nms[0].input) == ["nms/boxes_cxcywh", "nms/scores"]

    # Everything before the TensorRT-only plugin must run in onnxruntime and
    # deliver boxes = first 4 channels, scores = remaining C channels, both [B,N,·].
    pre = onnx.utils.Extractor(model).extract_model(["images"], ["nms/boxes_cxcywh", "nms/scores"])
    x = np.random.default_rng(1).standard_normal((1, 3, H, W)).astype(np.float32)
    boxes, scores = _run(pre, {"images": x})
    head = x.reshape(1, 4 + N_CLASSES, 12).transpose(0, 2, 1)
    np.testing.assert_allclose(boxes, head[..., :4])
    np.testing.assert_allclose(scores, head[..., 4:])
    assert boxes.shape == (1, 12, 4) and scores.shape == (1, 12, N_CLASSES)


def test_full_surgery_chain_keeps_uint8_input_and_nms_outputs() -> None:
    model = append_efficient_nms(prepend_uint8_preprocess(_yolo_like_model()))
    assert [i.name for i in model.graph.input] == [U8_INPUT_NAME]
    assert tuple(o.name for o in model.graph.output) == NMS_OUTPUT_NAMES
    with pytest.raises(ValueError, match="single-input/single-output"):
        head_layout(model)


def test_append_efficient_nms_rejects_bad_head() -> None:
    raw = _yolo_like_model()
    raw.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 4
    with pytest.raises(ValueError, match="num_classes"):
        append_efficient_nms(raw)
