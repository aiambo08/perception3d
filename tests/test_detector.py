"""CPU tests for F2: letterbox inverse, Detector on a fake engine, config, recall."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from percepcion3d.detection.config import load_detector_config
from percepcion3d.detection.detector_trt import (
    NMS_OUTPUTS,
    Detections,
    Detector,
    DetectorConfig,
    EngineBackend,
    InferenceHandle,
    postprocess_nms_outputs,
)
from percepcion3d.detection.letterbox import Letterboxer, letterbox_params
from percepcion3d.eval.detection import (
    GtBox,
    PredBox,
    format_recall,
    iou_xyxy,
    match_greedy,
    recall_by_type,
)
from percepcion3d.utils.profiling import StageTimer

KITTI_HW = (375, 1242)
NET_HW = (320, 1024)
REPO = Path(__file__).resolve().parents[1]


# ─── Letterbox ────────────────────────────────────────────────────────────────


def test_letterbox_params_kitti_to_1024x320() -> None:
    p = letterbox_params(KITTI_HW, NET_HW)
    assert (p.resized_h, p.resized_w) == (309, 1024)
    assert p.pad_x == 0 and p.pad_y == 5
    assert p.scale == pytest.approx(1024 / 1242)
    assert p.scale_y == pytest.approx(309 / 375)


@pytest.mark.parametrize(
    "src_hw,dst_hw",
    [
        (KITTI_HW, NET_HW),
        ((720, 1280), (640, 640)),
        ((480, 640), (320, 1024)),
        ((100, 100), (320, 1024)),
    ],
)
def test_letterbox_roundtrip_is_exact(src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> None:
    p = letterbox_params(src_hw, dst_hw)
    rng = np.random.default_rng(1)
    xy1 = rng.uniform(0, [src_hw[1] * 0.5, src_hw[0] * 0.5], size=(200, 2))
    xy2 = xy1 + rng.uniform(1, [src_hw[1] * 0.5, src_hw[0] * 0.5], size=(200, 2))
    boxes = np.hstack([xy1, xy2])
    back = p.to_frame(p.to_net(boxes))
    np.testing.assert_allclose(back, boxes, rtol=0, atol=1e-9)
    net = p.to_net(boxes)
    assert (
        net[:, 0::2].min() >= p.pad_x - 1e-9 and net[:, 0::2].max() <= p.pad_x + p.resized_w + 1e-9
    )
    assert (
        net[:, 1::2].min() >= p.pad_y - 1e-9 and net[:, 1::2].max() <= p.pad_y + p.resized_h + 1e-9
    )


def test_letterbox_corners_map_to_canvas_roi() -> None:
    p = letterbox_params(KITTI_HW, NET_HW)
    frame_full = np.array([[0.0, 0.0, 1242.0, 375.0]])
    net = p.to_net(frame_full)[0]
    np.testing.assert_allclose(net, [0.0, 5.0, 1024.0, 314.0], atol=1e-9)


def test_clip_to_frame_mutates_in_place() -> None:
    p = letterbox_params(KITTI_HW, NET_HW)
    b = np.array([[-10.0, -5.0, 2000.0, 400.0], [10.0, 20.0, 30.0, 40.0]])
    out = p.clip_to_frame(b)
    assert out is b
    np.testing.assert_allclose(b[0], [0.0, 0.0, 1242.0, 375.0])
    np.testing.assert_allclose(b[1], [10.0, 20.0, 30.0, 40.0])


def test_letterbox_params_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        letterbox_params((0, 10), NET_HW)


def test_letterboxer_writes_into_preallocated_canvas() -> None:
    lb = Letterboxer(NET_HW, fill=114)
    frame = np.full((*KITTI_HW, 3), 200, dtype=np.uint8)
    canvas, p = lb.apply(frame)
    assert canvas is lb.canvas
    assert canvas.shape == (*NET_HW, 3) and canvas.dtype == np.uint8
    assert (canvas[: p.pad_y] == 114).all() and (canvas[p.pad_y + p.resized_h :] == 114).all()
    assert (canvas[p.pad_y : p.pad_y + p.resized_h] == 200).all()
    # A second frame of another size reuses the canvas and refreshes the padding.
    frame2 = np.full((480, 640, 3), 50, dtype=np.uint8)
    canvas2, p2 = lb.apply(frame2)
    assert canvas2 is lb.canvas
    assert p2.pad_x > 0
    assert (canvas2[:, : p2.pad_x] == 114).all()
    assert (canvas2[:, p2.pad_x : p2.pad_x + p2.resized_w] == 50).all()


def test_letterboxer_pixel_position_matches_box_mapping() -> None:
    """A white square drawn in the frame lands where ``to_net`` says it should."""
    lb = Letterboxer(NET_HW, fill=0)
    frame = np.zeros((*KITTI_HW, 3), dtype=np.uint8)
    x1, y1, x2, y2 = 400, 100, 600, 300
    frame[y1:y2, x1:x2] = 255
    canvas, p = lb.apply(frame)
    ys, xs = np.nonzero(canvas[..., 0] > 127)
    nx1, ny1, nx2, ny2 = p.to_net(np.array([[x1, y1, x2, y2]], dtype=np.float64))[0]
    assert abs(xs.min() - nx1) <= 1.5 and abs(xs.max() + 1 - nx2) <= 1.5
    assert abs(ys.min() - ny1) <= 1.5 and abs(ys.max() + 1 - ny2) <= 1.5


def test_letterboxer_rejects_bad_frames() -> None:
    lb = Letterboxer(NET_HW)
    with pytest.raises(ValueError):
        lb.apply(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        lb.apply(np.zeros((10, 10, 3), dtype=np.float32))


# ─── Fake engine + Detector ───────────────────────────────────────────────────


class _FakeHandle:
    def __init__(self, outputs: dict[str, NDArray[Any]], gpu_ms: float | None) -> None:
        self._outputs = outputs
        self._gpu_ms = gpu_ms
        self.waited = 0

    def wait(self) -> dict[str, NDArray[Any]]:
        self.waited += 1
        return self._outputs

    @property
    def gpu_ms(self) -> float | None:
        return self._gpu_ms


class FakeEngine:
    """Returns canned EfficientNMS outputs; records the inputs it receives."""

    def __init__(self, hw: tuple[int, int] = NET_HW, max_out: int = 100) -> None:
        self._shape = (1, hw[0], hw[1], 3)
        self.max_out = max_out
        self.inputs: list[NDArray[np.uint8]] = []
        self.canned: list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]] = []
        self.closed = False

    @property
    def input_name(self) -> str:
        return "images_u8"

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def output_names(self) -> tuple[str, ...]:
        return NMS_OUTPUTS

    def infer_async(self, host_input: NDArray[Any]) -> InferenceHandle:
        assert host_input.shape == self._shape and host_input.dtype == np.uint8
        self.inputs.append(host_input.copy())
        boxes, scores, classes = (
            self.canned.pop(0)
            if self.canned
            else (
                np.zeros((0, 4)),
                np.zeros(0),
                np.zeros(0, dtype=np.int64),
            )
        )
        k = self.max_out
        n = boxes.shape[0]
        det_boxes = np.zeros((1, k, 4), np.float32)
        det_scores = np.zeros((1, k), np.float32)
        det_classes = np.zeros((1, k), np.int32)
        det_boxes[0, :n] = boxes
        det_scores[0, :n] = scores
        det_classes[0, :n] = classes
        return _FakeHandle(
            {
                "num_dets": np.array([[n]], np.int32),
                "det_boxes": det_boxes,
                "det_scores": det_scores,
                "det_classes": det_classes,
            },
            gpu_ms=1.5,
        )

    def close(self) -> None:
        self.closed = True


def test_fake_engine_satisfies_protocol() -> None:
    assert isinstance(FakeEngine(), EngineBackend)


def test_detector_maps_boxes_back_to_frame_coordinates() -> None:
    eng = FakeEngine()
    p = letterbox_params(KITTI_HW, NET_HW)
    frame_boxes = np.array([[100.0, 50.0, 300.0, 200.0], [900.0, 150.0, 1200.0, 370.0]])
    eng.canned.append((p.to_net(frame_boxes), np.array([0.9, 0.4]), np.array([2, 0])))
    timer = StageTimer()
    det = Detector(eng, DetectorConfig(class_names=("person", "bicycle", "car")), timer=timer)
    frame = np.zeros((*KITTI_HW, 3), dtype=np.uint8)

    handle = det.infer_async(frame)
    assert eng.inputs[0].shape == (1, *NET_HW, 3)
    result = handle.wait()
    assert handle.wait() is result  # idempotent

    assert len(result) == 2
    np.testing.assert_allclose(result.boxes, frame_boxes, atol=1e-4)
    np.testing.assert_allclose(result.scores, [0.9, 0.4], atol=1e-6)
    assert result.classes.tolist() == [2, 0]
    assert result.gpu_ms == 1.5
    assert result.letterbox == p
    arr = result.as_array()
    assert arr.shape == (2, 6) and arr.dtype == np.float32
    assert det.cfg.name_of(2) == "car" and det.cfg.name_of(7) == "7"
    assert {"det.preprocess", "det.postprocess", "det.gpu"} <= set(timer.stage_names)
    det.close()
    assert eng.closed


def test_detector_filters_score_size_and_classes_and_clips() -> None:
    eng = FakeEngine()
    p = letterbox_params(KITTI_HW, NET_HW)
    frame_boxes = np.array(
        [
            [10.0, 10.0, 100.0, 100.0],  # kept
            [10.0, 10.0, 100.0, 100.0],  # low score
            [10.0, 10.0, 11.0, 100.0],  # too thin
            [10.0, 10.0, 100.0, 100.0],  # class not kept
            [-50.0, -20.0, 1300.0, 400.0],  # clipped to frame
        ]
    )
    eng.canned.append(
        (p.to_net(frame_boxes), np.array([0.9, 0.1, 0.9, 0.9, 0.8]), np.array([2, 2, 2, 5, 0]))
    )
    cfg = DetectorConfig(score_threshold=0.25, min_box_px=2.0, keep_classes=frozenset({0, 2}))
    result = Detector(eng, cfg).infer(np.zeros((*KITTI_HW, 3), dtype=np.uint8))
    assert len(result) == 2
    np.testing.assert_allclose(result.boxes[0], [10, 10, 100, 100], atol=1e-4)
    np.testing.assert_allclose(result.boxes[1], [0, 0, 1242, 375], atol=1e-4)
    assert result.classes.tolist() == [2, 0]


def test_detector_empty_output_and_filter() -> None:
    eng = FakeEngine()
    det = Detector(eng)
    result = det.infer(np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(result) == 0 and result.as_array().shape == (0, 6)
    sub = result.filter(np.zeros(0, dtype=bool))
    assert len(sub) == 0
    assert Detections.empty(result.letterbox).letterbox == result.letterbox


def test_detector_rejects_incompatible_engines() -> None:
    class Bad(FakeEngine):
        @property
        def input_shape(self) -> tuple[int, ...]:
            return (1, 3, 320, 1024)

    class NoNms(FakeEngine):
        @property
        def output_names(self) -> tuple[str, ...]:
            return ("output0",)

    with pytest.raises(ValueError, match="1,H,W,3"):
        Detector(Bad())
    with pytest.raises(ValueError, match="EfficientNMS"):
        Detector(NoNms())


def test_postprocess_handles_num_dets_zero() -> None:
    p = letterbox_params(KITTI_HW, NET_HW)
    out = {
        "num_dets": np.zeros((1, 1), np.int32),
        "det_boxes": np.zeros((1, 100, 4), np.float32),
        "det_scores": np.zeros((1, 100), np.float32),
        "det_classes": np.zeros((1, 100), np.int32),
    }
    d = postprocess_nms_outputs(out, p, DetectorConfig())
    assert len(d) == 0


def test_trt_engine_import_is_lazy() -> None:
    """Importing the detector module must not pull TensorRT/CUDA into the CPU core."""
    assert "tensorrt" not in sys.modules
    assert not any(m == "cuda" or m.startswith("cuda.") for m in sys.modules)


def test_trt_engine_reports_missing_tensorrt(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    if "tensorrt" in sys.modules or _module_available("tensorrt"):
        pytest.skip("tensorrt installed")
    from percepcion3d.detection.detector_trt import TrtEngine

    with pytest.raises(ImportError):
        TrtEngine(tmp_path / "none.engine")


def _module_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


# ─── Config ───────────────────────────────────────────────────────────────────


def test_models_yaml_loads_and_builds_runtime_config() -> None:
    cfg = load_detector_config(REPO / "configs" / "models.yaml")
    assert cfg.input_hw == (320, 1024)
    assert cfg.engine.suffix == ".engine" and cfg.engine.is_absolute()
    assert len(cfg.class_names) == 80 and cfg.class_names[2] == "car"
    rt = cfg.runtime_config()
    assert rt.keep_classes == frozenset({0, 1, 2, 3, 5, 7})
    assert rt.score_threshold == cfg.nms.score_threshold
    assert cfg.kitti_to_coco["Pedestrian"] == ("person",)


def test_models_yaml_validation(tmp_path: Path) -> None:
    bad = tmp_path / "m.yaml"
    bad.write_text("detector:\n  weights: x.pt\n  input_hw: [300, 1024]\n  onnx: a\n  engine: b\n")
    with pytest.raises(ValueError, match="multiples of 32"):
        load_detector_config(bad)
    bad.write_text("other: 1\n")
    with pytest.raises(ValueError, match="detector"):
        load_detector_config(bad)


# ─── Eval: IoU / matching / recall ────────────────────────────────────────────


def test_iou_xyxy_known_values() -> None:
    a = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    b = np.array([[5, 5, 15, 15], [0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    iou = iou_xyxy(a, b)
    assert iou.shape == (2, 3)
    assert iou[0, 0] == pytest.approx(25 / 175)
    assert iou[0, 1] == pytest.approx(1.0)
    assert iou[0, 2] == 0.0
    assert iou_xyxy(np.zeros((0, 4)), b).shape == (0, 3)
    assert iou_xyxy(np.array([[0, 0, 0, 0]]), np.array([[0, 0, 0, 0]]))[0, 0] == 0.0


def test_match_greedy_prefers_high_score_one_to_one() -> None:
    gt = np.array([[0, 0, 10, 10]], dtype=float)
    preds = np.array([[0, 0, 10, 10], [1, 1, 10, 10]], dtype=float)
    assign, matched = match_greedy(preds, np.array([0.5, 0.9]), gt)
    assert assign.tolist() == [-1, 0]  # the 0.9 prediction takes the only GT
    assert matched.tolist() == [True]
    assign, matched = match_greedy(preds, np.array([0.5, 0.9]), gt, iou_threshold=0.99)
    assert assign.tolist() == [0, -1]  # 0.9 has IoU 0.81 < 0.99 → next candidate wins


def test_recall_by_type_maps_coco_to_kitti() -> None:
    gts = [
        GtBox(0, "Car", (0, 0, 100, 60)),
        GtBox(0, "Pedestrian", (200, 0, 240, 100)),
        GtBox(1, "Car", (0, 0, 100, 60)),
        GtBox(1, "Car", (300, 0, 310, 10)),  # < 25 px high: ignored
        GtBox(1, "DontCare", (0, 0, 50, 50)),
    ]
    preds = [
        PredBox(0, "car", 0.9, (2, 1, 99, 61)),
        PredBox(0, "person", 0.8, (201, 0, 241, 99)),
        PredBox(1, "truck", 0.7, (0, 0, 100, 60)),  # truck counts as Car
        PredBox(1, "person", 0.7, (500, 0, 540, 100)),  # false positive
    ]
    rows = recall_by_type(
        preds, gts, type_to_classes={"Car": ["car", "truck"], "Pedestrian": ["person"]}
    )
    assert rows["Car"].n_gt == 2 and rows["Car"].n_matched == 2 and rows["Car"].recall == 1.0
    assert rows["Pedestrian"].n_gt == 1 and rows["Pedestrian"].n_matched == 1
    assert rows["Pedestrian"].n_pred == 2 and rows["Pedestrian"].precision == pytest.approx(0.5)
    table = format_recall(rows)
    assert "Car" in table and "Pedestrian" in table


def test_recall_nan_when_no_gt() -> None:
    rows = recall_by_type([], [GtBox(0, "Car", (0, 0, 10, 100))], type_to_classes={"Car": ["car"]})
    assert rows["Car"].recall == 0.0 and np.isnan(rows["Car"].precision)
    rows = recall_by_type([], [], type_to_classes={"Car": ["car"]})
    assert np.isnan(rows["Car"].recall)
