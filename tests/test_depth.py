"""F3 CPU tests: depth resize mapping, DepthEstimator on a fake engine, config, contention loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from percepcion3d.depth.config import DepthModelConfig, load_depth_config
from percepcion3d.depth.depth_trt import (
    DepthConfig,
    DepthEstimator,
    DepthMap,
    extract_depth_output,
)
from percepcion3d.depth.preprocess import DepthResize, DepthResizer
from percepcion3d.runtime.buffer import FrameStamped
from percepcion3d.runtime.contention import run_contention
from percepcion3d.runtime.trt_engine import InferenceHandle
from percepcion3d.utils.profiling import StageTimer

NET_HW = (28, 84)  # multiples of 14, 3:1 like the real 280x924
FRAME_HW = (60, 200)
REPO = Path(__file__).resolve().parents[1]


class _FakeHandle:
    def __init__(self, out: NDArray[np.float32], name: str, ready: bool) -> None:
        self._out = out
        self._name = name
        self._ready = ready
        self.waited = 0

    def wait(self) -> dict[str, NDArray[Any]]:
        self.waited += 1
        self._ready = True
        return {self._name: self._out}

    def ready(self) -> bool:
        return self._ready

    @property
    def gpu_ms(self) -> float | None:
        return 7.5


class FakeDepthEngine:
    """Returns ``1/(row+1)`` ramps (closer at the bottom), records inputs, fakes GPU timing."""

    def __init__(
        self,
        hw: tuple[int, int] = NET_HW,
        four_d: bool = False,
        name: str = "predicted_depth",
        ready_immediately: bool = True,
    ) -> None:
        self._shape = (1, hw[0], hw[1], 3)
        self.four_d = four_d
        self.name = name
        self.ready_immediately = ready_immediately
        self.inputs: list[NDArray[np.uint8]] = []
        self.handles: list[_FakeHandle] = []
        self.closed = False

    @property
    def input_name(self) -> str:
        return "images_u8"

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def output_names(self) -> tuple[str, ...]:
        return (self.name,)

    def infer_async(self, host_input: NDArray[Any]) -> InferenceHandle:
        assert host_input.shape == self._shape and host_input.dtype == np.uint8
        self.inputs.append(host_input.copy())
        h, w = self._shape[1], self._shape[2]
        rows = np.arange(h, dtype=np.float32)[:, None] + 1.0
        out = np.broadcast_to(rows, (h, w)).astype(np.float32)
        out = out.reshape(1, 1, h, w) if self.four_d else out.reshape(1, h, w)
        hd = _FakeHandle(out, self.name, self.ready_immediately)
        self.handles.append(hd)
        return hd

    def close(self) -> None:
        self.closed = True


def _frame(hw: tuple[int, int] = FRAME_HW, seed: int = 0) -> NDArray[np.uint8]:
    return np.random.default_rng(seed).integers(0, 256, size=(*hw, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- resize


def test_depth_resize_round_trip_and_index() -> None:
    r = DepthResize(src_h=375, src_w=1242, dst_h=280, dst_w=924)
    uv = np.array([[0.0, 0.0], [1241.0, 374.0], [621.0, 187.0]])
    back = r.to_frame(r.to_net(uv))
    np.testing.assert_allclose(back, uv, atol=1e-9)
    assert r.anisotropy < 0.01  # 3.31 vs 3.30 aspect: nearly isotropic

    row, col = r.frame_to_index(np.array([0.0, 1241.0]), np.array([0.0, 374.0]))
    assert row.tolist() == [0, 279] and col.tolist() == [0, 923]
    # frame pixel centre (u+0.5) maps into the canvas cell, never past its edge
    row, col = r.frame_to_index(np.array([1241.9]), np.array([374.9]))
    assert row[0] == 279 and col[0] == 923


def test_depth_resizer_preallocated_canvas_and_interpolation() -> None:
    rs = DepthResizer(NET_HW)
    frame = _frame()
    canvas, resize = rs.apply(frame)
    assert canvas is rs.canvas and canvas.shape == (*NET_HW, 3) and canvas.dtype == np.uint8
    assert (resize.src_h, resize.src_w, resize.dst_h, resize.dst_w) == (*FRAME_HW, *NET_HW)
    # constant image survives resize exactly (INTER_AREA is an average)
    flat = np.full((*FRAME_HW, 3), 77, dtype=np.uint8)
    canvas, _ = rs.apply(flat)
    assert np.all(canvas == 77)
    # upscale path (frame smaller than net)
    small, resize = rs.apply(_frame((10, 30)))
    assert small.shape == (*NET_HW, 3) and resize.scale_x > 1
    with pytest.raises(ValueError):
        rs.apply(frame.astype(np.float32))
    with pytest.raises(ValueError):
        rs.apply(frame[..., 0])


# --------------------------------------------------------------------------- estimator


@pytest.mark.parametrize("four_d", [False, True])
def test_depth_estimator_returns_stamped_fp16_map(four_d: bool) -> None:
    eng = FakeDepthEngine(four_d=four_d)
    timer = StageTimer()
    est = DepthEstimator(eng, DepthConfig(), timer=timer)
    m = est.infer(_frame(), frame_id=17, t_capture_ns=123_456)
    assert isinstance(m, DepthMap)
    assert (m.frame_id, m.t_capture_ns) == (17, 123_456)
    assert m.values.dtype == np.float16 and m.shape == NET_HW
    assert m.kind == "relative_disparity" and m.gpu_ms == 7.5
    assert (m.resize.src_h, m.resize.src_w) == FRAME_HW
    assert eng.inputs[0].shape == (1, *NET_HW, 3)
    rep = timer.report()
    assert {"depth.preprocess", "depth.postprocess", "depth.gpu"} <= set(rep)
    assert rep["depth.gpu"].p50_ms == pytest.approx(7.5)


def test_depth_estimator_store_fp32_and_output_name() -> None:
    eng = FakeDepthEngine(name="disp")
    est = DepthEstimator(eng, DepthConfig(output_name="disp", store_fp16=False))
    m = est.infer(_frame())
    assert m.values.dtype == np.float32
    with pytest.raises(ValueError, match="not among engine outputs"):
        DepthEstimator(FakeDepthEngine(), DepthConfig(output_name="missing"))


def test_depth_estimator_rejects_non_nhwc_engine() -> None:
    eng = FakeDepthEngine()
    eng._shape = (1, 3, *NET_HW)
    with pytest.raises(ValueError, match="1,H,W,3"):
        DepthEstimator(eng, DepthConfig())


def test_depth_handle_wait_is_idempotent_and_ready_tracks_event() -> None:
    eng = FakeDepthEngine(ready_immediately=False)
    est = DepthEstimator(eng, DepthConfig())
    h = est.infer_async(_frame(), frame_id=3)
    assert h.frame_id == 3
    assert not h.ready()
    m1 = h.wait()
    assert h.ready()
    assert h.wait() is m1 and eng.handles[0].waited == 1


def test_depth_estimator_infer_stamped_propagates_metadata() -> None:
    est = DepthEstimator(FakeDepthEngine(), DepthConfig())
    fs = FrameStamped(frame_id=9, t_capture_ns=42, img=_frame())
    m = est.infer_stamped(fs).wait()
    assert (m.frame_id, m.t_capture_ns) == (9, 42)


def test_depth_map_sampling_and_inverse_depth_semantics() -> None:
    est = DepthEstimator(FakeDepthEngine(), DepthConfig(store_fp16=False))
    m = est.infer(_frame())
    # fake output grows with the row -> bottom of the image is "closer" (larger disparity)
    inv = m.inverse_depth()
    assert inv.shape == NET_HW and np.all(np.diff(inv[:, 0]) > 0)
    # sample at frame coordinates: row v of the frame -> canvas row floor((v+.5)*scale)
    u = np.array([0.0, 100.0, 199.0])
    v = np.array([0.0, 30.0, 59.0])
    s = m.sample(u, v)
    exp_rows = np.floor((v + 0.5) * NET_HW[0] / FRAME_HW[0]).astype(int) + 1
    np.testing.assert_allclose(s, exp_rows.astype(np.float32))
    full = m.to_frame_resolution()
    assert full.shape == FRAME_HW and full.dtype == np.float32

    metric = DepthMap(0, 0, np.full(NET_HW, 4.0, np.float32), "metric_depth", m.resize)
    np.testing.assert_allclose(metric.inverse_depth(), 0.25)
    with pytest.raises(AttributeError):
        _ = metric.disparity


def test_extract_depth_output_validation() -> None:
    hw = (4, 6)
    ok = {"d": np.zeros((1, 4, 6), np.float16)}
    out = extract_depth_output(ok, None, hw)
    assert out.shape == hw and out.dtype == np.float32 and out.flags.c_contiguous
    assert extract_depth_output({"d": np.zeros((1, 1, 4, 6))}, "d", hw).shape == hw
    with pytest.raises(ValueError, match="incompatible"):
        extract_depth_output({"d": np.zeros((1, 6, 4))}, None, hw)
    with pytest.raises(ValueError, match="incompatible"):
        extract_depth_output({"d": np.zeros((2, 4, 6))}, None, hw)
    with pytest.raises(ValueError, match="output_name"):
        extract_depth_output({"a": np.zeros((1, 4, 6)), "b": np.zeros((1, 4, 6))}, None, hw)
    with pytest.raises(KeyError):
        extract_depth_output(ok, "zzz", hw)


# --------------------------------------------------------------------------- config


def test_load_depth_config_from_repo_yaml() -> None:
    cfg = load_depth_config(REPO / "configs" / "models.yaml")
    assert cfg.variant == "relative" and cfg.kind == "relative_disparity"
    assert cfg.default_hw == (280, 924)
    assert all(h % 14 == 0 and w % 14 == 0 for h, w in cfg.input_sizes)
    assert cfg.onnx_path().name == "depth_924x280.onnx"
    assert cfg.engine_path((252, 840)).name == "depth_840x252_fp16.engine"
    assert cfg.mean == (0.485, 0.456, 0.406)
    rt = cfg.runtime_config()
    assert rt.kind == "relative_disparity" and rt.store_fp16 is True


def test_depth_model_config_validation() -> None:
    def make(
        sizes: tuple[tuple[int, int], ...],
        variant: str = "relative",
        precision: str = "fp16",
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> DepthModelConfig:
        return DepthModelConfig(
            weights="w",
            onnx="a.onnx",
            engine="a.engine",
            input_sizes=sizes,
            variant=variant,
            precision=precision,
            std=std,
        )

    make(((28, 84),))
    with pytest.raises(ValueError, match="multiple"):
        make(((30, 84),))
    with pytest.raises(ValueError, match="input_sizes"):
        make(())
    with pytest.raises(ValueError, match="variant"):
        make(((28, 84),), variant="absolute")
    with pytest.raises(ValueError, match="precision"):
        make(((28, 84),), precision="bf16")
    with pytest.raises(ValueError, match="std"):
        make(((28, 84),), std=(0.0, 1.0, 1.0))
    assert make(((28, 84),), variant="metric_outdoor").runtime_config().kind == "metric_depth"


# --------------------------------------------------------------------------- contention


def test_run_contention_depth_only_waits_every_map() -> None:
    eng = FakeDepthEngine()
    est = DepthEstimator(eng, DepthConfig())
    timer = StageTimer()
    frames = [_frame(seed=i) for i in range(6)]
    seen: list[tuple[int, int | None]] = []
    res = run_contention(
        None,
        est,
        frames,
        timer,
        depth_every=2,
        on_result=lambda k, d, m: seen.append((k, None if m is None else m.frame_id)),
    )
    assert res.frames == 6 and res.depth_maps == 3 and res.depth_skipped == 0
    assert res.lag_frames == [0, 0, 0] and res.depth_rate == pytest.approx(0.5)
    assert [m for _, m in seen] == [0, None, 2, None, 4, None]
    assert timer.report()["depth_e2e"].count == 3
    assert "det_e2e" not in timer.report()


def test_run_contention_skips_slots_while_depth_in_flight() -> None:
    from percepcion3d.detection.detector_trt import Detector, DetectorConfig
    from tests.test_detector import FakeEngine as FakeDetEngine

    eng = FakeDepthEngine(ready_immediately=False)  # never "ready" until waited
    est = DepthEstimator(eng, DepthConfig())
    det = Detector(FakeDetEngine(), DetectorConfig(), timer=None)
    timer = StageTimer()
    frames = [_frame(seed=i) for i in range(5)]
    res = run_contention(det, est, frames, timer, depth_every=1)
    # depth launched at k=0, never ready inside the loop, drained at the end
    assert res.frames == 5 and res.depth_maps == 1 and res.depth_skipped == 4
    assert res.lag_frames == [4] and res.max_lag == 4
    assert timer.report()["det_e2e"].count == 5
    assert timer.report()["depth_e2e"].count == 1
    with pytest.raises(ValueError):
        run_contention(None, None, frames, timer)
    with pytest.raises(ValueError):
        run_contention(None, est, frames, timer, depth_every=0)
