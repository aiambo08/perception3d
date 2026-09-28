"""GPU smoke tests (F8): real engines, one inference each, short F7 loop.

Run on the target machine with ``uv run pytest -m gpu -v``. Every test skips
(never fails) when TensorRT/cuda-python are missing, when the engines listed in
``configs/models.yaml`` have not been built, or — for the loop test — when the
KITTI tracking root is not given through ``KT`` / ``KITTI_TRACKING_ROOT``.

The percentile report of the detector/depth tests is archived through
:mod:`percepcion3d.utils.bench_history` as ``gpu_smoke`` so successive runs can
be diffed with ``scripts/bench_history.py check --name gpu_smoke``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from percepcion3d.depth.config import load_depth_config
from percepcion3d.depth.depth_trt import DepthEstimator
from percepcion3d.detection.config import load_detector_config
from percepcion3d.detection.detector_trt import Detector
from percepcion3d.detection.int8_calibration import CalibrationBatcher
from percepcion3d.runtime.buffer import FrameStamped
from percepcion3d.utils.bench_history import archive_report
from percepcion3d.utils.profiling import StageTimer

pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "configs" / "models.yaml"
BENCH_ROOT = ROOT / "data" / "outputs" / "bench"


def _require_cuda() -> None:
    pytest.importorskip("tensorrt", reason="tensorrt not installed")
    pytest.importorskip("cuda", reason="cuda-python not installed")


def _require(path: Path) -> Path:
    if not path.is_file():
        pytest.skip(f"engine not built: {path}")
    return path


@pytest.fixture(scope="module")
def det_engine_path() -> Path:
    _require_cuda()
    return _require(load_detector_config(MODELS).engine)


@pytest.fixture(scope="module")
def depth_engine_path() -> Path:
    _require_cuda()
    return _require(load_depth_config(MODELS).engine_path())


@pytest.fixture(scope="module")
def frames() -> list[NDArray[np.uint8]]:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, size=(375, 1242, 3), dtype=np.uint8) for _ in range(30)]


@pytest.fixture(scope="module")
def detector(det_engine_path: Path) -> Iterator[tuple[Detector, StageTimer]]:
    from percepcion3d.runtime.trt_engine import TrtEngine

    timer = StageTimer(capacity=4096)
    cfg = load_detector_config(MODELS)
    det = Detector(
        TrtEngine(det_engine_path, use_cuda_graph=True, high_priority_stream=True),
        cfg.runtime_config(),
        timer=timer,
    )
    yield det, timer
    det.close()


@pytest.fixture(scope="module")
def depth(depth_engine_path: Path) -> Iterator[tuple[DepthEstimator, StageTimer]]:
    from percepcion3d.runtime.trt_engine import TrtEngine

    timer = StageTimer(capacity=4096)
    cfg = load_depth_config(MODELS)
    est = DepthEstimator(
        TrtEngine(depth_engine_path, use_cuda_graph=True, high_priority_stream=False),
        cfg.runtime_config(),
        timer=timer,
    )
    yield est, timer
    est.close()


def test_detector_engine_infers(
    detector: tuple[Detector, StageTimer], frames: list[NDArray[np.uint8]]
) -> None:
    det, timer = detector
    for f in frames:
        d = det.infer(f)
        assert d.boxes.shape[1] == 4
        assert d.boxes.shape[0] == d.scores.shape[0] == d.classes.shape[0]
        if d.boxes.shape[0]:
            assert np.all(d.boxes[:, 2] >= d.boxes[:, 0]) and np.all(d.boxes[:, 3] >= d.boxes[:, 1])
            assert np.all(d.boxes[:, [0, 2]] <= f.shape[1]) and np.all(
                d.boxes[:, [1, 3]] <= f.shape[0]
            )
    rep = timer.report()
    assert rep["det.gpu"].count == len(frames)
    assert 0.0 < rep["det.gpu"].p50_ms < 100.0


def test_depth_engine_infers(
    depth: tuple[DepthEstimator, StageTimer], frames: list[NDArray[np.uint8]]
) -> None:
    est, timer = depth
    for i, f in enumerate(frames):
        m = est.infer(f, frame_id=i, t_capture_ns=i * 16_666_667)
        assert m.frame_id == i
        assert m.values.ndim == 2 and m.values.size > 0
        assert np.isfinite(m.values).all()
    rep = timer.report()
    assert rep["depth.gpu"].count == len(frames)
    assert 0.0 < rep["depth.gpu"].p50_ms < 200.0


def test_detector_and_depth_overlap_on_separate_streams(
    detector: tuple[Detector, StageTimer],
    depth: tuple[DepthEstimator, StageTimer],
    frames: list[NDArray[np.uint8]],
) -> None:
    det, _ = detector
    est, _ = depth
    for i, f in enumerate(frames[:10]):
        fs = FrameStamped(frame_id=i, t_capture_ns=i * 16_666_667, img=f)
        h_boxes = det.infer_async(f)
        h_depth = est.infer_stamped(fs)
        boxes = h_boxes.wait()
        dm = h_depth.wait()
        assert boxes.boxes.shape[1] == 4
        assert dm.frame_id == i


def test_archive_gpu_smoke_percentiles(
    detector: tuple[Detector, StageTimer], depth: tuple[DepthEstimator, StageTimer]
) -> None:
    _, t_det = detector
    _, t_depth = depth
    stages = {**t_det.report(), **t_depth.report()}
    report = {"stages": {n: asdict(s) for n, s in stages.items()}, "frames": 30}
    path = archive_report(report, "gpu_smoke", BENCH_ROOT)
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["bench_history"]["name"] == "gpu_smoke"
    assert "det.gpu" in data["stages"] and "depth.gpu" in data["stages"]


def test_int8_calibrator_feeds_device_batches(det_engine_path: Path, tmp_path: Path) -> None:
    import cv2

    from percepcion3d.detection.int8_calibration import make_entropy_calibrator

    rng = np.random.default_rng(1)
    paths = []
    for i in range(3):
        p = tmp_path / f"{i:06d}.png"
        cv2.imwrite(str(p), rng.integers(0, 256, size=(375, 1242, 3), dtype=np.uint8))
        paths.append(p)
    hw = load_detector_config(MODELS).input_hw
    batcher = CalibrationBatcher(paths, hw, tmp_path / "calib.cache")
    calib = make_entropy_calibrator(batcher)
    assert calib.get_batch_size() == 1
    ptrs = [calib.get_batch(["images"]) for _ in range(4)]
    assert all(p is not None and p[0] > 0 for p in ptrs[:3])
    assert ptrs[3] is None
    calib.write_calibration_cache(b"TRT-cache")
    assert calib.read_calibration_cache() == b"TRT-cache"


def test_f7_loop_short_run(det_engine_path: Path, depth_engine_path: Path, tmp_path: Path) -> None:
    kt = os.environ.get("KITTI_TRACKING_ROOT") or os.environ.get("KT")
    if not kt:
        pytest.skip("set KT or KITTI_TRACKING_ROOT to run the F7 loop test")
    root = Path(kt).expanduser()
    if not (root / "image_02" / "0001").is_dir():
        pytest.skip(f"KITTI tracking sequence 0001 not found under {root}")
    out = tmp_path / "f7.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_pipeline.py"),
        "--kitti-tracking",
        str(root),
        "--seq",
        "0001",
        "--boxes",
        "detector",
        "--det-engine",
        str(det_engine_path),
        "--depth-engine",
        str(depth_engine_path),
        "--duration",
        "5",
        "--loop",
        "--telemetry",
        "none",
        "--json",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert out.is_file(), proc.stderr[-2000:]
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["stats"]["frames"] > 60, data["stats"]
    assert data["stats"]["depth"]["maps"] > 0
    assert "e2e_alert_p99" in data["verdicts"]
    assert data["verdicts"]["drops"] is not False, data["stats"]
