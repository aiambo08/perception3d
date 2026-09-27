"""Depth / box providers shared by the KITTI evaluation scripts (F4 fusion, F5 tracking).

The TensorRT-backed providers need the ``runtime`` extra and a GPU only when called
(``TrtEngine`` imports TensorRT lazily).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from percepcion3d.depth.config import load_depth_config
from percepcion3d.depth.depth_trt import DepthEstimator, DepthMap
from percepcion3d.depth.preprocess import DepthResize
from percepcion3d.detection.config import load_detector_config
from percepcion3d.detection.detector_trt import Detector
from percepcion3d.eval.fusion_kitti import BoxProvider, DepthProvider
from percepcion3d.runtime.buffer import FrameStamped
from percepcion3d.runtime.trt_engine import TrtEngine
from percepcion3d.utils.profiling import StageTimer


def npy_depth_provider(depth_dir: Path) -> DepthProvider:
    """Relative inverse-depth maps pre-computed as ``depth_dir/<frame:06d>.npy``."""

    def provide(fs: FrameStamped) -> DepthMap | None:
        p = depth_dir / f"{fs.frame_id:06d}.npy"
        if not p.is_file():
            return None
        arr: NDArray[np.float32] = np.load(p).astype(np.float32)
        rs = DepthResize(fs.img.shape[0], fs.img.shape[1], arr.shape[0], arr.shape[1])
        return DepthMap(fs.frame_id, fs.t_capture_ns, arr, "relative_disparity", rs)

    return provide


def engine_depth_provider(engine: Path, models_cfg: Path, timer: StageTimer) -> DepthProvider:
    depth_cfg = load_depth_config(models_cfg)
    est = DepthEstimator(
        TrtEngine(engine, use_cuda_graph=True, high_priority_stream=False),
        depth_cfg.runtime_config(),
        timer=timer,
    )

    def provide(fs: FrameStamped) -> DepthMap | None:
        return est.infer_stamped(fs).wait()

    return provide


def detector_box_provider(
    models_cfg: Path, det_engine: Path | None, timer: StageTimer
) -> BoxProvider:
    det_cfg = load_detector_config(models_cfg)
    det = Detector(
        TrtEngine(det_engine or det_cfg.engine, use_cuda_graph=True, high_priority_stream=True),
        det_cfg.runtime_config(),
        timer=timer,
    )
    names = det_cfg.runtime_config()

    def provide(fs: FrameStamped) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
        d = det.infer(fs.img)
        return (
            d.boxes.astype(np.float64),
            d.scores.astype(np.float64),
            [names.name_of(int(c)) for c in d.classes],
        )

    return provide
