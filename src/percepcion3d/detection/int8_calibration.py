"""INT8 calibration for the uint8-input detector engine (F8).

Two layers so the sampling/caching logic is testable without TensorRT:

* :class:`CalibrationBatcher` — pure NumPy: picks a deterministic, evenly
  spaced subset of KITTI frames, letterboxes each one into the engine's
  ``uint8 [1,H,W,3]`` input (the same :class:`Letterboxer` the runtime uses, so
  the calibration histograms see exactly the runtime distribution) and owns the
  calibration-cache bytes.
* :func:`make_entropy_calibrator` — wraps a batcher in a
  ``trt.IInt8EntropyCalibrator2`` (imported lazily) that uploads each batch to
  a device buffer through :class:`CudaRuntime`.

Acceptance (plan F8): INT8 is adopted only if pedestrian recall drops ≤ 2 pt
and the detector latency drops ≥ 25 % versus FP16 (see
:mod:`percepcion3d.eval.engine_compare`).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from percepcion3d.detection.letterbox import Letterboxer
from percepcion3d.runtime.cuda import CudaRuntime

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def list_calibration_images(dirs: Iterable[Path], n: int, seed: int = 0) -> list[Path]:
    """Deterministic subset of ``n`` image paths, evenly spread over ``dirs``.

    Frames within a sequence are highly correlated; taking every k-th frame from
    each directory (offset by ``seed``) covers the sequences instead of the
    first seconds of the first one.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    per_dir: list[list[Path]] = []
    for d in dirs:
        paths = sorted(p for p in Path(d).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if paths:
            per_dir.append(paths)
    total = sum(len(p) for p in per_dir)
    if total == 0:
        raise FileNotFoundError("no calibration images found")
    n = min(n, total)
    out: list[Path] = []
    remaining = n
    for i, paths in enumerate(per_dir):
        share = remaining if i == len(per_dir) - 1 else round(n * len(paths) / total)
        share = min(share, len(paths), remaining)
        if share <= 0:
            continue
        idx = np.linspace(0, len(paths) - 1, num=share, dtype=np.int64)
        idx = (idx + seed) % len(paths)
        out.extend(paths[int(j)] for j in sorted(set(idx.tolist())))
        remaining -= share
    return out


@dataclass
class CalibrationBatcher:
    """Yields letterboxed ``uint8 [1,H,W,3]`` batches and holds the calibration cache."""

    paths: Sequence[Path]
    input_hw: tuple[int, int]
    cache_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.paths and self.read_cache() is None:
            raise ValueError("no calibration images and no existing cache")
        self._lb = Letterboxer(self.input_hw)
        self._i = 0
        self.served = 0

    @property
    def batch_shape(self) -> tuple[int, int, int, int]:
        return (1, self.input_hw[0], self.input_hw[1], 3)

    def reset(self) -> None:
        self._i = 0

    def next_batch(self) -> NDArray[np.uint8] | None:
        """Next ``[1,H,W,3]`` uint8 batch (BGR, as the runtime feeds the engine) or ``None``."""
        if self._i >= len(self.paths):
            return None
        img = cv2.imread(str(self.paths[self._i]), cv2.IMREAD_COLOR)
        self._i += 1
        if img is None:
            raise OSError(f"cannot read calibration image {self.paths[self._i - 1]}")
        frame: NDArray[np.uint8] = np.asarray(img, dtype=np.uint8)
        canvas, _ = self._lb.apply(frame)
        self.served += 1
        return np.ascontiguousarray(canvas)[None].copy()

    def read_cache(self) -> bytes | None:
        if self.cache_path is not None and self.cache_path.is_file():
            return self.cache_path.read_bytes()
        return None

    def write_cache(self, blob: bytes) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(blob)


def make_entropy_calibrator(batcher: CalibrationBatcher, rt: CudaRuntime | None = None) -> Any:
    """``trt.IInt8EntropyCalibrator2`` feeding ``batcher`` through a pinned+device buffer.

    Imported lazily: only callable where ``tensorrt`` and ``cuda-python`` exist.
    """
    import tensorrt as trt  # noqa: PLC0415

    runtime = rt if rt is not None else CudaRuntime()
    stream = runtime.create_stream(high_priority=False)
    buf = runtime.alloc_pinned(batcher.batch_shape, np.dtype(np.uint8))

    class _Calibrator(trt.IInt8EntropyCalibrator2):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()

        def get_batch_size(self) -> int:
            return 1

        def get_batch(self, names: list[str]) -> list[int] | None:
            batch = batcher.next_batch()
            if batch is None:
                return None
            buf.host[...] = batch
            runtime.copy_h2d_async(buf, stream)
            stream.synchronize()
            return [buf.device_ptr] * len(names)

        def read_calibration_cache(self) -> bytes | None:
            return batcher.read_cache()

        def write_calibration_cache(self, cache: bytes) -> None:
            batcher.write_cache(bytes(cache))

    return _Calibrator()
