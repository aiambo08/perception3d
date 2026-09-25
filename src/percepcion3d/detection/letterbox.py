"""Aspect-preserving resize + pad ("letterbox") and its exact inverse for boxes.

The detector engine has a fixed rectangular input (e.g. 1024×320 for KITTI's
3.3:1 sensor). Frames are scaled by a single factor ``s`` and pasted at an
integer offset ``(pad_x, pad_y)`` into a preallocated ``uint8`` canvas, so:

    u_net = s · u_frame + pad_x        u_frame = (u_net − pad_x) / s
    v_net = s · v_frame + pad_y        v_frame = (v_net − pad_y) / s

Because ``s`` is stored as the *actual* ratio of resized/original size (not a
rounded target), the inverse is exact to floating point: boxes mapped
frame → net → frame round-trip to within 1e-9 px (see tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LetterboxParams:
    """Mapping frame → network canvas. ``scale`` is identical on both axes."""

    src_h: int
    src_w: int
    dst_h: int
    dst_w: int
    resized_h: int
    resized_w: int
    pad_x: int
    pad_y: int

    @property
    def scale(self) -> float:
        """Pixels of network canvas per pixel of frame (same on both axes)."""
        return self.resized_w / self.src_w

    @property
    def scale_y(self) -> float:
        """Vertical ratio; equals :attr:`scale` up to integer rounding of ``resized_h``."""
        return self.resized_h / self.src_h

    def to_frame(self, boxes_net: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Map ``[N,4]`` xyxy boxes from canvas to frame coordinates (no clipping)."""
        b = np.asarray(boxes_net, dtype=np.float64).reshape(-1, 4).copy()
        b[:, [0, 2]] = (b[:, [0, 2]] - self.pad_x) / self.scale
        b[:, [1, 3]] = (b[:, [1, 3]] - self.pad_y) / self.scale_y
        return b

    def to_net(self, boxes_frame: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Map ``[N,4]`` xyxy boxes from frame to canvas coordinates."""
        b = np.asarray(boxes_frame, dtype=np.float64).reshape(-1, 4).copy()
        b[:, [0, 2]] = b[:, [0, 2]] * self.scale + self.pad_x
        b[:, [1, 3]] = b[:, [1, 3]] * self.scale_y + self.pad_y
        return b

    def clip_to_frame(self, boxes_frame: NDArray[np.float64]) -> NDArray[np.float64]:
        """Clip xyxy boxes to ``[0, src_w] × [0, src_h]`` in place and return them."""
        np.clip(boxes_frame[:, 0::2], 0.0, float(self.src_w), out=boxes_frame[:, 0::2])
        np.clip(boxes_frame[:, 1::2], 0.0, float(self.src_h), out=boxes_frame[:, 1::2])
        return boxes_frame


def letterbox_params(src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> LetterboxParams:
    """Compute the centred letterbox mapping for a frame of ``src_hw`` into ``dst_hw``."""
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
        raise ValueError("letterbox sizes must be positive")
    s = min(dst_w / src_w, dst_h / src_h)
    resized_w = max(1, min(dst_w, round(src_w * s)))
    resized_h = max(1, min(dst_h, round(src_h * s)))
    pad_x = (dst_w - resized_w) // 2
    pad_y = (dst_h - resized_h) // 2
    return LetterboxParams(src_h, src_w, dst_h, dst_w, resized_h, resized_w, pad_x, pad_y)


class Letterboxer:
    """Resize + pad frames into a preallocated ``uint8`` HWC canvas.

    The canvas is allocated once per (source size, destination size) pair and
    reused, so the hot path performs a single ``cv2.resize`` write and no
    allocation. Border pixels are filled with ``fill`` (YOLO convention 114).
    """

    def __init__(self, dst_hw: tuple[int, int], fill: int = 114, channels: int = 3) -> None:
        self.dst_hw = dst_hw
        self.fill = fill
        self.channels = channels
        self._params: LetterboxParams | None = None
        self._canvas: NDArray[np.uint8] = np.full(
            (dst_hw[0], dst_hw[1], channels), fill, dtype=np.uint8
        )

    @property
    def canvas(self) -> NDArray[np.uint8]:
        """The preallocated canvas (valid after the first :meth:`apply`)."""
        return self._canvas

    def params_for(self, src_hw: tuple[int, int]) -> LetterboxParams:
        p = self._params
        if p is None or (p.src_h, p.src_w) != src_hw:
            p = letterbox_params(src_hw, self.dst_hw)
            self._params = p
            self._canvas.fill(self.fill)
        return p

    def apply(self, frame: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], LetterboxParams]:
        """Letterbox ``frame`` (HWC uint8) into the canvas; returns ``(canvas, params)``.

        The returned canvas is the internal buffer — copy it if it must outlive
        the next call.
        """
        if frame.ndim != 3 or frame.shape[2] != self.channels or frame.dtype != np.uint8:
            raise ValueError(
                f"expected HxWx{self.channels} uint8 frame, got {frame.shape} {frame.dtype}"
            )
        p = self.params_for((frame.shape[0], frame.shape[1]))
        roi = self._canvas[p.pad_y : p.pad_y + p.resized_h, p.pad_x : p.pad_x + p.resized_w]
        if (p.resized_h, p.resized_w) == (p.src_h, p.src_w):
            roi[...] = frame
        else:
            interp = cv2.INTER_AREA if p.scale < 1.0 else cv2.INTER_LINEAR
            cv2.resize(frame, (p.resized_w, p.resized_h), dst=roi, interpolation=interp)
        return self._canvas, p
