"""Plain resize (no padding) into a preallocated ``uint8`` canvas for the depth net.

Unlike the detector, the depth network gets a **stretched** frame rather than a
letterbox: grey borders would enter the ViT's global attention and shift the
affine-invariant disparity of every pixel, and padded tokens are wasted compute
(the token count is what sets the latency). The engine input size is chosen at
export time to match the sensor aspect (KITTI 1242×375 ≈ 3.31:1 → 924×280,
both multiples of 14), so the residual anisotropy is ≲ 1 %.

    u_net = u_frame · dst_w / src_w        v_net = v_frame · dst_h / src_h

:class:`DepthResize` stores the mapping so that boxes/pixels in frame
coordinates can be sampled from the low-resolution map without resizing the map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DepthResize:
    """Frame → network-canvas mapping of a plain (anisotropic) resize."""

    src_h: int
    src_w: int
    dst_h: int
    dst_w: int

    @property
    def scale_x(self) -> float:
        return self.dst_w / self.src_w

    @property
    def scale_y(self) -> float:
        return self.dst_h / self.src_h

    @property
    def anisotropy(self) -> float:
        """``|scale_x / scale_y − 1|``: 0 for an aspect-preserving resize."""
        return abs(self.scale_x / self.scale_y - 1.0)

    def to_net(self, uv_frame: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Map ``[N,2]`` pixel coordinates from frame to canvas (continuous)."""
        p = np.asarray(uv_frame, dtype=np.float64).reshape(-1, 2).copy()
        p[:, 0] *= self.scale_x
        p[:, 1] *= self.scale_y
        return p

    def to_frame(self, uv_net: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        p = np.asarray(uv_net, dtype=np.float64).reshape(-1, 2).copy()
        p[:, 0] /= self.scale_x
        p[:, 1] /= self.scale_y
        return p

    def frame_to_index(
        self, u: NDArray[np.floating[Any]], v: NDArray[np.floating[Any]]
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Nearest ``(row, col)`` of the canvas for frame pixel centres ``(u, v)``.

        Pixel ``(u, v)`` of the frame covers ``[u, u+1)``; its centre ``u+0.5``
        maps to canvas column ``floor((u+0.5)·scale_x)``, clipped to the canvas.
        """
        col = np.floor((np.asarray(u, dtype=np.float64) + 0.5) * self.scale_x).astype(np.intp)
        row = np.floor((np.asarray(v, dtype=np.float64) + 0.5) * self.scale_y).astype(np.intp)
        np.clip(col, 0, self.dst_w - 1, out=col)
        np.clip(row, 0, self.dst_h - 1, out=row)
        return row, col


class DepthResizer:
    """Resize HxWx3 uint8 frames into a reusable ``(dst_h, dst_w, 3)`` canvas."""

    def __init__(self, dst_hw: tuple[int, int]) -> None:
        self.dst_h, self.dst_w = int(dst_hw[0]), int(dst_hw[1])
        if self.dst_h <= 0 or self.dst_w <= 0:
            raise ValueError(f"invalid canvas size {dst_hw}")
        self.canvas: NDArray[np.uint8] = np.empty((self.dst_h, self.dst_w, 3), dtype=np.uint8)

    def apply(self, frame: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], DepthResize]:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError(f"expected HxWx3 uint8 frame, got {frame.shape} {frame.dtype}")
        src_h, src_w = int(frame.shape[0]), int(frame.shape[1])
        params = DepthResize(src_h, src_w, self.dst_h, self.dst_w)
        if (src_h, src_w) == (self.dst_h, self.dst_w):
            np.copyto(self.canvas, frame)
        else:
            shrinking = self.dst_w < src_w or self.dst_h < src_h
            interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
            cv2.resize(frame, (self.dst_w, self.dst_h), dst=self.canvas, interpolation=interp)
        return self.canvas, params
