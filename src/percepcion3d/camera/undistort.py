"""Lens undistortion via precomputed LUT for real-time GPU/CPU remapping.

The Brown-Conrady distortion model is used, supporting radial (k1, k2, k3)
and tangential (p1, p2) distortion coefficients.

Distortion coefficient ordering in CameraIntrinsics.distortion_coeffs:
    [k1, k2, p1, p2, k3]  (OpenCV convention)

Performance contract:
    - LUT is computed ONCE at construction (cv2.initUndistortRectifyMap).
    - Per-frame undistortion via cv2.remap: target <= 1.5 ms at 1080p.
    - Map arrays are C-contiguous int16/uint16 (CV_16SC2) for fastest remap.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from percepcion3d.camera.calibration import CameraIntrinsics


class UndistortionLUT:
    """Precomputed undistortion look-up table for a given camera intrinsics.

    The remapping maps are allocated once during ``__init__`` using
    ``cv2.initUndistortRectifyMap`` with ``CV_16SC2`` maps, which are the
    fastest format accepted by ``cv2.remap`` and can be uploaded to the GPU
    without format conversion.

    Example::

        lut = UndistortionLUT(intrinsics)
        undistorted_frame = lut.apply(raw_frame)
    """

    def __init__(self, intrinsics: CameraIntrinsics) -> None:
        """Precompute cv2 remapping maps at initialisation.

        Args:
            intrinsics: Camera intrinsic parameters including distortion coefficients.
                ``distortion_coeffs`` must be a 5-element float64 array
                ``[k1, k2, p1, p2, k3]`` (OpenCV Brown-Conrady convention).

        Raises:
            ValueError: If ``distortion_coeffs`` does not have exactly 5 elements.
        """
        if intrinsics.distortion_coeffs.shape != (5,):
            raise ValueError(
                "distortion_coeffs must have shape (5,) — [k1, k2, p1, p2, k3], "
                f"got {intrinsics.distortion_coeffs.shape}."
            )
        k_mat = intrinsics.k_matrix
        dist = intrinsics.distortion_coeffs
        w, h = intrinsics.width, intrinsics.height

        # Compute the optimal new camera matrix that minimises black borders.
        # alpha=0.0 crops to retain only valid pixels (no black borders).
        new_k, _ = cv2.getOptimalNewCameraMatrix(
            k_mat, dist, (w, h), alpha=0.0, newImgSize=(w, h)
        )

        # CV_16SC2 integer maps are the fastest format for cv2.remap.
        # They avoid per-pixel float interpolation of map coordinates.
        # R=np.eye(3) is equivalent to identity rotation (no rectification).
        map1_raw: Any
        map2_raw: Any
        map1_raw, map2_raw = cv2.initUndistortRectifyMap(
            k_mat,
            dist,
            np.eye(3, dtype=np.float64),
            new_k,
            (w, h),
            cv2.CV_16SC2,
        )

        # Store as C-contiguous for zero-copy GPU upload via cuda.GpuMat.upload().
        # Typed as Any at the cv2 interop boundary; runtime dtype is guaranteed
        # by np.ascontiguousarray with explicit dtype= argument.
        self._map1: Any = np.ascontiguousarray(map1_raw, dtype=np.int16)
        self._map2: Any = np.ascontiguousarray(map2_raw, dtype=np.uint16)
        self._new_k: NDArray[np.float64] = new_k.astype(np.float64)
        self._intrinsics = intrinsics

    @property
    def rectified_k_matrix(self) -> NDArray[np.float64]:
        """The optimal camera matrix K' for the undistorted image."""
        return self._new_k

    def apply(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Apply undistortion to a raw frame using the precomputed LUT.

        Args:
            image: H x W x C or H x W uint8 BGR/grayscale frame from the sensor.

        Returns:
            Undistorted image with the same dtype and shape.

        Note:
            For CUDA acceleration, upload ``self._map1`` and ``self._map2``
            to ``cv2.cuda.GpuMat`` objects and call ``cv2.cuda.remap`` instead.
            This CPU path targets <= 1.5 ms at 1080p on modern hardware.
        """
        result: NDArray[np.uint8] = cv2.remap(  # type: ignore[assignment]
            image,
            self._map1,
            self._map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        return result
