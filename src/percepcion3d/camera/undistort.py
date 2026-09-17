"""Lens undistortion via precomputed LUT for real-time GPU/CPU remapping.

The Brown-Conrady distortion model is used, supporting radial (k1, k2, k3)
and tangential (p1, p2) distortion coefficients.

Distortion coefficient ordering in CameraIntrinsics.distortion_coeffs:
    [k1, k2, p1, p2, k3]  (OpenCV convention)

Performance contract:
    - LUT is computed ONCE at construction (cv2.initUndistortRectifyMap).
    - Per-frame undistortion via cv2.remap: target <= 1.5 ms at 1080p on GPU,
      < 5.0 ms on host CPU before GPU dispatch.
    - Map arrays are C-contiguous float32 (CV_32FC1) for GPU-compatible upload.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from percepcion3d.camera.calibration import CameraIntrinsics


class ImageRectifier:
    """Optical distortion correction with precomputed rectification lookup maps.

    The rectification maps are allocated exactly once during ``__init__`` using
    ``cv2.initUndistortRectifyMap`` with ``CV_32FC1`` maps (DoD §2 compliant).
    ``getOptimalNewCameraMatrix`` is also invoked only at construction.

    Example::

        rectifier = ImageRectifier(intrinsics)
        undistorted = rectifier.rectify(raw_frame)

    Note:
        For CUDA acceleration, upload ``self.map_x`` and ``self.map_y``
        to ``cv2.cuda.GpuMat`` objects and call ``cv2.cuda.remap`` instead.
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
        self.intrinsics: CameraIntrinsics = intrinsics
        k_mat: NDArray[np.float64] = intrinsics.k_matrix
        dist: NDArray[np.float64] = intrinsics.distortion_coeffs
        w, h = intrinsics.width, intrinsics.height

        # Compute the optimal new camera matrix that minimises black borders.
        # alpha=0.0 crops to retain only valid pixels (no black borders).
        # roi is stored for downstream cropping if needed.
        new_k: NDArray[np.float64]
        new_k, self.roi = cv2.getOptimalNewCameraMatrix(  # type: ignore[assignment]
            k_mat, dist, (w, h), alpha=0.0, newImgSize=(w, h)
        )
        self.rectified_k_matrix: NDArray[np.float64] = np.asarray(new_k, dtype=np.float64)

        # CV_32FC1 float maps: DoD-compliant format for numerical stability.
        # C-contiguous layout for zero-copy GPU upload via cuda.GpuMat.upload().
        raw_map_x: NDArray[np.float32]
        raw_map_y: NDArray[np.float32]
        raw_map_x, raw_map_y = cv2.initUndistortRectifyMap(  # type: ignore[assignment]
            k_mat,
            dist,
            np.eye(3, dtype=np.float64),  # R = identity (no stereo rectification)
            new_k,
            (w, h),
            cv2.CV_32FC1,
        )
        # Enforce C-contiguous layout and explicit dtype for deterministic GPU upload.
        self.map_x: NDArray[np.float32] = np.ascontiguousarray(raw_map_x, dtype=np.float32)
        self.map_y: NDArray[np.float32] = np.ascontiguousarray(raw_map_y, dtype=np.float32)

    def rectify(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Remove lens distortion using precomputed rectification maps.

        Args:
            frame: ``H × W × C`` or ``H × W`` uint8 BGR/grayscale frame from
                the sensor. Must match the calibrated resolution exactly.

        Returns:
            Undistorted image with the same dtype and shape as ``frame``.

        Raises:
            ValueError: If frame spatial dimensions do not match the calibrated
                resolution — guards against silent misuse before OpenCV C bindings.
        """
        h, w = frame.shape[:2]
        if (w, h) != (self.intrinsics.width, self.intrinsics.height):
            raise ValueError(
                f"Frame dimensions ({w}×{h}) do not match calibration "
                f"({self.intrinsics.width}×{self.intrinsics.height})."
            )
        result: NDArray[np.uint8] = cv2.remap(  # type: ignore[assignment]
            frame,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        return result
