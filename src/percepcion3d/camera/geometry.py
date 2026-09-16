"""Pinhole camera geometry, Euclidean deprojections, and ground plane equations.

Coordinate convention: X right, Y down, Z forward (standard optical camera frame).
All floating-point arithmetic is performed in float64 to prevent truncation error
accumulation before GPU tensor dispatch.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from percepcion3d.camera.calibration import CameraIntrinsics, ExtrinsicMountConfig

#: Minimum angle guard (radians) below which sin/tan diverge unacceptably.
_ANGLE_GUARD_RAD: float = 1e-4


class PinholeGeometry:
    """Encapsulates pinhole projection/deprojection and ground-plane geometry.

    All depth and distance quantities are in metres (SI).
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        extrinsics: ExtrinsicMountConfig,
    ) -> None:
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        # Precompute horizon pixel row for fast per-frame gating (no recomputation).
        self._v_horizon: float = float(
            intrinsics.cy - intrinsics.fy * np.tan(extrinsics.pitch_rad)
        )

    # ------------------------------------------------------------------
    # Projection / Deprojection
    # ------------------------------------------------------------------

    def project_point(self, p_camera: NDArray[np.float64]) -> tuple[float, float]:
        """Project a metric 3D point [X_c, Y_c, Z_c] to pixel (u, v).

        Args:
            p_camera: Shape (3,) float64 array in the camera optical frame
                (X right, Y down, Z forward).

        Returns:
            (u, v) pixel coordinates as Python floats.

        Raises:
            ValueError: If Z_c <= 0 (point behind or on the optical plane).
        """
        p = np.asarray(p_camera, dtype=np.float64)
        z = float(p[2])
        if z <= 1e-4:
            raise ValueError(
                f"Cannot project points with Z_c={z:.6f} <= 0 "
                "(point is behind or on the camera optical plane)."
            )
        u = float(self.intrinsics.fx * p[0] / z + self.intrinsics.cx)
        v = float(self.intrinsics.fy * p[1] / z + self.intrinsics.cy)
        return u, v

    def deproject_pixel_to_ray(self, u: float, v: float) -> NDArray[np.float64]:
        """Compute the normalised unit direction vector for pixel (u, v).

        Returns:
            Shape (3,) float64 unit vector [dx, dy, dz=1 normalised].
        """
        x = (u - self.intrinsics.cx) / self.intrinsics.fx
        y = (v - self.intrinsics.cy) / self.intrinsics.fy
        ray: NDArray[np.float64] = np.array([x, y, 1.0], dtype=np.float64)
        return ray / np.linalg.norm(ray)

    def deproject_pixel_with_depth(
        self, u: float, v: float, z_depth: float
    ) -> NDArray[np.float64]:
        """Reconstruct metric 3D point [X_c, Y_c, Z_c] from pixel and known optical depth Z.

        Note:
            z_depth is the **optical axis depth** (Z_c), not the Euclidean
            ray length. Use this for the DoD geometric reversibility test.

        Args:
            u: Pixel column coordinate.
            v: Pixel row coordinate.
            z_depth: Metric depth along the optical axis (Z_c), in metres.

        Returns:
            Shape (3,) float64 array [X_c, Y_c, Z_c].

        Raises:
            ValueError: If z_depth <= 0.
        """
        if z_depth <= 0.0:
            raise ValueError(f"Metric depth Z_c must be > 0, got {z_depth:.6f}.")
        x = (u - self.intrinsics.cx) * z_depth / self.intrinsics.fx
        y = (v - self.intrinsics.cy) * z_depth / self.intrinsics.fy
        return np.array([x, y, z_depth], dtype=np.float64)

    # ------------------------------------------------------------------
    # Ground Plane Geometry
    # ------------------------------------------------------------------

    def get_horizon_v(self) -> float:
        """Pixel row of the optical horizon: v_h = c_y - f_y * tan(theta).

        Returns:
            Horizon pixel row (may be fractional and outside sensor bounds).
        """
        return self._v_horizon

    def is_below_horizon(self, v: float) -> bool:
        """Return True if pixel row v lies strictly below the horizon."""
        return v > self._v_horizon

    def compute_ground_distances(self, v: float) -> tuple[float, float]:
        """Compute ground-contact distances for pixel row v.

        Uses the ground plane equations from the v3.0 technical report::

            alpha      = arctan((v - c_y) / f_y)     [elevation angle, rad]
            Z_c_suelo  = h_cam / sin(theta + alpha)   [optical-ray metric depth, m]
            d_long     = h_cam / tan(theta + alpha)   [forward planar distance, m]

        Physical identity: Z_c_suelo > d_long always holds for theta > 0,
        because the ray hypotenuse exceeds the planar adjacent side
        (sin(x) < tan(x) for x in (0, pi/2)).

        Args:
            v: Pixel row coordinate.

        Returns:
            (z_cam, d_long) where:
                - z_cam: depth along the optical axis to the ground contact (metres).
                - d_long: forward planar (longitudinal) Euclidean distance (metres).

        Raises:
            ValueError: If v is at or above the horizon line.
            ValueError: If theta + alpha <= _ANGLE_GUARD_RAD (ray nearly grazes ground).
        """
        if not self.is_below_horizon(v):
            raise ValueError(
                f"Pixel row v={v:.2f} is at or above the horizon line "
                f"(v_horizon={self._v_horizon:.2f}). "
                "Ground distance computation is undefined for such pixels."
            )

        alpha: float = float(np.arctan((v - self.intrinsics.cy) / self.intrinsics.fy))
        total_angle: float = self.extrinsics.pitch_rad + alpha

        if total_angle <= _ANGLE_GUARD_RAD:
            raise ValueError(
                f"Combined angle theta+alpha={total_angle:.6f} rad <= {_ANGLE_GUARD_RAD} rad. "
                "Ray is nearly parallel to the ground plane; distance diverges."
            )

        h = self.extrinsics.camera_height_m
        # Z_c_suelo: depth along the optical axis to the ground contact point
        z_cam: float = h / float(np.sin(total_angle))
        # d_long: forward planar (longitudinal) Euclidean distance on the ground
        d_long: float = h / float(np.tan(total_angle))

        return z_cam, d_long
