"""Pinhole camera geometry, Euclidean deprojections, and ground plane equations.

Coordinate convention: X right, Y down, Z forward (standard optical camera frame).
All floating-point arithmetic is performed in float64 to prevent truncation error
accumulation before GPU tensor dispatch.

Ground-plane model
------------------
The *ground frame* G shares the camera optical centre but has Z horizontal
(forward) and Y pointing straight down at the ground, which is the plane
``Y_g = h`` (``h`` = camera height). The camera frame C is obtained from G by
pitching down by ``theta`` about X and then rolling by ``rho`` about the
optical axis. For a pixel ``(u, v)`` with normalised ray
``r_c = ((u - c_x)/f_x, (v - c_y)/f_y, 1)`` the ground intersection is::

    r_g   = R_cg · r_c                 (R_cg: camera -> ground rotation)
    t     = h / r_g.y                  (ray parameter, r_g.y > 0 below horizon)
    Z_c   = t                          (optical depth, since r_c.z == 1)
    d_lon = t · r_g.z                  (forward distance on the ground)
    x_lat = t · r_g.x                  (lateral offset on the ground)
    ‖r‖   = t · |r_c|                  (Euclidean ray length)

With ``rho = 0`` and ``alpha = arctan((v - c_y)/f_y)`` this reduces to the
closed forms ``Z_c = h·cos(alpha)/sin(theta+alpha)`` and
``d_lon = h/tan(theta+alpha)``. Note that ``h/sin(theta+alpha)`` is the ray
length, **not** the optical depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from percepcion3d.camera.calibration import CameraIntrinsics, ExtrinsicMountConfig

#: Minimum downward ray component (sin of the grazing angle) accepted as a
#: valid ground intersection. Below this the distance diverges as 1/angle.
_ANGLE_GUARD_RAD: float = 1e-4

#: Finite-difference steps for the sensitivity (variance) model.
_D_PITCH_RAD: float = 1e-6
_D_ROW_PX: float = 1e-3


@dataclass(frozen=True)
class GroundNoiseModel:
    """1-sigma uncertainties feeding the ground-distance variance model.

    Attributes:
        sigma_pitch_rad: Pitch uncertainty (mount error, braking, load, bumps).
        sigma_v_px: Uncertainty of the ground-contact pixel row (box bottom).
    """

    sigma_pitch_rad: float = float(np.deg2rad(0.5))
    sigma_v_px: float = 1.0

    def __post_init__(self) -> None:
        if self.sigma_pitch_rad < 0.0 or self.sigma_v_px < 0.0:
            raise ValueError("Noise sigmas must be >= 0.")


@dataclass(frozen=True)
class GroundHit:
    """Vectorised ground-plane intersection result (all arrays share one shape).

    Entries where ``valid`` is False (pixel at/above the horizon or grazing the
    ground) hold ``nan`` in every metric field instead of raising.

    Attributes:
        z_c: Optical-axis depth to the ground contact (m).
        d_long: Forward planar distance on the ground (m).
        x_lat: Lateral offset on the ground, positive to the right (m).
        ray_length: Euclidean distance camera -> contact point (m).
        sigma_z: 1-sigma uncertainty of ``z_c`` (m).
        sigma_d: 1-sigma uncertainty of ``d_long`` (m).
        valid: Mask of pixels with a finite ground intersection.
    """

    z_c: NDArray[np.float64]
    d_long: NDArray[np.float64]
    x_lat: NDArray[np.float64]
    ray_length: NDArray[np.float64]
    sigma_z: NDArray[np.float64]
    sigma_d: NDArray[np.float64]
    valid: NDArray[np.bool_]


def camera_to_ground_rotation(pitch_rad: float, roll_rad: float = 0.0) -> NDArray[np.float64]:
    """Rotation ``R_cg`` mapping camera-frame vectors to the ground frame.

    The camera is first pitched down by ``pitch_rad`` about its X axis and then
    rolled by ``roll_rad`` about its optical axis (positive = clockwise as seen
    from behind the camera, i.e. the right-hand side of the image dips).
    Columns of ``R_cg`` are the camera axes expressed in the ground frame.
    """
    ct, st = float(np.cos(pitch_rad)), float(np.sin(pitch_rad))
    cr, sr = float(np.cos(roll_rad)), float(np.sin(roll_rad))
    pitch = np.array([[1.0, 0.0, 0.0], [0.0, ct, st], [0.0, -st, ct]], dtype=np.float64)
    roll = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return np.asarray(pitch @ roll, dtype=np.float64)


class PinholeGeometry:
    """Encapsulates pinhole projection/deprojection and ground-plane geometry.

    All depth and distance quantities are in metres (SI).
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        extrinsics: ExtrinsicMountConfig,
        noise: GroundNoiseModel | None = None,
    ) -> None:
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.noise = noise if noise is not None else GroundNoiseModel()
        self._r_cg: NDArray[np.float64] = camera_to_ground_rotation(
            extrinsics.pitch_rad, extrinsics.roll_rad
        )
        # Horizon row at the principal column; with roll the horizon is a line
        # (see get_horizon_v(u)).
        self._v_horizon: float = self.get_horizon_v(intrinsics.cx)

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

    def deproject_pixel_with_depth(self, u: float, v: float, z_depth: float) -> NDArray[np.float64]:
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

    def camera_to_ground(self, p_camera: NDArray[np.float64]) -> NDArray[np.float64]:
        """Rotate camera-frame points (…, 3) into the ground frame (…, 3).

        Ground-plane points satisfy ``p_ground[..., 1] == camera_height_m``.
        """
        p = np.asarray(p_camera, dtype=np.float64)
        return np.asarray(p @ self._r_cg.T, dtype=np.float64)

    # ------------------------------------------------------------------
    # Ground Plane Geometry
    # ------------------------------------------------------------------

    def get_horizon_v(self, u: float | None = None) -> float:
        """Pixel row of the optical horizon at column ``u`` (default: principal column).

        Without roll this is ``v_h = c_y - f_y * tan(theta)`` for every column;
        with roll the horizon is the line ``n · r_c = 0`` where ``n`` is the
        ground-frame Y axis expressed in camera coordinates.

        Returns:
            Horizon pixel row (may be fractional and outside sensor bounds).
        """
        n = self._r_cg[1]
        x = 0.0 if u is None else (u - self.intrinsics.cx) / self.intrinsics.fx
        y = -(n[0] * x + n[2]) / n[1]
        return float(self.intrinsics.cy + self.intrinsics.fy * y)

    def is_below_horizon(self, v: float, u: float | None = None) -> bool:
        """Return True if pixel (u, v) lies strictly below the horizon line."""
        return v > self.get_horizon_v(u)

    def compute_ground_distances(self, v: float, u: float | None = None) -> tuple[float, float]:
        """Compute ground-contact distances for pixel (u, v) — scalar, raising API.

        Closed forms for ``roll = 0`` and ``alpha = arctan((v - c_y)/f_y)``::

            Z_c    = h_cam · cos(alpha) / sin(theta + alpha)   [optical depth, m]
            d_long = h_cam / tan(theta + alpha)                [forward planar distance, m]

        The Euclidean ray length ``h_cam / sin(theta + alpha)`` is available via
        :meth:`ground_hits` (``ray_length``); it always exceeds both values.

        Args:
            v: Pixel row coordinate.
            u: Pixel column coordinate; only affects the result when the mount
                has roll. Defaults to the principal column.

        Returns:
            (z_cam, d_long) where:
                - z_cam: depth along the optical axis to the ground contact (metres).
                - d_long: forward planar (longitudinal) distance (metres).

        Raises:
            ValueError: If (u, v) is at or above the horizon line.
            ValueError: If the ray grazes the ground (downward component
                <= _ANGLE_GUARD_RAD); the distance diverges.
        """
        u_eff = self.intrinsics.cx if u is None else u
        v_h = self.get_horizon_v(u_eff)
        if not v > v_h:
            raise ValueError(
                f"Pixel row v={v:.2f} is at or above the horizon line "
                f"(v_horizon={v_h:.2f}). "
                "Ground distance computation is undefined for such pixels."
            )

        hit = self.ground_hits(np.array([u_eff]), np.array([v]))
        if not bool(hit.valid[0]):
            raise ValueError(
                f"Ray through v={v:.2f} grazes the ground plane "
                f"(downward component <= {_ANGLE_GUARD_RAD} rad); distance diverges."
            )
        return float(hit.z_c[0]), float(hit.d_long[0])

    def ground_hits(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
    ) -> GroundHit:
        """Vectorised ground-plane intersection with first-order uncertainty.

        Never raises for individual pixels: invalid rays (at/above horizon or
        grazing) are flagged in ``GroundHit.valid`` and filled with ``nan``.

        The uncertainty propagates the mount pitch (``sigma_pitch_rad``) and
        the contact row (``sigma_v_px``) through the exact geometry by central
        finite differences, so it stays consistent with roll and with the
        closed form ``sigma_d ≈ (d²/h)·sigma_theta`` far from the camera.

        Args:
            u: Pixel columns, any shape (broadcast with ``v``).
            v: Pixel rows, same/broadcastable shape.

        Returns:
            :class:`GroundHit` with arrays of the broadcast shape.
        """
        u_arr = np.asarray(u, dtype=np.float64)
        v_arr = np.asarray(v, dtype=np.float64)
        u_arr, v_arr = np.broadcast_arrays(u_arr, v_arr)

        z_c, d_long, x_lat, ray_length, valid = self._intersect(u_arr, v_arr, self._r_cg)

        r_plus = camera_to_ground_rotation(
            self.extrinsics.pitch_rad + _D_PITCH_RAD, self.extrinsics.roll_rad
        )
        r_minus = camera_to_ground_rotation(
            self.extrinsics.pitch_rad - _D_PITCH_RAD, self.extrinsics.roll_rad
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            z_tp, d_tp, _, _, _ = self._intersect(u_arr, v_arr, r_plus)
            z_tm, d_tm, _, _, _ = self._intersect(u_arr, v_arr, r_minus)
            z_vp, d_vp, _, _, _ = self._intersect(u_arr, v_arr + _D_ROW_PX, self._r_cg)
            z_vm, d_vm, _, _, _ = self._intersect(u_arr, v_arr - _D_ROW_PX, self._r_cg)

            dz_dtheta = (z_tp - z_tm) / (2.0 * _D_PITCH_RAD)
            dd_dtheta = (d_tp - d_tm) / (2.0 * _D_PITCH_RAD)
            dz_dv = (z_vp - z_vm) / (2.0 * _D_ROW_PX)
            dd_dv = (d_vp - d_vm) / (2.0 * _D_ROW_PX)

            s_t, s_v = self.noise.sigma_pitch_rad, self.noise.sigma_v_px
            sigma_z = np.sqrt((dz_dtheta * s_t) ** 2 + (dz_dv * s_v) ** 2)
            sigma_d = np.sqrt((dd_dtheta * s_t) ** 2 + (dd_dv * s_v) ** 2)

        nan = np.nan
        sigma_z = np.where(valid, sigma_z, nan)
        sigma_d = np.where(valid, sigma_d, nan)

        return GroundHit(
            z_c=z_c,
            d_long=d_long,
            x_lat=x_lat,
            ray_length=ray_length,
            sigma_z=np.asarray(sigma_z, dtype=np.float64),
            sigma_d=np.asarray(sigma_d, dtype=np.float64),
            valid=valid,
        )

    def _intersect(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
        r_cg: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
    ]:
        """Ray/ground-plane intersection for pixel arrays under rotation ``r_cg``."""
        x = (u - self.intrinsics.cx) / self.intrinsics.fx
        y = (v - self.intrinsics.cy) / self.intrinsics.fy
        h = self.extrinsics.camera_height_m

        # r_g = R_cg · (x, y, 1)
        gx = r_cg[0, 0] * x + r_cg[0, 1] * y + r_cg[0, 2]
        gy = r_cg[1, 0] * x + r_cg[1, 1] * y + r_cg[1, 2]
        gz = r_cg[2, 0] * x + r_cg[2, 1] * y + r_cg[2, 2]

        norm_c = np.sqrt(x * x + y * y + 1.0)
        # sin(grazing angle) = gy / |r|; guard against horizon/grazing rays.
        valid = np.asarray(gy / norm_c > _ANGLE_GUARD_RAD, dtype=np.bool_)

        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(valid, h / gy, np.nan)
        z_c = np.asarray(t, dtype=np.float64)
        d_long = np.asarray(t * gz, dtype=np.float64)
        x_lat = np.asarray(t * gx, dtype=np.float64)
        ray_length = np.asarray(t * norm_c, dtype=np.float64)
        return z_c, d_long, x_lat, ray_length, valid
