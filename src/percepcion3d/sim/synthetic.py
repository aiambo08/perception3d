"""Deterministic synthetic scenes: 3D constant-velocity objects → 2D boxes + relative disparity.

The generator produces exactly the observables the pipeline consumes
(bounding boxes, ground-contact rows, an *affine* relative inverse depth as a
monocular depth network would output) together with the ground truth needed
to test fusion, tracking, ego-motion compensation and TTC on CPU-only CI.

Conventions
-----------
* Ground frame G (see :mod:`percepcion3d.camera.geometry`): X right, Y down,
  Z forward; the ground plane is ``Y_g = h``. Object positions are the centre
  of the footprint on the ground.
* Velocities are given in the world; the camera moves with ``ego_velocity``.
  Relative motion is ``v_obj - v_ego``; ego yaw rate is not modelled (pure
  translation), which is the regime the EKF of F5 must handle before adding
  rotation.
* Relative inverse depth ``inv_depth = disparity_scale / Z_c + disparity_shift``
  (+ multiplicative noise): scale/shift unknown to the consumer, as with
  Depth-Anything-style outputs.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from percepcion3d.camera.geometry import PinholeGeometry

_MIN_DEPTH_M = 0.5


@dataclass(frozen=True)
class SyntheticObject:
    """A rigid cuboid moving at constant velocity on the ground plane."""

    track_id: int
    cls: str
    x0_m: float
    z0_m: float
    vx_mps: float = 0.0
    vz_mps: float = 0.0
    width_m: float = 1.8
    height_m: float = 1.5
    length_m: float = 4.2


@dataclass(frozen=True)
class SyntheticNoise:
    """Measurement noise applied to the observables (1-sigma)."""

    box_px: float = 1.0
    inv_depth_rel: float = 0.02
    contact_row_px: float = 0.0


@dataclass(frozen=True)
class SyntheticFrame:
    """Observables (noisy) and ground truth (exact) for visible objects of one frame."""

    frame_id: int
    t_ns: int
    boxes: NDArray[np.float64]
    """(N, 4) ``x1, y1, x2, y2`` in pixels, clipped to the image."""
    contact_row: NDArray[np.float64]
    """(N,) row of the footprint centre (ground-contact pixel), noisy if configured."""
    inv_depth: NDArray[np.float64]
    """(N,) relative inverse depth at the footprint centre."""
    track_ids: NDArray[np.int64]
    classes: list[str]
    gt_xyz_camera: NDArray[np.float64]
    """(N, 3) footprint centre in the camera frame (``Z`` is the optical depth)."""
    gt_xz_ground: NDArray[np.float64]
    """(N, 2) footprint centre ``(x_lat, z_fwd)`` in the ground frame."""
    gt_vel_ground: NDArray[np.float64]
    """(N, 2) relative velocity ``(vx, vz)`` in the ground frame (m/s)."""
    truncated: NDArray[np.bool_]
    """(N,) True when the unclipped box exceeded the image (contact row may be off-image)."""

    def __len__(self) -> int:
        return int(self.boxes.shape[0])


@dataclass
class SyntheticScene:
    """Deterministic scene; ``frame(k)`` depends only on ``(seed, k)``."""

    geometry: PinholeGeometry
    objects: Sequence[SyntheticObject]
    fps: float = 30.0
    ego_velocity_mps: tuple[float, float] = (0.0, 0.0)
    """Camera velocity ``(vx, vz)`` in the ground frame."""
    noise: SyntheticNoise = field(default_factory=SyntheticNoise)
    disparity_scale: float = 1.0
    disparity_shift: float = 0.0
    seed: int = 0

    @property
    def dt_s(self) -> float:
        return 1.0 / self.fps

    def relative_velocity(self, obj: SyntheticObject) -> NDArray[np.float64]:
        return np.array(
            [obj.vx_mps - self.ego_velocity_mps[0], obj.vz_mps - self.ego_velocity_mps[1]],
            dtype=np.float64,
        )

    def ground_position(self, obj: SyntheticObject, t_s: float) -> NDArray[np.float64]:
        """Footprint centre ``(x_lat, z_fwd)`` in the ground frame at time ``t_s``."""
        return np.array([obj.x0_m, obj.z0_m], dtype=np.float64) + self.relative_velocity(obj) * t_s

    def frame(self, k: int) -> SyntheticFrame:
        t_s = k * self.dt_s
        rng = np.random.default_rng([self.seed, k])
        geo = self.geometry
        h = geo.extrinsics.camera_height_m
        r_gc = geo.r_cg.T
        w_img, h_img = geo.intrinsics.width, geo.intrinsics.height

        boxes: list[NDArray[np.float64]] = []
        rows: list[float] = []
        inv_depths: list[float] = []
        ids: list[int] = []
        classes: list[str] = []
        xyz_cam: list[NDArray[np.float64]] = []
        xz_ground: list[NDArray[np.float64]] = []
        vel_ground: list[NDArray[np.float64]] = []
        truncated: list[bool] = []

        for obj in self.objects:
            x, z = self.ground_position(obj, t_s)
            base_g = np.array([x, h, z], dtype=np.float64)
            base_c = r_gc @ base_g
            if base_c[2] < _MIN_DEPTH_M:
                continue

            corners_g = _cuboid_corners(x, z, h, obj.width_m, obj.height_m, obj.length_m)
            corners_c = corners_g @ r_gc.T
            if np.any(corners_c[:, 2] < _MIN_DEPTH_M):
                continue
            u = geo.intrinsics.fx * corners_c[:, 0] / corners_c[:, 2] + geo.intrinsics.cx
            v = geo.intrinsics.fy * corners_c[:, 1] / corners_c[:, 2] + geo.intrinsics.cy
            raw = np.array([u.min(), v.min(), u.max(), v.max()], dtype=np.float64)
            clipped = np.array(
                [
                    max(raw[0], 0.0),
                    max(raw[1], 0.0),
                    min(raw[2], w_img - 1.0),
                    min(raw[3], h_img - 1.0),
                ]
            )
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue

            noisy = clipped + rng.normal(0.0, self.noise.box_px, size=4)
            noisy = np.clip(
                noisy, [0.0, 0.0, 0.0, 0.0], [w_img - 1.0, h_img - 1.0, w_img - 1.0, h_img - 1.0]
            )

            _, v_base = geo.project_point(base_c)
            row = (
                v_base + rng.normal(0.0, self.noise.contact_row_px)
                if self.noise.contact_row_px > 0
                else v_base
            )
            inv_depth = self.disparity_scale / base_c[2] + self.disparity_shift
            inv_depth *= 1.0 + rng.normal(0.0, self.noise.inv_depth_rel)

            boxes.append(noisy)
            rows.append(float(row))
            inv_depths.append(float(inv_depth))
            ids.append(obj.track_id)
            classes.append(obj.cls)
            xyz_cam.append(base_c)
            xz_ground.append(np.array([x, z], dtype=np.float64))
            vel_ground.append(self.relative_velocity(obj))
            truncated.append(bool(np.any(raw != clipped)))

        n = len(boxes)
        return SyntheticFrame(
            frame_id=k,
            t_ns=int(round(t_s * 1e9)),
            boxes=np.asarray(boxes, dtype=np.float64).reshape(n, 4),
            contact_row=np.asarray(rows, dtype=np.float64),
            inv_depth=np.asarray(inv_depths, dtype=np.float64),
            track_ids=np.asarray(ids, dtype=np.int64),
            classes=classes,
            gt_xyz_camera=np.asarray(xyz_cam, dtype=np.float64).reshape(n, 3),
            gt_xz_ground=np.asarray(xz_ground, dtype=np.float64).reshape(n, 2),
            gt_vel_ground=np.asarray(vel_ground, dtype=np.float64).reshape(n, 2),
            truncated=np.asarray(truncated, dtype=np.bool_),
        )

    def frames(self, n: int) -> Iterator[SyntheticFrame]:
        for k in range(n):
            yield self.frame(k)

    def ground_inv_depth_map(self) -> NDArray[np.float64]:
        """Dense relative inverse depth of the bare ground plane, ``nan`` at/above the horizon."""
        w_img, h_img = self.geometry.intrinsics.width, self.geometry.intrinsics.height
        vv, uu = np.mgrid[0:h_img, 0:w_img].astype(np.float64)
        hit = self.geometry.ground_hits(uu, vv)
        out = self.disparity_scale / hit.z_c + self.disparity_shift
        out[~hit.valid] = np.nan
        return np.asarray(out, dtype=np.float64)


def _cuboid_corners(
    x: float, z: float, h: float, width: float, height: float, length: float
) -> NDArray[np.float64]:
    """8 corners (ground frame) of an axis-aligned cuboid standing on ``Y_g = h``."""
    dx, dz = width / 2.0, length / 2.0
    xs = (x - dx, x + dx)
    ys = (h, h - height)
    zs = (z - dz, z + dz)
    return np.array([[cx, cy, cz] for cx in xs for cy in ys for cz in zs], dtype=np.float64)


def default_highway_scene(geometry: PinholeGeometry, seed: int = 0) -> SyntheticScene:
    """A small reference scene: lead car, overtaking car, oncoming car, pedestrian on the kerb."""
    objects = [
        SyntheticObject(1, "car", x0_m=0.0, z0_m=25.0, vz_mps=-2.0),
        SyntheticObject(2, "car", x0_m=3.5, z0_m=8.0, vz_mps=4.0),
        SyntheticObject(3, "car", x0_m=-3.5, z0_m=60.0, vz_mps=-25.0),
        SyntheticObject(
            4,
            "pedestrian",
            x0_m=5.0,
            z0_m=15.0,
            vx_mps=-0.8,
            width_m=0.6,
            height_m=1.7,
            length_m=0.6,
        ),
    ]
    return SyntheticScene(geometry, objects, fps=30.0, ego_velocity_mps=(0.0, 12.0), seed=seed)
