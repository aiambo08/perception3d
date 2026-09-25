"""KITTI Velodyne → rectified camera projection (ground truth for depth sanity checks).

Raw KITTI (``calib_velo_to_cam.txt`` + ``calib_cam_to_cam.txt``)::

    x_cam = P_rect_0c · [R_rect_00 0; 0 1] · [R T; 0 1] · x_velo

Tracking benchmark (``calib/XXXX.txt``)::

    x_cam = P2 · [R_rect 0; 0 1] · Tr_velo_cam · x_velo

In both cases the depth used for ``1/Z`` is the z of the rectified camera-0
frame (the horizontal baseline to camera 2 only shifts ``u``). Only points with
``z > 0`` inside the image are kept; :func:`nearest_per_pixel` resolves several
returns landing on one pixel by keeping the closest (the visible surface).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ProjectedLidar:
    """LiDAR returns projected into the image (continuous pixel coords, metres)."""

    u: NDArray[np.float64]
    v: NDArray[np.float64]
    z: NDArray[np.float64]

    def __len__(self) -> int:
        return int(self.u.shape[0])

    def select(self, keep: NDArray[np.bool_]) -> ProjectedLidar:
        return ProjectedLidar(self.u[keep], self.v[keep], self.z[keep])


def load_velodyne_bin(path: Path | str) -> NDArray[np.float32]:
    """``[N, 4]`` float32 ``(x, y, z, reflectance)`` from a KITTI ``.bin`` scan."""
    raw = np.fromfile(Path(path), dtype=np.float32)
    if raw.size % 4:
        raise ValueError(f"{path}: size {raw.size} is not a multiple of 4 floats")
    return raw.reshape(-1, 4)


def _read_kv(path: Path) -> dict[str, NDArray[np.float64]]:
    data: dict[str, NDArray[np.float64]] = {}
    with path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, values = line.split(":", 1)
            try:
                data[key.strip()] = np.array(values.split(), dtype=np.float64)
            except ValueError:
                continue
    return data


class LidarProjector:
    """Project Velodyne points with a ``3×4`` camera matrix and a ``4×4`` velo→rect-cam transform."""

    def __init__(
        self,
        p_rect: NDArray[np.float64],
        velo_to_rect: NDArray[np.float64],
        image_hw: tuple[int, int],
    ) -> None:
        p = np.asarray(p_rect, dtype=np.float64)
        t = np.asarray(velo_to_rect, dtype=np.float64)
        if p.shape != (3, 4) or t.shape != (4, 4):
            raise ValueError(f"expected P (3,4) and T (4,4), got {p.shape} {t.shape}")
        self.p_rect = p
        self.velo_to_rect = t
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))

    @classmethod
    def from_kitti_raw(
        cls,
        calib_dir: Path | str,
        cam_idx: int = 2,
        image_hw: tuple[int, int] | None = None,
    ) -> LidarProjector:
        d = Path(calib_dir)
        cam = _read_kv(d / "calib_cam_to_cam.txt")
        velo = _read_kv(d / "calib_velo_to_cam.txt")
        r_rect = np.eye(4)
        r_rect[:3, :3] = cam["R_rect_00"].reshape(3, 3)
        tr = np.eye(4)
        tr[:3, :3] = velo["R"].reshape(3, 3)
        tr[:3, 3] = velo["T"]
        p = cam[f"P_rect_0{cam_idx}"].reshape(3, 4)
        if image_hw is None:
            s = cam.get(f"S_rect_0{cam_idx}")
            image_hw = (int(s[1]), int(s[0])) if s is not None else (375, 1242)
        return cls(p, r_rect @ tr, image_hw)

    @classmethod
    def from_kitti_tracking(
        cls,
        calib_file: Path | str,
        cam_idx: int = 2,
        image_hw: tuple[int, int] = (375, 1242),
    ) -> LidarProjector:
        d = _read_kv(Path(calib_file))
        r_rect = np.eye(4)
        r_rect[:3, :3] = d["R_rect"].reshape(3, 3)
        tr = np.eye(4)
        tr[:3, :] = d["Tr_velo_cam"].reshape(3, 4)
        return cls(d[f"P{cam_idx}"].reshape(3, 4), r_rect @ tr, image_hw)

    def project(self, points_xyz: NDArray[np.floating[Any]]) -> ProjectedLidar:
        """Keep returns with ``z > 0`` that land inside the image."""
        pts = np.asarray(points_xyz, dtype=np.float64)[:, :3]
        hom = np.hstack([pts, np.ones((pts.shape[0], 1))])
        cam = hom @ self.velo_to_rect.T
        z = cam[:, 2]
        front = z > 1e-3
        cam = cam[front]
        z = z[front]
        img = cam @ self.p_rect.T
        u = img[:, 0] / img[:, 2]
        v = img[:, 1] / img[:, 2]
        h, w = self.image_hw
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        return ProjectedLidar(u[inside], v[inside], z[inside])


def nearest_per_pixel(proj: ProjectedLidar, image_hw: tuple[int, int]) -> ProjectedLidar:
    """Keep, for every integer pixel, only the closest return."""
    if len(proj) == 0:
        return proj
    h, w = image_hw
    col = np.clip(np.floor(proj.u).astype(np.intp), 0, w - 1)
    row = np.clip(np.floor(proj.v).astype(np.intp), 0, h - 1)
    lin = row * w + col
    order = np.lexsort((proj.z, lin))
    lin_sorted = lin[order]
    first = np.ones(lin_sorted.shape[0], dtype=bool)
    first[1:] = lin_sorted[1:] != lin_sorted[:-1]
    keep = np.zeros(len(proj), dtype=bool)
    keep[order[first]] = True
    return proj.select(keep)
