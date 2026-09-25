"""Camera calibration data structures and configuration loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters (immutable, float64-enforced)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion_coeffs: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(5, dtype=np.float64),
        hash=False,
        compare=False,
    )

    @property
    def k_matrix(self) -> NDArray[np.float64]:
        """Returns the 3×3 camera intrinsic matrix K (float64)."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class ExtrinsicMountConfig:
    """Camera extrinsic mount parameters.

    Attributes:
        camera_height_m: h_cam > 0 — height of the camera optical centre
            above the ground plane in metres.
        pitch_rad: θ — positive downwards relative to the horizon.
        roll_rad: ρ — rotation about the optical axis, positive clockwise as
            seen from behind the camera (right side of the image dips).
    """

    camera_height_m: float
    pitch_rad: float
    roll_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.camera_height_m <= 0.0:
            raise ValueError(f"camera_height_m must be > 0, got {self.camera_height_m}")


def parse_kitti_calib_txt(
    calib_file: Path | str,
    cam_idx: int = 2,
) -> CameraIntrinsics:
    """Parse a KITTI calib_cam_to_cam.txt and extract the rectified camera matrix.

    Args:
        calib_file: Path to KITTI calibration text file.
        cam_idx: Camera index (default 2 = left colour camera).

    Returns:
        CameraIntrinsics populated from the P_rect_0{cam_idx} projection matrix.

    Raises:
        FileNotFoundError: If the calibration file does not exist.
        KeyError: If the required projection matrix key is absent.
        ValueError: If the projection matrix cannot be reshaped to (3, 4).
    """
    path = Path(calib_file)
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    data: dict[str, NDArray[np.float64]] = {}
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, values = line.split(":", 1)
            data[key.strip()] = np.fromstring(values.strip(), sep=" ", dtype=np.float64)

    size_key = f"S_rect_0{cam_idx}"
    width, height = 1242, 375
    if size_key in data:
        width, height = int(data[size_key][0]), int(data[size_key][1])

    p_key = f"P_rect_0{cam_idx}"
    if p_key not in data:
        raise KeyError(
            f"Projection matrix key '{p_key}' not found in {path}. "
            f"Available keys: {list(data.keys())}"
        )

    p_rect = data[p_key].reshape(3, 4)

    return CameraIntrinsics(
        fx=float(p_rect[0, 0]),
        fy=float(p_rect[1, 1]),
        cx=float(p_rect[0, 2]),
        cy=float(p_rect[1, 2]),
        width=width,
        height=height,
        distortion_coeffs=np.zeros(5, dtype=np.float64),
    )


def load_camera_config_yaml(
    yaml_file: Path | str,
) -> tuple[CameraIntrinsics, ExtrinsicMountConfig]:
    """Load camera intrinsic and extrinsic configuration from a YAML file.

    YAML schema expected::

        intrinsics: {fx, fy, cx, cy, width, height, distortion_coeffs (optional)}
        extrinsics: {camera_height_m, pitch_deg, roll_deg (optional, default 0)}

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        KeyError: If a required field is missing from the YAML.
        ValueError: If field values are physically inconsistent.
    """
    path = Path(yaml_file)
    if not path.is_file():
        raise FileNotFoundError(f"YAML config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    if "intrinsics" not in raw_cfg:
        raise KeyError("YAML must contain top-level 'intrinsics' section.")
    if "extrinsics" not in raw_cfg:
        raise KeyError("YAML must contain top-level 'extrinsics' section.")

    intr_raw: dict[str, Any] = raw_cfg["intrinsics"]
    extr_raw: dict[str, Any] = raw_cfg["extrinsics"]

    _require_keys(intr_raw, ["fx", "fy", "cx", "cy", "width", "height"], section="intrinsics")
    _require_keys(extr_raw, ["camera_height_m", "pitch_deg"], section="extrinsics")

    dist_coeffs = np.array(
        intr_raw.get("distortion_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if dist_coeffs.shape != (5,):
        raise ValueError(
            f"distortion_coeffs must have exactly 5 elements [k1,k2,p1,p2,k3], "
            f"got shape {dist_coeffs.shape}."
        )

    intrinsics = CameraIntrinsics(
        fx=float(intr_raw["fx"]),
        fy=float(intr_raw["fy"]),
        cx=float(intr_raw["cx"]),
        cy=float(intr_raw["cy"]),
        width=int(intr_raw["width"]),
        height=int(intr_raw["height"]),
        distortion_coeffs=dist_coeffs,
    )
    extrinsics = ExtrinsicMountConfig(
        camera_height_m=float(extr_raw["camera_height_m"]),
        pitch_rad=float(np.deg2rad(float(extr_raw["pitch_deg"]))),
        roll_rad=float(np.deg2rad(float(extr_raw.get("roll_deg", 0.0)))),
    )
    return intrinsics, extrinsics


def _require_keys(mapping: dict[str, Any], keys: list[str], section: str) -> None:
    """Assert that all required keys are present in a mapping, raising KeyError otherwise."""
    for k in keys:
        if k not in mapping:
            raise KeyError(f"Required field '{k}' missing from YAML section '{section}'.")
