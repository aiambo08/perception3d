from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from percepcion3d.camera.calibration import (
    CameraIntrinsics,
    ExtrinsicMountConfig,
    load_camera_config_yaml,
)
from percepcion3d.camera.geometry import PinholeGeometry
from percepcion3d.camera.undistort import ImageRectifier


@pytest.fixture
def kitti_geometry() -> PinholeGeometry:
    intrinsics = CameraIntrinsics(
        fx=721.5377,
        fy=721.5377,
        cx=609.5593,
        cy=172.8540,
        width=1242,
        height=375,
        distortion_coeffs=np.zeros(5, dtype=np.float64),
    )
    extrinsics = ExtrinsicMountConfig(
        camera_height_m=1.65,
        pitch_rad=np.deg2rad(2.5),
    )
    return PinholeGeometry(intrinsics, extrinsics)


# ─── DoD §1: Geometric Reversibility ──────────────────────────────────────────


@pytest.mark.parametrize(
    "p_orig",
    [
        np.array([0.5, 0.2, 2.0]),
        np.array([-1.5, 0.8, 8.5]),
        np.array([2.4, -0.4, 14.2]),
        np.array([-3.2, 1.2, 25.0]),
        np.array([0.0, 1.65, 5.0]),  # on-ground-level
        np.array([5.0, -2.0, 50.0]),  # far range
        np.array([0.0, 0.0, 0.5]),  # near range boundary
        np.array([-10.0, 5.0, 30.0]),  # lateral extreme
        np.array([3.0, 3.0, 10.0]),
        np.array([-5.0, -1.0, 20.0]),
    ],
)
def test_dod_geometric_reversibility(
    kitti_geometry: PinholeGeometry, p_orig: NDArray[np.float64]
) -> None:
    """DoD §1: ||P_orig - P_recon||_2 < 1e-5 m across operational range [0.5 m, 50 m]."""
    u, v = kitti_geometry.project_point(p_orig)
    p_recon = kitti_geometry.deproject_pixel_with_depth(u, v, z_depth=p_orig[2])
    error = float(np.linalg.norm(p_orig - p_recon))
    assert error < 1e-5, f"DoD §1 FAILED: Point {p_orig} -> error {error:.2e} m"


# ─── DoD §3: Horizon Gating ───────────────────────────────────────────────────


def test_horizon_gate_rejection(kitti_geometry: PinholeGeometry) -> None:
    """Pixels at or above the horizon line must raise ValueError."""
    v_h = kitti_geometry.get_horizon_v()

    with pytest.raises(ValueError, match="at or above the horizon line"):
        kitti_geometry.compute_ground_distances(v_h)

    with pytest.raises(ValueError, match="at or above the horizon line"):
        kitti_geometry.compute_ground_distances(v_h - 5.0)

    with pytest.raises(ValueError, match="at or above the horizon line"):
        kitti_geometry.compute_ground_distances(0.0)  # top of sensor

    # Strictly below horizon → valid
    z_cam, d_long = kitti_geometry.compute_ground_distances(v_h + 10.0)
    assert z_cam > 0.0
    assert d_long > 0.0


# ─── DoD §1+§3: Physical Invariant Z_c > d_long ──────────────────────────────


@pytest.mark.parametrize("pitch_deg", [0.5, 1.0, 2.5, 5.0, 10.0])
def test_z_cam_greater_than_d_long(pitch_deg: float) -> None:
    """Z_c_suelo must always exceed d_long when theta > 0 (hypotenuse > adjacent)."""
    intrinsics = CameraIntrinsics(
        fx=721.5377,
        fy=721.5377,
        cx=609.5593,
        cy=172.8540,
        width=1242,
        height=375,
        distortion_coeffs=np.zeros(5, dtype=np.float64),
    )
    extrinsics = ExtrinsicMountConfig(camera_height_m=1.65, pitch_rad=np.deg2rad(pitch_deg))
    geom = PinholeGeometry(intrinsics, extrinsics)
    v_test = geom.get_horizon_v() + 20.0
    if v_test >= intrinsics.height:
        pytest.skip("Horizon is below sensor boundary for this pitch.")
    z_cam, d_long = geom.compute_ground_distances(v_test)
    assert z_cam > d_long, (
        f"Invariant violated at theta={pitch_deg} deg: z_cam={z_cam:.4f} <= d_long={d_long:.4f}"
    )


# ─── DoD §2: Monotonicity ─────────────────────────────────────────────────────


def test_ground_distance_monotonicity(kitti_geometry: PinholeGeometry) -> None:
    """d_long must decrease monotonically as v increases (closer objects lower in image)."""
    v_h = kitti_geometry.get_horizon_v()
    v_values = np.linspace(v_h + 5.0, kitti_geometry.intrinsics.height - 1, num=20)
    prev_d_long = float("inf")
    for v in v_values:
        _, d_long = kitti_geometry.compute_ground_distances(float(v))
        assert d_long < prev_d_long, (
            f"Monotonicity violated at v={v:.1f}: d_long={d_long:.4f} >= prev={prev_d_long:.4f}"
        )
        assert d_long > 0.0
        prev_d_long = d_long


# ─── DoD §2: Defensive Coding ─────────────────────────────────────────────────


def test_project_point_rejects_nonpositive_z(kitti_geometry: PinholeGeometry) -> None:
    """Points with Z <= 0 must raise ValueError."""
    with pytest.raises(ValueError):
        kitti_geometry.project_point(np.array([1.0, 1.0, 0.0]))

    with pytest.raises(ValueError):
        kitti_geometry.project_point(np.array([1.0, 1.0, -5.0]))


def test_deproject_rejects_nonpositive_depth(kitti_geometry: PinholeGeometry) -> None:
    """Deprojection with depth <= 0 must raise ValueError."""
    with pytest.raises(ValueError):
        kitti_geometry.deproject_pixel_with_depth(609.0, 200.0, z_depth=0.0)

    with pytest.raises(ValueError):
        kitti_geometry.deproject_pixel_with_depth(609.0, 200.0, z_depth=-1.0)


def test_negative_camera_height_rejected() -> None:
    """ExtrinsicMountConfig must reject non-positive camera heights."""
    with pytest.raises(ValueError):
        ExtrinsicMountConfig(camera_height_m=-1.0, pitch_rad=0.04)

    with pytest.raises(ValueError):
        ExtrinsicMountConfig(camera_height_m=0.0, pitch_rad=0.04)


# ─── DoD §4b: YAML Config Parsing ────────────────────────────────────────────


def test_yaml_config_loading() -> None:
    """Validate config round-trip from configs/camera_kitti.yaml."""
    yaml_path = Path("configs/camera_kitti.yaml")
    assert yaml_path.is_file(), "configs/camera_kitti.yaml must exist."

    intrinsics, extrinsics = load_camera_config_yaml(yaml_path)
    geom = PinholeGeometry(intrinsics, extrinsics)

    assert geom.intrinsics.width == 1242
    assert geom.intrinsics.height == 375
    assert np.isclose(geom.extrinsics.camera_height_m, 1.65)
    assert np.isclose(np.rad2deg(geom.extrinsics.pitch_rad), 2.5, atol=1e-4)


def test_yaml_missing_key_raises() -> None:
    """Missing required YAML keys must raise KeyError explicitly."""
    bad_yaml = "intrinsics:\n  fx: 700.0\nextrinsics:\n  camera_height_m: 1.5\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(bad_yaml)
        tmp_path = Path(tmp.name)

    try:
        with pytest.raises(KeyError):
            load_camera_config_yaml(tmp_path)
    finally:
        tmp_path.unlink()


@pytest.mark.slow
def test_rectifier_latency_and_output_shape(kitti_geometry: PinholeGeometry) -> None:
    """
    DoD F0/F1: Verifies undistort preserves frame shape and executes within budget.
    Marked slow — excluded from CI runners (requires dedicated hardware timing).
    """
    rectifier = ImageRectifier(kitti_geometry.intrinsics)
    dummy_frame = np.zeros(
        (kitti_geometry.intrinsics.height, kitti_geometry.intrinsics.width, 3),
        dtype=np.uint8,
    )

    # Warmup
    for _ in range(10):
        _ = rectifier.rectify(dummy_frame)

    # Timed loop (100 iterations)
    start_time = time.perf_counter()
    iterations = 100
    for _ in range(iterations):
        rectified = rectifier.rectify(dummy_frame)
    elapsed_ms = ((time.perf_counter() - start_time) / iterations) * 1000.0

    assert rectified.shape == dummy_frame.shape
    # Host CPU execution should be well under 5ms; on GPU remap it is <= 1.5ms
    assert elapsed_ms < 5.0, f"Undistortion too slow on CPU: {elapsed_ms:.2f} ms"
