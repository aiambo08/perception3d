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
from percepcion3d.camera.geometry import (
    GroundNoiseModel,
    PinholeGeometry,
    camera_to_ground_rotation,
)
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


# ─── DoD §1+§3: Physical invariants of the ground intersection ────────────────


@pytest.mark.parametrize("pitch_deg", [0.5, 1.0, 2.5, 5.0, 10.0])
def test_ground_intersection_invariants(pitch_deg: float) -> None:
    """Ray length is the hypotenuse (> d_long and >= Z_c); Z_c = d_long cos(theta) + h sin(theta)."""
    intrinsics = CameraIntrinsics(
        fx=721.5377,
        fy=721.5377,
        cx=609.5593,
        cy=172.8540,
        width=1242,
        height=375,
        distortion_coeffs=np.zeros(5, dtype=np.float64),
    )
    theta = float(np.deg2rad(pitch_deg))
    extrinsics = ExtrinsicMountConfig(camera_height_m=1.65, pitch_rad=theta)
    geom = PinholeGeometry(intrinsics, extrinsics)
    v_h = geom.get_horizon_v()
    v_rows = np.linspace(v_h + 20.0, intrinsics.height - 1.0, num=8)
    hit = geom.ground_hits(np.full_like(v_rows, intrinsics.cx), v_rows)

    assert hit.valid.all()
    assert np.all(hit.ray_length > hit.d_long)
    assert np.all(hit.ray_length >= hit.z_c - 1e-12)
    expected_z = hit.d_long * np.cos(theta) + 1.65 * np.sin(theta)
    np.testing.assert_allclose(hit.z_c, expected_z, rtol=0, atol=1e-9)
    np.testing.assert_allclose(hit.ray_length, np.hypot(hit.d_long, 1.65), rtol=0, atol=1e-9)


# ─── F0.1 DoD: closed-form value, consistency, variance ──────────────────────────


def test_ground_distances_closed_form_kitti(kitti_geometry: PinholeGeometry) -> None:
    """KITTI, bottom row v=374: Z_c = h cos(alpha)/sin(theta+alpha) ~ 5.12 m, d_long ~ 5.06 m."""
    z_c, d_long = kitti_geometry.compute_ground_distances(374.0)
    assert abs(z_c - 5.122) < 0.01
    assert abs(d_long - 5.055) < 0.01

    intr, extr = kitti_geometry.intrinsics, kitti_geometry.extrinsics
    alpha = np.arctan((374.0 - intr.cy) / intr.fy)
    phi = extr.pitch_rad + alpha
    assert np.isclose(z_c, extr.camera_height_m * np.cos(alpha) / np.sin(phi), atol=1e-9)
    assert np.isclose(d_long, extr.camera_height_m / np.tan(phi), atol=1e-9)

    hit = kitti_geometry.ground_hits(np.array([intr.cx]), np.array([374.0]))
    assert np.isclose(hit.ray_length[0], extr.camera_height_m / np.sin(phi), atol=1e-9)


@pytest.mark.parametrize("pitch_deg", [0.0, 2.5, 7.0])
@pytest.mark.parametrize("u", [50.0, 609.5593, 1200.0])
def test_deprojected_contact_point_lies_on_ground(pitch_deg: float, u: float) -> None:
    """deproject_pixel_with_depth(u, v, Z_c) rotated by R_cg has Y equal to the camera height."""
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
    for v in (250.0, 300.0, 374.0):
        z_c, d_long = geom.compute_ground_distances(v, u)
        p_cam = geom.deproject_pixel_with_depth(u, v, z_c)
        p_ground = geom.camera_to_ground(p_cam)
        assert abs(p_ground[1] - 1.65) < 1e-3, f"Y_g={p_ground[1]:.6f} at v={v}"
        assert np.isclose(p_ground[2], d_long, atol=1e-9)
        if pitch_deg == 0.0:
            assert abs(p_cam[1] - 1.65) < 1e-3


def test_ground_variance_model_kitti() -> None:
    """sigma_d(d=20 m, sigma_theta=0.5 deg) ~ (d^2/h) sigma_theta ~ 2.1-2.2 m (R2 model)."""
    intrinsics = CameraIntrinsics(
        fx=721.5377,
        fy=721.5377,
        cx=609.5593,
        cy=172.8540,
        width=1242,
        height=375,
        distortion_coeffs=np.zeros(5, dtype=np.float64),
    )
    extrinsics = ExtrinsicMountConfig(camera_height_m=1.65, pitch_rad=np.deg2rad(2.5))
    sigma_theta = float(np.deg2rad(0.5))
    geom = PinholeGeometry(intrinsics, extrinsics, GroundNoiseModel(sigma_theta, 0.0))

    # Pixel row whose ground distance is 20 m: alpha = arctan(h/d) - theta.
    phi = np.arctan(1.65 / 20.0)
    v_20 = intrinsics.cy + intrinsics.fy * np.tan(phi - extrinsics.pitch_rad)
    hit = geom.ground_hits(np.array([intrinsics.cx]), np.array([v_20]))
    assert np.isclose(hit.d_long[0], 20.0, atol=1e-6)

    exact = 1.65 / np.sin(phi) ** 2 * sigma_theta  # |dd/dtheta| = h / sin^2(phi)
    assert np.isclose(hit.sigma_d[0], exact, rtol=1e-3)
    assert abs(hit.sigma_d[0] - 2.2) / 2.2 < 0.05

    # Pixel-row term alone: |dd/dv| = (h / sin^2 phi) * cos^2(alpha) / f_y.
    geom_px = PinholeGeometry(intrinsics, extrinsics, GroundNoiseModel(0.0, 1.0))
    hit_px = geom_px.ground_hits(np.array([intrinsics.cx]), np.array([v_20]))
    alpha = phi - extrinsics.pitch_rad
    exact_px = 1.65 / np.sin(phi) ** 2 * np.cos(alpha) ** 2 / intrinsics.fy
    assert np.isclose(hit_px.sigma_d[0], exact_px, rtol=1e-3)

    # Uncertainty grows monotonically with distance.
    v_rows = np.linspace(geom.get_horizon_v() + 5.0, intrinsics.height - 1.0, num=30)
    sig = geom.ground_hits(np.full_like(v_rows, intrinsics.cx), v_rows).sigma_d
    assert np.all(np.diff(sig) < 0.0)


def test_ground_hits_vectorised_masks_invalid(kitti_geometry: PinholeGeometry) -> None:
    """Above-horizon pixels yield valid=False and nan instead of raising."""
    v_h = kitti_geometry.get_horizon_v()
    v = np.array([0.0, v_h - 1.0, v_h, v_h + 0.01, v_h + 10.0, 374.0])
    hit = kitti_geometry.ground_hits(np.full_like(v, kitti_geometry.intrinsics.cx), v)

    np.testing.assert_array_equal(hit.valid, [False, False, False, False, True, True])
    assert np.isnan(hit.z_c[~hit.valid]).all()
    assert np.isnan(hit.sigma_z[~hit.valid]).all()
    assert np.isfinite(hit.z_c[hit.valid]).all()

    z_scalar, d_scalar = kitti_geometry.compute_ground_distances(374.0)
    assert np.isclose(hit.z_c[-1], z_scalar) and np.isclose(hit.d_long[-1], d_scalar)

    with pytest.raises(ValueError, match="grazes the ground"):
        kitti_geometry.compute_ground_distances(v_h + 0.01)

    # Broadcasting over a (rows, cols) grid.
    grid = kitti_geometry.ground_hits(np.arange(0.0, 1242.0, 200.0)[None, :], v[:, None])
    assert grid.z_c.shape == (6, 7)


# ─── F0.1: roll ────────────────────────────────────────────────────────────────────────


def test_camera_to_ground_rotation_is_orthonormal() -> None:
    r = camera_to_ground_rotation(np.deg2rad(2.5), np.deg2rad(1.0))
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(r), 1.0)
    # Optical axis of a pitched-down camera points forward and down in the ground frame.
    axis = camera_to_ground_rotation(np.deg2rad(10.0)) @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(axis, [0.0, np.sin(np.deg2rad(10.0)), np.cos(np.deg2rad(10.0))])


def test_roll_tilts_horizon_line_and_keeps_contact_on_ground() -> None:
    """With roll the horizon is a line (right side up for positive roll); contacts stay on Y_g = h."""
    intrinsics = CameraIntrinsics(
        fx=721.5377,
        fy=721.5377,
        cx=609.5593,
        cy=172.8540,
        width=1242,
        height=375,
        distortion_coeffs=np.zeros(5, dtype=np.float64),
    )
    flat = PinholeGeometry(intrinsics, ExtrinsicMountConfig(1.65, np.deg2rad(2.5)))
    rolled = PinholeGeometry(
        intrinsics, ExtrinsicMountConfig(1.65, np.deg2rad(2.5), np.deg2rad(1.0))
    )

    assert np.isclose(flat.get_horizon_v(0.0), flat.get_horizon_v(1241.0))
    assert rolled.get_horizon_v(0.0) > rolled.get_horizon_v(1241.0)
    # Horizon slope ~ tan(roll): ~1 deg over 1241 px -> ~21.7 px.
    drop = rolled.get_horizon_v(0.0) - rolled.get_horizon_v(1241.0)
    assert abs(drop - 1241.0 * np.tan(np.deg2rad(1.0))) < 0.5

    for u in (50.0, 609.5593, 1200.0):
        z_c, _ = rolled.compute_ground_distances(300.0, u)
        p_ground = rolled.camera_to_ground(rolled.deproject_pixel_with_depth(u, 300.0, z_c))
        assert abs(p_ground[1] - 1.65) < 1e-6

    # Roll does not change the depth at the principal column.
    assert np.isclose(
        flat.compute_ground_distances(300.0)[0],
        rolled.compute_ground_distances(300.0)[0],
        atol=1e-3,
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
    assert geom.extrinsics.roll_rad == 0.0


def test_yaml_optional_roll() -> None:
    cfg = (
        "intrinsics:\n  fx: 700.0\n  fy: 700.0\n  cx: 320.0\n  cy: 240.0\n"
        "  width: 640\n  height: 480\n"
        "extrinsics:\n  camera_height_m: 1.5\n  pitch_deg: 3.0\n  roll_deg: -1.5\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(cfg)
        tmp_path = Path(tmp.name)
    try:
        _, extr = load_camera_config_yaml(tmp_path)
    finally:
        tmp_path.unlink()
    assert np.isclose(np.rad2deg(extr.roll_rad), -1.5)


# ─── F0.1: undistort_points ───────────────────────────────────────────────────────────


def test_undistort_points_identity_without_distortion(kitti_geometry: PinholeGeometry) -> None:
    rectifier = ImageRectifier(kitti_geometry.intrinsics)
    pts = np.array([[10.0, 20.0], [609.5593, 172.854], [1200.0, 370.0]])
    np.testing.assert_allclose(rectifier.undistort_points(pts), pts, atol=1e-6)
    assert rectifier.undistort_points(np.empty((0, 2))).shape == (0, 2)


def test_undistort_points_consistent_with_remap_lut() -> None:
    """map_x/map_y give the raw pixel sampled at each rectified pixel; undistort_points inverts it."""
    intrinsics = CameraIntrinsics(
        fx=721.5377,
        fy=721.5377,
        cx=609.5593,
        cy=172.8540,
        width=1242,
        height=375,
        distortion_coeffs=np.array([-0.15, 0.03, 0.001, -0.0005, 0.0]),
    )
    rectifier = ImageRectifier(intrinsics)
    rect_uv = np.array([[100.0, 60.0], [620.0, 180.0], [1100.0, 320.0], [300.0, 350.0]])
    cols, rows = rect_uv[:, 0].astype(int), rect_uv[:, 1].astype(int)
    raw_uv = np.stack([rectifier.map_x[rows, cols], rectifier.map_y[rows, cols]], axis=1)
    recovered = rectifier.undistort_points(raw_uv.astype(np.float64))
    np.testing.assert_allclose(recovered, rect_uv, atol=0.05)


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
