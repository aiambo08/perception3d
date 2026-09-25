"""LiDAR projection, road mask and disparity-vs-1/Z sanity metrics (synthetic geometry)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from percepcion3d.depth.depth_trt import DepthMap
from percepcion3d.depth.preprocess import DepthResize
from percepcion3d.eval.depth_sanity import (
    affine_abs_rel,
    evaluate_depth_map,
    road_mask,
    spearman_vs_inverse_depth,
)
from percepcion3d.eval.lidar import (
    LidarProjector,
    ProjectedLidar,
    load_velodyne_bin,
    nearest_per_pixel,
)

H, W = 40, 120
FX = FY = 100.0
CX, CY = 60.0, 20.0
P = np.array([[FX, 0, CX, 0], [0, FY, CY, 0], [0, 0, 1, 0]], dtype=np.float64)
# KITTI-like velo→cam: x_cam = -y_velo, y_cam = -z_velo, z_cam = x_velo
T_VELO_CAM = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def _ground_points(n: int = 400, h_cam: float = 1.65, seed: int = 0) -> NDArray[np.float64]:
    """Velodyne points on the ground plane in front of the car (x forward, z up)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(4.0, 60.0, n)  # forward
    y = rng.uniform(-8.0, 8.0, n)  # left
    z = np.full(n, -h_cam)  # ground (camera at the origin, h above)
    return np.stack([x, y, z], axis=1)


def test_projector_matches_pinhole_and_filters_outside() -> None:
    proj = LidarProjector(P, T_VELO_CAM, (H, W))
    pts = np.array(
        [
            [10.0, 0.0, -1.65],  # ahead, on the ground → below the principal point
            [10.0, 0.0, 0.0],  # optical axis → (cx, cy)
            [-5.0, 0.0, 0.0],  # behind the camera → dropped
            [1.0, 50.0, 0.0],  # far left → outside the image → dropped
        ]
    )
    out = proj.project(pts)
    assert len(out) == 2
    np.testing.assert_allclose(out.z, [10.0, 10.0])
    np.testing.assert_allclose(out.u, [CX, CX])
    np.testing.assert_allclose(out.v, [CY + FY * 1.65 / 10.0, CY])


def test_projector_rejects_bad_shapes_and_reads_kitti_calib(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        LidarProjector(np.eye(3), np.eye(4), (H, W))
    cam = tmp_path / "calib_cam_to_cam.txt"
    cam.write_text(
        "calib_time: 09-Jan-2012 13:57:47\n"
        "R_rect_00: 1 0 0 0 1 0 0 0 1\n"
        f"S_rect_02: {W} {H}\n"
        "P_rect_02: " + " ".join(str(x) for x in P.ravel()) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "calib_velo_to_cam.txt").write_text(
        "R: 0 -1 0 0 0 -1 1 0 0\nT: 0 0 0\n", encoding="utf-8"
    )
    proj = LidarProjector.from_kitti_raw(tmp_path)
    assert proj.image_hw == (H, W)
    np.testing.assert_allclose(proj.velo_to_rect, T_VELO_CAM)
    np.testing.assert_allclose(proj.p_rect, P)

    bin_path = tmp_path / "0000000000.bin"
    np.array([[1, 2, 3, 0.5], [4, 5, 6, 0.1]], dtype=np.float32).tofile(bin_path)
    pts = load_velodyne_bin(bin_path)
    assert pts.shape == (2, 4) and pts.dtype == np.float32
    with pytest.raises(ValueError):
        np.zeros(5, np.float32).tofile(tmp_path / "bad.bin")
        load_velodyne_bin(tmp_path / "bad.bin")


def test_nearest_per_pixel_keeps_closest_return() -> None:
    proj = ProjectedLidar(
        u=np.array([3.2, 3.7, 3.9, 10.0]),
        v=np.array([5.1, 5.6, 5.2, 7.0]),
        z=np.array([20.0, 8.0, 12.0, 30.0]),
    )
    near = nearest_per_pixel(proj, (H, W))
    assert len(near) == 2
    assert sorted(near.z.tolist()) == [8.0, 30.0]


def test_road_mask_below_horizon_minus_boxes() -> None:
    m = road_mask((H, W), horizon_v=CY, horizon_margin_px=4.0, bottom_margin_px=2.0)
    assert not m[: int(CY) + 4].any()
    assert m[int(CY) + 4 : H - 2].all()
    assert not m[H - 2 :].any()
    boxes = np.array([[10.0, 25.0, 20.0, 35.0]])
    m2 = road_mask((H, W), CY, boxes, horizon_margin_px=4.0)
    assert not m2[25:35, 10:20].any() and m2[25:35, 21:].all()


def test_spearman_and_affine_fit_recover_synthetic_relation() -> None:
    rng = np.random.default_rng(1)
    z = rng.uniform(5.0, 60.0, 500)
    s, t = 40.0, 0.3
    pred = s / z + t + rng.normal(0, 1e-3, z.shape)
    rho = spearman_vs_inverse_depth(pred, z)
    assert rho > 0.999
    s_hat, t_hat, abs_rel = affine_abs_rel(pred, z)
    assert s_hat == pytest.approx(s, rel=1e-2)
    assert t_hat == pytest.approx(t, abs=2e-2)
    assert abs_rel < 0.02
    # anti-correlated (a *depth* fed where disparity is expected) → ρ ≈ −1
    assert spearman_vs_inverse_depth(z, z) < -0.99
    assert np.isnan(spearman_vs_inverse_depth(np.ones(2), np.ones(2)))


def test_evaluate_depth_map_on_synthetic_ground_plane() -> None:
    """Perfect disparity (1/Z of the ground plane at every net pixel) → ρ ≈ 1 after resize."""
    proj = LidarProjector(P, T_VELO_CAM, (H, W))
    lidar = nearest_per_pixel(proj.project(_ground_points()), (H, W))
    assert len(lidar) > 100

    net_h, net_w = 28, 84  # coarser than the frame, anisotropic on purpose
    resize = DepthResize(src_h=H, src_w=W, dst_h=net_h, dst_w=net_w)
    # Ground plane: Z = fy * h / (v - cy) at frame row v; evaluate at each net-pixel centre.
    v_net = (np.arange(net_h, dtype=np.float64) + 0.5) / resize.scale_y
    with np.errstate(divide="ignore"):
        inv_z_rows = np.clip(v_net - CY, 1e-3, None) / (FY * 1.65)
    disparity = (2.5 * inv_z_rows + 0.7)[:, None].repeat(net_w, axis=1).astype(np.float32)
    depth = DepthMap(0, 0, disparity, "relative_disparity", resize)

    res = evaluate_depth_map(depth, lidar, road_mask((H, W), CY, horizon_margin_px=2.0))
    assert res.n > 100
    assert res.spearman > 0.97  # quantisation by the coarser net grid costs a little
    assert res.affine_scale == pytest.approx(2.5, rel=0.1)
    assert res.abs_rel < 0.1

    with pytest.raises(ValueError, match="mask"):
        evaluate_depth_map(depth, lidar, np.ones((H + 1, W), bool))
