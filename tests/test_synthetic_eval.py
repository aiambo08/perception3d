from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from percepcion3d.camera.calibration import CameraIntrinsics, ExtrinsicMountConfig
from percepcion3d.camera.geometry import PinholeGeometry
from percepcion3d.eval.kitti import (
    depth_metrics_by_bin,
    format_depth_metrics,
    load_kitti_tracking_labels,
    parse_kitti_tracking_line,
)
from percepcion3d.sim.synthetic import (
    SyntheticNoise,
    SyntheticObject,
    SyntheticScene,
    default_highway_scene,
)


@pytest.fixture
def geo() -> PinholeGeometry:
    intr = CameraIntrinsics(
        fx=721.5377, fy=721.5377, cx=609.5593, cy=172.854, width=1242, height=375
    )
    return PinholeGeometry(
        intr, ExtrinsicMountConfig(camera_height_m=1.65, pitch_rad=np.deg2rad(2.5))
    )


def _noiseless(
    geo: PinholeGeometry, objects: list[SyntheticObject], **kw: object
) -> SyntheticScene:
    return SyntheticScene(
        geo,
        objects,
        noise=SyntheticNoise(box_px=0.0, inv_depth_rel=0.0),
        **kw,  # type: ignore[arg-type]
    )


# ─── Synthetic scene ─────────────────────────────────────────────────────────


def test_contact_row_is_consistent_with_ground_geometry(geo: PinholeGeometry) -> None:
    scene = _noiseless(geo, [SyntheticObject(1, "car", x0_m=1.0, z0_m=20.0)])
    f = scene.frame(0)
    assert len(f) == 1
    u_mid = 0.5 * (f.boxes[0, 0] + f.boxes[0, 2])
    hit = geo.ground_hits(np.array([u_mid]), f.contact_row)
    assert hit.valid[0]
    # The footprint centre lies on the ground plane: inverting its pixel gives back the GT depth.
    z_c, d_long = geo.compute_ground_distances(float(f.contact_row[0]), u=None)
    assert d_long == pytest.approx(20.0, abs=1e-6)
    assert f.gt_xyz_camera[0, 2] == pytest.approx(z_c, abs=1e-6)
    np.testing.assert_allclose(f.gt_xz_ground[0], [1.0, 20.0])
    # The box bottom edge is at/below the footprint centre row (near corners project lower).
    assert f.boxes[0, 3] >= f.contact_row[0] - 1e-9
    assert not f.truncated[0]


def test_constant_velocity_and_ego_compensation(geo: PinholeGeometry) -> None:
    obj = SyntheticObject(1, "car", x0_m=0.0, z0_m=30.0, vz_mps=5.0)
    scene = _noiseless(geo, [obj], fps=10.0, ego_velocity_mps=(0.0, 8.0))
    np.testing.assert_allclose(scene.relative_velocity(obj), [0.0, -3.0])
    f0, f5 = scene.frame(0), scene.frame(5)
    assert f5.t_ns == 500_000_000
    assert f5.gt_xz_ground[0, 1] == pytest.approx(30.0 - 3.0 * 0.5)
    assert f5.contact_row[0] > f0.contact_row[0]  # approaching → lower in the image
    np.testing.assert_allclose(f5.gt_vel_ground[0], [0.0, -3.0])


def test_inverse_depth_is_affine_in_true_inverse_depth(geo: PinholeGeometry) -> None:
    objs = [SyntheticObject(i, "car", x0_m=0.0, z0_m=z) for i, z in enumerate((8.0, 15.0, 40.0))]
    scene = _noiseless(geo, objs, disparity_scale=3.0, disparity_shift=0.25)
    f = scene.frame(0)
    expected = 3.0 / f.gt_xyz_camera[:, 2] + 0.25
    np.testing.assert_allclose(f.inv_depth, expected)


def test_determinism_and_noise(geo: PinholeGeometry) -> None:
    a = default_highway_scene(geo, seed=3)
    b = default_highway_scene(geo, seed=3)
    c = default_highway_scene(geo, seed=4)
    fa, fb, fc = a.frame(7), b.frame(7), c.frame(7)
    np.testing.assert_array_equal(fa.boxes, fb.boxes)
    np.testing.assert_array_equal(fa.inv_depth, fb.inv_depth)
    assert not np.array_equal(fa.boxes, fc.boxes)
    assert len(fa) >= 2
    assert fa.boxes.shape == (len(fa), 4)
    assert np.all(fa.boxes[:, 0] <= fa.boxes[:, 2]) and np.all(fa.boxes[:, 1] <= fa.boxes[:, 3])
    assert np.all(fa.boxes >= 0.0)


def test_objects_behind_camera_or_off_image_are_dropped(geo: PinholeGeometry) -> None:
    scene = _noiseless(
        geo,
        [
            SyntheticObject(1, "car", x0_m=0.0, z0_m=-5.0),
            SyntheticObject(2, "car", x0_m=200.0, z0_m=20.0),
            SyntheticObject(3, "car", x0_m=0.0, z0_m=20.0),
        ],
    )
    f = scene.frame(0)
    assert f.track_ids.tolist() == [3]
    assert f.classes == ["car"]


def test_ground_inv_depth_map_shape_and_horizon(geo: PinholeGeometry) -> None:
    scene = _noiseless(geo, [], disparity_scale=2.0)
    m = scene.ground_inv_depth_map()
    assert m.shape == (375, 1242)
    v_h = int(np.floor(geo.get_horizon_v()))
    assert np.all(np.isnan(m[: v_h - 1]))
    col = m[v_h + 5 :, 600]
    assert np.all(np.isfinite(col)) and np.all(np.diff(col) > 0)  # closer rows → larger 1/Z


# ─── KITTI tracking labels + metrics ─────────────────────────────────────────

_LINE = "12 3 Car 0.00 1 -1.57 614.24 181.78 727.31 284.77 1.57 1.73 4.15 1.00 1.75 13.22 -1.62"


def test_parse_kitti_tracking_line() -> None:
    lab = parse_kitti_tracking_line(_LINE)
    assert (lab.frame, lab.track_id, lab.obj_type) == (12, 3, "Car")
    assert lab.bbox == (614.24, 181.78, 727.31, 284.77)
    assert lab.dims_hwl == (1.57, 1.73, 4.15)
    assert lab.depth_m == 13.22
    assert lab.occluded == 1
    with pytest.raises(ValueError):
        parse_kitti_tracking_line("0 1 Car")


def test_load_kitti_tracking_labels_groups_and_filters(tmp_path: Path) -> None:
    p = tmp_path / "0000.txt"
    p.write_text(
        _LINE + "\n"
        "12 -1 DontCare -1 -1 -10 0 0 10 10 -1 -1 -1 -1000 -1000 -1000 -10\n"
        "13 3 Car 0.00 0 -1.57 600 180 720 280 1.57 1.73 4.15 1.00 1.75 12.80 -1.62\n"
        "13 9 Pedestrian 0.00 0 0.1 100 150 130 250 1.8 0.6 0.7 -6.0 1.7 20.0 0.0\n"
    )
    by_frame = load_kitti_tracking_labels(p)
    assert sorted(by_frame) == [12, 13]
    assert len(by_frame[12]) == 1  # DontCare removed
    assert [lab.track_id for lab in by_frame[13]] == [3, 9]
    only_cars = load_kitti_tracking_labels(p, keep_types=("Car",))
    assert [lab.track_id for lab in only_cars[13]] == [3]


def test_depth_metrics_by_bin_values() -> None:
    gt = np.array([5.0, 8.0, 15.0, 25.0, 35.0, np.nan, 60.0])
    pred = np.array([5.5, 7.2, 15.0, 30.0, np.nan, 10.0, 66.0])
    rows = depth_metrics_by_bin(pred, gt, bins_m=(0.0, 10.0, 20.0, 30.0, 50.0))
    assert [r.count for r in rows] == [2, 1, 1, 0]
    r0 = rows[0]
    assert r0.abs_rel == pytest.approx((0.5 / 5.0 + 0.8 / 8.0) / 2)
    assert r0.rmse_m == pytest.approx(np.sqrt((0.25 + 0.64) / 2))
    assert r0.bias_m == pytest.approx((0.5 - 0.8) / 2)
    assert rows[1].abs_rel == 0.0 and rows[1].rmse_m == 0.0
    assert rows[2].bias_m == pytest.approx(5.0)
    assert np.isnan(rows[3].abs_rel)
    table = format_depth_metrics(rows)
    assert "0-10" in table and "30-50" in table

    with pytest.raises(ValueError):
        depth_metrics_by_bin([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        depth_metrics_by_bin([1.0], [1.0], bins_m=(10.0, 0.0))
