"""F4 KITTI-tracking evaluation harness on a synthetic sequence (CPU, no dataset)."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from percepcion3d.camera.calibration import CameraIntrinsics, ExtrinsicMountConfig
from percepcion3d.camera.geometry import PinholeGeometry
from percepcion3d.depth.depth_trt import DepthMap
from percepcion3d.depth.fusion import MetricFusionStage, load_fusion_config, load_solver_configs
from percepcion3d.eval.fusion_kitti import (
    KittiTrackingSequence,
    gt_near_face_depth_m,
    is_scored,
    labels_to_boxes,
    parse_kitti_tracking_calib,
    run_fusion_kitti,
)
from percepcion3d.eval.kitti import KittiTrackLabel, parse_kitti_tracking_line
from percepcion3d.runtime.buffer import FrameStamped
from percepcion3d.sim.synthetic import SyntheticNoise, SyntheticObject, SyntheticScene

ROOT = Path(__file__).resolve().parents[1]
INTR = CameraIntrinsics(fx=721.5377, fy=721.5377, cx=609.5593, cy=172.854, width=1242, height=375)
EXTR = ExtrinsicMountConfig(camera_height_m=1.65, pitch_rad=float(np.deg2rad(2.5)))


def _label(
    z: float, ry: float, hwl: tuple[float, float, float] = (1.5, 1.8, 4.2)
) -> KittiTrackLabel:
    return KittiTrackLabel(0, 0, "Car", 0.0, 0, 0.0, (0, 0, 100, 60), hwl, (0.0, 1.65, z), ry)


def test_near_face_depth_uses_kitti_rotation_convention() -> None:
    assert gt_near_face_depth_m(_label(20.0, math.pi / 2)) == pytest.approx(20.0 - 2.1)
    assert gt_near_face_depth_m(_label(20.0, 0.0)) == pytest.approx(20.0 - 0.9)
    assert gt_near_face_depth_m(_label(20.0, -math.pi / 2)) == pytest.approx(20.0 - 2.1)


def test_is_scored_filters_type_truncation_occlusion_height() -> None:
    ok = _label(20.0, 1.5)
    assert is_scored(ok)
    assert not is_scored(replace(ok, obj_type="Van"))
    assert not is_scored(replace(ok, truncated=0.1))
    assert not is_scored(replace(ok, occluded=2))
    assert not is_scored(replace(ok, bbox=(0.0, 0.0, 20.0, 20.0)))
    assert is_scored(replace(ok, occluded=2), max_occluded=2)


def test_labels_to_boxes_maps_types_and_drops_dontcare() -> None:
    labs = [
        parse_kitti_tracking_line("0 1 Car 0 0 0 10 20 110 80 1.5 1.8 4.2 0 1.65 20 1.5"),
        parse_kitti_tracking_line("0 2 Pedestrian 0 0 0 300 20 330 90 1.7 0.6 0.5 2 1.65 12 0"),
        parse_kitti_tracking_line(
            "0 -1 DontCare -1 -1 -10 500 20 530 40 -1 -1 -1 -1000 -1000 -1000 -10"
        ),
    ]
    boxes, classes = labels_to_boxes(labs)
    assert boxes.shape == (2, 4) and classes == ["car", "pedestrian"]


def _write_tracking_layout(root: Path, n_frames: int, label_lines: list[str]) -> None:
    (root / "image_02" / "0000").mkdir(parents=True)
    (root / "label_02").mkdir()
    (root / "calib").mkdir()
    img = np.zeros((INTR.height, INTR.width, 3), dtype=np.uint8)
    for k in range(n_frames):
        cv2.imwrite(str(root / "image_02" / "0000" / f"{k:06d}.png"), img)
    (root / "label_02" / "0000.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    p2 = f"P2: {INTR.fx} 0 {INTR.cx} 44.857 0 {INTR.fy} {INTR.cy} 0.2163 0 0 1 0.00274\n"
    (root / "calib" / "0000.txt").write_text("P0: 1 0 0 0 0 1 0 0 0 0 1 0\n" + p2, encoding="utf-8")


def test_tracking_sequence_layout_and_calib(tmp_path: Path) -> None:
    _write_tracking_layout(tmp_path, 3, ["0 1 Car 0 0 0 10 20 110 80 1.5 1.8 4.2 0 1.65 20 1.5"])
    seq = KittiTrackingSequence(tmp_path, "0000", max_frames=2)
    k = seq.intrinsics()
    assert (k.fx, k.cx, k.cy) == pytest.approx((INTR.fx, INTR.cx, INTR.cy))
    assert parse_kitti_tracking_calib(tmp_path / "calib" / "0000.txt").fy == pytest.approx(INTR.fy)
    src = seq.source()
    frames = list(src)
    assert len(frames) == 2 and frames[1].t_capture_ns == 100_000_000
    assert set(seq.labels()) == {0}
    with pytest.raises(FileNotFoundError):
        KittiTrackingSequence(tmp_path, "0001")


def _synthetic_labels(scene: SyntheticScene, n_frames: int) -> dict[int, list[KittiTrackLabel]]:
    """Convert synthetic frames to KITTI labels: centre = near face + l/2, r_y = π/2."""
    out: dict[int, list[KittiTrackLabel]] = {}
    objs = {o.track_id: o for o in scene.objects}
    for k in range(n_frames):
        f = scene.frame(k)
        labs: list[KittiTrackLabel] = []
        for i in range(len(f)):
            o = objs[int(f.track_ids[i])]
            x, y, _ = f.gt_xyz_camera[i]
            typ = "Car" if o.cls == "car" else "Pedestrian"
            labs.append(
                KittiTrackLabel(
                    frame=k,
                    track_id=o.track_id,
                    obj_type=typ,
                    truncated=1.0 if f.truncated[i] else 0.0,
                    occluded=0,
                    alpha=0.0,
                    bbox=(
                        float(f.gt_boxes[i, 0]),
                        float(f.gt_boxes[i, 1]),
                        float(f.gt_boxes[i, 2]),
                        float(f.gt_boxes[i, 3]),
                    ),
                    dims_hwl=(o.height_m, o.width_m, o.length_m),
                    location=(float(x), float(y), float(f.gt_z_front_m[i]) + o.length_m / 2),
                    rotation_y=math.pi / 2,
                )
            )
        out[k] = labs
    return out


def test_run_fusion_kitti_synthetic_scores_cars_and_passes_dod() -> None:
    geo = PinholeGeometry(INTR, EXTR)
    objs = [
        SyntheticObject(1, "car", x0_m=0.5, z0_m=12.0, vz_mps=1.0, height_m=1.52),
        SyntheticObject(2, "car", x0_m=-3.5, z0_m=40.0, vz_mps=-3.0, height_m=1.45),
        SyntheticObject(3, "pedestrian", x0_m=4.0, z0_m=18.0, width_m=0.6, height_m=1.75,
                        length_m=0.5, silhouette_frac=0.4),
    ]  # fmt: skip
    scene = SyntheticScene(
        geo, objs, noise=SyntheticNoise(box_px=0.5, inv_depth_rel=0.01), fps=10.0,
        disparity_scale=3.0, disparity_shift=0.2,
    )  # fmt: skip
    n = 25
    labels = _synthetic_labels(scene, n)
    frames = [
        FrameStamped(k, scene.frame(k).t_ns, np.zeros((INTR.height, INTR.width, 3), np.uint8))
        for k in range(n)
    ]

    def depth(fs: FrameStamped) -> DepthMap | None:
        f = scene.frame(fs.frame_id)
        inv, rs = scene.dense_inv_depth_map(f, (140, 462), pixel_noise_rel=0.01)
        return DepthMap(
            fs.frame_id, fs.t_capture_ns, inv.astype(np.float32), "relative_disparity", rs
        )

    cfg = load_fusion_config(ROOT / "configs" / "fusion.yaml")
    road, aff, pit = load_solver_configs(ROOT / "configs" / "fusion.yaml")
    stage = MetricFusionStage(INTR, geo, cfg, road, aff, pit)
    res = run_fusion_kitti(frames, labels, stage, depth, timer_ms=lambda: 0.0)

    assert res.n_frames == n and res.n_frames_with_depth == n
    assert res.n_boxes_fed > res.n_scored_labels  # pedestrian fed but not scored
    assert len(res.samples) == res.n_scored_labels
    assert all(s.track_id in (1, 2) for s in res.samples)
    dod = res.dod()
    assert dod["n_0_30m"] > 0 and dod["n_30_60m"] > 0
    assert dod["pass"], dod
    assert dod["abs_rel_0_30m"] < 0.05
    rows = res.metrics()["fused (near face)"]
    assert rows[0].count == dod["n_0_30m"]
    assert "ground only" in res.metrics() and "AbsRel" in res.format()
    assert res.affine_s == pytest.approx(3.0, rel=0.05)


def test_run_fusion_kitti_detector_mode_matches_by_iou() -> None:
    geo = PinholeGeometry(INTR, EXTR)
    scene = SyntheticScene(
        geo, [SyntheticObject(1, "car", x0_m=0.5, z0_m=15.0)], noise=SyntheticNoise(0.0, 0.0)
    )
    labels = _synthetic_labels(scene, 3)
    frames = [FrameStamped(k, k * 10**8, np.zeros((4, 4, 3), np.uint8)) for k in range(3)]

    def boxes(fs: FrameStamped) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
        gt = labels[fs.frame_id][0].bbox
        shifted = np.array([[gt[0] + 3, gt[1] + 3, gt[2] + 3, gt[3] + 3]], dtype=np.float64)
        far = np.array([[5.0, 300.0, 60.0, 360.0]])
        return np.vstack([far, shifted]), np.array([0.9, 0.8]), ["car", "car"]

    cfg = load_fusion_config(ROOT / "configs" / "fusion.yaml")
    stage = MetricFusionStage(INTR, geo, cfg)
    res = run_fusion_kitti(frames, labels, stage, None, box_provider=boxes)
    assert res.n_boxes_fed == 6 and len(res.samples) == 3
    assert all(np.isfinite(s.z_cam_m) and np.isnan(s.z_net_m) for s in res.samples)
    assert res.dod()["abs_rel_0_30m"] < 0.1
