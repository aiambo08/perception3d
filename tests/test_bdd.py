"""BDD100K loaders and camera YAML (F2 real-camera path)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from percepcion3d.camera.calibration import load_camera_config_yaml
from percepcion3d.eval.bdd import (
    DEFAULT_BDD_TO_COCO,
    flatten_det_gts,
    load_det_frame_meta,
    load_det_labels,
    load_mot_sequence,
    pick_demo_candidates,
    summary_counts,
)
from percepcion3d.eval.detection import HEIGHT_ONLY, PredBox, recall_by_type


def test_camera_bdd_yaml_estimated_intrinsics() -> None:
    yaml_path = Path(__file__).resolve().parents[1] / "configs" / "camera_bdd.yaml"
    intr, extr = load_camera_config_yaml(yaml_path)
    assert intr.width == 1280 and intr.height == 720
    # fx = width / (2·tan(HFOV/2)) with HFOV ≈ 78°
    expected = 1280.0 / (2.0 * math.tan(math.radians(78.0 / 2.0)))
    assert np.isclose(intr.fx, expected, rtol=1e-3)
    assert np.isclose(intr.fy, expected, rtol=1e-3)
    assert np.isclose(intr.cx, 640.0) and np.isclose(intr.cy, 360.0)
    assert np.isclose(extr.camera_height_m, 1.30)
    assert np.isclose(extr.pitch_rad, 0.0)


def test_load_det_labels_scalabel(tmp_path: Path) -> None:
    payload = [
        {
            "name": "aaa.jpg",
            "attributes": {"weather": "rainy", "scene": "highway", "timeofday": "night"},
            "labels": [
                {
                    "id": "1",
                    "category": "car",
                    "box2d": {"x1": 10, "y1": 20, "x2": 100, "y2": 80},
                },
                {
                    "id": "2",
                    "category": "pedestrian",
                    "box2d": {"x1": 200, "y1": 100, "x2": 240, "y2": 220},
                },
                {"id": "3", "category": "traffic sign"},  # no box2d → skipped
            ],
        },
        {
            "name": "bbb.jpg",
            "attributes": {"weather": "clear", "scene": "city street", "timeofday": "daytime"},
            "labels": [],
        },
    ]
    p = tmp_path / "det_val.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    by_name = load_det_labels(p, keep_categories=("car", "pedestrian", "rider"))
    assert set(by_name) == {"aaa.jpg", "bbb.jpg"}
    assert len(by_name["aaa.jpg"]) == 2
    assert by_name["aaa.jpg"][0].kitti_type == "car"
    assert by_name["aaa.jpg"][0].box == (10.0, 20.0, 100.0, 80.0)

    meta = load_det_frame_meta(p)
    buckets = pick_demo_candidates(meta, limit_per_bucket=3)
    assert "aaa" in buckets["night"]
    assert "aaa" in buckets["rain"]
    assert "aaa" in buckets["highway"]

    gts, name_to_frame = flatten_det_gts(by_name, ["aaa.jpg", "bbb.jpg"])
    assert name_to_frame["aaa.jpg"] == 0
    assert all(g.frame == 0 for g in gts)
    stats = summary_counts(by_name)
    assert stats["n_images"] == 2 and stats["n_boxes"] == 2


def test_bdd_recall_maps_rider_to_person() -> None:
    from percepcion3d.eval.detection import GtBox

    gts = [
        GtBox(0, "pedestrian", (0, 0, 40, 100)),
        GtBox(0, "rider", (100, 0, 140, 100)),
        GtBox(0, "car", (200, 0, 300, 80)),
    ]
    preds = [
        PredBox(0, "person", 0.9, (0, 0, 40, 100)),
        PredBox(0, "person", 0.8, (100, 0, 140, 100)),
        PredBox(0, "car", 0.9, (200, 0, 300, 80)),
    ]
    rows = recall_by_type(preds, gts, type_to_classes=DEFAULT_BDD_TO_COCO, gt_filter=HEIGHT_ONLY)
    assert rows["pedestrian"].n_matched == 1
    assert rows["rider"].n_matched == 1
    assert rows["car"].n_matched == 1


def test_load_mot_sequence(tmp_path: Path) -> None:
    payload = [
        {
            "name": "vid-0000001.jpg",
            "frameIndex": 1,
            "labels": [
                {
                    "id": "7",
                    "category": "car",
                    "box2d": {"x1": 1, "y1": 2, "x2": 30, "y2": 40},
                }
            ],
        },
        {
            "name": "vid-0000000.jpg",
            "frameIndex": 0,
            "labels": [
                {
                    "id": "7",
                    "category": "car",
                    "box2d": {"x1": 2, "y1": 3, "x2": 31, "y2": 41},
                }
            ],
        },
    ]
    p = tmp_path / "vid.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    frames = load_mot_sequence(p)
    assert [f.frame_idx for f in frames] == [0, 1]
    assert frames[0].track_ids == (7,)
    assert frames[1].boxes[0].kitti_type == "car"
