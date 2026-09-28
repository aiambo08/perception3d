"""F7 — ``scripts/run_pipeline.py`` end to end on a synthetic KITTI tracking layout (CPU)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_pipeline.py"
W, H = 1242, 375
N_FRAMES = 24


def _write_layout(root: Path) -> None:
    (root / "image_02" / "0000").mkdir(parents=True)
    (root / "label_02").mkdir()
    (root / "calib").mkdir()
    (root / "oxts").mkdir()
    img = np.zeros((H, W, 3), dtype=np.uint8)
    lines = []
    for k in range(N_FRAMES):
        cv2.imwrite(str(root / "image_02" / "0000" / f"{k:06d}.png"), img)
        # A car approaching along the optical axis: box grows and its base line sinks.
        z = 30.0 - 0.8 * k
        v = 172.854 + 721.5377 * 1.65 / z
        w = 721.5377 * 1.8 / z
        lines.append(
            f"{k} 1 Car 0 0 0 {609.6 - w / 2:.2f} {v - 1.5 * w / 1.8:.2f} "
            f"{609.6 + w / 2:.2f} {v:.2f} 1.5 1.8 4.2 0 1.65 {z:.2f} -1.57"
        )
    (root / "label_02" / "0000.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    p2 = "P2: 721.5377 0 609.5593 44.857 0 721.5377 172.854 0.2163 0 0 1 0.00274\n"
    (root / "calib" / "0000.txt").write_text("P0: 1 0 0 0 0 1 0 0 0 0 1 0\n" + p2, encoding="utf-8")
    # 30 OXTS fields: lat lon alt roll pitch yaw vn ve vf vl vu ax ay az af al au wx wy wz wf wl wu ...
    oxts = " ".join(["49.0", "8.4", "112.0", "0", "0", "0.5", "0", "0"] + ["0"] * 22)
    (root / "oxts" / "0000.txt").write_text((oxts + "\n") * N_FRAMES, encoding="utf-8")


def _run(*extra: str) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *extra],
        capture_output=True,
        text=True,
        timeout=240,
        cwd=ROOT,
    )
    assert proc.returncode in (0, 1), proc.stderr  # 1 = a CPU DoD verdict failed, not a crash
    return proc


@pytest.fixture(scope="module")
def kitti_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("kt")
    _write_layout(root)
    return root


def test_runner_paced_loop_gt_boxes_jsonl(kitti_root: Path, tmp_path: Path) -> None:
    out, jsonl = tmp_path / "f7.json", tmp_path / "tel.jsonl"
    proc = _run(
        "--kitti-tracking",
        str(kitti_root),
        "--seq",
        "0000",
        "--boxes",
        "gt",
        "--hz",
        "0",
        "--telemetry",
        "jsonl",
        "--jsonl",
        str(jsonl),
        "--json",
        str(out),
    )
    rep = json.loads(out.read_text(encoding="utf-8"))
    st = rep["stats"]
    assert rep["delivery"] == "paced_loop" and rep["image_hw"] == [H, W]
    assert st["frames"] == N_FRAMES and st["dropped"] == 0
    assert st["depth"]["maps"] == 0 and rep["verdicts"]["depth_hz"] is None
    assert st["e2e_alert_ms"]["n"] == N_FRAMES and st["e2e_alert_ms"]["p99"] > 0
    assert rep["telemetry"]["backend"] == "JsonlBackend"
    assert rep["telemetry"]["logged"] == N_FRAMES and rep["telemetry"]["errors"] == 0
    assert rep["verdicts"]["vram_peak"] is None  # no GPU requested → not measured
    recs = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert [r["frame_id"] for r in recs] == list(range(N_FRAMES))
    assert all(len(r["boxes"]) == 1 and r["classes"] == ["car"] for r in recs)
    assert any(r["tracks"] for r in recs[3:])  # tracker confirmed the approaching car
    assert "capture→alert ms" in proc.stdout and "DoD:" in proc.stdout


def test_runner_capture_thread_loop_oxts_and_max_frames(kitti_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "f7_ct.json"
    _run(
        "--kitti-tracking",
        str(kitti_root),
        "--seq",
        "0000",
        "--boxes",
        "gt",
        "--ego",
        "oxts",
        "--loop",
        "--hz",
        "400",
        "--max-frames",
        "60",
        "--capture-thread",
        "--telemetry",
        "null",
        "--json",
        str(out),
    )
    rep = json.loads(out.read_text(encoding="utf-8"))
    st = rep["stats"]
    assert rep["delivery"] == "capture_thread"
    assert 0 < st["frames"] <= 60 and st["frames"] + st["dropped"] == 60
    assert st["e2e_alert_ms"]["n"] == st["frames"]
    assert rep["telemetry"]["backend"] == "NullBackend"
    assert rep["args"]["ego"] == "oxts" and rep["args"]["loop"] is True


def test_runner_rejects_ambiguous_sources(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--video", "a.mp4", "--kitti-raw", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=ROOT,
    )
    assert proc.returncode == 2 and "exactly one of" in proc.stderr
