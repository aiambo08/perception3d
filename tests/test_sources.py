from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from percepcion3d.io.sources import (
    FrameSource,
    ImageSequenceSource,
    KittiSequenceSource,
    VideoFileSource,
    parse_kitti_timestamps,
    parse_oxts_line,
)

_CALIB = """calib_time: 09-Jan-2012 13:57:47
S_rect_02: 1.242000e+03 3.750000e+02
P_rect_02: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03
"""

_OXTS = (
    "49.011212804408 8.4228850417969 112.83492279053 0.022447 1e-05 -1.2219096732051 "
    "-3.3256321642087 1.1384311271831 3.5147680214713 0.037625160413037 -0.03878884255623 "
    "-0.2560652063917 -0.62926763262956 9.8035359531966 -0.34681738614616 -0.62926763262956 "
    "9.7843523543958 0.03848729963693 -0.0016399994794 0.023173007447776 "
    "0.036986999288349 -0.0097636947478502 0.023173007447776 0.015 0.02 4 11 6 6 6\n"
)


def _write_kitti_drive(root: Path, n: int, with_oxts: bool = True) -> Path:
    date = root / "2011_09_26"
    drive = date / "2011_09_26_drive_0001_sync"
    img_dir = drive / "image_02" / "data"
    img_dir.mkdir(parents=True)
    (date / "calib_cam_to_cam.txt").write_text(_CALIB)
    stamps = []
    for i in range(n):
        img = np.full((8, 12, 3), i * 10, dtype=np.uint8)
        assert cv2.imwrite(str(img_dir / f"{i:010d}.png"), img)
        stamps.append(f"2011-09-26 13:02:{25 + i // 10:02d}.{(i % 10) * 100_000_000:09d}")
    (drive / "image_02" / "timestamps.txt").write_text("\n".join(stamps) + "\n")
    if with_oxts:
        oxts_dir = drive / "oxts" / "data"
        oxts_dir.mkdir(parents=True)
        for i in range(n):
            (oxts_dir / f"{i:010d}.txt").write_text(_OXTS)
    return drive


# ─── KITTI parsing ───────────────────────────────────────────────────────────


def test_parse_kitti_timestamps_nanosecond_precision(tmp_path: Path) -> None:
    p = tmp_path / "timestamps.txt"
    p.write_text("2011-09-26 13:02:25.964389445\n2011-09-26 13:02:26.067835000\n\n")
    t = parse_kitti_timestamps(p)
    assert len(t) == 2
    assert t[0] % 1_000_000_000 == 964389445
    assert t[1] - t[0] == 103_445_555


def test_parse_oxts_line_fields() -> None:
    rec = parse_oxts_line(_OXTS)
    assert rec.lat == pytest.approx(49.011212804408)
    assert rec.vf == pytest.approx(3.5147680214713)
    assert rec.wu == pytest.approx(0.023173007447776)
    assert rec.speed_mps == pytest.approx(np.hypot(rec.vf, rec.vl))
    with pytest.raises(ValueError):
        parse_oxts_line("1 2 3")


def test_kitti_sequence_source_reads_frames_calib_and_oxts(tmp_path: Path) -> None:
    drive = _write_kitti_drive(tmp_path, n=5)
    src = KittiSequenceSource(drive)
    assert isinstance(src, FrameSource)
    assert len(src) == 5
    assert src.intrinsics is not None and src.intrinsics.fx == pytest.approx(721.5377)
    assert src.fps is not None and src.fps == pytest.approx(10.0, rel=1e-3)
    assert src.oxts is not None and len(src.oxts) == 5

    frames = list(src)
    assert [f.frame_id for f in frames] == list(range(5))
    assert frames[0].img.shape == (8, 12, 3) and frames[0].img.dtype == np.uint8
    assert int(frames[3].img[0, 0, 0]) == 30
    assert frames[1].t_capture_ns - frames[0].t_capture_ns == 100_000_000
    src.close()


def test_kitti_sequence_source_max_frames_and_missing_oxts(tmp_path: Path) -> None:
    drive = _write_kitti_drive(tmp_path, n=4, with_oxts=False)
    src = KittiSequenceSource(drive, max_frames=2)
    assert len(src) == 2
    assert src.oxts is None
    with pytest.raises(FileNotFoundError):
        KittiSequenceSource(tmp_path / "nope")


def test_image_sequence_source_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ImageSequenceSource([], [])
    with pytest.raises(ValueError):
        ImageSequenceSource([tmp_path / "a.png"], [1, 2])
    src = ImageSequenceSource([tmp_path / "missing.png"], [0])
    assert src.fps is None
    with pytest.raises(RuntimeError):
        list(src)


# ─── Video file ──────────────────────────────────────────────────────────────


def test_video_file_source_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 25.0, (32, 16))
    if not writer.isOpened():
        pytest.skip("OpenCV build has no MJPG writer")
    for i in range(6):
        writer.write(np.full((16, 32, 3), 40 * i, dtype=np.uint8))
    writer.release()

    src = VideoFileSource(path)
    assert src.fps == pytest.approx(25.0)
    frames = list(src)
    src.close()
    assert len(frames) == 6
    assert frames[0].img.shape == (16, 32, 3)
    assert frames[1].t_capture_ns - frames[0].t_capture_ns == 40_000_000
    assert abs(int(frames[2].img[8, 16, 1]) - 80) <= 8  # MJPG is lossy

    with pytest.raises(FileNotFoundError):
        VideoFileSource(tmp_path / "none.avi")
