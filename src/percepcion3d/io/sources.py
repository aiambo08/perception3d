"""Frame sources: video files, KITTI raw drives, V4L2 cameras.

Every source yields :class:`~percepcion3d.runtime.buffer.FrameStamped` and
exposes nominal ``fps`` and (when known) ``intrinsics``. Timestamps are
nanoseconds: dataset timestamps for replays, ``time.monotonic_ns`` for live
cameras, so downstream code can compute ``dt`` uniformly.

KITTI raw layout expected by :class:`KittiSequenceSource`::

    <date>/
      calib_cam_to_cam.txt
      <date>_drive_XXXX_sync/
        image_02/data/0000000000.png ...
        image_02/timestamps.txt
        oxts/data/0000000000.txt ...          (optional)
        oxts/timestamps.txt                   (optional)
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np
from numpy.typing import NDArray

from percepcion3d.camera.calibration import CameraIntrinsics, parse_kitti_calib_txt
from percepcion3d.runtime.buffer import FrameStamped


@runtime_checkable
class FrameSource(Protocol):
    """Iterable producer of stamped frames."""

    @property
    def fps(self) -> float | None:
        """Nominal frame rate, ``None`` if unknown/variable."""

    @property
    def intrinsics(self) -> CameraIntrinsics | None:
        """Rectified intrinsics when the source knows them."""

    def __iter__(self) -> Iterator[FrameStamped]: ...

    def close(self) -> None: ...


# ----------------------------------------------------------------------
# Video file
# ----------------------------------------------------------------------


class VideoFileSource:
    """Decode a video file with OpenCV; timestamps are ``frame_idx / fps`` (deterministic).

    ``CAP_PROP_POS_MSEC`` is not used because several backends return 0 or
    jump on B-frames.
    """

    def __init__(self, path: Path | str, intrinsics: CameraIntrinsics | None = None) -> None:
        self._path = Path(path)
        if not self._path.is_file():
            raise FileNotFoundError(f"Video file not found: {self._path}")
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {self._path}")
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._fps: float | None = fps if fps > 0.0 else None
        self._intrinsics = intrinsics
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self) -> float | None:
        return self._fps

    @property
    def intrinsics(self) -> CameraIntrinsics | None:
        return self._intrinsics

    @property
    def frame_count(self) -> int:
        """Container-reported frame count (may be approximate for some codecs)."""
        return self._frame_count

    def __iter__(self) -> Iterator[FrameStamped]:
        period_ns = int(round(1e9 / self._fps)) if self._fps else 0
        idx = 0
        while True:
            ok, img = self._cap.read()
            if not ok:
                return
            yield FrameStamped(frame_id=idx, t_capture_ns=idx * period_ns, img=_as_u8(img))
            idx += 1

    def close(self) -> None:
        self._cap.release()


# ----------------------------------------------------------------------
# Image sequences (generic + KITTI)
# ----------------------------------------------------------------------


class ImageSequenceSource:
    """Replay a list of image files with explicit timestamps."""

    def __init__(
        self,
        paths: Sequence[Path],
        timestamps_ns: Sequence[int],
        intrinsics: CameraIntrinsics | None = None,
        grayscale: bool = False,
    ) -> None:
        if len(paths) != len(timestamps_ns):
            raise ValueError(
                f"paths ({len(paths)}) and timestamps ({len(timestamps_ns)}) differ in length"
            )
        if len(paths) == 0:
            raise ValueError("Empty image sequence")
        self._paths = [Path(p) for p in paths]
        self._t_ns = [int(t) for t in timestamps_ns]
        self._intrinsics = intrinsics
        self._flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        self._fps = _nominal_fps(self._t_ns)

    @property
    def fps(self) -> float | None:
        return self._fps

    @property
    def intrinsics(self) -> CameraIntrinsics | None:
        return self._intrinsics

    @property
    def timestamps_ns(self) -> list[int]:
        return list(self._t_ns)

    def __len__(self) -> int:
        return len(self._paths)

    def __iter__(self) -> Iterator[FrameStamped]:
        for idx, (path, t_ns) in enumerate(zip(self._paths, self._t_ns, strict=True)):
            img = cv2.imread(str(path), self._flag)
            if img is None:
                raise RuntimeError(f"Could not read image: {path}")
            yield FrameStamped(frame_id=idx, t_capture_ns=t_ns, img=_as_u8(img))

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class OxtsRecord:
    """Subset of a KITTI OXTS/INS record relevant to ego-motion.

    Velocities are in the vehicle frame (``vf`` forward, ``vl`` left, ``vu`` up)
    in m/s; angular rates ``wf/wl/wu`` about the same axes in rad/s; ``roll``,
    ``pitch``, ``yaw`` in rad; accelerations in m/s².
    """

    lat: float
    lon: float
    alt: float
    roll: float
    pitch: float
    yaw: float
    vn: float
    ve: float
    vf: float
    vl: float
    vu: float
    ax: float
    ay: float
    az: float
    af: float
    al: float
    au: float
    wx: float
    wy: float
    wz: float
    wf: float
    wl: float
    wu: float

    @property
    def speed_mps(self) -> float:
        return float(np.hypot(self.vf, self.vl))


_OXTS_FIELDS = 23  # numeric fields consumed above; the remaining 7 are status flags


def parse_oxts_line(line: str) -> OxtsRecord:
    vals = line.split()
    if len(vals) < _OXTS_FIELDS:
        raise ValueError(f"OXTS record has {len(vals)} fields, expected >= {_OXTS_FIELDS}")
    nums = [float(v) for v in vals[:_OXTS_FIELDS]]
    return OxtsRecord(*nums)


def parse_kitti_timestamps(path: Path | str) -> list[int]:
    """Parse ``timestamps.txt`` (``YYYY-MM-DD HH:MM:SS.fffffffff``) into epoch nanoseconds."""
    out: list[int] = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            date_part, frac = line.split(".") if "." in line else (line, "0")
            frac = (frac + "000000000")[:9]
            dt = datetime.strptime(date_part, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
            out.append(int(dt.timestamp()) * 1_000_000_000 + int(frac))
    return out


class KittiSequenceSource(ImageSequenceSource):
    """A KITTI raw ``*_drive_XXXX_sync`` folder: images, timestamps, calibration and OXTS."""

    def __init__(
        self,
        drive_dir: Path | str,
        cam_idx: int = 2,
        calib_file: Path | str | None = None,
        max_frames: int | None = None,
    ) -> None:
        drive = Path(drive_dir)
        cam_dir = drive / f"image_{cam_idx:02d}"
        img_dir = cam_dir / "data"
        if not img_dir.is_dir():
            raise FileNotFoundError(f"KITTI image folder not found: {img_dir}")
        paths = sorted(img_dir.glob("*.png")) or sorted(img_dir.glob("*.jpg"))
        t_ns = parse_kitti_timestamps(cam_dir / "timestamps.txt")
        n = min(len(paths), len(t_ns))
        if max_frames is not None:
            n = min(n, max_frames)
        if n == 0:
            raise ValueError(f"No frames in {img_dir}")

        calib = (
            Path(calib_file) if calib_file is not None else drive.parent / "calib_cam_to_cam.txt"
        )
        intrinsics = parse_kitti_calib_txt(calib, cam_idx=cam_idx) if calib.is_file() else None

        super().__init__(paths[:n], t_ns[:n], intrinsics=intrinsics, grayscale=cam_idx < 2)

        self._oxts: list[OxtsRecord] | None = None
        oxts_dir = drive / "oxts" / "data"
        if oxts_dir.is_dir():
            records: list[OxtsRecord] = []
            for p in sorted(oxts_dir.glob("*.txt"))[:n]:
                records.append(parse_oxts_line(p.read_text(encoding="utf-8")))
            if len(records) == n:
                self._oxts = records
        self._drive = drive

    @property
    def drive_dir(self) -> Path:
        return self._drive

    @property
    def oxts(self) -> list[OxtsRecord] | None:
        """One OXTS record per frame, or ``None`` if the drive has no/incomplete OXTS data."""
        return self._oxts


# ----------------------------------------------------------------------
# Live camera (V4L2)
# ----------------------------------------------------------------------


class V4L2Source:
    """Live capture through OpenCV's V4L2 backend (Linux/WSL2 with usbipd).

    Requests MJPG at the given resolution/fps to avoid USB-bandwidth caps of
    raw YUYV; the driver may silently pick another mode, so check ``fps`` and
    the first frame's shape.
    """

    def __init__(
        self,
        device: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: float = 60.0,
        fourcc: str = "MJPG",
        intrinsics: CameraIntrinsics | None = None,
    ) -> None:
        self._cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open V4L2 device {device!r}")
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*fourcc))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        got = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._fps: float | None = got if got > 0.0 else None
        self._intrinsics = intrinsics

    @property
    def fps(self) -> float | None:
        return self._fps

    @property
    def intrinsics(self) -> CameraIntrinsics | None:
        return self._intrinsics

    def __iter__(self) -> Iterator[FrameStamped]:
        idx = 0
        while True:
            ok = self._cap.grab()
            t_ns = time.monotonic_ns()
            if not ok:
                return
            ok, img = self._cap.retrieve()
            if not ok:
                return
            yield FrameStamped(frame_id=idx, t_capture_ns=t_ns, img=_as_u8(img))
            idx += 1

    def close(self) -> None:
        self._cap.release()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _as_u8(img: cv2.typing.MatLike) -> NDArray[np.uint8]:
    return np.asarray(img, dtype=np.uint8)


def _nominal_fps(t_ns: Sequence[int]) -> float | None:
    if len(t_ns) < 2:
        return None
    diffs: NDArray[np.float64] = np.diff(np.asarray(t_ns, dtype=np.float64))
    med = float(np.median(diffs))
    return 1e9 / med if med > 0.0 else None
