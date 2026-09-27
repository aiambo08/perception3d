"""ByteTrack-style 2D association (IoU + two score thresholds, Hungarian, no ReID).

Each track carries a constant-velocity box filter on ``[cx, cy, w, h]`` driven by
capture timestamps (variable ``Δt``), with noise proportional to the box height
as in ByteTrack. Per frame:

1. predict every live track to ``t``;
2. match *high*-score detections (``≥ high_thresh``) to all tracks on
   ``1 − IoU`` (Hungarian), keeping pairs with ``IoU ≥ match_iou``;
3. match *low*-score detections (``[low_thresh, high_thresh)``) to the remaining
   confirmed, not-lost tracks with the stricter ``IoU ≥ second_iou`` — this is
   what recovers partially occluded objects the detector scores low;
4. unmatched tentative tracks die; unmatched confirmed tracks become ``LOST`` and
   are removed after ``max_lost_s`` without a match;
5. unmatched high detections with ``score ≥ new_track_thresh`` start tentative
   tracks, confirmed after ``min_hits`` matches.

Association is restricted to detections of the same class *group*
(``class_groups``), so a car/truck label flicker does not break identity while a
pedestrian never inherits a car track.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from percepcion3d.eval.detection import iou_xyxy

DEFAULT_CLASS_GROUPS: dict[str, str] = {
    "car": "vehicle",
    "van": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "person": "person",
    "pedestrian": "person",
    "bicycle": "cycle",
    "cyclist": "cycle",
    "motorcycle": "cycle",
}


@dataclass(frozen=True)
class ByteTrackConfig:
    high_thresh: float = 0.5
    low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    match_iou: float = 0.2
    second_iou: float = 0.5
    min_hits: int = 2
    max_lost_s: float = 1.0
    accel_rel: float = 2.0
    """Box acceleration noise per pixel of height (1/s²): ``q = (accel_rel·h)²`` px²/s³."""
    meas_rel: float = 0.05
    """Box measurement noise per pixel of height (σ = ``meas_rel·h`` px)."""
    class_groups: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_CLASS_GROUPS))

    def group(self, cls: str) -> str:
        return self.class_groups.get(cls, cls)


class TrackState(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"


def _xyxy_to_cxcywh(b: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array(
        [0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]), b[2] - b[0], b[3] - b[1]], dtype=np.float64
    )


@dataclass
class Track2D:
    track_id: int
    cls: str
    score: float
    x: NDArray[np.float64]
    """``[cx, cy, w, h, vcx, vcy, vw, vh]`` (px, px/s)."""
    P: NDArray[np.float64]
    t_ns: int
    t_last_match_ns: int
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    det_index: int = -1
    """Detection matched in the latest :meth:`ByteTracker.update` (``-1`` if none)."""

    @property
    def box(self) -> NDArray[np.float64]:
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float64)


class ByteTracker:
    def __init__(self, cfg: ByteTrackConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else ByteTrackConfig()
        self.tracks: list[Track2D] = []
        self._next_id = 1

    # -- box filter (4 independent [position, velocity] pairs per track) --------
    def _predict_all(self, t_ns: int) -> None:
        if not self.tracks:
            return
        x = np.stack([t.x for t in self.tracks])
        p = np.stack([t.P for t in self.tracks])
        dt = np.maximum(0.0, np.array([(t_ns - t.t_ns) * 1e-9 for t in self.tracks]))
        f = np.broadcast_to(np.eye(8), (dt.size, 8, 8)).copy()
        idx = np.arange(4)
        f[:, idx, idx + 4] = dt[:, None]
        q = (self.cfg.accel_rel * np.maximum(x[:, 3], 1.0)) ** 2
        g = np.zeros_like(f)
        g[:, idx, idx] = (q * dt**3 / 3.0)[:, None]
        g[:, idx, idx + 4] = g[:, idx + 4, idx] = (q * dt**2 / 2.0)[:, None]
        g[:, idx + 4, idx + 4] = (q * dt)[:, None]
        x = np.einsum("nij,nj->ni", f, x)
        x[:, 2:4] = np.maximum(x[:, 2:4], 1.0)
        p = f @ p @ np.transpose(f, (0, 2, 1)) + g
        for i, tr in enumerate(self.tracks):
            tr.x, tr.P, tr.t_ns = x[i], p[i], t_ns

    def _correct_all(self, tracks: Sequence[Track2D], boxes: NDArray[np.float64]) -> None:
        if not tracks:
            return
        x = np.stack([t.x for t in tracks])
        p = np.stack([t.P for t in tracks])
        z = np.stack(
            [
                0.5 * (boxes[:, 0] + boxes[:, 2]),
                0.5 * (boxes[:, 1] + boxes[:, 3]),
                boxes[:, 2] - boxes[:, 0],
                boxes[:, 3] - boxes[:, 1],
            ],
            axis=1,
        )
        r = (self.cfg.meas_rel * np.maximum(z[:, 3], 1.0)) ** 2
        # P stays block-diagonal per axis (diagonal prior, per-axis F/Q/R), so each axis is a
        # scalar-measurement update of its [position, velocity] pair.
        idx = np.arange(4)
        pp = p[:, idx, idx]
        pv = p[:, idx, idx + 4]
        vv = p[:, idx + 4, idx + 4]
        sinv = 1.0 / (pp + r[:, None])
        kp, kv = pp * sinv, pv * sinv
        nu = z - x[:, :4]
        x = x.copy()
        x[:, :4] += kp * nu
        x[:, 4:] += kv * nu
        p = p.copy()
        p[:, idx, idx] = pp - kp * pp
        p[:, idx, idx + 4] = p[:, idx + 4, idx] = pv - kp * pv
        p[:, idx + 4, idx + 4] = vv - kv * pv
        for i, tr in enumerate(tracks):
            tr.x, tr.P = x[i], p[i]

    def _new(self, box: NDArray[np.float64], score: float, cls: str, t_ns: int, i: int) -> Track2D:
        z = _xyxy_to_cxcywh(box)
        h = max(z[3], 1.0)
        p = np.diag(
            np.r_[
                np.full(4, (2 * self.cfg.meas_rel * h) ** 2),
                np.full(4, (10 * self.cfg.meas_rel * h) ** 2),
            ]
        )
        state = TrackState.CONFIRMED if self.cfg.min_hits <= 1 else TrackState.TENTATIVE
        tr = Track2D(self._next_id, cls, score, np.r_[z, np.zeros(4)], p, t_ns, t_ns, state, 1, i)
        self._next_id += 1
        return tr

    # -- association ------------------------------------------------------------
    def _match(
        self,
        tracks: Sequence[Track2D],
        boxes: NDArray[np.float64],
        det_idx: NDArray[np.intp],
        groups: Sequence[str],
        min_iou: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Hungarian on ``1 − IoU``; returns ``(pairs, unmatched_tracks, unmatched_dets)`` as
        positions into ``tracks`` / ``det_idx``."""
        if not tracks or det_idx.size == 0:
            return [], list(range(len(tracks))), list(range(det_idx.size))
        c = np.stack([t.x[:4] for t in tracks])
        tb = np.concatenate([c[:, :2] - c[:, 2:] / 2, c[:, :2] + c[:, 2:] / 2], axis=1)
        iou = iou_xyxy(tb, boxes[det_idx])
        codes = {g: i for i, g in enumerate(set(groups))}
        tg = np.array([codes.get(self.cfg.group(t.cls), -1) for t in tracks])
        dg = np.array([codes[groups[i]] for i in det_idx])
        iou = np.where(tg[:, None] == dg[None, :], iou, 0.0)
        rows, cols = linear_sum_assignment(1.0 - iou)
        pairs = [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if iou[r, c] >= min_iou]
        mt = {r for r, _ in pairs}
        md = {c for _, c in pairs}
        return (
            pairs,
            [r for r in range(len(tracks)) if r not in mt],
            [c for c in range(det_idx.size) if c not in md],
        )

    def update(
        self,
        t_ns: int,
        boxes: NDArray[np.floating[Any]],
        scores: NDArray[np.floating[Any]],
        classes: Sequence[str],
    ) -> list[Track2D]:
        """Advance to ``t_ns`` with this frame's detections; returns the live tracks.

        ``Track2D.det_index`` of each returned track is the detection it was
        matched to in this call (``-1`` when it coasted).
        """
        cfg = self.cfg
        b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        s = np.asarray(scores, dtype=np.float64).reshape(-1)
        groups = [cfg.group(c) for c in classes]
        self._predict_all(t_ns)
        for tr in self.tracks:
            tr.det_index = -1

        high = np.flatnonzero(s >= cfg.high_thresh)
        low = np.flatnonzero((s >= cfg.low_thresh) & (s < cfg.high_thresh))

        pairs1, um_t1, um_d1 = self._match(self.tracks, b, high, groups, cfg.match_iou)
        for ti, di in pairs1:
            self._assign(self.tracks[ti], s, classes, int(high[di]), t_ns)

        rest = [self.tracks[i] for i in um_t1 if self.tracks[i].state is TrackState.CONFIRMED]
        pairs2, _, _ = self._match(rest, b, low, groups, cfg.second_iou)
        for ti, di in pairs2:
            self._assign(rest[ti], s, classes, int(low[di]), t_ns, update_cls=False)
        matched = [t for t in self.tracks if t.det_index >= 0]
        self._correct_all(matched, b[[t.det_index for t in matched]])

        survivors: list[Track2D] = []
        for tr in self.tracks:
            if tr.det_index >= 0:
                survivors.append(tr)
            elif tr.state is TrackState.TENTATIVE:
                continue
            elif (t_ns - tr.t_last_match_ns) * 1e-9 <= cfg.max_lost_s:
                tr.state = TrackState.LOST
                survivors.append(tr)
        for di in um_d1:
            i = int(high[di])
            if s[i] >= cfg.new_track_thresh:
                survivors.append(self._new(b[i], float(s[i]), classes[i], t_ns, i))
        self.tracks = survivors
        return list(self.tracks)

    def _assign(
        self,
        tr: Track2D,
        s: NDArray[np.float64],
        classes: Sequence[str],
        i: int,
        t_ns: int,
        update_cls: bool = True,
    ) -> None:
        tr.det_index = i
        tr.score = float(s[i])
        if update_cls:
            tr.cls = classes[i]
        tr.hits += 1
        tr.t_last_match_ns = t_ns
        if tr.state is TrackState.LOST or tr.hits >= self.cfg.min_hits:
            tr.state = TrackState.CONFIRMED
