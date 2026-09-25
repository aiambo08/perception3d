"""Paced replay of a frame source into a :class:`LatestFrameSlot` with jitter statistics.

This is the capture-thread stand-in for offline runs: it pushes frames at a
target rate (or as fast as the decoder allows) and measures how well the
schedule is kept, which bounds the timing noise injected into ``dt`` for the
kinematic filters downstream.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from percepcion3d.runtime.buffer import FrameStamped, LatestFrameSlot
from percepcion3d.utils.profiling import percentiles_ms


@dataclass(frozen=True)
class PlaybackStats:
    """Timing summary of one replay.

    ``jitter_*`` is ``|actual_period - target_period|`` per frame in ms (only
    meaningful when ``target_hz`` was set); ``period_*`` is the raw inter-frame
    period in ms.
    """

    frames: int
    elapsed_s: float
    achieved_hz: float
    target_hz: float | None
    period_p50_ms: float
    period_p95_ms: float
    period_p99_ms: float
    jitter_p50_ms: float
    jitter_p95_ms: float
    jitter_p99_ms: float
    dropped: int


def play_into_slot(
    source: Iterable[FrameStamped],
    slot: LatestFrameSlot[FrameStamped],
    target_hz: float | None = None,
    max_frames: int | None = None,
    stop_event: threading.Event | None = None,
    restamp: bool = True,
) -> PlaybackStats:
    """Push frames from ``source`` into ``slot`` at ``target_hz`` (``None`` = unpaced).

    Args:
        restamp: Replace ``t_capture_ns`` with ``time.monotonic_ns()`` at push time
            so the consumer sees wall-clock timing, as with a live camera.
    """
    period_s = 1.0 / target_hz if target_hz else 0.0
    t_push: list[int] = []
    t_start = time.perf_counter()
    next_t = t_start
    n = 0
    for frame in source:
        if stop_event is not None and stop_event.is_set():
            break
        if period_s > 0.0:
            remaining = next_t - time.perf_counter()
            if remaining > 0.0:
                _sleep_precise(remaining)
            next_t += period_s
        now_ns = time.monotonic_ns()
        if restamp:
            frame = FrameStamped(frame.frame_id, now_ns, frame.img)
        slot.put(frame)
        t_push.append(now_ns)
        n += 1
        if max_frames is not None and n >= max_frames:
            break
    elapsed = time.perf_counter() - t_start

    if len(t_push) >= 2:
        periods_ms = np.diff(np.asarray(t_push, dtype=np.float64)) * 1e-6
        p50, p95, p99 = percentiles_ms(periods_ms)
        if target_hz:
            jitter = np.abs(periods_ms - 1e3 / target_hz)
        else:
            jitter = np.abs(periods_ms - float(np.median(periods_ms)))
        j50, j95, j99 = percentiles_ms(jitter)
    else:
        p50 = p95 = p99 = j50 = j95 = j99 = float("nan")

    return PlaybackStats(
        frames=n,
        elapsed_s=elapsed,
        achieved_hz=n / elapsed if elapsed > 0.0 else float("nan"),
        target_hz=target_hz,
        period_p50_ms=p50,
        period_p95_ms=p95,
        period_p99_ms=p99,
        jitter_p50_ms=j50,
        jitter_p95_ms=j95,
        jitter_p99_ms=j99,
        dropped=slot.dropped,
    )


def _sleep_precise(seconds: float, spin_threshold_s: float = 0.0015) -> None:
    """Sleep most of the interval, then spin the last ~1.5 ms for sub-ms accuracy on Linux."""
    deadline = time.perf_counter() + seconds
    coarse = seconds - spin_threshold_s
    if coarse > 0.0:
        time.sleep(coarse)
    while time.perf_counter() < deadline:
        pass
