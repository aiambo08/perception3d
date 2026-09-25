from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from percepcion3d.runtime.buffer import FrameStamped, LatestFrameSlot
from percepcion3d.runtime.playback import play_into_slot


def _frame(i: int) -> FrameStamped:
    return FrameStamped(frame_id=i, t_capture_ns=i * 1_000_000, img=np.zeros((2, 2, 3), np.uint8))


# ─── LatestFrameSlot ─────────────────────────────────────────────────────────


def test_latest_wins_and_dropped_counter() -> None:
    slot: LatestFrameSlot[FrameStamped] = LatestFrameSlot()
    assert slot.get(timeout=0.0) is None
    slot.put(_frame(0))
    slot.put(_frame(1))
    slot.put(_frame(2))
    got = slot.get(timeout=0.1)
    assert got is not None and got.frame_id == 2
    assert slot.dropped == 2
    assert slot.produced == 3
    # Nothing new: blocks until timeout.
    assert slot.get(timeout=0.02) is None
    # Same frame is not returned twice, and peek does not consume.
    slot.put(_frame(3))
    peeked = slot.peek()
    assert peeked is not None and peeked.frame_id == 3
    got = slot.get(timeout=0.1)
    assert got is not None and got.frame_id == 3
    assert slot.dropped == 2


def test_consumer_wakes_on_put_and_on_close() -> None:
    slot: LatestFrameSlot[FrameStamped] = LatestFrameSlot()
    results: list[int | None] = []

    def consumer() -> None:
        f = slot.get(timeout=2.0)
        results.append(f.frame_id if f is not None else None)
        results.append(None if slot.get(timeout=2.0) is None else -1)

    th = threading.Thread(target=consumer)
    th.start()
    time.sleep(0.02)
    slot.put(_frame(7))
    time.sleep(0.02)
    slot.close()
    th.join(timeout=3.0)
    assert not th.is_alive()
    assert results == [7, None]
    assert slot.closed
    with pytest.raises(RuntimeError):
        slot.put(_frame(8))


def test_fast_producer_slow_consumer_bounded_memory() -> None:
    slot: LatestFrameSlot[FrameStamped] = LatestFrameSlot()
    n_produced = 300
    consumed: list[int] = []
    stop = threading.Event()

    def consumer() -> None:
        while not stop.is_set() or slot.peek() is not None:
            f = slot.get(timeout=0.01)
            if f is None:
                if stop.is_set():
                    break
                continue
            consumed.append(f.frame_id)
            time.sleep(0.002)  # slower than the producer

    th = threading.Thread(target=consumer)
    th.start()
    for i in range(n_produced):
        slot.put(_frame(i))
        time.sleep(0.0002)
    stop.set()
    slot.close()
    th.join(timeout=5.0)

    assert consumed == sorted(consumed)  # monotone, never re-delivered
    assert len(set(consumed)) == len(consumed)
    assert slot.dropped + len(consumed) == n_produced
    assert slot.dropped > 0


# ─── play_into_slot ──────────────────────────────────────────────────────────


def test_unpaced_playback_reports_counts() -> None:
    slot: LatestFrameSlot[FrameStamped] = LatestFrameSlot()
    stats = play_into_slot((_frame(i) for i in range(50)), slot, target_hz=None)
    assert stats.frames == 50
    assert stats.target_hz is None
    assert stats.dropped == 49  # no consumer
    assert np.isfinite(stats.period_p50_ms)
    got = slot.get(timeout=0.0)
    assert got is not None and got.frame_id == 49
    assert got.t_capture_ns != 49 * 1_000_000  # restamped to monotonic time


def test_playback_keeps_original_stamp_when_restamp_false() -> None:
    slot: LatestFrameSlot[FrameStamped] = LatestFrameSlot()
    play_into_slot([_frame(3)], slot, restamp=False)
    got = slot.get(timeout=0.0)
    assert got is not None and got.t_capture_ns == 3_000_000


def test_playback_max_frames_and_stop_event() -> None:
    slot: LatestFrameSlot[FrameStamped] = LatestFrameSlot()
    stats = play_into_slot((_frame(i) for i in range(10_000)), slot, max_frames=25)
    assert stats.frames == 25
    ev = threading.Event()
    ev.set()
    stats = play_into_slot((_frame(i) for i in range(10)), slot, stop_event=ev)
    assert stats.frames == 0
    assert np.isnan(stats.period_p50_ms)


@pytest.mark.slow
def test_paced_playback_holds_60hz_with_low_jitter() -> None:
    slot: LatestFrameSlot[FrameStamped] = LatestFrameSlot()
    stats = play_into_slot((_frame(i) for i in range(120)), slot, target_hz=60.0)
    assert stats.frames == 120
    assert 57.0 <= stats.achieved_hz <= 61.0
    assert stats.period_p50_ms == pytest.approx(1000.0 / 60.0, abs=0.5)
    assert stats.jitter_p99_ms < 3.0
