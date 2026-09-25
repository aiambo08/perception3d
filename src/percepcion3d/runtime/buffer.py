"""Single-slot *latest-wins* hand-off between producer and consumer threads.

A capture thread running at 60 Hz must never block on a slower consumer
(3D inference at 25-35 Hz). ``LatestFrameSlot`` keeps exactly one frame:
``put`` overwrites any unconsumed frame (counting it as dropped) and ``get``
blocks until a frame newer than the last one returned is available. Memory is
bounded to one frame regardless of the rate mismatch.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FrameStamped:
    """A frame with its monotonic capture timestamp.

    Attributes:
        frame_id: Sequential index from the source (may skip after drops).
        t_capture_ns: ``time.monotonic_ns()`` at capture, or the dataset timestamp
            mapped to nanoseconds for replayed sequences.
        img: HxWx3 uint8 BGR (OpenCV convention) or HxW for grayscale.
    """

    frame_id: int
    t_capture_ns: int
    img: NDArray[np.uint8]


T = TypeVar("T")


class LatestFrameSlot(Generic[T]):
    """Thread-safe one-element mailbox where the newest item replaces the previous one."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item: T | None = None
        self._seq = 0
        self._consumed_seq = 0
        self._dropped = 0
        self._closed = False

    def put(self, item: T) -> None:
        """Store ``item``; an unconsumed previous item is discarded and counted as dropped."""
        with self._cond:
            if self._closed:
                raise RuntimeError("LatestFrameSlot is closed")
            if self._item is not None and self._seq > self._consumed_seq:
                self._dropped += 1
            self._item = item
            self._seq += 1
            self._cond.notify_all()

    def get(self, timeout: float | None = None) -> T | None:
        """Block until an item newer than the last returned one arrives.

        Returns ``None`` on timeout or once the slot is closed and drained.
        """
        with self._cond:
            if not self._cond.wait_for(
                lambda: self._seq > self._consumed_seq or self._closed, timeout=timeout
            ):
                return None
            if self._seq <= self._consumed_seq:
                return None
            self._consumed_seq = self._seq
            return self._item

    def peek(self) -> T | None:
        """Return the current item (consumed or not) without blocking or marking it consumed."""
        with self._cond:
            return self._item

    def close(self) -> None:
        """Wake blocked consumers; further ``put`` calls raise."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def dropped(self) -> int:
        with self._cond:
            return self._dropped

    @property
    def produced(self) -> int:
        with self._cond:
            return self._seq

    @property
    def closed(self) -> bool:
        with self._cond:
            return self._closed
