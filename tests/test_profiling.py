from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from percepcion3d.utils.profiling import StageTimer, percentiles_ms
from percepcion3d.utils.vram import VramPeak, VramSampler

# ─── StageTimer ──────────────────────────────────────────────────────────────


def test_percentiles_of_known_sample() -> None:
    timer = StageTimer(capacity=1000)
    for x in range(1, 101):  # 1..100 ms
        timer.record("s", float(x))
    s = timer.stats("s")
    assert s.count == 100
    assert s.mean_ms == pytest.approx(50.5)
    assert s.p50_ms == pytest.approx(50.5)
    assert s.p95_ms == pytest.approx(95.05)
    assert s.p99_ms == pytest.approx(99.01)
    assert s.max_ms == 100.0


def test_ring_buffer_keeps_latest_samples_and_total_count() -> None:
    timer = StageTimer(capacity=8)
    for x in range(20):
        timer.record("s", float(x))
    np.testing.assert_array_equal(timer.samples("s"), np.arange(12, 20, dtype=np.float64))
    assert timer.stats("s").count == 8
    assert timer.total_count("s") == 20
    assert timer.stats("s").max_ms == 19.0


def test_stage_context_manager_measures_sleep() -> None:
    timer = StageTimer()
    with timer.stage("nap"):
        time.sleep(0.02)
    s = timer.stats("nap")
    assert s.count == 1
    assert 15.0 < s.p50_ms < 200.0


def test_stage_records_even_when_block_raises() -> None:
    timer = StageTimer()
    with pytest.raises(ValueError), timer.stage("boom"):
        raise ValueError("x")
    assert timer.stats("boom").count == 1


def test_json_and_csv_export_roundtrip(tmp_path: Path) -> None:
    timer = StageTimer()
    for x in (1.0, 2.0, 3.0):
        timer.record("a", x)
    timer.record("b", 10.0)

    js = tmp_path / "lat.json"
    cs = tmp_path / "lat.csv"
    timer.to_json(js)
    timer.to_csv(cs)

    data = json.loads(js.read_text())
    assert set(data) == {"a", "b"}
    assert data["a"]["count"] == 3
    assert data["a"]["p50_ms"] == pytest.approx(2.0)
    assert data["b"]["max_ms"] == 10.0

    with open(cs, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["name"] for r in rows] == ["a", "b"]
    assert float(rows[0]["mean_ms"]) == pytest.approx(2.0)


def test_format_table_lists_every_stage() -> None:
    timer = StageTimer()
    timer.record("rectify", 1.5)
    timer.record("detector", 4.0)
    table = timer.format_table()
    assert "rectify" in table and "detector" in table
    assert "p99" in table.splitlines()[0]


def test_empty_stage_and_invalid_capacity() -> None:
    with pytest.raises(ValueError):
        StageTimer(capacity=0)
    with pytest.raises(ValueError):
        percentiles_ms([])
    assert percentiles_ms([5.0]) == (5.0, 5.0, 5.0)


# ─── VramSampler ─────────────────────────────────────────────────────────────


def test_vram_sampler_tracks_peaks_from_injected_fn() -> None:
    values = iter([(100.0, 900.0), (350.0, 1200.0), (200.0, 1000.0)])
    last = [(200.0, 1000.0)]

    def fake() -> tuple[float, float]:
        try:
            last[0] = next(values)
        except StopIteration:
            pass
        return last[0]

    sampler = VramSampler(interval_s=0.005, sample_fn=fake)
    with sampler:
        time.sleep(0.08)
    peak = sampler.peak
    assert isinstance(peak, VramPeak)
    assert peak.process_mb == 350.0
    assert peak.gpu_mb == 1200.0
    assert peak.samples >= 3


def test_vram_sampler_rate_is_close_to_requested() -> None:
    calls = 0
    lock = threading.Lock()

    def fake() -> tuple[float, float]:
        nonlocal calls
        with lock:
            calls += 1
        return 0.0, 0.0

    sampler = VramSampler(interval_s=0.01, sample_fn=fake)
    sampler.start()
    time.sleep(0.2)
    peak = sampler.stop()
    # ~20 expected at 100 Hz; allow generous scheduling slack on CI.
    assert 8 <= peak.samples <= 40
    assert peak.samples == calls


def test_vram_sampler_sample_once_and_double_start() -> None:
    sampler = VramSampler(sample_fn=lambda: (1.0, 2.0))
    assert sampler.sample_once() == VramPeak(1.0, 2.0, 1)
    sampler.start()
    with pytest.raises(RuntimeError):
        sampler.start()
    sampler.stop()
    with pytest.raises(ValueError):
        VramSampler(interval_s=0.0, sample_fn=lambda: (0.0, 0.0))
