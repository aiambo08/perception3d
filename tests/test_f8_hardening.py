"""CPU tests for F8: INT8 calibration batcher, FP16/INT8 A/B verdict, bench history."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from percepcion3d.detection.int8_calibration import (
    CalibrationBatcher,
    list_calibration_images,
)
from percepcion3d.eval.engine_compare import (
    LatencyDelta,
    RecallDelta,
    compare_reports,
    format_compare,
)
from percepcion3d.utils.bench_history import (
    archive_report,
    find_regressions,
    flatten_metrics,
    format_regressions,
    list_reports,
    load_report,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


# ─── INT8 calibration batcher ─────────────────────────────────────────────────


def _write_seq(d: Path, n: int, hw: tuple[int, int] = (375, 1242), seed: int = 0) -> list[Path]:
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        p = d / f"{i:06d}.png"
        cv2.imwrite(str(p), rng.integers(0, 256, size=(*hw, 3), dtype=np.uint8))
        out.append(p)
    return out


def test_list_calibration_images_spreads_over_dirs_and_is_deterministic(tmp_path: Path) -> None:
    _write_seq(tmp_path / "0000", 20, hw=(8, 8))
    _write_seq(tmp_path / "0001", 60, hw=(8, 8))
    (tmp_path / "0001" / "notes.txt").write_text("ignored")
    a = list_calibration_images([tmp_path / "0000", tmp_path / "0001"], 8)
    b = list_calibration_images([tmp_path / "0000", tmp_path / "0001"], 8)
    assert a == b and len(a) == 8
    per_dir = {
        d.name: sum(p.parent == d for p in a) for d in (tmp_path / "0000", tmp_path / "0001")
    }
    assert per_dir == {"0000": 2, "0001": 6}  # proportional to sequence length
    idx = sorted(int(p.stem) for p in a if p.parent.name == "0001")
    assert idx[0] == 0 and idx[-1] == 59  # evenly spread, not the first seconds


def test_list_calibration_images_caps_at_available_and_seed_shifts(tmp_path: Path) -> None:
    _write_seq(tmp_path / "s", 5, hw=(8, 8))
    assert len(list_calibration_images([tmp_path / "s"], 50)) == 5
    a = list_calibration_images([tmp_path / "s"], 2, seed=0)
    b = list_calibration_images([tmp_path / "s"], 2, seed=1)
    assert a != b


def test_list_calibration_images_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        list_calibration_images([tmp_path], 0)
    with pytest.raises(FileNotFoundError):
        list_calibration_images([tmp_path], 3)


def test_batcher_letterboxes_to_engine_input_and_exhausts(tmp_path: Path) -> None:
    paths = _write_seq(tmp_path / "s", 3)
    b = CalibrationBatcher(paths, (320, 1024))
    assert b.batch_shape == (1, 320, 1024, 3)
    seen = []
    while (batch := b.next_batch()) is not None:
        assert batch.shape == (1, 320, 1024, 3) and batch.dtype == np.uint8
        assert batch.flags["C_CONTIGUOUS"]
        # KITTI 375x1242 → scale 1024/1242 → 309 rows: top/bottom pad rows are fill 114
        assert np.all(batch[0, 0] == 114) and np.all(batch[0, -1] == 114)
        seen.append(batch.copy())
    assert len(seen) == 3 and b.served == 3
    assert b.next_batch() is None
    assert not np.array_equal(seen[0], seen[1])
    b.reset()
    again = b.next_batch()
    assert again is not None and np.array_equal(again, seen[0])


def test_batcher_batches_are_independent_copies(tmp_path: Path) -> None:
    paths = _write_seq(tmp_path / "s", 2)
    b = CalibrationBatcher(paths, (64, 128))
    first = b.next_batch()
    assert first is not None
    snapshot = first.copy()
    b.next_batch()
    assert np.array_equal(first, snapshot)


def test_batcher_cache_roundtrip_and_requirements(tmp_path: Path) -> None:
    cache = tmp_path / "c" / "calib.cache"
    with pytest.raises(ValueError):
        CalibrationBatcher([], (64, 64), cache)
    paths = _write_seq(tmp_path / "s", 1, hw=(16, 16))
    b = CalibrationBatcher(paths, (64, 64), cache)
    assert b.read_cache() is None
    b.write_cache(b"\x00\x01TRT")
    assert b.read_cache() == b"\x00\x01TRT"
    only_cache = CalibrationBatcher([], (64, 64), cache)
    assert only_cache.next_batch() is None
    no_cache = CalibrationBatcher(paths, (64, 64), None)
    no_cache.write_cache(b"x")
    assert no_cache.read_cache() is None


def test_batcher_rejects_unreadable_image(tmp_path: Path) -> None:
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not a png")
    with pytest.raises(OSError):
        CalibrationBatcher([bad], (64, 64)).next_batch()


# ─── FP16 vs INT8 verdict ─────────────────────────────────────────────────────


def _bench_report(
    ped: tuple[int, int], car: tuple[int, int], gpu_p95: float, e2e_p95: float
) -> dict[str, Any]:
    def row(k: int, n: int) -> dict[str, Any]:
        return {"kitti_type": "x", "n_gt": n, "n_matched": k, "n_pred": k, "recall": k / n}

    return {
        "engine": "e",
        "stages": {
            "det.gpu": {"p50_ms": gpu_p95 * 0.9, "p95_ms": gpu_p95, "p99_ms": gpu_p95 * 1.1},
            "detector_e2e": {"p50_ms": e2e_p95 * 0.9, "p95_ms": e2e_p95, "p99_ms": e2e_p95 * 1.1},
        },
        "recall": {"pooled": {"Pedestrian": row(*ped), "Car": row(*car)}},
    }


def test_compare_accepts_within_thresholds() -> None:
    base = _bench_report((900, 1000), (1900, 2000), 1.10, 2.0)
    cand = _bench_report((895, 1000), (1890, 2000), 0.70, 1.6)
    res = compare_reports(base, cand)
    assert res.recall["Pedestrian"].delta_pt == pytest.approx(-0.5)
    assert res.latency["det.gpu"].gain == pytest.approx(1 - 0.70 / 1.10)
    assert res.verdicts == {"recall_pedestrian": "PASS", "latency_det.gpu": "PASS"}
    assert res.accept
    assert not res.recall_conclusive  # Δ −0.5 pt on 1000 GT: CI ≈ ±2.6 pt straddles −2
    txt = format_compare(res)
    assert "ACCEPT" in txt and "Pedestrian" in txt and "det.gpu" in txt and "straddles" in txt
    big = compare_reports(
        _bench_report((90000, 100000), (1, 1), 1.0, 2.0),
        _bench_report((89500, 100000), (1, 1), 0.7, 1.6),
    )
    assert big.accept and big.recall_conclusive
    assert "straddles" not in format_compare(big)


def test_compare_rejects_recall_drop_and_small_latency_gain() -> None:
    base = _bench_report((900, 1000), (1900, 2000), 1.0, 2.0)
    slow = _bench_report((900, 1000), (1900, 2000), 0.9, 1.9)
    res = compare_reports(base, slow)
    assert res.verdicts["latency_det.gpu"] == "FAIL" and not res.accept
    bad = _bench_report((800, 1000), (1900, 2000), 0.5, 1.0)
    res = compare_reports(base, bad)
    assert res.recall["Pedestrian"].delta_pt == pytest.approx(-10.0)
    assert res.verdicts["recall_pedestrian"] == "FAIL" and not res.accept


def test_compare_flags_inconclusive_recall_ci() -> None:
    # Δ = −2.5 pt on 200 pedestrians: CI half-width ≈ 6 pt → straddles −2 pt
    base = _bench_report((180, 200), (10, 10), 1.0, 2.0)
    cand = _bench_report((175, 200), (10, 10), 0.5, 1.0)
    res = compare_reports(base, cand)
    lo, hi = res.recall["Pedestrian"].ci95_pt
    assert lo < -2.0 < hi
    assert res.verdicts["recall_pedestrian"] == "FAIL" and not res.accept
    assert not res.recall_conclusive
    # Δ = −5 pt on 20 000 pedestrians: clearly below the threshold.
    base = _bench_report((18000, 20000), (10, 10), 1.0, 2.0)
    cand = _bench_report((17000, 20000), (10, 10), 0.5, 1.0)
    res = compare_reports(base, cand)
    assert res.verdicts["recall_pedestrian"] == "FAIL" and res.recall_conclusive


def test_compare_handles_missing_data() -> None:
    res = compare_reports({}, {})
    assert res.verdicts == {"recall_pedestrian": "NO-DATA", "latency_det.gpu": "NO-DATA"}
    assert not res.accept
    d = res.to_dict()
    json.dumps(d)  # NaN allowed by json but structure must serialise
    assert d["accept"] is False


def test_recall_and_latency_delta_edge_cases() -> None:
    r = RecallDelta("Pedestrian", 0.9, 0.9, 0, 0)
    assert r.verdict(2.0) == "NO-DATA"
    zero = LatencyDelta("det.gpu", "p95_ms", 0.0, 1.0)
    assert zero.verdict(0.25) == "NO-DATA"
    slower = LatencyDelta("det.gpu", "p95_ms", 1.0, 1.5)
    assert slower.gain == pytest.approx(-0.5) and slower.verdict(0.25) == "FAIL"


@pytest.mark.parametrize("seed", range(5))
def test_compare_random_reports_verdicts_consistent_with_rule(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(50, 5000))
    kb, kc = int(rng.integers(0, n + 1)), int(rng.integers(0, n + 1))
    pb, pc = float(rng.uniform(0.5, 3.0)), float(rng.uniform(0.3, 3.0))
    res = compare_reports(
        _bench_report((kb, n), (1, 1), pb, 1.0), _bench_report((kc, n), (1, 1), pc, 1.0)
    )
    r = res.recall["Pedestrian"]
    lo, hi = r.ci95_pt
    assert lo <= r.delta_pt <= hi
    assert res.verdicts["recall_pedestrian"] == ("PASS" if r.delta_pt >= -2.0 else "FAIL")
    assert res.recall_conclusive == (lo >= -2.0 or hi < -2.0)
    expected_lat = "PASS" if (pb - pc) / pb >= 0.25 else "FAIL"
    assert res.verdicts["latency_det.gpu"] == expected_lat
    assert res.accept == all(v == "PASS" for v in res.verdicts.values())


def test_compare_engines_script(tmp_path: Path) -> None:
    base = tmp_path / "b.json"
    cand = tmp_path / "c.json"
    out = tmp_path / "out.json"
    base.write_text(json.dumps(_bench_report((900, 1000), (1, 1), 1.0, 2.0)))
    cand.write_text(json.dumps(_bench_report((899, 1000), (1, 1), 0.6, 1.5)))
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "compare_engines.py"),
            str(base),
            str(cand),
            "--json",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ACCEPT" in proc.stdout
    data = json.loads(out.read_text())
    assert data["accept"] is True and data["verdicts"]["recall_pedestrian"] == "PASS"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "compare_engines.py"),
            str(base),
            str(tmp_path / "missing.json"),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode == 2


# ─── Bench history ────────────────────────────────────────────────────────────


def _pipeline_report(p99: float, hz: float, drop: float, vram: float) -> dict[str, Any]:
    return {
        "stats": {
            "frames": 3600,
            "drop_frac": drop,
            "depth": {"hz": hz, "age_ms": {"p50": 16.0, "p95": 33.3, "p99": 33.4, "n": 100}},
            "e2e_alert_ms": {"p50": 6.0, "p95": 7.5, "p99": p99, "n": 3600},
        },
        "stages": {
            "boxes": {
                "count": 3600,
                "mean_ms": 3.0,
                "p50_ms": 3.0,
                "p95_ms": 3.6,
                "p99_ms": 4.1,
                "max_ms": 9.0,
            }
        },
        "vram_peak_over_baseline_mb": vram,
        "verdicts": {"e2e_alert_p99": True},
        "recall": {"pooled": {"Pedestrian": {"recall": 0.9, "n_gt": 100}}},
    }


def test_flatten_metrics_picks_percentiles_memory_recall_and_skips_bools() -> None:
    m = flatten_metrics(_pipeline_report(9.0, 29.9, 0.004, 539.0))
    assert m["stats.e2e_alert_ms.p99"] == 9.0
    assert m["stats.depth.hz"] == 29.9
    assert m["stats.drop_frac"] == 0.004
    assert m["stages.boxes.p99_ms"] == 4.1 and m["stages.boxes.max_ms"] == 9.0
    assert m["vram_peak_over_baseline_mb"] == 539.0
    assert m["recall.pooled.Pedestrian.recall"] == 0.9
    assert "stats.frames" not in m and "stages.boxes.count" not in m
    assert "verdicts.e2e_alert_p99" not in m and "stats.e2e_alert_ms.n" not in m


def test_find_regressions_direction_and_tolerances() -> None:
    prev = _pipeline_report(9.0, 29.9, 0.004, 539.0)
    same = _pipeline_report(9.5, 29.5, 0.004, 560.0)  # +5.5 %, −1.3 %, +3.9 %: within 10 %
    assert find_regressions(prev, same) == []
    worse = _pipeline_report(12.0, 20.0, 0.02, 700.0)
    regs = {r.metric: r for r in find_regressions(prev, worse)}
    assert set(regs) == {
        "stats.e2e_alert_ms.p99",
        "stats.depth.hz",
        "vram_peak_over_baseline_mb",
    }
    assert regs["stats.depth.hz"].higher_is_better and regs[
        "stats.depth.hz"
    ].rel_change == pytest.approx(9.9 / 29.9)
    assert not regs["stats.e2e_alert_ms.p99"].higher_is_better
    # drop_frac 0.004 → 0.02 is +400 % but only +0.016 absolute: below abs_tol by default…
    assert "stats.drop_frac" not in regs
    # …and caught with a tighter absolute tolerance.
    tight = {r.metric for r in find_regressions(prev, worse, abs_tol=0.001)}
    assert "stats.drop_frac" in tight
    # Improvements never count as regressions.
    better = _pipeline_report(5.0, 60.0, 0.0, 100.0)
    assert find_regressions(prev, better) == []
    assert format_regressions([]) == "no regressions"
    assert "stats.depth.hz" in format_regressions(list(regs.values()))


def test_find_regressions_ignores_missing_and_nan() -> None:
    prev = {"stages": {"a": {"p99_ms": 1.0}, "b": {"p99_ms": float("nan")}}}
    curr = {"stages": {"a": {"p99_ms": 1.0}, "b": {"p99_ms": 5.0}, "c": {"p99_ms": 99.0}}}
    assert find_regressions(prev, curr) == []


def test_archive_and_list_reports_sorted_by_stamp(tmp_path: Path) -> None:
    t0 = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)
    p1 = archive_report({"x": {"p99_ms": 1.0}}, "pipeline", tmp_path, now=t0)
    p2 = archive_report({"x": {"p99_ms": 2.0}}, "pipeline", tmp_path, now=t0 + timedelta(hours=1))
    p3 = archive_report(
        {"x": {"p99_ms": 3.0}}, "detector", tmp_path, now=t0 + timedelta(minutes=30)
    )
    assert p1.name == "20260928_120000_pipeline.json"
    assert list_reports(tmp_path) == [p1, p3, p2]
    assert list_reports(tmp_path, "pipeline") == [p1, p2]
    assert list_reports(tmp_path / "nope") == []
    (tmp_path / "random.json").write_text("{}")
    assert len(list_reports(tmp_path)) == 3
    data = load_report(p1)
    assert data["bench_history"] == {"name": "pipeline", "archived_at": t0.isoformat()}
    assert data["x"]["p99_ms"] == 1.0
    with pytest.raises(ValueError):
        archive_report({}, "bad name/with slash", tmp_path)


def test_archive_same_second_does_not_overwrite(tmp_path: Path) -> None:
    t0 = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)
    p1 = archive_report({"x": {"p99_ms": 1.0}}, "pipeline", tmp_path, now=t0)
    p2 = archive_report({"x": {"p99_ms": 2.0}}, "pipeline", tmp_path, now=t0)
    p3 = archive_report({"x": {"p99_ms": 3.0}}, "pipeline", tmp_path, now=t0)
    assert p2.name == "20260928_120000-01_pipeline.json"
    assert list_reports(tmp_path, "pipeline") == [p1, p2, p3]
    assert [load_report(p)["x"]["p99_ms"] for p in (p1, p2, p3)] == [1.0, 2.0, 3.0]


def test_load_report_rejects_non_object(tmp_path: Path) -> None:
    p = tmp_path / "list.json"
    p.write_text("[1, 2]")
    with pytest.raises(ValueError):
        load_report(p)


def test_bench_history_script_add_list_check(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    r1, r2 = tmp_path / "r1.json", tmp_path / "r2.json"
    r1.write_text(json.dumps(_pipeline_report(9.0, 29.9, 0.004, 539.0)))
    r2.write_text(json.dumps(_pipeline_report(14.0, 29.9, 0.004, 539.0)))

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "bench_history.py"), "--root", str(root), *args],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )

    assert run("check", "--name", "pipeline").returncode == 0  # fewer than two runs: no verdict
    assert run("add", str(r1), "--name", "pipeline").returncode == 0
    assert run("check", "--name", "pipeline").returncode == 0
    assert run("add", str(r2), "--name", "pipeline").returncode == 0
    ls = run("list")
    assert ls.returncode == 0 and ls.stdout.count("_pipeline.json") == 2
    out = tmp_path / "check.json"
    chk = run("check", "--name", "pipeline", "--json", str(out))
    assert chk.returncode == 1
    assert "stats.e2e_alert_ms.p99" in chk.stdout
    data = json.loads(out.read_text())
    assert [r["metric"] for r in data["regressions"]] == ["stats.e2e_alert_ms.p99"]
    assert run("check", "--prev", str(r2), "--curr", str(r1)).returncode == 0
    assert run("check", "--prev", str(r1)).returncode == 2
    assert run("check").returncode == 2
