"""Benchmark history: archive JSON reports and flag regressions between runs (F8).

Reports from ``bench_detector.py``, ``bench_depth.py`` and ``run_pipeline.py``
are archived as ``data/outputs/bench/<YYYYMMDD_HHMMSS>_<name>.json`` and the
last two runs of the same ``name`` are compared metric by metric. Metrics are
discovered by flattening the report and keeping the numeric leaves whose key
matches :data:`METRIC_PATTERN`; direction (higher or lower is worse) comes from
the key name, so new percentiles or recall fields are picked up automatically.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("data/outputs/bench")
_STAMP = "%Y%m%d_%H%M%S"
_FILE_RE = re.compile(r"^(?P<stamp>\d{8}_\d{6}(?:-\d{2})?)_(?P<name>[A-Za-z0-9._-]+)\.json$")

METRIC_PATTERN = re.compile(
    r"(?:^|[._])(?:p50|p95|p99|max)(?:_ms)?$"  # latencies / ages
    r"|(?:^|[._])(?:frac|fraction)$"  # drop fraction
    r"|(?:^|[._])(?:peak|vram_peak_over_baseline_mb|gpu_peak)$"  # memory
    r"|(?:^|[._])recall$|(?:^|[._])hz$"  # higher is better
)
_HIGHER_IS_BETTER = re.compile(r"(?:^|[._])recall$|(?:^|[._])hz$")


@dataclass(frozen=True)
class Regression:
    metric: str
    prev: float
    curr: float
    rel_change: float
    higher_is_better: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def archive_report(
    report: dict[str, Any],
    name: str,
    root: Path = DEFAULT_ROOT,
    now: datetime | None = None,
) -> Path:
    """Write ``report`` to ``root/<stamp>_<name>.json`` and return the path."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise ValueError(f"invalid bench name {name!r}")
    ts = now if now is not None else datetime.now(UTC)
    root.mkdir(parents=True, exist_ok=True)
    stamp = ts.strftime(_STAMP)
    path = root / f"{stamp}_{name}.json"
    for i in range(1, 100):
        if not path.exists():
            break
        path = root / f"{stamp}-{i:02d}_{name}.json"
    else:
        raise FileExistsError(f"too many '{name}' reports archived at {stamp}")
    payload = {**report, "bench_history": {"name": name, "archived_at": ts.isoformat()}}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def list_reports(root: Path = DEFAULT_ROOT, name: str | None = None) -> list[Path]:
    """Archived reports, oldest first; optionally only one bench ``name``."""
    if not root.is_dir():
        return []
    out: list[tuple[str, Path]] = []
    for p in root.iterdir():
        m = _FILE_RE.match(p.name)
        if m and (name is None or m["name"] == name):
            out.append((m["stamp"], p))
    return [p for _, p in sorted(out)]


def flatten_metrics(report: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Numeric leaves of ``report`` whose dotted key matches :data:`METRIC_PATTERN`."""
    out: dict[str, float] = {}
    for k, v in report.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_metrics(v, key))
        elif isinstance(v, bool):
            continue
        elif isinstance(v, int | float) and METRIC_PATTERN.search(key):
            f = float(v)
            if f == f:  # drop NaN
                out[key] = f
    return out


def find_regressions(
    prev: dict[str, Any],
    curr: dict[str, Any],
    rel_tol: float = 0.10,
    abs_tol: float = 0.2,
) -> list[Regression]:
    """Metrics that got worse by more than ``rel_tol`` (relative) **and** ``abs_tol`` (absolute).

    Latencies, ages, drops and memory regress upward; ``recall``/``hz`` regress
    downward. Metrics missing from either report are ignored.
    """
    a, b = flatten_metrics(prev), flatten_metrics(curr)
    out: list[Regression] = []
    for key in sorted(a.keys() & b.keys()):
        p, c = a[key], b[key]
        hib = bool(_HIGHER_IS_BETTER.search(key))
        worse = (p - c) if hib else (c - p)
        if worse <= abs_tol:
            continue
        rel = worse / abs(p) if p != 0 else float("inf")
        if rel > rel_tol:
            out.append(Regression(key, p, c, rel, hib))
    return out


def format_regressions(regs: list[Regression]) -> str:
    if not regs:
        return "no regressions"
    rows = [f"{'metric':<48}{'prev':>12}{'curr':>12}{'change':>9}"]
    for r in regs:
        rows.append(f"{r.metric:<48}{r.prev:>12.3f}{r.curr:>12.3f}{r.rel_change:>+9.1%}")
    return "\n".join(rows)


def load_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: report must be a JSON object")
    return data
