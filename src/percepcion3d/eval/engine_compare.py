"""A/B verdict between two ``bench_detector.py`` reports (F8: FP16 vs INT8).

Acceptance rule from the plan: adopt the candidate engine only if pedestrian
recall drops by at most ``max_recall_drop_pt`` points **and** the detector
latency drops by at least ``min_latency_gain`` (fraction). Verdicts use the
point estimates, as the plan states; each recall delta also carries a 95 %
interval (normal approximation of the difference of two proportions, which is
conservative here because both engines see the same frames) and
``conclusive`` tells whether that interval lies entirely on one side of the
threshold, so a PASS from a handful of pedestrians is visibly weak.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

_Z = 1.96


@dataclass(frozen=True)
class RecallDelta:
    kitti_type: str
    base_recall: float
    cand_recall: float
    n_gt_base: int
    n_gt_cand: int

    @property
    def delta_pt(self) -> float:
        return 100.0 * (self.cand_recall - self.base_recall)

    @property
    def ci95_pt(self) -> tuple[float, float]:
        if self.n_gt_base <= 0 or self.n_gt_cand <= 0:
            return float("nan"), float("nan")
        var = self.base_recall * (1.0 - self.base_recall) / self.n_gt_base
        var += self.cand_recall * (1.0 - self.cand_recall) / self.n_gt_cand
        half = 100.0 * _Z * math.sqrt(var)
        return self.delta_pt - half, self.delta_pt + half

    def verdict(self, max_drop_pt: float) -> str:
        if self.n_gt_base <= 0 or self.n_gt_cand <= 0:
            return "NO-DATA"
        return "PASS" if self.delta_pt >= -max_drop_pt else "FAIL"

    def conclusive(self, max_drop_pt: float) -> bool:
        """``True`` when the 95 % interval does not straddle ``-max_drop_pt``."""
        lo, hi = self.ci95_pt
        return not math.isnan(lo) and (lo >= -max_drop_pt or hi < -max_drop_pt)


@dataclass(frozen=True)
class LatencyDelta:
    stage: str
    percentile: str
    base_ms: float
    cand_ms: float

    @property
    def gain(self) -> float:
        """Fractional reduction: 0.3 means the candidate is 30 % faster."""
        return (self.base_ms - self.cand_ms) / self.base_ms if self.base_ms > 0 else float("nan")

    def verdict(self, min_gain: float) -> str:
        if math.isnan(self.gain):
            return "NO-DATA"
        return "PASS" if self.gain >= min_gain else "FAIL"


@dataclass
class CompareResult:
    recall: dict[str, RecallDelta]
    latency: dict[str, LatencyDelta]
    verdicts: dict[str, str] = field(default_factory=dict)
    accept: bool = False
    recall_conclusive: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "recall": {
                k: {**asdict(r), "delta_pt": r.delta_pt, "ci95_pt": list(r.ci95_pt)}
                for k, r in self.recall.items()
            },
            "latency": {k: {**asdict(d), "gain": d.gain} for k, d in self.latency.items()},
            "verdicts": dict(self.verdicts),
            "accept": self.accept,
            "recall_conclusive": self.recall_conclusive,
        }


def _pooled(report: dict[str, Any], kitti_type: str) -> tuple[float, int]:
    row = report.get("recall", {}).get("pooled", {}).get(kitti_type)
    if row is None:
        return float("nan"), 0
    return float(row["recall"]), int(row["n_gt"])


def _stage(report: dict[str, Any], stage: str, key: str) -> float:
    st = report.get("stages", {}).get(stage)
    return float(st[key]) if st is not None and key in st else float("nan")


def compare_reports(
    base: dict[str, Any],
    cand: dict[str, Any],
    *,
    max_recall_drop_pt: float = 2.0,
    min_latency_gain: float = 0.25,
    recall_gate_type: str = "Pedestrian",
    latency_stage: str = "det.gpu",
    latency_key: str = "p95_ms",
    extra_types: tuple[str, ...] = ("Car",),
) -> CompareResult:
    """Apply the F8 acceptance rule to two ``bench_detector.py`` JSON reports."""
    recall: dict[str, RecallDelta] = {}
    for t in (recall_gate_type, *extra_types):
        rb, nb = _pooled(base, t)
        rc, nc = _pooled(cand, t)
        recall[t] = RecallDelta(t, rb, rc, nb, nc)
    latency = {
        s: LatencyDelta(s, latency_key, _stage(base, s, latency_key), _stage(cand, s, latency_key))
        for s in (latency_stage, "detector_e2e")
    }
    verdicts = {
        f"recall_{recall_gate_type.lower()}": recall[recall_gate_type].verdict(max_recall_drop_pt),
        f"latency_{latency_stage}": latency[latency_stage].verdict(min_latency_gain),
    }
    accept = all(v == "PASS" for v in verdicts.values())
    conclusive = recall[recall_gate_type].conclusive(max_recall_drop_pt)
    return CompareResult(recall, latency, verdicts, accept, conclusive)


def format_compare(res: CompareResult) -> str:
    rows = [f"{'recall':<12}{'base':>8}{'cand':>8}{'Δpt':>8}{'CI95 (pt)':>20}"]
    for r in res.recall.values():
        lo, hi = r.ci95_pt
        rows.append(
            f"{r.kitti_type:<12}{r.base_recall:>8.3f}{r.cand_recall:>8.3f}{r.delta_pt:>+8.2f}"
            f"{f'[{lo:+.2f}, {hi:+.2f}]':>20}  n={r.n_gt_base}/{r.n_gt_cand}"
        )
    rows.append("")
    rows.append(f"{'latency':<16}{'base ms':>9}{'cand ms':>9}{'gain':>8}")
    for d in res.latency.values():
        rows.append(
            f"{d.stage + ' ' + d.percentile:<16}{d.base_ms:>9.3f}{d.cand_ms:>9.3f}{d.gain:>+8.1%}"
        )
    rows.append("")
    rows.append(f"verdicts: {res.verdicts} → {'ACCEPT' if res.accept else 'REJECT'}")
    if not res.recall_conclusive:
        rows.append("note: the recall CI95 straddles the threshold; more frames would firm this up")
    return "\n".join(rows)
