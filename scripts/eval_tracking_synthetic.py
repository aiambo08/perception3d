"""F5 synthetic DoD: relative-velocity RMSE, Δt jitter, static labelling, CPU P95 (30 tracks).

Runs entirely on CPU (no GPU, no dataset)::

    uv run python scripts/eval_tracking_synthetic.py
    uv run python scripts/eval_tracking_synthetic.py --seeds 5 --json reports/f5_synth.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from percepcion3d.camera.calibration import load_camera_config_yaml  # noqa: E402
from percepcion3d.camera.geometry import PinholeGeometry  # noqa: E402
from percepcion3d.eval.tracking_synthetic import (  # noqa: E402
    TrackingScenarioResult,
    cpu_benchmark,
    default_tracking_scenarios,
    run_tracking_scenario,
    tracking_dod,
)
from percepcion3d.tracking.tracker3d import load_tracker_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _merge(parts: list[TrackingScenarioResult]) -> TrackingScenarioResult:
    out = TrackingScenarioResult(parts[0].name)
    for r in parts:
        out.n_samples += r.n_samples
        out.vz_err += r.vz_err
        out.vx_err += r.vx_err
        out.nees += r.nees
        out.static_samples += r.static_samples
        out.static_labelled += r.static_labelled
        out.moving_samples += r.moving_samples
        out.moving_as_static += r.moving_as_static
        out.id_switches += r.id_switches
        out.step_ms += r.step_ms
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", type=Path, default=ROOT / "configs" / "camera_kitti.yaml")
    ap.add_argument("--tracking", type=Path, default=ROOT / "configs" / "tracking.yaml")
    ap.add_argument("--seeds", type=int, default=3, help="Monte-Carlo runs per scenario")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    geo = PinholeGeometry(*load_camera_config_yaml(args.camera))
    cfg = load_tracker_config(args.tracking)
    results: dict[str, TrackingScenarioResult] = {}
    for sc in default_tracking_scenarios():
        results[sc.name] = _merge(
            [run_tracking_scenario(geo, sc, cfg, seed=s) for s in range(args.seeds)]
        )
    cpu = cpu_benchmark(geo)
    dod = tracking_dod(results, cpu)

    hdr = f"{'scenario':<24}{'n':>6}{'RMSE vz':>9}{'RMSE vx':>9}{'NEES':>7}{'>χ²':>7}{'static':>8}{'mov→st':>8}{'IDSW':>6}"
    print(hdr)
    for name, r in results.items():
        d = r.to_dict()
        print(
            f"{name:<24}{d['n_scored']:>6}{d['rmse_vz_rel_mps']:>9.3f}{d['rmse_vx_rel_mps']:>9.3f}"
            f"{d['nees_mean']:>7.2f}{d['nees_exceed_frac']:>7.3f}{d['static_frac']:>8.3f}"
            f"{d['moving_as_static_frac']:>8.3f}"
            f"{d['id_switches']:>6}"
        )
    p = np.percentile(cpu, [50, 95, 99])
    print(f"\nTracker3D.step, 30 tracks [ms]: P50 {p[0]:.3f} · P95 {p[1]:.3f} · P99 {p[2]:.3f}")
    print("\nDoD:")
    for k, v in dod["checks"].items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    if args.json is not None:
        payload: dict[str, Any] = {
            "scenarios": {k: v.to_dict() for k, v in results.items()},
            "cpu_ms_p50_p95_p99": [float(x) for x in p],
            "dod": dod,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0 if dod["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
