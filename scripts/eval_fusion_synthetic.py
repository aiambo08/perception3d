"""F4 synthetic DoD: fusion vs single cues, 1° pitch bias, CPU P95 with 20 boxes.

Runs entirely on CPU (no GPU, no dataset)::

    uv run python scripts/eval_fusion_synthetic.py
    uv run python scripts/eval_fusion_synthetic.py --frames 200 --seed 3 --json out/f4_synth.json
    uv run python scripts/eval_fusion_synthetic.py --no-online-pitch   # ablation
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from percepcion3d.camera.calibration import load_camera_config_yaml  # noqa: E402
from percepcion3d.depth.fusion import load_fusion_config, load_solver_configs  # noqa: E402
from percepcion3d.eval.fusion_synthetic import (  # noqa: E402
    ScenarioResult,
    default_scenarios,
    dod_summary,
    format_results,
    run_scenario,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", type=Path, default=ROOT / "configs" / "camera_kitti.yaml")
    ap.add_argument("--fusion", type=Path, default=ROOT / "configs" / "fusion.yaml")
    ap.add_argument("--frames", type=int, default=None, help="Override frames per scenario")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-online-pitch", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    intr, extr = load_camera_config_yaml(args.camera)
    fcfg = load_fusion_config(args.fusion)
    road, aff, pit = load_solver_configs(args.fusion)

    results: dict[str, ScenarioResult] = {}
    for sc in default_scenarios():
        if args.frames is not None:
            sc = replace(sc, n_frames=args.frames)
        if args.no_online_pitch:
            sc = replace(sc, online_pitch=False)
        results[sc.name] = run_scenario(intr, extr, fcfg, road, aff, pit, sc, seed=args.seed)
    summary = dod_summary(results)
    print(format_results(results, summary))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {"scenarios": {k: v.to_dict() for k, v in results.items()}, "dod": summary},
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
