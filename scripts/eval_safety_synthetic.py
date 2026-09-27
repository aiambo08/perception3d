"""F6 synthetic DoD: alert battery (head-on, static, passing, cut-in, crossing, intermittent).

Runs entirely on CPU (no GPU, no dataset)::

    uv run python scripts/eval_safety_synthetic.py
    uv run python scripts/eval_safety_synthetic.py --seeds 10 --json reports/f6_synth.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from percepcion3d.eval.safety_synthetic import (  # noqa: E402
    SafetyScenarioResult,
    cpu_p95_us,
    default_safety_scenarios,
    is_deterministic,
    run_safety_scenario,
    safety_dod,
)
from percepcion3d.safety.gates import load_safety_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--safety", type=Path, default=ROOT / "configs" / "safety.yaml")
    ap.add_argument("--seeds", type=int, default=5, help="noise realisations per scenario")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_safety_config(args.safety)
    scenarios = default_safety_scenarios(cfg)
    per_seed: list[dict[str, SafetyScenarioResult]] = [
        {sc.name: run_safety_scenario(sc, cfg, seed) for sc in scenarios}
        for seed in range(args.seeds)
    ]
    dods = [safety_dod(r, scenarios) for r in per_seed]
    checks = {sc.name: all(d[sc.name] for d in dods) for sc in scenarios}
    checks["deterministic"] = all(is_deterministic(sc, cfg) for sc in scenarios)
    p95_us = cpu_p95_us(per_seed[0])

    hdr = f"{'scenario':<26}{'peak':>9}{'lead[s]':>9}{'trans/s':>9}{'allowed':>10}{'required':>10}"
    print(hdr)
    for sc in scenarios:
        peaks = {r[sc.name].peak for r in per_seed}
        leads = [r[sc.name].critical_lead_s for r in per_seed]
        tps = max(r[sc.name].transitions_per_s for r in per_seed)
        lead = min((x for x in leads if x is not None), default=None)
        print(
            f"{sc.name:<26}{'/'.join(sorted(p.name[:4] for p in peaks)):>9}"
            f"{(f'{lead:.2f}' if lead is not None else '-'):>9}{tps:>9.2f}"
            f"{sc.max_level.name:>10}{sc.min_level.name:>10}"
        )
    print(f"\nsafety.step per frame [µs]: P95 {p95_us:.1f}")
    print("\nDoD:")
    for k, v in checks.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    ok = all(checks.values())
    if args.json is not None:
        payload: dict[str, Any] = {
            "seeds": args.seeds,
            "scenarios": {sc.name: [r[sc.name].summary() for r in per_seed] for sc in scenarios},
            "step_us_p95": p95_us,
            "dod": {"checks": checks, "pass": ok},
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
