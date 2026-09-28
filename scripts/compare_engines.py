"""F8 A/B: decide whether an INT8 detector engine replaces the FP16 one.

Both inputs are ``bench_detector.py --json`` reports measured on the same
frames (``--kitti-root … --seqs …``)::

    uv run python scripts/bench_detector.py --engine models/detector_1024x320_fp16.engine \\
        --kitti-root "$KT" --seqs 0000 0001 0020 --frames 200 --json reports/det_fp16.json
    uv run python scripts/bench_detector.py --engine models/detector_1024x320_int8.engine \\
        --kitti-root "$KT" --seqs 0000 0001 0020 --frames 200 --json reports/det_int8.json
    uv run python scripts/compare_engines.py reports/det_fp16.json reports/det_int8.json \\
        --json reports/det_fp16_vs_int8.json

Exit code 0 = ACCEPT, 1 = REJECT (or MARGINAL), 2 = bad input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from percepcion3d.eval.engine_compare import compare_reports, format_compare  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("base", type=Path, help="bench_detector.py JSON of the reference engine (FP16)")
    ap.add_argument("cand", type=Path, help="bench_detector.py JSON of the candidate engine (INT8)")
    ap.add_argument("--max-recall-drop-pt", type=float, default=2.0)
    ap.add_argument("--min-latency-gain", type=float, default=0.25)
    ap.add_argument("--recall-type", default="Pedestrian")
    ap.add_argument("--latency-stage", default="det.gpu")
    ap.add_argument("--latency-key", default="p95_ms", choices=("p50_ms", "p95_ms", "p99_ms"))
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    try:
        base = json.loads(args.base.read_text(encoding="utf-8"))
        cand = json.loads(args.cand.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read reports: {exc}", file=sys.stderr)
        return 2

    res = compare_reports(
        base,
        cand,
        max_recall_drop_pt=args.max_recall_drop_pt,
        min_latency_gain=args.min_latency_gain,
        recall_gate_type=args.recall_type,
        latency_stage=args.latency_stage,
        latency_key=args.latency_key,
    )
    print(f"base: {base.get('engine')}\ncand: {cand.get('engine')}\n")
    print(format_compare(res))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        out = {"base": str(args.base), "cand": str(args.cand), **res.to_dict()}
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0 if res.accept else 1


if __name__ == "__main__":
    raise SystemExit(main())
