"""Archive benchmark JSON reports and check the newest run for regressions (F8).

    # archive (the bench_* / run_pipeline scripts do this with --archive)
    uv run python scripts/bench_history.py add reports/det_fp16.json --name detector

    # list what is archived
    uv run python scripts/bench_history.py list [--name detector]

    # compare the two newest 'pipeline' runs (exit 1 if any metric regressed)
    uv run python scripts/bench_history.py check --name pipeline --rel-tol 0.10

    # compare an arbitrary pair
    uv run python scripts/bench_history.py check --prev a.json --curr b.json

Archive root: data/outputs/bench (override with --root). Files are named
``<YYYYMMDD_HHMMSS>_<name>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from percepcion3d.utils.bench_history import (  # noqa: E402
    DEFAULT_ROOT,
    archive_report,
    find_regressions,
    flatten_metrics,
    format_regressions,
    list_reports,
    load_report,
)


def _cmd_add(args: argparse.Namespace) -> int:
    path = archive_report(load_report(args.report), args.name, args.root)
    print(f"archived {path}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    paths = list_reports(args.root, args.name)
    if not paths:
        print("no archived reports")
        return 0
    for p in paths:
        n = len(flatten_metrics(load_report(p)))
        print(f"{p.name}  ({n} metrics)")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    if (args.prev is None) != (args.curr is None):
        print("--prev and --curr go together", file=sys.stderr)
        return 2
    if args.prev is not None:
        prev_path, curr_path = args.prev, args.curr
    else:
        if args.name is None:
            print("check needs --name or --prev/--curr", file=sys.stderr)
            return 2
        paths = list_reports(args.root, args.name)
        if len(paths) < 2:
            print(f"need at least two archived '{args.name}' reports, have {len(paths)}")
            return 0
        prev_path, curr_path = paths[-2], paths[-1]
    prev, curr = load_report(prev_path), load_report(curr_path)
    regs = find_regressions(prev, curr, rel_tol=args.rel_tol, abs_tol=args.abs_tol)
    print(f"prev: {prev_path}\ncurr: {curr_path}\n")
    print(format_regressions(regs))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "prev": str(prev_path),
            "curr": str(curr_path),
            "rel_tol": args.rel_tol,
            "abs_tol": args.abs_tol,
            "regressions": [r.to_dict() for r in regs],
        }
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 1 if regs else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="archive a report")
    a.add_argument("report", type=Path)
    a.add_argument("--name", required=True, help="bench name, e.g. detector | depth | pipeline")
    a.set_defaults(fn=_cmd_add)

    ls = sub.add_parser("list", help="list archived reports")
    ls.add_argument("--name")
    ls.set_defaults(fn=_cmd_list)

    c = sub.add_parser("check", help="regressions between two runs")
    c.add_argument("--name")
    c.add_argument("--prev", type=Path)
    c.add_argument("--curr", type=Path)
    c.add_argument("--rel-tol", type=float, default=0.10, help="relative worsening tolerated")
    c.add_argument(
        "--abs-tol", type=float, default=0.2, help="absolute worsening tolerated (ms, MB, pt)"
    )
    c.add_argument("--json", type=Path)
    c.set_defaults(fn=_cmd_check)

    args = ap.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
