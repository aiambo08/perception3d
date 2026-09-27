"""Build a TensorRT engine from a surgically-prepared ONNX using the Python API.

Drop-in replacement for scripts/export_trt.sh when trtexec is not available
(e.g. tensorrt-cu12 pip wheel, which does not ship the trtexec binary).

Usage::

    # fp16 (default)
    uv run python scripts/export_trt.py \
        models/detector_1024x320.onnx \
        models/detector_1024x320_fp16.engine

    # explicit precision
    uv run python scripts/export_trt.py \
        models/depth_924x280.onnx \
        models/depth_924x280_fp16.engine --precision fp16

    # fp32
    uv run python scripts/export_trt.py in.onnx out.engine --precision fp32

Exit codes: 0 success, 1 build error, 2 argument error.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _build_engine(
    onnx_path: Path,
    engine_path: Path,
    precision: str,
    workspace_mb: int,
    builder_opt_level: int,
    verbose: bool,
) -> None:
    try:
        import tensorrt as trt  # noqa: PLC0415
    except ImportError as exc:
        print(
            f"tensorrt not importable: {exc}\n"
            "Install with: uv pip install -e '.[runtime]'",
            file=sys.stderr,
        )
        sys.exit(1)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, namespace="")

    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    print(f"Parsing ONNX: {onnx_path}")
    with onnx_path.open("rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error {i}: {parser.get_error(i)}", file=sys.stderr)
            sys.exit(1)
    print(f"  inputs : {[network.get_input(i).name for i in range(network.num_inputs)]}")
    print(f"  outputs: {[network.get_output(i).name for i in range(network.num_outputs)]}")

    config = builder.create_builder_config()
    # workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)
    # builder optimisation level (TRT >= 8.6)
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = builder_opt_level

    # precision flags
    if precision in ("fp16", "int8"):
        if not builder.platform_has_fast_fp16:
            print("WARNING: GPU does not report fast FP16; building anyway.", file=sys.stderr)
        config.set_flag(trt.BuilderFlag.FP16)
    if precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)

    print(f"Building engine (precision={precision}, workspace={workspace_mb} MB) …")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    elapsed = time.perf_counter() - t0

    if serialized is None:
        print("Engine build failed (serialized is None).", file=sys.stderr)
        sys.exit(1)

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    size_mb = engine_path.stat().st_size / 1024 / 1024
    print(f"Engine written to {engine_path}  ({size_mb:.1f} MB, built in {elapsed:.1f}s)")
    print(f"Next: uv run python scripts/bench_detector.py --engine {engine_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build TensorRT engine from ONNX (Python API, no trtexec needed)."
    )
    ap.add_argument("onnx", type=Path, help="Input ONNX path")
    ap.add_argument("engine", type=Path, help="Output .engine path")
    ap.add_argument(
        "--precision",
        choices=("fp16", "fp32", "int8"),
        default="fp16",
        help="Build precision (default: fp16)",
    )
    ap.add_argument(
        "--workspace-mb",
        type=int,
        default=1024,
        help="Builder workspace in MB (default: 1024)",
    )
    ap.add_argument(
        "--opt-level",
        type=int,
        default=4,
        choices=range(6),
        metavar="{0-5}",
        help="Builder optimization level (default: 4)",
    )
    ap.add_argument("--verbose", action="store_true", help="Enable TRT verbose logging")
    args = ap.parse_args()

    if not args.onnx.is_file():
        print(f"ONNX not found: {args.onnx}", file=sys.stderr)
        sys.exit(2)

    _build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        precision=args.precision,
        workspace_mb=args.workspace_mb,
        builder_opt_level=args.opt_level,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
