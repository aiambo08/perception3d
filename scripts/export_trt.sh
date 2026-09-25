#!/usr/bin/env bash
# Build a TensorRT engine from the surgically-prepared ONNX (run on the target GPU).
#
#   bash scripts/export_trt.sh <in.onnx> <out.engine> [fp16|int8] [extra trtexec args...]
#
# Uses `trtexec` from PATH or $TRTEXEC (the tensorrt-cu12 wheel ships it under
# tensorrt_libs/). Static shapes: the surgery fixes N=1, so no optimization
# profiles are needed; the uint8 NHWC input + Cast is parsed natively by TRT ≥ 8.5.
# trtexec's own timing loop is only a sanity check; the runtime numbers come
# from scripts/bench_detector.py (TrtEngine with its own CUDA Graph).
set -euo pipefail

ONNX=${1:?onnx path}
ENGINE=${2:?engine path}
PRECISION=${3:-fp16}
shift $(( $# >= 3 ? 3 : $# ))

TRTEXEC=${TRTEXEC:-$(command -v trtexec || true)}
if [[ -z "$TRTEXEC" ]]; then
  # tensorrt-cu12 wheels ship trtexec under the package's bin/ but not always on PATH.
  TRTEXEC=$(python -c 'import pathlib,tensorrt_libs as t;print(pathlib.Path(t.__file__).parent/"trtexec")' 2>/dev/null || true)
fi
[[ -x "$TRTEXEC" ]] || { echo "trtexec not found; set TRTEXEC=/path/to/trtexec" >&2; exit 1; }

PREC_FLAGS=()
case "$PRECISION" in
  fp16) PREC_FLAGS=(--fp16) ;;
  int8) PREC_FLAGS=(--fp16 --int8 "--calib=${CALIB_CACHE:?CALIB_CACHE required for int8}") ;;
  fp32) PREC_FLAGS=() ;;
  *) echo "unknown precision $PRECISION" >&2; exit 1 ;;
esac

mkdir -p "$(dirname "$ENGINE")"
"$TRTEXEC" \
  --onnx="$ONNX" \
  --saveEngine="$ENGINE" \
  ${PREC_FLAGS[@]+"${PREC_FLAGS[@]}"} \
  --builderOptimizationLevel=4 \
  --memPoolSize=workspace:1024 \
  --warmUp=500 --iterations=300 --percentile=50,95,99 \
  --useSpinWait --noDataTransfers \
  "$@"

echo "engine written to $ENGINE"
echo "next: uv run python scripts/bench_detector.py --engine $ENGINE"
