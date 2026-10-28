#!/usr/bin/env bash
# Export a smaller DepthAnything size for BDD F7 depth-Hz (700×210, multiples of 14).
# ONNX export is CPU; TensorRT build needs the GPU free (run after F2).
#
#   bash scripts/export_depth_bdd_small_wsl.sh          # onnx only
#   bash scripts/export_depth_bdd_small_wsl.sh --trt    # + engine
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=wsl_env.sh
source "$HERE/wsl_env.sh"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"
DO_TRT=0
for a in "$@"; do [[ "$a" == "--trt" ]] && DO_TRT=1; done

SIZE=700x210
# Isolated env so we do not fight the runtime/TRT venv used by F2/F7.
EXPORT_ENV="${UV_PROJECT_ENVIRONMENT}-depth-export"
echo "=== export venv: $EXPORT_ENV ==="
uv venv --python 3.11 "$EXPORT_ENV"
# shellcheck disable=SC1091
source "$EXPORT_ENV/bin/activate"
export UV_PROJECT_ENVIRONMENT="$EXPORT_ENV"

if [[ ! -f "models/depth_${SIZE}.onnx" ]]; then
  echo "=== ONNX export $SIZE (CPU torch) ==="
  uv pip install -e .
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  uv pip install transformers onnx
  python scripts/export_depth.py --config configs/models_bdd.yaml --size "$SIZE"
fi
ls -lh "models/depth_${SIZE}.onnx"

if [[ "$DO_TRT" == "1" ]]; then
  echo "=== TensorRT FP16 $SIZE ==="
  # TRT needs the runtime venv (GPU).
  # shellcheck disable=SC1091
  source "$HERE/wsl_env.sh"
  # shellcheck disable=SC1091
  source "$UV_PROJECT_ENVIRONMENT/bin/activate"
  uv pip install -e '.[runtime]'
  python scripts/export_trt.py \
    "models/depth_${SIZE}.onnx" \
    "models/depth_${SIZE}_fp16.engine" \
    --precision fp16
  ls -lh "models/depth_${SIZE}_fp16.engine"
fi
echo "DONE depth $SIZE"
