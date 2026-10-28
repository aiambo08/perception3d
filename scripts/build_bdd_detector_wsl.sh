#!/usr/bin/env bash
# Export YOLO → ONNX (16:9) + TensorRT FP16 for BDD, then optional F2/F7.
# WSL only — open Ubuntu and run from the repo:
#
#   cd "/mnt/c/Users/niaib/OneDrive - Universidad Politécnica de Madrid/106 - PROYECTOS/22. Edge-Ready 3D Spatial Perception & Multi-Object Tracking Pipeline/perception3d"
#   bash scripts/build_bdd_detector_wsl.sh
#   bash scripts/build_bdd_detector_wsl.sh --f2-smoke --f7
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=wsl_env.sh
source "$HERE/wsl_env.sh"

REPO="$(cd "$HERE/.." && pwd)"
CFG="$REPO/configs/models_bdd.yaml"
cd "$REPO"

DO_F7=0
DO_F2=0
for a in "$@"; do
  case "$a" in
    --f7) DO_F7=1 ;;
    --f2-smoke) DO_F2=1 ;;
  esac
done

echo "=== venv: $UV_PROJECT_ENVIRONMENT ==="
echo "=== repo: $REPO ==="
uv venv --python 3.11 "$UV_PROJECT_ENVIRONMENT"
# Activate so we do not fight uv run / --no-sync Python mismatches.
# shellcheck disable=SC1091
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
python -V
which python

echo "=== [1/2] ONNX export (CPU torch + ultralytics) ==="
uv pip install -e .
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install ultralytics onnx onnxslim
python scripts/export_detector.py --config "$CFG"
test -f models/detector_672x384.onnx

echo "=== [2/2] TensorRT FP16 engine ==="
uv pip install -e '.[runtime]'
python scripts/export_trt.py \
  models/detector_672x384.onnx \
  models/detector_672x384_fp16.engine \
  --precision fp16
ls -lh models/detector_672x384*

if [[ "$DO_F2" == "1" ]]; then
  echo "=== F2 smoke (200 images) ==="
  python scripts/eval_detector_bdd.py \
    --root "$BDD" \
    --config "$CFG" \
    --engine models/detector_672x384_fp16.engine \
    --max-images 200 \
    --json reports/f2_bdd_672x384_smoke.json
fi

if [[ "$DO_F7" == "1" ]]; then
  echo "=== F7 remasure ==="
  DET_ENGINE=models/detector_672x384_fp16.engine \
  MODELS_CFG=configs/models_bdd.yaml \
  DEPTH_ENGINE=models/depth_840x252_fp16.engine \
  REPORT=reports/f7_bdd_672x384.json \
  DURATION=60 \
  bash scripts/run_bdd_f7_wsl.sh
fi

echo "DONE: models/detector_672x384_fp16.engine"
