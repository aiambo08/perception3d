#!/usr/bin/env bash
# F7 dual-rate loop on a BDD 720p clip (WSL + TensorRT).
#
#   bash scripts/run_bdd_f7_wsl.sh
#   DURATION=60 DEPTH_ENGINE=models/depth_840x252_fp16.engine bash scripts/run_bdd_f7_wsl.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=wsl_env.sh
source "$HERE/wsl_env.sh"

REPO="$(cd "$HERE/.." && pwd)"
DURATION="${DURATION:-60}"
HZ="${HZ:-30}"
HOLD="${HOLD:-8}"
KEYFRAMES="${KEYFRAMES:-90}"
VIDEO="${VIDEO:-$REPO/data/outputs/bdd_demo_hold_720p30.mp4}"
REPORT="${REPORT:-$REPO/reports/f7_bdd.json}"
DET_ENGINE="${DET_ENGINE:-models/detector_1024x320_fp16.engine}"
MODELS_CFG="${MODELS_CFG:-configs/models.yaml}"
# Prefer locally rebuilt 840x252 when present; else 924x280.
if [[ -z "${DEPTH_ENGINE:-}" ]]; then
  if [[ -f "$REPO/models/depth_840x252_fp16.engine" ]]; then
    DEPTH_ENGINE=models/depth_840x252_fp16.engine
  else
    DEPTH_ENGINE=models/depth_924x280_fp16.engine
  fi
fi
DEPTH_N="${DEPTH_N:-1 1 4}"
DEPTH_MARGIN="${DEPTH_MARGIN:-1.0}"
FORCE_VIDEO="${FORCE_VIDEO:-0}"
cd "$REPO"

echo "=== demo video (hold=${HOLD}) ==="
if [[ ! -f "$VIDEO" || "$FORCE_VIDEO" == "1" ]]; then
  uv run --extra runtime python scripts/make_bdd_demo_video.py \
    --root "$BDD" --out "$VIDEO" --frames "$KEYFRAMES" --hold "$HOLD" --fps 30
fi
ls -lh "$VIDEO"

echo "=== F7 loop ${HZ} Hz × ${DURATION}s  det=${DET_ENGINE} models=${MODELS_CFG} depth=${DEPTH_ENGINE} n=(${DEPTH_N}) margin=${DEPTH_MARGIN} ==="
# shellcheck disable=SC2086
uv run --extra runtime python scripts/run_pipeline.py \
  --video "$VIDEO" \
  --camera configs/camera_bdd.yaml \
  --models "$MODELS_CFG" \
  --boxes detector \
  --det-engine "$DET_ENGINE" \
  --depth-engine "$DEPTH_ENGINE" \
  --depth-n $DEPTH_N \
  --depth-margin "$DEPTH_MARGIN" \
  --hz "$HZ" \
  --duration "$DURATION" \
  --loop \
  --telemetry none \
  --json "$REPORT" \
  "$@"

python3 -c "
import json
from pathlib import Path
d = json.loads(Path(r'''$REPORT''').read_text())
e = (d.get('stats') or {}).get('e2e_alert_ms') or {}
dep = (d.get('stats') or {}).get('depth') or {}
print(f\"capture→alert P50/P95/P99 = {e.get('p50'):.2f} / {e.get('p95'):.2f} / {e.get('p99'):.2f} ms\")
print(f\"depth Hz = {dep.get('hz')}  age P95 = {(dep.get('age_ms') or {}).get('p95')}\")
print('verdicts:', {k: ('PASS' if v else ('FAIL' if v is False else 'n/a')) for k,v in (d.get('verdicts') or {}).items()})
"
