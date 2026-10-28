#!/usr/bin/env bash
# Run F2 BDD detector eval inside WSL (venv in $HOME + TensorRT).
#
#   bash scripts/run_bdd_eval_wsl.sh
#   bash scripts/run_bdd_eval_wsl.sh --max-images 200 --json reports/f2_bdd_smoke.json
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=wsl_env.sh
source "$HERE/wsl_env.sh"

REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

echo "=== repo: $REPO ==="
echo "=== BDD:  $BDD ==="
echo "=== uv:   $(command -v uv) ==="
test -f "$BDD/labels/det_20/det_val.json" || {
  echo "missing det_val.json under $BDD" >&2
  exit 1
}
echo "val jpgs: $(find "$BDD/images/100k/val" -maxdepth 1 -name '*.jpg' | wc -l)"
ls -1 models/*.engine 2>/dev/null || echo "WARN: no TensorRT engines in models/"

# Ensure TensorRT / NVML extras are present (idempotent).
uv pip install -e '.[runtime]'

JSON_OUT=reports/f2_bdd_val.json
EXTRA=()
if [[ $# -gt 0 ]]; then
  EXTRA=("$@")
else
  EXTRA=(--json "$JSON_OUT")
fi

# If caller did not pass --json, keep the default.
has_json=0
for a in "${EXTRA[@]+"${EXTRA[@]}"}"; do
  [[ "$a" == "--json" ]] && has_json=1
done
if [[ $has_json -eq 0 ]]; then
  EXTRA+=(--json "$JSON_OUT")
fi

exec uv run --extra runtime python scripts/eval_detector_bdd.py \
  --root "$BDD" \
  "${EXTRA[@]}"
