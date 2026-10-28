#!/usr/bin/env bash
# Descarga BDD100K desde el espejo Berkeley (HTTP) que sigue online.
#
# Uso:
#   bash scripts/download_bdd100k.sh eval     # labels 2018→det_20 + images 100k (~5.5 GB)
#   bash scripts/download_bdd100k.sh labels   # solo labels (+ conversión val)
#   bash scripts/download_bdd100k.sh images   # solo images 100k
#
# Destino: ~/datasets/bdd100k  (override: DEST=...)
# Espejo: http://dl.yf.io/bdd100k/  (UC Berkeley / Fisher Yu)
#
# Nota (2026): el ZIP "bdd100k_det_20_labels.zip" del portal está mal
# empaquetado (2k JPG). Usamos bdd100k_labels.zip (2018) y
# scripts/convert_bdd_labels_2018_to_det20.py → labels/det_20/det_val.json.
# El espejo ETH (dl.cv.ethz.ch) está caído (NXDOMAIN).
set -euo pipefail

DEST="${DEST:-$HOME/datasets/bdd100k}"
BASE="${BDD100K_MIRROR:-http://dl.yf.io/bdd100k}"
MODE=${1:-eval}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
PY=${PY:-"uv run --no-sync python"}

mkdir -p "$DEST"
cd "$DEST"

need_cmd() { command -v "$1" >/dev/null || { echo "falta $1" >&2; exit 1; }; }
need_cmd curl
need_cmd unzip || need_cmd tar

fetch() {
  local z=$1
  echo "→ $BASE/$z"
  curl -L --retry 5 -C - --progress-bar -o "$z" "$BASE/$z"
  if command -v unzip >/dev/null; then
    unzip -tq "$z" >/dev/null || { echo "$z corrupto" >&2; exit 1; }
    unzip -nq "$z"
  else
    tar -xf "$z"
  fi
  echo "OK $z"
}

flatten_bdd() {
  if [[ -d bdd100k/images ]] || [[ -d bdd100k/labels ]]; then
    shopt -s dotglob nullglob
    mv -n bdd100k/* . 2>/dev/null || true
    rmdir bdd100k 2>/dev/null || true
  fi
  # labels zip extracts as 100k/{train,val,test}/*.json
  if [[ -d 100k/val && ! -d labels/100k ]]; then
    mkdir -p labels
    mv 100k labels/100k
  fi
  # images zip may extract as train/val/test or images/100k
  if [[ -d train && ! -d images/100k ]]; then
    mkdir -p images/100k
    mv train images/100k/train
    [[ -d val ]] && mv val images/100k/val
    [[ -d test ]] && mv test images/100k/test
  fi
  if [[ -d 100k/val && -f 100k/val/*.jpg ]]; then
    mkdir -p images
    mv 100k images/100k
  fi
}

convert_val() {
  $PY "$ROOT/scripts/convert_bdd_labels_2018_to_det20.py" --root "$DEST" --split val
}

case "$MODE" in
  labels)
    curl -fsSL -o LICENSE.txt "$BASE/LICENSE.txt" || true
    fetch bdd100k_labels.zip
    flatten_bdd
    convert_val
    ;;
  images)
    fetch bdd100k_images_100k.zip
    flatten_bdd
    ;;
  eval)
    curl -fsSL -o LICENSE.txt "$BASE/LICENSE.txt" || true
    fetch bdd100k_labels.zip
    flatten_bdd
    convert_val
    fetch bdd100k_images_100k.zip
    flatten_bdd
    ;;
  *)
    echo "modo: eval | labels | images" >&2
    exit 1
    ;;
esac

echo ""
echo "Raíz: $DEST"
echo "  export BDD=$DEST"
echo "  uv run python scripts/eval_detector_bdd.py --root \$BDD --json reports/f2_bdd_val.json"
