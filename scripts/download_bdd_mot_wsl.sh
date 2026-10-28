#!/usr/bin/env bash
# Download BDD100K MOT 2020 val images (+ labels if a known mirror has them).
#
#   bash scripts/download_bdd_mot_wsl.sh
#   DEST=/mnt/c/Users/$USER/datasets/bdd100k bash scripts/download_bdd_mot_wsl.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=wsl_env.sh
source "$HERE/wsl_env.sh"

DEST="${DEST:-$BDD}"
BASE="${BDD100K_MIRROR:-http://dl.yf.io/bdd100k}"
mkdir -p "$DEST"
cd "$DEST"

fetch() {
  local url=$1 z=$2
  echo "→ $url"
  curl -L --retry 5 -C - --progress-bar -o "$z" "$url"
  unzip -tq "$z" >/dev/null
  unzip -nq "$z"
  echo "OK $z"
}

# Val images (single zip on Berkeley mirror).
fetch "$BASE/mot20/images20-track-val-1.zip" "images20-track-val-1.zip"

# Flatten common layouts → images/track/val/<video>/*.jpg
if [[ -d bdd100k/images/track/val ]]; then
  mkdir -p images/track
  mv -n bdd100k/images/track/val images/track/ 2>/dev/null || true
fi
if [[ -d images/track/val ]]; then
  echo "images: $(find images/track/val -type d | wc -l) video dirs"
elif [[ -d track/val ]]; then
  mkdir -p images
  mv track images/track
  echo "images: $(find images/track/val -type d | wc -l) video dirs"
else
  echo "AVISO: layout de imágenes MOT inesperado bajo $DEST" >&2
  find . -maxdepth 3 -type d | head -40
fi

# Labels: Berkeley does not host box_track_20; try known alternate names then give up.
LABEL_CANDIDATES=(
  "$BASE/bdd100k_box_track_20_labels.zip"
  "$BASE/mot20/bdd100k_box_track_20_labels.zip"
  "$BASE/mot20/box_track_20_labels.zip"
)
got_labels=0
for url in "${LABEL_CANDIDATES[@]}"; do
  z=$(basename "$url")
  code=$(curl -fsSL -o /dev/null -w "%{http_code}" --max-time 15 "$url" || true)
  if [[ "$code" == "200" ]]; then
    fetch "$url" "$z"
    got_labels=1
    break
  fi
  echo "skip labels $url ($code)"
done

if [[ -d bdd100k/labels/box_track_20 ]]; then
  mkdir -p labels
  mv -n bdd100k/labels/box_track_20 labels/ 2>/dev/null || true
fi

if [[ "$got_labels" -eq 0 && ! -d labels/box_track_20/val ]]; then
  echo ""
  echo "MOT images OK; labels box_track_20 AUSENTES en el espejo público."
  echo "Descárgalos desde https://doc.bdd100k.com/download.html (MOT 2020 Labels)"
  echo "y descomprime a: $DEST/labels/box_track_20/val/*.json"
  exit 0
fi

echo "labels: $(find labels/box_track_20/val -name '*.json' 2>/dev/null | wc -l) videos"
echo "export BDD=$DEST"
echo "uv run python scripts/eval_tracking_bdd.py --root \$BDD --engine models/detector_672x384_fp16.engine --max-videos 20 --json reports/f5_bdd_mot.json"
