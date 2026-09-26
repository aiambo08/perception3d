#!/usr/bin/env bash
# Descarga uno o varios drives de KITTI Raw (sync) desde el bucket S3 oficial.
#
# Uso:
#   bash scripts/download_kitti_raw.sh                          # drive por defecto: 0005
#   bash scripts/download_kitti_raw.sh 2011_09_26_drive_0009   # drive especifico
#   bash scripts/download_kitti_raw.sh 2011_09_26_drive_0005 2011_09_26_drive_0009
#
# Requisitos: wget, unzip  (sudo apt install -y wget unzip)
#
# Estructura resultante:
#   ~/datasets/kitti/raw/
#     2011_09_26/
#       calib_cam_to_cam.txt
#       calib_velo_to_cam.txt
#       calib_imu_to_velo.txt
#       2011_09_26_drive_0005_sync/
#         image_02/data/*.png
#         image_02/timestamps.txt
#         velodyne_points/data/*.bin
#         oxts/data/*.txt            (opcional)
#
# Variable de entorno:
#   export KR=~/datasets/kitti/raw/2011_09_26/2011_09_26_drive_0005_sync
#
# Uso en scripts del proyecto:
#   uv run python scripts/bench_depth.py --mode matrix \
#       --kitti-drive "$KR" --frames 150

set -euo pipefail

DEST="$HOME/datasets/kitti/raw"
B=https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data
DRIVES=("${@:-2011_09_26_drive_0005}")

# Validar nombres de drive (formato: YYYY_MM_DD_drive_XXXX, sin _sync)
for d in "${DRIVES[@]}"; do
  [[ $d =~ ^20[0-9]{2}_[0-9]{2}_[0-9]{2}_drive_[0-9]{4}$ ]] \
    || { echo "Nombre de drive invalido: '$d'  (formato: 2011_09_26_drive_0005)" >&2; exit 1; }
done

mkdir -p "$DEST" && cd "$DEST"

# Preflight: ~5 GB por drive (imagenes + LiDAR + ZIPs temporales)
need_gb=$(( 5 * ${#DRIVES[@]} ))
avail_gb=$(df --output=avail -BG . | tail -1 | tr -dc 0-9)
(( avail_gb >= need_gb )) \
  || { echo "Poco espacio: ${avail_gb} GB libres, se necesitan ~${need_gb} GB" >&2; exit 1; }

# Descarga con reanudacion, test de integridad y extraccion sin sobrescribir
fetch() {
  local url=$1 z=${1##*/}
  wget --continue --tries=5 --show-progress "$url"
  unzip -tq "$z" >/dev/null || { echo "$z corrupto -- borraló y relanza" >&2; exit 1; }
  unzip -nq "$z"
  sha256sum "$z" >> SHA256SUMS.local
  sort -u -o SHA256SUMS.local SHA256SUMS.local
}

# Contar archivos; devuelve 0 si el directorio no existe
count() { [[ -d $1 ]] && find "$1" -name "$2" | wc -l || printf '0'; }

for d in "${DRIVES[@]}"; do
  date=${d:0:10}   # 2011_09_26

  # Imagenes + LiDAR + OXTS (ZIP sync, carpeta sin _sync en S3, verificado 200 OK)
  fetch "$B/$d/${d}_sync.zip"

  # Calibracion de la fecha (compartida entre drives del mismo dia)
  [[ -f "$date/calib_cam_to_cam.txt" ]] || fetch "$B/${date}_calib.zip"

  # Verificar archivos requeridos por LidarProjector.from_kitti_raw
  for f in calib_cam_to_cam.txt calib_velo_to_cam.txt; do
    [[ -f "$date/$f" ]] || { echo "Falta calibracion: $date/$f" >&2; exit 1; }
  done

  dir="$date/${d}_sync"
  n_img=$(count "$dir/image_02/data"        '*.png')
  n_vel=$(count "$dir/velodyne_points/data" '*.bin')
  n_oxt=$(count "$dir/oxts/data"            '*.txt')

  (( n_img > 0 && n_img == n_vel )) \
    || { echo "$dir incompleto: img=$n_img velo=$n_vel" >&2; exit 1; }

  msg="OK $dir: $n_img frames (imagen + LiDAR"
  (( n_oxt > 0 )) && msg+=", OXTS=$n_oxt"
  echo "$msg)"
done

FIRST_DRIVE="${DRIVES[0]}"
echo ""
echo "Añade a ~/.bashrc:"
echo "  export KR=$DEST/${FIRST_DRIVE:0:10}/${FIRST_DRIVE}_sync"
