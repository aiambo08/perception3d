#!/usr/bin/env bash

export KT="${KT:-/home/aiambo/datasets/kitti/tracking/training}"
SEQS=$(seq -s ' ' -f %04g 0 20)
E="models/depth_924x280_fp16.engine"

echo "========================================================================"
echo "F5 CONVERGENCE SWEEP: 8 RUNS ACROSS 21 KITTI SEQUENCES"
echo "========================================================================"

# 1. Escala de R (Q = 1.0 nominal)
for R in 1 2 4; do
  if [ -f "reports/f5_conv_r${R}.json" ]; then
    echo ">>> [SKIP R=${R}] reports/f5_conv_r${R}.json already exists."
  else
    echo ""
    echo ">>> [RUN R=${R}] --r-scale ${R} -> reports/f5_conv_r${R}.json"
    .venv-linux/bin/python scripts/eval_tracking_kitti.py \
      --root "$KT" \
      --seqs $SEQS \
      --ego oxts \
      --engine "$E" \
      --r-scale "$R" \
      --json "reports/f5_conv_r${R}.json" || true
  fi
done

# 2. Ruido de proceso de vehículos (R = 1.0 nominal)
for Q in 0.25 4; do
  if [ -f "reports/f5_conv_q${Q}.json" ]; then
    echo ">>> [SKIP Q=${Q}] reports/f5_conv_q${Q}.json already exists."
  else
    echo ""
    echo ">>> [RUN Q=${Q}] --q-vehicle ${Q} -> reports/f5_conv_q${Q}.json"
    .venv-linux/bin/python scripts/eval_tracking_kitti.py \
      --root "$KT" \
      --seqs $SEQS \
      --ego oxts \
      --engine "$E" \
      --q-vehicle "$Q" \
      --json "reports/f5_conv_q${Q}.json" || true
  fi
done

# 3. Exploración fina de R reducido con Q=4 y Q=1
for R in 0.25 0.5; do
  if [ -f "reports/f5_conv_r${R}_q4.json" ]; then
    echo ">>> [SKIP R=${R}_q4] reports/f5_conv_r${R}_q4.json already exists."
  else
    echo ""
    echo ">>> [RUN R=${R}_q4] --r-scale ${R} --q-vehicle 4 -> reports/f5_conv_r${R}_q4.json"
    .venv-linux/bin/python scripts/eval_tracking_kitti.py \
      --root "$KT" \
      --seqs $SEQS \
      --ego oxts \
      --engine "$E" \
      --r-scale "$R" \
      --q-vehicle 4 \
      --json "reports/f5_conv_r${R}_q4.json" || true
  fi
done

if [ -f "reports/f5_conv_r0.5_q1.json" ]; then
  echo ">>> [SKIP R=0.5_q1] reports/f5_conv_r0.5_q1.json already exists."
else
  echo ""
  echo ">>> [RUN R=0.5_q1] --r-scale 0.5 -> reports/f5_conv_r0.5_q1.json"
  .venv-linux/bin/python scripts/eval_tracking_kitti.py \
    --root "$KT" \
    --seqs $SEQS \
    --ego oxts \
    --engine "$E" \
    --r-scale 0.5 \
    --json "reports/f5_conv_r0.5_q1.json" || true
fi

echo ""
echo "========================================================================"
echo "ALL CONVERGENCE RUNS COMPLETE!"
echo "========================================================================"
