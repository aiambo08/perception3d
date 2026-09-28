#!/usr/bin/env bash

export KT="${KT:-/home/aiambo/datasets/kitti/tracking/training}"
SEQS=$(seq -s ' ' -f %04g 0 20)
E="models/depth_924x280_fp16.engine"

echo "========================================================================"
echo "F5 CONVERGENCE SWEEP: 5 RUNS ACROSS 21 KITTI SEQUENCES"
echo "========================================================================"

# 1. Escala de R (r1 ya se ejecuto, pero si existe podemos verificar o continuar)
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

# 2. Ruido de proceso de vehículos (por defecto 1.0)
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

echo ""
echo "========================================================================"
echo "ALL 5 CONVERGENCE RUNS COMPLETE!"
echo "========================================================================"
