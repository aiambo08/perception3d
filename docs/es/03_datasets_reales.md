# Datasets reales (más allá de KITTI)

> **Idiomas:** [English](../03_real_datasets.md) (principal) · Español (esta página)


## BDD100K (fase 1 — dashcam)

| Uso | Paquete | Estado del espejo (2026-10) |
|---|---|---|
| F2 recall | `bdd100k_images_100k` + labels 2018→det_20 | OK en `http://dl.yf.io/bdd100k/` |
| F7 lazo 720p | keyframes → `make_bdd_demo_video.py` | OK |
| F5 tracking 2D | MOT 2020 images + `box_track_20` labels | Images OK (`mot20/images20-track-val-1.zip`); **labels offline** (ETH NXDOMAIN, Berkeley sin el ZIP; ver [bdd100k#369](https://github.com/bdd100k/bdd100k/issues/369)) |

Scripts WSL: `scripts/download_bdd100k.sh`, `scripts/download_bdd_mot_wsl.sh`,
`scripts/run_bdd_eval_wsl.sh`, `scripts/run_bdd_f7_wsl.sh`,
`scripts/make_bdd_readme_gif.py`.

## nuScenes mini (fase 2 — métricas 3D no-KITTI)

Solo si un revisor pide AbsRel (F4) / RMSE de velocidad (F5) fuera de KITTI.

| | |
|---|---|
| Licencia | CC BY-NC-SA 4.0 — **registro** en [nuscenes.org](https://www.nuscenes.org/nuscenes) |
| Split | `v1.0-mini` (~4 GB) |
| Qué aporta | 6 cámaras, LiDAR, calibración, ego pose |
| Coste de código | loader + `nuscenes-devkit`; no hay YAML de cámara único |

Hasta tener cuenta/descarga, el repo no embebe el dataset. Checklist cuando
proceda:

1. Registrar y descargar `v1.0-mini` a `~/datasets/nuscenes`.
2. `uv pip install nuscenes-devkit` (extra opcional futuro).
3. Añadir `src/percepcion3d/eval/nuscenes.py` + `configs/camera_nuscenes_*.yaml`
   por cámara (CAM_FRONT primero).
4. Reusar `eval_fusion_kitti.py` / tracking eval con proyecciones LiDAR del
   sample.

No bloquea el camino BDD100K de cámara real 2D / F7.
