# Real-world datasets (beyond KITTI)

> **Languages:** English (this page) · [Español](es/03_datasets_reales.md)

## BDD100K (phase 1 — dashcam)

| Use | Package | Mirror status (2026-10) |
|---|---|---|
| F2 recall | `bdd100k_images_100k` + labels 2018→det_20 | OK at `http://dl.yf.io/bdd100k/` |
| F7 720p loop | keyframes → `make_bdd_demo_video.py` | OK |
| F5 2D tracking | MOT 2020 images + `box_track_20` labels | Images OK (`mot20/images20-track-val-1.zip`); **labels offline** (ETH NXDOMAIN, Berkeley missing the ZIP; see [bdd100k#369](https://github.com/bdd100k/bdd100k/issues/369)) |

WSL scripts: `scripts/download_bdd100k.sh`, `scripts/download_bdd_mot_wsl.sh`,
`scripts/run_bdd_eval_wsl.sh`, `scripts/run_bdd_f7_wsl.sh`,
`scripts/make_bdd_readme_gif.py`.

## nuScenes mini (phase 2 — non-KITTI 3D metrics)

Only if a reviewer asks for AbsRel (F4) / velocity RMSE (F5) outside KITTI.

| | |
|---|---|
| License | CC BY-NC-SA 4.0 — **registration** at [nuscenes.org](https://www.nuscenes.org/nuscenes) |
| Split | `v1.0-mini` (~4 GB) |
| Provides | 6 cameras, LiDAR, calibration, ego pose |
| Code cost | loader + `nuscenes-devkit`; no single camera YAML |

Until an account/download exists, the repo does not vendor the dataset. Checklist when
ready:

1. Register and download `v1.0-mini` to `~/datasets/nuscenes`.
2. `uv pip install nuscenes-devkit` (future optional extra).
3. Add `src/percepcion3d/eval/nuscenes.py` + `configs/camera_nuscenes_*.yaml`
   per camera (CAM_FRONT first).
4. Reuse `eval_fusion_kitti.py` / tracking eval with the sample’s LiDAR projections.

Does not block the BDD100K real-camera 2D / F7 path.
