# percepcion3d

Pipeline de percepción espacial 3D monocular en tiempo real (edge, GPU Ada 8 GB,
Linux/WSL2). Transforma píxeles en coordenadas métricas $X, Y, Z$, velocidad
relativa y alertas de colisión (TTC/CPA) con un diseño asíncrono *dual-rate*.

## Documentación

- [`docs/00_analisis_critico.md`](docs/00_analisis_critico.md) — riesgos,
  trade-offs y mitigaciones por etapa; hallazgos sobre el código actual.
- [`docs/01_plan_fases_mvp.md`](docs/01_plan_fases_mvp.md) — fases F0–F8,
  Definition of Done y estrategia de medición (P50/P95/P99, VRAM).

## Estado

Fase 0 + F0.1 (geometría con roll, intersección con el suelo vectorizada y
modelo de varianza, calibración, rectificación) y F1 (medición P50/P95/P99 y
VRAM, `LatestFrameSlot`, fuentes KITTI/vídeo/V4L2, escena sintética, GT y
métricas KITTI) y F2 (detector 2D: letterbox rectangular con inversa exacta,
cirugía ONNX con preprocesado uint8 + `EfficientNMS_TRT`, `TrtEngine`
asíncrono y `Detector` con backend inyectable, recall COCO→KITTI) y F3
(profundidad relativa: Depth Anything V2-Small → ONNX con normalización
ImageNet en grafo, `DepthEstimator.infer_async` → `DepthMap` fp16 con
metadatos de frame, lazo de contención detector+depth en dos streams, sanidad
Spearman disparidad vs 1/Z LiDAR) y F4 (fusión métrica CPU/NumPy: ajuste
afín robusto $(s,t)$ disparidad↔$1/Z_c$ sobre la calzada con Kalman y gating
$\chi^2$, mediana/MAD con bimodalidad por caja, BLUE en profundidad inversa de
suelo + red + altura con error de pitch correlado, pitch en línea desde
alturas de clase, `Measurement3D` con flags y timestamps) implementadas y
testeadas en CPU/ONNX Runtime. Las métricas GPU de F2 y F3 (P95/P99, VRAM,
recall, Spearman) y el DoD KITTI de F4 (AbsRel por bins) están pendientes de
medirse en la GPU objetivo; los DoD sintéticos y de coste CPU de F4 se cumplen
en local (`scripts/eval_fusion_synthetic.py`). El resto de módulos
(`tracking/`, `safety/`, `runtime/pipeline.py`) son esqueletos pendientes de
las fases F5–F7.

```bash
uv run python scripts/profile_stage.py --stage rectify           # P50/P95/P99 de una etapa
uv run python scripts/profile_stage.py --stage playback --hz 60  # replay + jitter
uv run python scripts/profile_stage.py --stage playback --kitti <drive_sync> --hz 60
```

Detector (F2), en la máquina con GPU:

```bash
uv pip install -e ".[export]"
uv run python scripts/export_detector.py --config configs/models.yaml   # YOLO → ONNX (uint8 + NMS)
uv pip install -e ".[runtime]"
bash scripts/export_trt.sh models/detector_1024x320.onnx models/detector_1024x320_fp16.engine fp16
uv run python scripts/bench_detector.py --kitti <image_02/0000> --labels <label_02/0000.txt> \
    --frames 200 --json reports/f2_detector.json                        # P50/P95/P99 + VRAM + recall
```

Profundidad (F3), en la máquina con GPU:

```bash
uv run python scripts/export_depth.py --config configs/models.yaml       # HF → ONNX (uint8 + ImageNet) ×3 tamaños
for s in 924x280 840x252 1064x322; do
  bash scripts/export_trt.sh models/depth_$s.onnx models/depth_${s}_fp16.engine fp16
done
uv run python scripts/bench_depth.py --mode depth --frames 300           # P50/P95/P99 + VRAM por tamaño
uv run python scripts/bench_depth.py --mode matrix --size 924x280 --pace-hz 60 \
    --kitti-drive <2011_09_26_drive_0005_sync> --frames 150 \
    --json reports/f3_matrix.json                                        # det / depth / ambos + Spearman LiDAR
```

Fusión métrica (F4):

```bash
uv run python scripts/eval_fusion_synthetic.py --json out/f4_synth.json   # DoD sintéticos + P95 CPU (sin GPU)
uv run python scripts/eval_fusion_kitti.py --root <kitti_tracking/training> --seq 0000 \
    --engine models/depth_924x280_fp16.engine --json out/f4_kitti.json    # AbsRel 0–30 / 30–60 m (GPU)
```

## Desarrollo

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"      # núcleo CPU + herramientas
ruff check src tests && ruff format --check src tests
mypy src tests
pytest tests -m "not slow and not gpu"
```

Grupos opcionales (`pyproject.toml`):

| Extra | Contenido | Cuándo |
|---|---|---|
| *(núcleo)* | numpy, opencv-headless, pyyaml, scipy | siempre; CI |
| `runtime` | tensorrt-cu12 10.x, cuda-python, rerun-sdk, nvidia-ml-py | inferencia y VRAM en la GPU objetivo |
| `export` | torch, torchvision, ultralytics, transformers, onnx | exportar ONNX (detector y depth) |
| `dev` | pytest, ruff, mypy, onnx, onnxruntime | desarrollo y CI (tests de cirugía ONNX en CPU) |
