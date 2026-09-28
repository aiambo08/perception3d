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
alturas de clase, `Measurement3D` con flags y timestamps) y F5 (tracking: ByteTrack 2D, KF CV
en $[X, Z, V_X, V_Z]$ con $\Delta t$ variable, $R_k$ desde F4 y mediciones
retrasadas, ego-motion `Zero`/`Constant`/`Oxts`, velocidad relativa y
etiqueta estático/móvil) y F6 (seguridad: $t_{CPA}$/$d_{CPA}$ con σ,
$TTC_{low}$ conservador, compuertas de proximidad/trayectoria y máquina de
alertas pura con histéresis, dwell y decaimiento) y F7 (integración: lazo
dual-rate mono-hilo detector/profundidad con `frame_id` sellado, cadencia de
profundidad adaptativa, buffer de cajas por frame para mapas antiguos,
envoltorio `cuda-python` con streams por prioridad, telemetría asíncrona con
cola acotada y Rerun perezoso, runner KITTI/vídeo con informe de DoD)
implementadas y testeadas en CPU/ONNX Runtime. Las métricas GPU de F2 y F3 (P95/P99, VRAM,
recall, Spearman) y el DoD KITTI de F4 (AbsRel por bins) están pendientes de
medirse en la GPU objetivo; los DoD sintéticos y de coste CPU de F4 se cumplen
en local (`scripts/eval_fusion_synthetic.py`), igual que los sintéticos de F5
(`scripts/eval_tracking_synthetic.py`) y F6 (`scripts/eval_safety_synthetic.py`).
F5 en KITTI cumple estáticos, ID switches y coste CPU, pero no el RMSE de
velocidad en tráfico urbano (0000/0001: 1.2–1.5 m/s frente a 1.0) por un
sesgo del pitch del cue de suelo de F4 — limitación documentada en
`docs/01_plan_fases_mvp.md`. El DoD de F7 (P99 captura→alerta ≤ 16.7 ms,
profundidad ≥ 25 Hz, antigüedad P95 ≤ 70 ms, VRAM ≤ 3.5 GB) sólo puede
medirse en la GPU objetivo con `scripts/run_pipeline.sh`.

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
uv run python scripts/bench_detector.py --kitti-root <tracking/training> --seqs 0000 0001 0020 \
    --frames 200 --json reports/f2_detector.json                        # P50/P95/P99 + VRAM + recall (CI95)
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

Tracking 3D (F5):

```bash
uv run python scripts/eval_tracking_synthetic.py --json out/f5_synth.json  # RMSE V, Δt jitter, estáticos, P95 CPU
uv run python scripts/eval_tracking_kitti.py --root <kitti_tracking/training> \
    --seqs 0000 0001 0020 --ego oxts --json out/f5_kitti.json               # necesita training/oxts/
```

Seguridad (F6):

```bash
uv run python scripts/eval_safety_synthetic.py --seeds 10 --json out/f6_synth.json  # batería de alertas + determinismo
```

Lazo integrado y telemetría (F7):

```bash
# Arnés CPU (cajas GT, sin red): latencias, cadencia y telemetría JSONL, sin GPU
uv run python scripts/run_pipeline.py --kitti-tracking <kitti_tracking/training> --seq 0001 \
    --boxes gt --hz 60 --loop --duration 20 --telemetry jsonl --json out/f7_cpu.json
# DoD en la GPU objetivo: engines F2/F3, 5 min en bucle, visor Rerun (uv pip install -e ".[runtime]")
bash scripts/run_pipeline.sh kitti-tracking <kitti_tracking/training> 0001 --ego oxts
# Variante con hilo de captura (LatestFrameSlot) si el P99 falla por CPU; sin visor
CAPTURE_THREAD=1 TELEMETRY=none bash scripts/run_pipeline.sh kitti-tracking <kitti_tracking/training> 0001
# Vídeo cualquiera (intrínsecas de --camera) o KITTI raw
bash scripts/run_pipeline.sh video <clip.mp4> --camera configs/camera_kitti.yaml
bash scripts/run_pipeline.sh kitti-raw <2011_09_26_drive_0005_sync>
```

El informe JSON (`reports/f7_<modo>.json`) trae `stats` (P50/P95/P99 de
captura→alerta, Hz de profundidad, antigüedad y retraso del mapa, descartes,
coste de `log()`), `stages` (`StageTimer`), `telemetry` (encolados / emitidos /
descartados / errores) y `verdicts` por criterio del DoD (`null` = no medible
en esa configuración, p. ej. VRAM sin GPU).

## Desarrollo

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"      # núcleo CPU + herramientas
ruff check src scripts tests && ruff format --check src scripts tests
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
