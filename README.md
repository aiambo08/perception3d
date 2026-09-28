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
`docs/01_plan_fases_mvp.md`. El DoD de F7 se midió en la GPU objetivo
(RTX Ada 8 GB bajo WSL2, KITTI 0001 en bucle, 60 s): P99 captura→alerta
9.00 ms (P50 5.93), profundidad 29.9 Hz con antigüedad P95 33 ms, 0.39 % de
descartes, 539 MB de VRAM sobre la base y `log()` P95 0.30 ms; todos los
criterios cumplen (detalle en el plan). F8 (endurecimiento) añade la
calibración INT8 del detector con frames KITTI y su veredicto FP16 frente a
INT8, el histórico de benchmarks con detección de regresiones, los tests
`gpu` de humo, la imagen `nvidia/cuda` para regenerar engines y la guía de
arranque en WSL2 de abajo; las medidas INT8 quedan pendientes de la GPU.

## Arranque en WSL2

Pasos para dejar la máquina lista desde cero (Windows 11 + WSL2 Ubuntu 22.04
o 24.04). Los comandos se ejecutan dentro de WSL salvo que se indique
PowerShell.

1. **Driver NVIDIA en Windows** (no se instala ningún driver dentro de WSL).
   Instala el Game Ready/Studio Driver actual desde nvidia.com y comprueba
   desde WSL que la GPU se ve:
   ```bash
   nvidia-smi          # debe listar la GPU y "CUDA Version: 12.x"; si falla, actualiza el driver de Windows
   ```
   `nvidia-smi` en WSL informa memoria del proceso con menos detalle que en
   Linux nativo (WDDM); `utils/vram.py` usa NVML y devuelve `null` en los
   campos que WDDM no expone.
2. **Herramientas de sistema y `uv`:**
   ```bash
   sudo apt update && sudo apt install -y git libglib2.0-0
   curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
   ```
3. **Repositorio y entorno (Python 3.11 gestionado por uv):**
   ```bash
   git clone https://github.com/aiambo08/perception3d.git ~/perception3d && cd ~/perception3d
   uv venv --python 3.11 && source .venv/bin/activate
   uv pip install -e ".[dev]"                      # CPU: tests, ruff, mypy, onnxruntime
   uv pip install -e ".[runtime]"                  # GPU: tensorrt-cu12 10.x, cuda-python, rerun-sdk, nvidia-ml-py
   uv pip install -e ".[export]"                   # sólo para exportar ONNX (torch, ultralytics, transformers)
   ```
4. **Comprobar TensorRT y cuda-python:**
   ```bash
   python -c "import tensorrt as trt, cuda; print(trt.__version__)"   # 10.x
   pytest tests -q -m "not slow and not gpu"                          # batería CPU
   ```
5. **Datos y engines:**
   ```bash
   export KT=~/datasets/kitti/tracking/training     # image_02/<seq>/, label_02/<seq>.txt, oxts/<seq>.txt
   ls "$KT/image_02/0001" | head -3
   # engines (F2/F3): ver "Detector" y "Profundidad" más abajo; comprobar:
   ls models/detector_1024x320_fp16.engine models/depth_924x280_fp16.engine
   pytest tests -q -m gpu -v                        # humo GPU: una inferencia por engine y lazo F7 de 5 s si KT está definido
   ```
   Guarda el dataset en el disco de Linux (`~/datasets`), no en `/mnt/c`: el
   acceso a NTFS desde WSL es varias veces más lento y aparece en el P99.
6. **Cámara USB en vivo (opcional, `V4L2Source`).** WSL2 no ve USB por
   defecto; se comparte con `usbipd-win`. En PowerShell como administrador:
   ```powershell
   winget install usbipd
   usbipd list                                     # anota el BUSID de la cámara
   usbipd bind --busid <BUSID>
   usbipd attach --wsl --busid <BUSID>             # repetir tras cada reconexión
   ```
   En WSL: `ls /dev/video*` debe mostrar la cámara. El kernel de WSL por
   defecto no trae `uvcvideo`; si no aparece `/dev/video0`, hace falta un
   kernel WSL compilado con UVC (fuera del alcance de este repo). Sin cámara
   todo el pipeline se prueba con KITTI o vídeo.
7. **Visor Rerun.** Bajo WSLg el visor se renderiza con `llvmpipe` (CPU) y
   compite con el lazo (medido: P99 18.8 frente a 12.8 ms). Para medir,
   `TELEMETRY=none` o `RERUN_SAVE=reports/x.rrd`; para ver, abre el visor
   nativo en Windows (`pip install rerun-sdk` y `rerun` en PowerShell) y usa
   `RERUN_CONNECT=<ip-de-windows>:9876` desde WSL.

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

Endurecimiento (F8), en la máquina con GPU:

```bash
# 1. INT8 del detector con calibración de entropía sobre ~500 frames KITTI (guarda el caché)
uv run python scripts/export_trt.py models/detector_1024x320.onnx models/detector_1024x320_int8.engine \
    --precision int8 --calib-dir "$KT/image_02/0000" "$KT/image_02/0001" "$KT/image_02/0020" \
    --calib-frames 500 --calib-cache models/detector_1024x320_int8.cache
# 2. Mismo benchmark con los dos engines (mismas secuencias y frames) y veredicto
uv run python scripts/bench_detector.py --engine models/detector_1024x320_fp16.engine \
    --kitti-root "$KT" --seqs 0000 0001 0020 --frames 200 --json reports/det_fp16.json --archive
uv run python scripts/bench_detector.py --engine models/detector_1024x320_int8.engine \
    --kitti-root "$KT" --seqs 0000 0001 0020 --frames 200 --json reports/det_int8.json --archive
uv run python scripts/compare_engines.py reports/det_fp16.json reports/det_int8.json \
    --json reports/det_fp16_vs_int8.json     # ACCEPT si Δrecall peatones ≥ −2 pt y det.gpu P95 baja ≥ 25 %
# 3. Histórico: --archive en bench_detector/bench_depth/run_pipeline guarda una copia en data/outputs/bench/
uv run python scripts/bench_history.py list
uv run python scripts/bench_history.py check --name pipeline   # regresión si P95/P99/VRAM/descartes empeoran > 10 % y > 0.2
# 4. Tests de humo en GPU (se saltan sin tensorrt/cuda-python o sin engines)
KT="$KT" uv run pytest tests -q -m gpu -v
# 5. Regenerar engines en una imagen nvidia/cuda pinada (requiere NVIDIA Container Toolkit)
docker build -f docker/Dockerfile.trt -t percepcion3d-trt .
docker run --rm --gpus all -v "$PWD/models:/workspace/models" percepcion3d-trt \
    models/detector_1024x320.onnx models/detector_1024x320_fp16.engine --precision fp16
```

`compare_engines.py` devuelve 0 con `ACCEPT` y 1 con `REJECT`; el JSON trae
Δrecall por clase con su IC95 (`recall_conclusive=false` avisa de que el
intervalo cruza el umbral y conviene más frames) y Δlatencia por etapa.
`bench_history.py check` devuelve 1 si hay regresiones y 2 si falta la
entrada; los ficheros `data/outputs/bench/<AAAAMMDD_HHMMSS>_<nombre>.json`
se versionan a propósito (son pequeños) para que el histórico viaje con el
repo.

## Desarrollo

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"      # núcleo CPU + herramientas
ruff check src scripts tests && ruff format --check src scripts tests
mypy src tests
pytest tests -m "not slow and not gpu"
pytest tests -m gpu                                       # sólo con GPU + engines; se salta si faltan
```

Grupos opcionales (`pyproject.toml`):

| Extra | Contenido | Cuándo |
|---|---|---|
| *(núcleo)* | numpy, opencv-headless, pyyaml, scipy | siempre; CI |
| `runtime` | tensorrt-cu12 10.x, cuda-python, rerun-sdk, nvidia-ml-py | inferencia y VRAM en la GPU objetivo |
| `export` | torch, torchvision, ultralytics, transformers, onnx | exportar ONNX (detector y depth) |
| `dev` | pytest, ruff, mypy, onnx, onnxruntime | desarrollo y CI (tests de cirugía ONNX en CPU) |
