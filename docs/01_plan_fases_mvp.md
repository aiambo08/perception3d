# 01 · Plan modular por fases (MVPs aislables)

> Entregable 2 (§5.2 del brief). Cada fase produce un MVP que se prueba de
> forma aislada (tests CPU en CI, benchmarks GPU en local) antes de integrarse
> en el lazo asíncrono. Los *Definition of Done* (DoD) son medibles; cuando
> dependen del hardware se expresan como percentiles. Las referencias
> R*/H* apuntan a `00_analisis_critico.md`.

## Mapa de fases y dependencias

```
F0 Geometría ✔ ──► F0.1 correcciones ──► F1 Infra de medición y E/S
                                            │
                          ┌─────────────────┼──────────────────┐
                          ▼                 ▼                  │
                  F2 Detector TRT    F3 Profundidad TRT        │
                          │                 │                  │
                          └───────► F4 Fusión métrica ◄────────┘
                                            │
                                    F5 Tracking 3D + ego
                                            │
                                    F6 Seguridad (TTC/CPA)
                                            │
                                    F7 Integración dual-rate + Rerun
                                            │
                                    F8 Endurecimiento (INT8, CI, empaquetado)
```

F2 y F3 son independientes entre sí; F4–F6 pueden desarrollarse **sin GPU**
usando mediciones grabadas (JSON/NPZ) de F2/F3 y el generador sintético de F1.

---

## F0.1 · Correcciones de geometría (cierre de fase 0) — ✔ implementado

**Alcance**
- `compute_ground_distances`: devolver `(z_c, d_long)` con
  $Z_c = h\cos\alpha/\sin(\theta+\alpha)$ (H1); exponer también `ray_length`
  si se necesita.
- Versión vectorizada `PinholeGeometry.ground_hits(u, v) -> GroundHit`
  (`z_c, d_long, x_lat, ray_length, sigma_z, sigma_d, valid`; `nan` en vez de
  excepciones) con el modelo de varianza de R2 parametrizado por
  `GroundNoiseModel(sigma_pitch_rad, sigma_v_px)`. La propagación se hace por
  diferencias centrales sobre la geometría exacta, así es coherente con el roll
  y con la forma cerrada $\sigma_d \approx (d^2/h)\,\sigma_\theta$.
- `undistort_points(uv)` en `ImageRectifier` (opción A de §3.1).
- Roll opcional en `ExtrinsicMountConfig` (por defecto 0) y horizonte como
  recta, no como fila.
- Mover `torch`/`torchvision` a `[project.optional-dependencies].export` (H4) y
  `tensorrt-cu12` (fijado a 10.x), `cuda-python`, `rerun-sdk` a `runtime`
  (R11). El núcleo instalable en CI es puro CPU.

**Nota sobre el test heredado `z_cam > d_long`.** Ese invariante solo es
cierto para la longitud del rayo. Para la profundidad óptica se cumple
$Z_c = d_{long}\cos\theta + h\sin\theta$, que es **menor** que $d_{long}$ a
partir de $d \gtrsim 2h/\theta$ (≈ 19 m con 10° de pitch). El test se
sustituye por los invariantes correctos: `ray_length > d_long`,
`ray_length ≥ z_c` y la identidad anterior.

**DoD**
- Test de valor cerrado: KITTI, `v=374` → `z_c ≈ 5.12 m`, `d_long ≈ 5.06 m`
  (tolerancia 1 cm).
- Test de consistencia: `deproject_pixel_with_depth(u, v, z_c)` devuelve un
  punto con $Y$ igual a la altura de la cámara ±1 mm cuando `pitch = 0`; con
  pitch ≠ 0, tras rotar por $R_x(\theta)$.
- Test de varianza: `sigma_z(d=20 m, σ_θ=0.5°) ≈ 2.2 m` (±5 %).
- CI verde sin torch instalado.

---

## F1 · Infraestructura de medición y entrada/salida — ✔ implementado (CPU)

**Alcance**
- `utils/profiling.py`: `StageTimer` (CPU, `perf_counter_ns`, ring buffer por
  etapa) con reporte P50/P95/P99/max, tabla, CSV/JSON y contexto
  `with timer.stage("detector"):`. Las etapas GPU se alimentan con
  `timer.record(name, ms)` desde CUDA events; el `CudaStageTimer` se añade en
  F2 junto al primer motor TensorRT, cuando pueda medirse en la GPU objetivo.
- `utils/vram.py`: `VramSampler` (hilo a 10 Hz, `sample_fn` inyectable; NVML
  vía `nvidia-ml-py` en el extra `runtime`) → pico de VRAM del proceso y de la GPU.
- `runtime/buffer.py`: `LatestFrameSlot` (un único slot, *latest-wins*,
  contador de frames descartados) y `FrameStamped(frame_id, t_capture_ns, img)`.
  `runtime/playback.py`: `play_into_slot` (reproducción a ritmo fijo con
  P50/P95/P99 de periodo y jitter).
- `io/sources.py`: `FrameSource` (protocolo) con `VideoFileSource`,
  `ImageSequenceSource`, `KittiSequenceSource` (imágenes + timestamps ns +
  calib + OXTS) y `V4L2Source` (MJPG, `CAP_PROP_BUFFERSIZE=1`).
- `sim/synthetic.py`: `SyntheticScene` determinista (`(seed, k)`): cuboides
  a velocidad constante + ego-velocidad → cajas con ruido, fila de contacto,
  inversa de profundidad *afín* (escala/desplazamiento desconocidos) y GT
  3D/velocidad; `ground_inv_depth_map()` denso del suelo. Base de F3–F6.
- `eval/kitti.py`: GT por objeto desde las etiquetas de *tracking* de KITTI
  (`location.z` = profundidad óptica del centro de la caja 3D, sin proyectar
  LiDAR) + `depth_metrics_by_bin` (AbsRel/RMSE/sesgo por bin de distancia).
- `scripts/profile_stage.py --stage {rectify,ground_hits,playback}`
  (`--kitti`, `--video`, `--hz`, `--vram`, `--json/--csv`).

**Medido (CPU de desarrollo, KITTI 1242×375, sin GPU):** `rectify` P50 0.37 ms
/ P99 0.71 ms; `ground_hits` con 20 000 píxeles P50 1.97 ms / P99 2.66 ms;
reproducción a 60 Hz: 60.5 Hz logrados, jitter P99 0.01 ms (sleep + spin de
1.5 ms). Cifras orientativas: la máquina objetivo se mide en F2.

**DoD**
- Reproducir una secuencia KITTI a ≥ 60 Hz (fuente + `LatestFrameSlot`)
  con P99 de jitter entre frames reportado.
- Informe de percentiles generado con un solo comando
  (`scripts/profile_stage.py --stage rectify`).
- 100 % de las utilidades cubiertas por tests CPU.

---

## F2 · Detector 2D en TensorRT — ✔ implementado (CPU/ONNX); métricas GPU pendientes

**Alcance**
- `scripts/export_detector.py` (grupo `export`): YOLO(n|s) → ONNX con
  preprocesado en grafo (uint8 NHWC → float NCHW normalizado) + EfficientNMS →
  `.engine` FP16 con perfil de entrada rectangular (p. ej. 1024×320 para KITTI).
- `detection/detector_trt.py`: `TrtEngine` genérico (carga, buffers
  preasignados, `execute_async_v3`, CUDA Graph opcional, stream/priority) y
  `Detector(engine).infer_async(frame, stream) -> DetectionsHandle` con
  `wait() -> Detections[N,6]` (x1,y1,x2,y2,score,cls) **en coordenadas del
  frame de entrada**, incluyendo la inversa del letterbox.
- Bench `scripts/bench_detector.py`: {aislado} con P50/P95/P99 + VRAM.

**Cómo quedó**
- `detection/letterbox.py`: `Letterboxer` con lienzo `uint8` preasignado
  (relleno 114, centrado, `INTER_AREA` al reducir) y `LetterboxParams`
  con `to_net`/`to_frame`/`clip_to_frame`; la inversa es exacta porque guarda
  las dimensiones redimensionadas reales (escala X e Y independientes).
- `detection/onnx_surgery.py`: `prepend_uint8_preprocess` (entrada
  `images_u8` uint8 `[1,H,W,3]` → Cast → Transpose → Gather BGR→RGB → ×1/255)
  y `append_efficient_nms` (`Transpose` + `Split` → `EfficientNMS_TRT`,
  `box_coding=1`, salidas `num_dets/det_boxes/det_scores/det_classes` de forma
  estática). Ambas verificadas con onnxruntime sobre grafos sintéticos.
- `detection/detector_trt.py`: `Detector(engine: EngineBackend)` con backend
  inyectable (protocolo `EngineBackend`/`InferenceHandle`), de modo que la
  lógica de coordenadas, filtros y temporización se testea en CPU con un motor
  falso. `TrtEngine` importa TensorRT/cuda-python de forma perezosa, usa
  `set_tensor_address` + `execute_async_v3`, host *pinned*, eventos CUDA para
  `gpu_ms` y para `wait()` (sin `cudaDeviceSynchronize`), stream de alta
  prioridad y CUDA Graph opcionales.
- `detection/config.py` + `configs/models.yaml`: pesos, `input_hw`, NMS,
  clases a conservar, mapeo COCO→KITTI y umbrales DoD en un solo sitio.
- `eval/detection.py`: IoU, *matching* codicioso por score y `recall_by_type`
  (COCO→KITTI, GT < 25 px ignorado).
- `scripts/export_trt.sh` (`trtexec`, FP16/INT8) y `scripts/bench_detector.py`
  (P50/P95/P99 de `det.gpu`, `det.postprocess`, `detector_e2e`; VRAM NVML
  baseline/carga/pico; recall opcional con etiquetas KITTI; JSON/CSV).

**DoD**
- P95 ≤ 6 ms aislado en la GPU objetivo (FP16, YOLO-n, 1024×320) **[medir]**;
  si no, documentar y elegir resolución/variante.
- Recall ≥ 0.85 (IoU 0.5) para `Car` y ≥ 0.6 para `Pedestrian` en 200 frames
  KITTI etiquetados (validación de exportación, no del modelo) **[medir]**.
- VRAM del proceso (NVML) ≤ 900 MB con contexto incluido **[medir]**.
- Test CPU: la inversa del letterbox es exacta (round-trip de cajas). ✔

Los tres puntos **[medir]** requieren la GPU objetivo; el entorno de
desarrollo de esta fase no dispone de driver CUDA, así que `TrtEngine` está
escrito contra la API de TensorRT 10 / cuda-python 12 pero **no ejecutado**.
Primera cosa que hacer en la máquina objetivo:

```bash
uv pip install -e ".[export]" && uv run python scripts/export_detector.py
uv pip install -e ".[runtime]" && bash scripts/export_trt.sh \
    models/detector_1024x320.onnx models/detector_1024x320_fp16.engine fp16
uv run python scripts/bench_detector.py --kitti <image_02/0000> \
    --labels <label_02/0000.txt> --frames 200 --json reports/f2_detector.json
```

---

## F3 · Profundidad relativa en TensorRT — implementado (GPU pendiente)

**Alcance** (implementado, PR F3)
- `scripts/export_depth.py`: `depth-anything/Depth-Anything-V2-Small-hf`
  (transformers; extra `export`) → ONNX estático `pixel_values [1,3,H,W]` →
  cirugía `prepend_uint8_preprocess(mean, std)`: entrada `images_u8 [1,H,W,3]`
  BGR, swap RGB, `/255` y normalización ImageNet plegadas en `Mul+Add`. Un
  ONNX por tamaño de `depth.input_sizes` (ancho, múltiplo de 14: 280×924 por
  defecto ≈ 1 320 tokens, 252×840, 322×1064). La variante métrica outdoor
  (`variant: metric_outdoor`, `max_depth: 80` en el checkpoint) usa el mismo
  camino con `kind="metric_depth"`.
- `runtime/trt_engine.py`: `TrtEngine`/`EngineBackend`/`InferenceHandle`
  extraídos del detector y compartidos; `InferenceHandle.ready()`
  (`cudaEventQuery`) permite sondear sin bloquear.
- `depth/preprocess.py`: `DepthResizer` (stretch a resolución de red en un
  canvas uint8 preasignado, sin letterbox: el padding introduciría tokens
  artificiales en la atención global) + `DepthResize` (mapeo exacto
  frame↔red, `frame_to_index`).
- `depth/depth_trt.py`: `DepthEstimator.infer_async(frame, frame_id,
  t_capture_ns) -> DepthHandle` → `DepthMap(frame_id, t_capture_ns,
  values: float16[h,w], kind, resize, gpu_ms)` con `sample(u,v)`,
  `inverse_depth()` y `to_frame_resolution()`. La salida relativa se llama
  `disparity` y nunca se interpreta como distancia: $\hat d = s\,/Z + t$ con
  $(s,t)$ desconocidos por frame (los resuelve F4).
- `runtime/contention.py`: lazo R3 con dos engines — depth encolado **antes**
  que el detector (peor caso para su cola), `wait()` solo del detector,
  `ready()` para recoger el mapa cuando termina; registra `det_e2e`,
  `depth_e2e`, mapas producidos, slots saltados y retraso en frames.
- `eval/lidar.py` + `eval/depth_sanity.py`: proyección Velodyne→imagen
  (KITTI raw y tracking), retorno más cercano por píxel, máscara de calzada
  (bajo horizonte ∧ fuera de cajas), Spearman $\rho$ disparidad vs $1/Z$ y
  AbsRel tras ajuste afín Theil–Sen.
- `scripts/bench_depth.py`: `--mode depth` (todos los tamaños) y
  `--mode matrix` ({detector, depth, ambos}); P50/P95/P99 de `depth.gpu`,
  `depth_e2e`, `det.gpu` bajo contención; VRAM NVML; `--kitti-drive` añade la
  sanidad LiDAR; informe DoD y JSON. Sustituye al proxy EfficientNet-B3 (H2).

**DoD**
- P95 ≤ 25 ms aislado para el tamaño elegido **[medir]**; P99 del
  **detector** bajo contención ≤ 8 ms **[medir]**.
- VRAM total (ambos engines, NVML) ≤ 2.0 GB **[medir]**.
- Correlación de Spearman ≥ 0.95 entre disparidad predicha y 1/Z LiDAR en
  píxeles de calzada **[medir]**. Valida orden relativo, canal BGR/RGB,
  normalización, resize y construcción FP16 — **no** la escala métrica.
  Si falla: `polygraphy run --trt --onnxrt` sobre el ONNX.
- Tests CPU: mapeo frame↔red y `sample()` exactos, `DepthEstimator` con
  engine simulado (fp16, metadatos, `[1,H,W]` y `[1,1,H,W]`), cirugía
  mean/std en ONNX Runtime, proyección LiDAR sintética con $\rho \approx 1$,
  lazo de contención con engines simulados. ✔

Sin driver CUDA en el entorno de desarrollo; primera cosa en la GPU objetivo:

```bash
uv pip install -e ".[export]" && uv run python scripts/export_depth.py
uv pip install -e ".[runtime]"
for s in 924x280 840x252 1064x322; do
  bash scripts/export_trt.sh models/depth_$s.onnx models/depth_${s}_fp16.engine fp16
done
uv run python scripts/bench_depth.py --mode depth --frames 300            # elegir tamaño
uv run python scripts/bench_depth.py --mode matrix --size 924x280 --pace-hz 60 \
    --kitti-drive <2011_09_26_drive_0005_sync> --frames 150 --json reports/f3_matrix.json
```

Decisiones que dependen de la medida: tamaño de entrada (280×924 vs
252×840 si P95 > 25 ms), `depth_every` (1 → 30 Hz a 60 fps del detector si
`depth_e2e` P95 < 33 ms; 2 si no) y si hace falta INT8 para el ViT (en
general no compensa: la atención en FP16 ya domina y la calibración INT8 de
ViT degrada el orden relativo).

---

## F4 · Fusión métrica y restricción del suelo

**Alcance** (todo CPU/NumPy, testeable sin GPU)
- `depth/ground_solver.py`:
  - `RoadMask`: píxeles bajo el horizonte ∧ fuera de cajas ∧ banda central.
  - `AffineScaleSolver`: Theil–Sen/RANSAC de $\hat d$ vs $1/Z_c$ en
    ~2 000 píxeles → $(s,t)$, nº inliers, covarianza; filtro temporal con
    gating $\chi^2$.
  - `PlaneFitter`: plano a los puntos deproyectados de calzada → pitch/roll en
    línea (invariante a escala) → actualiza `ExtrinsicMountConfig` efectivo
    y $\sigma_\theta$.
- `depth/sampling.py`: ROI interior, mediana/MAD, detección de bimodalidad
  (modo de mayor disparidad para clases delgadas), σ de la mediana.
- `depth/fusion.py`: estimadores $Z_g, Z_n, Z_h$ con gates (truncado,
  ocluido, clase sin contacto, $d > d_{max}$) → fusión por varianza inversa
  con test de consistencia → `Measurement3D(X, Z, sigma_X, sigma_Z, flags,
  t_capture_ns)`.
- Config `configs/fusion.yaml`: alturas por clase y σ, $\sigma_\theta$,
  $\sigma_v$, k de la red, umbrales de gates.

**DoD**
- Sintético: con ruido nominal, error de $\hat Z$ ≤ el mejor estimador
  individual en el 95 % de las muestras; con pitch perturbado 1°, la fusión
  degrada < 30 % mientras $Z_g$ solo degrada > 100 %.
- KITTI (mediciones grabadas de F2/F3): AbsRel ≤ 10 % en 0–30 m para `Car`
  con caja no truncada; ≤ 20 % en 30–60 m **[medir]**.
- Coste CPU total por frame (solver + muestreo de 20 cajas) P95 ≤ 2 ms.

---

## F5 · Tracking 3D y ego-motion

**Alcance**
- `tracking/byte_tracker.py`: ByteTrack 2D (IoU, dos umbrales, Hungarian con
  `scipy.optimize.linear_sum_assignment`), ids estables, edad/hits.
- `tracking/kalman_filter.py`: KF lineal CV en $[X, Z, \dot X, \dot Z]$
  con $F(\Delta t)$, $Q(\Delta t)$ por clase, $R_k$ por medición desde
  `Measurement3D` (diseño I); soporte de medición retrasada (R6).
- `tracking/ego_motion.py`: `EgoMotionProvider` (protocolo) con `Zero`,
  `KittiOxts`, `ConstantVelocity`; compensación en la predicción; velocidad
  absoluta derivada y etiqueta estático/móvil con histéresis.
- `tracking/tracker3d.py`: orquestación (asociación 2D → actualización 3D →
  gestión de vida) → `Track3D(id, x, P, cls, age, flags)`.

**DoD**
- Sintético: RMSE de $\dot Z$ ≤ 0.5 m/s a 15 m con ruido nominal tras 1 s
  de track; sin divergencia con $\Delta t$ ∈ [10, 50] ms aleatorio.
- KITTI tracking (3 secuencias): ID switches ≤ ByteTrack de referencia +10 %;
  RMSE de velocidad relativa vs. GT ≤ 1.0 m/s en 0–30 m **[medir]**.
- Con `KittiOxts`, ≥ 90 % de los objetos estáticos del GT etiquetados como
  estáticos tras 0.5 s; con `Zero`, la velocidad relativa no cambia (invariante
  para TTC).
- Coste CPU P95 ≤ 1 ms para 30 tracks.

---

## F6 · Cinemática y seguridad

**Alcance**
- `safety/kinematics.py`: $t_{CPA}, d_{CPA}, TTC_{low}$ con propagación de σ.
- `safety/gates.py`: corredor y compuertas de proximidad/trayectoria por
  configuración (ancho del ego, márgenes, por clase).
- `safety/ttc.py`: máquina de estados pura `step(state, kin, cfg, t_ns) ->
  state` con histéresis, dwell, decaimiento en pérdida de track; salida
  `Alert(track_id, level, ttc_low, d_cpa, reason)`.
- Config `configs/safety.yaml`.

**DoD**
- Batería sintética en CI: colisión frontal (alerta CRITICAL ≥ 1.0 s antes),
  objeto estático a 2 m con ego parado (WARNING/CRITICAL por proximidad),
  adelantamiento lateral a 1.5 m (nunca > CAUTION), cruce que no intersecta
  (nunca > CAUTION), track intermitente (sin parpadeo: ≤ 1 transición por
  segundo con ruido nominal).
- Determinismo: mismas entradas → misma secuencia de estados (test con hash).
- Propiedades (`hypothesis`, opcional): el nivel nunca sube con un track no
  confirmado; el nivel nunca baja antes del dwell mínimo.

---

## F7 · Integración dual-rate y telemetría

**Alcance**
- `runtime/pipeline.py`: lazo mono-hilo con dos streams (R5), `frame_id`
  sellado extremo a extremo, política `depth_every_n_frames` adaptativa,
  buffer de cajas por frame para muestrear mapas antiguos (R6).
- `runtime/cuda.py`: envoltorio de `cuda-python` (streams con prioridad,
  events, pinned buffers, H2D/D2H asíncronos).
- `utils/logger.py` → `telemetry/rerun_sink.py`: visor externo, tiempos de
  captura, tracks 3D, cajas, mapa de profundidad submuestreado cada N frames,
  escalares de latencia por etapa.
- `scripts/run_pipeline.sh` operativo sobre KITTI y vídeo.

**DoD (en la GPU objetivo, [medir])**
- Lazo reactivo (captura → alerta): P99 ≤ 16.7 ms sostenido 5 min.
- Profundidad: ≥ 25 Hz efectivos; antigüedad P95 del mapa usado ≤ 70 ms.
- VRAM pico (NVML, GPU completa menos baseline) ≤ 3.5 GB.
- Frames descartados en `LatestFrameSlot` < 1 % a 60 Hz de entrada.
- Coste del logging P95 ≤ 1 ms; el visor puede desconectarse sin afectar al
  lazo.
- Si el P99 falla por CPU → variante con hilo de captura; si falla por GIL →
  variante multiproceso; ambas medidas con el mismo arnés antes de decidir.

---

## F8 · Endurecimiento

- INT8 en detector con calibración KITTI: aceptar solo si Δrecall peatones
  ≥ −2 pt y Δlatencia ≤ −25 %.
- Imagen `nvidia/cuda` opcional para regenerar engines de forma reproducible
  (no para desarrollo).
- Tests `gpu` ejecutables en local con `pytest -m gpu` y reporte de
  percentiles archivado en `data/outputs/bench/<fecha>.json` (histórico para
  detectar regresiones).
- README con guía de arranque en WSL2 (driver, `usbipd` si cámara en vivo).

---

## Estrategia de medición transversal

| Qué | Cómo | Dónde vive |
|---|---|---|
| Latencia por etapa (CPU) | `perf_counter_ns`, ventana deslizante ≥ 2 000 muestras, P50/P95/P99/max | `utils/profiling.py` |
| Latencia GPU por engine | CUDA events por stream; separar `enqueue` de `execute` | `utils/profiling.py` |
| Contención | Matriz {A}, {B}, {A+B} con el mismo arnés | `scripts/bench_*.py` |
| VRAM | NVML (pico del proceso y de la GPU), no `torch.cuda` | `utils/vram.py` |
| Extremo a extremo | `t_alerta − t_captura` usando el timestamp sellado en el frame | `runtime/pipeline.py` |
| Precisión métrica | AbsRel/RMSE vs. LiDAR por bins de distancia y por clase | `eval/kitti.py` |
| Velocidad y tracking | RMSE vs. GT derivada; ID switches | `eval/kitti.py` |
| Seguridad | Lead time de alerta, falsos positivos/h, transiciones/s | `tests/test_safety.py` + sintético |

Regla operativa: **ninguna decisión de la tabla §4 del análisis crítico se
toma sin su fila correspondiente medida en la GPU objetivo**; los valores del
plan son umbrales de aceptación, no predicciones.
