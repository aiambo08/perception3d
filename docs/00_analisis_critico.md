# 00 · Análisis crítico previo — Percepción 3D monocular en el edge

> Entregable 1 de la dinámica de trabajo (§5.1 del brief): riesgos, trade-offs y
> mitigaciones **antes** de fijar la arquitectura detallada. Todo lo que aquí se
> afirma como número es una estimación de orden de magnitud o una derivación
> analítica; los valores definitivos salen del arnés de medición de la fase F1
> (ver `01_plan_fases_mvp.md`). Donde se marca **[medir]** la decisión se
> pospone deliberadamente hasta tener percentiles P50/P95/P99 reales.

---

## 0. Estado del repositorio (punto de partida)

| Módulo | Estado | Observaciones |
|---|---|---|
| `camera/calibration.py` | Implementado | Dataclasses inmutables, parser KITTI y YAML. |
| `camera/geometry.py` | Implementado | Proyección/deproyección, horizonte, distancias al suelo. **Ver hallazgo H1.** |
| `camera/undistort.py` | Implementado | LUT `CV_32FC1` precomputada, `cv2.remap` en CPU. |
| `tests/test_geometry.py` | 22 tests verdes | Reversibilidad, monotonía, gating de horizonte. |
| `scripts/benchmark_depth.py` | Implementado | Proxy EfficientNet-B3 con CUDA events. **Ver hallazgo H2.** (Sustituido en F3 por `scripts/bench_depth.py` sobre el engine real.) |
| `detection/`, `depth/`, `tracking/`, `safety/`, `runtime/`, `utils/` | Ficheros vacíos | Solo esqueleto. |
| `configs/models.yaml`, `README.md` | Vacíos | |

Verificado localmente (CPU, sin GPU): `ruff check`, `ruff format --check`,
`mypy --strict` y `pytest -m "not slow and not gpu"` pasan.

### Hallazgos sobre el código existente

**H1 — `compute_ground_distances` devuelve la longitud del rayo, no la profundidad óptica $Z_c$.**
El docstring dice que `z_cam` es la "depth along the optical axis", pero
`h / sin(θ+α)` es la **hipotenusa** (distancia euclídea cámara→punto de contacto).
La profundidad sobre el eje óptico es:

$$
Z_c = \frac{h\,\cos\alpha}{\sin(\theta+\alpha)}, \qquad
d_{long} = \frac{h}{\tan(\theta+\alpha)}, \qquad
\|r\| = \frac{h}{\sin(\theta+\alpha)}
$$

Impacto con la calibración KITTI (`h=1.65 m`, `θ=2.5°`):

| fila `v` | α | `d_long` | `‖r‖` (código) | `Z_c` (correcto) | error |
|---|---|---|---|---|---|
| 200 | 2.2° | 20.27 m | 20.33 m | 20.32 m | 0.07 % |
| 250 | 6.1° | 10.91 m | 11.03 m | 10.97 m | 0.57 % |
| 300 | 10.0° | 7.45 m | 7.63 m | 7.51 m | 1.54 % |
| 374 | 15.6° | 5.06 m | 5.32 m | 5.12 m | 3.81 % |

Es pequeño a media distancia pero **sistemático y máximo justo en el rango
crítico de seguridad (< 8 m)**. Importa porque las redes de profundidad
predicen $Z_c$ por píxel (o su inversa), y la fusión métrica (§3.3) compara
ambas magnitudes: mezclar hipotenusa con $Z_c$ introduce un sesgo de escala
que después el ajuste afín "absorbe" mal (no es constante con $v$). Los tests
actuales no lo detectan porque el invariante `z_cam > d_long` se cumple también
para $Z_c$ ($Z_c/d_{long} = \cos\alpha/\cos(\theta+\alpha) > 1$).
→ Corregir en F0.1 y añadir un test con valor cerrado.

**H2 — El benchmark de profundidad usa un proxy convolucional para un modelo ViT.**
EfficientNet-B3 no reproduce el perfil de cómputo de un ViT-S (atención
cuadrática en tokens, GEMMs grandes, sin convoluciones separables). Los
percentiles obtenidos no son transferibles: el benchmark real debe ejecutar el
**engine TensorRT del modelo candidato**, no un proxy PyTorch. Además,
`torch.cuda.max_memory_allocated` solo ve el allocator de PyTorch; el pico real
de VRAM (contexto CUDA, workspace TRT, cuDNN) hay que leerlo con NVML.
→ Resuelto en F3: `scripts/bench_depth.py` mide el engine TensorRT de Depth
Anything V2-Small (CUDA events + NVML) y la matriz detector+depth.

**H3 — `ImageRectifier` está pensado para trabajar a resolución de sensor en CPU.**
Correcto para KITTI (1242×375, ya rectificado → coeficientes cero) pero a 1080p
`cv2.remap` monohilo cuesta 8–12 ms típicos **[medir]**, incompatible con el
lazo reactivo de 60 Hz (16.7 ms). Ver §3.1.

**H4 — El runtime declara `torch` y `torchvision` como dependencias obligatorias.**
Importar torch en el proceso de inferencia cuesta ~1–2 s de arranque y, si se
toca CUDA, reserva contexto y caché propios (cientos de MB). Debe pasar a un
grupo opcional `export`/`bench`.

---

## 1. Riesgos globales y mitigaciones

Clasificación: **P** = probabilidad, **I** = impacto (A/M/B).

### R1 · Divergencia métrica (P: A, I: A)

*El problema.* Los modelos ligeros de profundidad (Depth Anything V2-Small,
DPT-Hybrid, MiDaS-small) devuelven **disparidad relativa afín-invariante**:
$\hat d = s\cdot\frac{1}{Z} + t$ con $s, t$ desconocidos y **variables frame
a frame**. Un único punto de contacto con el suelo da una ecuación para dos
incógnitas. Si se asume $t = 0$ (solo escala) el error se dispara en los
extremos del rango; si se estiman $s,t$ a partir de pocas cajas, el ajuste es
frágil frente a una detección errónea.

*Mitigación propuesta.*
1. **Ajustar $(s,t)$ sobre los píxeles de calzada, no sobre las cajas.** Todo
   píxel por debajo del horizonte y fuera de las cajas detectadas tiene una
   profundidad geométrica conocida $Z_c(v)$ (ecuación H1). Eso son miles de
   pares $(\hat d, 1/Z_c)$ por frame → regresión robusta (Theil–Sen o RANSAC
   sobre una submuestra de ~2 000 píxeles, coste < 1 ms en NumPy). Decorrela la
   calibración de escala de los objetos que luego se miden con ella.
2. **Filtrar $(s,t)$ temporalmente** con un filtro de Kalman 1D/2D con gating
   $\chi^2$: un bache cambia el horizonte un instante; la escala real de la
   escena no cambia a 30 Hz.
3. **Evaluar también las variantes métricas del mismo tamaño.** Existen
   fine-tunings métricos de DA-V2-Small para exteriores (VKITTI) con el mismo
   coste de inferencia que la versión relativa. La premisa "los métricos exceden
   el presupuesto" es cierta para UniDepth/Metric3D-L/DepthPro, no para ViT-S.
   El trade-off es generalización: un modelo métrico queda atado a la focal de
   entrenamiento; con otra cámara sigue haciendo falta la corrección afín, pero
   parte de $t\approx 0$ y $s\approx 1$ y se degrada con elegancia.
   → **[medir]** AbsRel por bins de distancia con y sin corrección afín.
4. Publicar siempre la **incertidumbre** de la escala: si la regresión tiene
   pocos inliers (calzada ocluida, noche, lluvia) el módulo de fusión debe
   inflar $\sigma_n$ y apoyarse más en las otras cotas (§3.3).

### R2 · Sensibilidad extrema de la cota de suelo al pitch (P: A, I: A)

Derivando $d = h/\tan\varphi$ con $\varphi = \theta+\alpha$:

$$
\frac{\partial d}{\partial \theta} = -\frac{h}{\sin^2\varphi} \approx -\frac{d^2}{h}
$$

El error crece con el **cuadrado de la distancia**. Con KITTI y un error de
pitch de solo **0.5°** (frenada, carga, bache):

| `d_long` | error por 0.5° de pitch | error por 1 px en `v` |
|---|---|---|
| 5 m | 0.15 m (3 %) | 0.02 m |
| 7.5 m | 0.31 m (4 %) | 0.05 m |
| 11 m | 0.64 m (6 %) | 0.10 m |
| 15 m | 1.22 m (8 %) | 0.19 m |
| 20 m | 2.19 m (11 %) | 0.35 m |

A 30 m, 0.5° son ~5 m. Conclusión: la restricción del suelo es una **cota
excelente en el campo cercano (< 10 m) y casi inútil más allá de 20 m sin
estimación de pitch en línea**. Esto fija el diseño de la fusión: los pesos
deben derivarse de este modelo de varianza, no de constantes.

*Mitigación.*
- Modelo de varianza explícito: $\sigma_g^2(d) = \left(\frac{d^2}{h}\right)^2\sigma_\theta^2 + \left(\frac{d^2}{h f_y}\right)^2\sigma_v^2$.
- **Estimación de pitch en línea desde la propia red de profundidad**: la
  normal del plano ajustado a los puntos de calzada deproyectados es
  **invariante a la escala** (una homotecia no cambia direcciones). Se obtiene
  $\theta$ (y roll) sin conocer $s,t$; luego $h$ conocida fija la escala.
  Rompe el círculo "necesito la escala para el suelo y el suelo para la escala".
- Fallback: punto de fuga de marcas viales, o IMU si el vehículo lo expone.

### R3 · Contención en GPU entre la ruta reactiva (60 Hz) y la métrica (30 Hz) (P: A, I: M)

Dos engines TensorRT en la misma GPU laptop (pocos SMs) se ejecutan en
*time-slicing*: los kernels de la red de profundidad (grandes GEMMs de ViT,
varios ms cada uno) pueden **retrasar la cola del detector** aunque este sea
rápido en aislamiento. El P50 del detector no se moverá; su **P99 sí**.

*Mitigación.*
- Streams CUDA separados, **con prioridad**: `cudaStreamCreateWithPriority`
  (mayor prioridad al detector). Reduce la latencia de cola, no la elimina
  (no hay preempción a mitad de kernel).
- **CUDA Graphs** en ambos engines para eliminar el overhead de lanzamiento
  desde Python (decenas de µs × decenas de kernels).
- Matriz de medición obligatoria: {detector solo, depth solo, ambos} × {P50,
  P95, P99}. Si el P99 del detector bajo contención supera ~8 ms, bajar la
  resolución de la red de profundidad antes que su frecuencia (la latencia por
  frame es lo que bloquea; la frecuencia se degrada sola).
- Fallback: `depth_every_n_frames` adaptativo (2 → 3) por presión medida.

### R4 · Presupuesto de VRAM (P: M, I: A)

Estimación de orden de magnitud (FP16, **[medir]** con NVML):

| Componente | Estimación |
|---|---|
| Contexto CUDA + runtime TRT | 300–500 MB |
| Detector (YOLO-n/s 640, pesos + workspace + activaciones) | 100–250 MB |
| Profundidad ViT-S a ~518² (pesos 50 MB + activaciones 300–600 MB) | 400–700 MB |
| Buffers de E/S pinned/device, mapas de rectificación | < 50 MB |
| Segundo contexto CUDA si se usa multiproceso | +300–500 MB |
| **Total esperado** | **1.0–2.0 GB** (mono‑proceso) |

Hay margen frente a los 3.5–4.0 GB, **siempre que torch no viva en el proceso
de runtime** (H4) y que Rerun corra como visor externo. El riesgo real es
la resolución de profundidad: activaciones ViT escalan ~O(tokens²) en
atención; pasar de 518² a 700² puede duplicar activaciones.

### R5 · GIL de Python y arquitectura dual-rate (P: A, I: M)

Con `threading`, las llamadas a TensorRT (`execute_async_v3`) son cortas, pero
el trabajo NumPy de fusión, tracking y logging **sí** compite por el GIL con
el hilo de captura. Con `multiprocessing` se paga un segundo contexto CUDA
(R4) y copias/IPC de frames.

*Propuesta de diseño (a validar en F7).* Empezar por el diseño **más
determinista**: un único hilo con **dos streams asíncronos** y CUDA events.

```
loop:
  frame_k = fuente.ultimo()                      # latest-wins, sin cola
  H2D(frame_k) → stream_det; enqueue(detector); event_det_k
  if k % N == 0 and not depth_en_vuelo:
      enqueue(depth, stream_depth); event_depth_k
  esperar(event_det_k)                            # ~4-6 ms
  if event_depth_j.query(): publicar mapa_j (frame j ≤ k)
  fusión(boxes_k, mapa_j, predicción_KF) → tracks → TTC → alertas → Rerun
```

La GPU ejecuta ambas redes en paralelo, Python no bloquea en el mapa de
profundidad, y no hay GIL ni locks. Solo si la parte CPU por frame supera ~5 ms
**[medir]** se escala a hilos (captura en hilo propio) o procesos.

### R6 · Coherencia temporal entre profundidad y detección (P: A, I: M)

El mapa de profundidad disponible en el frame $k$ pertenece al frame
$j \le k$ (33–66 ms de antigüedad). Un coche a 20 m/s relativo se ha movido
0.7–1.3 m. Muestrear el mapa antiguo con la caja nueva contamina la mediana.

*Mitigación.* (a) Sellar cada mapa con su `frame_id`/timestamp y mantener un
buffer de **cajas por frame** para muestrear el mapa $j$ con las cajas del
frame $j$ (asociadas por track id); (b) la medición de profundidad entra al
KF con su **timestamp real** (medición retrasada: predecir hasta $t_j$,
corregir, re‑predecir hasta $t_k$, o inflar $R$ con la antigüedad); (c)
las alertas del lazo de 60 Hz usan siempre la predicción del KF, nunca la
última medición cruda.

### R7 · Cadena de captura en WSL2 (P: A, I: M)

WSL2 **no expone cámaras USB de forma nativa**: hace falta `usbipd-win` y un
kernel con UVC; la latencia y el jitter de captura no están garantizados.
La GPU sí funciona (CUDA/TensorRT vía `dxgkrnl`) con overhead pequeño.

*Mitigación.* Abstraer `FrameSource` (fichero de vídeo / secuencia KITTI /
V4L2 / GStreamer) y desarrollar toda la primera iteración con **secuencias
grabadas con timestamps reales**. La captura en vivo es un hito propio, no un
prerequisito. Medir el jitter de `perf_counter_ns` entre frames como métrica.

### R8 · La base de la caja no es el punto de contacto (P: A, I: M)

Objetos truncados por el borde inferior, ocluidos por otro vehículo o que no
tocan el suelo (señales, semáforos) rompen la cota geométrica: la distancia
sale **mayor** de la real (el punto visible más bajo está más lejos que el
contacto real) → error del lado peligroso.

*Mitigación.* Gates: caja tocando el borde inferior → no usar cota de suelo;
solapamiento con otra caja más cercana en la banda inferior → marcar oclusión;
clases sin contacto → solo red + prior de altura. Añadir un **tercer
estimador**: $Z_h = f_y H_{clase}/h_{px}$ (altura física conocida: coche
1.5±0.2 m, peatón 1.7±0.15 m) con $\sigma$ de la dispersión de la clase
(~10–13 %). Tres cotas independientes → fusión por varianza inversa robusta.

### R9 · Telemetría que bloquea el lazo (P: M, I: M)

Loguear un mapa de profundidad float32 a 30 Hz en Rerun son ~20–30 MB/s solo
en tensores; serializar en el hilo principal introduce picos. Rerun es
asíncrono en el envío, pero la **codificación** ocurre en la llamada.

*Mitigación.* Visor en proceso separado (`rr.connect_grpc`/`spawn`); imágenes y
mapas cada N frames y submuestreados (¼); escalares y tracks cada frame;
`rr.set_time` con el timestamp de captura, no el de log. Medir el coste del
logging como una etapa más del perfil.

### R10 · Falta de verdad terreno para validar métrica y velocidad (P: M, I: A)

Sin ground truth, "funciona" es una opinión. KITTI ofrece LiDAR (profundidad
por objeto), *tracking* (identidades y trayectorias) y OXTS (ego‑motion
GPS/IMU), con la misma cámara ya cargada en `configs/camera_kitti.yaml`.

*Mitigación.* Definir desde F1 el conjunto de evaluación: N secuencias KITTI
`tracking`, métricas AbsRel/RMSE por bins de distancia (0–10, 10–30, 30–60 m),
RMSE de velocidad relativa vs. GT derivada, y **generador sintético
determinista** (trayectorias cinemáticas con ruido controlado) para EKF y TTC
sin GPU, ejecutable en CI.

### R11 · Dependencias pesadas y reproducibilidad (P: M, I: B)

`tensorrt-cu12` (>1 GB), `torch`+`torchvision` (>2 GB) y `cuda-python` en el
mismo entorno hacen que `uv sync` sea lento y que el `uv.lock` sea frágil
frente a versiones de driver.

*Mitigación.* Grupos opcionales: `runtime` (numpy, opencv, tensorrt-cu12,
cuda-python, rerun-sdk, pyyaml, scipy), `export` (torch, torchvision, onnx,
onnxsim), `dev` (ruff, mypy, pytest, nvidia-ml-py). CI instala solo `dev` +
núcleo puro-CPU. Fijar la versión **mayor** de TensorRT (10.x) y documentar la
del driver con la que se generó cada `.engine` (los engines no son portables
entre versiones ni entre GPUs).

---

## 2. Análisis por etapa

### 3.1 · Calibración y geometría proyectiva

**Precomputar mapas sin penalizar el bucle.** Ya se hace (LUT en `__init__`).
La pregunta correcta es *dónde aplicar la rectificación*, porque el `remap`
completo a resolución de sensor es la operación más cara del lazo reactivo
fuera de la GPU. Opciones, de menor a mayor esfuerzo:

| Opción | Coste/frame | Pros | Contras |
|---|---|---|---|
| **A. No rectificar la imagen; rectificar solo puntos** (`cv2.undistortPoints` sobre esquinas de caja y píxeles muestreados) | ~µs | Elimina el `remap`; los detectores toleran distorsión moderada | La red de profundidad ve distorsión → pequeño sesgo radial; no vale para gran angular |
| **B. Fusionar undistort + resize + letterbox en un solo `remap`** al tamaño de entrada del modelo | ~1–2 ms CPU a 640×384 | Una sola interpolación, salida pequeña | Dos mapas si detector y depth usan tamaños distintos |
| **C. `GridSample` con rejilla constante dentro del grafo ONNX** | ~0.1 ms GPU | Todo en GPU, cero trabajo CPU, entra en el CUDA Graph | Resolución fija por engine; doble interpolación con el resize interno |
| **D. `cv2.cuda.remap`** | ~0.3 ms GPU | Rápido | `opencv-python-headless` de pip **no** incluye módulo CUDA → compilar OpenCV |

Recomendación exploratoria: **A** para el MVP con KITTI (coeficientes cero →
gratis) y **C** como diseño objetivo para cámaras reales: el engine acepta el
frame crudo `uint8 NHWC`, y la geometría (undistort, resize, normalización,
NCHW) vive dentro del grafo. Descartar D salvo que se compile OpenCV.

**Singularidades cerca del horizonte.** El guard angular actual (`1e-4 rad`)
evita la división por cero pero no el problema real: la varianza diverge como
$d^2$ (R2). Tres medidas:
1. Trabajar internamente en **profundidad inversa** $\rho = 1/Z$: cerca del
   horizonte $\rho \to 0$ de forma suave y su ruido es aproximadamente
   gaussiano (es lineal en $\tan\varphi$); la red también entrega
   disparidad. Convertir a metros solo en la salida.
2. **Distancia máxima útil** $d_{max}$ derivada del modelo de varianza, no
   fija: `d_max = argmax_d { σ_g(d) < σ_tol }` (con KITTI y σ_θ = 0.5°, ~15 m
   para σ_tol = 10 %). Más allá, la cota de suelo solo aporta a través de su
   varianza (peso ≈ 0), sin excepciones.
3. Vectorizar `compute_ground_distances` (arrays de `v`) y devolver
   $(Z_c, d_{long}, \sigma)$; las excepciones por píxel no escalan a miles
   de píxeles de calzada por frame.

Pendiente adicional: **roll** de la cámara (hoy no modelado). Un roll de 1°
inclina el horizonte ~11 px en los bordes de KITTI. La estimación de plano de
R2 lo entrega gratis.

### 3.2 · Percepción 2D y profundidad relativa

**Detector.** Candidatos YOLOv8n/YOLO11n (o `-s` si el P99 lo permite),
exportados a ONNX con **NMS dentro del engine** (plugin EfficientNMS) para no
hacer NMS en Python. Entrada **rectangular** acorde al aspecto del sensor
(KITTI 3.3:1 → p. ej. 1024×320 en vez de 640×640 con 70 % de padding). Dos
trade-offs a medir: INT8 con calibración (–30–40 % latencia, riesgo en
peatones pequeños) y resolución vs. recall de objetos lejanos.

**Profundidad.** Depth Anything V2‑Small (ViT‑S, ~25 M parámetros) como
candidato principal; variante métrica outdoor del mismo tamaño como segunda
opción (R1). Entrada múltiple de 14 (patch ViT); preferir aspecto ancho
(p. ej. 924×280 ≈ mismos tokens que 518×518) para no perder resolución
lateral. **[medir]** P95 y AbsRel para 3 tamaños.

**Minimizar el overhead del framework de inferencia.** Principios:
1. **Sin PyTorch en el runtime.** TensorRT 10 (`execute_async_v3`,
   `set_tensor_address`) + `cuda-python` para streams, events, memoria pinned
   y copias. Un solo contexto CUDA.
2. **Buffers preasignados** (device y pinned host) por engine, tamaño fijo,
   reutilizados; cero `malloc` por frame.
3. **Preprocesado dentro del grafo ONNX**: `uint8 NHWC → float NCHW`,
   `/255`, mean/std. Se sube 1 byte/píxel en vez de 4 (1242×375×3 ≈ 1.4 MB,
   ~0.1 ms PCIe) y no hay `astype` ni `transpose` en NumPy.
4. **CUDA Graph** por engine (captura del `enqueue`): el coste de lanzar 200
   kernels desde Python baja a una llamada.
5. **Salidas compactas**: detector → tensor `[N_max, 6]` ya con NMS;
   profundidad → `float16` a resolución de red (el upsample al tamaño del
   sensor es innecesario: se muestrea en coordenadas de red).
6. **Sincronización por events, no `synchronize()` global**, para que la red
   de profundidad siga corriendo mientras se procesa el detector.

Presupuesto orientativo del lazo reactivo a 60 Hz (16.7 ms), **[medir]**:
captura+H2D ≤ 1 ms · detector ≤ 5 ms (P95, bajo contención ≤ 8 ms) ·
tracking+fusión+TTC ≤ 3 ms · logging ≤ 1 ms → ~10 ms con margen.

### 3.3 · Fusión métrica y restricción del suelo

**Tres estimadores independientes por objeto**, cada uno con su varianza:

| Estimador | Fórmula | Modelo de σ | Dominio de validez |
|---|---|---|---|
| Suelo $Z_g$ | $h\cos\alpha/\sin(\theta+\alpha)$ en la base de la caja | $\sigma_g \approx \frac{d^2}{h}\sqrt{\sigma_\theta^2 + \sigma_v^2/f_y^2}$ | Caja no truncada ni ocluida, clase con contacto, $d < d_{max}$ |
| Red $Z_n$ | $1/((\hat d_{med} - t)/s)$ | $\sigma_n \approx k\,Z$ (k ≈ 5–10 % **[medir]**) + término de la incertidumbre de $(s,t)$ | Siempre; degradado si pocos inliers de calzada |
| Altura $Z_h$ | $f_y H_{clase}/h_{px}$ | $\sigma_h/Z = \sigma_H/H$ (~10–13 %) | Caja completa verticalmente |

Fusión por varianza inversa:
$\hat Z = \frac{\sum_i Z_i/\sigma_i^2}{\sum_i 1/\sigma_i^2}$,
$\hat\sigma^2 = 1/\sum_i 1/\sigma_i^2$, precedida de un **test de
consistencia** (si dos estimadores difieren > 3σ combinadas, se descarta el
menos fiable según los gates, y se marca el objeto como `inconsistente` para
que el KF infle R). Es más resiliente que un peso fijo porque:
- ante un **bache** (pitch transitorio) $\sigma_g$ se infla vía la
  incertidumbre de pitch estimada en línea, y la fusión migra a $Z_n, Z_h$
  sin lógica ad hoc;
- ante **fallo del punto de contacto** el gate anula $Z_g$ y quedan dos
  cotas;
- ante **calzada no visible** (atasco) $\sigma_n$ crece y $Z_g, Z_h$
  dominan en el campo cercano, donde son mejores.

**Muestreo robusto dentro de la caja.** Reglas propuestas:
1. ROI interior: recortar 25 % lateral y quedarse con la banda vertical
   central (35–75 % de la altura) → evita suelo por abajo y cielo/fondo por
   arriba.
2. Estadístico: **mediana y MAD** de la disparidad; inliers en
   $|\hat d - med| < 2.5 \cdot 1.4826\,MAD$; reestimar mediana sobre
   inliers; $\sigma_{med} \approx 1.2533\cdot 1.4826\,MAD/\sqrt{n}$.
3. Caso **objeto delgado** (peatón, poste): la ROI puede ser mayoritariamente
   fondo, y la mediana cae en el fondo (disparidad menor). Usar el **modo de
   mayor disparidad** (superficie más cercana) cuando el histograma es
   bimodal; para clases anchas, la mediana.
4. Trabajar siempre en el **espacio de disparidad de la red** (donde el ruido
   es más homogéneo), y convertir a $Z$ tras la robustificación.

### 3.4 · Tracking 3D y ego-motion

**Asociación.** ByteTrack en 2D a 60 Hz (IoU + score bajo/alto, sin ReID) es
suficiente y barato (< 0.5 ms para decenas de cajas). El estado 3D cuelga del
id 2D. Trade-off frente a asociación 3D (distancia de Mahalanobis en $(X,Z)$):
la 3D es más robusta a cruces pero depende de una métrica aún ruidosa; se puede
añadir como segunda etapa para las cajas que ByteTrack deja sin emparejar.

**¿EKF o KF lineal?** Depende de **dónde** se define la medición:

| Diseño | Estado | Medición | Filtro |
|---|---|---|---|
| **I. Medición convertida** | $[X, Z, \dot X, \dot Z]$ en el marco cámara-suelo | $(X, Z)$ ya fusionadas (§3.3) | **KF lineal**, con $R_k$ calculado por medición vía jacobiano de la conversión (`converted-measurement KF`) |
| II. Medición en imagen | igual | $(u_{centro}, v_{base}, \hat d)$ | **EKF** ($h(x)$ no lineal: $u = f X/Z + c_x$, $\rho = 1/Z$) |
| III. Profundidad inversa | $[u, \rho, \dot u, \dot\rho]$ | $(u, \rho)$ | KF lineal en imagen; no lineal para velocidad métrica |

Recomendación: **I para el MVP**. El modelo de proceso (velocidad constante)
es lineal en cartesianas; la no linealidad vive solo en la conversión de
medida, y se captura correctamente propagando la covarianza por el jacobiano:

$$
R_k = J\,\mathrm{diag}(\sigma_u^2, \sigma_Z^2)\,J^\top,\quad
\sigma_X \approx \frac{Z}{f}\sigma_u + \frac{|u-c_x|}{f}\sigma_Z,\quad
\sigma_Z \text{ de la fusión (crece }\propto Z\ldots Z^2)
$$

Esto responde a "cómo modelar el crecimiento del ruido con la distancia": no
con una función arbitraria sino **heredando la varianza fusionada** (§3.3),
que ya es $\propto Z$ (red) y $\propto Z^2$ (suelo). El EKF (II) queda
como alternativa si se decide medir directamente en píxeles/disparidad para
evitar la fusión previa; añade linealización y riesgo de divergencia con
$\Delta t$ variable.

**$\Delta t$ variable.** Usar timestamps de captura (`perf_counter_ns` en la
adquisición), nunca el reloj del procesamiento. Ruido de proceso *discrete
white noise acceleration* por eje:
$Q(\Delta t) = \sigma_a^2 \begin{bmatrix}\Delta t^4/4 & \Delta t^3/2\\ \Delta t^3/2 & \Delta t^2\end{bmatrix}$
con $\sigma_a$ por clase (coche 2–3 m/s², peatón 1 m/s²). Mediciones
fuera de orden (mapa de profundidad antiguo, R6): retro‑predicción o inflado
de $R$ proporcional a la antigüedad.

**Ego-motion.** Sin compensación, un objeto estático parece moverse a
$-v_{ego}$; eso es **correcto para TTC** (lo que importa es la velocidad
relativa) pero **incorrecto para clasificar estático/móvil y para el modelo
CV en giros** (en un giro, un objeto estático describe un arco en el marco
cámara; el KF lo interpreta como aceleración). Diseño:
- Interfaz `EgoMotionProvider` → $(R_{k-1\to k}, t_{k-1\to k})$ o
  $(v, \omega)$; implementaciones: KITTI OXTS (GT para validar), CAN/odometría
  de ruedas, VO monocular con escala del suelo (fase avanzada), y `Zero`.
- Compensación en la **predicción**: transformar el estado previo al marco
  actual antes de aplicar $F(\Delta t)$; la velocidad relativa sigue en el
  estado, y la absoluta se deriva como $v_{rel} + v_{ego}$ para la lógica
  de clasificación.
- Tratar la incertidumbre del ego como ruido de proceso adicional (no como
  verdad).

### 3.5 · Cinemática y seguridad (TTC)

**Por qué falla $-Z/\dot Z$.** Con objeto estático y ego lento, $\dot Z\to
0^-$ y TTC → ∞ aunque esté a 1 m; con un objeto lateral que **no** va a
cruzar nuestra trayectoria, TTC es finito y genera falsa alarma. Hay que
separar *cuándo* (tiempo) de *dónde* (geometría de paso).

**Magnitudes.** Con posición relativa $p = (X, Z)$ y velocidad relativa
$v = (\dot X, \dot Z)$:

$$
t_{CPA} = -\frac{p\cdot v}{\|v\|^2}, \qquad
d_{CPA} = \|p + v\,t_{CPA}\|, \qquad
TTC_{corr} = t_{CPA}\ \text{si } d_{CPA} < w_{ego}/2 + w_{obj}/2 + m
$$

- **Compuerta de proximidad** (independiente de velocidad): $Z < Z_{near}$
  y $|X| < w_{corredor}/2$ → alerta aunque $v \approx 0$. Cubre el objeto
  estático cercano.
- **Compuerta de trayectoria**: solo se evalúa TTC si $d_{CPA}$ cae en el
  corredor; si no, el objeto es "de paso" y como mucho `CAUTION`.
- **Robustez al ruido**: usar la **cota conservadora** $TTC_{low}$ calculada
  con $\dot Z + k\sigma_{\dot Z}$ (k = 1–2) y $Z - k\sigma_Z$; no
  escalar la alerta si `traza(P)` del track supera un umbral o la edad del
  track es < N frames (track sin confirmar).

**Máquina de estados determinista con histéresis** por track:

```
NONE → CAUTION   si TTC_low < 4.0 s  ó  Z < 12 m en corredor,  durante ≥ 3 frames
CAUTION → WARNING si TTC_low < 2.5 s  ó  Z < 6 m en corredor,   durante ≥ 2 frames
WARNING → CRITICAL si TTC_low < 1.2 s ó  Z < 3 m en corredor,   inmediato
bajada de nivel:  umbral de salida = umbral de entrada × 1.3 (tiempo) / × 1.2 (distancia),
                  sostenido ≥ 10 frames (~170 ms a 60 Hz)
track perdido:    mantener nivel con decaimiento 1 nivel/500 ms, nunca subir
```

Los umbrales son parámetros de configuración (por clase y por velocidad del
ego); la lógica es una función pura `step(estado, medida, cfg) → estado` sin
aleatoriedad ni dependencia del reloj de pared (solo de los timestamps de
entrada), de modo que la misma secuencia produce la misma salida y se puede
testear exhaustivamente con trayectorias sintéticas (colisión frontal,
adelantamiento, cruce lateral, objeto estático, track intermitente).

---

## 3. Crítica del stack propuesto

| Elección | Veredicto | Matices |
|---|---|---|
| Python 3.11 + `uv` | Adecuado | Runtime en Python es viable si la GPU hace el trabajo pesado y el código por frame es NumPy vectorizado. Presupuesto CPU/frame ≤ 5 ms **[medir]**. |
| TensorRT vía `tensorrt-cu12` (pip) | Adecuado, con cautela | Los `.engine` son específicos de GPU + versión TRT: generarlos en la máquina objetivo (`scripts/export_trt.sh`), nunca versionarlos. Plugins (EfficientNMS) vienen en la wheel. |
| `cuda-python` | Adecuado | API de bajo nivel verbosa; encapsular en `runtime/cuda.py` (streams, events, buffers pinned). Alternativa `cupy` si se necesitan kernels ad hoc (p. ej. muestreo por caja en GPU). |
| PyTorch solo para export | Correcto | Debe salir del grupo de dependencias del runtime (H4). |
| `opencv-python-headless` | Adecuado para CPU | Sin módulo CUDA; no planificar `cv2.cuda.*`. |
| NumPy < 2 | Aceptable | Fijado por compatibilidad de bindings; revisar cuando `tensorrt`/`cv2` lo permitan. |
| SciPy | Adecuado | `linear_sum_assignment` (asociación), `stats` (Theil–Sen), poco más. |
| Rerun SDK | Adecuado | Visor externo; controlar el volumen (R9). Versión ≥ 0.18 usa gRPC. |
| ruff + mypy strict + pytest | Adecuado | Mantener `-m "not gpu and not slow"` en CI; tests GPU se ejecutan en local con `scripts/`. |
| Docker solo para CI | Correcto | La imagen CI no necesita CUDA; validar solo lógica pura. Una imagen `nvidia/cuda` opcional para reproducir engines, no para desarrollo. |

**Ausencias sugeridas:** `nvidia-ml-py` (NVML, medir VRAM real), `onnx` +
`onnxsim`/`onnxslim` (export), `polygraphy` (depurar precisión FP16 del engine
capa a capa), y un `hypothesis` opcional para tests de propiedades de la
geometría y la máquina de estados.

---

## 4. Decisiones que se posponen hasta tener métricas

| Decisión | Métrica que la desbloquea | Fase |
|---|---|---|
| Resolución/entrada del detector y variante n/s | P95 aislado y bajo contención; recall KITTI coches/peatones a > 30 m | F2 |
| Resolución y variante (relativa vs. métrica) de profundidad | P95, VRAM NVML, AbsRel por bins tras corrección afín | F3 |
| Frecuencia de profundidad (cada 2 ó 3 frames) | P99 del detector bajo contención | F3/F7 |
| Rectificación A vs. C | Coste CPU del `remap` a la resolución objetivo | F1 |
| Hilos vs. mono‑hilo con dos streams | P99 extremo a extremo del lazo reactivo; ms de CPU por frame | F7 |
| Necesidad de EKF (II) | Error de velocidad del KF lineal (I) vs. GT KITTI | F5 |
| INT8 en detector | Δ latencia vs. Δ recall en peatones | F8 |

---

## 5. Resumen ejecutivo

1. La **cota de suelo es excelente por debajo de ~10 m y peligrosa por encima
   de 20 m** sin pitch en línea: el diseño de fusión debe nacer del modelo de
   varianza $\sigma_g \propto d^2$, y el pitch debe estimarse desde el plano de
   calzada de la red (invariante a escala).
2. La **escala afín se calibra en la calzada**, no en las cajas; la caja
   aporta tres cotas (suelo, red, altura) fusionadas por varianza inversa con
   test de consistencia.
3. **Un hilo, dos streams, CUDA Graphs, sin torch**: el diseño dual‑rate más
   simple y determinista; escalar a hilos/procesos solo si la medición lo
   exige.
4. **KF lineal con medición convertida** y $R(Z)$ heredada de la fusión;
   EKF solo si se decide medir en píxeles.
5. **TTC = tiempo + geometría**: CPA + compuertas + histéresis en una función
   pura testeable.
6. Antes de nada: corregir H1, sacar torch del runtime, sustituir el proxy del
   benchmark por engines reales y montar el arnés de percentiles + NVML.
