# 00 · Prior critical analysis — Monocular 3D perception on the edge

> **Languages:** English (this page) · [Español](es/00_analisis_critico.md)
>
> Deliverable 1 of the working process (§5.1 of the brief): risks, trade-offs and
> mitigations **before** locking the detailed architecture. Everything stated here
> as a number is an order-of-magnitude estimate or an analytical derivation;
> definitive values come from the measurement harness in phase F1
> (see `01_mvp_phase_plan.md`). Where **[measure]** is marked, the decision is
> deliberately deferred until real P50/P95/P99 percentiles are available.

---

## 0. Repository status (starting point)

| Module | Status | Notes |
|---|---|---|
| `camera/calibration.py` | Implemented | Immutable dataclasses, KITTI and YAML parsers. |
| `camera/geometry.py` | Implemented | Projection/deprojection, horizon, ground distances. **See finding H1.** |
| `camera/undistort.py` | Implemented | Precomputed `CV_32FC1` LUT, `cv2.remap` on CPU. |
| `tests/test_geometry.py` | 22 tests green | Reversibility, monotonicity, horizon gating. |
| `scripts/benchmark_depth.py` | Implemented | EfficientNet-B3 proxy with CUDA events. **See finding H2.** (Replaced in F3 by `scripts/bench_depth.py` on the real engine.) |
| `detection/`, `depth/`, `tracking/`, `safety/`, `runtime/`, `utils/` | Empty files | Skeleton only. |
| `configs/models.yaml`, `README.md` | Empty | |

Verified locally (CPU, no GPU): `ruff check`, `ruff format --check`,
`mypy --strict` and `pytest -m "not slow and not gpu"` pass.

### Findings on the existing code

**H1 — `compute_ground_distances` returns the ray length, not the optical depth $Z_c$.**
The docstring says that `z_cam` is the "depth along the optical axis", but
`h / sin(θ+α)` is the **hypotenuse** (Euclidean camera→contact-point distance).
The depth along the optical axis is:

$$
Z_c = \frac{h\,\cos\alpha}{\sin(\theta+\alpha)}, \qquad
d_{long} = \frac{h}{\tan(\theta+\alpha)}, \qquad
\|r\| = \frac{h}{\sin(\theta+\alpha)}
$$

Impact with the KITTI calibration (`h=1.65 m`, `θ=2.5°`):

| row `v` | α | `d_long` | `‖r‖` (code) | `Z_c` (correct) | error |
|---|---|---|---|---|---|
| 200 | 2.2° | 20.27 m | 20.33 m | 20.32 m | 0.07 % |
| 250 | 6.1° | 10.91 m | 11.03 m | 10.97 m | 0.57 % |
| 300 | 10.0° | 7.45 m | 7.63 m | 7.51 m | 1.54 % |
| 374 | 15.6° | 5.06 m | 5.32 m | 5.12 m | 3.81 % |

It is small at mid-range but **systematic and largest precisely in the critical
safety range (< 8 m)**. It matters because depth networks predict $Z_c$ per
pixel (or its inverse), and metric fusion (§3.3) compares both quantities:
mixing hypotenuse with $Z_c$ introduces a scale bias that the affine fit later
"absorbs" poorly (it is not constant in $v$). Current tests do not catch it
because the invariant `z_cam > d_long` also holds for $Z_c$
($Z_c/d_{long} = \cos\alpha/\cos(\theta+\alpha) > 1$).
→ Fix in F0.1 and add a closed-form value test.

**H2 — The depth benchmark uses a convolutional proxy for a ViT model.**
EfficientNet-B3 does not reproduce the compute profile of a ViT-S (quadratic
attention over tokens, large GEMMs, no separable convolutions). The percentiles
obtained are not transferable: the real benchmark must run the **TensorRT
engine of the candidate model**, not a PyTorch proxy. Furthermore,
`torch.cuda.max_memory_allocated` only sees the PyTorch allocator; the real
VRAM peak (CUDA context, TRT workspace, cuDNN) must be read with NVML.
→ Resolved in F3: `scripts/bench_depth.py` measures the TensorRT engine of Depth
Anything V2-Small (CUDA events + NVML) and the detector+depth matrix.

**H3 — `ImageRectifier` is designed to run at sensor resolution on CPU.**
Correct for KITTI (1242×375, already rectified → zero coefficients) but at 1080p
single-threaded `cv2.remap` typically costs 8–12 ms **[measure]**, incompatible
with the reactive loop at 60 Hz (16.7 ms). See §3.1.

**H4 — The runtime declares `torch` and `torchvision` as required dependencies.**
Importing torch in the inference process costs ~1–2 s of startup and, if CUDA
is touched, reserves its own context and cache (hundreds of MB). It must move
to an optional `export`/`bench` group.

---

## 1. Global risks and mitigations

Classification: **P** = probability, **I** = impact (H/M/L).

### R1 · Metric divergence (P: H, I: H)

*The problem.* Lightweight depth models (Depth Anything V2-Small,
DPT-Hybrid, MiDaS-small) return **affine-invariant relative disparity**:
$\hat d = s\cdot\frac{1}{Z} + t$ with $s, t$ unknown and **variable frame
to frame**. A single ground-contact point yields one equation for two
unknowns. If $t = 0$ is assumed (scale only) the error blows up at the
extremes of the range; if $s,t$ are estimated from few boxes, the fit is
fragile to a wrong detection.

*Proposed mitigation.*
1. **Fit $(s,t)$ on roadway pixels, not on boxes.** Every pixel below the
   horizon and outside detected boxes has a known geometric depth $Z_c(v)$
   (equation H1). That is thousands of $(\hat d, 1/Z_c)$ pairs per frame →
   robust regression (Theil–Sen or RANSAC on a subsample of ~2 000 pixels,
   cost < 1 ms in NumPy). Decouples the scale calibration from the objects
   later measured with it.
2. **Filter $(s,t)$ temporally** with a 1D/2D Kalman filter with $\chi^2$
   gating: a bump changes the horizon for an instant; the real scene scale
   does not change at 30 Hz.
3. **Also evaluate metric variants of the same size.** Metric fine-tunings of
   DA-V2-Small for outdoors (VKITTI) exist with the same inference cost as the
   relative version. The premise "metric models exceed the budget" is true for
   UniDepth/Metric3D-L/DepthPro, not for ViT-S. The trade-off is generalization:
   a metric model is tied to the training focal length; with another camera
   the affine correction is still needed, but it starts from $t\approx 0$ and
   $s\approx 1$ and degrades gracefully.
   → **[measure]** AbsRel by distance bins with and without affine correction.
4. Always publish the scale **uncertainty**: if the regression has few inliers
   (occluded roadway, night, rain) the fusion module must inflate $\sigma_n$
   and lean more on the other bounds (§3.3).

### R2 · Extreme sensitivity of the ground bound to pitch (P: H, I: H)

Differentiating $d = h/\tan\varphi$ with $\varphi = \theta+\alpha$:

$$
\frac{\partial d}{\partial \theta} = -\frac{h}{\sin^2\varphi} \approx -\frac{d^2}{h}
$$

The error grows with the **square of the distance**. With KITTI and a pitch
error of only **0.5°** (braking, load, bump):

| `d_long` | error per 0.5° of pitch | error per 1 px in `v` |
|---|---|---|
| 5 m | 0.15 m (3 %) | 0.02 m |
| 7.5 m | 0.31 m (4 %) | 0.05 m |
| 11 m | 0.64 m (6 %) | 0.10 m |
| 15 m | 1.22 m (8 %) | 0.19 m |
| 20 m | 2.19 m (11 %) | 0.35 m |

At 30 m, 0.5° is ~5 m. Conclusion: the ground constraint is an **excellent
bound in the near field (< 10 m) and almost useless beyond 20 m without
online pitch estimation**. This fixes the fusion design: weights must be
derived from this variance model, not from constants.

*Mitigation.*
- Explicit variance model: $\sigma_g^2(d) = \left(\frac{d^2}{h}\right)^2\sigma_\theta^2 + \left(\frac{d^2}{h f_y}\right)^2\sigma_v^2$.
- **Online pitch estimation from the depth network itself**: the normal of the
  plane fit to the deprojected roadway points is **scale-invariant** (a
  homothety does not change directions). $\theta$ (and roll) are obtained
  without knowing $s,t$; then known $h$ fixes the scale. Breaks the circle
  "I need the scale for the ground and the ground for the scale".
- Fallback: vanishing point of lane markings, or IMU if the vehicle exposes it.

### R3 · GPU contention between the reactive path (60 Hz) and the metric path (30 Hz) (P: H, I: M)

Two TensorRT engines on the same laptop GPU (few SMs) run under
*time-slicing*: the depth-network kernels (large ViT GEMMs, several ms each)
can **delay the detector queue** even though the detector is fast in
isolation. The detector's P50 will not move; its **P99 will**.

*Mitigation.*
- Separate CUDA streams, **with priority**: `cudaStreamCreateWithPriority`
  (higher priority for the detector). Reduces queue latency, does not eliminate
  it (no mid-kernel preemption).
- **CUDA Graphs** on both engines to remove launch overhead from Python
  (tens of µs × tens of kernels).
- Mandatory measurement matrix: {detector only, depth only, both} × {P50,
  P95, P99}. If the detector's P99 under contention exceeds ~8 ms, lower the
  depth network resolution before its frequency (per-frame latency is what
  blocks; frequency degrades on its own).
- Fallback: adaptive `depth_every_n_frames` (2 → 3) under measured pressure.

### R4 · VRAM budget (P: M, I: H)

Order-of-magnitude estimate (FP16, **[measure]** with NVML):

| Component | Estimate |
|---|---|
| CUDA context + TRT runtime | 300–500 MB |
| Detector (YOLO-n/s 640, weights + workspace + activations) | 100–250 MB |
| Depth ViT-S at ~518² (weights 50 MB + activations 300–600 MB) | 400–700 MB |
| Pinned/device I/O buffers, rectification maps | < 50 MB |
| Second CUDA context if multiprocessing is used | +300–500 MB |
| **Expected total** | **1.0–2.0 GB** (single process) |

There is headroom vs. 3.5–4.0 GB, **as long as torch does not live in the
runtime process** (H4) and Rerun runs as an external viewer. The real risk is
depth resolution: ViT activations scale ~O(tokens²) in attention; going from
518² to 700² can double activations.

### R5 · Python GIL and dual-rate architecture (P: H, I: M)

With `threading`, TensorRT calls (`execute_async_v3`) are short, but the NumPy
work of fusion, tracking and logging **does** contend for the GIL with the
capture thread. With `multiprocessing` one pays a second CUDA context
(R4) and frame copies/IPC.

*Design proposal (to validate in F7).* Start with the **most deterministic**
design: a single thread with **two asynchronous streams** and CUDA events.

```
loop:
  frame_k = source.latest()                      # latest-wins, no queue
  H2D(frame_k) → stream_det; enqueue(detector); event_det_k
  if k % N == 0 and not depth_in_flight:
      enqueue(depth, stream_depth); event_depth_k
  wait(event_det_k)                               # ~4-6 ms
  if event_depth_j.query(): publish map_j (frame j ≤ k)
  fusion(boxes_k, map_j, KF_prediction) → tracks → TTC → alerts → Rerun
```

The GPU runs both networks in parallel, Python does not block on the depth
map, and there is no GIL or locks. Only if the CPU work per frame exceeds ~5 ms
**[measure]** do we scale to threads (capture on its own thread) or processes.

### R6 · Temporal coherence between depth and detection (P: H, I: M)

The depth map available at frame $k$ belongs to frame
$j \le k$ (33–66 ms old). A car at 20 m/s relative has moved
0.7–1.3 m. Sampling the old map with the new box contaminates the median.

*Mitigation.* (a) Stamp each map with its `frame_id`/timestamp and keep a
buffer of **boxes per frame** to sample map $j$ with the boxes of
frame $j$ (associated by track id); (b) the depth measurement enters the
KF with its **real timestamp** (delayed measurement: predict to $t_j$,
correct, re-predict to $t_k$, or inflate $R$ with age); (c)
the 60 Hz loop alerts always use the KF prediction, never the
last raw measurement.

### R7 · Capture chain on WSL2 (P: H, I: M)

WSL2 **does not expose USB cameras natively**: `usbipd-win` and a kernel with
UVC are needed; capture latency and jitter are not guaranteed.
The GPU does work (CUDA/TensorRT via `dxgkrnl`) with small overhead.

*Mitigation.* Abstract `FrameSource` (video file / KITTI sequence /
V4L2 / GStreamer) and develop the entire first iteration with **recorded
sequences with real timestamps**. Live capture is its own milestone, not a
prerequisite. Measure `perf_counter_ns` jitter between frames as a metric.

### R8 · The box bottom is not the contact point (P: H, I: M)

Objects truncated by the bottom edge, occluded by another vehicle, or that do
not touch the ground (signs, traffic lights) break the geometric bound: the
distance comes out **larger** than reality (the lowest visible point is farther
than the real contact) → error on the dangerous side.

*Mitigation.* Gates: box touching the bottom edge → do not use the ground bound;
overlap with another closer box in the lower band → mark occlusion;
classes without contact → network + height prior only. Add a **third
estimator**: $Z_h = f_y H_{class}/h_{px}$ (known physical height: car
1.5±0.2 m, pedestrian 1.7±0.15 m) with $\sigma$ from the class dispersion
(~10–13 %). Three independent bounds → robust inverse-variance fusion.

### R9 · Telemetry that blocks the loop (P: M, I: M)

Logging a float32 depth map at 30 Hz in Rerun is ~20–30 MB/s in tensors alone;
serializing on the main thread introduces spikes. Rerun is asynchronous on
send, but **encoding** happens in the call.

*Mitigation.* Viewer in a separate process (`rr.connect_grpc`/`spawn`); images and
maps every N frames and downsampled (¼); scalars and tracks every frame;
`rr.set_time` with the capture timestamp, not the log timestamp. Measure logging
cost as one more stage of the profile.

### R10 · Lack of ground truth to validate metric and velocity (P: M, I: H)

Without ground truth, "it works" is an opinion. KITTI offers LiDAR (per-object
depth), *tracking* (identities and trajectories) and OXTS (ego-motion
GPS/IMU), with the same camera already loaded in `configs/camera_kitti.yaml`.

*Mitigation.* Define from F1 the evaluation set: N KITTI `tracking`
sequences, AbsRel/RMSE metrics by distance bins (0–10, 10–30, 30–60 m),
relative velocity RMSE vs. derived GT, and a **deterministic synthetic
generator** (kinematic trajectories with controlled noise) for EKF and TTC
without GPU, runnable in CI.

### R11 · Heavy dependencies and reproducibility (P: M, I: L)

`tensorrt-cu12` (>1 GB), `torch`+`torchvision` (>2 GB) and `cuda-python` in the
same environment make `uv sync` slow and the `uv.lock` fragile against driver
versions.

*Mitigation.* Optional groups: `runtime` (numpy, opencv, tensorrt-cu12,
cuda-python, rerun-sdk, pyyaml, scipy), `export` (torch, torchvision, onnx,
onnxsim), `dev` (ruff, mypy, pytest, nvidia-ml-py). CI installs only `dev` +
pure-CPU core. Pin the **major** TensorRT version (10.x) and document the
driver version with which each `.engine` was generated (engines are not
portable across versions or across GPUs).

---

## 2. Stage-by-stage analysis

### 3.1 · Calibration and projective geometry

**Precompute maps without penalizing the loop.** Already done (LUT in `__init__`).
The right question is *where to apply rectification*, because a full `remap`
at sensor resolution is the most expensive operation of the reactive loop
outside the GPU. Options, from least to most effort:

| Option | Cost/frame | Pros | Cons |
|---|---|---|---|
| **A. Do not rectify the image; rectify points only** (`cv2.undistortPoints` on box corners and sampled pixels) | ~µs | Eliminates `remap`; detectors tolerate moderate distortion | The depth network sees distortion → small radial bias; not valid for wide angle |
| **B. Fuse undistort + resize + letterbox into a single `remap`** at the model input size | ~1–2 ms CPU at 640×384 | Single interpolation, small output | Two maps if detector and depth use different sizes |
| **C. `GridSample` with constant grid inside the ONNX graph** | ~0.1 ms GPU | Everything on GPU, zero CPU work, enters the CUDA Graph | Fixed resolution per engine; double interpolation with the internal resize |
| **D. `cv2.cuda.remap`** | ~0.3 ms GPU | Fast | pip `opencv-python-headless` does **not** include the CUDA module → compile OpenCV |

Exploratory recommendation: **A** for the MVP with KITTI (zero coefficients →
free) and **C** as the target design for real cameras: the engine accepts the
raw `uint8 NHWC` frame, and geometry (undistort, resize, normalization,
NCHW) lives inside the graph. Discard D unless OpenCV is compiled.

**Singularities near the horizon.** The current angular guard (`1e-4 rad`)
avoids division by zero but not the real problem: variance diverges as
$d^2$ (R2). Three measures:
1. Work internally in **inverse depth** $\rho = 1/Z$: near the
   horizon $\rho \to 0$ smoothly and its noise is approximately
   Gaussian (it is linear in $\tan\varphi$); the network also delivers
   disparity. Convert to meters only at the output.
2. **Useful maximum distance** $d_{max}$ derived from the variance model, not
   fixed: `d_max = argmax_d { σ_g(d) < σ_tol }` (with KITTI and σ_θ = 0.5°, ~15 m
   for σ_tol = 10 %). Beyond that, the ground bound only contributes through its
   variance (weight ≈ 0), with no exceptions.
3. Vectorize `compute_ground_distances` (arrays of `v`) and return
   $(Z_c, d_{long}, \sigma)$; per-pixel exceptions do not scale to thousands
   of roadway pixels per frame.

Additional pending item: camera **roll** (not modeled today). A 1° roll
tilts the horizon ~11 px at the edges of KITTI. The plane estimation of
R2 delivers it for free.

### 3.2 · 2D perception and relative depth

**Detector.** Candidates YOLOv8n/YOLO11n (or `-s` if P99 allows),
exported to ONNX with **NMS inside the engine** (EfficientNMS plugin) so NMS
is not done in Python. **Rectangular** input matching the sensor aspect
(KITTI 3.3:1 → e.g. 1024×320 instead of 640×640 with 70 % padding). Two
trade-offs to measure: INT8 with calibration (–30–40 % latency, risk on
small pedestrians) and resolution vs. recall of distant objects.

**Depth.** Depth Anything V2‑Small (ViT‑S, ~25 M parameters) as the
primary candidate; outdoor metric variant of the same size as second
option (R1). Multiple of 14 input (ViT patch); prefer wide aspect
(e.g. 924×280 ≈ same tokens as 518×518) so lateral resolution is not lost.
**[measure]** P95 and AbsRel for 3 sizes.

**Minimize inference-framework overhead.** Principles:
1. **No PyTorch in the runtime.** TensorRT 10 (`execute_async_v3`,
   `set_tensor_address`) + `cuda-python` for streams, events, pinned memory
   and copies. A single CUDA context.
2. **Preallocated buffers** (device and pinned host) per engine, fixed size,
   reused; zero `malloc` per frame.
3. **Preprocessing inside the ONNX graph**: `uint8 NHWC → float NCHW`,
   `/255`, mean/std. Upload 1 byte/pixel instead of 4 (1242×375×3 ≈ 1.4 MB,
   ~0.1 ms PCIe) and no `astype` or `transpose` in NumPy.
4. **CUDA Graph** per engine (capture of `enqueue`): the cost of launching 200
   kernels from Python drops to one call.
5. **Compact outputs**: detector → `[N_max, 6]` tensor already with NMS;
   depth → `float16` at network resolution (upsample to sensor size is
   unnecessary: sampling is done in network coordinates).
6. **Synchronization via events, not global `synchronize()`**, so the depth
   network keeps running while the detector is processed.

Indicative budget for the reactive loop at 60 Hz (16.7 ms), **[measure]**:
capture+H2D ≤ 1 ms · detector ≤ 5 ms (P95, under contention ≤ 8 ms) ·
tracking+fusion+TTC ≤ 3 ms · logging ≤ 1 ms → ~10 ms with margin.

### 3.3 · Metric fusion and ground constraint

**Three independent estimators per object**, each with its variance:

| Estimator | Formula | σ model | Validity domain |
|---|---|---|---|
| Ground $Z_g$ | $h\cos\alpha/\sin(\theta+\alpha)$ at the box base | $\sigma_g \approx \frac{d^2}{h}\sqrt{\sigma_\theta^2 + \sigma_v^2/f_y^2}$ | Box not truncated or occluded, class with contact, $d < d_{max}$ |
| Network $Z_n$ | $1/((\hat d_{med} - t)/s)$ | $\sigma_n \approx k\,Z$ (k ≈ 5–10 % **[measure]**) + term from $(s,t)$ uncertainty | Always; degraded if few roadway inliers |
| Height $Z_h$ | $f_y H_{class}/h_{px}$ | $\sigma_h/Z = \sigma_H/H$ (~10–13 %) | Box complete vertically |

Inverse-variance fusion:
$\hat Z = \frac{\sum_i Z_i/\sigma_i^2}{\sum_i 1/\sigma_i^2}$,
$\hat\sigma^2 = 1/\sum_i 1/\sigma_i^2$, preceded by a **consistency
test** (if two estimators differ by > 3σ combined, discard the
less reliable one per the gates, and mark the object as `inconsistent` so
the KF inflates R). More resilient than a fixed weight because:
- under a **bump** (transient pitch) $\sigma_g$ inflates via the
  online-estimated pitch uncertainty, and fusion migrates to $Z_n, Z_h$
  with no ad hoc logic;
- under **contact-point failure** the gate nulls $Z_g$ and two
  bounds remain;
- under **non-visible roadway** (traffic jam) $\sigma_n$ grows and $Z_g, Z_h$
  dominate in the near field, where they are better.

**Robust sampling inside the box.** Proposed rules:
1. Interior ROI: crop 25 % laterally and keep the central vertical band
   (35–75 % of height) → avoids ground below and sky/background above.
2. Statistic: **median and MAD** of disparity; inliers within
   $|\hat d - med| < 2.5 \cdot 1.4826\,MAD$; re-estimate median over
   inliers; $\sigma_{med} \approx 1.2533\cdot 1.4826\,MAD/\sqrt{n}$.
3. **Thin object** case (pedestrian, pole): the ROI may be mostly
   background, and the median falls on the background (lower disparity). Use the
   **highest-disparity mode** (closest surface) when the histogram is
   bimodal; for wide classes, the median.
4. Always work in the network's **disparity space** (where noise
   is more homogeneous), and convert to $Z$ after robustification.

### 3.4 · 3D tracking and ego-motion

**Association.** ByteTrack in 2D at 60 Hz (IoU + low/high score, no ReID) is
sufficient and cheap (< 0.5 ms for tens of boxes). The 3D state hangs off the
2D id. Trade-off vs. 3D association (Mahalanobis distance in $(X,Z)$):
3D is more robust to crossings but depends on a still-noisy metric; it can be
added as a second stage for boxes that ByteTrack leaves unmatched.

**EKF or linear KF?** Depends on **where** the measurement is defined:

| Design | State | Measurement | Filter |
|---|---|---|---|
| **I. Converted measurement** | $[X, Z, \dot X, \dot Z]$ in the camera-ground frame | already fused $(X, Z)$ (§3.3) | **Linear KF**, with $R_k$ computed per measurement via the conversion Jacobian (`converted-measurement KF`) |
| II. Image-space measurement | same | $(u_{center}, v_{base}, \hat d)$ | **EKF** (nonlinear $h(x)$: $u = f X/Z + c_x$, $\rho = 1/Z$) |
| III. Inverse depth | $[u, \rho, \dot u, \dot\rho]$ | $(u, \rho)$ | Linear KF in image; nonlinear for metric velocity |

Recommendation: **I for the MVP**. The process model (constant velocity)
is linear in Cartesian coordinates; nonlinearity lives only in the measurement
conversion, and is correctly captured by propagating the covariance through the Jacobian:

$$
R_k = J\,\mathrm{diag}(\sigma_u^2, \sigma_Z^2)\,J^\top,\quad
\sigma_X \approx \frac{Z}{f}\sigma_u + \frac{|u-c_x|}{f}\sigma_Z,\quad
\sigma_Z \text{ from fusion (grows }\propto Z\ldots Z^2)
$$

This answers "how to model noise growth with distance": not with an arbitrary
function but by **inheriting the fused variance** (§3.3),
which is already $\propto Z$ (network) and $\propto Z^2$ (ground). The EKF (II)
remains as an alternative if one decides to measure directly in pixels/disparity
to avoid prior fusion; it adds linearization and divergence risk with
variable $\Delta t$.

**Variable $\Delta t$.** Use capture timestamps (`perf_counter_ns` at
acquisition), never the processing clock. Process noise *discrete
white noise acceleration* per axis:
$Q(\Delta t) = \sigma_a^2 \begin{bmatrix}\Delta t^4/4 & \Delta t^3/2\\ \Delta t^3/2 & \Delta t^2\end{bmatrix}$
with $\sigma_a$ per class (car 2–3 m/s², pedestrian 1 m/s²). Out-of-order
measurements (old depth map, R6): back-prediction or inflation
of $R$ proportional to age.

**Ego-motion.** Without compensation, a static object appears to move at
$-v_{ego}$; that is **correct for TTC** (what matters is relative
velocity) but **incorrect for classifying static/moving and for the CV
model in turns** (in a turn, a static object describes an arc in the camera
frame; the KF interprets it as acceleration). Design:
- `EgoMotionProvider` interface → $(R_{k-1\to k}, t_{k-1\to k})$ or
  $(v, \omega)$; implementations: KITTI OXTS (GT for validation), CAN/wheel
  odometry, monocular VO with ground scale (advanced phase), and `Zero`.
- Compensation in the **prediction**: transform the previous state to the
  current frame before applying $F(\Delta t)$; relative velocity remains in the
  state, and absolute velocity is derived as $v_{rel} + v_{ego}$ for the
  classification logic.
- Treat ego uncertainty as additional process noise (not as
  truth).

### 3.5 · Kinematics and safety (TTC)

**Why $-Z/\dot Z$ fails.** With a static object and slow ego, $\dot Z\to
0^-$ and TTC → ∞ even at 1 m; with a lateral object that will **not**
cross our trajectory, TTC is finite and generates a false alarm. One must
separate *when* (time) from *where* (passage geometry).

**Quantities.** With relative position $p = (X, Z)$ and relative velocity
$v = (\dot X, \dot Z)$:

$$
t_{CPA} = -\frac{p\cdot v}{\|v\|^2}, \qquad
d_{CPA} = \|p + v\,t_{CPA}\|, \qquad
TTC_{corr} = t_{CPA}\ \text{if } d_{CPA} < w_{ego}/2 + w_{obj}/2 + m
$$

- **Proximity gate** (velocity-independent): $Z < Z_{near}$
  and $|X| < w_{corridor}/2$ → alert even if $v \approx 0$. Covers the nearby
  static object.
- **Trajectory gate**: TTC is only evaluated if $d_{CPA}$ falls in the
  corridor; otherwise the object is "passing" and at most `CAUTION`.
- **Noise robustness**: use the **conservative bound** $TTC_{low}$ computed
  with $\dot Z + k\sigma_{\dot Z}$ (k = 1–2) and $Z - k\sigma_Z$; do not
  escalate the alert if the track's `trace(P)` exceeds a threshold or the track
  age is < N frames (unconfirmed track).

**Deterministic state machine with hysteresis** per track:

```
NONE → CAUTION   if TTC_low < 4.0 s  or  Z < 12 m in corridor,  for ≥ 3 frames
CAUTION → WARNING if TTC_low < 2.5 s  or  Z < 6 m in corridor,   for ≥ 2 frames
WARNING → CRITICAL if TTC_low < 1.2 s or  Z < 3 m in corridor,   immediate
level down:  exit threshold = entry threshold × 1.3 (time) / × 1.2 (distance),
             sustained ≥ 10 frames (~170 ms at 60 Hz)
lost track:  keep level with decay 1 level/500 ms, never escalate
```

Thresholds are configuration parameters (per class and per ego
speed); the logic is a pure function `step(state, measurement, cfg) → state` with
no randomness or wall-clock dependence (only input timestamps),
so the same sequence produces the same output and can be tested
exhaustively with synthetic trajectories (frontal collision,
overtaking, lateral crossing, static object, intermittent track).

---

## 3. Critique of the proposed stack

| Choice | Verdict | Nuances |
|---|---|---|
| Python 3.11 + `uv` | Adequate | Runtime in Python is viable if the GPU does the heavy work and per-frame code is vectorized NumPy. CPU/frame budget ≤ 5 ms **[measure]**. |
| TensorRT via `tensorrt-cu12` (pip) | Adequate, with caution | `.engine` files are GPU + TRT version specific: generate them on the target machine (`scripts/export_trt.sh`), never version them. Plugins (EfficientNMS) come in the wheel. |
| `cuda-python` | Adequate | Verbose low-level API; encapsulate in `runtime/cuda.py` (streams, events, pinned buffers). Alternative `cupy` if ad hoc kernels are needed (e.g. per-box sampling on GPU). |
| PyTorch only for export | Correct | Must leave the runtime dependency group (H4). |
| `opencv-python-headless` | Adequate for CPU | No CUDA module; do not plan `cv2.cuda.*`. |
| NumPy < 2 | Acceptable | Fixed by binding compatibility; revisit when `tensorrt`/`cv2` allow it. |
| SciPy | Adequate | `linear_sum_assignment` (association), `stats` (Theil–Sen), little else. |
| Rerun SDK | Adequate | External viewer; control volume (R9). Version ≥ 0.18 uses gRPC. |
| ruff + mypy strict + pytest | Adequate | Keep `-m "not gpu and not slow"` in CI; GPU tests run locally with `scripts/`. |
| Docker only for CI | Correct | The CI image does not need CUDA; validate pure logic only. An optional `nvidia/cuda` image to reproduce engines, not for development. |

**Suggested absences:** `nvidia-ml-py` (NVML, measure real VRAM), `onnx` +
`onnxsim`/`onnxslim` (export), `polygraphy` (debug FP16 engine precision
layer by layer), and optional `hypothesis` for property tests of
geometry and the state machine.

---

## 4. Decisions deferred until metrics are available

| Decision | Metric that unlocks it | Phase |
|---|---|---|
| Detector resolution/input and n/s variant | Isolated and under-contention P95; KITTI car/pedestrian recall at > 30 m | F2 |
| Depth resolution and variant (relative vs. metric) | P95, NVML VRAM, AbsRel by bins after affine correction | F3 |
| Depth frequency (every 2 or 3 frames) | Detector P99 under contention | F3/F7 |
| Rectification A vs. C | CPU cost of `remap` at the target resolution | F1 |
| Threads vs. single-thread with two streams | End-to-end P99 of the reactive loop; CPU ms per frame | F7 |
| Need for EKF (II) | Linear KF (I) velocity error vs. KITTI GT | F5 |
| INT8 on detector | Δ latency vs. Δ recall on pedestrians | F8 |

---

## 5. Executive summary

1. The **ground bound is excellent below ~10 m and dangerous above
   20 m** without online pitch: the fusion design must be born from the
   variance model $\sigma_g \propto d^2$, and pitch must be estimated from the
   roadway plane of the network (scale-invariant).
2. The **affine scale is calibrated on the roadway**, not on boxes; the box
   contributes three bounds (ground, network, height) fused by inverse variance with
   a consistency test.
3. **One thread, two streams, CUDA Graphs, no torch**: the simplest and most
   deterministic dual-rate design; scale to threads/processes only if measurement
   requires it.
4. **Linear KF with converted measurement** and $R(Z)$ inherited from fusion;
   EKF only if measuring in pixels.
5. **TTC = time + geometry**: CPA + gates + hysteresis in a pure
   testable function.
6. Before anything else: fix H1, remove torch from the runtime, replace the
   benchmark proxy with real engines, and set up the percentiles + NVML harness.
