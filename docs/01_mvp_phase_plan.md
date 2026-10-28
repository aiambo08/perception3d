# 01 · Modular phase plan (isolatable MVPs)

> **Languages:** English (this page) · [Español](es/01_plan_fases_mvp.md)

> Deliverable 2 (§5.2 of the brief). Each phase produces an MVP that is tested
> in isolation (CPU tests in CI, GPU benchmarks locally) before being integrated
> into the asynchronous loop. The *Definition of Done* (DoD) items are measurable;
> when they depend on hardware they are expressed as percentiles. References
> R*/H* point to `00_critical_analysis.md`.

## Phase map and dependencies

```
F0 Geometry ✔ ──► F0.1 corrections ──► F1 Measurement and I/O infra
                                            │
                          ┌─────────────────┼──────────────────┐
                          ▼                 ▼                  │
                  F2 Detector TRT    F3 Depth TRT              │
                          │                 │                  │
                          └───────► F4 Metric fusion ◄─────────┘
                                            │
                                    F5 Tracking 3D + ego
                                            │
                                    F6 Safety (TTC/CPA)
                                            │
                                    F7 Dual-rate integration + Rerun
                                            │
                                    F8 Hardening (INT8, CI, packaging)
```

F2 and F3 are independent of each other; F4–F6 can be developed **without a GPU**
using recorded measurements (JSON/NPZ) from F2/F3 and the synthetic generator from F1.

---

## F0.1 · Geometry corrections (phase 0 closeout) — ✔ implemented

**Scope**
- `compute_ground_distances`: return `(z_c, d_long)` with
  $Z_c = h\cos\alpha/\sin(\theta+\alpha)$ (H1); also expose `ray_length`
  if needed.
- Vectorized version `PinholeGeometry.ground_hits(u, v) -> GroundHit`
  (`z_c, d_long, x_lat, ray_length, sigma_z, sigma_d, valid`; `nan` instead of
  exceptions) with the R2 variance model parameterized by
  `GroundNoiseModel(sigma_pitch_rad, sigma_v_px)`. Propagation uses
  central differences over the exact geometry, so it is consistent with roll
  and with the closed form $\sigma_d \approx (d^2/h)\,\sigma_\theta$.
- `undistort_points(uv)` in `ImageRectifier` (option A of §3.1).
- Optional roll in `ExtrinsicMountConfig` (default 0) and horizon as a
  line, not a row.
- Move `torch`/`torchvision` to `[project.optional-dependencies].export` (H4) and
  `tensorrt-cu12` (pinned to 10.x), `cuda-python`, `rerun-sdk` to `runtime`
  (R11). The CI-installable core is pure CPU.

**Note on the inherited test `z_cam > d_long`.** That invariant only holds
for ray length. For optical depth one has
$Z_c = d_{long}\cos\theta + h\sin\theta$, which is **smaller** than $d_{long}$
from $d \gtrsim 2h/\theta$ onward (≈ 19 m with 10° pitch). The test is
replaced by the correct invariants: `ray_length > d_long`,
`ray_length ≥ z_c` and the identity above.

**DoD**
- Closed-form value test: KITTI, `v=374` → `z_c ≈ 5.12 m`, `d_long ≈ 5.06 m`
  (1 cm tolerance).
- Consistency test: `deproject_pixel_with_depth(u, v, z_c)` returns a
  point with $Y$ equal to camera height ±1 mm when `pitch = 0`; with
  pitch ≠ 0, after rotating by $R_x(\theta)$.
- Variance test: `sigma_z(d=20 m, σ_θ=0.5°) ≈ 2.2 m` (±5 %).
- Green CI without torch installed.

---

## F1 · Measurement and I/O infrastructure — ✔ implemented (CPU)

**Scope**
- `utils/profiling.py`: `StageTimer` (CPU, `perf_counter_ns`, ring buffer per
  stage) with P50/P95/P99/max report, table, CSV/JSON and context
  `with timer.stage("detector"):`. GPU stages are fed with
  `timer.record(name, ms)` from CUDA events; `CudaStageTimer` is added in
  F2 alongside the first TensorRT engine, when it can be measured on the target GPU.
- `utils/vram.py`: `VramSampler` (thread at 10 Hz, injectable `sample_fn`; NVML
  via `nvidia-ml-py` in the `runtime` extra) → process and GPU VRAM peak.
- `runtime/buffer.py`: `LatestFrameSlot` (single slot, *latest-wins*,
  dropped-frame counter) and `FrameStamped(frame_id, t_capture_ns, img)`.
  `runtime/playback.py`: `play_into_slot` (fixed-rate playback with
  P50/P95/P99 of period and jitter).
- `io/sources.py`: `FrameSource` (protocol) with `VideoFileSource`,
  `ImageSequenceSource`, `KittiSequenceSource` (images + timestamps ns +
  calib + OXTS) and `V4L2Source` (MJPG, `CAP_PROP_BUFFERSIZE=1`).
- `sim/synthetic.py`: deterministic `SyntheticScene` (`(seed, k)`): constant-velocity
  cuboids + ego-velocity → noisy boxes, contact row,
  *affine* inverse depth (unknown scale/offset) and 3D/velocity GT;
  dense `ground_inv_depth_map()` of the ground. Base for F3–F6.
- `eval/kitti.py`: per-object GT from KITTI *tracking* labels
  (`location.z` = optical depth of the 3D box center, without projecting
  LiDAR) + `depth_metrics_by_bin` (AbsRel/RMSE/bias by distance bin).
- `scripts/profile_stage.py --stage {rectify,ground_hits,playback}`
  (`--kitti`, `--video`, `--hz`, `--vram`, `--json/--csv`).

**Measured (dev CPU, KITTI 1242×375, no GPU):** `rectify` P50 0.37 ms
/ P99 0.71 ms; `ground_hits` with 20 000 pixels P50 1.97 ms / P99 2.66 ms;
playback at 60 Hz: 60.5 Hz achieved, jitter P99 0.01 ms (sleep + spin of
1.5 ms). Indicative figures: the target machine is measured in F2.

**DoD**
- Play back a KITTI sequence at ≥ 60 Hz (source + `LatestFrameSlot`)
  with reported inter-frame jitter P99.
- Percentile report generated with a single command
  (`scripts/profile_stage.py --stage rectify`).
- 100 % of utilities covered by CPU tests.

---

## F2 · 2D detector on TensorRT — ✔ implemented (CPU/ONNX); GPU metrics pending

**Scope**
- `scripts/export_detector.py` (`export` group): YOLO(n|s) → ONNX with
  in-graph preprocessing (uint8 NHWC → normalized float NCHW) + EfficientNMS →
  FP16 `.engine` with rectangular input profile (e.g. 1024×320 for KITTI).
- `detection/detector_trt.py`: generic `TrtEngine` (load, preallocated
  buffers, `execute_async_v3`, optional CUDA Graph, stream/priority) and
  `Detector(engine).infer_async(frame, stream) -> DetectionsHandle` with
  `wait() -> Detections[N,6]` (x1,y1,x2,y2,score,cls) **in input-frame
  coordinates**, including the letterbox inverse.
- Bench `scripts/bench_detector.py`: {isolated} with P50/P95/P99 + VRAM.

**How it landed**
- `detection/letterbox.py`: `Letterboxer` with preallocated `uint8` canvas
  (fill 114, centered, `INTER_AREA` when downscaling) and `LetterboxParams`
  with `to_net`/`to_frame`/`clip_to_frame`; the inverse is exact because it stores
  the actual resized dimensions (independent X and Y scales).
- `detection/onnx_surgery.py`: `prepend_uint8_preprocess` (input
  `images_u8` uint8 `[1,H,W,3]` → Cast → Transpose → Gather BGR→RGB → ×1/255)
  and `append_efficient_nms` (`Transpose` + `Split` → `EfficientNMS_TRT`,
  `box_coding=1`, outputs `num_dets/det_boxes/det_scores/det_classes` with
  static shape). Both verified with onnxruntime on synthetic graphs.
- `detection/detector_trt.py`: `Detector(engine: EngineBackend)` with injectable
  backend (`EngineBackend`/`InferenceHandle` protocol), so coordinate
  logic, filters and timing are tested on CPU with a fake engine. `TrtEngine`
  lazily imports TensorRT/cuda-python, uses
  `set_tensor_address` + `execute_async_v3`, *pinned* host, CUDA events for
  `gpu_ms` and for `wait()` (no `cudaDeviceSynchronize`), high-priority
  stream and optional CUDA Graph.
- `detection/config.py` + `configs/models.yaml`: weights, `input_hw`, NMS,
  classes to keep, COCO→KITTI mapping and DoD thresholds in one place.
- `eval/detection.py`: IoU, greedy *matching* by score and `recall_by_type`
  (COCO→KITTI). Only KITTI *moderate* difficulty GTs score (height
  ≥ 25 px, `occluded` ≤ 1, `truncated` ≤ 0.3); the rest are ignored (a hit
  on them is neither TP nor FP). Aggregated recall over several sequences
  (`merge_recall`) with 95 % Wilson interval: the DoD verdict is
  PASS/FAIL only if the interval lies entirely on one side of the threshold, otherwise
  INCONCLUSIVE.
- `scripts/export_trt.sh` (`trtexec`, FP16/INT8) and `scripts/bench_detector.py`
  (P50/P95/P99 of `det.gpu`, `det.postprocess`, `detector_e2e`; NVML VRAM
  baseline/load/peak; optional recall with KITTI labels; JSON/CSV).
- Context for each measurement (F2 and F3): `utils/gpu_state.py` samples via NVML
  SM clock, temperature, power and clock-throttling reasons, and the JSON
  reports what fraction of the run was limited by temperature or power.
  Under WDDM (native Windows **and** WSL2) NVML does not give per-process memory: process
  VRAM is then estimated as *device usage − baseline before
  the CUDA context* (`process_source: device_delta`, upper bound).

**DoD**
- P95 ≤ 6 ms isolated on the target GPU (FP16, YOLO-n, 1024×320) **[measure]**;
  otherwise document and choose resolution/variant.
- Recall ≥ 0.85 (IoU 0.5) for `Car` and ≥ 0.6 for `Pedestrian` on 200
  labeled KITTI frames (export validation, not of the model) **[measure]**.
- Process VRAM (NVML) ≤ 900 MB including context **[measure]**.
- CPU test: letterbox inverse is exact (box round-trip). ✔

The three **[measure]** items require the target GPU; this phase's
development environment has no CUDA driver, so `TrtEngine` is
written against the TensorRT 10 / cuda-python 12 API but **not executed**.
First thing to do on the target machine:

```bash
uv pip install -e ".[export]" && uv run python scripts/export_detector.py
uv pip install -e ".[runtime]" && bash scripts/export_trt.sh \
    models/detector_1024x320.onnx models/detector_1024x320_fp16.engine fp16
uv run python scripts/bench_detector.py --kitti-root <tracking/training> \
    --seqs 0000 0001 0020 --frames 200 --json reports/f2_detector.json
```

**First measurement** (RTX 4060 Laptop, Windows 11, TRT 10.16 FP16):
`detector_e2e` P95 = 2.41 ms ✔. Recall `Car` 84.4 % on 0000 with the old
filter (height only), which counted heavily occluded or truncated cars and with a
single sequence does not distinguish 84 % from 85 %; remains to be repeated with *moderate* and
several sequences. Per-process VRAM was unavailable (WDDM). In F3,
Spearman 0.959 ✔ and total VRAM 1370 MB ✔, but latency went from a cold P50 of
15 ms to 40–46 ms after hours under load, pointing to thermal
or power throttling. WSL2 shares the WDDM driver with Windows, so it does not
remove that source; the repeat must be accompanied by
`gpu_state` logging.

---

## F3 · Relative depth on TensorRT — implemented (GPU pending)

**Scope** (implemented, PR F3)
- `scripts/export_depth.py`: `depth-anything/Depth-Anything-V2-Small-hf`
  (transformers; `export` extra) → static ONNX `pixel_values [1,3,H,W]` →
  surgery `prepend_uint8_preprocess(mean, std)`: input `images_u8 [1,H,W,3]`
  BGR, RGB swap, `/255` and ImageNet normalization folded into `Mul+Add`. One
  ONNX per `depth.input_sizes` size (width, multiple of 14: 280×924 by
  default ≈ 1 320 tokens, 252×840, 322×1064). The outdoor metric
  variant (`variant: metric_outdoor`, `max_depth: 80` in the checkpoint) uses the same
  path with `kind="metric_depth"`.
- `runtime/trt_engine.py`: `TrtEngine`/`EngineBackend`/`InferenceHandle`
  extracted from the detector and shared; `InferenceHandle.ready()`
  (`cudaEventQuery`) allows non-blocking polling.
- `depth/preprocess.py`: `DepthResizer` (stretch to network resolution on a
  preallocated uint8 canvas, no letterbox: padding would introduce artificial
  tokens in global attention) + `DepthResize` (exact
  frame↔net mapping, `frame_to_index`).
- `depth/depth_trt.py`: `DepthEstimator.infer_async(frame, frame_id,
  t_capture_ns) -> DepthHandle` → `DepthMap(frame_id, t_capture_ns,
  values: float16[h,w], kind, resize, gpu_ms)` with `sample(u,v)`,
  `inverse_depth()` and `to_frame_resolution()`. Relative output is called
  `disparity` and is never interpreted as distance: $\hat d = s\,/Z + t$ with
  $(s,t)$ unknown per frame (resolved by F4).
- `runtime/contention.py`: R3 loop with two engines — depth enqueued **before**
  the detector (worst case for its queue), `wait()` only on the detector,
  `ready()` to collect the map when it finishes; logs `det_e2e`,
  `depth_e2e`, maps produced, skipped slots and frame lag.
- `eval/lidar.py` + `eval/depth_sanity.py`: Velodyne→image projection
  (KITTI raw and tracking), nearest return per pixel, roadway mask
  (below horizon ∧ outside boxes), Spearman $\rho$ disparity vs $1/Z$ and
  AbsRel after Theil–Sen affine fit.
- `scripts/bench_depth.py`: `--mode depth` (all sizes) and
  `--mode matrix` ({detector, depth, both}); P50/P95/P99 of `depth.gpu`,
  `depth_e2e`, `det.gpu` under contention; NVML VRAM; `--kitti-drive` adds
  LiDAR sanity; DoD report and JSON. Replaces the EfficientNet-B3 proxy (H2).

**DoD**
- P95 ≤ 25 ms isolated for the chosen size **[measure]**; detector
  P99 under contention ≤ 8 ms **[measure]**.
- Total VRAM (both engines, NVML) ≤ 2.0 GB **[measure]**.
- Spearman correlation ≥ 0.95 between predicted disparity and LiDAR 1/Z on
  roadway pixels **[measure]**. Validates relative order, BGR/RGB channel,
  normalization, resize and FP16 construction — **not** metric scale.
  If it fails: `polygraphy run --trt --onnxrt` on the ONNX.
- CPU tests: exact frame↔net mapping and `sample()`, `DepthEstimator` with
  simulated engine (fp16, metadata, `[1,H,W]` and `[1,1,H,W]`), mean/std
  surgery in ONNX Runtime, synthetic LiDAR projection with $\rho \approx 1$,
  contention loop with simulated engines. ✔

No CUDA driver in the development environment; first thing on the target GPU:

```bash
uv pip install -e ".[export]" && uv run python scripts/export_depth.py
uv pip install -e ".[runtime]"
for s in 924x280 840x252 1064x322; do
  bash scripts/export_trt.sh models/depth_$s.onnx models/depth_${s}_fp16.engine fp16
done
uv run python scripts/bench_depth.py --mode depth --frames 300            # choose size
uv run python scripts/bench_depth.py --mode matrix --size 924x280 --pace-hz 60 \
    --kitti-drive <2011_09_26_drive_0005_sync> --frames 150 --json reports/f3_matrix.json
```

Decisions that depend on the measurement: input size (280×924 vs
252×840 if P95 > 25 ms), `depth_every` (1 → 30 Hz at 60 fps detector if
`depth_e2e` P95 < 33 ms; 2 otherwise) and whether INT8 is needed for the ViT (in
general not worth it: FP16 attention already dominates and INT8 calibration of
ViT degrades relative order).

---

## F4 · Metric fusion and ground constraint — ✔ implemented (CPU/NumPy)

**Implemented scope**
- `depth/ground_solver.py`:
  - `GroundGrid`/`GroundGridCache`: $1/Z_c$, $\partial Z/\partial v$,
    $\partial Z/\partial\theta$ and rays at **network resolution**, cached by
    (resize, intrinsics) and by (resize, $h$, pitch quantized to 0.1°, roll);
    separable construction (rows × columns), ~0.9 ms at 462×140 and ~3.7 ms at
    924×280 on the development CPU — hence pitch quantization.
  - `sample_road`: static mask (below horizon, central band, $Z_c$
    in range) ∧ outside rasterized boxes; ≤ 2 000 pixels.
  - `robust_affine_fit`: median of random-pair slopes →
    least squares with inliers (3σ MAD) → $(s,t)$, 2×2 covariance,
    inlier ratio; rejects $s \le 0$ and degenerate ranges.
  - `AffineKalman`: 2D random walk with $\chi^2_{2}$ gating (99 %) and
    reset after $n$ consecutive rejects (scene/model change).
  - `PitchEstimator`: online pitch from objects with known height
    (`pitch_from_contact` solves $Z_c(v_b;\theta)=f_yH/h_{px}$ per
    detection; robust median → 1D Kalman with max step).
  - `fit_plane`: PCA plane on the deprojected roadway → pitch/roll/height;
    kept as a diagnostic, **not** as a corrector (see finding).
- `depth/sampling.py`: interior ROI (`shrink`), regular spatial subsampling
  to bound cost by `max_pixels`, median/MAD, median σ
  ($1.2533\,\sigma/\sqrt{n}$), bimodality via 1D Otsu (thin classes →
  near mode), dilated exclusion of occluding boxes;
  `disparity_to_depth_batch` with delta method on $(\hat d, s, t)$.
- `depth/fusion.py`: cues $\rho_g, \rho_n, \rho_h$ in **inverse depth**
  with gates (truncated contact/above horizon, no prior, no map, box
  outside fit range, height < 12 px, outside $[1, 80]$ m), BLUE
  under $\Sigma = D + cc^\top$ with $c_i = (\partial\rho_i/\partial\theta)\sigma_\theta$
  (height prior has $c_h = 0$), $\chi^2$ consistency test that
  inflates variance and marks `INCONSISTENT`, `Measurement3D` (meters, σ,
  individual cues, weights, flags, `frame_id`, `t_capture_ns`,
  `center_offset_m`), `MetricFusionStage.process()` (latest available map +
  `depth_lag_frames`).
- `configs/fusion.yaml`: per-class heights/lengths with COCO↔KITTI aliases,
  noises, gates and sampling.
- Evaluation: `eval/fusion_synthetic.py` + `scripts/eval_fusion_synthetic.py`
  (dense synthetic scene from F1 extended with $\hat d$ maps painted per
  object, several seeds); `eval/fusion_kitti.py` +
  `scripts/eval_fusion_kitti.py` (KITTI tracking: near face and center vs
  3D labels, `gt`/`detector`, maps from TensorRT engine, precomputed
  `.npy` or no network).

**Finding: pitch is not observable from the roadway.** With roll 0,
$1/Z_c(v) = (\sin\theta + y\cos\theta)/h$ is *exactly* affine in the row;
if the assumed pitch is $\theta'$, $\hat d$ remains exactly affine in
$1/Z_c(\theta')$: the fit is perfect and $(s,t)$ absorb the error (especially
$t$). Deprojecting the roadway with that $(s,t)$ and fitting a plane returns
$\theta'$, not the true pitch (`test_pitch_is_unobservable_from_road_plus_affine_map`).
The original plan's `PlaneFitter` cannot correct pitch; a second
**metric** cue can: known-height objects (implemented) or a
metric model with $t \equiv 0$. Consequence for fusion: $\rho_g$ and
$\rho_n$ share the pitch error (hence the correlated covariance), and the
height prior is what rescues the far field when pitch is wrong.

**Why fuse in $\rho = 1/Z$.** $\rho_g$ is linear in the row with
sensitivity $\partial\rho_g/\partial\theta \approx 1/h$ independent of
range; $\rho_n$ is affine in the network output by construction; $\rho_h
\propto h_{px}$. In $Z$ the same errors are heavy-tailed toward far
($Z=1/\rho$ blows up near the horizon) and a Gaussian BLUE in $Z$
underweights exactly the bad far ground samples. Output in meters:
$Z=1/\hat\rho$, $\sigma_Z=\sigma_\rho/\hat\rho^2$.

**Conventions.** $Z$ of `Measurement3D` is the optical depth of the
**nearest contact** (what the bottom box row sees);
`center_offset_m` (mean class length) maps to the object center used by
KITTI labels and the tracker. In KITTI evaluation the primary reference
is the near face, $z - \tfrac12(l|\sin r_y| + w|\cos r_y|)$.

**DoD (measured on CPU, synthetic, `scripts/eval_fusion_synthetic.py`)**
- Nominal: fused P95 18.2 % ≤ best individual cue 22.6 % (height);
  P50 5.1 % vs 5.9 % (network) / 6.2 % (height) / 8.3 % (ground). ✔
- Pitch perturbed 1°: with online pitch the fused median degrades 0.1 %
  (< 30 %); without correction, $Z_g$ degrades 123 % (> 100 %) and fusion 84 %. ✔
- CPU cost solver + sampling, 20 boxes, 924×280 map (F3 size): P95 **1.93 ms** in
  steady state (cached grid) ≤ 2 ms ✔; **3.02 ms** counting
  frames that rebuild the grid on quantized pitch change. Measured on
  the development CPU, not the target laptop. Times **exclude**
  synthetic map generation.
- KITTI: AbsRel ≤ 10 % in 0–30 m and ≤ 20 % in 30–60 m for non-truncated `Car`
  **[measure]** — requires the F3 depth engine on the target GPU
  (`scripts/eval_fusion_kitti.py --engine …`) or `.npy` maps dumped from
  it. The harness is tested with a synthetic sequence in KITTI
  tracking format, which validates wiring, not the DoD.

```bash
uv run python scripts/eval_fusion_synthetic.py --json out/f4_synth.json
# Target GPU (KITTI tracking training/):
uv run python scripts/eval_fusion_kitti.py --root <training> --seq 0000 \
    --engine models/depth_924x280_fp16.engine --json out/f4_kitti_0000.json
uv run python scripts/eval_fusion_kitti.py --root <training> --seq 0000     # no network: ground + height
```

---

## F5 · 3D tracking and ego-motion — ✔ implemented; KITTI velocity DoD **not met** (accepted limitation)

**Scope**
- `tracking/byte_tracker.py`: 2D ByteTrack (IoU, two thresholds, Hungarian with
  `scipy.optimize.linear_sum_assignment`), stable ids, age/hits. Box KF
  $[c_x, c_y, w, h]$ + velocities with real $\Delta t$; association restricted
  by class group (`car/van/truck/bus` → vehicle, etc.).
- `tracking/kalman_filter.py`: linear CV KF in $[X, Z, V_X, V_Z]$
  with $F(\Delta t)$, $Q(\Delta t)$ CWNA per class, $R_k$ per measurement from
  `Measurement3D` (design I); delayed measurement with $H = [I, -\ell I]$ and
  $R + q\ell^3/3$ (R6); NIS $\chi^2$ gating. Vectorized prediction/update
  for all tracks in the frame (`predict_batch`/`update_batch`).
- `tracking/ego_motion.py`: `EgoMotionProvider` (protocol) with
  `ZeroEgoMotion`, `ConstantEgoMotion` and `OxtsEgoMotion` (interpolated
  `vf`, `vl`, `wu`, IMU→camera lever arm). Prediction expresses state
  in the new pose axes: $p' = R^\top(\psi)(p + V\Delta t - t)$,
  $V' = R^\top V$.
- `tracking/tracker3d.py`: orchestration (ByteTrack → F4 measurement →
  3D predict/update → life) → `Track3D` with position, absolute and
  relative velocity ($V - v_{ego}$) and their covariances, NIS and
  `MotionState` (static/moving/unknown).

**Decisions**
- **Linear KF, not EKF.** With the measurement already in metric $(X, Z)$ (F4) the
  observation model is linear; the pixel→metric nonlinearity and the growth
  $\sigma_Z \propto Z^2$ are already in $R_k$, including the along-ray
  correlation $\mathrm{cov}(X,Z) = (X/Z)\,\sigma_Z^2$
  (`position_covariance`). An EKF would only help if measuring in
  $(u, 1/Z)$, at the cost of per-track Jacobians.
- **Static/moving by statistical test**, not only by threshold: static
  candidate if $|V| < 1$ m/s **or** $V^\top P_{VV}^{-1} V < \chi^2_{2,0.95}$;
  moving if $|V| > 2$ m/s **and** $\chi^2 > \chi^2_{2,0.999}$; 0.3 s hysteresis.
  With a fixed threshold at 30–40 m the $V$ noise exceeded 1 m/s and only 45 % of
  statics in a curve were labeled (now 100 %, 2.6 % of movers as
  static). Without absolute ego-motion (`Zero`) the label is `unknown`.
- **Identities only in 2D.** The 3D stage never changes ids, so F5 ID
  switches are those of ByteTrack with the given configuration; the DoD
  "≤ reference +10 %" holds by construction and the script reports them
  for comparing configurations (thresholds, GT boxes vs. detector). As an
  absolute criterion `idsw_per_100_matches` is reported (proposed threshold,
  pending fixation with data: ≤ 1 IDSW/100 matches with detector).

**DoD (measured on CPU, synthetic, `scripts/eval_tracking_synthetic.py`, 3 seeds)**
- Relative $\dot Z$ RMSE at 15 m after 1 s: **0.39 m/s** ≤ 0.5 ✔.
- Random $\Delta t \in [10, 50]$ ms: RMSE 0.36 m/s, NEES > $\chi^2_{4,0.99}$
  on 0 % of samples (no divergence) ✔.
- Turn 0.2 rad/s with true ego-motion: 100 % of statics labeled
  static after 0.5 s ✔ (synthetic stand-in for the OXTS DoD).
- `Zero` on a straight: relative velocity identical to the ego-motion case ✔.
  On a curve `Zero` diverges (NEES ≫, $V_X$ RMSE ≈ 4 m/s): unmodeled camera
  yaw is a $\omega \times p$ term that no CV absorbs; that is the
  reason to use a yaw-aware provider in production.
- CPU cost `Tracker3D.step`, 30 tracks: P95 **1.0–1.3 ms** on the development
  CPU (shared Xeon, high run-to-run variance) vs
  1 ms target — **[measure]** on the laptop. If it does not drop, next lever:
  structure-of-arrays state (no per-track objects).
- KITTI tracking (3 sequences, `scripts/eval_tracking_kitti.py`): relative
  velocity RMSE ≤ 1.0 m/s in 0–30 m and ≥ 90 % statics with OXTS
  **[measure]**. Also needs `data_tracking_oxts.zip` (→ `training/oxts/`).
  Velocity GT: ±2-frame central difference of the label in G and, with
  OXTS, $v_{rel} = \dot p + \omega(-Z, X)$; static if $|v_{rel} + v_{ego}| < 0.5$ m/s.
  First measurement (user laptop, GT boxes, OXTS): RMSE 1.20 / 1.62 /
  0.86 m/s on 0000 / 0001 / 0020 ✘; statics 99–100 % ✔. Diagnosis in the
  JSON (`diagnostics`): error by Z bins (0–10/10–20/20–30/30–60 m) with
  bias, RMSE, P50/P95; GT dispersion (±2 vs `--gt-alt-window`
  ±5) as a lower bound on RMSE; error on straight vs. turn (|ω| ≥ 0.05 rad/s);
  and separate timings `fusion_ms` (F4) / `tracker_ms` (F5).
  With network (0000/0001/0020): GT dispersion 0.07–0.40 m/s (not limiting);
  `tracker_ms` P95 0.58–0.61 ms ✔; $V_Z$ bias ≈ constant with
  distance (+0.75 m/s on 0000, +0.75–1.8 on 0001, ≈ 0 on 0020) — signature of
  scale error × ego speed, not noise ∝ Z² — and tails on 0001
  (P50 1.2, P95 7–10 m/s). Added: regression `vz_err_vs_ref_vz` /
  `vz_err_vs_ego_fwd` (slope $k-1$ ⇒ range scale; slope only
  vs $v_{ego}$ ⇒ OXTS) and robust KF update
  (`filter.robust_chi2`: NIS > 5.99 ⇒ $R\cdot\mathrm{NIS}/5.99$;
  inliers unchanged, synthetic DoDs identical).
  Result: robust with no effect (RMSE 0001 3.31 → 3.30); slopes ≈ 0 vs
  GT and OXTS, and the +0.8–1.0 m/s shift persists with `--ego zero`
  ⇒ neither scale nor OXTS. F4 on 0001 passes per frame (AbsRel 9.9 %/11.4 %) but with
  468 `NET_BIMODAL` and 257 `INCONSISTENT` on 1432 boxes. Next diagnosis:
  `vz_err_vs_ref_az` (slope $-\tau$ = CV delay), `by_ref_az_mps2`,
  `by_track_age_s`; A/B with `--q-vehicle` and `--arbitration inflate`
  (F4: BLUE of the three tracks with inflated σ instead of discarding the discrepant one,
  no dominant-source jump). `eval_fusion_kitti.py` reports `stability`:
  |ΔZ_fus − ΔZ_gt| between consecutive frames per track (P50/P95/RMS), with and without
  dominant-track change.
  A/B result (0001): `inflate` lowers inter-frame jump P95 from 1.25 to
  0.70 m (with dominant-track change: 17.7 → 1.2 m), improves AbsRel
  (9.9 → 8.7 % and 11.4 → 9.1 %) and leaves F5 with network equal to without network (RMSE 0001
  3.30 → 1.64 m/s, statics 79 → 99.9 %) ⇒ `gates.arbitration: inflate` by
  default. CV delay does not explain the bias (slope vs $a_z$
  −0.18 s; with $a_z≈0$ still +0.94 m/s; `--q-vehicle 4` no effect). Per-track
  drift of each F4 track's error localizes it in the ground: d(err)/dt
  +2.2 m/s (network +0.2, height −0.01, fused +0.9 ≈ tracker bias); the ground
  underestimates Z from −5 % at 5 m to −15 % at 35 m and down to −25 % on frames 50–150,
  signature of badly estimated pitch/slope (≈ 0.5–1.2°). Added diagnosis:
  `pitch_series` / `pitch` in the `eval_fusion_kitti.py` JSON (filtered pitch
  vs median implicit pitch from heights; the residual is the pitch
  error left in $Z_g$) and `--pitch-q`, `--sigma-pitch-deg` overrides in
  both KITTI scripts.
  Result (0001, with network): filtered pitch averages 0.48° above
  the height-implied one (residual P95 1.9°); with $Z≈20$ m, $h=1.65$ m and
  a car approaching at 9 m/s that is worth $2Z\,\Delta\theta/h\cdot|\dot Z|≈1.8$ m/s,
  almost all of the ground drift (+2.2 m/s). `--pitch-q 1` improves AbsRel
  (8.7 → 8.0 % and 9.1 → 8.1 %) but barely the drift (2.2 → 2.0 m/s), worsens the
  inter-frame jump (P95 0.70 → 0.95 m) and leaves F5 unchanged (0001: 1.54 → 1.47 m/s,
  bias +0.92). Two causes identified in the pitch series: (1) the KITTI nominal
  in `camera_kitti.yaml` (2.5°) vs −0.06° from heights, with
  `max_step_deg` = 3° clipping measurements below −0.5° (inference: the
  JSON does not count rejects); (2) the filter locks on its $\chi^2$ gate: the
  measurement variance $\pi/2/\sum w$ (σ ≈ 0.1°) is much smaller than the real
  dispersion (±1° between frames), follows noise up to 2.9° (frames 52–96) and stays
  frozen at 1.88° (frames 108–128, the stretch with −25 % in $Z_g$) until its
  uncertainty grows.

**Accepted limitation (project decision).** F5 closes with the relative-velocity
DoD unmet in urban traffic: 0–30 m RMSE of 1.22 m/s on
0000 and 1.47–1.54 m/s on 0001 (limit 1.0), 0.88 m/s on 0020 ✔; statics
≥ 98.8 % ✔; 0 ID switches with GT boxes ✔; `Tracker3D.step` P95 0.58–0.61 ms ✔.
The $+0.9$ m/s bias in $V_Z$ is a systematic error of F4's ground cue
(pitch), not of the tracker, and F6 absorbs it with conservative bounds
($TTC_{low}$ with $-\dot Z + k\sigma_{\dot Z}$).

**Pitch filter correction (implemented and measured).**
(1) `configs/camera_kitti.yaml` changes the nominal pitch from 2.5° to 0°, so
`max_step_deg` = 3° no longer clips the low side; (2) `robust_r`: median
variance is replaced by $\pi/2\cdot\sigma_{MAD}^2/n$ when it exceeds 4× the formal
$\pi/2/\sum w$; (3) `soft_gate`: outside the $\chi^2$ gate $R$ is inflated (Huber)
instead of discarded, and `reset_after_gated` = 10 consecutive frames outside the gate
reset the filter to the measurement; (4) F4 JSON `pitch_series` and `pitch`
count discarded measurements per frame (`n_discarded`) and whether the frame was
outside the gate (`gated`). `--pitch-filter legacy --nominal-pitch-deg 2.5`
reproduces the previous behavior for A/B. In simulation (random walk,
−1.5° step, 1° inter-object dispersion, 10 % outliers) pitch RMSE
drops from 0.47–1.01° to 0.32–0.70°; F4 synthetic DoDs still hold.

Measurement on KITTI (FP16 network, GT boxes, OXTS), previous filter → robust:

| | 0000 | 0001 | 0020 |
|---|---|---|---|
| RMSE V 0–30 m (m/s) | 1.21 → 1.21 | **1.64 → 1.25** | 0.87 → 0.85 ✔ |
| Bias $V_Z$ 10–20 m (m/s) | +0.74 → +0.77 | **+0.93 → +0.08** | −0.15 → −0.17 |
| Bias $V_Z$ in curve (\|ω\| ≥ 0.05 rad/s) | +0.82 → +0.77 | +1.26 → +1.11 | — |

F4 0001: AbsRel 0–30 / 30–60 m 8.7 / 9.1 % → 6.5 / 5.7 %, mean pitch residual
−0.48° → +0.18°, 4 of 447 frames outside the gate. On a straight the 0001
$V_Z$ bias disappears (+0.77 → +0.09 m/s); the DoD is still unmet on 0000/0001 due to
curve bias (72 of 81 samples on 0000 are in curve), which does not
depend on pitch.

**Curve bias diagnosis.** `run_tracking_kitti` stores per sample the
signed yaw rate, $X$ and position error (tracker − GT), and
`diagnostics()["turn"]` separates candidate causes:
- `vz_err_vs_omega_x` / `vx_err_vs_omega_z`: fit of error vs the rotating-frame
  term $\omega\cdot(-Z, X)$; slope ≈ ±1 ⇒ term missing or with
  flipped sign (the static-point-in-curve test fixes the
  GT reference convention);
- `by_turn_direction` (left / straight / right): mean position error
  `mean_ez_m` (includes the constant object-center ↔ label bottom-center
  offset), `mean_ez_rel` and its derivative `mean_dez_dt_mps`; if
  `vz_err_minus_range_drift` ≈ 0, the bias comes from range drift during
  the curve (depth/ground), not from ego-motion compensation.

The JSON adds `pooled_diagnostics` (all sequences together), `lever_arm_m` and
`oxts_offset_ms`. `--oxts-offset-ms` shifts the OXTS clock relative to the camera
(ego-motion and GT rotation term at once) and `--lever-arm` (1.08 m by
default) the lever arm, to sweep both hypotheses.

```bash
uv run python scripts/eval_tracking_synthetic.py --json reports/f5_synth.json
# KITTI tracking training/ (+ oxts/):
uv run python scripts/eval_tracking_kitti.py --root $KT --seqs 0000 0001 0020 --ego oxts \
    --json reports/f5_kitti.json                                   # no network: ground + height
uv run python scripts/eval_tracking_kitti.py --root $KT --seqs 0000 0001 0020 --ego oxts \
    --engine models/depth_924x280_fp16.engine --boxes detector --json reports/f5_kitti_det.json
# A/B of the pitch filter correction (with network, GT boxes):
uv run python scripts/eval_tracking_kitti.py --root $KT --seqs 0000 0001 0020 --ego oxts \
    --engine models/depth_924x280_fp16.engine --pitch-filter legacy --nominal-pitch-deg 2.5 \
    --json reports/f5_kitti_pitch_legacy.json
uv run python scripts/eval_tracking_kitti.py --root $KT --seqs 0000 0001 0020 --ego oxts \
    --engine models/depth_924x280_fp16.engine --json reports/f5_kitti_pitch_robust.json
# Curve bias diagnosis: 21 sequences, then OXTS offset and lever-arm sweep
SEQS=$(seq -f %04g 0 20)
uv run python scripts/eval_tracking_kitti.py --root $KT --seqs $SEQS --ego oxts \
    --engine models/depth_924x280_fp16.engine --json reports/f5_turn_base.json
for OFF in -100 -50 50 100; do
  uv run python scripts/eval_tracking_kitti.py --root $KT --seqs $SEQS --ego oxts \
      --engine models/depth_924x280_fp16.engine --oxts-offset-ms $OFF --json reports/f5_turn_off$OFF.json
done
for ARM in 0 2.16; do
  uv run python scripts/eval_tracking_kitti.py --root $KT --seqs $SEQS --ego oxts \
      --engine models/depth_924x280_fp16.engine --lever-arm $ARM --json reports/f5_turn_arm$ARM.json
done
```

**Sweep result (21 sequences).** Slope vs `ω·X` between −0.09 and −0.13
across all variants: the rotation term is present and with the correct sign. Offset
±100 ms and arm 0–2.16 m move curve bias < 0.05 m/s: discarded as cause.
In left turns (483 of 13930 samples) V_Z bias is +0.96 m/s, of which
+0.45 is F4 range drift; concentrated in few sequences (0007: +4.7 m/s with 26
samples). What blocks the DoD is not bias but convergence: RMSE 1.83 m/s on
1–2 s tracks vs 0.90 m/s with more than 4 s.

**Filter consistency.** With a wide velocity prior (σ_v0 = 15 m/s) the KF already
equals, after two updates, the finite difference of the first measurements, so
initializing V by differences would change nothing. What decides convergence is whether
`R` (from F4) and `q` are realistic. `diagnostics.consistency` gives, by track age, mean
NIS (≈ 2 if `R`/`q` are correct; 5 % above 5.99) and `ratio` = real V_Z RMSE /
predicted V_Z σ (> 1 ⇒ the filter trusts measurements too much). `filter.r_scale` /
`--r-scale` scales `R`; together with `--q-vehicle` allows sweeping them:

```bash
for R in 1 2 4; do
  uv run python scripts/eval_tracking_kitti.py --root $KT --seqs $SEQS --ego oxts \
      --engine models/depth_924x280_fp16.engine --r-scale $R --json reports/f5_conv_r$R.json
done
for Q in 0.25 4; do
  uv run python scripts/eval_tracking_kitti.py --root $KT --seqs $SEQS --ego oxts \
      --engine models/depth_924x280_fp16.engine --q-vehicle $Q --json reports/f5_conv_q$Q.json
done
```

**Sweep result (21 sequences, pool RMSE V 0–30 m).**

| R · q | 0–30 m | track > 4 s (RMSE / P95) | track 1–2 s | mean NIS | min `static_frac` |
|---|---|---|---|---|---|
| 1 · 1 | 1.09 | 0.90 / 1.88 | 1.83 | 0.27 | 0.83 |
| 2 · 1 / 4 · 1 | 1.15 / 1.22 | 0.99 / 1.10 | 1.90 / 2.01 | 0.19 / 0.13 | 0.99 / 1.00 |
| 1 · 0.25 | 1.17 | 1.09 / 2.24 | 1.87 | 0.48 | 0.63 |
| 0.5 · 1 | 1.07 | 0.84 / 1.73 | 1.83 | 0.41 | 0.70 |
| **1 · 4** | 1.10 | **0.82 / 1.67** | 1.83 | 0.16 | **1.00** |
| 0.5 · 4 / 0.25 · 4 | 1.13 / 1.18 | 0.83 / 0.85 | 1.85 / 1.88 | 0.24 / 0.38 | 1.00 / 0.91 |

0–30 m RMSE stays at 1.07–1.22 m/s and 1–2 s tracks at 1.83–2.01
with any tuning. NIS remains well below 2 even with `R·0.25`, while
absolute error is high. Therefore F4 error is time-correlated (range
drift), and a KF with white noise cannot average it out: the limit is in
depth, not the tracker. **Decision:** `dynamics.vehicle.q = 4`, `r_scale = 1` in
`configs/tracking_kitti.yaml`, by default in `eval_tracking_kitti.py` and `run_pipeline.py`. Gives
better tails on mature tracks and `static_frac` 1.00 on all sequences, at the
cost of 0–10 m RMSE of 1.08 m/s (was 0.87). The F5 DoD remains an accepted
limitation, attributed to F4.
`configs/tracking.yaml` keeps q = 1 for the synthetic DoD: there objects move at constant
velocity, and with q = 4 the Δt-jitter scenario rises from 0.40 to 0.61 m/s (limit 0.5).

**Range-bias state (optional, `filter.range_bias`, off by default).** 5-state KF
`[X, Z, V_X, V_Z, b]` with measurement `z = (1+b)·(p − V·lag)` (EKF) and Gauss-Markov
`b` (`sigma`, `tau_s`). Simulation (`TrackingScenario.range_bias_sigma`, 2 % white
error + 5 % drift with τ = 2 s, declared σ = total as in F4, 4 seeds):

| | RMSE V_Z | mean NEES (ideal 4) |
|---|---|---|
| q 1 (base) | 0.85 m/s | 19.7 |
| q 1 + bias (0.05, 2 s) | 0.79 m/s | 3.2 |
| q 4 / q 4 + bias | 1.02 / 0.95 m/s | 15.1 / 3.0 |

Recovers filter consistency but improves RMSE little (≈ 7 %): with a single
constant-velocity track a constant scale bias is not observable (only
`(1+b)(V − v_ego)` is fixed), so the state only separates the part of the drift that decorrelates
in τ. If F4 error is common to all objects in a frame (affine-fit
scale), a **shared** per-frame bias would be observable from static objects; the
JSON `range_error` diagnosis (`frame_share`, lag autocorrelation) decides
between both paths. KITTI sweep: `--range-bias SIGMA TAU_S`.

---

## F6 · Kinematics and safety — ✔ implemented (CPU)

**Scope**
- `safety/kinematics.py`: `RelativeKinematics` (ego-front frame;
  `kinematics_from_track` shifts `ego.front_m` from the camera origin),
  $t_{CPA}, d_{CPA}$ with $\sigma_{d}$ via Jacobian ($r ⟂ v$ at CPA ⇒
  $\partial d/\partial p = \hat r$, $\partial d/\partial v = t_{CPA}\hat r$,
  with $p$–$v$ cross block) and $TTC_{low} = \max(Z - L/2 - k\sigma_Z, 0) /
  (-\dot Z + k\sigma_{\dot Z})$ (∞ if not approaching).
- `safety/gates.py`: `evaluate_gates` → `enter`/`hold` (entry /
  exit thresholds ×1.3 time, ×1.2 distance). **Proximity** gate without
  velocity (in corridor: $|X| - k\sigma_X <$ half-corridor, gap $<$ threshold) and
  **path** gate: $TTC_{low}$ only counts *on path* (in corridor
  now, or approaching with $d_{CPA} + k_{path}\sigma_d <$ half-corridor); off
  path the object is "passing" (`CAUTION` at most if $d_{CPA}$ falls in
  corridor + `passing_margin_m`). Confirmation: `min_updates`, `max_coast_s`,
  `max_pos_sigma_m`.
- `safety/ttc.py`: pure `step(state, kinematics, cfg, t_ns) → (state, alerts)`
  (frozen dataclasses, no wall clock): escalate after consecutive `enter_frames`
  (`CRITICAL` 1 frame in corridor; `path_frames` = 3 if only via
  CPA), an unconfirmed track never escalates; de-escalate to `hold` level only after
  `min_dwell_s` and with `hold < level` sustained for `exit_dwell_s`; lost track:
  −1 level every `lost_decay_s`, never escalates, forget at `forget_s`.
  `Alert(track_id, level, ttc_low_s, d_cpa_m, reason)`.
- `configs/safety.yaml`; `eval/safety_synthetic.py` + `scripts/eval_safety_synthetic.py`.

**Design finding: the lower bound on $d_{CPA}$ produces false alarms.**
$\sigma_d ≈ t_{CPA}\,\sigma_V$ grows with horizon: with σ_V = 0.5 m/s and
$t_{CPA}$ = 4 s, $d_{CPA} - 1.5\sigma_d$ turns a car passing at 1.5 m
into an on-path target (`WARNING`/`CRITICAL` on 100 % of seeds).
Hence the path gate uses the *upper* bound ($d_{CPA} +
k_{path}\sigma_d$, "on path with confidence"), current corridor presence
does not depend on velocity, and CPA-only targets need
consecutive `path_frames` before `CRITICAL` (a single-frame 2.5σ $V_X$ outlier
triggered immediate entry). Conservative bounds are
kept where they are safe: gap, closing speed and $|X|$.

**Measured (synthetic, 10 seeds, σ_Z = 5 % Z, σ_V = 0.5 m/s):** frontal
`CRITICAL` 1.26–1.36 s before contact ✔ (≥ 1.0); static at 2 m with ego
stopped `CRITICAL` by proximity ✔; overtake at 1.5 m `CAUTION` max ✔;
crossing path that intersects `CRITICAL` ✔; crossing with no intersection
`NONE` ✔; intermittent track (visible 0.7 s of every 1 s) without drops, ≤ 0.6
transitions/s ✔; deterministic (level hash) ✔; `step` P95 ≈ 9 µs/frame.

**DoD**
- Synthetic battery in CI: frontal collision (CRITICAL alert ≥ 1.0 s before),
  static object at 2 m with ego stopped (WARNING/CRITICAL by proximity),
  lateral overtake at 1.5 m (never > CAUTION), crossing that does not intersect
  (never > CAUTION), intermittent track (no flicker: ≤ 1 transition per
  second under nominal noise).
- Determinism: same inputs → same state sequence (hash test).
- Properties (`hypothesis`, optional): level never escalates with an unconfirmed
  track; level never drops before minimum dwell.
  *(Implemented as tests with seeded random inputs, without
  `hypothesis`, to avoid adding dependencies.)*

```bash
uv run python scripts/eval_safety_synthetic.py --seeds 10 --json reports/f6_synth.json
```

---

## F7 · Dual-rate integration and telemetry — ✔ implemented; GPU DoD met

**Scope**
- `runtime/pipeline.py`: single-threaded `Pipeline.process(fs, t_ingress)`. Per
  frame: `boxes.submit(fs)` (reactive detector or GT boxes); if
  `DepthCadence.due(k)` and no depth in flight, enqueue
  `depth.infer_stamped(fs)` (own stream, never waited here; if already in
  flight count `skipped` and bump `n` by 1); then `wait()` on boxes.
  Enqueuing the detector first prevents its kernels from waiting behind a
  depth pass already running (stream priority does not preempt kernels);
  `PipelineConfig.depth_first` / `--depth-first` restores the R3 worst-case
  order. `_poll_depth()` collects the map if its event
  has finished, `BoxHistory` stores boxes from frame `k`, and fusion receives
  the latest map $j \le k$ along with boxes *from frame j* (R6:
  `net_boxes`; `MetricFusionStage` inflates $\sigma_n$ with `net_age_sigma_mps ·
  age`). Tracker (real Δt from `t_capture_ns`) → `kinematics_from_track` →
  `safety.step` → `FrameResult` (`frame_id`, `depth_frame_id`, `depth_lag_frames`,
  `depth_age_ms`, per-stage timings) → `sink.log()`.
- `DepthCadence`: adaptive `n` in `[n_min, n_max]` from network EMA turnaround:
  `n = ceil(margin · turnaround / period)`; decreases one by one
  when the network speeds up, jumps up on spikes.
- Frame delivery: `FramePacer` (paced loop, drops lagged frames)
  and capture-thread variant (`start_pump` → `LatestFrameSlot` →
  `iter_slot`, latest-wins with drop counter). Same `Pipeline` and
  same report for both. `LoopedSource`/`LoopedEgoMotion` repeat a
  sequence with continuous timestamps (fold OXTS onto the original stretch) to
  sustain the 5 min DoD.
- `runtime/cuda.py`: `CudaRuntime` (high/low priority streams, timed/untimed
  events, `PinnedBuffer`, `copy_h2d_async`/`copy_d2h_async`),
  `cuda_check` over the `(err, *values)` convention of `cuda-python`; lazy
  import, used by `TrtEngine` (same behavior, CUDA Graph included).
- `telemetry/rerun_sink.py` (replaces `utils/logger.py`): `AsyncSink`
  (bounded deque, *drop-oldest*, own worker; `log()` never blocks or
  throws, `logged/emitted/dropped/errors` counters), `build_record` from
  `FrameResult` (boxes, tracks with σ and level, alerts, latencies, map
  age, depth subsampled every N frames), backends `Rerun`
  (import inside `__init__`; spawn / connect / save `.rrd`), `Jsonl`,
  `Null`, `Recording` (tests).
- `scripts/run_pipeline.py` + `run_pipeline.sh`: KITTI tracking (GT or
  detector, `--ego zero|oxts`), KITTI raw, video; `--hz --loop --duration
  --max-frames --capture-thread --depth-n INIT MIN MAX --telemetry
  none|null|jsonl|rerun --json`. JSON carries `stats`, `stages`, `telemetry`,
  `vram_peak_over_baseline_mb` and DoD `verdicts` (`null` when not
  measurable, e.g. VRAM without GPU). `stats.by_depth_enqueue` separates `boxes` and
  capture→alert for frames that enqueued depth vs those that did not;
  `stats.series` gives one row per second (frames, drops, maps, P50/P99/max
  of capture→alert and P99 of `boxes`) to locate spikes.

**Measured on CPU (tests + harness with GT boxes, no network):** `frame_id` and
timestamps propagate without mixing frames; depth never blocks the
loop and the map used is always from a frame $j \le k$ with its own boxes;
`n` rises/falls with turnaround and respects `[n_min, n_max]`; the sink drops
old data with a slow or broken backend and `log()` costs µs; clean shutdown with
depth in flight and with the capture thread; both delivery variants
produce the same report. None of this substitutes for the following DoD.

**Measured on the target GPU (KITTI 0001, detector + Depth Anything FP16,
60 Hz, 5 min, first-version depth-first order):** paced loop
P99 12.95 ms, capture thread 14.08 ms, 0 % drops, VRAM ≤ 0.73 GB; depth
(7.3 ms turnaround) fits in the period and cadence stays
at `n = 1` (60 Hz, age 0). Rerun recording to `.rrd`: P99 12.78 ms,
`log()` P95 0.30 ms; with the viewer under WSLg (`llvmpipe`, ~300 % CPU) P99
rises to 18.8 ms from CPU contention, not the sink. Forcing `n = 2`
(`--depth-n 2 2 8`) P99 was 16.96 ms: `boxes` P99 12.3 ms on frames
with depth enqueued before the detector → reason for
detector-first order.

**Re-measured with detector-first (60 s, no telemetry):** by default P99
9.00 ms (P50 5.93), `boxes` P50/P99 3.0/4.1 ms and equal with or without depth
enqueued on the frame (P99 4.12 vs 3.99 ms: contention disappears);
with `--depth-n 2 2 8` P99 9.24 ms, 0 drops. In both cases cadence ends at
`n = 2` (30 Hz, age P95 33 ms) because measured turnaround goes to
18.5 ms: depth finishes after its own frame's poll and is
collected on the next, so the measurement includes one wait period
(`depth.gpu` ≈ 8 ms). To return to 60 Hz one would measure turnaround with
the CUDA event instead of polling. `stats.series` located the two spikes of
the default run: startup (max 52 ms, 2 drops) and an isolated stall
of 230 ms at second 49 (12 drops; P50 and `boxes` normal that
second → host/WSL, unverified). All criteria below are met.

**DoD (on the target GPU, ✔ measured)**
- Reactive loop (capture → alert): P99 ≤ 16.7 ms sustained 5 min.
- Depth: ≥ 25 Hz effective; used-map age P95 ≤ 70 ms.
- Peak VRAM (NVML, full GPU minus baseline) ≤ 3.5 GB.
- Frames dropped in `LatestFrameSlot` < 1 % at 60 Hz input.
- Logging cost P95 ≤ 1 ms; the viewer can disconnect without affecting the
  loop.
- If P99 fails due to CPU → capture-thread variant; if due to GIL →
  multiprocess variant; both measured with the same harness before deciding.

---

## F8 · Hardening — ✔ closed; detector in FP16 (INT8 rejected)

- INT8 on detector with KITTI calibration: accept only if pedestrian Δrecall
  ≥ −2 pt and Δlatency ≤ −25 %.
- Optional `nvidia/cuda` image to regenerate engines reproducibly
  (not for development).
- `gpu` tests runnable locally with `pytest -m gpu` and percentile
  report archived in `data/outputs/bench/<date>.json` (history to
  detect regressions).
- README with WSL2 bring-up guide (driver, `usbipd` if live camera).

**Implemented.**

- `detection/int8_calibration.py`: `list_calibration_images` picks
  deterministically N frames distributed across sequences proportional to their
  length and spaced in time (not the first seconds);
  `CalibrationBatcher` applies the same detector `Letterboxer` and serves
  `uint8 (1,H,W,3)` batches — the graph input after ONNX surgery, so
  calibration sees exactly what the engine will see, including fill
  114 —; `make_entropy_calibrator` wraps them in an
  `IInt8EntropyCalibrator2` (pinned memory + H2D on its own stream) with
  on-disk cache. `scripts/export_trt.py --precision int8 --calib-dir …
  --calib-cache …` builds the engine with `INT8 + FP16` (layers without
  INT8 scale fall back to FP16, not FP32); with the cache already written no
  frames are needed. TensorRT/CUDA are imported only inside the function.
- `eval/engine_compare.py` + `scripts/compare_engines.py`: read two
  `bench_detector.py` JSONs and give `PASS/FAIL` per criterion with the plan's
  point estimates (Δrecall `Pedestrian` ≥ −2 pt; `det.gpu` P95 drop
  ≥ 25 %; `Car` is reported without a gate). Each Δrecall carries its 95 % CI
  (difference of proportions; conservative because both engines see the
  same frames) and `recall_conclusive` warns when the interval crosses the
  threshold: with ~1000 pedestrians the CI of a −0.5 pt Δ is ±2.6 pt, so
  an `ACCEPT` with few frames is seen as weak instead of hiding.
- `utils/bench_history.py` + `scripts/bench_history.py`: `--archive` in
  `bench_detector`, `bench_depth` and `run_pipeline` copies the report to
  `data/outputs/bench/<YYYYMMDD_HHMMSS>_<name>.json` (with `-NN` suffix if
  they collide within the second). `check` flattens the JSON, keeps percentile,
  max, VRAM, drop, recall and Hz keys (with the correct direction
  for each) and flags regression if they worsen by more than 10 %
  **and** more than 0.2 units — the dual threshold avoids false alarms on
  near-zero metrics such as `drop_frac`. NaNs and keys that only
  exist in one of the two runs are ignored.
- `tests/test_gpu_smoke.py` (`gpu` marker, 6 tests): one real inference of
  detector and depth with engines from `configs/models.yaml`
  (shapes, finiteness, boxes inside frame, `det.gpu`/`depth.gpu`
  recorded), overlap on two streams, calibrator batch on device,
  report archive as `gpu_smoke` and the F7 5 s loop on KITTI 0001
  when `KT` is set (frames > 60, maps > 0, drops ≤ 1 %). All
  `skip`, never `fail`, without `tensorrt`/`cuda-python`, without engines or
  without dataset, so the CI CPU battery is unchanged.
- `docker/Dockerfile.trt`: `nvidia/cuda:12.6.3-runtime-ubuntu22.04` +
  uv-managed CPython 3.11 + `runtime` extra; `ENTRYPOINT` on
  `scripts/export_trt.py`. Models, datasets and engines are mounted, not
  copied. 25 new CPU tests (`tests/test_f8_hardening.py`) cover
  frame sampling, the batcher (shapes, fill, independent copies,
  cache), verdicts (including random reports vs the rule),
  history and the two CLI scripts.

**Measured on the target GPU (RTX Ada 8 GB, WSL2).** Entropy calibration
on 500 frames from 0000/0001/0020; benchmark with `--frames 500 --warmup 200`
on 0001/0013/0015/0016/0017/0019 (2017 frames, 6743 moderate pedestrians):

| | FP16 | INT8 | Δ |
|---|---|---|---|
| Recall `Pedestrian` | 79.9 % | 72.7 % | **−7.3 pt** (IC95 [−8.7, −5.9]) → FAIL |
| Recall `Car` | 81.2 % | 78.1 % | −3.1 pt (IC95 [−5.2, −1.0]) |
| Recall `Van` / `Cyclist` | 94.3 % / 29.1 % | 82.1 % / 24.9 % | −12.2 / −4.2 pt |
| `det.gpu` P50 / P95 | 0.93 / 2.64 ms | 0.70 / 1.64 ms | −24 % / **−38 %** → PASS |
| `detector_e2e` P95 | 3.76 ms | 2.88 ms | −0.9 ms |

**Decision: `REJECT`, detector stays in FP16.** The recall drop is
conclusive (the entire 95 % CI lies below −2 pt) and systematic: it falls on
all six sequences and prediction count drops 28 % across all
classes (10 794 → 7 822), consistent with INT8 shifting confidences
below the in-graph NMS `score_threshold: 0.25` (inference from
counts; scores were not measured). The latency gain is
real but irrelevant to the loop: ≈ 0.25 ms at P50 and ≈ 0.9 ms at P95 vs
a capture→alert P99 of 9.0 ms with a 16.7 ms budget. A first
run on 0000/0001/0020 (only 67 pedestrians) gave −9.0 pt with 95 % CI
[−24, +6], inconclusive: deciding needs a set with hundreds of
pedestrians. The FP16-run percentiles came out inflated because the GPU
was idle 42 % of samples (F2 `det.gpu` P95 was
≈ 1.1 ms); P50 is the more stable comparison.

Open option, unscheduled: if hardware gets tight (e.g. Jetson),
retry INT8 with MinMax calibrator, INT8-engine-specific confidence
threshold or detection head locked in FP16, and repeat the same verdict.

---

## Cross-cutting measurement strategy

| What | How | Where it lives |
|---|---|---|
| Per-stage latency (CPU) | `perf_counter_ns`, sliding window ≥ 2 000 samples, P50/P95/P99/max | `utils/profiling.py` |
| Per-engine GPU latency | CUDA events per stream; separate `enqueue` from `execute` | `utils/profiling.py` |
| Contention | Matrix {A}, {B}, {A+B} with the same harness | `scripts/bench_*.py` |
| VRAM | NVML (process and GPU peak), not `torch.cuda` | `utils/vram.py` |
| End to end | `t_alert − t_capture` using the timestamp stamped on the frame | `runtime/pipeline.py` |
| Metric accuracy | AbsRel/RMSE vs. LiDAR by distance bins and by class | `eval/kitti.py` |
| Velocity and tracking | RMSE vs. derived GT; ID switches | `eval/kitti.py` |
| Safety | Alert lead time, false positives/h, transitions/s | `tests/test_safety.py` + synthetic |

Operating rule: **no decision from table §4 of the critical analysis is
taken without its corresponding row measured on the target GPU**; plan values are
acceptance thresholds, not predictions.
