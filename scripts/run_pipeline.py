#!/usr/bin/env python3
"""F7: run the dual-rate loop on KITTI (tracking / raw) or a video and measure its DoD.

Examples::

    # KITTI tracking 0001 with GT boxes, no GPU (CPU harness for latency/cadence)
    python scripts/run_pipeline.py --kitti-tracking "$KT" --seq 0001 --boxes gt --hz 60 \\
        --telemetry jsonl --json reports/f7_cpu.json

    # Real engines, 5 min looped, Rerun viewer spawned
    python scripts/run_pipeline.py --kitti-tracking "$KT" --seq 0001 --boxes detector \\
        --det-engine models/detector_1024x320_fp16.engine \\
        --depth-engine models/depth_924x280_fp16.engine \\
        --hz 60 --loop --duration 300 --telemetry rerun --rerun-spawn --json reports/f7.json

Delivery: ``--capture-thread`` moves the source into a thread feeding a
``LatestFrameSlot`` (variant to measure when the P99 fails on CPU time);
otherwise the loop thread paces the replay itself with latest-wins skipping.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from percepcion3d.camera.calibration import (  # noqa: E402
    CameraIntrinsics,
    load_camera_config_yaml,
)
from percepcion3d.camera.geometry import PinholeGeometry  # noqa: E402
from percepcion3d.depth.config import load_depth_config  # noqa: E402
from percepcion3d.depth.depth_trt import DepthEstimator  # noqa: E402
from percepcion3d.depth.fusion import (  # noqa: E402
    MetricFusionStage,
    load_fusion_config,
    load_solver_configs,
)
from percepcion3d.detection.config import load_detector_config  # noqa: E402
from percepcion3d.detection.detector_trt import Detector  # noqa: E402
from percepcion3d.eval.fusion_kitti import KittiTrackingSequence, labels_to_boxes  # noqa: E402
from percepcion3d.io.sources import (  # noqa: E402
    FrameSource,
    KittiSequenceSource,
    OxtsRecord,
    VideoFileSource,
)
from percepcion3d.runtime.buffer import FrameStamped, LatestFrameSlot  # noqa: E402
from percepcion3d.runtime.pipeline import (  # noqa: E402
    Boxes2D,
    BoxStage,
    CallableBoxStage,
    DepthCadenceConfig,
    DepthStage,
    DetectorBoxStage,
    FramePacer,
    IngressFrame,
    LoopedEgoMotion,
    LoopedSource,
    Pipeline,
    PipelineConfig,
    PreloadedSource,
    iter_slot,
    start_pump,
)
from percepcion3d.runtime.trt_engine import TrtEngine  # noqa: E402
from percepcion3d.safety.gates import load_safety_config  # noqa: E402
from percepcion3d.telemetry.rerun_sink import (  # noqa: E402
    AsyncSink,
    JsonlBackend,
    NullBackend,
    RerunBackend,
    SinkReport,
    TelemetryBackend,
)
from percepcion3d.tracking.ego_motion import (  # noqa: E402
    EgoMotionProvider,
    OxtsEgoMotion,
    ZeroEgoMotion,
    load_oxts_file,
)
from percepcion3d.tracking.tracker3d import Tracker3D, load_tracker_config  # noqa: E402
from percepcion3d.utils.profiling import StageTimer  # noqa: E402
from percepcion3d.utils.vram import VramPeak, VramSampler, nvml_sample_fn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

DOD = {
    "e2e_alert_p99_ms": 16.7,
    "depth_hz_min": 25.0,
    "depth_age_p95_ms": 70.0,
    "vram_peak_mb": 3500.0,
    "drop_frac_max": 0.01,
    "telemetry_p95_ms": 1.0,
}


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_argument_group("source")
    src.add_argument("--kitti-tracking", type=Path, help="KITTI tracking training root")
    src.add_argument("--seq", default="0001", help="Tracking sequence id (with --kitti-tracking)")
    src.add_argument("--kitti-raw", type=Path, help="KITTI raw *_drive_XXXX_sync folder")
    src.add_argument("--video", type=Path, help="Video file")
    src.add_argument("--frames", type=int, default=None, help="Max source frames to load")
    src.add_argument("--hz", type=float, default=60.0, help="Replay rate (0 = as fast as possible)")
    src.add_argument("--loop", action="store_true", help="Repeat the source until --duration")
    src.add_argument("--duration", type=float, default=None, help="Stop after this many seconds")
    src.add_argument("--max-frames", type=int, default=None, help="Stop after delivering N frames")
    src.add_argument("--no-preload", action="store_true", help="Decode images while replaying")
    src.add_argument(
        "--capture-thread", action="store_true", help="Capture thread + LatestFrameSlot"
    )

    mdl = ap.add_argument_group("models")
    mdl.add_argument("--camera", type=Path, default=ROOT / "configs/camera_kitti.yaml")
    mdl.add_argument("--fusion", type=Path, default=ROOT / "configs/fusion.yaml")
    mdl.add_argument("--tracking", type=Path, default=ROOT / "configs/tracking.yaml")
    mdl.add_argument("--safety", type=Path, default=ROOT / "configs/safety.yaml")
    mdl.add_argument("--models", type=Path, default=ROOT / "configs/models.yaml")
    mdl.add_argument("--boxes", choices=("gt", "detector", "none"), default=None)
    mdl.add_argument("--det-engine", type=Path, default=None)
    mdl.add_argument("--depth-engine", type=Path, default=None, help="Omit for no depth network")
    mdl.add_argument("--no-cuda-graph", action="store_true")
    mdl.add_argument("--ego", choices=("zero", "oxts"), default="zero")
    mdl.add_argument(
        "--depth-n", type=int, nargs=3, metavar=("INIT", "MIN", "MAX"), default=(2, 1, 8)
    )
    mdl.add_argument("--depth-margin", type=float, default=1.1)
    mdl.add_argument(
        "--depth-first",
        action="store_true",
        help="Enqueue depth before the detector (R3 worst case); default: detector first",
    )

    tel = ap.add_argument_group("telemetry")
    tel.add_argument("--telemetry", choices=("none", "null", "jsonl", "rerun"), default="none")
    tel.add_argument("--jsonl", type=Path, default=Path("reports/f7_telemetry.jsonl"))
    tel.add_argument("--rerun-spawn", action="store_true")
    tel.add_argument("--rerun-connect", default=None, help="Viewer address (e.g. 127.0.0.1:9876)")
    tel.add_argument("--rerun-save", type=Path, default=None, help="Write an .rrd instead")
    tel.add_argument("--tel-depth-every", type=int, default=10, help="Log 1 of N depth maps")
    tel.add_argument("--tel-depth-stride", type=int, default=4)
    tel.add_argument("--tel-queue", type=int, default=64)

    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    n_src = sum(x is not None for x in (args.kitti_tracking, args.kitti_raw, args.video))
    if n_src != 1:
        ap.error("choose exactly one of --kitti-tracking, --kitti-raw, --video")
    if args.boxes is None:
        args.boxes = (
            "detector" if args.det_engine is not None else ("gt" if args.kitti_tracking else "none")
        )
    if args.boxes == "gt" and args.kitti_tracking is None:
        ap.error("--boxes gt needs --kitti-tracking")
    if args.boxes == "detector" and args.det_engine is None:
        ap.error("--boxes detector needs --det-engine")
    return args


def _oxts_ego(
    t_ns: Sequence[int], oxts: Sequence[OxtsRecord], loop_period_ns: int | None
) -> EgoMotionProvider:
    n = min(len(t_ns), len(oxts))
    ego: EgoMotionProvider = OxtsEgoMotion(t_ns[:n], oxts[:n])
    if loop_period_ns is not None:
        ego = LoopedEgoMotion(ego, int(t_ns[0]), int(t_ns[n - 1]), loop_period_ns)
    return ego


def _open_source(
    args: argparse.Namespace, loop_period_ns: int | None
) -> tuple[FrameSource, CameraIntrinsics | None, BoxStage | None, EgoMotionProvider]:
    """Source, its intrinsics (if the dataset provides them), GT box stage and ego-motion."""
    ego: EgoMotionProvider = ZeroEgoMotion()
    if args.kitti_tracking is not None:
        seq = KittiTrackingSequence(args.kitti_tracking, args.seq, max_frames=args.frames)
        src = seq.source()
        intr: CameraIntrinsics | None = seq.intrinsics()
        boxes: BoxStage | None = None
        if args.boxes == "gt":
            labels = seq.labels()
            n_src = len(src.timestamps_ns)

            def gt_boxes(fs: FrameStamped) -> Boxes2D:
                b, cls = labels_to_boxes(labels.get(fs.frame_id % n_src, []))
                return Boxes2D(b, np.ones(b.shape[0]), cls)

            boxes = CallableBoxStage(gt_boxes)
        if args.ego == "oxts":
            oxts_file = args.kitti_tracking / "oxts" / f"{args.seq}.txt"
            if not oxts_file.is_file():
                raise SystemExit(f"{oxts_file} not found (download data_tracking_oxts.zip)")
            ego = _oxts_ego(src.timestamps_ns, load_oxts_file(oxts_file), loop_period_ns)
        return src, intr, boxes, ego
    if args.kitti_raw is not None:
        raw = KittiSequenceSource(args.kitti_raw, max_frames=args.frames)
        if args.ego == "oxts":
            if raw.oxts is None:
                raise SystemExit("--ego oxts: the drive has no oxts/data folder")
            ego = _oxts_ego(raw.timestamps_ns, raw.oxts, loop_period_ns)
        return raw, raw.intrinsics, None, ego
    intr_yaml, _ = load_camera_config_yaml(args.camera)
    return VideoFileSource(args.video, intrinsics=intr_yaml), intr_yaml, None, ego


def _backend(args: argparse.Namespace, camera_hw: tuple[int, int]) -> TelemetryBackend | None:
    if args.telemetry == "none":
        return None
    if args.telemetry == "null":
        return NullBackend()
    if args.telemetry == "jsonl":
        return JsonlBackend(args.jsonl)
    return RerunBackend(
        spawn=args.rerun_spawn,
        connect=args.rerun_connect,
        save=args.rerun_save,
        camera_hw=camera_hw,
    )


def _verdicts(
    stats: dict[str, Any], vram_mb: float | None, has_depth: bool
) -> dict[str, bool | None]:
    v: dict[str, bool | None] = {
        "e2e_alert_p99": stats["e2e_alert_ms"]["p99"] <= DOD["e2e_alert_p99_ms"],
        "drops": stats["drop_frac"] < DOD["drop_frac_max"],
        "telemetry_p95": (
            stats["telemetry_ms"]["p95"] <= DOD["telemetry_p95_ms"]
            if stats["telemetry_ms"]["n"]
            else None
        ),
        "depth_hz": stats["depth"]["hz"] >= DOD["depth_hz_min"] if has_depth else None,
        "depth_age_p95": (
            stats["depth"]["age_ms"]["p95"] <= DOD["depth_age_p95_ms"] if has_depth else None
        ),
        "vram_peak": vram_mb <= DOD["vram_peak_mb"] if vram_mb is not None else None,
    }
    return v


def main() -> int:
    args = _args()
    period_ns = int(round(1e9 / args.hz)) if args.hz > 0 else 100_000_000
    source, intr, boxes, ego = _open_source(args, period_ns if args.loop else None)
    frames_src: Iterable[FrameStamped] = source
    if not args.no_preload:
        pre = PreloadedSource(source, max_frames=args.frames)
        print(f"preloaded {len(pre)} frames")
        frames_src = pre
    if args.loop:
        frames_src = LoopedSource(frames_src, repeats=1_000_000, period_ns=period_ns)
    it = iter(frames_src)
    first = next(it)
    if args.no_preload:
        frames_src = itertools.chain([first], it)
    hw = (int(first.img.shape[0]), int(first.img.shape[1]))
    if intr is None:
        intr, _ = load_camera_config_yaml(args.camera)
    _, extr = load_camera_config_yaml(args.camera)
    fusion_cfg = load_fusion_config(args.fusion)
    road_cfg, aff_cfg, pit_cfg = load_solver_configs(args.fusion)
    trk_cfg = load_tracker_config(args.tracking)
    safety_cfg = load_safety_config(args.safety)
    timer = StageTimer(capacity=1 << 16)
    geo = PinholeGeometry(intr, extr)
    stage = MetricFusionStage(intr, geo, fusion_cfg, road_cfg, aff_cfg, pit_cfg, timer=timer)
    tracker = Tracker3D(trk_cfg, ego)

    need_gpu = args.det_engine is not None or args.depth_engine is not None
    vram_fn = nvml_sample_fn() if need_gpu else None
    vram = VramSampler(interval_s=0.05, sample_fn=vram_fn) if vram_fn is not None else None
    base: VramPeak | None = vram.sample_once() if vram is not None else None
    if vram is not None:
        vram.start()

    det: Detector | None = None
    depth_est: DepthEstimator | None = None
    depth: DepthStage | None = None
    try:
        if args.boxes == "detector":
            det_cfg = load_detector_config(args.models)
            det = Detector(
                TrtEngine(
                    args.det_engine or det_cfg.engine,
                    use_cuda_graph=not args.no_cuda_graph,
                    high_priority_stream=True,
                ),
                det_cfg.runtime_config(),
                timer=timer,
            )
            boxes = DetectorBoxStage(det)
        elif boxes is None:
            boxes = CallableBoxStage(lambda fs: Boxes2D.empty())
        if args.depth_engine is not None:
            depth_cfg = load_depth_config(args.models)
            depth_est = DepthEstimator(
                TrtEngine(
                    args.depth_engine,
                    use_cuda_graph=not args.no_cuda_graph,
                    high_priority_stream=False,
                ),
                depth_cfg.runtime_config(),
                timer=timer,
            )
            depth = depth_est

        backend = _backend(args, hw)
        sink = (
            AsyncSink(
                backend,
                maxlen=args.tel_queue,
                depth_every=args.tel_depth_every,
                depth_stride=args.tel_depth_stride,
            )
            if backend is not None
            else None
        )
        n_init, n_min, n_max = args.depth_n
        cfg = PipelineConfig(
            frame_hz=args.hz if args.hz > 0 else 60.0,
            cadence=DepthCadenceConfig(
                n_min=n_min, n_max=n_max, n_init=n_init, margin=args.depth_margin
            ),
            ego_front_m=safety_cfg.ego_front_m,
            depth_first=args.depth_first,
        )
        # Warm up the engines (CUDA graphs, allocations) outside the measured loop.
        for _ in range(3):
            if det is not None:
                det.infer(first.img)
            if depth_est is not None:
                depth_est.infer_stamped(first).wait()
        timer.reset()
        pipe = Pipeline(boxes, stage, tracker, safety_cfg, depth, sink, cfg, timer)

        hz = args.hz if args.hz > 0 else None
        if args.capture_thread:
            slot: LatestFrameSlot[IngressFrame] = LatestFrameSlot()
            stop = threading.Event()
            th = start_pump(frames_src, slot, hz, stop, max_frames=args.max_frames)
            stats = pipe.run(
                iter_slot(slot, stop, duration_s=args.duration), dropped=lambda: slot.dropped
            )
            stop.set()
            th.join(timeout=2.0)
            delivery = "capture_thread"
        else:
            pacer = FramePacer(frames_src, hz, max_frames=args.max_frames, duration_s=args.duration)
            stats = pipe.run(pacer, dropped=lambda: pacer.dropped)
            delivery = "paced_loop"
        sink_report = SinkReport.from_sink(sink)
    finally:
        if det is not None:
            det.close()
        if depth_est is not None:
            depth_est.close()
        source.close()
    peak = vram.stop() if vram is not None else None

    st = stats.to_dict()
    vram_mb = None
    if peak is not None and base is not None:
        vram_mb = peak.process_mb if peak.process_mb > 0 else peak.gpu_mb - base.gpu_mb
    verdicts = _verdicts(st, vram_mb, depth is not None)

    print(timer.format_table())
    print(
        f"\n{delivery}: {st['frames']} frames in {st['elapsed_s']:.1f} s "
        f"(input {st['input_hz']:.1f} Hz, processed {st['processed_hz']:.1f} Hz, "
        f"dropped {st['dropped']} = {100 * st['drop_frac']:.2f} %)"
    )
    e = st["e2e_alert_ms"]
    print(f"capture→alert ms: P50 {e['p50']:.2f} P95 {e['p95']:.2f} P99 {e['p99']:.2f}")
    if depth is not None:
        d = st["depth"]
        print(
            f"depth: {d['maps']} maps = {d['hz']:.1f} Hz, n_final {d['n_final']}, skipped "
            f"{d['skipped']}, age P95 {d['age_ms']['p95']:.1f} ms, lag P95 "
            f"{d['lag_frames']['p95']:.1f} frames, turnaround P95 {d['turnaround_ms']['p95']:.1f} ms"
        )
    bd = st["by_depth_enqueue"]
    for key in ("boxes_ms", "e2e_alert_ms"):
        w, wo = bd[key]["depth"], bd[key]["no_depth"]
        print(
            f"{key} with depth enqueued (n={w['n']}): P95 {w['p95']:.2f} P99 {w['p99']:.2f} | "
            f"without (n={wo['n']}): P95 {wo['p95']:.2f} P99 {wo['p99']:.2f}"
        )
    if st["series"]:
        worst = max(st["series"], key=lambda r: r["e2e_p99_ms"])
        print(
            f"worst {worst['t_s']:.0f} s window: e2e P99 {worst['e2e_p99_ms']:.2f} ms, max "
            f"{worst['e2e_max_ms']:.2f} ms, dropped {worst['dropped']:.0f}, "
            f"{worst['frames']:.0f} frames"
        )
    if st["telemetry_ms"]["n"]:
        t = st["telemetry_ms"]
        print(
            f"telemetry [{sink_report.backend}]: cost P95 {t['p95']:.3f} ms, logged "
            f"{sink_report.logged}, emitted {sink_report.emitted}, dropped {sink_report.dropped}, "
            f"errors {sink_report.errors}"
        )
    if vram_mb is not None:
        print(f"VRAM peak over baseline: {vram_mb:.0f} MB")
    print("alerts:", st["alerts_by_level"])
    print(
        "DoD:",
        {k: ("n/a" if v is None else ("PASS" if v else "FAIL")) for k, v in verdicts.items()},
    )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "delivery": delivery,
            "image_hw": list(hw),
            "stats": st,
            "stages": {n: s.__dict__ for n, s in timer.report().items()},
            "telemetry": sink_report.__dict__,
            "vram_peak_over_baseline_mb": vram_mb,
            "dod": DOD,
            "verdicts": verdicts,
        }
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0 if all(v is not False for v in verdicts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
