#!/usr/bin/env python3
"""Render the README demo (docs/assets/) from the real F4 → F5 → F6 code on a synthetic scene.

Runs without GPU or dataset: :class:`SyntheticScene` supplies noisy boxes and an affine
relative inverse-depth map (what Depth Anything would output); ``MetricFusionStage``,
``Tracker3D`` and the F6 alert machine run unchanged. Writes ``demo.gif`` (needs
``ffmpeg``), ``demo_frame.png`` and ``latency_budget.svg`` (measured F7 figures).

    uv run python scripts/make_readme_assets.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from percepcion3d.camera.calibration import CameraIntrinsics, ExtrinsicMountConfig  # noqa: E402
from percepcion3d.camera.geometry import PinholeGeometry  # noqa: E402
from percepcion3d.depth.depth_trt import DepthMap  # noqa: E402
from percepcion3d.depth.fusion import MetricFusionStage, load_fusion_config  # noqa: E402
from percepcion3d.safety.gates import AlertLevel, load_safety_config  # noqa: E402
from percepcion3d.safety.kinematics import kinematics_from_track  # noqa: E402
from percepcion3d.safety.ttc import initial_state, step  # noqa: E402
from percepcion3d.sim.synthetic import (  # noqa: E402
    SyntheticFrame,
    SyntheticNoise,
    SyntheticObject,
    SyntheticScene,
    cuboid_corners,
)
from percepcion3d.tracking.ego_motion import ConstantEgoMotion  # noqa: E402
from percepcion3d.tracking.tracker3d import (  # noqa: E402
    Track3D,
    Tracker3D,
    load_tracker_config,
    measurement_from_fusion,
)

INTR = CameraIntrinsics(fx=721.5377, fy=721.5377, cx=609.5593, cy=172.854, width=1242, height=375)
EXTR = ExtrinsicMountConfig(camera_height_m=1.65, pitch_rad=0.0)
EGO_V = 12.0
FPS = 30.0
DEPTH_HW = (280, 924)
SCALE = 0.6
CROP_TOP = 70
"""Rows of empty sky removed from the camera and depth panels (full resolution)."""
BEV_W, BEV_Z = 300, 45.0
LEVEL_BGR = {
    AlertLevel.NONE: (120, 200, 90),
    AlertLevel.CAUTION: (40, 210, 240),
    AlertLevel.WARNING: (30, 140, 255),
    AlertLevel.CRITICAL: (50, 50, 235),
}
CLS_BGR = {"car": (170, 120, 60), "pedestrian": (90, 60, 170)}
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Measured on the target GPU (RTX Ada 8 GB, WSL2), docs/01_plan_fases_mvp.md F7/F8.
LATENCY_MS = [
    ("Detector 2D (TensorRT FP16), P95", 2.64),
    ("Detector extremo a extremo, P95", 3.76),
    ("Fusión métrica F4 (CPU), P95", 3.3),
    ("Captura → alerta, P50", 5.93),
    ("Captura → alerta, P99", 9.00),
]
BUDGET_MS = 16.7


def demo_scene(geo: PinholeGeometry) -> SyntheticScene:
    """Ego at 12 m/s: a braking lead car it closes on, an overtaking car, an oncoming car
    and a pedestrian crossing from the kerb."""
    objs = [
        SyntheticObject(1, "car", x0_m=0.3, z0_m=34.0, vz_mps=6.5, height_m=1.45),
        SyntheticObject(2, "car", x0_m=3.6, z0_m=7.0, vz_mps=16.0, height_m=1.6),
        SyntheticObject(3, "car", x0_m=-3.6, z0_m=95.0, vz_mps=-13.0, height_m=1.5),
        SyntheticObject(
            4,
            "pedestrian",
            x0_m=9.0,
            z0_m=70.0,
            vx_mps=-1.2,
            vz_mps=0.0,
            width_m=0.6,
            height_m=1.75,
            length_m=0.6,
            silhouette_frac=0.4,
        ),
    ]
    return SyntheticScene(
        geo,
        objs,
        fps=FPS,
        ego_velocity_mps=(0.0, EGO_V),
        noise=SyntheticNoise(box_px=1.0, inv_depth_rel=0.02),
        seed=3,
    )


def _project(geo: PinholeGeometry, pts_g: NDArray[np.float64]) -> NDArray[np.float64] | None:
    pc = pts_g @ geo.r_cg
    if np.any(pc[:, 2] < 0.5):
        return None
    k = geo.intrinsics
    uv = np.stack([k.fx * pc[:, 0] / pc[:, 2] + k.cx, k.fy * pc[:, 1] / pc[:, 2] + k.cy], 1)
    return np.asarray(uv, dtype=np.float64)


def render_camera(
    geo: PinholeGeometry, scene: SyntheticScene, f: SyntheticFrame, t_s: float
) -> NDArray[np.uint8]:
    k = geo.intrinsics
    h = geo.extrinsics.camera_height_m
    img = np.zeros((k.height, k.width, 3), np.uint8)
    hor = int(round(k.cy))
    for r in range(hor):
        a = r / max(hor, 1)
        img[r] = (int(235 - 50 * a), int(205 - 30 * a), int(160 - 20 * a))
    img[hor:] = (95, 100, 100)
    road = _project(
        geo, np.array([[-7.0, h, 2.0], [7.0, h, 2.0], [7.0, h, 150.0], [-7.0, h, 150.0]])
    )
    if road is not None:
        cv2.fillPoly(img, [road.astype(np.int32)], (70, 72, 74))
    off = (EGO_V * t_s) % 6.0
    for x in (-5.4, -1.8, 1.8, 5.4):
        dashed = abs(x) < 3.0
        for z0 in np.arange(2.0 - off, 120.0, 6.0 if dashed else 120.0):
            z1 = z0 + (3.0 if dashed else 120.0)
            q = _project(
                geo,
                np.array(
                    [
                        [x - 0.08, h, max(z0, 2.0)],
                        [x + 0.08, h, max(z0, 2.0)],
                        [x + 0.08, h, z1],
                        [x - 0.08, h, z1],
                    ]
                ),
            )
            if q is not None:
                cv2.fillPoly(img, [q.astype(np.int32)], (230, 230, 230), cv2.LINE_AA)
    objs = {o.track_id: o for o in scene.objects}
    for i in np.argsort(-f.gt_xz_ground[:, 1]):
        o = objs[int(f.track_ids[i])]
        x, z = f.gt_xz_ground[i]
        c = cuboid_corners(float(x), float(z), h, o.width_m, o.height_m, o.length_m)
        uv = _project(geo, c)
        if uv is None:
            continue
        base = np.array(CLS_BGR[o.cls], np.float64)
        # corner index = 4·xi + 2·yi + zi (x: left/right, y: bottom/top, z: near/far)
        faces = [
            ((0, 2, 6, 4), 1.0),
            ((1, 3, 7, 5), 0.7),
            ((0, 1, 3, 2), 0.8),
            ((4, 5, 7, 6), 0.8),
            ((2, 3, 7, 6), 1.2),
        ]
        for idx, shade in sorted(faces, key=lambda fs: -float(np.mean(c[list(fs[0]), 2]))):
            col = tuple(int(v) for v in np.clip(base * shade, 0, 255))
            cv2.fillPoly(img, [uv[list(idx)].astype(np.int32)], col, cv2.LINE_AA)
    return img


def draw_tracks(
    img: NDArray[np.uint8], tracks: list[Track3D], levels: dict[int, AlertLevel]
) -> None:
    for tr in tracks:
        if tr.time_since_update_s > 0.0:
            continue
        lv = levels.get(tr.track_id, AlertLevel.NONE)
        col = LEVEL_BGR[lv]
        x1, y1, x2, y2 = (int(v) for v in tr.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 3, cv2.LINE_AA)
        txt = f"#{tr.track_id} {tr.position_xz[1]:.0f} m"
        (tw, th), _ = cv2.getTextSize(txt, FONT, 0.6, 2)
        y0 = max(y1 - 8, th + 6)
        cv2.rectangle(img, (x1, y0 - th - 6), (x1 + tw + 8, y0 + 4), col, -1)
        cv2.putText(img, txt, (x1 + 4, y0), FONT, 0.6, (20, 20, 20), 2, cv2.LINE_AA)


def render_depth(inv: NDArray[np.float64], w: int, h: int) -> NDArray[np.uint8]:
    lo, hi = np.nanpercentile(inv, [2, 99.5])
    u8 = np.clip((inv - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
    col = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    return np.asarray(cv2.resize(col, (w, h), interpolation=cv2.INTER_AREA), dtype=np.uint8)


def render_bev(
    h_px: int, tracks: list[Track3D], levels: dict[int, AlertLevel], f: SyntheticFrame
) -> NDArray[np.uint8]:
    img = np.full((h_px, BEV_W, 3), 28, np.uint8)
    s = (h_px - 40) / BEV_Z

    def px(x: float, z: float) -> tuple[int, int]:
        return int(BEV_W / 2 + x * s), int(h_px - 20 - z * s)

    for x in (-5.4, -1.8, 1.8, 5.4):
        cv2.line(img, px(x, 0), px(x, BEV_Z), (80, 80, 80), 1, cv2.LINE_AA)
    for z in (10.0, 20.0, 30.0, 40.0):
        cv2.putText(
            img, f"{z:.0f} m", (4, px(0, z)[1] + 4), FONT, 0.4, (130, 130, 130), 1, cv2.LINE_AA
        )
        cv2.line(img, (40, px(0, z)[1]), (BEV_W, px(0, z)[1]), (45, 45, 45), 1)
    cv2.rectangle(img, px(-0.9, -1.5), px(0.9, 1.5), (230, 230, 230), -1)
    for p in f.gt_xz_ground:
        cv2.drawMarker(img, px(float(p[0]), float(p[1])), (200, 200, 200), cv2.MARKER_CROSS, 8, 1)
    for tr in tracks:
        if tr.time_since_update_s > 0.2:
            continue
        col = LEVEL_BGR[levels.get(tr.track_id, AlertLevel.NONE)]
        x, z = (float(v) for v in tr.position_xz)
        c = tr.cov[:2, :2]
        ev, evec = np.linalg.eigh(c)
        ax = tuple(int(np.clip(2.0 * np.sqrt(max(e, 0.0)) * s, 2.0, 40.0)) for e in ev)
        ang = float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1])))
        cv2.ellipse(img, px(x, z), (ax[1], ax[0]), -ang + 90, 0, 360, col, 1, cv2.LINE_AA)
        cv2.circle(img, px(x, z), 4, col, -1, cv2.LINE_AA)
        v = tr.velocity_rel_xz
        cv2.arrowedLine(
            img,
            px(x, z),
            px(x + 0.5 * float(v[0]), z + 0.5 * float(v[1])),
            col,
            2,
            cv2.LINE_AA,
            tipLength=0.25,
        )
        cv2.putText(
            img,
            f"#{tr.track_id}",
            (px(x, z)[0] + 6, px(x, z)[1] - 6),
            FONT,
            0.45,
            col,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(img, "vista cenital (ego)", (8, 18), FONT, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return img


def render_banner(
    w: int,
    levels: dict[int, AlertLevel],
    alerts: list[tuple[int, AlertLevel, float, str]],
    t_s: float,
) -> NDArray[np.uint8]:
    img = np.full((44, w, 3), 18, np.uint8)
    top = max(levels.values(), default=AlertLevel.NONE)
    cv2.rectangle(img, (0, 0), (170, 44), LEVEL_BGR[top], -1)
    cv2.putText(img, top.name, (10, 30), FONT, 0.8, (15, 15, 15), 2, cv2.LINE_AA)
    parts = [
        f"#{tid} {lv.name.lower()} TTC {ttc:.1f}s ({why})"
        if np.isfinite(ttc)
        else f"#{tid} {lv.name.lower()} ({why})"
        for tid, lv, ttc, why in alerts
    ]
    cv2.putText(
        img,
        "  ".join(parts) or "sin alertas",
        (185, 29),
        FONT,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img, f"t = {t_s:4.1f} s", (w - 110, 29), FONT, 0.55, (180, 180, 180), 1, cv2.LINE_AA
    )
    return img


def label(img: NDArray[np.uint8], text: str) -> None:
    cv2.rectangle(img, (0, 0), (len(text) * 9 + 14, 22), (0, 0, 0), -1)
    cv2.putText(img, text, (7, 16), FONT, 0.5, (240, 240, 240), 1, cv2.LINE_AA)


def run(n_frames: int) -> list[NDArray[np.uint8]]:
    geo = PinholeGeometry(INTR, EXTR)
    scene = demo_scene(geo)
    stage = MetricFusionStage(INTR, geo, load_fusion_config(ROOT / "configs" / "fusion.yaml"))
    trk = Tracker3D(
        load_tracker_config(ROOT / "configs" / "tracking.yaml"), ConstantEgoMotion(EGO_V)
    )
    scfg = load_safety_config(ROOT / "configs" / "safety.yaml")
    sstate = initial_state()
    w, h = int(INTR.width * SCALE), int((INTR.height - CROP_TOP) * SCALE)
    out = []
    for k in range(n_frames):
        f = scene.frame(k)
        t_s = f.t_ns * 1e-9
        inv, rs = scene.dense_inv_depth_map(f, dst_hw=DEPTH_HW, pixel_noise_rel=0.02)
        dm = DepthMap(k, f.t_ns, inv.astype(np.float32), "relative_disparity", rs)
        fo = stage.process(k, f.t_ns, f.boxes, f.classes, dm)
        meas = [measurement_from_fusion(m) for m in fo.measurements]
        tracks = trk.step(f.t_ns, f.boxes, np.full(len(f), 0.9), f.classes, meas)
        sstate, al = step(
            sstate, [kinematics_from_track(t, scfg.ego_front_m) for t in tracks], scfg, f.t_ns
        )
        levels = {a.track_id: a.level for a in al}
        cam = render_camera(geo, scene, f, t_s)
        draw_tracks(cam, tracks, levels)
        cam = np.asarray(
            cv2.resize(cam[CROP_TOP:], (w, h), interpolation=cv2.INTER_AREA), dtype=np.uint8
        )
        label(cam, "camara + detector + tracking 3D")
        dep = render_depth(inv[int(CROP_TOP * rs.scale_y) :], w, h)
        label(dep, "profundidad relativa (1/Z afin)")
        left = np.vstack([cam, dep])
        bev = render_bev(left.shape[0], tracks, levels, f)
        body = np.hstack([left, bev])
        banner = render_banner(
            body.shape[1], levels, [(a.track_id, a.level, a.ttc_low_s, a.reason) for a in al], t_s
        )
        out.append(np.vstack([banner, body]))
    return out


def latency_svg(path: Path) -> None:
    w, row, x0, bar_w = 760, 34, 290, 420
    hgt = 60 + row * len(LATENCY_MS) + 30
    sx = bar_w / 20.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{hgt}" '
        f'font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="13">',
        f'<rect width="{w}" height="{hgt}" rx="10" fill="#0d1117"/>',
        '<text x="20" y="30" fill="#e6edf3" font-size="15" font-weight="600">'
        "Latencia medida en RTX Ada 8 GB (WSL2, KITTI 0001) · presupuesto 60 Hz</text>",
    ]
    for i, (name, ms) in enumerate(LATENCY_MS):
        y = 55 + i * row
        col = "#3fb950" if ms < BUDGET_MS else "#f85149"
        parts += [
            f'<text x="20" y="{y + 16}" fill="#c9d1d9">{name}</text>',
            f'<rect x="{x0}" y="{y + 3}" width="{bar_w}" height="18" rx="4" fill="#161b22"/>',
            f'<rect x="{x0}" y="{y + 3}" width="{ms * sx:.1f}" height="18" rx="4" fill="{col}"/>',
            f'<text x="{x0 + ms * sx + 6:.1f}" y="{y + 16}" fill="#e6edf3">{ms:.2f} ms</text>',
        ]
    bx = x0 + BUDGET_MS * sx
    parts += [
        f'<line x1="{bx:.1f}" y1="48" x2="{bx:.1f}" y2="{hgt - 22}" stroke="#d29922" '
        'stroke-width="2" stroke-dasharray="5 4"/>',
        f'<text x="{bx - 60:.1f}" y="{hgt - 8}" fill="#d29922">16.7 ms (60 Hz)</text>',
        "</svg>",
    ]
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_gif(frames: list[NDArray[np.uint8]], path: Path, fps: int, width: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found (needed for the GIF)")
    with tempfile.TemporaryDirectory() as tmp:
        for i, fr in enumerate(frames):
            cv2.imwrite(str(Path(tmp) / f"{i:04d}.png"), fr)
        vf = (
            f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
            "[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-i",
                str(Path(tmp) / "%04d.png"),
                "-vf",
                vf,
                "-loop",
                "0",
                str(path),
            ],
            check=True,
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "assets")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--gif-fps", type=int, default=15)
    ap.add_argument("--gif-width", type=int, default=900)
    ap.add_argument("--still-s", type=float, default=4.3, help="time of demo_frame.png")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames = run(int(args.seconds * FPS))
    step_n = max(1, int(round(FPS / args.gif_fps)))
    write_gif(frames[::step_n], args.out / "demo.gif", args.gif_fps, args.gif_width)
    cv2.imwrite(
        str(args.out / "demo_frame.png"), frames[min(int(args.still_s * FPS), len(frames) - 1)]
    )
    latency_svg(args.out / "latency_budget.svg")
    for p in ("demo.gif", "demo_frame.png", "latency_budget.svg"):
        print(f"wrote {args.out / p} ({(args.out / p).stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
