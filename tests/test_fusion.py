"""F4 — ground solver, robust box sampling and correlated metric fusion (CPU only)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from percepcion3d.camera.calibration import CameraIntrinsics, ExtrinsicMountConfig
from percepcion3d.camera.geometry import PinholeGeometry
from percepcion3d.depth.depth_trt import DepthMap
from percepcion3d.depth.fusion import (
    ClassPrior,
    FusionConfig,
    FusionFlag,
    MetricFuser,
    MetricFusionStage,
    fuse_correlated,
    fuse_correlated_batch,
    load_fusion_config,
    load_solver_configs,
    occlusion_matrix,
)
from percepcion3d.depth.ground_solver import (
    AffineFilterConfig,
    AffineKalman,
    AffineState,
    GroundGrid,
    GroundGridCache,
    PitchEstimator,
    PitchFilterConfig,
    RoadSampleConfig,
    contact_depth,
    contact_depth_batch,
    fit_plane,
    pitch_from_contact,
    road_static_mask,
    robust_affine_fit,
    sample_road,
)
from percepcion3d.depth.preprocess import DepthResize
from percepcion3d.depth.sampling import (
    BoxSample,
    BoxSampleConfig,
    disparity_to_depth,
    disparity_to_depth_batch,
    sample_box,
    sample_values,
)
from percepcion3d.sim.synthetic import SyntheticNoise, SyntheticObject, SyntheticScene

ROOT = Path(__file__).resolve().parents[1]
INTR = CameraIntrinsics(fx=721.5377, fy=721.5377, cx=609.5593, cy=172.854, width=1242, height=375)
EXTR = ExtrinsicMountConfig(camera_height_m=1.65, pitch_rad=float(np.deg2rad(2.5)))
RESIZE = DepthResize(src_h=375, src_w=1242, dst_h=140, dst_w=462)


@pytest.fixture
def geo() -> PinholeGeometry:
    return PinholeGeometry(INTR, EXTR)


def _priors() -> FusionConfig:
    return FusionConfig(
        priors={
            "car": ClassPrior(height_m=1.52, sigma_height_m=0.14, length_m=4.3),
            "pedestrian": ClassPrior(height_m=1.70, sigma_height_m=0.12, length_m=0.5, thin=True),
        }
    )


# ─── Contact depth: scalar/batch parity and analytic derivatives ─────────────


def test_contact_depth_batch_matches_scalar_and_finite_differences(geo: PinholeGeometry) -> None:
    u = np.array([100.0, 609.6, 1100.0, 600.0])
    v = np.array([300.0, 250.0, 374.0, 141.0])  # horizon ≈ row 141.4 → last one is nan
    z, dz_dv, dz_dth = contact_depth_batch(geo, u, v)
    for i in range(3):
        s = contact_depth(geo, float(u[i]), float(v[i]))
        assert s is not None
        assert z[i] == pytest.approx(s.z_m, rel=1e-12)
        assert dz_dv[i] == pytest.approx(s.dz_dv, rel=1e-12)
        assert dz_dth[i] == pytest.approx(s.dz_dpitch, rel=1e-12)
    assert np.isnan(z[3]) and contact_depth(geo, 600.0, 141.0) is None
    eps = 1e-4
    z_p, _, _ = contact_depth_batch(geo, u[:3], v[:3] + eps)
    z_m, _, _ = contact_depth_batch(geo, u[:3], v[:3] - eps)
    np.testing.assert_allclose((z_p - z_m) / (2 * eps), dz_dv[:3], rtol=1e-5)
    geo_p = PinholeGeometry(INTR, replace(EXTR, pitch_rad=EXTR.pitch_rad + eps))
    geo_m = PinholeGeometry(INTR, replace(EXTR, pitch_rad=EXTR.pitch_rad - eps))
    zp, _, _ = contact_depth_batch(geo_p, u[:3], v[:3])
    zm, _, _ = contact_depth_batch(geo_m, u[:3], v[:3])
    np.testing.assert_allclose((zp - zm) / (2 * eps), dz_dth[:3], rtol=1e-5)


def test_contact_depth_matches_legacy_geometry_at_principal_column(geo: PinholeGeometry) -> None:
    z_c, _ = geo.compute_ground_distances(300.0, u=None)
    s = contact_depth(geo, INTR.cx, 300.0)
    assert s is not None and s.z_m == pytest.approx(z_c, rel=1e-9)


# ─── Ground grid + cache ─────────────────────────────────────────────────────


def test_ground_grid_matches_contact_depth_and_cache_quantises_pitch(geo: PinholeGeometry) -> None:
    grid = GroundGrid.build(geo, RESIZE)
    assert grid.inv_z_ground.shape == (RESIZE.dst_h, RESIZE.dst_w)
    r, c = 120, 231
    v = (r + 0.5) / RESIZE.scale_y - 0.5
    u = (c + 0.5) / RESIZE.scale_x - 0.5
    s = contact_depth(geo, u, v)
    assert s is not None
    assert grid.inv_z_ground[r, c] == pytest.approx(1.0 / s.z_m, rel=1e-9)
    assert np.isnan(grid.inv_z_ground[0, c])  # above the horizon
    hz = int(np.ceil(grid.horizon_row))
    assert np.all(np.isnan(grid.inv_z_ground[: hz - 1, c]))
    assert np.all(np.isfinite(grid.inv_z_ground[hz + 2 :, c]))

    cache = GroundGridCache(angle_quantum_rad=float(np.deg2rad(0.1)))
    g0 = cache.get(geo, RESIZE)
    tiny = PinholeGeometry(INTR, replace(EXTR, pitch_rad=EXTR.pitch_rad + np.deg2rad(0.02)))
    assert cache.get(tiny, RESIZE) is g0 and cache.n_builds == 1
    big = PinholeGeometry(INTR, replace(EXTR, pitch_rad=EXTR.pitch_rad + np.deg2rad(0.3)))
    g1 = cache.get(big, RESIZE)
    assert g1 is not g0 and cache.n_builds == 2
    assert cache.rays(INTR, RESIZE) is cache.rays(INTR, RESIZE)


# ─── Road sampling + robust affine fit ───────────────────────────────────────


def _road_map(
    grid: GroundGrid, s: float, t: float, rng: np.random.Generator
) -> NDArray[np.float32]:
    inv = grid.inv_z_ground
    d = s * inv + t
    return np.where(np.isfinite(d), d + rng.normal(0.0, 0.002, d.shape), 0.5).astype(np.float32)


def test_sample_road_avoids_boxes_and_fit_recovers_affine(geo: PinholeGeometry) -> None:
    rng = np.random.default_rng(0)
    grid = GroundGrid.build(geo, RESIZE)
    cfg = RoadSampleConfig(max_points=1500)
    d_map = _road_map(grid, 3.0, 0.2, rng)
    static_idx = np.flatnonzero(road_static_mask(grid, cfg))
    box = np.array([[400.0, 200.0, 800.0, 374.0]])
    # Poison the box interior: a road sample drawn there would be off the affine line.
    r0, c0 = int(200 * RESIZE.scale_y), int(400 * RESIZE.scale_x)
    c1 = int(np.ceil(800 * RESIZE.scale_x)) + 1
    d_map[r0:, c0:c1] = 50.0
    smp = sample_road(d_map, grid, static_idx, box, cfg, rng)
    assert 500 <= smp.d_hat.shape[0] <= cfg.max_points
    assert smp.n_candidates >= smp.d_hat.shape[0]
    assert np.all(smp.d_hat < 10.0)
    assert np.all(1.0 / smp.inv_z >= cfg.z_min_m - 1e-9) and np.all(1.0 / smp.inv_z <= cfg.z_max_m)
    # Without the box the poisoned pixels are sampled.
    smp_nobox = sample_road(d_map, grid, static_idx, None, cfg, rng)
    assert np.any(smp_nobox.d_hat == 50.0)
    fit = robust_affine_fit(smp.inv_z, smp.d_hat, rng)
    assert fit is not None
    assert fit.s == pytest.approx(3.0, rel=0.02)
    assert fit.t == pytest.approx(0.2, abs=0.005)
    assert fit.inlier_ratio > 0.9
    assert fit.cov.shape == (2, 2) and np.all(np.diag(fit.cov) > 0)


def test_robust_affine_fit_survives_outliers_and_rejects_degenerate() -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(0.02, 0.3, 2000)
    y = 2.0 * x + 0.1 + rng.normal(0, 0.001, 2000)
    y[:500] = rng.uniform(0.0, 2.0, 500)  # 25 % gross outliers
    fit = robust_affine_fit(x, y, rng)
    assert fit is not None
    assert fit.s == pytest.approx(2.0, rel=0.02) and fit.t == pytest.approx(0.1, abs=0.005)
    assert robust_affine_fit(np.full(100, 0.1), rng.normal(size=100), rng) is None  # no x spread
    assert robust_affine_fit(x[:5], y[:5], rng) is None  # too few
    assert robust_affine_fit(x, -y, rng) is None  # negative slope: not a disparity


def test_affine_kalman_gates_outlier_and_resets_after_streak() -> None:
    rng = np.random.default_rng(2)
    x = rng.uniform(0.02, 0.3, 1000)
    kf = AffineKalman(AffineFilterConfig(reset_after_rejects=3))
    good = robust_affine_fit(x, 2.0 * x + 0.1 + rng.normal(0, 0.001, 1000), rng)
    assert good is not None
    st = kf.update(good, 0)
    assert st is not None and st.n_updates == 1
    bad = robust_affine_fit(x, 5.0 * x + 0.9 + rng.normal(0, 0.001, 1000), rng)
    assert bad is not None
    st = kf.update(bad, 33_000_000)
    assert st is not None and st.s == pytest.approx(2.0, rel=0.02) and st.reject_streak == 1
    kf.update(bad, 66_000_000)
    st = kf.update(bad, 99_000_000)
    assert st is not None and st.s == pytest.approx(5.0, rel=0.02) and st.reject_streak == 0
    # Predict-only step grows the covariance.
    p_before = st.cov[0, 0]
    st2 = kf.update(None, 1_099_000_000)
    assert st2 is not None and st2.cov[0, 0] > p_before


# ─── Box sampling ────────────────────────────────────────────────────────────


def test_sample_values_unimodal_bimodal_and_thin_selection() -> None:
    rng = np.random.default_rng(3)
    cfg = BoxSampleConfig()
    uni = rng.normal(0.10, 0.001, 400)
    s = sample_values(uni, cfg)
    assert s is not None and not s.bimodal
    assert s.value == pytest.approx(0.10, abs=2e-4)
    assert 0 < s.sigma < 0.0005
    near, far = rng.normal(0.20, 0.002, 120), rng.normal(0.05, 0.002, 280)
    mixed = np.concatenate([near, far])
    s_car = sample_values(mixed, cfg, thin=False)
    assert s_car is not None and s_car.bimodal
    assert s_car.value == pytest.approx(0.05, abs=1e-3)  # majority cluster (the object)
    s_ped = sample_values(mixed, cfg, thin=True)
    assert s_ped is not None and s_ped.bimodal
    assert s_ped.value == pytest.approx(0.20, abs=1e-3)  # thin: the near mode
    assert sample_values(np.full(5, np.nan), cfg) is None
    assert sample_values(np.arange(4, dtype=np.float64), cfg) is None  # < min_pixels


def test_sample_box_indexing_subsampling_and_exclusion() -> None:
    cfg = BoxSampleConfig(shrink=0.0, max_pixels=64, exclude_dilate_px=0.0)
    m = np.zeros((RESIZE.dst_h, RESIZE.dst_w), dtype=np.float32)
    m[40:80, 100:200] = 1.0

    def r2f(r: int) -> float:
        return (r + 0.5) / RESIZE.scale_y - 0.5

    def c2f(c: int) -> float:
        return (c + 0.5) / RESIZE.scale_x - 0.5

    box = np.array([c2f(100), r2f(40), c2f(199), r2f(79)])
    s = sample_box(m, RESIZE, box, cfg)
    assert s is not None and s.value == 1.0 and s.mad == 0.0 and s.n <= 4 * cfg.max_pixels
    s_all = sample_box(m, RESIZE, box, replace(cfg, max_pixels=10**6))
    assert s_all is not None and s_all.n == 40 * 100
    # An excluder covering the top half leaves only the bottom rows.
    m[40:60, 100:200] = 9.0
    ex = np.array([[c2f(100), r2f(40), c2f(199), r2f(59)]])
    s_ex = sample_box(m, RESIZE, box, replace(cfg, max_pixels=10**6), exclude_boxes=ex)
    assert s_ex is not None and s_ex.value == 1.0 and s_ex.n == 20 * 100
    s_dil = sample_box(
        m, RESIZE, box, replace(cfg, max_pixels=10**6, exclude_dilate_px=10.0), exclude_boxes=ex
    )
    assert s_dil is not None and s_dil.n < s_ex.n
    assert sample_box(m, RESIZE, np.array([2000.0, 2000.0, 2100.0, 2100.0]), cfg) is None


def test_disparity_to_depth_propagation_scalar_batch_and_beyond() -> None:
    state = AffineState(
        s=3.0, t=0.2, cov=np.diag([0.01**2, 0.002**2]), n_updates=5, reject_streak=0, t_ns=0
    )
    z_true = np.array([5.0, 20.0, 60.0])
    d = state.s / z_true + state.t
    nd = disparity_to_depth_batch(d, np.full(3, 1e-3), state, net_rel_sigma=0.05)
    np.testing.assert_allclose(nd.z_m, z_true, rtol=1e-12)
    np.testing.assert_allclose(nd.inv_z, 1.0 / z_true, rtol=1e-12)
    # σ_Z = σ_ρ · Z²  and the fit share is the delta method of (s, t).
    np.testing.assert_allclose(nd.sigma_m, nd.sigma_inv * z_true**2, rtol=1e-12)
    jac = np.stack([z_true / state.s, z_true**2 / state.s], axis=1)
    var_fit = np.einsum("ij,jk,ik->i", jac, state.cov, jac)
    np.testing.assert_allclose(nd.sigma_fit_m, np.sqrt(var_fit), rtol=1e-9)
    assert np.all(np.diff(nd.sigma_m / nd.z_m) > 0)  # relative error grows with range
    smp = BoxSample(
        value=float(d[1]), sigma=1e-3, median=float(d[1]), mad=0.0, n=100, n_cluster=100,
        bimodal=False, cluster_lo=0.0, cluster_hi=0.0,
    )  # fmt: skip
    sc = disparity_to_depth(smp, state, 0.05)
    assert (
        sc is not None
        and sc.z_m == pytest.approx(20.0)
        and sc.sigma_m == pytest.approx(nd.sigma_m[1])
    )
    beyond = disparity_to_depth_batch(np.array([state.t - 0.01]), np.array([1e-3]), state, 0.05)
    assert np.isnan(beyond.z_m[0]) and np.isfinite(beyond.inv_z[0]) and beyond.inv_z[0] < 0


# ─── Correlated BLUE ─────────────────────────────────────────────────────────


def test_fuse_correlated_reduces_to_inverse_variance_and_downweights_correlated() -> None:
    z = np.array([0.10, 0.11, 0.09])
    sig = np.array([0.01, 0.02, 0.005])
    z_hat, s_hat, w, chi2 = fuse_correlated(z, sig, np.zeros(3))
    w_iv = (1 / sig**2) / (1 / sig**2).sum()
    np.testing.assert_allclose(w, w_iv, rtol=1e-12)
    assert z_hat == pytest.approx(float(w_iv @ z)) and s_hat == pytest.approx(
        1 / np.sqrt((1 / sig**2).sum())
    )
    assert chi2 >= 0.0
    # Strong shared pitch term on cues 0 and 1 → the pitch-free cue 2 gains weight and the
    # fused σ is above the naive value.
    c = np.array([0.05, 0.05, 0.0])
    z2, s2, w2, _ = fuse_correlated(z, sig, c)
    assert w2[2] > w_iv[2] and s2 > s_hat
    assert w2.sum() == pytest.approx(1.0)


def test_fuse_correlated_batch_matches_scalar_for_every_active_pattern() -> None:
    rng = np.random.default_rng(4)
    n, m = 40, 3
    z = rng.uniform(0.02, 0.5, (n, m))
    sig = rng.uniform(1e-3, 5e-2, (n, m))
    c = rng.uniform(0.0, 3e-2, (n, m))
    c[:, 2] = 0.0
    patterns = [
        [True, True, True],
        [True, True, False],
        [True, False, True],
        [False, True, True],
        [True, False, False],
        [False, True, False],
        [False, False, True],
        [False, False, False],
    ]
    active = np.array([patterns[i % len(patterns)] for i in range(n)])
    z_hat, s_hat, w, chi2 = fuse_correlated_batch(z, sig, c, active)
    for i in range(n):
        idx = np.flatnonzero(active[i])
        if idx.size == 0:
            assert np.isnan(z_hat[i]) and np.isnan(s_hat[i]) and chi2[i] == 0.0
            assert np.all(w[i] == 0.0)
            continue
        zs, ss, ws, cs = fuse_correlated(z[i, idx], sig[i, idx], c[i, idx])
        # Inactive cues carry σ = 1e4 inside the batch, i.e. a ~1e-11 relative weight leak.
        assert z_hat[i] == pytest.approx(zs, rel=1e-8)
        assert s_hat[i] == pytest.approx(ss, rel=1e-8)
        np.testing.assert_allclose(w[i, idx], ws, rtol=1e-7, atol=1e-10)
        assert np.all(w[i, ~active[i]] == 0.0)
        assert chi2[i] == pytest.approx(cs if idx.size > 1 else 0.0, rel=1e-7, abs=1e-10)


def test_occlusion_matrix_marks_only_nearer_overlapping_boxes() -> None:
    boxes = np.array(
        [
            [100.0, 100.0, 300.0, 300.0],  # 0: far (bottom 300)
            [200.0, 150.0, 400.0, 360.0],  # 1: overlaps 0, nearer
            [900.0, 100.0, 1000.0, 350.0],  # 2: disjoint
        ]
    )
    occ = occlusion_matrix(boxes, margin_px=2.0)
    assert occ[0, 1] and not occ[1, 0] and not occ[0, 2] and not occ.diagonal().any()


# ─── Frame-level fuser: gates, flags, pitch sensitivity ──────────────────────


def _fuser_inputs(
    geo: PinholeGeometry, z: float, cls: str = "car", h_obj: float = 1.52
) -> tuple[NDArray[np.float64], list[str]]:
    """Box of an object of height ``h_obj`` standing at optical depth ``z`` under the camera column."""
    k = geo.intrinsics
    ext = geo.extrinsics
    y = (ext.camera_height_m / z - np.sin(ext.pitch_rad)) / np.cos(ext.pitch_rad)
    hit_v = float(k.cy + y * k.fy)
    s = contact_depth(geo, k.cx, hit_v)
    assert s is not None and s.z_m == pytest.approx(z, rel=1e-9)
    h_px = k.fy * h_obj / z
    return np.array([[k.cx - 60.0, hit_v - h_px, k.cx + 60.0, hit_v]]), [cls]


def test_fuser_ground_and_height_agree_on_consistent_object(geo: PinholeGeometry) -> None:
    fuser = MetricFuser(_priors())
    boxes, classes = _fuser_inputs(geo, 20.0)
    m = fuser.fuse(boxes, classes, geo, None, None, None, frame_id=0, t_ns=0)[0]
    assert FusionFlag.NET_UNAVAILABLE in m.flags
    assert m.z_ground_m == pytest.approx(20.0, rel=0.01)
    assert m.z_height_m == pytest.approx(20.0, rel=0.01)
    assert np.isnan(m.z_net_m) and m.weights[1] == 0.0
    assert m.z_cam_m == pytest.approx(20.0, rel=0.01)
    assert m.weights[0] + m.weights[2] == pytest.approx(1.0)
    assert FusionFlag.INCONSISTENT not in m.flags
    assert 0 < m.sigma_z_m < min(m.sigma_ground_m, m.sigma_height_m)
    assert m.pitch_meas_rad == pytest.approx(EXTR.pitch_rad, abs=np.deg2rad(0.2))
    assert m.z_fwd_m > 0 and abs(m.x_lat_m) < 0.2 and m.center_offset_m == pytest.approx(2.15)


def test_fuser_flags_truncation_no_prior_and_out_of_range(geo: PinholeGeometry) -> None:
    fuser = MetricFuser(_priors())
    k = geo.intrinsics
    boxes = np.array(
        [
            [500.0, 200.0, 700.0, k.height - 1.0],  # bottom on the border → ground+height off
            [500.0, 250.0, 520.0, 258.0],  # 8 px tall → height too small
            [500.0, 200.0, 700.0, 330.0],  # unknown class
        ]
    )
    ms = fuser.fuse(boxes, ["car", "car", "kangaroo"], geo, None, None, None, 0, 0)
    assert FusionFlag.GROUND_TRUNCATED in ms[0].flags and FusionFlag.HEIGHT_TRUNCATED in ms[0].flags
    assert FusionFlag.NO_ESTIMATE in ms[0].flags and np.isnan(ms[0].z_cam_m)
    assert FusionFlag.HEIGHT_TOO_SMALL in ms[1].flags and np.isnan(ms[1].z_height_m)
    assert np.isfinite(ms[1].z_cam_m)  # ground alone
    assert FusionFlag.NO_PRIOR in ms[2].flags and np.isnan(ms[2].z_height_m)
    assert ms[2].weights[0] == pytest.approx(1.0)
    # Contact row a fraction of a pixel below the horizon → Z_g of kilometres: the cue is
    # dropped as implausible, the height cue alone survives.
    horizon = k.cy - k.fy * np.tan(EXTR.pitch_rad)
    far = np.array([[600.0, horizon - 40.0, 620.0, horizon + 0.3]])
    m_far = fuser.fuse(far, ["car"], geo, None, None, None, 0, 0)[0]
    assert FusionFlag.OUT_OF_RANGE in m_far.flags and np.isnan(m_far.z_ground_m)
    assert np.isfinite(m_far.z_height_m) and m_far.weights[2] == pytest.approx(1.0)


def test_fuser_inconsistent_cues_are_arbitrated(geo: PinholeGeometry) -> None:
    cfg = replace(_priors(), sigma_pitch_rad=1e-6)  # no shared error → plain χ²
    fuser = MetricFuser(cfg)
    boxes, classes = _fuser_inputs(geo, 20.0, h_obj=3.2)  # a "car" twice as tall as the prior
    m = fuser.fuse(boxes, classes, geo, None, None, None, 0, 0)[0]
    assert m.z_ground_m == pytest.approx(20.0, rel=0.01)
    assert m.z_height_m == pytest.approx(9.5, rel=0.05)
    assert FusionFlag.INCONSISTENT in m.flags
    naive = fuse_correlated(
        np.array([1 / m.z_ground_m, 1 / m.z_height_m]),
        np.array([m.sigma_ground_m / m.z_ground_m**2, m.sigma_height_m / m.z_height_m**2]),
        np.zeros(2),
    )
    assert m.sigma_z_m > naive[1] * m.z_cam_m**2 * 1.5  # σ inflated by the disagreement


def test_pitch_error_moves_ground_and_height_apart_and_flags_it(geo: PinholeGeometry) -> None:
    """Road-only cues are blind to a pitch bias; the height prior is the term that sees it."""
    fuser = MetricFuser(_priors())
    boxes, classes = _fuser_inputs(geo, 40.0)
    wrong = PinholeGeometry(INTR, replace(EXTR, pitch_rad=EXTR.pitch_rad + np.deg2rad(1.0)))
    m_ok = fuser.fuse(boxes, classes, geo, None, None, None, 0, 0)[0]
    m_bad = fuser.fuse(boxes, classes, wrong, None, None, None, 0, 0)[0]
    assert m_ok.z_ground_m == pytest.approx(40.0, rel=0.02)
    assert abs(m_bad.z_ground_m - 40.0) / 40.0 > 0.2  # ∂Z/∂θ ≈ Z²/h → 1° at 40 m ≈ 30 %
    assert m_bad.z_height_m == pytest.approx(m_ok.z_height_m)  # pitch-free
    assert abs(m_bad.z_cam_m - 40.0) < abs(m_bad.z_ground_m - 40.0)
    # The per-object pitch measurement recovers the true pitch from the height cue.
    assert m_bad.pitch_meas_rad == pytest.approx(EXTR.pitch_rad, abs=np.deg2rad(0.3))


def test_pitch_from_contact_inverts_geometry_and_estimator_converges(geo: PinholeGeometry) -> None:
    k = geo.intrinsics
    v = np.array([300.0, 260.0, 230.0])
    z, _, _ = contact_depth_batch(geo, np.full(3, k.cx), v)
    th, ok = pitch_from_contact((v - k.cy) / k.fy, z, EXTR.camera_height_m)
    assert ok.all()
    np.testing.assert_allclose(th, EXTR.pitch_rad, atol=1e-9)
    _, bad = pitch_from_contact(np.array([0.0]), np.array([0.5]), 1.65)  # closer than h
    assert not bad[0]

    est = PitchEstimator(
        replace(EXTR, pitch_rad=EXTR.pitch_rad + np.deg2rad(1.0)),
        PitchFilterConfig(sigma0_rad=np.deg2rad(1.0)),
    )
    rng = np.random.default_rng(5)
    for i in range(40):
        meas = EXTR.pitch_rad + rng.normal(0.0, np.deg2rad(0.3), 5)
        est.update(meas, np.full(5, np.deg2rad(0.3)), i * 33_000_000)
    assert est.state.pitch_rad == pytest.approx(EXTR.pitch_rad, abs=np.deg2rad(0.1))
    assert est.state.sigma_rad < np.deg2rad(0.1)
    n_before = est.state.n_updates
    est.update(np.array([EXTR.pitch_rad + np.deg2rad(5.0)]), np.array([1e-4]), 41 * 33_000_000)
    assert est.state.n_updates == n_before  # beyond max_step → discarded


def test_fit_plane_recovers_pitch_and_height(geo: PinholeGeometry) -> None:
    rng = np.random.default_rng(6)
    grid = GroundGrid.build(geo, RESIZE)
    pts = grid.deproject(grid.inv_z_ground).reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    pts = pts[rng.choice(pts.shape[0], 3000, replace=False)]
    pf = fit_plane(pts)
    assert pf is not None
    assert pf.pitch_rad == pytest.approx(EXTR.pitch_rad, abs=1e-6)
    assert pf.roll_rad == pytest.approx(0.0, abs=1e-6)
    assert pf.height_m == pytest.approx(EXTR.camera_height_m, abs=1e-6)
    assert fit_plane(pts[:2]) is None


def test_pitch_is_unobservable_from_road_plus_affine_map(geo: PinholeGeometry) -> None:
    """A wrong assumed pitch leaves the affine fit perfect and the plane fit returns
    the *assumed* pitch: road pixels alone cannot correct it."""
    rng = np.random.default_rng(7)
    true_grid = GroundGrid.build(geo, RESIZE)
    d_hat = 3.0 * true_grid.inv_z_ground + 0.2  # network output on a perfectly flat road
    wrong = PinholeGeometry(INTR, replace(EXTR, pitch_rad=EXTR.pitch_rad + np.deg2rad(1.0)))
    wrong_grid = GroundGrid.build(wrong, RESIZE)
    ok = np.isfinite(d_hat) & np.isfinite(wrong_grid.inv_z_ground)
    ok &= wrong_grid.inv_z_ground > 1.0 / 60.0
    fit = robust_affine_fit(wrong_grid.inv_z_ground[ok], d_hat[ok], rng)
    assert fit is not None
    assert fit.sigma_resid < 1e-6 and fit.inlier_ratio > 0.99  # still exactly affine
    assert fit.t != pytest.approx(0.2, abs=1e-3)  # the offset absorbed the pitch error
    pts = wrong_grid.deproject((d_hat - fit.t) / fit.s).reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    pf = fit_plane(pts[rng.choice(pts.shape[0], 3000, replace=False)])
    assert pf is not None
    assert pf.pitch_rad == pytest.approx(wrong.extrinsics.pitch_rad, abs=np.deg2rad(0.01))


# ─── Config + end-to-end stage on the synthetic scene ────────────────────────


def test_fusion_yaml_loads_with_aliases() -> None:
    cfg = load_fusion_config(ROOT / "configs" / "fusion.yaml")
    assert cfg.prior_for("pedestrian") is cfg.prior_for("person")
    assert cfg.prior_for("van") is cfg.prior_for("car")
    car = cfg.prior_for("car")
    assert car is not None and car.height_m == pytest.approx(1.52)
    assert cfg.prior_for("unicorn") is None
    assert cfg.range_slack == pytest.approx(
        1.5
    ) and cfg.sampling.exclude_dilate_px == pytest.approx(3.0)
    road, aff, pit = load_solver_configs(ROOT / "configs" / "fusion.yaml")
    assert road.max_points == 2000 and aff.min_inlier_ratio > 0 and pit.max_step_rad > 0


def _stage_on_scene(
    geo_true: PinholeGeometry,
    geo_assumed: PinholeGeometry,
    online_pitch: bool,
    n_frames: int = 30,
) -> tuple[list[float], list[float], MetricFusionStage]:
    cfg = load_fusion_config(ROOT / "configs" / "fusion.yaml")
    road, aff, pit = load_solver_configs(ROOT / "configs" / "fusion.yaml")
    objs = [
        SyntheticObject(1, "car", x0_m=0.5, z0_m=12.0, vz_mps=1.0, height_m=1.52),
        SyntheticObject(2, "car", x0_m=-3.5, z0_m=30.0, vz_mps=-2.0, height_m=1.45),
        SyntheticObject(3, "pedestrian", x0_m=4.0, z0_m=18.0, width_m=0.6, height_m=1.75,
                        length_m=0.5, silhouette_frac=0.4),
    ]  # fmt: skip
    scene = SyntheticScene(
        geo_true, objs, noise=SyntheticNoise(box_px=0.5, inv_depth_rel=0.01), fps=30.0
    )
    scene.disparity_scale, scene.disparity_shift = 3.0, 0.2
    stage = MetricFusionStage(INTR, geo_assumed, cfg, road, aff, pit, online_pitch=online_pitch)
    errs_fused: list[float] = []
    errs_ground: list[float] = []
    for k in range(n_frames):
        f = scene.frame(k)
        inv, rs = scene.dense_inv_depth_map(f, (140, 462), pixel_noise_rel=0.01)
        dm = DepthMap(f.frame_id, f.t_ns, inv.astype(np.float32), "relative_disparity", rs)
        out = stage.process(f.frame_id, f.t_ns, f.boxes, f.classes, dm)
        assert out.depth_lag_frames == 0 and out.solver is not None
        if k < 10:
            continue
        assert out.affine is not None and out.affine.s == pytest.approx(3.0, rel=0.05)
        for m, zt in zip(out.measurements, f.gt_z_front_m, strict=True):
            assert m.frame_id == f.frame_id and m.t_capture_ns == f.t_ns
            assert np.isfinite(m.z_cam_m) and m.sigma_z_m > 0
            errs_fused.append(abs(m.z_cam_m - zt) / zt)
            errs_ground.append(abs(m.z_ground_m - zt) / zt)
    return errs_fused, errs_ground, stage


def test_stage_end_to_end_nominal(geo: PinholeGeometry) -> None:
    fused, ground, stage = _stage_on_scene(geo, geo, online_pitch=True)
    assert len(fused) >= 40
    assert float(np.median(fused)) < 0.05
    assert float(np.percentile(fused, 95)) < 0.15
    assert stage.pitch.state.pitch_rad == pytest.approx(EXTR.pitch_rad, abs=np.deg2rad(0.3))
    assert stage.solver.grids.n_builds < 10


def test_stage_online_pitch_recovers_one_degree_bias(geo: PinholeGeometry) -> None:
    wrong = PinholeGeometry(INTR, replace(EXTR, pitch_rad=EXTR.pitch_rad + np.deg2rad(1.0)))
    fused_off, ground_off, st_off = _stage_on_scene(geo, wrong, online_pitch=False)
    fused_on, ground_on, st_on = _stage_on_scene(geo, wrong, online_pitch=True)
    assert st_off.geometry().extrinsics.pitch_rad == pytest.approx(wrong.extrinsics.pitch_rad)
    assert st_on.geometry().extrinsics.pitch_rad == pytest.approx(
        EXTR.pitch_rad, abs=np.deg2rad(0.3)
    )
    assert float(np.median(ground_off)) > 2.0 * float(np.median(ground_on))
    assert float(np.median(fused_on)) < 0.05
    assert float(np.median(fused_on)) < float(np.median(fused_off))


def test_stage_without_depth_map_runs_on_geometry_and_prior(geo: PinholeGeometry) -> None:
    stage = MetricFusionStage(INTR, geo, _priors())
    boxes, classes = _fuser_inputs(geo, 15.0)
    out = stage.process(0, 0, boxes, classes, None)
    assert out.solver is None and out.affine is None
    m = out.measurements[0]
    assert FusionFlag.NET_UNAVAILABLE in m.flags and m.z_cam_m == pytest.approx(15.0, rel=0.02)
    empty = stage.process(1, 33_000_000, np.zeros((0, 4)), [], None)
    assert empty.measurements == []
