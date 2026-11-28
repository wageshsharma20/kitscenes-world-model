"""Unit tests for Stage B.5 night ego-mask fix, native t0 preservation, and progressive sharpen."""

from __future__ import annotations

import numpy as np

from kitscenes import spec
from kitscenes.depth_reproj_world_model import Depth3DFullFoVWorldModel, compute_ego_body_mask
from scripts.eval_world_model_qalign import VARIANT_SPECS


def test_ego_static_min_luma_prevents_dark_road_false_ego_mask() -> None:
    """Dark moving nighttime asphalt (diff < 6) must not trigger a 20% ego-body mask."""
    h, w = spec.HEIGHT, spec.WIDTH
    k_mat = np.array([[500.0, 0.0, w / 2.0], [0.0, 500.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    r_mat = np.eye(3, dtype=np.float64)
    t_vec = np.zeros(3, dtype=np.float64)

    rng = np.random.default_rng(7)
    t0 = rng.integers(8, 13, size=(h, w, 3), dtype=np.uint8)
    tm04 = rng.integers(8, 13, size=(h, w, 3), dtype=np.uint8)

    unfiltered = compute_ego_body_mask(
        "rear_right", None, k_mat, r_mat, t_vec, h, w,
        t0_rgb=t0, past_rgb_tm04=tm04, ego_speed_mps=10.0, ego_static_min_luma=0.0,
    )
    filtered = compute_ego_body_mask(
        "rear_right", None, k_mat, r_mat, t_vec, h, w,
        t0_rgb=t0, past_rgb_tm04=tm04, ego_speed_mps=10.0, ego_static_min_luma=45.0,
    )
    assert unfiltered.mean() >= 0.19
    assert filtered.mean() < 0.05


def test_ego_static_min_luma_keeps_textured_bright_static_body() -> None:
    """Bright textured pixels stationary across t=-0.4s and t=0 are still detected."""
    h, w = spec.HEIGHT, spec.WIDTH
    k_mat = np.array([[500.0, 0.0, w / 2.0], [0.0, 500.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    r_mat = np.eye(3, dtype=np.float64)
    t_vec = np.zeros(3, dtype=np.float64)

    # Textured checkerboard badge in the bottom strip
    ys, xs = np.mgrid[0:h, 0:w]
    pattern = (((xs // 2 + ys // 2) % 2) * 80 + 120).astype(np.uint8)
    img = np.stack([pattern, pattern, pattern], axis=-1)

    filtered = compute_ego_body_mask(
        "rear_right", None, k_mat, r_mat, t_vec, h, w,
        t0_rgb=img, past_rgb_tm04=img.copy(), ego_speed_mps=10.0, ego_static_min_luma=45.0,
    )
    assert filtered[int(0.84 * h):, int(0.1 * w):int(0.5 * w)].all()


def test_preserve_native_t0_and_progressive_sharpen_specs() -> None:
    """Stage B.5 candidate configs and constructor attributes match expectations."""
    model = Depth3DFullFoVWorldModel(
        device="cpu",
        render_scale=1,
        crf_sharpen=0.20,
        crf_sharpen_end=0.52,
        min_lama_hole_pixels=1200,
        min_lama_luma=45.0,
        ego_static_min_luma=45.0,
        preserve_native_t0=True,
    )
    assert model.ego_static_min_luma == 45.0
    assert model.preserve_native_t0 is True
    assert model.crf_sharpen == 0.20
    assert model.crf_sharpen_end == 0.52

    for name in ("cand_j_egofix", "cand_k_native", "cand_l_progsharp", "cand_m_sharp38"):
        assert name in VARIANT_SPECS
        assert VARIANT_SPECS[name]["ego_static_min_luma"] == 45.0
    assert VARIANT_SPECS["cand_j_egofix"]["preserve_native_t0"] is False
    assert VARIANT_SPECS["cand_k_native"]["preserve_native_t0"] is True
    assert VARIANT_SPECS["cand_l_progsharp"]["crf_sharpen_end"] == 0.52
    assert VARIANT_SPECS["cand_m_sharp38"]["crf_sharpen"] == 0.38
