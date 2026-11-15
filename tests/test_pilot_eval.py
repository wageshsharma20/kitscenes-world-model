"""Unit tests for Phase 3 Pilot comparative evaluation metrics and contracts."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.eval_pilot_comparison import (
    compute_frame_ssim,
    compute_laplacian_sharpness,
    compute_lateral_motion_center,
    compute_optical_flow_coherence,
    compute_psnr,
)


def test_compute_psnr_identical():
    """Verify PSNR on identical images is capped at 100 dB."""
    val = compute_psnr(0.0)
    assert val == 100.0


def test_compute_psnr_noisy():
    """Verify PSNR decreases monotonically with MSE."""
    psnr_low_noise = compute_psnr(1.0)
    psnr_high_noise = compute_psnr(100.0)
    assert psnr_low_noise > psnr_high_noise
    assert round(psnr_low_noise, 1) == 48.1


def test_compute_frame_ssim_identical():
    """Verify SSIM on identical images is 1.0."""
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    ssim = compute_frame_ssim(img, img)
    assert abs(ssim - 1.0) < 1e-3


def test_compute_frame_ssim_different():
    """Verify SSIM on distinct images is less than 1.0."""
    img1 = np.full((128, 128, 3), 50, dtype=np.uint8)
    img2 = np.full((128, 128, 3), 200, dtype=np.uint8)
    ssim = compute_frame_ssim(img1, img2)
    assert ssim < 0.5


def test_laplacian_sharpness_blurry_vs_sharp():
    """Verify Laplacian sharpness proxy detects edge degradation."""
    # Sharp pattern: high frequency checkerboard
    sharp_frames = np.zeros((5, 128, 128, 3), dtype=np.uint8)
    sharp_frames[:, ::4, ::4] = 255

    # Blurry pattern: constant smooth gradient
    blurry_frames = np.ones((5, 128, 128, 3), dtype=np.uint8) * 128

    score_sharp = compute_laplacian_sharpness(sharp_frames)
    score_blurry = compute_laplacian_sharpness(blurry_frames)

    assert score_sharp > score_blurry
    assert score_blurry < 1.0


def test_lateral_motion_center_directional_response():
    """Verify lateral center-of-mass detects left vs right motion."""
    # Video 1: motion on the left half of the image (coords in [-1.0, 0.0])
    frames_left = np.zeros((5, 64, 128, 3), dtype=np.uint8)
    for t in range(5):
        frames_left[t, :, :40] = t * 40  # Motion concentrated on left

    com_left = compute_lateral_motion_center(frames_left)
    assert np.all(com_left < 0.0), f"Left motion must have negative CoM! Got: {com_left}"

    # Video 2: motion on the right half of the image (coords in [0.0, 1.0])
    frames_right = np.zeros((5, 64, 128, 3), dtype=np.uint8)
    for t in range(5):
        frames_right[t, :, 88:] = t * 40  # Motion concentrated on right

    com_right = compute_lateral_motion_center(frames_right)
    assert np.all(com_right > 0.0), f"Right motion must have positive CoM! Got: {com_right}"
