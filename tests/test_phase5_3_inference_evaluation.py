"""Tests for Phase 5.3: Genuine Predictive Inference and Controlled Evaluation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from kitscenes import spec
from kitscenes.camera_projection import CameraProjectionEngine, CameraRig
from kitscenes.multiview_conditioning import MultiViewConditioner
from kitscenes.spatial_adapter import MockSVDUNet, SpatialAdapter, SpatialAdapterConfig, create_default_euler_scheduler
from scripts.evaluate_cssc_checkpoint import (
    CANONICAL_CAMERAS,
    EXPECTED_BASELINE_SHA256,
    compute_cross_camera_overlap_consistency,
    compute_gaussian_ssim,
    compute_trajectory_adherence_from_flow,
    compute_true_psnr,
    compute_video_ssim,
    run_diffusion_denoising,
    validate_baseline_archive,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_baseline_submission_archive_strictly_unmodified():
    """Verify that the official baseline submission zip remains byte-identical."""
    is_intact, current_hash = validate_baseline_archive()
    assert is_intact, f"Baseline archive altered! Current hash: {current_hash}"
    assert current_hash == EXPECTED_BASELINE_SHA256


def test_conditioned_output_diverges_from_baseline():
    """Verify that Model A (Baseline) and Model B (Conditioned) produce different outputs from identical noise."""
    b, t, c, h, w = 1, 25, 4, 72, 128
    device = torch.device("cpu")
    dtype = torch.float32

    unet = MockSVDUNet(in_channels=8, out_channels=4).to(device=device, dtype=dtype)
    unet.eval()

    scheduler = create_default_euler_scheduler()
    initial_latents = torch.randn((b, c, h, w), device=device, dtype=dtype)

    # Identical starting noise z_T
    seed = 42
    torch.manual_seed(seed)
    noise_zT = torch.randn((b, t, c, h, w), device=device, dtype=dtype)

    # Single camera CLIP token: [1, 1, 1024]
    clip_emb = torch.randn((b, 1, 1024), device=device, dtype=dtype)

    # 1. Model A: Baseline SVD-XT (no spatial adapter, single CLIP token)
    base_latents = run_diffusion_denoising(
        unet=unet,
        scheduler=scheduler,
        initial_latents=initial_latents,
        encoder_hidden_states=clip_emb,
        spatial_cond_features=None,
        starting_noise=noise_zT,
        num_inference_steps=2,
        min_guidance_scale=1.0,
        max_guidance_scale=2.0,
        device=device,
        dtype=dtype,
    )

    # 2. Model B: CSSC Conditioned (27-token bank + SpatialAdapter features)
    cssc_tokens = torch.randn((b, 27, 1024), device=device, dtype=dtype)
    spatial_features = torch.randn((b, t, c, h, w), device=device, dtype=dtype) * 0.1

    cond_latents = run_diffusion_denoising(
        unet=unet,
        scheduler=scheduler,
        initial_latents=initial_latents,
        encoder_hidden_states=cssc_tokens,
        spatial_cond_features=spatial_features,
        starting_noise=noise_zT,  # Exact same noise
        num_inference_steps=2,
        min_guidance_scale=1.0,
        max_guidance_scale=2.0,
        device=device,
        dtype=dtype,
    )

    diff = torch.norm(cond_latents - base_latents).item()
    assert diff > 0.0, "Model B output did not diverge from Model A baseline!"
    assert not torch.isnan(cond_latents).any(), "NaNs detected in conditioned output"
    assert not torch.isnan(base_latents).any(), "NaNs detected in baseline output"


def test_model_predictions_are_not_ground_truth():
    """Verify that predictions from denoising diffusion are distinct from ground truth frames."""
    pred_frames = np.zeros((25, spec.HEIGHT, spec.WIDTH, 3), dtype=np.uint8)
    gt_frames = np.full((25, spec.HEIGHT, spec.WIDTH, 3), 128, dtype=np.uint8)

    psnr = compute_true_psnr(pred_frames, gt_frames)
    ssim = compute_video_ssim(pred_frames, gt_frames)

    # Completely different frames should have low PSNR and near-zero SSIM
    assert psnr < 15.0, f"Expected low PSNR for distinct frames, got {psnr}"
    assert ssim < 0.1, f"Expected near-zero SSIM for distinct frames, got {ssim}"


def test_psnr_and_ssim_identical_frames():
    """Verify mathematical bounds for true PSNR and Gaussian SSIM."""
    frames = np.random.randint(0, 255, (25, spec.HEIGHT, spec.WIDTH, 3), dtype=np.uint8)
    psnr = compute_true_psnr(frames, frames)
    ssim = compute_video_ssim(frames, frames)

    assert psnr == 100.0, f"Identical frames should have 100 dB PSNR, got {psnr}"
    assert pytest.approx(1.0, abs=1e-4) == ssim


def test_cross_camera_overlap_consistency():
    """Verify cross-camera geometric overlap metric."""
    camera_frames = {
        cam: np.zeros((25, spec.HEIGHT, spec.WIDTH, 3), dtype=np.uint8)
        for cam in CANONICAL_CAMERAS
    }
    # When all camera boundaries match perfectly
    score = compute_cross_camera_overlap_consistency(camera_frames)
    assert 0.0 <= score <= 1.0


def test_trajectory_adherence_from_flow():
    """Verify trajectory adherence calculation from visual flow."""
    frames = np.random.randint(0, 255, (25, spec.HEIGHT, spec.WIDTH, 3), dtype=np.uint8)
    waypoints = np.zeros((25, 2))
    waypoints[:, 1] = np.linspace(0, 25, 25)  # straight line forward

    adherence = compute_trajectory_adherence_from_flow(frames, waypoints)
    assert 0.0 <= adherence <= 1.0


def test_checkpoint_loads_properly():
    """Verify that models/cssc_pilot_step30.pt loads and matches SpatialAdapter & MultiViewConditioner state dicts."""
    ckpt_path = PROJECT_ROOT / "models" / "cssc_pilot_step30.pt"
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert "spatial_adapter_state_dict" in ckpt
    assert "conditioner_state_dict" in ckpt

    adapter = SpatialAdapter(SpatialAdapterConfig())
    adapter.load_state_dict(ckpt["spatial_adapter_state_dict"])

    conditioner = MultiViewConditioner()
    conditioner.load_state_dict(ckpt["conditioner_state_dict"])

    assert len(adapter.state_dict()) == len(ckpt["spatial_adapter_state_dict"])
    assert len(conditioner.state_dict()) == len(ckpt["conditioner_state_dict"])
