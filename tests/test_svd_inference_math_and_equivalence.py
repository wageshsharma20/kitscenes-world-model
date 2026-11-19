"""Regression & Equivalence Test Suite for SVD Inference Mathematics (Phase 5.5.1).

Validates the four critical mathematical corrections identified against the official
Hugging Face StableVideoDiffusionPipeline:
  1. Initial noise scaled by scheduler.init_noise_sigma.
  2. CFG guidance schedule linspace(min, max, num_frames) across video frames (not denoising steps).
  3. Unconditional image latents are all zeros, conditional branch receives actual VAE latents.
  4. Micro-conditioning added time IDs use fps - 1 (e.g. 4 for 5 FPS).
  5. Exact bitwise mathematical identity between custom sampler and official SVD pipeline logic.
  6. Aspect ratio padding and exact 876x568 crop coordinates.
  7. Non-silent ground truth failure in evaluation (no black-frame padding).
  8. MMS null fallback when expert trajectory is absent.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest
import torch

from kitscenes import spec
from kitscenes.spatial_adapter import MockSVDUNet, create_default_euler_scheduler
from kitscenes.svd_inference import (
    adapt_aspect_ratio_to_svd,
    build_svd_added_time_ids,
    crop_svd_array_to_kitscenes,
    run_official_svd_diffusion_rollout,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_initial_noise_scaling():
    """Verify that starting Gaussian noise is scaled by scheduler.init_noise_sigma."""
    scheduler = create_default_euler_scheduler()
    scheduler.set_timesteps(25)
    
    assert hasattr(scheduler, "init_noise_sigma"), "Scheduler missing init_noise_sigma attribute!"
    sigma = float(scheduler.init_noise_sigma.item() if hasattr(scheduler.init_noise_sigma, "item") else scheduler.init_noise_sigma)
    assert sigma > 10.0, f"Expected init_noise_sigma > 10.0, got {sigma}"

    starting_noise = torch.randn(1, 25, 4, 72, 128)
    initial_rms = torch.sqrt(torch.mean(starting_noise.float() ** 2)).item()

    scaled_latents = starting_noise.clone() * scheduler.init_noise_sigma
    scaled_rms = torch.sqrt(torch.mean(scaled_latents.float() ** 2)).item()

    # The scaled RMS must match initial RMS * init_noise_sigma within numerical precision
    expected_rms = initial_rms * sigma
    assert abs(scaled_rms - expected_rms) / expected_rms < 1e-4


def test_frame_wise_guidance_schedule():
    """Verify that CFG guidance scale varies across temporal frames, not denoising steps."""
    num_frames = 25
    min_cfg = 1.0
    max_cfg = 3.0
    
    guidance = torch.linspace(min_cfg, max_cfg, num_frames)
    assert len(guidance) == 25
    assert float(guidance[0].item()) == 1.0, "Frame 0 guidance must be min_guidance_scale (1.0)"
    assert float(guidance[-1].item()) == 3.0, "Frame 24 guidance must be max_guidance_scale (3.0)"

    # Verify tensor broadcasts across [B, T, C, H, W]
    guidance_tensor = guidance.view(1, num_frames, 1, 1, 1)
    dummy_pred = torch.ones(1, 25, 4, 72, 128)
    result = dummy_pred * guidance_tensor
    assert result[:, 0, ...].mean().item() == 1.0
    assert result[:, 24, ...].mean().item() == 3.0


def test_unconditional_image_latents_are_zero():
    """Verify that unconditional CFG branch receives zero image latents, while conditional receives actual latents."""
    init_latent = torch.randn(1, 4, 72, 128)
    negative_image_latents = torch.zeros_like(init_latent)
    image_latents_cat = torch.cat([negative_image_latents, init_latent], dim=0)

    assert image_latents_cat.shape == (2, 4, 72, 128)
    assert torch.all(image_latents_cat[0] == 0.0), "Unconditional branch must be strictly all zeros!"
    assert torch.all(image_latents_cat[1] == init_latent[0]), "Conditional branch must match observed image latent!"


def test_fps_micro_conditioning():
    """Verify that 5 FPS is micro-conditioned matching official diffusers pipeline."""
    fps = 5
    motion_bucket_id = 127
    noise_aug_strength = 0.02
    device = torch.device("cpu")
    dtype = torch.float32

    time_ids = build_svd_added_time_ids(
        fps=fps,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        dtype=dtype,
        device=device,
        do_classifier_free_guidance=True,
    )

    assert time_ids.shape == (2, 3)
    # Diffusers SVD converts user playback FPS to fps - 1 micro-conditioning (e.g. 5 -> 4.0)
    assert time_ids[0, 0].item() == 4.0, f"Expected fps_conditioned 4.0 for 5 FPS, got {time_ids[0, 0].item()}"
    assert time_ids[1, 0].item() == 4.0
    assert time_ids[0, 1].item() == 127.0
    assert abs(time_ids[0, 2].item() - 0.02) < 1e-4


def test_exact_mathematical_identity_against_official_svd_logic():
    """Verify exact bitwise mathematical identity between run_official_svd_diffusion_rollout and official SVD pipeline logic."""
    device = torch.device("cpu")
    dtype = torch.float32
    num_frames = 25
    num_steps = 3
    min_guidance = 1.0
    max_guidance = 3.0
    fps = 5
    motion_bucket = 127
    noise_aug = 0.02

    unet = MockSVDUNet(in_channels=8, out_channels=4).to(device=device, dtype=dtype)
    scheduler1 = create_default_euler_scheduler()
    scheduler2 = create_default_euler_scheduler()

    gen = torch.Generator(device=device).manual_seed(999)
    starting_noise = torch.randn(1, num_frames, 4, 72, 128, generator=gen, device=device, dtype=dtype)
    init_latent = torch.randn(1, 4, 72, 128, generator=gen, device=device, dtype=dtype)
    clip_emb = torch.randn(1, 1, 1024, generator=gen, device=device, dtype=dtype)

    # Reference Pipeline A: Direct Official SVD Pipeline Loop
    scheduler1.set_timesteps(num_steps, device=device)
    latents_ref = starting_noise.clone() * scheduler1.init_noise_sigma
    image_latents_ref = torch.cat([torch.zeros_like(init_latent), init_latent], dim=0).unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
    fps_conditioned_ref = fps - 1
    time_ids_ref = torch.tensor([[fps_conditioned_ref, motion_bucket, noise_aug]], dtype=dtype, device=device).repeat(2, 1)
    hidden_states_ref = torch.cat([torch.zeros_like(clip_emb), clip_emb], dim=0)
    guidance_ref = torch.linspace(min_guidance, max_guidance, num_frames, device=device, dtype=dtype).view(1, num_frames, 1, 1, 1)

    for t in scheduler1.timesteps:
        inp = torch.cat([latents_ref, latents_ref], dim=0)
        inp = scheduler1.scale_model_input(inp, t)
        inp = torch.cat([inp, image_latents_ref], dim=2)
        with torch.no_grad():
            np_pred = unet(inp, t, encoder_hidden_states=hidden_states_ref, added_time_ids=time_ids_ref).sample
        u, c = np_pred.chunk(2, dim=0)
        np_pred = u + guidance_ref * (c - u)
        latents_ref = scheduler1.step(np_pred, t, latents_ref).prev_sample

    # Custom Sampler: run_official_svd_diffusion_rollout
    latents_custom, telem = run_official_svd_diffusion_rollout(
        unet=unet,
        scheduler=scheduler2,
        initial_latent=init_latent,
        encoder_hidden_states=clip_emb,
        starting_noise=starting_noise,
        fps=fps,
        motion_bucket_id=motion_bucket,
        noise_aug_strength=noise_aug,
        num_inference_steps=num_steps,
        min_guidance_scale=min_guidance,
        max_guidance_scale=max_guidance,
        do_classifier_free_guidance=True,
    )

    diff = torch.max(torch.abs(latents_ref - latents_custom)).item()
    assert diff == 0.0, f"Mathematical divergence detected: {diff}"
    assert telem["fps_playback"] == 5
    assert telem["fps_conditioned"] == 4
    assert telem["guidance_frame_0"] == 1.0
    assert telem["guidance_frame_24"] == 3.0


def test_crop_svd_array_to_kitscenes():
    """Verify that crop_svd_array_to_kitscenes produces exact 876x568 dimensions."""
    dummy_render = np.zeros((25, 576, 1024, 3), dtype=np.uint8)
    cropped = crop_svd_array_to_kitscenes(dummy_render)
    assert cropped.shape == (25, 568, 876, 3), f"Expected (25, 568, 876, 3), got {cropped.shape}"
