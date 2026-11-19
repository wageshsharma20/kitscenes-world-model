"""Official Stable Video Diffusion (SVD-XT) Inference Engine & Mathematical Sampler.

Implements the exact inference mathematics from the official Hugging Face
StableVideoDiffusionPipeline:
  1. Scheduler-scaled initial Gaussian noise: z_T = starting_noise * scheduler.init_noise_sigma
  2. Frame-wise CFG schedule: guidance_scale = torch.linspace(min_guidance, max_guidance, num_frames)
     broadcast to [B, num_frames, 1, 1, 1], modulating temporal extrapolation across the 25 frames.
  3. Unconditional image latent zero-conditioning: unconditional branch receives torch.zeros_like,
     while conditional branch receives actual mode-extracted VAE image latents.
  4. Micro-conditioning added time IDs: fps is passed as fps - 1 (e.g. 4 for 5.0 FPS),
     alongside motion_bucket_id and noise_aug_strength.
  5. Multi-view token bank integration: cleanly accepts both single-camera CLIP embeddings [1, 1, 1024]
     and CSSC 27-token surround banks [1, 27, 1024].
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from . import spec
from .spatial_adapter import decode_latents_temporal


def adapt_aspect_ratio_to_svd(
    img: Image.Image, target_width: int = 1024, target_height: int = 576
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """Pad 876x568 to SVD target dimensions 1024x576 using edge replication.
    
    Returns:
        (padded_image, (pad_top, pad_bottom, pad_left, pad_right))
    """
    w, h = img.size
    pad_h = target_height - h
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_w = target_width - w
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    arr = np.array(img)
    padded = np.pad(
        arr,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="edge",
    )
    return Image.fromarray(padded), (pad_top, pad_bottom, pad_left, pad_right)


def crop_svd_array_to_kitscenes(arr: np.ndarray) -> np.ndarray:
    """Crop 1024x576 SVD output array back to exact KITScenes dimensions 876x568.
    
    Accepts shape [T, 576, 1024, 3] or [576, 1024, 3].
    """
    if arr.ndim == 4:
        return arr[:, 4:572, 74:950, :]
    elif arr.ndim == 3:
        return arr[4:572, 74:950, :]
    else:
        raise ValueError(f"Unexpected array shape for cropping: {arr.shape}")


def encode_image_to_vae_latents(
    vae: Any,
    image: Image.Image,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    noise_aug_strength: float = 0.02,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Encode initial PIL image to mode-extracted VAE latent [1, 4, 72, 128] with noise augmentation."""
    padded_img, _ = adapt_aspect_ratio_to_svd(image)
    arr = np.array(padded_img).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
    
    # Safely match VAE parameter dtype (e.g. float32 to avoid half-precision conv bias mismatch)
    vae_dtype = next(vae.parameters()).dtype if hasattr(vae, "parameters") and list(vae.parameters()) else dtype
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=vae_dtype)

    if noise_aug_strength > 0:
        noise = torch.randn(tensor.shape, generator=generator, device=device, dtype=vae_dtype)
        tensor = tensor + noise_aug_strength * noise

    with torch.no_grad():
        dist = vae.encode(tensor).latent_dist
        latent = dist.mode()
        shift_factor = getattr(vae.config, "shift_factor", None)
        if shift_factor is not None:
            latent = latent - shift_factor
        scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
        latent = latent * scaling_factor
    return latent.to(dtype=dtype)


def encode_image_to_clip_embedding(
    image_encoder: Any,
    feature_extractor: Any,
    image: Image.Image,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Extract normalized CLIP image embedding [1, 1, 1024] from observed frame."""
    padded_img, _ = adapt_aspect_ratio_to_svd(image)
    px = feature_extractor(images=padded_img, return_tensors="pt").pixel_values.to(
        device=device, dtype=dtype
    )
    with torch.no_grad():
        clip_emb = image_encoder(px).image_embeds.unsqueeze(1)  # [1, 1, 1024]
    return clip_emb


def build_svd_added_time_ids(
    fps: int,
    motion_bucket_id: int,
    noise_aug_strength: float,
    dtype: torch.dtype,
    device: torch.device,
    do_classifier_free_guidance: bool = True,
) -> torch.Tensor:
    """Construct added time IDs strictly matching the official SVD pipeline.
    
    NOTE: Stable Video Diffusion UNet was micro-conditioned on (fps - 1) during training.
    """
    fps_conditioned = fps - 1
    add_time_ids = torch.tensor([[fps_conditioned, motion_bucket_id, noise_aug_strength]], dtype=dtype, device=device)
    if do_classifier_free_guidance:
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
    return add_time_ids


def run_official_svd_diffusion_rollout(
    unet: Any,
    scheduler: Any,
    initial_latent: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    starting_noise: torch.Tensor,
    fps: int = 5,
    motion_bucket_id: int = 127,
    noise_aug_strength: float = 0.02,
    num_inference_steps: int = 25,
    min_guidance_scale: float = 1.0,
    max_guidance_scale: float = 3.0,
    do_classifier_free_guidance: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Execute SVD-XT diffusion rollout strictly adhering to official Hugging Face SVD mathematics.
    
    Args:
        unet: Pretrained or mock SVD UNet
        scheduler: EulerDiscreteScheduler instance
        initial_latent: Mode-extracted VAE latents from observed t=0 frame [1, 4, 72, 128]
        encoder_hidden_states: Conditioning hidden states [1, seq_len, 1024] (CLIP or CSSC bank)
        starting_noise: Pure Gaussian noise z_T ~ N(0, I) [1, num_frames, 4, 72, 128]
        fps: Video playback rate (UNet is conditioned on fps - 1)
        motion_bucket_id: Motion bucket conditioning scalar
        noise_aug_strength: Noise augmentation strength for added time IDs
        num_inference_steps: Total denoising steps
        min_guidance_scale: CFG scale for first frame
        max_guidance_scale: CFG scale for last frame
        do_classifier_free_guidance: Whether CFG is active
        
    Returns:
        (latents, telemetry_dict)
    """
    device = starting_noise.device
    dtype = starting_noise.dtype
    num_frames = starting_noise.shape[1]

    # 1. Prepare timesteps and scheduler noise scale
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    init_noise_sigma = scheduler.init_noise_sigma

    # 2. Scale initial noise by scheduler's init_noise_sigma
    initial_noise_rms_before = float(torch.sqrt(torch.mean(starting_noise.float() ** 2)).item())
    latents = starting_noise.clone() * init_noise_sigma
    initial_latent_rms_after = float(torch.sqrt(torch.mean(latents.float() ** 2)).item())

    # 3. Prepare image conditioning: unconditional half is zero, conditional half is initial latent
    if do_classifier_free_guidance:
        negative_image_latents = torch.zeros_like(initial_latent)
        image_latents_cat = torch.cat([negative_image_latents, initial_latent], dim=0)  # [2, 4, 72, 128]
    else:
        image_latents_cat = initial_latent

    # Repeat image latents across temporal frames
    image_latents_cat = image_latents_cat.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)  # [2, num_frames, 4, 72, 128]

    # 4. Prepare encoder hidden states (zero-conditioned for unconditional branch)
    if do_classifier_free_guidance:
        negative_hidden_states = torch.zeros_like(encoder_hidden_states)
        hidden_states_cat = torch.cat([negative_hidden_states, encoder_hidden_states], dim=0)
    else:
        hidden_states_cat = encoder_hidden_states

    # 5. Prepare added time IDs with fps - 1 micro-conditioning
    added_time_ids = build_svd_added_time_ids(
        fps=fps,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        dtype=dtype,
        device=device,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )

    # 6. Prepare per-frame guidance scale tensor [1, num_frames, 1, 1, 1]
    guidance_schedule = torch.linspace(
        min_guidance_scale, max_guidance_scale, num_frames, device=device, dtype=dtype
    )
    guidance_tensor = guidance_schedule.view(1, num_frames, 1, 1, 1)

    # 7. Execute Denoising Loop
    step_telemetry: List[Dict[str, Any]] = []

    for step_idx, t in enumerate(timesteps):
        if do_classifier_free_guidance:
            latent_model_input = torch.cat([latents, latents], dim=0)
        else:
            latent_model_input = latents

        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # Concatenate image latents along channel dimension -> [2, num_frames, 8, 72, 128]
        latent_model_input = torch.cat([latent_model_input, image_latents_cat], dim=2)

        # Predict noise residual with UNet
        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=hidden_states_cat,
                added_time_ids=added_time_ids,
            ).sample

        # Perform classifier-free guidance across frames
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + guidance_tensor * (noise_pred_cond - noise_pred_uncond)

        # Step scheduler
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        step_telemetry.append({
            "step": step_idx,
            "timestep": float(t.item()) if hasattr(t, "item") else float(t),
            "latent_rms": float(torch.sqrt(torch.mean(latents.float() ** 2)).item()),
            "has_nan": bool(torch.isnan(latents).any().item()),
            "has_inf": bool(torch.isinf(latents).any().item()),
        })

    telemetry = {
        "scheduler_class": scheduler.__class__.__name__,
        "num_inference_steps": num_inference_steps,
        "init_noise_sigma": float(init_noise_sigma.item() if hasattr(init_noise_sigma, "item") else init_noise_sigma),
        "initial_noise_rms_before_scaling": round(initial_noise_rms_before, 6),
        "initial_latent_rms_after_scaling": round(initial_latent_rms_after, 6),
        "first_timestep": float(timesteps[0].item()) if hasattr(timesteps[0], "item") else float(timesteps[0]),
        "first_sigma": float(scheduler.sigmas[0].item()) if hasattr(scheduler.sigmas[0], "item") else float(scheduler.sigmas[0]),
        "fps_conditioned": fps - 1,
        "motion_bucket_id": motion_bucket_id,
        "noise_aug_strength": noise_aug_strength,
        "min_guidance_scale": min_guidance_scale,
        "max_guidance_scale": max_guidance_scale,
        "guidance_frame_0": float(guidance_schedule[0].item()),
        "guidance_frame_24": float(guidance_schedule[-1].item()),
        "has_nan": any(s["has_nan"] for s in step_telemetry),
        "has_inf": any(s["has_inf"] for s in step_telemetry),
        "step_telemetry": step_telemetry,
    }

    return latents, telemetry
