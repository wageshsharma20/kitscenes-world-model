#!/usr/bin/env python3
"""Diagnostic script to isolate the exact step/tensor where custom rollout diverges from official SVD pipeline."""

import sys
from pathlib import Path
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitscenes.spatial_adapter import create_default_euler_scheduler
from kitscenes.svd_inference import (
    adapt_aspect_ratio_to_svd,
    build_svd_added_time_ids,
    encode_image_to_clip_embedding,
    encode_image_to_vae_latents,
    run_official_svd_diffusion_rollout,
)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"[*] Running diagnostic on {device} ({dtype})...")

    from diffusers import AutoencoderKLTemporalDecoder, StableVideoDiffusionPipeline, UNetSpatioTemporalConditionModel
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
    unet = UNetSpatioTemporalConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    unet.eval().requires_grad_(False)
    if hasattr(unet, "enable_forward_chunking"):
        try:
            unet.enable_forward_chunking(chunk_size=5, dim=0)
        except Exception:
            pass

    vae = AutoencoderKLTemporalDecoder.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, low_cpu_mem_usage=True).to(device)
    vae.eval().requires_grad_(False)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    image_encoder.eval().requires_grad_(False)

    feature_extractor = CLIPImageProcessor.from_pretrained(model_id, subfolder="feature_extractor")

    pipe = StableVideoDiffusionPipeline(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=create_default_euler_scheduler(),
        feature_extractor=feature_extractor,
    ).to(device)

    obs_file = PROJECT_ROOT / "data" / "val_observed_t0" / "0000900" / "front.jpg"
    pil_front = Image.open(obs_file).convert("RGB")

    gen = torch.Generator(device=device.type).manual_seed(42)
    z_gold = torch.randn(1, 25, 4, 72, 128, generator=gen, device=device, dtype=dtype)

    print("\n--- 1. VAE Image Latents Comparison ---")
    padded_img, _ = adapt_aspect_ratio_to_svd(pil_front)
    # Custom VAE latent
    custom_vae_lat = encode_image_to_vae_latents(vae, padded_img, device, noise_aug_strength=0.0, dtype=dtype)
    print(f"Custom VAE Latent: shape={custom_vae_lat.shape}, rms={torch.sqrt(torch.mean(custom_vae_lat.float()**2)).item():.6f}")

    # Official Pipeline VAE latent
    pipe_preprocessed = pipe.video_processor.preprocess(padded_img, height=576, width=1024).to(device)
    pipe_vae_lat = pipe._encode_vae_image(pipe_preprocessed, device=device, num_videos_per_prompt=1, do_classifier_free_guidance=False)
    print(f"Pipe VAE Latent:   shape={pipe_vae_lat.shape}, rms={torch.sqrt(torch.mean(pipe_vae_lat.float()**2)).item():.6f}")
    vae_diff = torch.max(torch.abs(custom_vae_lat.float() - pipe_vae_lat.float())).item()
    print(f"Max Absolute Diff (VAE Latents): {vae_diff:.6e}")

    print("\n--- 2. CLIP Image Embedding Comparison ---")
    custom_clip = encode_image_to_clip_embedding(image_encoder, feature_extractor, padded_img, device, dtype)
    pipe_clip = pipe._encode_image(padded_img, device=device, num_videos_per_prompt=1, do_classifier_free_guidance=False)
    print(f"Custom CLIP: shape={custom_clip.shape}, rms={torch.sqrt(torch.mean(custom_clip.float()**2)).item():.6f}")
    print(f"Pipe CLIP:   shape={pipe_clip.shape}, rms={torch.sqrt(torch.mean(pipe_clip.float()**2)).item():.6f}")
    clip_diff = torch.max(torch.abs(custom_clip.float() - pipe_clip.float())).item()
    print(f"Max Absolute Diff (CLIP Embeddings): {clip_diff:.6e}")

    print("\n--- 3. Added Time IDs Comparison ---")
    custom_time_ids = build_svd_added_time_ids(fps=5, motion_bucket_id=127, noise_aug_strength=0.0, dtype=dtype, device=device, do_classifier_free_guidance=False)
    pipe_time_ids = pipe._get_add_time_ids(fps=5, motion_bucket_id=127, noise_aug_strength=0.0, dtype=dtype, batch_size=1, num_videos_per_prompt=1, do_classifier_free_guidance=False)
    print(f"Custom time IDs: {custom_time_ids}")
    print(f"Pipe time IDs:   {pipe_time_ids}")
    time_ids_diff = torch.max(torch.abs(custom_time_ids.float() - pipe_time_ids.float())).item()
    print(f"Max Absolute Diff (Time IDs): {time_ids_diff:.6e}")

    print("\n--- 4. Initial Latents Scaling in prepare_latents vs Custom ---")
    pipe.scheduler.set_timesteps(2, device=device)
    pipe_latents = pipe.prepare_latents(
        batch_size=1,
        num_frames=25,
        num_channels_latents=4 * 2, # Note: diffusers passes 8 to get 4
        height=576,
        width=1024,
        dtype=dtype,
        device=device,
        generator=None,
        latents=z_gold.clone(),
    )
    custom_latents = z_gold.clone() * pipe.scheduler.init_noise_sigma
    print(f"Pipe Latents after prepare_latents: rms={torch.sqrt(torch.mean(pipe_latents.float()**2)).item():.6f}")
    print(f"Custom Latents (z_gold * sigma):   rms={torch.sqrt(torch.mean(custom_latents.float()**2)).item():.6f}")
    lat_prep_diff = torch.max(torch.abs(pipe_latents.float() - custom_latents.float())).item()
    print(f"Max Absolute Diff (Initial Latents): {lat_prep_diff:.6e}")

    print("\n--- 5. Full 1-Step Execution & UNet Forward Hook Comparison ---")
    hook_data_pipe = {}
    def unet_hook_pipe(module, args, kwargs, output):
        hook_data_pipe["sample"] = args[0] if len(args) > 0 else kwargs.get("sample")
        hook_data_pipe["timestep"] = args[1] if len(args) > 1 else kwargs.get("timestep")
        hook_data_pipe["encoder_hidden_states"] = kwargs.get("encoder_hidden_states")
        hook_data_pipe["added_time_ids"] = kwargs.get("added_time_ids")
        hook_data_pipe["output"] = output.sample if hasattr(output, "sample") else output[0]

    hook_handle_pipe = unet.register_forward_hook(unet_hook_pipe, with_kwargs=True)
    # Run official pipeline for 1 step
    pipe_out = pipe(
        image=padded_img,
        height=576,
        width=1024,
        num_frames=25,
        num_inference_steps=1,
        min_guidance_scale=1.0,
        max_guidance_scale=3.0,
        fps=5,
        motion_bucket_id=127,
        noise_aug_strength=0.0,
        latents=z_gold.clone(),
        output_type="latent",
    ).frames
    hook_handle_pipe.remove()

    hook_data_custom = {}
    def unet_hook_custom(module, args, kwargs, output):
        hook_data_custom["sample"] = args[0] if len(args) > 0 else kwargs.get("sample")
        hook_data_custom["timestep"] = args[1] if len(args) > 1 else kwargs.get("timestep")
        hook_data_custom["encoder_hidden_states"] = kwargs.get("encoder_hidden_states")
        hook_data_custom["added_time_ids"] = kwargs.get("added_time_ids")
        hook_data_custom["output"] = output.sample if hasattr(output, "sample") else output[0]

    hook_handle_custom = unet.register_forward_hook(unet_hook_custom, with_kwargs=True)
    # Run custom rollout for 1 step
    custom_out, telem = run_official_svd_diffusion_rollout(
        unet=unet,
        scheduler=create_default_euler_scheduler(),
        initial_latent=custom_vae_lat,
        encoder_hidden_states=custom_clip,
        starting_noise=z_gold.clone(),
        fps=5,
        motion_bucket_id=127,
        noise_aug_strength=0.0,
        num_inference_steps=1,
        min_guidance_scale=1.0,
        max_guidance_scale=3.0,
        do_classifier_free_guidance=True,
    )
    hook_handle_custom.remove()

    print("\n--- Granular Hook Comparison at Step 0 ---")
    p_s = hook_data_pipe["sample"].float()
    c_s = hook_data_custom["sample"].float()
    diff_s_noise = torch.max(torch.abs(p_s[:, :, :4] - c_s[:, :, :4])).item()
    diff_s_img = torch.max(torch.abs(p_s[:, :, 4:] - c_s[:, :, 4:])).item()
    print(f"UNet Input Sample (Noise channels 0-3) Diff: {diff_s_noise:.6e}")
    print(f"UNet Input Sample (Image channels 4-7) Diff: {diff_s_img:.6e}")

    diff_ts = torch.max(torch.abs(hook_data_pipe["timestep"].float() - hook_data_custom["timestep"].float())).item()
    print(f"UNet Timestep Diff:                         {diff_ts:.6e}")

    diff_emb = torch.max(torch.abs(hook_data_pipe["encoder_hidden_states"].float() - hook_data_custom["encoder_hidden_states"].float())).item()
    print(f"UNet Encoder Hidden States Diff:             {diff_emb:.6e}")

    diff_tid = torch.max(torch.abs(hook_data_pipe["added_time_ids"].float() - hook_data_custom["added_time_ids"].float())).item()
    print(f"UNet Added Time IDs Diff:                    {diff_tid:.6e}")

    diff_out = torch.max(torch.abs(hook_data_pipe["output"].float() - hook_data_custom["output"].float())).item()
    print(f"UNet Raw Output (noise_pred) Diff:           {diff_out:.6e}")

    step1_diff = torch.max(torch.abs(pipe_out.float() - custom_out.float())).item()
    print(f"Final Latents Diff after Step 1:             {step1_diff:.6e}")
    print(f"Pipe out RMS:   {torch.sqrt(torch.mean(pipe_out.float()**2)).item():.6f}")
    print(f"Custom out RMS: {torch.sqrt(torch.mean(custom_out.float()**2)).item():.6f}")

if __name__ == "__main__":
    main()
