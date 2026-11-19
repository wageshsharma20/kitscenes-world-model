#!/usr/bin/env python3
"""Diagnostic script to rigorously verify bitwise SVD-XT golden equivalence against Hugging Face diffusers."""

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitscenes.spatial_adapter import create_default_euler_scheduler, decode_latents_temporal
from kitscenes.svd_inference import (
    adapt_aspect_ratio_to_svd,
    build_svd_added_time_ids,
    encode_image_to_clip_embedding,
    encode_image_to_vae_latents,
    run_official_svd_diffusion_rollout,
)


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT)).decode("utf-8").strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Phase 5.5.1 Rigorous Golden Equivalence Diagnostic")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=2, help="Denoising steps to compare (e.g. 2 or 25)")
    parser.add_argument("--test-chunks", action="store_true", help="Run VAE decode chunk size comparison (2, 8, 25)")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Import dependencies and record environment
    try:
        import diffusers
        from diffusers import AutoencoderKLTemporalDecoder, StableVideoDiffusionPipeline, UNetSpatioTemporalConditionModel
        diffusers_ver = diffusers.__version__
    except ImportError:
        diffusers = None
        diffusers_ver = "unavailable"

    try:
        import transformers
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        transformers_ver = transformers.__version__
    except ImportError:
        transformers = None
        transformers_ver = "unavailable"

    git_hash = get_git_commit()

    print("=" * 80)
    print("PHASE 5.5.1: RIGOROUS SVD-XT GOLDEN EQUIVALENCE & HOOK DIAGNOSTIC")
    print(f"Device: {device} ({dtype}) | Steps: {args.steps} | Git Commit: {git_hash}")
    print(f"diffusers: {diffusers_ver} | transformers: {transformers_ver} | torch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | CUDA: {torch.version.cuda}")
    print("=" * 80)

    if diffusers is None or transformers is None:
        print("[!] diffusers or transformers not installed. Golden diagnostic requires diffusers environment.")
        return 0

    model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
    print(f"\n[*] Loading pretrained SVD-XT model: {model_id}...")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    unet.eval().requires_grad_(False)
    if hasattr(unet, "enable_forward_chunking"):
        try:
            unet.enable_forward_chunking(chunk_size=1, dim=1)
        except Exception:
            pass

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).to(device)
    vae.eval().requires_grad_(False)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
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
    padded_img, _ = adapt_aspect_ratio_to_svd(pil_front)

    gen = torch.Generator(device=device.type).manual_seed(42)
    z_gold = torch.randn(1, 25, 4, 72, 128, generator=gen, device=device, dtype=dtype)

    diag_report: Dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": git_hash,
        "diffusers_version": diffusers_ver,
        "transformers_version": transformers_ver,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "model_id": model_id,
        "scenario_id": "0000900",
        "denoising_steps": args.steps,
    }

    # 1. VAE Latents
    print("\n--- 1. VAE Image Latents Comparison ---")
    custom_vae_lat = encode_image_to_vae_latents(vae, padded_img, device, noise_aug_strength=0.0, dtype=dtype, video_processor=pipe.video_processor)
    pipe_preprocessed = pipe.video_processor.preprocess(padded_img, height=576, width=1024).to(device)
    needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
    if needs_upcasting:
        pipe.vae.to(dtype=torch.float32)
    pipe_vae_lat = pipe._encode_vae_image(pipe_preprocessed, device=device, num_videos_per_prompt=1, do_classifier_free_guidance=False).to(dtype)
    if needs_upcasting:
        pipe.vae.to(dtype=torch.float16)
    vae_diff = float(torch.max(torch.abs(custom_vae_lat.float() - pipe_vae_lat.float())).item())
    print(f"Custom VAE Latent RMS: {torch.sqrt(torch.mean(custom_vae_lat.float()**2)).item():.6f}")
    print(f"Pipe VAE Latent RMS:   {torch.sqrt(torch.mean(pipe_vae_lat.float()**2)).item():.6f}")
    print(f"Max Absolute Diff (VAE Latents): {vae_diff:.6e} (fp16 tolerance: < 2e-2)")
    diag_report["vae_latent_diff"] = vae_diff

    # 2. CLIP Embeddings
    print("\n--- 2. CLIP Image Embedding Comparison ---")
    custom_clip = encode_image_to_clip_embedding(image_encoder, feature_extractor, padded_img, device, dtype)
    pipe_clip = pipe._encode_image(padded_img, device=device, num_videos_per_prompt=1, do_classifier_free_guidance=False)
    clip_diff = float(torch.max(torch.abs(custom_clip.float() - pipe_clip.float())).item())
    print(f"Custom CLIP RMS: {torch.sqrt(torch.mean(custom_clip.float()**2)).item():.6f}")
    print(f"Pipe CLIP RMS:   {torch.sqrt(torch.mean(pipe_clip.float()**2)).item():.6f}")
    print(f"Max Absolute Diff (CLIP Embeddings): {clip_diff:.6e}")
    diag_report["clip_embedding_diff"] = clip_diff

    # 3. UNet Forward Hook Full Execution Comparison
    print(f"\n--- 3. UNet Forward Hook & End-to-End {args.steps}-Step Execution Comparison ---")
    hook_data_pipe = {}
    def unet_hook_pipe(module, args, kwargs, output):
        if "sample" not in hook_data_pipe:
            hook_data_pipe["sample"] = (args[0] if len(args) > 0 else kwargs.get("sample")).detach().clone()
            hook_data_pipe["timestep"] = (args[1] if len(args) > 1 else kwargs.get("timestep")).detach().clone()
            hook_data_pipe["encoder_hidden_states"] = kwargs.get("encoder_hidden_states").detach().clone()
            hook_data_pipe["added_time_ids"] = kwargs.get("added_time_ids").detach().clone()
            out_sample = output.sample if hasattr(output, "sample") else output[0]
            hook_data_pipe["output"] = out_sample.detach().clone()

    hook_handle_pipe = unet.register_forward_hook(unet_hook_pipe, with_kwargs=True)
    pipe_out = pipe(
        image=padded_img,
        height=576,
        width=1024,
        num_frames=25,
        num_inference_steps=args.steps,
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
        if "sample" not in hook_data_custom:
            hook_data_custom["sample"] = (args[0] if len(args) > 0 else kwargs.get("sample")).detach().clone()
            hook_data_custom["timestep"] = (args[1] if len(args) > 1 else kwargs.get("timestep")).detach().clone()
            hook_data_custom["encoder_hidden_states"] = kwargs.get("encoder_hidden_states").detach().clone()
            hook_data_custom["added_time_ids"] = kwargs.get("added_time_ids").detach().clone()
            out_sample = output.sample if hasattr(output, "sample") else output[0]
            hook_data_custom["output"] = out_sample.detach().clone()

    hook_handle_custom = unet.register_forward_hook(unet_hook_custom, with_kwargs=True)
    custom_out, telem = run_official_svd_diffusion_rollout(
        unet=unet,
        scheduler=create_default_euler_scheduler(),
        initial_latent=custom_vae_lat,
        encoder_hidden_states=custom_clip,
        starting_noise=z_gold.clone(),
        fps=5,
        motion_bucket_id=127,
        noise_aug_strength=0.0,
        num_inference_steps=args.steps,
        min_guidance_scale=1.0,
        max_guidance_scale=3.0,
        do_classifier_free_guidance=True,
    )
    hook_handle_custom.remove()

    print("\n--- 4. Granular UNet Hook Tap Audit at Step 0 ---")
    p_s = hook_data_pipe["sample"].float()
    c_s = hook_data_custom["sample"].float()
    diff_s_noise = float(torch.max(torch.abs(p_s[:, :, :4] - c_s[:, :, :4])).item())
    diff_s_img = float(torch.max(torch.abs(p_s[:, :, 4:] - c_s[:, :, 4:])).item())
    print(f"UNet Input Sample (Noise channels 0-3) Diff: {diff_s_noise:.6e}")
    print(f"UNet Input Sample (Image channels 4-7) Diff: {diff_s_img:.6e}")

    diff_ts = float(torch.max(torch.abs(hook_data_pipe["timestep"].float() - hook_data_custom["timestep"].float())).item())
    print(f"UNet Timestep Diff:                         {diff_ts:.6e}")

    diff_emb = float(torch.max(torch.abs(hook_data_pipe["encoder_hidden_states"].float() - hook_data_custom["encoder_hidden_states"].float())).item())
    print(f"UNet Encoder Hidden States Diff:             {diff_emb:.6e}")

    p_tid = hook_data_pipe["added_time_ids"].float()
    c_tid = hook_data_custom["added_time_ids"].float()
    diff_tid = float(torch.max(torch.abs(p_tid - c_tid)).item())
    print(f"Pipe actual UNet added_time_ids:             {p_tid[0].cpu().numpy().tolist()} (fps-1={p_tid[0,0].item():.1f})")
    print(f"Custom actual UNet added_time_ids:           {c_tid[0].cpu().numpy().tolist()} (fps-1={c_tid[0,0].item():.1f})")
    print(f"UNet Added Time IDs Max Diff:                {diff_tid:.6e}")

    diff_out = float(torch.max(torch.abs(hook_data_pipe["output"].float() - hook_data_custom["output"].float())).item())
    print(f"UNet Raw Noise Prediction Diff:              {diff_out:.6e}")

    final_latents_diff = float(torch.max(torch.abs(pipe_out.float() - custom_out.float())).item())
    print(f"\nFinal Latents Diff after {args.steps} Steps:          {final_latents_diff:.6e}")
    print(f"Pipe out RMS:                                {torch.sqrt(torch.mean(pipe_out.float()**2)).item():.6f}")
    print(f"Custom out RMS:                              {torch.sqrt(torch.mean(custom_out.float()**2)).item():.6f}")

    diag_report["unet_hook_taps"] = {
        "noise_channels_diff": diff_s_noise,
        "image_channels_diff": diff_s_img,
        "timestep_diff": diff_ts,
        "encoder_hidden_states_diff": diff_emb,
        "added_time_ids_pipe": p_tid[0].cpu().numpy().tolist(),
        "added_time_ids_custom": c_tid[0].cpu().numpy().tolist(),
        "added_time_ids_diff": diff_tid,
        "raw_noise_pred_diff": diff_out,
        "final_latents_diff": final_latents_diff,
        "pipe_latents_rms": float(torch.sqrt(torch.mean(pipe_out.float()**2)).item()),
        "custom_latents_rms": float(torch.sqrt(torch.mean(custom_out.float()**2)).item()),
    }

    # 5. VAE Decode Chunk Size Audit
    if args.test_chunks:
        print("\n--- 5. VAE Decode Chunk Size & Temporal Continuity Audit ---")
        chunk_results = {}
        for csz in [2, 8, 25]:
            torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
            t0 = datetime.datetime.now()
            dec = decode_latents_temporal(vae, custom_out, decode_chunk_size=csz)
            dur = (datetime.datetime.now() - t0).total_seconds()
            peak_vram = (torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0

            # Inter-frame L1 jumps across temporal dimension
            frame_diffs = torch.mean(torch.abs(dec[0, 1:] - dec[0, :-1]), dim=[1, 2, 3]).cpu().numpy().tolist()
            chunk_results[csz] = {
                "chunk_size": csz,
                "decode_duration_s": round(dur, 2),
                "peak_vram_mb": round(peak_vram, 1),
                "mean_temporal_diff": round(float(np.mean(frame_diffs)), 6),
                "max_temporal_jump": round(float(np.max(frame_diffs)), 6),
                "tensor": dec.detach().clone(),
            }
            print(f"  Chunk {csz:2d}: {dur:.2f}s | Peak VRAM: {peak_vram:.1f} MB | Mean Inter-Frame Diff: {np.mean(frame_diffs):.4f}")

        diff_chunk2_vs_25 = float(torch.max(torch.abs(chunk_results[2]["tensor"] - chunk_results[25]["tensor"])).item())
        diff_chunk8_vs_25 = float(torch.max(torch.abs(chunk_results[8]["tensor"] - chunk_results[25]["tensor"])).item())
        print(f"\n  Max Pixel Diff (Chunk 2 vs Full 25): {diff_chunk2_vs_25:.6f}")
        print(f"  Max Pixel Diff (Chunk 8 vs Full 25): {diff_chunk8_vs_25:.6f}")

        diag_report["vae_chunk_audit"] = {
            "diff_chunk2_vs_25": diff_chunk2_vs_25,
            "diff_chunk8_vs_25": diff_chunk8_vs_25,
            "benchmarks": {
                csz: {k: v for k, v in chunk_results[csz].items() if k != "tensor"}
                for csz in chunk_results
            }
        }

    out_json = PROJECT_ROOT / "reports" / "golden_diagnostics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(diag_report, f, indent=2)
    print(f"\n[✓] Diagnostic report saved to: {out_json}")

    # Assertion check
    print("\n" + "=" * 80)
    if final_latents_diff < 1e-3 and diff_tid == 0.0:
        print(f"[✓] GOLDEN EQUIVALENCE VERIFIED: diff {final_latents_diff:.6e} < 1e-3 and added_time_ids match exactly!")
        return 0
    else:
        print(f"[X] GOLDEN EQUIVALENCE FAILED: diff {final_latents_diff:.6e} (time_ids diff: {diff_tid:.6e})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
