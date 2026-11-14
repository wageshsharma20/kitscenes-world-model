#!/usr/bin/env python3
"""Run genuine Stable Video Diffusion (SVD) inference on a single NVIDIA GPU (e.g. Kaggle T4).

Protocol (Smoke-Test Milestone):
- 1 Scenario: 0000902 (Intersection Right Turn)
- 1 Camera: front
- 25 Frames @ 5.0 FPS (876x568 target contract)
- Precision: FP16 (stabilityai/stable-video-diffusion-img2vid-xt)
- Aspect Ratio: Symmetric reflection letterbox to 1024x576, inverse-cropped to 888x576, resized to 876x568
- Micro-conditioning: fps=5 passed directly to the diffusion pipeline
- Memory Optimizations:
    * pipeline.enable_model_cpu_offload()
    * pipeline.unet.enable_forward_chunking()
    * decode_chunk_size=2
- Verification:
    * Denoising step counter callback (proves all 15 steps executed in UNet)
    * Raw frame-to-frame differences computed BEFORE MP4 encoding
    * Real peak allocated and reserved VRAM logging

Usage on Kaggle (1x T4):
    python scripts/run_kaggle_svd_single_camera.py --device cuda --steps 15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitscenes import knn_trajectory, pq_proxy, spec, video_engine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run genuine single-GPU SVD inference for Scenario 0000902 (front camera)."
    )
    parser.add_argument(
        "--scenario-id",
        type=str,
        default="0000902",
        help="Target scenario ID (default: 0000902)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="front",
        choices=spec.CAMERAS,
        help="Target camera (default: front)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="stabilityai/stable-video-diffusion-img2vid-xt",
        help="Hugging Face model repository ID (default: stabilityai/stable-video-diffusion-img2vid-xt)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=15,
        help="Number of diffusion denoising steps (default: 15 for initial smoke test)",
    )
    parser.add_argument(
        "--decode-chunk-size",
        type=int,
        default=2,
        help="VAE temporal decode chunk size (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed random seed for deterministic inference (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Execution device (cuda, mps, cpu)",
    )
    parser.add_argument(
        "--allow-non-cuda",
        action="store_true",
        help="Allow running on non-CUDA devices (e.g. MPS or CPU) for testing",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("renders/verify_902"),
        help="Output directory for generated MP4 video",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("reports/kaggle_svd_single_camera_run.json"),
        help="Output JSON execution report path",
    )
    return parser


def adapt_aspect_ratio_to_svd(
    image: Image.Image,
    target_width: int = 1024,
    target_height: int = 576,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Scale image preserving aspect ratio and pad to target_width x target_height.

    For KITScenes (876x568, aspect 1.54225) to SVD (1024x576, aspect 1.77778):
    1. Scale height to 576 -> scaled width = 888 (aspect preserved).
    2. Symmetrically pad 68 pixels on left and right with edge replication -> 1024x576.
    Returns:
        padded_image: PIL Image of shape (1024, 576)
        crop_box: (pad_left, 0, pad_left + scaled_w, target_height) for inverse cropping
    """
    orig_w, orig_h = image.size
    scale = target_height / orig_h
    scaled_w = int(round(orig_w * scale))

    # Resize preserving aspect ratio
    resized = image.resize((scaled_w, target_height), Image.Resampling.LANCZOS)
    arr = np.array(resized)

    # Calculate symmetric horizontal padding
    pad_total = target_width - scaled_w
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    # Edge reflection/replicate padding to avoid artificial black-border gradients
    padded_arr = np.pad(arr, ((0, 0), (pad_left, pad_right), (0, 0)), mode="edge")
    padded_image = Image.fromarray(padded_arr)

    crop_box = (pad_left, 0, pad_left + scaled_w, target_height)
    return padded_image, crop_box


def unpad_and_resize_frames(
    frames: list[Image.Image],
    crop_box: tuple[int, int, int, int],
    target_width: int = spec.WIDTH,
    target_height: int = spec.HEIGHT,
) -> np.ndarray:
    """Crop padding and resize frames back to exact KITScenes contract (876x568)."""
    processed = []
    for f in frames:
        # Crop away the side padding (returns 888x576)
        cropped = f.crop(crop_box)
        # Clean downscale to exact contract (876x568)
        resized = cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
        processed.append(np.array(resized, dtype=np.uint8))
    return np.stack(processed, axis=0)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    print("=" * 80)
    print("GENUINE PRETRAINED SVD INFERENCE RUNNER (SINGLE CAMERA & GPU)")
    print("=" * 80)
    print(f"Scenario:            {args.scenario_id}")
    print(f"Camera:              {args.camera}")
    print(f"Model ID:            {args.model_id}")
    print("Precision:           FP16 (torch.float16, variant='fp16')")
    print(f"Denoising Steps:     {args.steps} (Smoke Test)")
    print(f"Decode Chunk Size:   {args.decode_chunk_size}")
    print(f"Fixed Random Seed:   {args.seed}")
    print("Model Conditioning:  fps=5.0 (micro-conditioning parameter)")
    print(f"Contract Output:     {spec.WIDTH}x{spec.HEIGHT} (25 frames @ 5 FPS)")
    print(f"Target Device:       {args.device}")
    print("=" * 80)

    # 1. Dependency and Hardware Verification
    try:
        import torch
        from diffusers import StableVideoDiffusionPipeline
    except ImportError as e:
        print(f"FATAL: Missing dependencies: {e}")
        print("Install required packages: pip install torch diffusers transformers accelerate safetensors")
        return 1

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        if not args.allow_non_cuda:
            print("FATAL: CUDA is not available on this machine.")
            print("To run genuine GPU inference, execute this script inside your Kaggle GPU notebook.")
            print("(Pass --allow-non-cuda to force execution on CPU/MPS if intended for mock testing).")
            return 2
        device_name = "mps" if (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()) else "cpu"
        print(f"NOTICE: Falling back to '{device_name}' because --allow-non-cuda was specified.")

    gpu_metadata: Dict[str, Any] = {"device": device_name}
    if device_name == "cuda":
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_metadata.update({
            "gpu_name": gpu_props.name,
            "total_vram_mb": round(gpu_props.total_memory / (1024**2), 2),
            "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
            "cuda_version": torch.version.cuda,
        })
        print(f"Detected GPU 0: {gpu_props.name} ({gpu_metadata['total_vram_mb']} MB VRAM, CC {gpu_metadata['compute_capability']})")

    # 2. Load Conditioning Image & Planned Trajectory
    cond_path = Path("data/val_observed_t0") / args.scenario_id / f"{args.camera}.jpg"
    if not cond_path.exists():
        print(f"FATAL: Conditioning image not found at {cond_path}")
        return 3

    print(f"Loading conditioning image from {cond_path}...")
    pil_image_raw = Image.open(cond_path).convert("RGB")
    raw_conditioning_arr = np.array(
        pil_image_raw.resize((spec.WIDTH, spec.HEIGHT), Image.Resampling.LANCZOS), dtype=np.uint8
    )

    # Derive trajectory motion bucket
    val_traj_path = Path("data/val_trajectories.jsonl")
    train_traj_path = Path("data/train_trajectories.jsonl")
    past_pts = []
    instruction = "turn right"
    stype = "intersection"

    if val_traj_path.exists():
        with open(val_traj_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if str(rec["scenario_id"]).zfill(7) == args.scenario_id:
                    past_pts = rec["past"]
                    instruction = rec.get("driving_instruction", instruction)
                    stype = rec.get("scenario_type", stype)
                    break

    knn_planner = knn_trajectory.KNNTrajectoryPlanner(donor_path=train_traj_path if train_traj_path.exists() else None)
    plan_knn = knn_planner.plan(args.scenario_id, past_pts, instruction, scenario_type=stype)
    pts = np.asarray(plan_knn, dtype=float)
    total_dist = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0
    mean_speed = total_dist / (len(pts) * spec.WAYPOINT_STEP_S)
    motion_bucket = int(np.clip(10 + mean_speed * 12.0, 1, 255))
    print(f"Planned Trajectory: length={total_dist:.1f}m | avg_speed={mean_speed:.1f}m/s | SVD motion_bucket={motion_bucket}")

    # Aspect ratio preserving adaptation: 876x568 -> 888x576 -> 1024x576 padded
    padded_input, crop_box = adapt_aspect_ratio_to_svd(pil_image_raw, target_width=1024, target_height=576)
    print(f"Aspect Ratio Adapted: input padded to 1024x576 (crop region: {crop_box})")

    # 3. Load Pretrained Pipeline Weights
    print(f"\n[Step 1/3] Loading pretrained checkpoint '{args.model_id}' (variant='fp16')...")
    dtype = torch.float16 if device_name in ("cuda", "mps") else torch.float32
    init_t0 = time.perf_counter()

    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )

    # Apply strict T4 memory optimizations per specification
    if device_name == "cuda":
        # 1. Model CPU offload (keeps pipeline components in RAM until invoked)
        pipeline.enable_model_cpu_offload()
        # 2. Feed-forward chunking on UNet spatial-temporal layers
        if hasattr(pipeline.unet, "enable_forward_chunking"):
            pipeline.unet.enable_forward_chunking(chunk_size=1, dim=1)
            print("  UNet forward chunking enabled.")
        # 3. Memory-efficient attention
        if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                print("  xformers attention enabled.")
            except Exception:
                pass
    elif device_name == "mps":
        pipeline.to("mps")
    else:
        pipeline.to("cpu")

    init_duration = time.perf_counter() - init_t0
    print(f"Pretrained weights loaded successfully in {init_duration:.2f}s.")

    # 4. Attach Denoising Step Callback Counter
    step_executions = 0

    def denoising_step_counter(pipe, step_idx, timestep, callback_kwargs):
        nonlocal step_executions
        step_executions += 1
        return callback_kwargs

    if device_name == "cuda":
        torch.cuda.reset_peak_memory_stats(0)

    print(f"\n[Step 2/3] Executing genuine neural diffusion ({args.steps} steps @ 1024x576, fps=5)...")
    infer_t0 = time.perf_counter()

    generator = torch.Generator(device=device_name).manual_seed(args.seed) if device_name != "mps" else None

    # Live SVD Diffusion Inference with fps=5 micro-conditioning
    output = pipeline(
        padded_input,
        num_frames=spec.FRAME_COUNT,
        fps=int(spec.FPS),  # micro-conditioning input: 5 fps
        height=576,
        width=1024,
        motion_bucket_id=motion_bucket,
        num_inference_steps=args.steps,
        decode_chunk_size=args.decode_chunk_size,
        callback_on_step_end=denoising_step_counter,
        generator=generator,
    )

    infer_duration = time.perf_counter() - infer_t0
    print(f"Inference complete in {infer_duration:.2f}s ({infer_duration / spec.FRAME_COUNT:.2f}s per frame).")
    print(f"Denoising Step Callback Counter: {step_executions}/{args.steps} steps verified.")

    # Query real peak VRAM
    peak_alloc_mb = 0.0
    peak_reserved_mb = 0.0
    if device_name == "cuda":
        peak_alloc_mb = round(torch.cuda.max_memory_allocated(0) / (1024**2), 2)
        peak_reserved_mb = round(torch.cuda.max_memory_reserved(0) / (1024**2), 2)
        print(f"Peak VRAM Allocated: {peak_alloc_mb} MB | Peak VRAM Reserved: {peak_reserved_mb} MB")

    # 5. Output Processing: Unpad & Inverse Crop to Exact Contract
    print(f"\n[Step 3/3] Inverse-cropping padding and mapping to exact contract {spec.WIDTH}x{spec.HEIGHT}...")
    pil_frames = output.frames[0]
    raw_contract_frames = unpad_and_resize_frames(
        pil_frames,
        crop_box=crop_box,
        target_width=spec.WIDTH,
        target_height=spec.HEIGHT,
    )
    assert raw_contract_frames.shape == (25, spec.HEIGHT, spec.WIDTH, 3)
    assert raw_contract_frames.dtype == np.uint8

    # 6. RAW FRAME-TO-FRAME DIFFERENCE COMPUTATION (BEFORE MP4 ENCODING)
    # Compute step-to-step consecutive frame difference on uncompressed uint8 frames
    consecutive_diffs = [
        float(np.mean(np.abs(raw_contract_frames[t + 1].astype(float) - raw_contract_frames[t].astype(float))))
        for t in range(spec.FRAME_COUNT - 1)
    ]
    mean_step_diff_l1 = float(np.mean(consecutive_diffs))
    total_drift_l1_from_t0 = float(
        np.mean(np.abs(raw_contract_frames[-1].astype(float) - raw_conditioning_arr.astype(float)))
    )
    temporal_variance = float(np.mean(np.std(raw_contract_frames.astype(float), axis=0)))

    print("--- RAW UNCOMPRESSED TEMPORAL METRICS (PRE-ENCODING) ---")
    print(f"Mean Consecutive Frame Diff L1: {mean_step_diff_l1:.4f} (Static hold is exactly 0.0000)")
    print(f"Total Trajectory Drift L1:      {total_drift_l1_from_t0:.4f}")
    print(f"Temporal Standard Deviation:    {temporal_variance:.4f}")

    # 7. Video Encoding & Local Visual Quality Proxy
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = args.output_dir / f"svd_genuine_{args.camera}_steps{args.steps}.mp4"
    encoder = video_engine.VideoEncoder(crf=24)
    encoder.encode(raw_contract_frames, out_video_path, validate=True)
    print(f"Saved verified video to: {out_video_path}")

    pq_score = pq_proxy.heuristic_perceptual_score(raw_contract_frames)
    print(f"Local PQ_proxy (front camera): {pq_score:.2f} (NOTE: heuristic proxy, not official Q-Align)")

    # 8. Save Structured Audit Report
    report: Dict[str, Any] = {
        "execution_status": "SUCCESSFUL_PRETRAINED_INFERENCE",
        "scenario_id": args.scenario_id,
        "camera": args.camera,
        "model_id": args.model_id,
        "pipeline_class": pipeline.__class__.__name__,
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "hardware": gpu_metadata,
        "inference_parameters": {
            "num_frames": spec.FRAME_COUNT,
            "fps_micro_conditioning": int(spec.FPS),
            "sampling_resolution": "1024x576",
            "aspect_ratio_adaptation": f"888x576 padded to 1024x576, inverse cropped at {crop_box}",
            "output_resolution": f"{spec.WIDTH}x{spec.HEIGHT}",
            "denoising_steps_requested": args.steps,
            "denoising_steps_executed": step_executions,
            "decode_chunk_size": args.decode_chunk_size,
            "motion_bucket_id": motion_bucket,
            "precision": str(dtype),
            "seed": args.seed,
        },
        "real_hardware_measurements": {
            "model_initialization_s": round(init_duration, 2),
            "inference_duration_s": round(infer_duration, 2),
            "sec_per_frame": round(infer_duration / spec.FRAME_COUNT, 3),
            "peak_vram_allocated_mb": peak_alloc_mb,
            "peak_vram_reserved_mb": peak_reserved_mb,
        },
        "raw_uncompressed_temporal_metrics": {
            "mean_consecutive_step_diff_l1": round(mean_step_diff_l1, 4),
            "total_drift_l1_from_t0": round(total_drift_l1_from_t0, 4),
            "temporal_std": round(temporal_variance, 4),
            "consecutive_step_diffs": [round(d, 4) for d in consecutive_diffs],
        },
        "empirical_video_metrics": {
            "local_pq_proxy": round(pq_score, 2),
            "official_qalign_pq": "Requires private competition evaluation server",
        },
        "output_video_path": str(out_video_path),
    }

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved structured audit report: {args.report_out}")
    print("=" * 80)
    print("PRETRAINED SVD INFERENCE TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
