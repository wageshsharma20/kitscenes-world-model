#!/usr/bin/env python3
"""KITScenes Production SVD-XT Video Generation Pipeline (Strategy 1).

Executes the frozen project_baseline (kNN trajectory conditioning + SVD-XT motion bucket)
at 12 denoising steps across the partitioned test scenarios.

Strategy 1 Target Allocation (1,025 Genuine Videos across 3 Kaggle Shards):
  * Shard 1 (sangita_sh):   Front cameras for scenarios 0000000 - 0000349 (350 videos · ~29.5 hrs)
  * Shard 2 (wageshsharma): Front cameras for scenarios 0000350 - 0000399 (50 videos)
                            + Surround cameras for scenarios 0000000 - 0000054 (275 videos)
                            (Total: 325 videos · ~27.4 hrs)
  * Shard 3 (wageshsh):     Surround cameras for scenarios 0000055 - 0000124 (350 videos · ~29.5 hrs)

Production Constraints:
  1. Denoising Steps: 12 steps (SVD-XT Euler Discrete Scheduler).
  2. Zero Procedural Mixing: Only genuine SVD-XT diffusion videos are produced.
  3. Video Contract: 25 frames @ 5.0 fps, 876x568 resolution, H.264 MP4, < 50 MiB.
  4. Resumption: Any existing, format-valid MP4 on disk is reused immediately.
  5. Resilient Archiving: Updates /kaggle/working/shard{N}_videos.zip every 25 videos.

Usage on Kaggle:
  # Account 1:
  python scripts/run_production_svd.py --shard 1 --steps 12 --auto-zip

  # Account 2:
  python scripts/run_production_svd.py --shard 2 --steps 12 --auto-zip

  # Account 3:
  python scripts/run_production_svd.py --shard 3 --steps 12 --auto-zip
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from PIL import Image
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitscenes import knn_trajectory, spec
from kitscenes.spatial_adapter import (
    MockSVDUNet,
    create_default_euler_scheduler,
    decode_latents_temporal,
)
from kitscenes.svd_inference import (
    adapt_aspect_ratio_to_svd,
    crop_svd_array_to_kitscenes,
    encode_image_to_clip_embedding,
    encode_image_to_vae_latents,
    run_official_svd_diffusion_rollout,
)

CAMERA_SEED_OFFSETS = {
    "front": 0,
    "front_left": 100,
    "front_right": 200,
    "rear": 300,
    "rear_left": 400,
    "rear_right": 500,
}

SURROUND_CAMERAS = ["front_left", "front_right", "rear", "rear_left", "rear_right"]


def get_shard_workload(shard_id: int) -> List[Tuple[str, str]]:
    """Return list of (scenario_id, camera) tuples assigned to a shard."""
    workload: List[Tuple[str, str]] = []

    if shard_id == 1:
        # Shard 1: All 400 front cameras (0000000 to 0000399 = 400 videos)
        for i in range(0, 400):
            sid = str(i).zfill(7)
            workload.append((sid, "front"))

    elif shard_id == 2:
        # Shard 2: Surround cameras for first half of 125 scenarios (0000000 - 0000061: 62 x 5 = 310 videos)
        for i in range(0, 62):
            sid = str(i).zfill(7)
            for cam in SURROUND_CAMERAS:
                workload.append((sid, cam))

    elif shard_id == 3:
        # Shard 3: Surround cameras for second half of 125 scenarios (0000062 - 0000124: 63 x 5 = 315 videos)
        for i in range(62, 125):
            sid = str(i).zfill(7)
            for cam in SURROUND_CAMERAS:
                workload.append((sid, cam))

    else:
        raise ValueError(f"Invalid shard_id: {shard_id}. Expected 1, 2, or 3.")

    return workload


def is_valid_mp4(file_path: Path) -> bool:
    """Validate encoded MP4 format against challenge specification."""
    if not file_path.exists() or file_path.stat().st_size < 1000:
        return False
    try:
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            return False
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = round(cap.get(cv2.CAP_PROP_FPS), 1)
        cap.release()
        return (
            frame_count == spec.FRAME_COUNT
            and w == spec.WIDTH
            and h == spec.HEIGHT
            and abs(fps - float(spec.FPS)) < 0.2
        )
    except Exception:
        return False


def sync_to_kaggle_working(source_path: Path, dest_filename: Optional[str] = None) -> Optional[Path]:
    """Copy an artifact to /kaggle/working/ if executing inside Kaggle."""
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists() and kaggle_working.is_dir():
        try:
            target_name = dest_filename if dest_filename else source_path.name
            dest_path = kaggle_working / target_name
            if source_path.resolve() == dest_path.resolve():
                return dest_path
            shutil.copy2(source_path, dest_path)
            return dest_path
        except Exception as exc:
            print(f"[-] Warning: Failed to mirror {source_path.name} to /kaggle/working/: {exc}")
    return None


def package_zip_archive(files_or_dir: Sequence[Path] | Path, zip_dest: Path, arc_prefix: str = "") -> Path:
    """Create a ZIP archive containing specified files or directory contents without duplicates."""
    zip_dest.parent.mkdir(parents=True, exist_ok=True)
    seen_arcnames: Set[str] = set()
    with zipfile.ZipFile(zip_dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if isinstance(files_or_dir, Path) and files_or_dir.is_dir():
            for item in sorted(files_or_dir.rglob("*")):
                if item.is_file():
                    arcname = item.relative_to(files_or_dir)
                    if arc_prefix:
                        arcname = Path(arc_prefix) / arcname
                    arcname_str = str(arcname)
                    if arcname_str not in seen_arcnames:
                        seen_arcnames.add(arcname_str)
                        zf.write(item, arcname=arcname_str)
        else:
            for item in files_or_dir:
                if item.exists() and item.is_file():
                    arcname = item.name if not arc_prefix else f"{arc_prefix}/{item.name}"
                    if arcname not in seen_arcnames:
                        seen_arcnames.add(arcname)
                        zf.write(item, arcname=arcname)
    return zip_dest


def resolve_conditioning_image(
    sid: str,
    cam: str,
    observed_t0_dir: Path,
) -> Image.Image:
    """Load conditioning image strictly for ``sid`` and ``cam``; fail loudly if absent."""
    candidate_paths = [
        observed_t0_dir / sid / f"{cam}.jpg",
        observed_t0_dir / sid / f"{cam}.png",
        PROJECT_ROOT / "data" / "observed_t0" / sid / f"{cam}.jpg",
        PROJECT_ROOT / "data" / "observed_t0" / sid / f"{cam}.png",
        PROJECT_ROOT / "data" / "val_observed_t0" / sid / f"{cam}.jpg",
        PROJECT_ROOT / "data" / "val_observed_t0" / sid / f"{cam}.png",
        PROJECT_ROOT / "data" / "train_observed_t0" / sid / f"{cam}.jpg",
        PROJECT_ROOT / "data" / "train_observed_t0" / sid / f"{cam}.png",
    ]

    for p in candidate_paths:
        if p.exists() and p.stat().st_size > 500:
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                pass

    raise FileNotFoundError(
        f"Missing observed t=0 conditioning image for scenario={sid}, camera={cam} "
        f"(searched under {observed_t0_dir} and {PROJECT_ROOT / 'data'}). "
        "Extract real observed_t0 frames first; silent fallback is disabled."
    )


def run_production_pipeline(
    workload: List[Tuple[str, str]],
    shard_id: int,
    output_dir: Path,
    steps: int = 12,
    device_str: str = "cuda:0",
    mock_mode: bool = False,
    resume: bool = True,
    decode_chunk_size: int = 4,
    batch_zip_interval: int = 25,
    auto_zip: bool = True,
    manifest_name: Optional[str] = None,
    zip_name: Optional[str] = None,
) -> Path:
    """Execute production SVD-XT generation loop for the assigned shard."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / manifest_name if manifest_name else output_dir / f"shard{shard_id}_manifest.json"
    zip_dest = output_dir / zip_name if zip_name else output_dir / f"shard{shard_id}_videos.zip"

    if "cuda" in device_str and not torch.cuda.is_available():
        print(f"[-] WARNING: CUDA requested ({device_str}) but not available. Falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    print("\n" + "=" * 70)
    print(f"KITSCENES PRODUCTION SVD-XT GENERATION: SHARD {shard_id}")
    print(f"Assigned Workload: {len(workload)} videos")
    print(f"Denoising Steps:   {steps} steps (Pareto Efficiency Setting)")
    print(f"Target Device:     {device}")
    print(f"Auto-Zip Interval: Every {batch_zip_interval} videos")
    print("=" * 70 + "\n")

    # Load frozen kNN donor planner
    donor_path = PROJECT_ROOT / "data" / "train_trajectories.jsonl"
    knn_planner = knn_trajectory.KNNTrajectoryPlanner(
        donor_path=donor_path if donor_path.exists() else None
    )

    # Load test metadata for past trajectories and instructions
    test_meta: Dict[str, Any] = {}
    test_meta_path = PROJECT_ROOT / "data" / "test_metadata.jsonl"
    if test_meta_path.exists():
        with open(test_meta_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    test_meta[rec["scenario_id"]] = rec

    # Load official SVD-XT weights
    if mock_mode or device_str == "cpu":
        print("[+] Mock mode enabled: using MockSVDUNet")
        unet = MockSVDUNet().to(device)
        vae = None
        image_encoder = None
        feature_extractor = None
    else:
        print("[+] Loading official pretrained SVD-XT weights (fp16)...")
        from diffusers import AutoencoderKLTemporalDecoder
        from diffusers.models.unets.unet_spatio_temporal_condition import (
            UNetSpatioTemporalConditionModel,
        )
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            torch_dtype=torch.float16,
            variant="fp16",
            low_cpu_mem_usage=True,
        ).to(device)
        unet.eval()
        unet.enable_forward_chunking()

        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)
        vae.eval()

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            model_id,
            subfolder="image_encoder",
            torch_dtype=torch.float16,
            variant="fp16",
            low_cpu_mem_usage=True,
        ).to(device)
        image_encoder.eval()

        feature_extractor = CLIPImageProcessor.from_pretrained(
            model_id,
            subfolder="feature_extractor",
        )

    # Load existing manifest records if resuming
    manifest_records: List[Dict[str, Any]] = []
    already_done: Set[Tuple[str, str]] = set()
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest_records = json.load(f).get("predictions", [])
                for r in manifest_records:
                    p = Path(r["file_path"])
                    if p.exists() and is_valid_mp4(p):
                        already_done.add((r["scenario_id"], r["camera"]))
        except Exception:
            manifest_records = []

    print(f"[*] Resumption check: {len(already_done)}/{len(workload)} videos already verified on disk.\n")

    rendered_in_session = 0
    total_workload = len(workload)
    pipeline_start_t = time.perf_counter()

    for idx, (sid, cam) in enumerate(workload, start=1):
        out_filename = f"{sid}_{cam}.mp4"
        out_path = output_dir / out_filename

        # Resumption: skip if verified
        if resume and (sid, cam) in already_done and is_valid_mp4(out_path):
            print(f"[{idx:4d}/{total_workload:4d}] SKIP {out_filename} (already verified)")
            continue

        if resume and out_path.exists() and is_valid_mp4(out_path):
            print(f"[{idx:4d}/{total_workload:4d}] REUSE {out_filename} (valid on disk)")
            manifest_records.append({
                "scenario_id": sid,
                "camera": cam,
                "file_path": str(out_path),
                "file_size": out_path.stat().st_size,
                "steps": steps,
                "runtime_sec": None,
            })
            already_done.add((sid, cam))
            continue

        # Trajectory & motion bucket calculation
        s_meta = test_meta.get(sid, {})
        past = s_meta.get("trajectory", {}).get("past", [])
        inst = s_meta.get("driving_instruction", "")
        stype = s_meta.get("scenario_type", "")

        if len(past) >= 4:
            plan_future = knn_planner.plan(sid, past, inst, scenario_type=stype)
            speeds = np.linalg.norm(np.diff(plan_future, axis=0), axis=1) / 0.2
            mean_speed = float(np.mean(speeds))
        else:
            mean_speed = 8.0

        motion_bucket = int(np.clip(12.0 + (mean_speed / 25.0) * (255.0 - 12.0), 12, 255))

        # Load conditioning image
        cond_img = resolve_conditioning_image(
            sid=sid,
            cam=cam,
            observed_t0_dir=PROJECT_ROOT / "data" / "observed_t0",
        )

        # Encode conditioning inputs
        if not mock_mode and device_str != "cpu":
            torch.cuda.reset_peak_memory_stats(device)
            aspect_img, _ = adapt_aspect_ratio_to_svd(cond_img)
            clip_emb = encode_image_to_clip_embedding(image_encoder, feature_extractor, aspect_img, device)
            vae_latents = encode_image_to_vae_latents(vae, aspect_img, device)
        else:
            clip_emb = torch.randn(1, 1, 1024, device=device)
            vae_latents = torch.randn(1, 4, 72, 128, device=device)

        start_vid_t = time.perf_counter()
        print(f"[{idx:4d}/{total_workload:4d}] RENDERING {out_filename} (steps={steps}, bucket={motion_bucket}, speed={mean_speed:.1f} m/s)...")

        # Reproducible random seed per camera view
        seed = 42 + CAMERA_SEED_OFFSETS.get(cam, 0)
        generator = torch.Generator(device=device).manual_seed(seed)
        starting_noise = torch.randn(
            1, spec.FRAME_COUNT, 4, 72, 128,
            device=device,
            dtype=torch.float16 if not mock_mode else torch.float32,
            generator=generator,
        )

        scheduler = create_default_euler_scheduler()

        with torch.inference_mode():
            latents, _ = run_official_svd_diffusion_rollout(
                unet=unet,
                scheduler=scheduler,
                initial_latent=vae_latents,
                encoder_hidden_states=clip_emb,
                starting_noise=starting_noise,
                fps=spec.FPS,
                motion_bucket_id=motion_bucket,
                num_inference_steps=steps,
                do_classifier_free_guidance=True,
            )

            if not mock_mode and device_str != "cpu":
                frames_raw = decode_latents_temporal(vae, latents, decode_chunk_size=decode_chunk_size)
                frames_uint8 = ((frames_raw[0].permute(0, 2, 3, 1).cpu().float().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                peak_res = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
                del latents, frames_raw
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                frames_uint8 = np.full((spec.FRAME_COUNT, 576, 1024, 3), 120, dtype=np.uint8)
                peak_alloc = 0.0
                peak_res = 0.0

        canonical_frames = crop_svd_array_to_kitscenes(frames_uint8)
        runtime_sec = time.perf_counter() - start_vid_t
        rendered_in_session += 1

        # Write validated H.264 MP4
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(str(out_path), fourcc, float(spec.FPS), (spec.WIDTH, spec.HEIGHT))
        for frame in canonical_frames:
            out_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out_writer.release()

        assert is_valid_mp4(out_path), f"Encoded video {out_path} failed contract validation!"

        rec = {
            "scenario_id": sid,
            "camera": cam,
            "file_path": str(out_path),
            "file_size": out_path.stat().st_size,
            "steps": steps,
            "runtime_sec": round(runtime_sec, 2),
            "peak_allocated_mb": round(peak_alloc, 2),
            "peak_reserved_mb": round(peak_res, 2),
            "motion_bucket": motion_bucket,
            "mean_speed": round(mean_speed, 2),
        }
        manifest_records.append(rec)
        already_done.add((sid, cam))

        print(f"      -> DONE in {runtime_sec:.1f}s ({runtime_sec/60:.2f} min) | File: {out_path.stat().st_size / 1024:.1f} KB (Alloc: {peak_alloc:.1f} MB)")

        # Flush manifest incrementally
        with open(manifest_path, "w") as f:
            json.dump({"shard": shard_id, "predictions": manifest_records}, f, indent=2)
        sync_to_kaggle_working(manifest_path)

        # Periodic intermediate packaging to protect against session timeouts
        if auto_zip and (rendered_in_session % batch_zip_interval == 0 or idx == total_workload):
            valid_files = [Path(r["file_path"]) for r in manifest_records if Path(r["file_path"]).exists()]
            print(f"\n[*] Packaging intermediate Shard {shard_id} archive ({len(valid_files)} videos)...")
            package_zip_archive(valid_files + [manifest_path], zip_dest)
            sync_to_kaggle_working(zip_dest)
            print(f"[+] Synced {zip_dest.name} ({zip_dest.stat().st_size / (1024**2):.1f} MiB) to /kaggle/working/.\n")

    # Final archive creation
    if auto_zip:
        valid_files = [Path(r["file_path"]) for r in manifest_records if Path(r["file_path"]).exists()]
        package_zip_archive(valid_files + [manifest_path], zip_dest)
        sync_to_kaggle_working(zip_dest)

    total_elapsed = time.perf_counter() - pipeline_start_t
    print("\n" + "=" * 70)
    print(f"SHARD {shard_id} EXECUTION COMPLETE!")
    print(f"  * Total Verified Videos:   {len(already_done)}/{total_workload}")
    print(f"  * Videos Rendered This Run: {rendered_in_session}")
    print(f"  * Total Elapsed Time:      {total_elapsed / 3600:.2f} hours")
    if auto_zip:
        print(f"  * Shard Archive:           {zip_dest} ({zip_dest.stat().st_size / (1024**2):.1f} MiB)")
        print(f"  * Kaggle Working Copy:     /kaggle/working/{zip_dest.name}")
    print("=" * 70 + "\n")

    return zip_dest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="KITScenes Production SVD-XT Video Generation (Strategy 1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shard",
        type=int,
        required=True,
        choices=[1, 2, 3],
        help="Shard ID to run (1: Front 000-349; 2: Front 350-399 + Surround 000-054; 3: Surround 055-124)",
    )
    parser.add_argument("--steps", type=int, default=12, help="SVD-XT denoising steps (default: 12)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device")
    parser.add_argument("--mock", action="store_true", default=False, help="Enable mock pipeline for local CPU test")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume and reuse verified MP4s")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Disable resumption")
    parser.add_argument("--decode-chunk-size", type=int, default=4, help="VAE temporal decoder chunk size")
    parser.add_argument("--batch-zip-interval", type=int, default=25, help="Update intermediate zip every N videos")
    parser.add_argument("--auto-zip", action="store_true", default=True, help="Auto-package shard zip archive")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to render (for testing)")
    parser.add_argument("--start-idx", type=int, default=None, help="Start index in shard workload (0-based)")
    parser.add_argument("--end-idx", type=int, default=None, help="End index in shard workload (exclusive)")
    parser.add_argument("--zip-name", type=str, default=None, help="Custom zip archive filename")
    parser.add_argument("--manifest-name", type=str, default=None, help="Custom manifest JSON filename")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "renders" / "production_svd")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    workload = get_shard_workload(args.shard)
    if args.start_idx is not None or args.end_idx is not None:
        start = args.start_idx if args.start_idx is not None else 0
        end = args.end_idx if args.end_idx is not None else len(workload)
        workload = workload[start:end]
    elif args.limit and args.limit > 0:
        workload = workload[:args.limit]

    run_production_pipeline(
        workload=workload,
        shard_id=args.shard,
        output_dir=args.output_dir,
        steps=args.steps,
        device_str=args.device,
        mock_mode=args.mock,
        resume=args.resume,
        decode_chunk_size=args.decode_chunk_size,
        batch_zip_interval=args.batch_zip_interval,
        auto_zip=args.auto_zip,
        manifest_name=args.manifest_name,
        zip_name=args.zip_name,
    )


if __name__ == "__main__":
    main()
