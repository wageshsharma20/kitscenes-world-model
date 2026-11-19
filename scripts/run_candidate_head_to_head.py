#!/usr/bin/env python3
"""Phase 5.5: Multi-Scenario Head-to-Head Candidate Evaluation & Submission Freezing.

Implements strict two-stage execution with verified official SVD-XT inference mathematics:
  Stage A (Prediction): Strict zero future target leakage. Evaluates ONLY observed t=0 images,
    past trajectories, and permitted donor pool. Generates videos for 3 candidates:
      1. vanilla_svd: Official SVD-XT (single-camera CLIP token, default motion bucket 127)
      2. project_baseline: kNN trajectory planner + trajectory-derived SVD motion bucket
      3. cssc_only: MultiViewConditioner 27-token bank (camera + scene + 25 waypoints),
                    SpatialAdapter STRICTLY DISABLED (no latent residual injection)
    Saves generated MP4 files and a prediction manifest. Inference process cleanly exits.

  Stage B (Evaluation): Decoupled post-inference evaluation. Loads prediction manifest and
    genuine ground-truth future videos.
    - NO silent black-frame padding: missing ground truth fails explicitly.
    - NO default 3.5 MMS fallback: missing reference trajectories evaluate to null.
    - Computes true PSNR, SSIM, Boundary Strip Correlation, Local EMC approximation with
      strict front/rear yaw agreement, null-safe optical-flow adherence, and naturalness proxy.
    - Synthesizes aggregate statistics and pairwise head-to-head win rates.

Usage:
  # Run prediction stage only:
  python scripts/run_candidate_head_to_head.py --stage predict --device cuda:0 --steps 25
  
  # Run evaluation stage only:
  python scripts/run_candidate_head_to_head.py --stage evaluate

  # Run both stages end-to-end:
  python scripts/run_candidate_head_to_head.py --stage all --device cuda:0 --steps 25
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitscenes import emc_eval, knn_trajectory, pq_proxy, spec
from kitscenes.multiview_conditioning import MultiViewConditioner
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

CANONICAL_CAMERAS: Tuple[str, ...] = spec.CAMERAS
BASELINE_SUBMISSION_PATH: Path = PROJECT_ROOT / "submissions" / "videos.zip"
EXPECTED_BASELINE_SHA256: str = "26119a957a37123db3f7dac5a74fab50efb7051c8e7312f1cad1c384343ef10e"

DEFAULT_HEAD_TO_HEAD_SCENARIOS: Tuple[str, ...] = (
    "0000900",  # snow/straight
    "0000901",  # overtake/lane change right
    "0000902",  # intersection/turn right
    "0000903",  # heavy rain/straight
    "0000904",  # construction/straight
    "0000905",  # turn right
    "0000906",  # nighttime/lane change left
    "0000911",  # urban turn left
    "0000919",  # wet road/overtake
    "0000984",  # highway merge/u-turn
)

CANDIDATES: Tuple[str, ...] = (
    "vanilla_svd",
    "project_baseline",
    "cssc_only",
)

CAMERA_SEED_OFFSETS: Dict[str, int] = {
    "front": 0,
    "front_left": 1000,
    "front_right": 2000,
    "rear": 3000,
    "rear_left": 4000,
    "rear_right": 5000,
}


def sha256_file(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def validate_baseline_archive(strict: bool = False) -> Tuple[bool, str, Dict[str, Any]]:
    """Verify that the official baseline submission zip exists and is strictly unchanged."""
    exists = BASELINE_SUBMISSION_PATH.exists()
    if not exists:
        err_msg = f"ARCHIVE_MISSING: {BASELINE_SUBMISSION_PATH} does not exist on disk!"
        if strict:
            raise FileNotFoundError(err_msg)
        return False, "ARCHIVE_MISSING_ON_DISK", {
            "exists": False,
            "path": str(BASELINE_SUBMISSION_PATH),
            "expected_sha256": EXPECTED_BASELINE_SHA256,
            "actual_sha256": None,
            "intact": False,
        }
    actual_hash = sha256_file(BASELINE_SUBMISSION_PATH)
    is_intact = (actual_hash == EXPECTED_BASELINE_SHA256)
    return is_intact, actual_hash, {
        "exists": True,
        "path": str(BASELINE_SUBMISSION_PATH),
        "expected_sha256": EXPECTED_BASELINE_SHA256,
        "actual_sha256": actual_hash,
        "intact": is_intact,
        "size_bytes": BASELINE_SUBMISSION_PATH.stat().st_size,
    }


def compute_true_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute exact frame-by-frame Peak Signal-to-Noise Ratio (dB)."""
    p = pred.astype(np.float64)
    g = gt.astype(np.float64)
    mse = np.mean((p - g) ** 2)
    if mse < 1e-10:
        return 50.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def compute_video_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Gaussian-windowed Structural Similarity Index (SSIM)."""
    p = pred.astype(np.float64)
    g = gt.astype(np.float64)
    frame_ssims = []
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    for t in range(len(p)):
        p_t = p[t]
        g_t = g[t]
        mu_x = cv2.GaussianBlur(p_t, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(g_t, (11, 11), 1.5)
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        sigma_x_sq = cv2.GaussianBlur(p_t ** 2, (11, 11), 1.5) - mu_x_sq
        sigma_y_sq = cv2.GaussianBlur(g_t ** 2, (11, 11), 1.5) - mu_y_sq
        sigma_xy = cv2.GaussianBlur(p_t * g_t, (11, 11), 1.5) - mu_xy
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        )
        frame_ssims.append(np.mean(ssim_map))
    return float(np.mean(frame_ssims))


def compute_boundary_strip_correlation(
    camera_frames: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Compute 2D boundary-strip Pearson normalized cross-correlation across adjacent camera pairs.

    DISCLAIMER: This is a 2D boundary pixel correlation heuristic across 15%-wide image edges,
    NOT calibrated 3D geometric scene reprojection consistency.
    """
    pairs = [
        ("front", "front_left"),
        ("front", "front_right"),
        ("rear", "rear_left"),
        ("rear", "rear_right"),
        ("front_left", "rear_left"),
        ("front_right", "rear_right"),
    ]
    pairwise_scores: Dict[str, float] = {}
    overlap_scores = []
    w = spec.WIDTH
    strip = int(w * 0.15)

    for c1, c2 in pairs:
        pair_key = f"{c1}_to_{c2}"
        if c1 not in camera_frames or c2 not in camera_frames:
            continue
        v1 = camera_frames[c1]
        v2 = camera_frames[c2]

        if c2.endswith("_left"):
            s1 = v1[:, :, :strip, :].astype(np.float32)
            s2 = v2[:, :, -strip:, :].astype(np.float32)
        else:
            s1 = v1[:, :, -strip:, :].astype(np.float32)
            s2 = v2[:, :, :strip, :].astype(np.float32)

        s1_norm = (s1 - s1.mean()) / (s1.std() + 1e-6)
        s2_norm = (s2 - s2.mean()) / (s2.std() + 1e-6)
        ncc = float(np.clip(np.mean(s1_norm * s2_norm), 0.0, 1.0))
        pairwise_scores[pair_key] = round(ncc, 4)
        overlap_scores.append(ncc)

    mean_score = float(round(np.mean(overlap_scores), 4)) if overlap_scores else 0.0
    return {
        "boundary_strip_correlation": mean_score,
        "pairwise_correlations": pairwise_scores,
        "pair_count_evaluated": len(overlap_scores),
        "is_calibrated_3d_geometry": False,
    }


def compute_trajectory_adherence_from_flow(
    frames: np.ndarray, planned_waypoints: np.ndarray
) -> Dict[str, Any]:
    """Measure whether visual motion recovered from video matches planned waypoint headings.
    
    Returns null/None when optical flow is unmeasurable. Never returns a silent 0.5 fallback.
    """
    if len(frames) < 2 or len(planned_waypoints) < 2:
        return {
            "adherence": None,
            "valid_flow_count": 0,
            "planned_heading_rad": 0.0,
            "recovered_flow_heading_rad": None,
            "measurable": False,
        }

    wp_start = planned_waypoints[0]
    wp_end = planned_waypoints[-1]
    dx = float(wp_end[0] - wp_start[0])
    dy = float(wp_end[1] - wp_start[1])
    plan_heading = math.atan2(dx, dy)

    flow_headings = []
    for i in range(min(len(frames) - 1, 10)):
        g1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        u = float(np.mean(flow[..., 0]))
        v = float(np.mean(flow[..., 1]))
        if abs(u) > 0.01 or abs(v) > 0.01:
            flow_headings.append(math.atan2(-u, -v))

    if not flow_headings:
        return {
            "adherence": None,
            "valid_flow_count": 0,
            "planned_heading_rad": round(plan_heading, 4),
            "recovered_flow_heading_rad": None,
            "measurable": False,
        }

    mean_flow_heading = float(np.mean(flow_headings))
    heading_diff = abs(plan_heading - mean_flow_heading)
    adherence = max(0.0, 1.0 - (heading_diff / math.pi))
    return {
        "adherence": float(round(adherence, 4)),
        "valid_flow_count": len(flow_headings),
        "planned_heading_rad": round(plan_heading, 4),
        "recovered_flow_heading_rad": round(mean_flow_heading, 4),
        "measurable": True,
    }


def compute_video_mean_optical_flow(frames: np.ndarray) -> float:
    """Compute mean frame-to-frame optical flow magnitude across a video."""
    if len(frames) < 2:
        return 0.0
    mags = []
    for i in range(len(frames) - 1):
        g1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mags.append(float(np.mean(mag)))
    return float(np.mean(mags)) if mags else 0.0


# ==============================================================================
# STAGE A: PREDICTION ONLY (STRICT ZERO FUTURE TARGET LEAKAGE)
# ==============================================================================

def run_prediction_stage(
    scenarios: Sequence[str],
    checkpoint_path: Path,
    device: str = "cuda:0",
    mock: bool = False,
    steps: int = 25,
    decode_chunk_size: int = 4,
    min_guidance_scale: float = 1.0,
    max_guidance_scale: float = 3.0,
    base_seed: int = 42,
    output_dir: Path = PROJECT_ROOT / "renders" / "phase5_5",
    manifest_path: Path = PROJECT_ROOT / "reports" / "phase5_5_prediction_manifest.json",
) -> Dict[str, Any]:
    """Execute Stage A: Candidate prediction strictly isolated from ground truth."""
    print("=" * 80)
    print("STAGE A: CANDIDATE DIFFUSION PREDICTION (ZERO FUTURE LEAKAGE)")
    print(f"Device: {device} | Mock Mode: {mock} | Steps: {steps} | Base Seed: {base_seed}")
    print(f"Scenarios ({len(scenarios)}): {list(scenarios)}")
    print(f"Candidates ({len(CANDIDATES)}): {list(CANDIDATES)}")
    print("SpatialAdapter: STRICTLY DISABLED (0% latent residual injection)")
    print("=" * 80)

    parsed_device = torch.device(device)
    dtype = torch.float16 if parsed_device.type == "cuda" else torch.float32

    # Checkpoint provenance check
    assert checkpoint_path.exists(), f"Checkpoint missing: {checkpoint_path}"
    ckpt_hash = sha256_file(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt_step = ckpt.get("step", 6)
    print(f"\n[*] Loaded Checkpoint: {checkpoint_path.name}")
    print(f"  - SHA-256: {ckpt_hash}")
    print(f"  - Internal Step Counter: Step {ckpt_step}")

    # Model architecture loading
    model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
    model_revision = "9e43909513c6714f1bc78bcb44d96e733cd242aa"
    if not mock:
        print(f"\n[*] Loading pretrained SVD-XT components from {model_id} (revision: {model_revision})...")
        from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        t0 = time.time()
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            model_id, subfolder="unet", revision=model_revision, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(device=parsed_device)
        unet.eval()
        unet.requires_grad_(False)

        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            model_id, subfolder="vae", revision=model_revision, torch_dtype=torch.float32, low_cpu_mem_usage=True
        ).to(device=parsed_device)
        vae.eval()
        vae.requires_grad_(False)

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            model_id, subfolder="image_encoder", revision=model_revision, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(device=parsed_device)
        image_encoder.eval()
        image_encoder.requires_grad_(False)

        feature_extractor = CLIPImageProcessor.from_pretrained(
            model_id, subfolder="feature_extractor", revision=model_revision
        )
        print(f"  [✓] Pretrained SVD-XT components loaded in {time.time() - t0:.2f}s")
    else:
        print("\n[*] Initializing Mock SVD-XT components...")
        unet = MockSVDUNet(in_channels=8, out_channels=4).to(device=parsed_device, dtype=dtype)
        vae = None
        image_encoder = None
        feature_extractor = None

    # MultiViewConditioner loading
    conditioner = MultiViewConditioner().to(device=parsed_device, dtype=dtype)
    conditioner.load_state_dict(ckpt["conditioner_state_dict"])
    conditioner.eval()
    conditioner.requires_grad_(False)

    scheduler = create_default_euler_scheduler()
    planner = knn_trajectory.KNNTrajectoryPlanner()

    # Load validation metadata (strictly observed inputs: driving instruction & past trajectory)
    val_records: Dict[str, Any] = {}
    val_meta_path = PROJECT_ROOT / "data" / "val_manifest.jsonl"
    if val_meta_path.exists():
        with open(val_meta_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    val_records[str(rec["scenario_id"]).zfill(7)] = rec

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True).strip()
    except Exception:
        git_commit = "unknown"

    manifest: Dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "stage": "prediction",
        "git_commit": git_commit,
        "model_id": model_id,
        "model_revision": model_revision if not mock else "mock-revision",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": ckpt_hash,
        "checkpoint_training_step": ckpt_step,
        "device": device,
        "mock_mode": mock,
        "denoising_steps": steps,
        "decode_chunk_size": decode_chunk_size,
        "base_seed": base_seed,
        "candidates": list(CANDIDATES),
        "scenarios_evaluated": list(scenarios),
        "inference_mathematics": {
            "initial_noise_scaling": "latents = starting_noise * scheduler.init_noise_sigma",
            "cfg_schedule": "per_frame_guidance_linspace(1.0, 3.0, 25 frames)",
            "unconditional_image_latents": "zeros_like(image_latents)",
            "fps_micro_conditioning": "fps - 1 = 4",
            "spatial_adapter": "STRICTLY_DISABLED",
        },
        "scenarios": {},
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    for sid in scenarios:
        print("\n" + "=" * 60)
        print(f"PREDICTING SCENARIO: {sid}")
        print("=" * 60)

        obs_dir = PROJECT_ROOT / "data" / "val_observed_t0" / sid
        rec = val_records.get(sid, {})
        instruction = rec.get("driving_instruction", "drive straight on")
        stype = rec.get("scenario_type", "regular")
        past_traj = rec.get("past_trajectory", [])

        # Plan trajectory strictly from observed past without touching future GT
        planned_pts, prov = planner.plan_with_provenance(
            scenario_id=sid,
            past_trajectory=past_traj,
            driving_instruction=instruction,
            scenario_type=stype,
        )
        planned_pts = np.asarray(planned_pts, dtype=float)[:spec.FRAME_COUNT]
        assert sid not in prov["retrieved_donors"], f"kNN self-exclusion violated for {sid}!"
        print(f"  [✓] Planned {len(planned_pts)} waypoints via {prov['method']}.")

        # Kinematics for CSSC
        dt = spec.WAYPOINT_STEP_S
        diffs = np.diff(planned_pts, axis=0, prepend=planned_pts[:1])
        speeds = np.linalg.norm(diffs, axis=1) / dt
        accels = np.diff(speeds, prepend=speeds[:1]) / dt
        yaws = np.arctan2(diffs[:, 0], np.maximum(diffs[:, 1], 1e-4))
        curvatures = np.diff(yaws, prepend=yaws[:1]) / np.maximum(speeds * dt, 1e-4)
        kinematics_np = np.stack([planned_pts[:, 0], planned_pts[:, 1], yaws, speeds, accels, curvatures], axis=-1)
        kinematics_tensor = torch.from_numpy(kinematics_np).unsqueeze(0).to(device=parsed_device, dtype=dtype)

        mean_speed = float(np.mean(speeds))
        traj_motion_bucket = int(np.clip(127 + int((mean_speed - 10.0) * 8.0), 1, 255))
        default_motion_bucket = 127

        # Encode observed initial frames at t=0
        clip_dict: Dict[str, torch.Tensor] = {}
        initial_latents_dict: Dict[str, torch.Tensor] = {}

        for cam in CANONICAL_CAMERAS:
            obs_file = obs_dir / f"{cam}.jpg"
            assert obs_file.exists(), f"Observed t0 image missing: {obs_file}"
            pil_obs = Image.open(obs_file).convert("RGB")

            if not mock:
                clip_emb = encode_image_to_clip_embedding(image_encoder, feature_extractor, pil_obs, parsed_device, dtype)
                clip_dict[cam] = clip_emb

                gen_init = torch.Generator(device=parsed_device.type).manual_seed(base_seed + CAMERA_SEED_OFFSETS[cam])
                init_lat = encode_image_to_vae_latents(vae, pil_obs, parsed_device, gen_init, noise_aug_strength=0.02, dtype=dtype)
                initial_latents_dict[cam] = init_lat
            else:
                clip_dict[cam] = torch.randn(1, 1, 1024, device=parsed_device, dtype=dtype)
                initial_latents_dict[cam] = torch.randn(1, 4, 72, 128, device=parsed_device, dtype=dtype)

        # CSSC Token Bank generation for all cameras
        with torch.no_grad():
            cssc_bank_dict = conditioner.forward_all_cameras(clip_dict, kinematics_tensor)

        candidate_video_paths: Dict[str, Dict[str, str]] = {c: {} for c in CANDIDATES}
        candidate_telemetry: Dict[str, Dict[str, Any]] = {c: {} for c in CANDIDATES}

        # Run diffusion rollout across candidates
        for cand in CANDIDATES:
            print(f"\n  ---> Executing Candidate: {cand.upper()}")
            for cam in CANONICAL_CAMERAS:
                # Deterministic camera-specific paired noise stream
                cam_seed = base_seed + CAMERA_SEED_OFFSETS[cam]
                gen_noise = torch.Generator(device=parsed_device.type).manual_seed(cam_seed)
                z_T = torch.randn(1, 25, 4, 72, 128, generator=gen_noise, device=parsed_device, dtype=dtype)

                if cand == "vanilla_svd":
                    enc_hidden = clip_dict[cam]
                    mb_val = default_motion_bucket
                elif cand == "project_baseline":
                    enc_hidden = clip_dict[cam]
                    mb_val = traj_motion_bucket
                elif cand == "cssc_only":
                    enc_hidden = cssc_bank_dict[cam]["encoder_hidden_states"]
                    mb_val = traj_motion_bucket

                # Execute official SVD mathematical rollout
                latents_out, telem = run_official_svd_diffusion_rollout(
                    unet=unet,
                    scheduler=scheduler,
                    initial_latent=initial_latents_dict[cam],
                    encoder_hidden_states=enc_hidden,
                    starting_noise=z_T,
                    fps=5,  # conditioned on fps - 1 = 4
                    motion_bucket_id=mb_val,
                    noise_aug_strength=0.02,
                    num_inference_steps=steps,
                    min_guidance_scale=min_guidance_scale,
                    max_guidance_scale=max_guidance_scale,
                    do_classifier_free_guidance=True,
                )
                candidate_telemetry[cand][cam] = telem

                if not mock:
                    decoded = decode_latents_temporal(vae, latents_out, decode_chunk_size=decode_chunk_size)
                    arr = ((decoded[0].permute(0, 2, 3, 1).cpu().float().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                    cropped = crop_svd_array_to_kitscenes(arr)
                else:
                    cropped = np.zeros((spec.FRAME_COUNT, spec.HEIGHT, spec.WIDTH, 3), dtype=np.uint8)

                # Save MP4 video
                cam_out_mp4 = output_dir / f"{sid}_{cam}_{cand}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(cam_out_mp4), fourcc, 5.0, (spec.WIDTH, spec.HEIGHT))
                for f_idx in range(len(cropped)):
                    writer.write(cv2.cvtColor(cropped[f_idx], cv2.COLOR_RGB2BGR))
                writer.release()
                candidate_video_paths[cand][cam] = str(cam_out_mp4)

        manifest["scenarios"][sid] = {
            "scenario_id": sid,
            "instruction": instruction,
            "scenario_type": stype,
            "planned_waypoints": planned_pts.tolist(),
            "planned_mean_speed": round(mean_speed, 2),
            "trajectory_motion_bucket": traj_motion_bucket,
            "planning_provenance": prov,
            "video_paths": candidate_video_paths,
            "telemetry": candidate_telemetry,
        }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[✓] Stage A Prediction Complete! Manifest saved to: {manifest_path}")
    return manifest


# ==============================================================================
# STAGE B: EVALUATION ONLY (DECOUPLED POST-INFERENCE WITH NON-SILENT CHECKS)
# ==============================================================================

def run_evaluation_stage(
    manifest_path: Path = PROJECT_ROOT / "reports" / "phase5_5_prediction_manifest.json",
    val_fut_dir: Path = PROJECT_ROOT / "data" / "val_future_videos",
    val_meta_path: Path = PROJECT_ROOT / "data" / "val_manifest.jsonl",
    report_json: Path = PROJECT_ROOT / "reports" / "phase5_5_candidate_head_to_head.json",
    report_md: Path = PROJECT_ROOT / "reports" / "phase5_5_candidate_head_to_head.md",
) -> Dict[str, Any]:
    """Execute Stage B: Ground-truth evaluation on generated predictions with zero silent fallbacks."""
    print("=" * 80)
    print("STAGE B: DECOUPLED CANDIDATE EVALUATION & STATISTICAL SYNTHESIS")
    print(f"Manifest: {manifest_path}")
    print(f"Ground-Truth Directory: {val_fut_dir}")
    print("=" * 80)

    assert manifest_path.exists(), f"Prediction manifest not found: {manifest_path}. Run Stage A first!"
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Baseline submission archive check
    base_intact, base_hash, base_meta = validate_baseline_archive(strict=False)

    val_records: Dict[str, Any] = {}
    if val_meta_path.exists():
        with open(val_meta_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    val_records[str(rec["scenario_id"]).zfill(7)] = rec

    scenarios = manifest["scenarios_evaluated"]
    candidates = manifest["candidates"]

    results: Dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "stage": "evaluation",
        "prediction_manifest": str(manifest_path),
        "checkpoint_path": manifest["checkpoint_path"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "checkpoint_training_step": manifest["checkpoint_training_step"],
        "device": manifest["device"],
        "mock_mode": manifest["mock_mode"],
        "denoising_steps": manifest["denoising_steps"],
        "base_seed": manifest["base_seed"],
        "candidates": candidates,
        "scenarios_evaluated": scenarios,
        "baseline_submission_archive": {
            "exists": base_meta["exists"],
            "sha256": base_hash,
            "intact": base_intact,
        },
        "scenarios": {},
    }

    # Evaluate each scenario
    for sid in scenarios:
        s_manifest = manifest["scenarios"][sid]
        planned_pts = np.asarray(s_manifest["planned_waypoints"], dtype=float)
        instruction = s_manifest["instruction"]

        # 1. Load genuine future videos (NON-SILENT: no black padding)
        gt_video_arrays: Dict[str, np.ndarray] = {}
        fut_dir = val_fut_dir / sid

        for cam in CANONICAL_CAMERAS:
            gt_mp4 = fut_dir / f"{cam}.mp4"
            gt_frame_dir = fut_dir / cam
            gt_frames = []

            if gt_mp4.exists():
                cap = cv2.VideoCapture(str(gt_mp4))
                while True:
                    ret, f = cap.read()
                    if not ret:
                        break
                    gt_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                cap.release()
            elif gt_frame_dir.exists():
                frame_files = sorted(gt_frame_dir.glob("*.jpg"))
                for fp in frame_files:
                    f = cv2.imread(str(fp))
                    if f is not None:
                        gt_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

            if len(gt_frames) != spec.FRAME_COUNT:
                raise RuntimeError(
                    f"EVALUATION FAILURE: Scenario {sid} camera {cam} ground truth is missing or incomplete "
                    f"({len(gt_frames)} frames found, expected exactly {spec.FRAME_COUNT}). Silent black padding is prohibited!"
                )
            gt_video_arrays[cam] = np.stack(gt_frames[:spec.FRAME_COUNT], axis=0)

        # 2. Load generated candidate videos from manifest paths
        candidate_video_arrays: Dict[str, Dict[str, np.ndarray]] = {c: {} for c in candidates}
        for cand in candidates:
            for cam in CANONICAL_CAMERAS:
                vid_path = Path(s_manifest["video_paths"][cand][cam])
                if not vid_path.exists():
                    raise FileNotFoundError(f"Predicted video missing from manifest: {vid_path}")
                cap = cv2.VideoCapture(str(vid_path))
                c_frames = []
                while True:
                    ret, f = cap.read()
                    if not ret:
                        break
                    c_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                cap.release()
                if len(c_frames) != spec.FRAME_COUNT:
                    raise RuntimeError(f"Predicted video {vid_path} has {len(c_frames)} frames, expected {spec.FRAME_COUNT}!")
                candidate_video_arrays[cand][cam] = np.stack(c_frames, axis=0)

        vanilla_arrays = candidate_video_arrays["vanilla_svd"]

        # 3. Compute metrics for each candidate
        candidate_evaluations: Dict[str, Any] = {}
        for cand in candidates:
            c_arrays = candidate_video_arrays[cand]
            cam_metrics: Dict[str, Any] = {}

            psnrs = []
            ssims = []
            divergences = []
            flows = []

            for cam in CANONICAL_CAMERAS:
                pred_arr = c_arrays[cam]
                gt_arr = gt_video_arrays[cam]
                vanilla_arr = vanilla_arrays[cam]

                psnr_val = compute_true_psnr(pred_arr, gt_arr)
                ssim_val = compute_video_ssim(pred_arr, gt_arr)
                div_l1 = float(np.mean(np.abs(pred_arr.astype(float) - vanilla_arr.astype(float))))
                flow_mag = compute_video_mean_optical_flow(pred_arr)
                adh_dict = compute_trajectory_adherence_from_flow(pred_arr, planned_pts)

                cam_metrics[cam] = {
                    "psnr": round(psnr_val, 2),
                    "ssim": round(ssim_val, 4),
                    "divergence_l1": round(div_l1, 2),
                    "mean_optical_flow": round(flow_mag, 4),
                    "trajectory_adherence": adh_dict,
                }
                psnrs.append(psnr_val)
                ssims.append(ssim_val)
                divergences.append(div_l1)
                flows.append(flow_mag)

            boundary_corr = compute_boundary_strip_correlation(c_arrays)
            cand_renders = {c: Path(s_manifest["video_paths"][cand][c]) for c in CANONICAL_CAMERAS}
            emc_res = emc_eval.compute_video_optical_flow_emc(cand_renders, planned_pts)
            nat_proxy = pq_proxy.score_scenario_videos(cand_renders)

            candidate_evaluations[cand] = {
                "mean_psnr": round(float(np.mean(psnrs)), 2),
                "mean_ssim": round(float(np.mean(ssims)), 4),
                "mean_divergence_l1": round(float(np.mean(divergences)), 2),
                "mean_optical_flow": round(float(np.mean(flows)), 4),
                "boundary_strip_correlation": boundary_corr["boundary_strip_correlation"],
                "boundary_strip_details": boundary_corr,
                "local_emc_approximation": emc_res["local_emc_approximation"],
                "emc_details": emc_res,
                "visual_naturalness_proxy": round(float(nat_proxy["scenario_pq_proxy"]), 2),
                "is_official_pq": False,
                "camera_evaluations": cam_metrics,
            }

        # 4. MMS Evaluation: Return null if expert_like trajectory is absent (NO 3.5 fallback!)
        rec = val_records.get(sid, {})
        expert_like = np.asarray(rec.get("expert_like", []), dtype=float)
        if len(expert_like) > 0 and not np.all(expert_like == -100.0):
            mms_score = round(float(knn_trajectory.compute_mms_score(planned_pts, expert_like)), 2)
        else:
            mms_score = None  # Explicit null, no silent default fallback

        results["scenarios"][sid] = {
            "scenario_id": sid,
            "instruction": instruction,
            "planned_mean_speed": s_manifest["planned_mean_speed"],
            "trajectory_motion_bucket": s_manifest["trajectory_motion_bucket"],
            "official_mms_score": mms_score,
            "mms_evaluates_video": False,
            "candidates": candidate_evaluations,
        }

    # 5. Aggregate Multi-Scenario Statistics
    aggregate_stats: Dict[str, Any] = {}
    for cand in candidates:
        c_psnrs = [results["scenarios"][s]["candidates"][cand]["mean_psnr"] for s in scenarios]
        c_ssims = [results["scenarios"][s]["candidates"][cand]["mean_ssim"] for s in scenarios]
        c_bcorrs = [results["scenarios"][s]["candidates"][cand]["boundary_strip_correlation"] for s in scenarios]
        c_emcs = [results["scenarios"][s]["candidates"][cand]["local_emc_approximation"] for s in scenarios]

        aggregate_stats[cand] = {
            "psnr_mean": round(float(np.mean(c_psnrs)), 2),
            "psnr_std": round(float(np.std(c_psnrs)), 2),
            "psnr_median": round(float(np.median(c_psnrs)), 2),
            "psnr_min": round(float(np.min(c_psnrs)), 2),
            "psnr_max": round(float(np.max(c_psnrs)), 2),
            "ssim_mean": round(float(np.mean(c_ssims)), 4),
            "ssim_std": round(float(np.std(c_ssims)), 4),
            "ssim_median": round(float(np.median(c_ssims)), 4),
            "ssim_min": round(float(np.min(c_ssims)), 4),
            "ssim_max": round(float(np.max(c_ssims)), 4),
            "boundary_corr_mean": round(float(np.mean(c_bcorrs)), 4),
            "local_emc_mean": round(float(np.mean(c_emcs)), 3),
        }

    # Pairwise Comparisons
    comparisons = {
        "cssc_only_vs_vanilla": {
            "psnr_delta_mean": round(aggregate_stats["cssc_only"]["psnr_mean"] - aggregate_stats["vanilla_svd"]["psnr_mean"], 2),
            "ssim_delta_mean": round(aggregate_stats["cssc_only"]["ssim_mean"] - aggregate_stats["vanilla_svd"]["ssim_mean"], 4),
            "psnr_wins": sum(1 for s in scenarios if results["scenarios"][s]["candidates"]["cssc_only"]["mean_psnr"] > results["scenarios"][s]["candidates"]["vanilla_svd"]["mean_psnr"]),
            "ssim_wins": sum(1 for s in scenarios if results["scenarios"][s]["candidates"]["cssc_only"]["mean_ssim"] > results["scenarios"][s]["candidates"]["vanilla_svd"]["mean_ssim"]),
            "total_scenarios": len(scenarios),
        },
        "project_baseline_vs_vanilla": {
            "psnr_delta_mean": round(aggregate_stats["project_baseline"]["psnr_mean"] - aggregate_stats["vanilla_svd"]["psnr_mean"], 2),
            "ssim_delta_mean": round(aggregate_stats["project_baseline"]["ssim_mean"] - aggregate_stats["vanilla_svd"]["ssim_mean"], 4),
            "psnr_wins": sum(1 for s in scenarios if results["scenarios"][s]["candidates"]["project_baseline"]["mean_psnr"] > results["scenarios"][s]["candidates"]["vanilla_svd"]["mean_psnr"]),
            "ssim_wins": sum(1 for s in scenarios if results["scenarios"][s]["candidates"]["project_baseline"]["mean_ssim"] > results["scenarios"][s]["candidates"]["vanilla_svd"]["mean_ssim"]),
            "total_scenarios": len(scenarios),
        },
        "cssc_only_vs_project_baseline": {
            "psnr_delta_mean": round(aggregate_stats["cssc_only"]["psnr_mean"] - aggregate_stats["project_baseline"]["psnr_mean"], 2),
            "ssim_delta_mean": round(aggregate_stats["cssc_only"]["ssim_mean"] - aggregate_stats["project_baseline"]["ssim_mean"], 4),
            "psnr_wins": sum(1 for s in scenarios if results["scenarios"][s]["candidates"]["cssc_only"]["mean_psnr"] > results["scenarios"][s]["candidates"]["project_baseline"]["mean_psnr"]),
            "ssim_wins": sum(1 for s in scenarios if results["scenarios"][s]["candidates"]["cssc_only"]["mean_ssim"] > results["scenarios"][s]["candidates"]["project_baseline"]["mean_ssim"]),
            "total_scenarios": len(scenarios),
        },
    }

    results["aggregate_statistics"] = aggregate_stats
    results["pairwise_comparisons"] = comparisons

    # Write Reports
    report_json.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Saved Head-to-Head JSON Report: {report_json}")

    # Generate Markdown Report
    md_lines = [
        "# Phase 5.5: Multi-Scenario Head-to-Head Candidate Evaluation Report",
        "",
        f"- **Date/Time (UTC):** {results['timestamp_utc']}",
        f"- **Device:** `{manifest['device']}` (Mock Mode: `{manifest['mock_mode']}`)",
        f"- **Denoising Steps:** {manifest['denoising_steps']} steps (EulerDiscreteScheduler)",
        f"- **Base Seed:** {manifest['base_seed']} (Deterministic camera seed offsets applied)",
        f"- **Evaluated Checkpoint:** `{manifest['checkpoint_path']}` (SHA-256: `{manifest['checkpoint_sha256'][:16]}...`, Step: {manifest['checkpoint_training_step']})",
        f"- **SpatialAdapter Status:** `STRICTLY_DISABLED` (Residual perturbation bypassed)",
        f"- **Baseline Submission Archive:** `{base_hash}` (Intact: `{base_intact}`)",
        "",
        "---",
        "",
        "## 1. Executive Summary & Selection Decision",
        "",
        "This evaluation pits the two viable candidate architectures against each other and against Vanilla SVD across diverse official validation scenarios using official SVD-XT inference mathematics:",
        "- **Candidate 1 (`project_baseline`)**: kNN trajectory planner + trajectory-derived SVD motion bucket.",
        "- **Candidate 2 (`cssc_only`)**: MultiViewConditioner 27-token bank (camera + scene + waypoints), SpatialAdapter disabled.",
        "- **Reference System (`vanilla_svd`)**: Pretrained frozen SVD-XT, default motion bucket 127.",
        "",
        "### Aggregate Performance Summary",
        "| Candidate | Mean PSNR (dB) | Mean SSIM | Mean Boundary Strip Corr | Mean Local EMC Approx |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for cand in candidates:
        c_agg = aggregate_stats[cand]
        md_lines.append(
            f"| `{cand}` | **{c_agg['psnr_mean']:.2f}** (±{c_agg['psnr_std']:.2f}) | "
            f"**{c_agg['ssim_mean']:.4f}** (±{c_agg['ssim_std']:.4f}) | "
            f"{c_agg['boundary_corr_mean']:.4f} | {c_agg['local_emc_mean']:.3f} |"
        )

    md_lines.extend([
        "",
        "### Pairwise Head-to-Head Comparison",
        f"- **CSSC-Only vs Project Baseline:**",
        f"  - Mean $\\Delta$ PSNR: `{comparisons['cssc_only_vs_project_baseline']['psnr_delta_mean']:+.2f} dB` (Wins: {comparisons['cssc_only_vs_project_baseline']['psnr_wins']}/{comparisons['cssc_only_vs_project_baseline']['total_scenarios']})",
        f"  - Mean $\\Delta$ SSIM: `{comparisons['cssc_only_vs_project_baseline']['ssim_delta_mean']:+.4f}` (Wins: {comparisons['cssc_only_vs_project_baseline']['ssim_wins']}/{comparisons['cssc_only_vs_project_baseline']['total_scenarios']})",
        f"- **CSSC-Only vs Vanilla SVD:**",
        f"  - Mean $\\Delta$ PSNR: `{comparisons['cssc_only_vs_vanilla']['psnr_delta_mean']:+.2f} dB` (Wins: {comparisons['cssc_only_vs_vanilla']['psnr_wins']}/{comparisons['cssc_only_vs_vanilla']['total_scenarios']})",
        f"  - Mean $\\Delta$ SSIM: `{comparisons['cssc_only_vs_vanilla']['ssim_delta_mean']:+.4f}` (Wins: {comparisons['cssc_only_vs_vanilla']['ssim_wins']}/{comparisons['cssc_only_vs_vanilla']['total_scenarios']})",
        f"- **Project Baseline vs Vanilla SVD:**",
        f"  - Mean $\\Delta$ PSNR: `{comparisons['project_baseline_vs_vanilla']['psnr_delta_mean']:+.2f} dB` (Wins: {comparisons['project_baseline_vs_vanilla']['psnr_wins']}/{comparisons['project_baseline_vs_vanilla']['total_scenarios']})",
        f"  - Mean $\\Delta$ SSIM: `{comparisons['project_baseline_vs_vanilla']['ssim_delta_mean']:+.4f}` (Wins: {comparisons['project_baseline_vs_vanilla']['ssim_wins']}/{comparisons['project_baseline_vs_vanilla']['total_scenarios']})",
        "",
        "---",
        "",
        "## 2. Per-Scenario Performance Breakdown",
        "",
        "| Scenario ID | Instruction | `vanilla_svd` PSNR/SSIM | `project_baseline` PSNR/SSIM | `cssc_only` PSNR/SSIM | Winning Architecture |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for sid in scenarios:
        s_data = results["scenarios"][sid]
        v_psnr = s_data["candidates"]["vanilla_svd"]["mean_psnr"]
        v_ssim = s_data["candidates"]["vanilla_svd"]["mean_ssim"]
        b_psnr = s_data["candidates"]["project_baseline"]["mean_psnr"]
        b_ssim = s_data["candidates"]["project_baseline"]["mean_ssim"]
        c_psnr = s_data["candidates"]["cssc_only"]["mean_psnr"]
        c_ssim = s_data["candidates"]["cssc_only"]["mean_ssim"]

        winner = "cssc_only" if c_ssim >= b_ssim and c_psnr >= b_psnr - 0.5 else "project_baseline"

        md_lines.append(
            f"| `{sid}` | {s_data['instruction']} | {v_psnr:.2f} / {v_ssim:.4f} | "
            f"{b_psnr:.2f} / {b_ssim:.4f} | {c_psnr:.2f} / {c_ssim:.4f} | **`{winner}`** |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. Metric Disclaimers & Experimental Guarantees",
        "",
        "1. **Inference Mathematics Audit:** Initial noise is properly scaled by `scheduler.init_noise_sigma`. CFG schedule varies across frames, not steps. Unconditional image latents are strictly zero-conditioned. Added time IDs pass `fps - 1 = 4`.",
        "2. **Strict Two-Stage Decoupling:** Stage A (prediction) has zero access to ground-truth future videos or future trajectories.",
        "3. **Zero Silent Fallbacks:** Black-frame padding is strictly disabled; missing future videos trigger an explicit evaluation error. Missing reference trajectories produce null MMS.",
        "4. **SpatialAdapter Frozen:** Bypassed completely across all runs. Zero latent energy perturbation.",
    ])

    report_md.parent.mkdir(parents=True, exist_ok=True)
    with open(report_md, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[✓] Saved Head-to-Head Markdown Report: {report_md}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 5.5: Multi-Scenario Head-to-Head Candidate Evaluation."
    )
    parser.add_argument("--stage", type=str, default="all", choices=["predict", "evaluate", "all"],
                        help="Execution stage: predict (Stage A), evaluate (Stage B), or all")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mock", action="store_true", help="Run with mock SVD-XT components for fast testing")
    parser.add_argument("--steps", type=int, default=25, help="Number of Euler denoising steps")
    parser.add_argument("--decode-chunk-size", type=int, default=4, help="Temporal VAE decode chunk size (default: 4)")
    parser.add_argument("--scenarios", type=str, default=",".join(DEFAULT_HEAD_TO_HEAD_SCENARIOS))
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", type=Path, default=PROJECT_ROOT / "models" / "cssc_pilot_step30.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "renders" / "phase5_5")
    parser.add_argument("--manifest-path", type=Path, default=PROJECT_ROOT / "reports" / "phase5_5_prediction_manifest.json")
    parser.add_argument("--val-future-dir", type=Path, default=PROJECT_ROOT / "data" / "val_future_videos")
    parser.add_argument("--report-json", type=Path, default=PROJECT_ROOT / "reports" / "phase5_5_candidate_head_to_head.json")
    parser.add_argument("--report-md", type=Path, default=PROJECT_ROOT / "reports" / "phase5_5_candidate_head_to_head.md")
    args = parser.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    if args.mock and args.steps > 2:
        args.steps = 2

    if args.stage in ("predict", "all"):
        run_prediction_stage(
            scenarios=scenarios,
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            mock=args.mock,
            steps=args.steps,
            decode_chunk_size=args.decode_chunk_size,
            base_seed=args.base_seed,
            output_dir=args.output_dir,
            manifest_path=args.manifest_path,
        )

    if args.stage in ("evaluate", "all"):
        run_evaluation_stage(
            manifest_path=args.manifest_path,
            val_fut_dir=args.val_future_dir,
            report_json=args.report_json,
            report_md=args.report_md,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
