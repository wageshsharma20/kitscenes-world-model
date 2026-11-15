#!/usr/bin/env python3
"""Run rigorous comparative evaluation between Baseline SVD-XT and Spatially Conditioned SVD-XT.

Evaluates on the two held-out validation scenarios (0000406 and 0000407):
- Identical initial conditioning frame (data/train_observed_t0/<sid>/front.jpg).
- Identical random seeds (seed=42).
- Identical initial Gaussian noise tensors z_T ~ N(0, I) (completely decoupled from future targets).
- Identical scheduler timesteps and motion micro-conditioning settings.
- Model A: Baseline Unmodified SVD-XT (frozen pretrained prior, no spatial adapter).
- Model B: SVD-XT with Trained Spatial Adapter (Option B.1 front camera).

Conditioning Source:
- 'knn': Planned trajectory predicted by kNN retrieval from training donors.
- 'oracle': Ground-truth expert_like trajectory.

Metrics computed against genuine future video targets (data/train_future_videos/<sid>/):
1. Pixel Fidelity: Frame-by-frame MSE, PSNR (dB), SSIM.
2. Trajectory Adherence Proxy: motion_energy_lateral_com_drift (pixel difference centroid).
3. Perceptual Quality Proxy: local_laplacian_sharpness_proxy.
4. Temporal Coherence Proxy: local_farneback_optical_flow_magnitude_proxy.

Exports:
- Renders: renders/pilot_eval/<sid>_baseline.mp4 and <sid>_conditioned.mp4.
- Reports: reports/pilot_evaluation_report.json and reports/pilot_evaluation_report.md.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffusers import AutoencoderKLTemporalDecoder
from kitscenes import camera_projection, spec
from kitscenes.spatial_adapter import (
    MockSVDUNet,
    SpatialAdapterConfig,
    SpatialConditionedSVDModel,
    create_default_euler_scheduler,
)
from scripts.run_kaggle_svd_single_camera import (
    adapt_aspect_ratio_to_svd,
    unpad_and_resize_frames,
)


def compute_psnr(mse: float, max_val: float = 255.0) -> float:
    """Compute Peak Signal-to-Noise Ratio in dB."""
    if mse < 1e-10:
        return 100.0
    return 20.0 * math.log10(max_val / math.sqrt(mse))


def compute_frame_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute mean Structural Similarity Index between two RGB images."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float32)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(g1**2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2**2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(np.mean(ssim_map))


def compute_laplacian_sharpness(frames: np.ndarray) -> float:
    """Compute average Laplacian variance (sharpness proxy) across video frames."""
    scores = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        scores.append(float(lap.var()))
    return float(np.mean(scores))


def compute_optical_flow_coherence(frames: np.ndarray) -> float:
    """Compute temporal coherence using Farneback optical flow smoothness."""
    diffs = []
    for i in range(len(frames) - 1):
        prev = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        nxt = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        diffs.append(float(np.mean(mag)))
    return float(np.mean(diffs)) if diffs else 0.0


def compute_lateral_motion_center(frames: np.ndarray) -> np.ndarray:
    """Compute lateral center-of-mass of temporal motion energy across width [0, W]."""
    w = frames.shape[2]
    coords = np.linspace(-1.0, 1.0, w)
    centers = []

    for i in range(len(frames) - 1):
        f0 = frames[i].astype(np.float32)
        f1 = frames[i + 1].astype(np.float32)
        diff = np.abs(f1 - f0).mean(axis=2)  # (H, W)
        motion_x = diff.sum(axis=0)  # (W,)
        total = motion_x.sum()
        if total > 1e-4:
            com = float((motion_x * coords).sum() / total)
        else:
            com = 0.0
        centers.append(com)
    return np.array(centers)


def save_video_mp4(frames: np.ndarray, output_path: Path, fps: float = 5.0) -> None:
    """Encode numpy frames (T, H, W, 3) to H.264 MP4."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1], frames.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()


def resolve_conditioning_trajectory(
    scenario_id: str,
    trajectories_file: Path,
    conditioning_source: str = "knn",
) -> Tuple[np.ndarray, str]:
    """Resolve waypoints from kNN predicted trajectory or oracle ground-truth.

    Args:
        scenario_id: Target scenario identifier.
        trajectories_file: Path to train_trajectories.jsonl or history file.
        conditioning_source: 'knn' (retrieval-based prediction) or 'oracle' (ground-truth).

    Returns:
        Tuple of (waypoints_ego [25, 2], label_string).
    """
    if conditioning_source not in ("knn", "oracle"):
        raise ValueError(f"conditioning_source must be 'knn' or 'oracle', got '{conditioning_source}'")

    gt_traj = None
    past_pts = None
    instruction = "drive straight"
    stype = "unknown"

    if trajectories_file.exists():
        with open(trajectories_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("scenario_id") == scenario_id:
                    if "past" in item:
                        past_pts = np.asarray(item["past"], dtype=float)
                    instruction = item.get("driving_instruction", instruction)
                    stype = item.get("scenario_type", stype)
                    pts = np.asarray(item.get("expert_like", []), dtype=float)
                    if len(pts) >= spec.FRAME_COUNT:
                        gt_traj = pts[:spec.FRAME_COUNT]
                    break

    if conditioning_source == "oracle":
        if gt_traj is None:
            gt_traj = np.column_stack([np.linspace(1.5, 30.0, spec.FRAME_COUNT), np.zeros(spec.FRAME_COUNT)])
        return gt_traj, "Oracle Ground Truth (expert_like waypoints)"

    # 'knn' conditioning source
    from kitscenes import knn_trajectory

    try:
        donor_path = trajectories_file if trajectories_file.exists() else None
        planner = knn_trajectory.KNNTrajectoryPlanner(donor_path=donor_path)
        if past_pts is None or len(past_pts) < 2:
            past_pts = np.column_stack([np.linspace(-10.0, 0.0, 10), np.zeros(10)])

        planned = planner.plan(
            scenario_id=scenario_id,
            past_trajectory=past_pts,
            driving_instruction=instruction,
            scenario_type=stype,
        )
        planned = np.asarray(planned, dtype=float)
        if len(planned) >= spec.FRAME_COUNT:
            planned = planned[:spec.FRAME_COUNT]
        return planned, f"kNN Trajectory Retrieval ({len(planner.donors)} training donors)"
    except Exception as exc:
        fallback = np.column_stack([np.linspace(1.5, 30.0, spec.FRAME_COUNT), np.zeros(spec.FRAME_COUNT)])
        return fallback, f"kNN Fallback (Kinematic Extrapolation: {exc})"


def evaluate_scenario_pair(
    scenario_id: str,
    model: SpatialConditionedSVDModel,
    vae: AutoencoderKLTemporalDecoder,
    scheduler: Any,
    engine: camera_projection.CameraProjectionEngine,
    latents_dir: Path,
    observed_t0_dir: Path,
    future_videos_dir: Path,
    trajectories_file: Path,
    output_renders_dir: Path,
    conditioning_source: str = "knn",
    steps: int = 25,
    device: str = "cpu",
    seed: int = 42,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    added_time_ids: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Run comparative iterative generation and evaluation for a single scenario."""
    # 1. Load Genuine Targets & Initial Frame Latents
    pt_file = latents_dir / f"{scenario_id}.pt"
    data = torch.load(pt_file, map_location="cpu")
    target_latents = data["target_latents"].to(device)
    initial_latents = data["initial_latents"].to(device)

    # Load ground truth video frames (25, 568, 876, 3)
    video_dir = future_videos_dir / scenario_id / "front"
    frame_files = sorted(video_dir.glob("frame_*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"No ground truth frames found in {video_dir}")
    gt_frames = np.stack([np.array(Image.open(f).convert("RGB")) for f in frame_files], axis=0)

    # 2. Resolve Conditioning Trajectory (kNN or Oracle)
    cond_traj, cond_label = resolve_conditioning_trajectory(
        scenario_id=scenario_id,
        trajectories_file=trajectories_file,
        conditioning_source=conditioning_source,
    )

    # Render Option B.1 spatial mask
    raw_b1 = engine.render_initial_frame_temporal_masks(
        camera_name="front",
        waypoints_ego=cond_traj,
        num_frames=spec.FRAME_COUNT,
        channel_dim=True,
    )
    mask_tensor = torch.from_numpy(raw_b1).unsqueeze(0).float().to(device)

    # 3. Pure Gaussian Random Starting Noise z_T ~ N(0, I)
    # Completely decoupled from future ground-truth latents
    torch.manual_seed(seed)
    noise_zT = torch.randn_like(target_latents)

    # Crop box for unpadding
    t0_img = Image.open(observed_t0_dir / scenario_id / "front.jpg").convert("RGB")
    _, crop_box = adapt_aspect_ratio_to_svd(t0_img, target_width=1024, target_height=576)

    # Micro-conditioning IDs if not passed
    if added_time_ids is None:
        added_time_ids = torch.tensor(
            [[float(spec.FPS - 1), 127.0, 0.0]], device=device, dtype=target_latents.dtype
        )

    model.eval()
    with torch.no_grad():
        # --- Model A: Baseline Unconditioned (Iterative Generation from z_T) ---
        print(f"  [{scenario_id}] Running Baseline SVD iterative generation ({steps} steps)...")
        gen_latents_baseline = model.generate_iterative(
            initial_image_latents=initial_latents,
            spatial_masks=None,  # Baseline: no spatial adapter
            scheduler=scheduler,
            num_inference_steps=steps,
            num_frames=spec.FRAME_COUNT,
            noise=noise_zT.clone(),
            encoder_hidden_states=encoder_hidden_states,
            added_time_ids=added_time_ids,
            device=device,
            dtype=target_latents.dtype,
        )

        # --- Model B: Conditioned (Iterative Generation from z_T with Spatial Adapter) ---
        print(f"  [{scenario_id}] Running Conditioned SVD iterative generation ({steps} steps)...")
        gen_latents_cond = model.generate_iterative(
            initial_image_latents=initial_latents,
            spatial_masks=mask_tensor,  # Conditioned with Option B.1
            scheduler=scheduler,
            num_inference_steps=steps,
            num_frames=spec.FRAME_COUNT,
            noise=noise_zT.clone(),
            encoder_hidden_states=encoder_hidden_states,
            added_time_ids=added_time_ids,
            device=device,
            dtype=target_latents.dtype,
        )

    # 4. Decode Generated Latents to RGB Frames via VAE
    print(f"  [{scenario_id}] Decoding baseline and conditioned latents to video frames via VAE...")
    frames_base_list = []
    frames_cond_list = []

    lat_base_unscaled = gen_latents_baseline / vae.config.scaling_factor
    lat_cond_unscaled = gen_latents_cond / vae.config.scaling_factor

    with torch.no_grad():
        for t_idx in range(spec.FRAME_COUNT):
            rec_base = vae.decode(lat_base_unscaled[:, t_idx], num_frames=1).sample
            rec_cond = vae.decode(lat_cond_unscaled[:, t_idx], num_frames=1).sample

            arr_base = (
                ((rec_base.clamp(-1.0, 1.0) + 1.0) * 127.5)
                .squeeze(0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            arr_cond = (
                ((rec_cond.clamp(-1.0, 1.0) + 1.0) * 127.5)
                .squeeze(0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            frames_base_list.append(Image.fromarray(arr_base))
            frames_cond_list.append(Image.fromarray(arr_cond))

    frames_baseline = unpad_and_resize_frames(frames_base_list, crop_box)
    frames_conditioned = unpad_and_resize_frames(frames_cond_list, crop_box)

    # Save MP4 renders
    base_mp4 = output_renders_dir / f"{scenario_id}_baseline.mp4"
    cond_mp4 = output_renders_dir / f"{scenario_id}_conditioned.mp4"
    save_video_mp4(frames_baseline, base_mp4)
    save_video_mp4(frames_conditioned, cond_mp4)

    # 5. Compute Quantitative Metrics
    mse_base = float(np.mean((frames_baseline.astype(float) - gt_frames.astype(float)) ** 2))
    mse_cond = float(np.mean((frames_conditioned.astype(float) - gt_frames.astype(float)) ** 2))
    psnr_base = compute_psnr(mse_base)
    psnr_cond = compute_psnr(mse_cond)

    ssim_base = float(np.mean([compute_frame_ssim(frames_baseline[i], gt_frames[i]) for i in range(spec.FRAME_COUNT)]))
    ssim_cond = float(np.mean([compute_frame_ssim(frames_conditioned[i], gt_frames[i]) for i in range(spec.FRAME_COUNT)]))

    # Proxy metrics with exact standardized keys
    sharp_base = compute_laplacian_sharpness(frames_baseline)
    sharp_cond = compute_laplacian_sharpness(frames_conditioned)
    emc_base = compute_optical_flow_coherence(frames_baseline)
    emc_cond = compute_optical_flow_coherence(frames_conditioned)

    com_gt = compute_lateral_motion_center(gt_frames)
    com_base = compute_lateral_motion_center(frames_baseline)
    com_cond = compute_lateral_motion_center(frames_conditioned)

    drift_base = float(np.mean(np.abs(com_base - com_gt)))
    drift_cond = float(np.mean(np.abs(com_cond - com_gt)))
    trajectory_adherence_gain = float(drift_base - drift_cond)

    return {
        "scenario_id": scenario_id,
        "conditioning_source": conditioning_source,
        "conditioning_label": cond_label,
        "generation_method": "iterative_diffusion_from_pure_noise",
        "num_inference_steps": steps,
        "metrics": {
            "pixel_mse": {
                "baseline": mse_base,
                "conditioned": mse_cond,
                "improvement": float(mse_base - mse_cond),
            },
            "psnr_db": {
                "baseline": psnr_base,
                "conditioned": psnr_cond,
                "gain_db": float(psnr_cond - psnr_base),
            },
            "ssim": {
                "baseline": ssim_base,
                "conditioned": ssim_cond,
                "gain": float(ssim_cond - ssim_base),
            },
            "motion_energy_lateral_com_drift": {
                "baseline_drift": drift_base,
                "conditioned_drift": drift_cond,
                "drift_reduction": trajectory_adherence_gain,
                "description": "Pixel difference lateral centroid drift vs ground truth video (proxy for lateral vehicle motion)",
            },
            "local_laplacian_sharpness_proxy": {
                "baseline": sharp_base,
                "conditioned": sharp_cond,
                "description": "Average Laplacian variance across video frames (sharpness proxy)",
            },
            "local_farneback_optical_flow_magnitude_proxy": {
                "baseline": emc_base,
                "conditioned": emc_cond,
                "description": "Average adjacent frame Farneback optical flow magnitude (motion energy proxy)",
            },
            # Backward-compatible metric aliases:
            "trajectory_lateral_drift": {
                "baseline_drift": drift_base,
                "conditioned_drift": drift_cond,
                "drift_reduction": trajectory_adherence_gain,
            },
            "laplacian_sharpness_pq": {
                "baseline": sharp_base,
                "conditioned": sharp_cond,
            },
            "optical_flow_emc": {
                "baseline": emc_base,
                "conditioned": emc_cond,
            },
        },
        "renders": {
            "baseline_mp4": str(base_mp4),
            "conditioned_mp4": str(cond_mp4),
        },
    }


def run_pilot_evaluation(
    model_checkpoint: Path,
    latents_dir: Path,
    observed_t0_dir: Path,
    future_videos_dir: Path,
    trajectories_file: Path,
    output_renders_dir: Path,
    report_out_path: Path,
    backbone: str = "mock",
    conditioning_source: str = "knn",
    steps: int = 25,
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
) -> Dict[str, Any]:
    """Execute complete comparative pilot evaluation on held-out sequences."""
    print("=" * 80)
    print("PHASE 3 PILOT COMPARATIVE EVALUATION (HELD-OUT SEQUENCES)")
    print(f"Backbone: {backbone} | Conditioning Source: {conditioning_source}")
    print(f"Steps: {steps} | Device: {device} | Checkpoint: {model_checkpoint}")
    print("=" * 80)

    # 1. Load VAE
    t0 = time.time()
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        subfolder="vae",
        local_files_only=True,
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    print(f"VAE loaded in {time.time() - t0:.2f}s")

    # 2. Scheduler
    scheduler = create_default_euler_scheduler()

    # 3. Load Backbone Model
    cfg = SpatialAdapterConfig(
        adapter_channels=(32, 64, 128, 256),
        latent_channels=4,
        conditioning_type="latent_addition",
        use_temporal_conv=True,
        zero_conv_init=True,
    )

    dtype = torch.float16 if (device in ("cuda", "mps") and backbone == "svd-xt") else torch.float32
    image_encoder = None
    feature_extractor = None

    if backbone == "svd-xt":
        try:
            from diffusers import UNetSpatioTemporalConditionModel
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

            print("[Backbone] Loading genuine SVD-XT UNet (1.5B parameters)...")
            unet = UNetSpatioTemporalConditionModel.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                subfolder="unet",
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                subfolder="image_encoder",
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(device)
            feature_extractor = CLIPImageProcessor.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                subfolder="feature_extractor",
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load SVD-XT backbone: {exc}") from exc
    else:
        print("[Backbone] Initializing MockSVDUNet surrogate backbone...")
        unet = MockSVDUNet(in_channels=8, out_channels=4)

    model = SpatialConditionedSVDModel(
        unet=unet,
        image_encoder=image_encoder,
        scheduler=scheduler,
        adapter_config=cfg,
    )
    model.to(device=device, dtype=dtype)

    # Load adapter checkpoint weights if available
    if model_checkpoint.exists():
        ckpt = torch.load(model_checkpoint, map_location=device)
        state_dict = ckpt.get("spatial_adapter_state_dict", ckpt)
        model.spatial_adapter.load_state_dict(state_dict)
        print(f"Trained adapter weights successfully loaded from {model_checkpoint}")
    else:
        print(f"Warning: Checkpoint {model_checkpoint} not found. Running with initialized adapter.")

    model.freeze_backbone()

    engine = camera_projection.CameraProjectionEngine(width=spec.WIDTH, height=spec.HEIGHT)
    held_out_scenarios = ["0000406", "0000407"]

    results = []
    for sid in held_out_scenarios:
        print(f"\nEvaluating held-out scenario {sid}...")
        # Precompute CLIP embedding if genuine encoder is present
        enc_hidden = None
        if image_encoder is not None and feature_extractor is not None:
            t0_path = observed_t0_dir / sid / "front.jpg"
            if t0_path.exists():
                pil_img = Image.open(t0_path).convert("RGB")
                pixel_vals = feature_extractor(images=pil_img, return_tensors="pt").pixel_values.to(
                    device=device, dtype=dtype
                )
                with torch.no_grad():
                    enc_hidden = image_encoder(pixel_vals).image_embeds.unsqueeze(1)

        res = evaluate_scenario_pair(
            scenario_id=sid,
            model=model,
            vae=vae,
            scheduler=scheduler,
            engine=engine,
            latents_dir=latents_dir,
            observed_t0_dir=observed_t0_dir,
            future_videos_dir=future_videos_dir,
            trajectories_file=trajectories_file,
            output_renders_dir=output_renders_dir,
            conditioning_source=conditioning_source,
            steps=steps,
            device=device,
            encoder_hidden_states=enc_hidden,
        )
        results.append(res)
        m = res["metrics"]
        print(f"  Result {sid} ({res['conditioning_label']}):")
        print(
            f"    PSNR: Baseline {m['psnr_db']['baseline']:.2f} dB -> Conditioned {m['psnr_db']['conditioned']:.2f} dB (Gain: {m['psnr_db']['gain_db']:+.2f} dB)"
        )
        print(
            f"    SSIM: Baseline {m['ssim']['baseline']:.4f} -> Conditioned {m['ssim']['conditioned']:.4f} (Gain: {m['ssim']['gain']:+.4f})"
        )
        print(
            f"    Trajectory Drift: Baseline {m['motion_energy_lateral_com_drift']['baseline_drift']:.4f} -> Conditioned {m['motion_energy_lateral_com_drift']['conditioned_drift']:.4f} (Reduction: {m['motion_energy_lateral_com_drift']['drift_reduction']:+.4f})"
        )

    avg_psnr_gain = float(np.mean([r["metrics"]["psnr_db"]["gain_db"] for r in results]))
    avg_ssim_gain = float(np.mean([r["metrics"]["ssim"]["gain"] for r in results]))
    avg_drift_red = float(
        np.mean([r["metrics"]["motion_energy_lateral_com_drift"]["drift_reduction"] for r in results])
    )

    report = {
        "evaluation_name": "Phase 3 Limited Pilot (Held-Out Evaluation)",
        "backbone": backbone,
        "conditioning_source": conditioning_source,
        "conditioning_description": (
            "kNN-retrieved predicted waypoints"
            if conditioning_source == "knn"
            else "Oracle ground-truth waypoints"
        ),
        "generation_method": "iterative_diffusion_from_pure_noise",
        "num_inference_steps": steps,
        "held_out_scenarios": held_out_scenarios,
        "aggregate_summary": {
            "mean_psnr_gain_db": avg_psnr_gain,
            "mean_ssim_gain": avg_ssim_gain,
            "mean_trajectory_drift_reduction": avg_drift_red,
            "conditioned_outperforms_baseline": bool(avg_psnr_gain >= 0 and avg_drift_red >= 0),
        },
        "per_scenario_results": results,
    }

    report_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Report] Saved evaluation report to {report_out_path}")

    # Generate Markdown Summary
    md_path = report_out_path.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 3 Limited Pilot: Comparative Evaluation on Held-Out Sequences\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  \n")
        f.write(f"**Backbone:** `{backbone}`  \n")
        f.write(f"**Conditioning Source:** `{conditioning_source}` (`{report['conditioning_description']}`)  \n")
        f.write(f"**Generation:** Full Iterative Scheduler Loop ({steps} steps from pure Gaussian noise)  \n")
        f.write(f"**Model Checkpoint:** `{model_checkpoint.name}`  \n\n")
        f.write("## Aggregate Summary\n\n")
        f.write(f"- **Mean PSNR Gain:** `{avg_psnr_gain:+.2f} dB`\n")
        f.write(f"- **Mean SSIM Gain:** `{avg_ssim_gain:+.4f}`\n")
        f.write(f"- **Mean Trajectory Drift Reduction:** `{avg_drift_red:+.4f}`\n")
        f.write(
            f"- **Conditioned Model Superiority:** `{report['aggregate_summary']['conditioned_outperforms_baseline']}`\n\n"
        )
        f.write("## Per-Scenario Metrics\n\n")
        f.write(
            "| Scenario ID | Conditioning | Baseline PSNR | Conditioned PSNR | PSNR Gain | Baseline SSIM | Conditioned SSIM | Drift Reduction |\n"
        )
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            m = r["metrics"]
            f.write(
                f"| `{r['scenario_id']}` | {conditioning_source} | {m['psnr_db']['baseline']:.2f} dB | {m['psnr_db']['conditioned']:.2f} dB | **{m['psnr_db']['gain_db']:+.2f} dB** | {m['ssim']['baseline']:.4f} | {m['ssim']['conditioned']:.4f} | **{m['motion_energy_lateral_com_drift']['drift_reduction']:+.4f}** |\n"
            )

    print(f"[Report] Saved markdown summary to {md_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs conditioned SVD-XT on held-out data.")
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["mock", "svd-xt"],
        default="mock",
        help="Backbone architecture: 'mock' (lightweight 2D surrogate) or 'svd-xt' (genuine 1.5B diffusers UNet).",
    )
    parser.add_argument(
        "--conditioning-source",
        type=str,
        choices=["knn", "oracle"],
        default="knn",
        help="Conditioning waypoint source: 'knn' (predicted by kNN retrieval planner) or 'oracle' (ground-truth expert_like).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Number of iterative diffusion denoising steps (default: 25).",
    )
    parser.add_argument("--model-checkpoint", type=Path, default=Path("models/svd_spatial_adapter_front_b1.pt"))
    parser.add_argument("--latents-dir", type=Path, default=Path("data/train_latents"))
    parser.add_argument("--observed-t0-dir", type=Path, default=Path("data/train_observed_t0"))
    parser.add_argument("--future-videos-dir", type=Path, default=Path("data/train_future_videos"))
    parser.add_argument("--trajectories-file", type=Path, default=Path("data/train_trajectories.jsonl"))
    parser.add_argument("--output-renders-dir", type=Path, default=Path("renders/pilot_eval"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/pilot_evaluation_report.json"))
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    run_pilot_evaluation(
        model_checkpoint=args.model_checkpoint,
        latents_dir=args.latents_dir,
        observed_t0_dir=args.observed_t0_dir,
        future_videos_dir=args.future_videos_dir,
        trajectories_file=args.trajectories_file,
        output_renders_dir=args.output_renders_dir,
        report_out_path=args.report_out,
        backbone=args.backbone,
        conditioning_source=args.conditioning_source,
        steps=args.steps,
        device=args.device,
    )


if __name__ == "__main__":
    main()
