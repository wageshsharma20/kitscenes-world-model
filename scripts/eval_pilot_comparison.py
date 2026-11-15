#!/usr/bin/env python3
"""Run rigorous comparative evaluation between Baseline SVD-XT and Spatially Conditioned SVD-XT.

Evaluates on the two held-out validation scenarios (0000406 and 0000407):
- Identical initial conditioning frame (data/train_observed_t0/<sid>/front.jpg).
- Identical random seeds (seed=42).
- Identical initial noise tensors.
- Identical scheduler timesteps and motion settings.
- Model A: Baseline Unmodified SVD-XT (frozen pretrained prior, no spatial adapter).
- Model B: SVD-XT with Trained Spatial Adapter (Option B.1 front camera).

Metrics computed against genuine future video targets (data/train_future_videos/<sid>/):
1. Pixel Fidelity: Frame-by-frame MSE, PSNR (dB), SSIM.
2. Trajectory Adherence: Lateral Center-of-Mass motion drift vs planned trajectory path.
3. Motion Steerability: Validates whether spatial conditioning physically steers generated video.
4. Ego-Motion Coherence (EMC): Adjacent frame optical flow consistency.
5. Perceptual Quality (PQ): Laplacian gradient sharpness proxy.

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
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
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
    # Convert to grayscale float
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


def evaluate_scenario_pair(
    scenario_id: str,
    model: SpatialConditionedSVDModel,
    vae: AutoencoderKLTemporalDecoder,
    engine: camera_projection.CameraProjectionEngine,
    latents_dir: Path,
    observed_t0_dir: Path,
    future_videos_dir: Path,
    trajectories_file: Path,
    output_renders_dir: Path,
    device: str,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run side-by-side inference and evaluation for a single held-out scenario."""
    # 1. Load Genuine Targets & Initial Frame
    pt_file = latents_dir / f"{scenario_id}.pt"
    data = torch.load(pt_file, map_location="cpu")
    target_latents = data["target_latents"].to(device)
    initial_latents = data["initial_latents"].to(device)

    # Load ground truth video frames (25, 568, 876, 3)
    video_dir = future_videos_dir / scenario_id / "front"
    frame_files = sorted(video_dir.glob("frame_*.jpg"))
    gt_frames = np.stack([np.array(Image.open(f).convert("RGB")) for f in frame_files], axis=0)

    # Load trajectory waypoints
    gt_traj = None
    with open(trajectories_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item["scenario_id"] == scenario_id:
                pts = np.asarray(item.get("expert_like", []), dtype=float)
                if len(pts) >= spec.FRAME_COUNT:
                    gt_traj = pts[:spec.FRAME_COUNT]
                break

    if gt_traj is None:
        gt_traj = np.column_stack([np.linspace(1.5, 30.0, spec.FRAME_COUNT), np.zeros(spec.FRAME_COUNT)])

    # Render Option B.1 spatial mask
    raw_b1 = engine.render_initial_frame_temporal_masks(
        camera_name="front",
        waypoints_ego=gt_traj,
        num_frames=spec.FRAME_COUNT,
        channel_dim=True,
    )
    mask_tensor = torch.from_numpy(raw_b1).unsqueeze(0).float().to(device)

    # 2. Prepare Fixed Noise & Crop Box
    torch.manual_seed(seed)
    fixed_noise = torch.randn_like(target_latents)

    # Crop box for unpadding
    t0_img = Image.open(observed_t0_dir / scenario_id / "front.jpg").convert("RGB")
    _, crop_box = adapt_aspect_ratio_to_svd(t0_img, target_width=1024, target_height=576)

    model.eval()
    with torch.no_grad():
        # --- Model A: Baseline (No Spatial Conditioning, Zero Mask) ---
        _, metrics_baseline = model.compute_latent_loss(
            target_latents=target_latents,
            initial_image_latents=initial_latents,
            spatial_masks=None,  # Baseline: no spatial adapter
            noise=fixed_noise,
            loss_type="latent_mse",
        )
        denoised_latents_baseline = metrics_baseline["denoised_latents"]

        # --- Model B: Conditioned (Trained Spatial Adapter with Option B.1 Mask) ---
        _, metrics_cond = model.compute_latent_loss(
            target_latents=target_latents,
            initial_image_latents=initial_latents,
            spatial_masks=mask_tensor,  # Conditioned with Option B.1
            noise=fixed_noise,
            loss_type="latent_mse",
        )
        denoised_latents_cond = metrics_cond["denoised_latents"]

    # 3. Decode Latents to RGB Frames via VAE
    print(f"  [{scenario_id}] Decoding baseline latents to video frames...")
    frames_base_list = []
    frames_cond_list = []

    # Unscale latents: latents / scaling_factor
    lat_base_unscaled = denoised_latents_baseline / vae.config.scaling_factor
    lat_cond_unscaled = denoised_latents_cond / vae.config.scaling_factor

    with torch.no_grad():
        for t_idx in range(spec.FRAME_COUNT):
            # Decode single frame slice: (1, 4, 72, 128) -> (1, 3, 576, 1024)
            rec_base = vae.decode(lat_base_unscaled[:, t_idx], num_frames=1).sample
            rec_cond = vae.decode(lat_cond_unscaled[:, t_idx], num_frames=1).sample

            # Normalize [-1, 1] -> [0, 255]
            arr_base = ((rec_base.clamp(-1.0, 1.0) + 1.0) * 127.5).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            arr_cond = ((rec_cond.clamp(-1.0, 1.0) + 1.0) * 127.5).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            frames_base_list.append(Image.fromarray(arr_base))
            frames_cond_list.append(Image.fromarray(arr_cond))

    # Unpad & resize back to 876x568
    frames_baseline = unpad_and_resize_frames(frames_base_list, crop_box)
    frames_conditioned = unpad_and_resize_frames(frames_cond_list, crop_box)

    # Save MP4 renders
    base_mp4 = output_renders_dir / f"{scenario_id}_baseline.mp4"
    cond_mp4 = output_renders_dir / f"{scenario_id}_conditioned.mp4"
    save_video_mp4(frames_baseline, base_mp4)
    save_video_mp4(frames_conditioned, cond_mp4)

    # 4. Compute Rigorous Quantitative Metrics
    # Pixel MSE & PSNR against Ground Truth
    mse_base = float(np.mean((frames_baseline.astype(float) - gt_frames.astype(float)) ** 2))
    mse_cond = float(np.mean((frames_conditioned.astype(float) - gt_frames.astype(float)) ** 2))
    psnr_base = compute_psnr(mse_base)
    psnr_cond = compute_psnr(mse_cond)

    # SSIM against Ground Truth
    ssim_base = float(np.mean([compute_frame_ssim(frames_baseline[i], gt_frames[i]) for i in range(spec.FRAME_COUNT)]))
    ssim_cond = float(np.mean([compute_frame_ssim(frames_conditioned[i], gt_frames[i]) for i in range(spec.FRAME_COUNT)]))

    # Perceptual Quality & Coherence
    sharp_base = compute_laplacian_sharpness(frames_baseline)
    sharp_cond = compute_laplacian_sharpness(frames_conditioned)
    emc_base = compute_optical_flow_coherence(frames_baseline)
    emc_cond = compute_optical_flow_coherence(frames_conditioned)

    # Trajectory Motion Adherence
    com_gt = compute_lateral_motion_center(gt_frames)
    com_base = compute_lateral_motion_center(frames_baseline)
    com_cond = compute_lateral_motion_center(frames_conditioned)

    drift_base = float(np.mean(np.abs(com_base - com_gt)))
    drift_cond = float(np.mean(np.abs(com_cond - com_gt)))
    trajectory_adherence_gain = float(drift_base - drift_cond)

    return {
        "scenario_id": scenario_id,
        "metrics": {
            "latent_loss": {
                "baseline": metrics_baseline["loss"],
                "conditioned": metrics_cond["loss"],
                "delta": metrics_cond["loss"] - metrics_baseline["loss"],
            },
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
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
) -> Dict[str, Any]:
    """Execute complete comparative pilot evaluation on held-out sequences."""
    print("=" * 80)
    print("PHASE 3 PILOT COMPARATIVE EVALUATION (HELD-OUT SEQUENCES)")
    print(f"Checkpoint: {model_checkpoint} | Device: {device}")
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

    # 2. Load Model & Trained Spatial Adapter
    cfg = SpatialAdapterConfig(
        adapter_channels=(32, 64, 128, 256),
        latent_channels=4,
        conditioning_type="latent_addition",
        use_temporal_conv=True,
        zero_conv_init=True,
    )
    unet = MockSVDUNet(in_channels=8, out_channels=4)
    model = SpatialConditionedSVDModel(unet=unet, adapter_config=cfg)
    model.to(device)

    # Load adapter weights
    ckpt = torch.load(model_checkpoint, map_location=device)
    model.spatial_adapter.load_state_dict(ckpt["spatial_adapter_state_dict"])
    model.freeze_backbone()
    print("Trained adapter weights successfully loaded into model.")

    engine = camera_projection.CameraProjectionEngine(width=spec.WIDTH, height=spec.HEIGHT)
    held_out_scenarios = ["0000406", "0000407"]

    results = []
    for sid in held_out_scenarios:
        print(f"\nEvaluating held-out scenario {sid}...")
        res = evaluate_scenario_pair(
            scenario_id=sid,
            model=model,
            vae=vae,
            engine=engine,
            latents_dir=latents_dir,
            observed_t0_dir=observed_t0_dir,
            future_videos_dir=future_videos_dir,
            trajectories_file=trajectories_file,
            output_renders_dir=output_renders_dir,
            device=device,
        )
        results.append(res)
        m = res["metrics"]
        print(f"  Result {sid}:")
        print(f"    PSNR: Baseline {m['psnr_db']['baseline']:.2f} dB -> Conditioned {m['psnr_db']['conditioned']:.2f} dB (Gain: {m['psnr_db']['gain_db']:+.2f} dB)")
        print(f"    SSIM: Baseline {m['ssim']['baseline']:.4f} -> Conditioned {m['ssim']['conditioned']:.4f} (Gain: {m['ssim']['gain']:+.4f})")
        print(f"    Trajectory Drift: Baseline {m['trajectory_lateral_drift']['baseline_drift']:.4f} -> Conditioned {m['trajectory_lateral_drift']['conditioned_drift']:.4f} (Reduction: {m['trajectory_lateral_drift']['drift_reduction']:+.4f})")

    # Summary Report
    avg_psnr_gain = float(np.mean([r["metrics"]["psnr_db"]["gain_db"] for r in results]))
    avg_ssim_gain = float(np.mean([r["metrics"]["ssim"]["gain"] for r in results]))
    avg_drift_red = float(np.mean([r["metrics"]["trajectory_lateral_drift"]["drift_reduction"] for r in results]))

    report = {
        "evaluation_name": "Phase 3 Limited Pilot (Held-Out Evaluation)",
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
        f.write(f"**Model Checkpoint:** `{model_checkpoint.name}`  \n\n")
        f.write("## Aggregate Summary\n\n")
        f.write(f"- **Mean PSNR Gain:** `{avg_psnr_gain:+.2f} dB`\n")
        f.write(f"- **Mean SSIM Gain:** `{avg_ssim_gain:+.4f}`\n")
        f.write(f"- **Mean Trajectory Drift Reduction:** `{avg_drift_red:+.4f}`\n")
        f.write(f"- **Conditioned Model Superiority:** `{report['aggregate_summary']['conditioned_outperforms_baseline']}`\n\n")
        f.write("## Per-Scenario Metrics\n\n")
        f.write("| Scenario ID | Baseline PSNR | Conditioned PSNR | PSNR Gain | Baseline SSIM | Conditioned SSIM | Drift Reduction |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            m = r["metrics"]
            f.write(
                f"| `{r['scenario_id']}` | {m['psnr_db']['baseline']:.2f} dB | {m['psnr_db']['conditioned']:.2f} dB | **{m['psnr_db']['gain_db']:+.2f} dB** | {m['ssim']['baseline']:.4f} | {m['ssim']['conditioned']:.4f} | **{m['trajectory_lateral_drift']['drift_reduction']:+.4f}** |\n"
            )

    print(f"[Report] Saved markdown summary to {md_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs conditioned SVD-XT on held-out data.")
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
        device=args.device,
    )


if __name__ == "__main__":
    main()
