#!/usr/bin/env python3
"""Train the Spatial Conditioning Adapter for Stable Video Diffusion (SVD-XT) on genuine latents.

Backbone Options:
- 'mock': Lightweight 2D neural surrogate for fast CPU development, testing, and CI.
- 'svd-xt': Genuine 1.5B diffusers UNetSpatioTemporalConditionModel backbone with real
  CLIP vision encoder conditioning and micro-conditioning time IDs (requires GPU/high-VRAM).

Conditioning Source:
- 'knn': Planned trajectory retrieved via kNN from training donors (fail-loud if retrieval fails).
- 'oracle': Ground-truth expert_like trajectory from train_trajectories.jsonl.

Diffusion Noise Formulation:
- Continuous log-normal noise levels: log(sigma) ~ Normal(1.0, 1.6^2)
- Timestep mapping: c_noise = 0.25 * log(sigma)
- Continuous SVD v-prediction objective
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitscenes import camera_projection, spec
from kitscenes.knn_trajectory import KNNTrajectoryPlanner
from kitscenes.spatial_adapter import (
    MockSVDUNet,
    SpatialAdapterConfig,
    SpatialConditionedSVDModel,
    create_default_euler_scheduler,
    sample_training_noise_levels,
    sigma_to_c_noise,
)


def validate_device_and_dtype(
    device: str | torch.device,
    backbone: str,
) -> Tuple[torch.device, torch.dtype]:
    """Validate compute device and precision according to backbone safety contracts.

    Contracts:
    - If backbone == 'svd-xt':
      * Check torch.cuda.is_available(). If False, raise RuntimeError.
      * Parse device using torch.device. Device type must be 'cuda'.
      * Validate device index against torch.cuda.device_count().
      * Precision dtype is torch.float16.
    - If backbone == 'mock':
      * CPU, MPS, and CUDA are permitted.
      * Precision dtype is torch.float32.
    """
    device_obj = torch.device(device) if not isinstance(device, torch.device) else device
    if backbone == "svd-xt":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is strictly required for backbone='svd-xt'. CPU and MPS execution are prohibited to prevent OOM and system crashes."
            )
        if device_obj.type != "cuda":
            raise RuntimeError(
                f"Invalid device '{device}' for backbone='svd-xt'. Must be a CUDA device (e.g. 'cuda' or 'cuda:0')."
            )
        if device_obj.index is not None and device_obj.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device_obj.index} is out of bounds for available device count {torch.cuda.device_count()}."
            )
        dtype = torch.float16
    elif backbone == "mock":
        dtype = torch.float32
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    return device_obj, dtype


def evaluate_validation_loss(
    model: SpatialConditionedSVDModel,
    val_records: List[Dict[str, Any]],
    val_noises: List[List[torch.Tensor]],
    val_sigma_grid: List[float],
    scheduler: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[float, Dict[float, float]]:
    """Evaluate held-out validation loss across multiple reproducible noise realizations and sigma grid."""
    model.eval()
    all_losses: List[float] = []
    per_sigma_losses: Dict[float, List[float]] = {s: [] for s in val_sigma_grid}

    with torch.no_grad():
        for r, noises in zip(val_records, val_noises):
            for sigma_val in val_sigma_grid:
                sigmas = torch.full((1,), sigma_val, device=device, dtype=dtype)
                c_n = sigma_to_c_noise(sigmas)
                for n in noises:
                    loss_v, _ = model.compute_latent_loss(
                        target_latents=r["target_latents"],
                        initial_image_latents=r["initial_latents"],
                        spatial_masks=r["spatial_masks"],
                        noise=n,
                        sigmas=sigmas,
                        c_noise=c_n,
                        scheduler=scheduler,
                        encoder_hidden_states=r.get("encoder_hidden_states"),
                        added_time_ids=r.get("added_time_ids"),
                        loss_type="v_mse",
                    )
                    all_losses.append(loss_v.item())
                    per_sigma_losses[sigma_val].append(loss_v.item())

    mean_loss = float(np.mean(all_losses)) if all_losses else 0.0
    sigma_means = {s: float(np.mean(losses)) for s, losses in per_sigma_losses.items()}
    return mean_loss, sigma_means


def load_dataset_records(
    latents_dir: Path,
    observed_t0_dir: Path,
    data_path: Path,
    train_ids: List[str],
    val_ids: List[str],
    engine: camera_projection.CameraProjectionEngine,
    device: str,
    dtype: torch.dtype = torch.float32,
    image_encoder: Optional[torch.nn.Module] = None,
    feature_extractor: Optional[Any] = None,
    conditioning_source: str = "knn",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load paired genuine latents, trajectory masks, and conditioning for train and val splits."""
    scenario_metadata: Dict[str, Dict[str, Any]] = {}
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            sid = item["scenario_id"]
            if sid in train_ids or sid in val_ids:
                scenario_metadata[sid] = item

    planner = None
    if conditioning_source == "knn":
        planner = KNNTrajectoryPlanner(donor_path=data_path)
        if len(planner.donors) == 0:
            raise RuntimeError(f"kNN Trajectory Planner failed to load any donors from {data_path}!")

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []

    # SVD micro-conditioning: fps - 1, motion_bucket_id=127, noise_aug_strength=0.02
    fps_micro = float(spec.FPS - 1)
    motion_bucket = 127.0
    aug_level = 0.02
    added_time_ids = torch.tensor([[fps_micro, motion_bucket, aug_level]], device=device, dtype=dtype)

    for sid in train_ids + val_ids:
        pt_file = latents_dir / f"{sid}.pt"
        if not pt_file.exists():
            raise FileNotFoundError(f"Latent file missing for scenario {sid}: {pt_file}")

        data = torch.load(pt_file, map_location="cpu")
        target_latents = data["target_latents"].to(device=device, dtype=dtype)
        initial_latents = data["initial_latents"].to(device=device, dtype=dtype)

        meta = scenario_metadata.get(sid)
        if meta is None:
            raise RuntimeError(f"Metadata for scenario {sid} not found in {data_path}!")

        if conditioning_source == "oracle":
            pts = meta.get("expert_like")
            if pts is None or len(pts) < spec.FRAME_COUNT:
                raise RuntimeError(f"Oracle trajectory missing or incomplete for scenario {sid}!")
            cond_traj = np.asarray(pts[:spec.FRAME_COUNT], dtype=float)
            cond_label = "Oracle Ground Truth (expert_like)"
        elif conditioning_source == "knn":
            past = meta.get("past")
            if past is None or len(past) < 2:
                raise RuntimeError(f"Past trajectory missing or insufficient for scenario {sid}!")
            instr = meta.get("driving_instruction", "drive straight")
            stype = meta.get("scenario_type", "")
            try:
                planned = planner.plan(
                    scenario_id=sid,
                    past_trajectory=past,
                    driving_instruction=instr,
                    scenario_type=stype,
                )
            except Exception as exc:
                raise RuntimeError(f"kNN donor retrieval failed for scenario {sid}: {exc}") from exc
            if planned is None or len(planned) < spec.FRAME_COUNT:
                raise RuntimeError(f"kNN planner returned invalid trajectory for scenario {sid}!")
            cond_traj = np.asarray(planned[:spec.FRAME_COUNT], dtype=float)
            cond_label = f"kNN Trajectory ({len(planner.donors)} donors)"
        else:
            raise ValueError(f"Invalid conditioning_source: {conditioning_source}")

        # Render Option B.1 spatial mask
        raw_b1 = engine.render_initial_frame_temporal_masks(
            camera_name="front",
            waypoints_ego=cond_traj,
            num_frames=spec.FRAME_COUNT,
            channel_dim=True,
        )

        mask_tensor = torch.from_numpy(raw_b1).unsqueeze(0).to(device=device, dtype=dtype)

        # Generate real CLIP embeddings if encoder and feature extractor provided
        if image_encoder is not None and feature_extractor is not None:
            t0_file = observed_t0_dir / sid / "front.jpg"
            if t0_file.exists():
                pil_img = Image.open(t0_file).convert("RGB")
                pixel_vals = feature_extractor(images=pil_img, return_tensors="pt").pixel_values.to(
                    device=device, dtype=dtype
                )
                with torch.no_grad():
                    encoder_hidden_states = image_encoder(pixel_vals).image_embeds.unsqueeze(1)
            else:
                encoder_hidden_states = torch.zeros(1, 1, 1024, device=device, dtype=dtype)
        else:
            encoder_hidden_states = torch.zeros(1, 1, 1024, device=device, dtype=dtype)

        record = {
            "scenario_id": sid,
            "target_latents": target_latents,
            "initial_latents": initial_latents,
            "spatial_masks": mask_tensor,
            "cond_trajectory": cond_traj,
            "cond_label": cond_label,
            "encoder_hidden_states": encoder_hidden_states,
            "added_time_ids": added_time_ids,
        }

        if sid in train_ids:
            train_records.append(record)
        else:
            val_records.append(record)

    return train_records, val_records


def train_pilot_adapter(
    latents_dir: Path,
    data_path: Path,
    output_model_path: Path,
    report_out_path: Path,
    observed_t0_dir: Path = Path("data/train_observed_t0"),
    backbone: str = "mock",
    conditioning_source: str = "knn",
    iterations: int = 25,
    lr: float = 1e-4,
    device: str | torch.device = "mps" if torch.backends.mps.is_available() else "cpu",
    train_ids: Optional[List[str]] = None,
    val_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Execute SVD-XT spatial adapter training pilot on genuine latents."""
    # 0. Strict device and precision safety validation
    device_obj, dtype = validate_device_and_dtype(device, backbone)

    title = f"Phase 3 Pilot ({'Mock Surrogate' if backbone == 'mock' else 'Genuine SVD-XT 1.5B'})"
    print("=" * 80)
    print(title.upper())
    print(f"Backbone: {backbone} | Conditioning Source: {conditioning_source} | Device: {device_obj} | Dtype: {dtype}")
    print(f"Iterations: {iterations} | Learning Rate: {lr}")
    if backbone == "mock":
        print("NOTE: 'mock' is a lightweight 2D neural surrogate for fast verification without 9GB SVD weights.")
    else:
        print("NOTE: 'svd-xt' loads the genuine 1.5B diffusers UNetSpatioTemporalConditionModel backbone.")
    print("=" * 80)

    if train_ids is None:
        train_ids = ["0000400", "0000401", "0000402", "0000403", "0000404", "0000405"]
    if val_ids is None:
        val_ids = ["0000406", "0000407"]

    print(f"Training split ({len(train_ids)} sequences): {train_ids}")
    print(f"Held-out validation split ({len(val_ids)} sequences): {val_ids}")

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
            print("[Backbone] Loading CLIP Vision Model and Feature Extractor...")
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                subfolder="image_encoder",
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(device_obj)
            feature_extractor = CLIPImageProcessor.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                subfolder="feature_extractor",
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load genuine SVD-XT components for backbone 'svd-xt': {exc}. "
                "Ensure huggingface weights are accessible or use '--backbone mock'."
            ) from exc
    else:
        print("[Backbone] Initializing MockSVDUNet surrogate backbone...")
        unet = MockSVDUNet(in_channels=8, out_channels=4)

    scheduler = create_default_euler_scheduler()

    engine = camera_projection.CameraProjectionEngine(width=spec.WIDTH, height=spec.HEIGHT)
    train_records, val_records = load_dataset_records(
        latents_dir=latents_dir,
        observed_t0_dir=observed_t0_dir,
        data_path=data_path,
        train_ids=train_ids,
        val_ids=val_ids,
        engine=engine,
        device=str(device_obj),
        dtype=dtype,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
        conditioning_source=conditioning_source,
    )

    train_sequences_count = len(train_records)
    optimizer_updates_per_iteration = len(train_records)
    total_optimizer_updates = iterations * len(train_records)

    print(f"Successfully loaded {train_sequences_count} training records and {len(val_records)} validation records.")
    print(f"Optimizer updates per iteration: {optimizer_updates_per_iteration} | Total updates: {total_optimizer_updates}")

    cfg = SpatialAdapterConfig(
        adapter_channels=(32, 64, 128, 256),
        latent_channels=4,
        conditioning_type="latent_addition",
        use_temporal_conv=True,
        zero_conv_init=True,
    )

    model = SpatialConditionedSVDModel(
        unet=unet,
        image_encoder=image_encoder,
        scheduler=scheduler,
        adapter_config=cfg,
    )
    model.to(device=device_obj, dtype=dtype)
    model.freeze_backbone()

    audit = model.count_parameters()
    print(f"[Model] Trainable Adapter Parameters: {audit['trainable_adapter_params']:,}")
    print(f"[Model] Frozen Backbone Parameters:    {audit['frozen_backbone_params']:,}")
    print(f"[Model] Parameter Budget Compliant:   {audit['adapter_budget_compliant']} (< 10M)")

    assert audit["adapter_budget_compliant"], "Adapter exceeds 10M parameter limit!"
    assert audit["frozen_backbone_params"] > 0, "Backbone must have frozen parameters!"
    assert all(not p.requires_grad for p in model.unet.parameters()), "All UNet parameters must be frozen!"
    assert all(p.requires_grad for p in model.spatial_adapter.parameters()), "All adapter parameters must be trainable!"

    optimizer = torch.optim.AdamW(model.spatial_adapter.parameters(), lr=lr, weight_decay=1e-4)

    # Multi-level reproducible validation noise across sigma grid [1.0, 15.0, 70.0]
    val_sigma_grid = [1.0, 15.0, 70.0]
    val_rng = torch.Generator(device="cpu").manual_seed(42)
    val_noise_realizations = 2
    val_noises = [
        [
            torch.randn(
                r["target_latents"].shape,
                generator=val_rng,
            ).to(device=r["target_latents"].device, dtype=r["target_latents"].dtype)
            for _ in range(val_noise_realizations)
        ]
        for r in val_records
    ]

    # Pre-Training Step 0 Baseline Validation Loss across sigma grid
    initial_val_loss, initial_val_per_sigma = evaluate_validation_loss(
        model=model,
        val_records=val_records,
        val_noises=val_noises,
        val_sigma_grid=val_sigma_grid,
        scheduler=scheduler,
        device=device_obj,
        dtype=dtype,
    )
    print(
        f"[Pre-Training Step 0 Baseline] Held-Out Validation Loss: {initial_val_loss:.6f} "
        f"(sigmas {val_sigma_grid}: {[round(initial_val_per_sigma[s], 6) for s in val_sigma_grid]})"
    )

    t_start = time.time()
    train_loss_history: List[float] = []
    val_loss_history: List[float] = [initial_val_loss]

    print("\n[Training Loop] Optimizing Spatial Adapter weights on genuine latents...")
    for it in range(iterations):
        model.train()
        it_train_losses = []

        # Blocker 4: Sample fresh Gaussian noise for each training example at each optimizer step
        for r in train_records:
            optimizer.zero_grad()
            noise = torch.randn_like(
                r["target_latents"], device=r["target_latents"].device, dtype=r["target_latents"].dtype
            )

            # Continuous training noise sampling: log(sigma) ~ Normal(1.0, 1.6^2)
            sigmas, c_n = sample_training_noise_levels(1, loc=1.0, scale=1.6, device=device_obj, dtype=dtype)

            loss, _ = model.compute_latent_loss(
                target_latents=r["target_latents"],
                initial_image_latents=r["initial_latents"],
                spatial_masks=r["spatial_masks"],
                noise=noise,
                sigmas=sigmas,
                c_noise=c_n,
                scheduler=scheduler,
                encoder_hidden_states=r.get("encoder_hidden_states"),
                added_time_ids=r.get("added_time_ids"),
                loss_type="v_mse",
            )
            loss.backward()

            for name, p in model.unet.named_parameters():
                assert p.grad is None, f"VIOLATION: Frozen backbone parameter '{name}' received gradient!"

            has_adapter_grad = False
            for p in model.spatial_adapter.parameters():
                if p.requires_grad and p.grad is not None:
                    has_adapter_grad = True
                    break
            assert has_adapter_grad, "VIOLATION: Spatial adapter received zero gradients!"

            torch.nn.utils.clip_grad_norm_(model.spatial_adapter.parameters(), max_norm=1.0)
            optimizer.step()
            it_train_losses.append(loss.item())

        mean_train_loss = float(np.mean(it_train_losses))
        train_loss_history.append(mean_train_loss)

        # Validation Step across noise realizations and sigma grid
        mean_val_loss, val_per_sigma = evaluate_validation_loss(
            model=model,
            val_records=val_records,
            val_noises=val_noises,
            val_sigma_grid=val_sigma_grid,
            scheduler=scheduler,
            device=device_obj,
            dtype=dtype,
        )
        val_loss_history.append(mean_val_loss)

        if it % 5 == 0 or it == iterations - 1:
            print(
                f"  Iteration {it:03d}/{iterations:03d} | Train Loss: {mean_train_loss:.6f} | Val Loss: {mean_val_loss:.6f}"
            )

    train_duration = time.time() - t_start
    l_init = train_loss_history[0]
    l_final = train_loss_history[-1]
    train_drop_pct = ((l_init - l_final) / max(l_init, 1e-7)) * 100.0

    val_init = initial_val_loss
    val_final = val_loss_history[-1]
    val_drop_pct = ((val_init - val_final) / max(val_init, 1e-7)) * 100.0

    print(f"\n[Training Completed in {train_duration:.2f}s]")
    print(f"  Train Loss: {l_init:.6f} -> {l_final:.6f} (drop: {train_drop_pct:.2f}%)")
    print(f"  Val Loss:   {val_init:.6f} -> {val_final:.6f} (drop: {val_drop_pct:.2f}%)")

    # Save Checkpoint with full provenance metadata
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "spatial_adapter_state_dict": model.spatial_adapter.state_dict(),
        "config": cfg.__dict__,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "metadata": {
            "model_version": "Phase 3 Limited Pilot (Option B.1 front)",
            "backbone": backbone,
            "conditioning_source": conditioning_source,
            "scheduler_config": dict(scheduler.config) if hasattr(scheduler, "config") else {},
            "adapter_config": cfg.__dict__,
            "train_sequences": train_ids,
            "train_sequences_count": train_sequences_count,
            "held_out_sequences": val_ids,
            "iterations": iterations,
            "optimizer_updates_per_iteration": optimizer_updates_per_iteration,
            "total_optimizer_updates": total_optimizer_updates,
            "lr": lr,
            "device": str(device_obj),
            "dtype": str(dtype),
            "model_revision": "main",
            "val_sigma_grid": val_sigma_grid,
            "val_noise_realizations": val_noise_realizations,
            "train_drop_pct": train_drop_pct,
            "val_drop_pct": val_drop_pct,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        },
    }
    torch.save(checkpoint, output_model_path)
    print(f"[Checkpoint] Saved trained adapter to {output_model_path}")

    # Build and Save Structured Report
    report = {
        "experiment": f"Phase 3 Pilot (SVD-XT Spatial Adapter - {backbone.upper()})",
        "title": title,
        "status": "COMPLETED_GENUINE_LATENTS",
        "backbone": backbone,
        "model_revision": "main",
        "device": str(device_obj),
        "dtype": str(dtype),
        "conditioning_source": conditioning_source,
        "conditioning_mode": "OPTION_B1_FRONT",
        "splits": {
            "train_sequences": train_ids,
            "train_sequences_count": train_sequences_count,
            "held_out_sequences": val_ids,
        },
        "parameter_audit": audit,
        "training_dynamics": {
            "initial_train_loss": l_init,
            "final_train_loss": l_final,
            "train_loss_reduction_pct": train_drop_pct,
            "initial_val_loss": val_init,
            "final_val_loss": val_final,
            "val_loss_reduction_pct": val_drop_pct,
            "strictly_decreased": bool(l_final < l_init),
            "generalizes_to_held_out": bool(val_final < val_init),
            "iterations": iterations,
            "optimizer_updates_per_iteration": optimizer_updates_per_iteration,
            "total_optimizer_updates": total_optimizer_updates,
            "val_sigma_grid": val_sigma_grid,
            "duration_s": train_duration,
        },
        "metadata": {
            "iterations": iterations,
            "train_sequences_count": train_sequences_count,
            "optimizer_updates_per_iteration": optimizer_updates_per_iteration,
            "total_optimizer_updates": total_optimizer_updates,
            "device": str(device_obj),
            "dtype": str(dtype),
            "backbone": backbone,
            "model_revision": "main",
        },
        "checkpoint_path": str(output_model_path),
    }

    report_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[Report] Saved training report to {report_out_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SVD-XT spatial adapter on genuine latents.")
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["mock", "svd-xt"],
        default="mock",
        help="Backbone architecture: 'mock' (lightweight 2D surrogate, fast CPU execution) or 'svd-xt' (genuine 1.5B diffusers UNetSpatioTemporalConditionModel).",
    )
    parser.add_argument(
        "--conditioning-source",
        type=str,
        choices=["knn", "oracle"],
        default="knn",
        help="Conditioning trajectory source: 'knn' (donor retrieval) or 'oracle' (ground-truth expert_like).",
    )
    parser.add_argument("--latents-dir", type=Path, default=Path("data/train_latents"))
    parser.add_argument("--observed-t0-dir", type=Path, default=Path("data/train_observed_t0"))
    parser.add_argument("--data-path", type=Path, default=Path("data/train_trajectories.jsonl"))
    parser.add_argument("--output-model", type=Path, default=Path("models/svd_spatial_adapter_front_b1.pt"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/svd_spatial_adapter_training_report.json"))
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    train_pilot_adapter(
        latents_dir=args.latents_dir,
        observed_t0_dir=args.observed_t0_dir,
        data_path=args.data_path,
        output_model_path=args.output_model,
        report_out_path=args.report_out,
        backbone=args.backbone,
        conditioning_source=args.conditioning_source,
        iterations=args.iterations,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
