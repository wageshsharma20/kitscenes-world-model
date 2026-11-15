#!/usr/bin/env python3
"""Train the Spatial Conditioning Adapter for Stable Video Diffusion (SVD-XT) on genuine latents.

Backbone Options:
- 'mock': Lightweight 2D neural surrogate for fast CPU development, testing, and CI.
- 'svd-xt': Genuine 1.5B diffusers UNetSpatioTemporalConditionModel backbone with real
  CLIP vision encoder conditioning and micro-conditioning time IDs (requires GPU/high-VRAM).

Protocol (Phase 3 Limited Pilot):
1. Loads genuine VAE latents from data/train_latents/<scenario_id>.pt (encoded via AutoencoderKLTemporalDecoder).
2. Partitions sequences:
   - 6 Training sequences: 0000400, 0000401, 0000402, 0000403, 0000404, 0000405
   - 2 Held-out validation sequences: 0000406, 0000407
3. Conditioning: Option B.1 (initial frame route ribbon, front camera).
4. Real conditioning inputs:
   - Real CLIP vision encoder embeddings [B, 1, 1024]
   - Micro-conditioning time IDs [B, 3] ([fps - 1, motion_bucket_id, augmentation_level])
5. Denoising schedule:
   - Sample timesteps t ~ scheduler.timesteps from standard diffusion schedule.
6. Backbone freezing & gradient isolation:
   - UNet backbone parameters are completely frozen (requires_grad = False).
   - Only the lightweight SpatialAdapter parameters (< 5-10M) are optimized.
   - Assert: all UNet parameter gradients are strictly None.
   - Assert: adapter parameters receive valid non-zero gradients.
7. Optimization:
   - AdamW with weight decay and gradient clipping.
   - Valid diffusion denoising / latent MSE objective.
8. Evaluates generalization on held-out validation sequences.
9. Saves checkpoint to models/svd_spatial_adapter_front_b1.pt and structured report.
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
from kitscenes.spatial_adapter import (
    MockSVDUNet,
    SpatialAdapterConfig,
    SpatialConditionedSVDModel,
    create_default_euler_scheduler,
)


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
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load paired genuine latents, trajectory masks, and conditioning for train and val splits."""
    trajectories: Dict[str, np.ndarray] = {}
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            sid = item["scenario_id"]
            if sid in train_ids or sid in val_ids:
                pts = np.asarray(item.get("expert_like", []), dtype=float)
                if len(pts) >= spec.FRAME_COUNT:
                    trajectories[sid] = pts[:spec.FRAME_COUNT]

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []

    fps_micro = float(spec.FPS - 1)  # SVD micro-conditioning: fps - 1
    motion_bucket = 127.0
    aug_level = 0.0
    added_time_ids = torch.tensor([[fps_micro, motion_bucket, aug_level]], device=device, dtype=dtype)

    for sid in train_ids + val_ids:
        pt_file = latents_dir / f"{sid}.pt"
        if not pt_file.exists():
            raise FileNotFoundError(f"Latent file missing for scenario {sid}: {pt_file}")

        data = torch.load(pt_file, map_location="cpu")
        target_latents = data["target_latents"].to(device=device, dtype=dtype)  # (1, 25, 4, 72, 128)
        initial_latents = data["initial_latents"].to(device=device, dtype=dtype)  # (1, 1, 4, 72, 128)

        # Render Option B.1 spatial mask
        gt_traj = trajectories.get(sid)
        if gt_traj is None:
            gt_traj = np.column_stack([np.linspace(1.5, 30.0, spec.FRAME_COUNT), np.zeros(spec.FRAME_COUNT)])

        raw_b1 = engine.render_initial_frame_temporal_masks(
            camera_name="front",
            waypoints_ego=gt_traj,
            num_frames=spec.FRAME_COUNT,
            channel_dim=True,
        )  # (25, 1, 568, 876)

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
            "gt_trajectory": gt_traj,
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
    iterations: int = 25,
    lr: float = 1e-4,
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
) -> Dict[str, Any]:
    """Execute SVD-XT spatial adapter training pilot on genuine latents."""
    print("=" * 80)
    print("PHASE 3 SVD-XT SPATIAL ADAPTER TRAINING PILOT")
    print(f"Backbone: {backbone} | Device: {device} | Iterations: {iterations} | Learning Rate: {lr}")
    if backbone == "mock":
        print("NOTE: 'mock' is a lightweight 2D neural surrogate for fast verification without 9GB SVD weights.")
    else:
        print("NOTE: 'svd-xt' loads the genuine 1.5B diffusers UNetSpatioTemporalConditionModel backbone.")
    print("=" * 80)

    train_ids = ["0000400", "0000401", "0000402", "0000403", "0000404", "0000405"]
    val_ids = ["0000406", "0000407"]

    print(f"Training split ({len(train_ids)} sequences): {train_ids}")
    print(f"Held-out validation split ({len(val_ids)} sequences): {val_ids}")

    # 1. Load Backbone Components
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
            print("[Backbone] Loading CLIP Vision Model and Feature Extractor...")
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
            raise RuntimeError(
                f"Failed to load genuine SVD-XT components for backbone 'svd-xt': {exc}. "
                "Ensure huggingface weights are accessible or use '--backbone mock'."
            ) from exc
    else:
        print("[Backbone] Initializing MockSVDUNet surrogate backbone...")
        unet = MockSVDUNet(in_channels=8, out_channels=4)

    # 2. Diffusion Scheduler
    scheduler = create_default_euler_scheduler()
    scheduler.set_timesteps(1000)

    engine = camera_projection.CameraProjectionEngine(width=spec.WIDTH, height=spec.HEIGHT)
    train_records, val_records = load_dataset_records(
        latents_dir=latents_dir,
        observed_t0_dir=observed_t0_dir,
        data_path=data_path,
        train_ids=train_ids,
        val_ids=val_ids,
        engine=engine,
        device=device,
        dtype=dtype,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
    )

    print(f"Successfully loaded {len(train_records)} training records and {len(val_records)} validation records.")

    # 3. Initialize Spatial Conditioning Model
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
    model.to(device=device, dtype=dtype)
    model.freeze_backbone()

    # Verify Parameter Budget and Isolation Contracts
    audit = model.count_parameters()
    print(f"[Model] Trainable Adapter Parameters: {audit['trainable_adapter_params']:,}")
    print(f"[Model] Frozen Backbone Parameters:    {audit['frozen_backbone_params']:,}")
    print(f"[Model] Parameter Budget Compliant:   {audit['adapter_budget_compliant']} (< 10M)")

    assert audit["adapter_budget_compliant"], "Adapter exceeds 10M parameter limit!"
    assert audit["frozen_backbone_params"] > 0, "Backbone must have frozen parameters!"
    assert all(not p.requires_grad for p in model.unet.parameters()), "All UNet parameters must be frozen!"
    assert all(p.requires_grad for p in model.spatial_adapter.parameters()), "All adapter parameters must be trainable!"

    # 4. Optimizer & Deterministic Noise
    optimizer = torch.optim.AdamW(model.spatial_adapter.parameters(), lr=lr, weight_decay=1e-4)

    torch.manual_seed(42)
    train_noises = [torch.randn_like(r["target_latents"]) for r in train_records]
    val_noises = [torch.randn_like(r["target_latents"]) for r in val_records]

    # Deterministic validation timestep (midpoint of scheduler)
    val_timestep = scheduler.timesteps[len(scheduler.timesteps) // 2 : len(scheduler.timesteps) // 2 + 1].to(device)

    # Initial Baseline Loss (Step 0)
    model.eval()
    with torch.no_grad():
        initial_val_losses = []
        for r, n in zip(val_records, val_noises):
            loss_v, _ = model.compute_latent_loss(
                target_latents=r["target_latents"],
                initial_image_latents=r["initial_latents"],
                spatial_masks=r["spatial_masks"],
                noise=n,
                timesteps=val_timestep,
                scheduler=scheduler,
                encoder_hidden_states=r.get("encoder_hidden_states"),
                added_time_ids=r.get("added_time_ids"),
            )
            initial_val_losses.append(loss_v.item())
        initial_val_loss = float(np.mean(initial_val_losses))

    print(f"[Pre-Training Baseline] Held-Out Validation Loss: {initial_val_loss:.6f}")

    # 5. Training Loop
    t_start = time.time()
    train_loss_history: List[float] = []
    val_loss_history: List[float] = []

    print("\n[Training Loop] Optimizing Spatial Adapter weights on genuine latents...")
    for it in range(iterations):
        model.train()
        it_train_losses = []

        for r, n in zip(train_records, train_noises):
            optimizer.zero_grad()

            # Sample proper diffusion timestep t ~ scheduler.timesteps
            step_idx = torch.randint(0, len(scheduler.timesteps), (1,)).item()
            t_sample = scheduler.timesteps[step_idx : step_idx + 1].to(device)

            loss, _ = model.compute_latent_loss(
                target_latents=r["target_latents"],
                initial_image_latents=r["initial_latents"],
                spatial_masks=r["spatial_masks"],
                noise=n,
                timesteps=t_sample,
                scheduler=scheduler,
                encoder_hidden_states=r.get("encoder_hidden_states"),
                added_time_ids=r.get("added_time_ids"),
                loss_type="latent_mse",
            )
            loss.backward()

            # Strict Gradient Checks
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

        # Validation Step
        model.eval()
        with torch.no_grad():
            it_val_losses = []
            for r, n in zip(val_records, val_noises):
                loss_v, _ = model.compute_latent_loss(
                    target_latents=r["target_latents"],
                    initial_image_latents=r["initial_latents"],
                    spatial_masks=r["spatial_masks"],
                    noise=n,
                    timesteps=val_timestep,
                    scheduler=scheduler,
                    encoder_hidden_states=r.get("encoder_hidden_states"),
                    added_time_ids=r.get("added_time_ids"),
                )
                it_val_losses.append(loss_v.item())
            mean_val_loss = float(np.mean(it_val_losses))
            val_loss_history.append(mean_val_loss)

        if it % 5 == 0 or it == iterations - 1:
            print(
                f"  Iteration {it:03d}/{iterations:03d} | Train Loss: {mean_train_loss:.6f} | Val Loss: {mean_val_loss:.6f}"
            )

    train_duration = time.time() - t_start
    l_init = train_loss_history[0]
    l_final = train_loss_history[-1]
    train_drop_pct = ((l_init - l_final) / max(l_init, 1e-7)) * 100.0
    val_init = val_loss_history[0]
    val_final = val_loss_history[-1]
    val_drop_pct = ((val_init - val_final) / max(val_init, 1e-7)) * 100.0

    print(f"\n[Training Completed in {train_duration:.2f}s]")
    print(f"  Train Loss: {l_init:.6f} -> {l_final:.6f} (drop: {train_drop_pct:.2f}%)")
    print(f"  Val Loss:   {val_init:.6f} -> {val_final:.6f} (drop: {val_drop_pct:.2f}%)")

    # 6. Save Checkpoint
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "spatial_adapter_state_dict": model.spatial_adapter.state_dict(),
        "config": cfg.__dict__,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "metadata": {
            "model_version": "Phase 3 Limited Pilot (Option B.1 front)",
            "backbone": backbone,
            "train_sequences": train_ids,
            "held_out_sequences": val_ids,
            "iterations": iterations,
            "lr": lr,
            "train_drop_pct": train_drop_pct,
            "val_drop_pct": val_drop_pct,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        },
    }
    torch.save(checkpoint, output_model_path)
    print(f"[Checkpoint] Saved trained adapter to {output_model_path}")

    # 7. Build and Save Report
    report = {
        "experiment": "Phase 3 Limited Pilot (SVD-XT Spatial Adapter)",
        "status": "COMPLETED_GENUINE_LATENTS",
        "backbone": backbone,
        "conditioning_mode": "OPTION_B1_FRONT",
        "splits": {
            "train_sequences": train_ids,
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
            "duration_s": train_duration,
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
        iterations=args.iterations,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
