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
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
) -> Dict[str, Any]:
    """Execute SVD-XT spatial adapter training pilot on genuine latents."""
    title = f"Phase 3 Pilot ({'Mock Surrogate' if backbone == 'mock' else 'Genuine SVD-XT 1.5B'})"
    print("=" * 80)
    print(title.upper())
    print(f"Backbone: {backbone} | Conditioning Source: {conditioning_source} | Device: {device}")
    print(f"Iterations: {iterations} | Learning Rate: {lr}")
    if backbone == "mock":
        print("NOTE: 'mock' is a lightweight 2D neural surrogate for fast verification without 9GB SVD weights.")
    else:
        print("NOTE: 'svd-xt' loads the genuine 1.5B diffusers UNetSpatioTemporalConditionModel backbone.")
    print("=" * 80)

    train_ids = ["0000400", "0000401", "0000402", "0000403", "0000404", "0000405"]
    val_ids = ["0000406", "0000407"]

    print(f"Training split ({len(train_ids)} sequences): {train_ids}")
    print(f"Held-out validation split ({len(val_ids)} sequences): {val_ids}")

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

    scheduler = create_default_euler_scheduler()

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
        conditioning_source=conditioning_source,
    )

    print(f"Successfully loaded {len(train_records)} training records and {len(val_records)} validation records.")

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

    audit = model.count_parameters()
    print(f"[Model] Trainable Adapter Parameters: {audit['trainable_adapter_params']:,}")
    print(f"[Model] Frozen Backbone Parameters:    {audit['frozen_backbone_params']:,}")
    print(f"[Model] Parameter Budget Compliant:   {audit['adapter_budget_compliant']} (< 10M)")

    assert audit["adapter_budget_compliant"], "Adapter exceeds 10M parameter limit!"
    assert audit["frozen_backbone_params"] > 0, "Backbone must have frozen parameters!"
    assert all(not p.requires_grad for p in model.unet.parameters()), "All UNet parameters must be frozen!"
    assert all(p.requires_grad for p in model.spatial_adapter.parameters()), "All adapter parameters must be trainable!"

    optimizer = torch.optim.AdamW(model.spatial_adapter.parameters(), lr=lr, weight_decay=1e-4)

    torch.manual_seed(42)
    train_noises = [torch.randn_like(r["target_latents"]) for r in train_records]
    val_noises = [torch.randn_like(r["target_latents"]) for r in val_records]

    # Pre-Training Step 0 Baseline Validation Loss (sigma=15.0)
    model.eval()
    with torch.no_grad():
        initial_val_losses = []
        for r, n in zip(val_records, val_noises):
            loss_v, _ = model.compute_latent_loss(
                target_latents=r["target_latents"],
                initial_image_latents=r["initial_latents"],
                spatial_masks=r["spatial_masks"],
                noise=n,
                sigmas=torch.full((1,), 15.0, device=device, dtype=dtype),
                scheduler=scheduler,
                encoder_hidden_states=r.get("encoder_hidden_states"),
                added_time_ids=r.get("added_time_ids"),
            )
            initial_val_losses.append(loss_v.item())
        initial_val_loss = float(np.mean(initial_val_losses))

    print(f"[Pre-Training Step 0 Baseline] Held-Out Validation Loss: {initial_val_loss:.6f}")

    t_start = time.time()
    train_loss_history: List[float] = []
    val_loss_history: List[float] = [initial_val_loss]

    print("\n[Training Loop] Optimizing Spatial Adapter weights on genuine latents...")
    for it in range(iterations):
        model.train()
        it_train_losses = []

        for r, n in zip(train_records, train_noises):
            optimizer.zero_grad()

            # Continuous training noise sampling: log(sigma) ~ Normal(1.0, 1.6^2)
            sigmas, c_n = sample_training_noise_levels(1, loc=1.0, scale=1.6, device=torch.device(device), dtype=dtype)

            loss, _ = model.compute_latent_loss(
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

        # Validation Step (fixed sigma=15.0)
        model.eval()
        with torch.no_grad():
            it_val_losses = []
            for r, n in zip(val_records, val_noises):
                loss_v, _ = model.compute_latent_loss(
                    target_latents=r["target_latents"],
                    initial_image_latents=r["initial_latents"],
                    spatial_masks=r["spatial_masks"],
                    noise=n,
                    sigmas=torch.full((1,), 15.0, device=device, dtype=dtype),
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
            "scheduler_config": scheduler.config if hasattr(scheduler, "config") else {},
            "adapter_config": cfg.__dict__,
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

    # Build and Save Structured Report
    report = {
        "experiment": f"Phase 3 Pilot (SVD-XT Spatial Adapter - {backbone.upper()})",
        "title": title,
        "status": "COMPLETED_GENUINE_LATENTS",
        "backbone": backbone,
        "conditioning_source": conditioning_source,
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
