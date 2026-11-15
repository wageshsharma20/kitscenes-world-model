"""Unit tests verifying SVD VAE latent encoding and exact temporal mapping contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import pytest

from kitscenes import camera_projection, spec


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LATENTS_DIR = PROJECT_ROOT / "data" / "train_latents"


def test_latent_temporal_mapping_contract():
    """Verify 25 RGB frames map 1-to-1 to 25 latent temporal slices."""
    manifest_file = LATENTS_DIR / "manifest.json"
    if not manifest_file.exists():
        pytest.skip("Latents not yet encoded on disk.")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["total_encoded"] >= 8

    for record in manifest["records"]:
        sid = record["scenario_id"]
        pt_file = LATENTS_DIR / f"{sid}.pt"
        assert pt_file.exists(), f"Missing latent file {pt_file}"

        data = torch.load(pt_file, map_location="cpu")
        target_latents = data["target_latents"]
        initial_latents = data["initial_latents"]
        meta = data["metadata"]

        # Shape contract assertions
        assert target_latents.ndim == 5, "Target latents must be 5D (B, T, C, H, W)"
        assert target_latents.shape == (1, spec.FRAME_COUNT, 4, 72, 128)
        assert initial_latents.shape == (1, 1, 4, 72, 128)

        # Integrity assertions
        assert not torch.isnan(target_latents).any(), "Target latents must not contain NaNs"
        assert not torch.isnan(initial_latents).any(), "Initial latents must not contain NaNs"
        assert target_latents.abs().sum() > 0, "Target latents must be non-zero"
        assert initial_latents.abs().sum() > 0, "Initial latents must be non-zero"

        # Temporal slice verification: each frame k corresponds to timestamp k * 0.2s
        assert meta["num_frames"] == 25
        assert meta["fps"] == 5.0
        assert meta["camera"] == "front"


def test_spatial_mask_to_latent_grid_alignment():
    """Verify that Option B.1 spatial mask matches the VAE latent spatial-temporal grid."""
    engine = camera_projection.CameraProjectionEngine(width=spec.WIDTH, height=spec.HEIGHT)

    # 25 waypoints
    straight_waypoints = np.column_stack([
        np.linspace(1.0, 40.0, spec.FRAME_COUNT),
        np.zeros(spec.FRAME_COUNT),
    ])

    mask_raw = engine.render_initial_frame_temporal_masks(
        camera_name="front",
        waypoints_ego=straight_waypoints,
        num_frames=spec.FRAME_COUNT,
        channel_dim=True,
    )  # (25, 1, 568, 876)

    assert mask_raw.shape == (25, 1, 568, 876)

    # Simulate aspect ratio pad: 568x876 -> 576x888 -> pad to 576x1024
    scale = 576.0 / 568.0
    scaled_w = int(round(876 * scale))  # 888
    pad_total = 1024 - scaled_w  # 136
    pad_left = pad_total // 2  # 68
    pad_right = pad_total - pad_left  # 68

    mask_tensor = torch.from_numpy(mask_raw).float()
    # Downsample directly to latent resolution (72, 128)
    mask_latent = torch.nn.functional.interpolate(
        mask_tensor,
        size=(72, 128),
        mode="bilinear",
        align_corners=False,
    )  # (25, 1, 72, 128)

    assert mask_latent.shape == (spec.FRAME_COUNT, 1, 72, 128)
    assert mask_latent.min() >= 0.0
    assert mask_latent.max() > 0.1, "Mask must contain active road corridor"
    # Ensure all 25 temporal slices have matching temporal dimension
    assert mask_latent.shape[0] == spec.FRAME_COUNT
