"""Unit tests for Calibration-Aware Shared-State Conditioning (CSSC) multi-view architecture."""

from __future__ import annotations

import torch

from kitscenes import spec
from kitscenes.multiview_conditioning import (
    CameraGeometryEncoder,
    MultiViewConditioner,
    SurroundSceneFusion,
)


def test_camera_geometry_encoder():
    """Verify geometry encoder produces [B, 6, 1024] distinct tokens for the 6 cameras."""
    encoder = CameraGeometryEncoder(num_cameras=6, model_dim=1024)
    encoder.eval()

    tokens = encoder(batch_size=2)
    assert tokens.shape == (2, 6, 1024)
    assert not torch.isnan(tokens).any()

    # Verify that each camera's token is distinct
    t0 = tokens[0]  # [6, 1024]
    for i in range(6):
        for j in range(i + 1, 6):
            cosine_sim = torch.cosine_similarity(t0[i], t0[j], dim=0)
            assert cosine_sim < 0.99, f"Camera {i} and {j} tokens must be distinct: sim={cosine_sim.item():.4f}"


def test_surround_scene_fusion():
    """Verify surround scene fusion enables cross-view information sharing."""
    fusion = SurroundSceneFusion(model_dim=256, num_heads=4, num_layers=1)
    fusion.eval()

    cam_tokens = torch.randn(1, 6, 256)
    geom_tokens = torch.randn(1, 6, 256)

    fused = fusion(cam_tokens, geom_tokens)
    assert fused.shape == (1, 6, 256)
    assert not torch.isnan(fused).any()

    # Perturbing camera 0 should affect fused representations
    perturbed_cams = cam_tokens.clone()
    perturbed_cams[0, 0] += torch.randn(256) * 5.0
    fused_perturbed = fusion(perturbed_cams, geom_tokens)

    diff_total = torch.norm(fused - fused_perturbed).item()
    assert diff_total > 0.1, "Cross-view attention must propagate visual context"
    diff_other = torch.norm(fused[0, 1:] - fused_perturbed[0, 1:]).item()
    assert diff_other > 0.0, "Cross-view attention must affect other surround camera tokens"


def test_multiview_conditioner_shapes_and_contracts():
    """Verify MultiViewConditioner produces correct conditioning shapes for all cameras."""
    conditioner = MultiViewConditioner(model_dim=512, num_heads=4, motion_in_dim=6)
    conditioner.eval()

    b = 2
    clip_embeds = torch.randn(b, 6, 512)
    motion_states = torch.randn(b, spec.FRAME_COUNT, 6)

    # 1. Forward for a single target camera ('front')
    out_front = conditioner(clip_embeds, motion_states, target_camera="front")
    assert out_front["conditioning_tokens"].shape == (b, 25, 512)
    assert out_front["pooled_token"].shape == (b, 1, 512)
    assert out_front["fused_surround"].shape == (b, 6, 512)
    assert out_front["temporal_motion"].shape == (b, 25, 512)

    # 2. Forward with dictionary input
    clip_dict = {cam: torch.randn(b, 1, 512) for cam in spec.CAMERAS}
    out_rear = conditioner(clip_dict, motion_states, target_camera="rear")
    assert out_rear["conditioning_tokens"].shape == (b, 25, 512)
    assert out_rear["pooled_token"].shape == (b, 1, 512)

    # 3. Forward for all 6 cameras simultaneously
    all_outs = conditioner.forward_all_cameras(clip_embeds, motion_states)
    assert set(all_outs.keys()) == set(spec.CAMERAS)
    for cam in spec.CAMERAS:
        assert all_outs[cam]["conditioning_tokens"].shape == (b, 25, 512)


def test_multiview_conditioner_gradients():
    """Verify backward gradient flow through all trainable weights of MultiViewConditioner."""
    conditioner = MultiViewConditioner(model_dim=256, num_heads=4, motion_in_dim=6)
    conditioner.train()

    clip_embeds = torch.randn(2, 6, 256, requires_grad=True)
    motion_states = torch.randn(2, spec.FRAME_COUNT, 6, requires_grad=True)

    out = conditioner(clip_embeds, motion_states, target_camera="front")
    loss = (out["conditioning_tokens"] ** 2).mean() + (out["pooled_token"] ** 2).mean()
    loss.backward()

    assert clip_embeds.grad is not None
    assert torch.norm(clip_embeds.grad) > 0.0

    assert motion_states.grad is not None
    assert torch.norm(motion_states.grad) > 0.0

    # Ensure key modules have gradients
    for name, p in conditioner.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient"
            assert torch.norm(p.grad) > 0.0, f"Parameter {name} has zero gradient"


def test_multiview_camera_differentiation():
    """Verify that different target cameras receive different conditioning tokens under identical inputs."""
    conditioner = MultiViewConditioner(model_dim=512, num_heads=4, motion_in_dim=6)
    conditioner.eval()

    clip_embeds = torch.randn(1, 6, 512)
    motion_states = torch.randn(1, spec.FRAME_COUNT, 6)

    out_front = conditioner(clip_embeds, motion_states, target_camera="front")
    out_rear = conditioner(clip_embeds, motion_states, target_camera="rear")

    diff = torch.norm(out_front["conditioning_tokens"] - out_rear["conditioning_tokens"]).item()
    assert diff > 1.0, f"Front and rear conditioning tokens must differ: diff={diff}"
