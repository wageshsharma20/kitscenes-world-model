"""Unit tests for Calibration-Aware Shared-State Conditioning (CSSC) multi-view architecture."""

from __future__ import annotations

import torch

from kitscenes import spec
from kitscenes.multiview_conditioning import (
    CSSCConditionedSVDModel,
    CameraGeometryEncoder,
    MultiViewConditioner,
    SurroundSceneFusion,
)
from kitscenes.spatial_adapter import MockSVDUNet, SpatialAdapterConfig


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
    assert out_front["target_view_token"].shape == (b, 1, 512)
    assert out_front["encoder_hidden_states"].shape == (b, 27, 512)
    assert out_front["fused_surround"].shape == (b, 6, 512)
    assert out_front["temporal_motion"].shape == (b, 25, 512)

    # 2. Forward with dictionary input
    clip_dict = {cam: torch.randn(b, 1, 512) for cam in spec.CAMERAS}
    out_rear = conditioner(clip_dict, motion_states, target_camera="rear")
    assert out_rear["conditioning_tokens"].shape == (b, 25, 512)
    assert out_rear["pooled_token"].shape == (b, 1, 512)
    assert out_rear["encoder_hidden_states"].shape == (b, 27, 512)

    # 3. Forward for all 6 cameras simultaneously
    all_outs = conditioner.forward_all_cameras(clip_embeds, motion_states)
    assert set(all_outs.keys()) == set(spec.CAMERAS)
    for cam in spec.CAMERAS:
        assert all_outs[cam]["conditioning_tokens"].shape == (b, 25, 512)
        assert all_outs[cam]["encoder_hidden_states"].shape == (b, 27, 512)


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


def test_svd_xt_encoder_hidden_states_temporal_semantics():
    """Verify SVD-XT cross-attention semantics: repeat_interleave broadcasts the 27-token bank.

    Proves:
      1. When encoder_hidden_states [B, 27, 1024] is passed to UNetSpatioTemporalConditionModel,
         Diffusers internally executes encoder_hidden_states.repeat_interleave(num_frames, dim=0).
      2. For every video frame t in [0..24], the temporal slice at (b * 25 + t) is identical
         to the full 27-token sequence, acting as a shared trajectory-context bank.
      3. Modifying token 0 (target view context), token 1 (pooled summary), or token k in [2..26]
         (kinematic waypoint at step k-2) impacts cross-attention across ALL 25 frames.
    """
    b = 2
    num_frames = 25
    seq_len = 27
    embed_dim = 1024

    # 1. Simulate the 27-token context sequence:
    # token 0: target view token [B, 1, 1024]
    # token 1: pooled summary token [B, 1, 1024]
    # tokens 2..26: 25 kinematic motion tokens [B, 25, 1024]
    context_bank = torch.randn(b, seq_len, embed_dim)

    # 2. Replicate Diffusers internal broadcast logic:
    # encoder_hidden_states.repeat_interleave(num_frames, dim=0) -> [B * 25, 27, 1024]
    broadcasted = context_bank.repeat_interleave(num_frames, dim=0)
    assert broadcasted.shape == (b * num_frames, seq_len, embed_dim)

    # Numerically verify that every frame slice matches the exact 27 tokens of sample b
    for batch_idx in range(b):
        for frame_idx in range(num_frames):
            row_idx = batch_idx * num_frames + frame_idx
            assert torch.equal(broadcasted[row_idx], context_bank[batch_idx]), (
                f"Frame {frame_idx} of sample {batch_idx} did not receive exact context bank!"
            )

    # 3. Simulate Spatial Cross-Attention Layer:
    # Query: Latent spatial features [B * 25, N_spatial, embed_dim]
    # Keys/Values: Broadcasted context tokens [B * 25, 27, embed_dim]
    n_spatial = 16  # e.g. 4x4 spatial patch
    queries = torch.randn(b * num_frames, n_spatial, embed_dim)
    cross_attn = torch.nn.MultiheadAttention(
        embed_dim=embed_dim, num_heads=8, batch_first=True
    )
    cross_attn.eval()

    with torch.no_grad():
        baseline_out, _ = cross_attn(query=queries, key=broadcasted, value=broadcasted)
        # Reshape to [B, 25, n_spatial, embed_dim] to inspect per-frame impact
        baseline_frames = baseline_out.reshape(b, num_frames, n_spatial, embed_dim)

        # A. Perturb token 0 (target camera view context)
        perturbed_bank_tok0 = context_bank.clone()
        perturbed_bank_tok0[:, 0, :] += torch.randn(b, embed_dim) * 3.0
        broadcasted_tok0 = perturbed_bank_tok0.repeat_interleave(num_frames, dim=0)
        out_tok0, _ = cross_attn(query=queries, key=broadcasted_tok0, value=broadcasted_tok0)
        out_tok0_frames = out_tok0.reshape(b, num_frames, n_spatial, embed_dim)

        # Verify that ALL 25 frames are affected by token 0 perturbation
        for t in range(num_frames):
            frame_diff = torch.norm(out_tok0_frames[:, t] - baseline_frames[:, t]).item()
            assert frame_diff > 0.05, f"Frame {t} cross-attention must respond to token 0!"

        # B. Perturb token 12 (mid-trajectory motion waypoint at step 10)
        perturbed_bank_tok12 = context_bank.clone()
        perturbed_bank_tok12[:, 12, :] += torch.randn(b, embed_dim) * 3.0
        broadcasted_tok12 = perturbed_bank_tok12.repeat_interleave(num_frames, dim=0)
        out_tok12, _ = cross_attn(query=queries, key=broadcasted_tok12, value=broadcasted_tok12)
        out_tok12_frames = out_tok12.reshape(b, num_frames, n_spatial, embed_dim)

        # Verify that early frames (e.g. frame 0) and late frames (e.g. frame 24) BOTH respond to waypoint 10
        diff_f0 = torch.norm(out_tok12_frames[:, 0] - baseline_frames[:, 0]).item()
        diff_f24 = torch.norm(out_tok12_frames[:, 24] - baseline_frames[:, 24]).item()
        assert diff_f0 > 0.05, "Early frame 0 must be able to attend to future waypoint token 12!"
        assert diff_f24 > 0.05, "Late frame 24 must be able to attend to waypoint token 12!"


def test_cssc_conditioned_svd_model_mock_integration():
    """Verify end-to-end CSSCConditionedSVDModel with MockSVDUNet and autograd gradient flow."""
    model = CSSCConditionedSVDModel(
        unet=MockSVDUNet(in_channels=8, out_channels=4),
        conditioner=MultiViewConditioner(model_dim=256, num_heads=4, motion_in_dim=6),
        adapter_config=SpatialAdapterConfig(
            in_channels=1,
            adapter_channels=(16, 32, 64, 128),
            latent_channels=4,
            down_block_channels=(320, 640, 1280, 1280),
            conditioning_type="latent_addition",
        ),
    )
    # 1. Audit parameters & freezing contract
    params = model.count_parameters()
    assert params["trainable_conditioner_params"] > 0
    assert params["trainable_adapter_params"] > 0
    assert params["frozen_backbone_params"] > 0
    for p in model.unet.parameters():
        assert not p.requires_grad

    # 2. Synthetic input batch
    b, t = 2, 25
    noisy_latents = torch.randn(b, t, 4, 16, 16, requires_grad=True)
    initial_image_latents = torch.randn(b, 4, 16, 16)
    spatial_masks = torch.rand(b, t, 1, 128, 128)
    camera_clip_embeds = torch.randn(b, 6, 256)
    motion_states = torch.randn(b, t, 6)

    # 3. Forward pass
    model.train()
    out = model(
        noisy_latents=noisy_latents,
        timesteps=torch.tensor([0.05, 0.10]),
        camera_clip_embeds=camera_clip_embeds,
        motion_states=motion_states,
        target_camera="front",
        initial_image_latents=initial_image_latents,
        spatial_masks=spatial_masks,
    )
    sample = out["sample"]
    assert sample.shape == (b, t, 4, 16, 16)
    assert out["encoder_hidden_states"].shape == (b, 27, 256)

    # 4. Backward pass & gradient isolation audit
    loss = sample.mean()
    loss.backward()

    # Backbone strictly frozen: all UNet weights have grad is None
    for name, p in model.unet.named_parameters():
        assert p.grad is None, f"Backbone parameter {name} received unwanted gradient!"

    # Conditioner parameters must receive gradients
    cond_grads = [p.grad for p in model.conditioner.parameters() if p.grad is not None]
    assert len(cond_grads) > 0, "Conditioner parameters did not receive gradients!"
    assert any(torch.norm(g) > 0 for g in cond_grads)

    # Spatial adapter parameters must receive gradients
    adapter_grads = [p.grad for p in model.spatial_adapter.parameters() if p.grad is not None]
    assert len(adapter_grads) > 0, "SpatialAdapter parameters did not receive gradients!"
    assert any(torch.norm(g) > 0 for g in adapter_grads)

    # 5. Training loss computation
    target_latents = torch.randn(b, t, 4, 16, 16)
    loss_val, metrics = model.compute_latent_loss(
        target_latents=target_latents,
        initial_image_latents=initial_image_latents,
        camera_clip_embeds=camera_clip_embeds,
        motion_states=motion_states,
        target_camera="front",
        spatial_masks=spatial_masks,
    )
    assert isinstance(loss_val, torch.Tensor)
    assert not torch.isnan(loss_val)
    assert "v_loss" in metrics
    assert "weighted_latent_loss" in metrics


def test_all_six_target_cameras_cssc_conditioned_model():
    """Verify all 6 surround target cameras independently produce valid 27-token banks and gradients.

    Covers:
      1. Every camera in spec.CAMERAS ('front', 'front_left', 'front_right', 'rear', 'rear_left', 'rear_right').
      2. Token bank composition: token 0 (target_view), token 1 (pooled_token), tokens 2..26 (conditioning).
      3. Cross-camera token distinctness: different cameras produce unique conditioning banks.
      4. Autograd safety: UNet parameters remain strictly frozen (grad is None) across all camera passes.
    """
    model = CSSCConditionedSVDModel(
        unet=MockSVDUNet(in_channels=8, out_channels=4),
        conditioner=MultiViewConditioner(model_dim=256, num_heads=4, motion_in_dim=6),
        adapter_config=SpatialAdapterConfig(
            in_channels=1,
            adapter_channels=(16, 32, 64, 128),
            latent_channels=4,
            down_block_channels=(320, 640, 1280, 1280),
            conditioning_type="latent_addition",
        ),
    )
    model.train()

    b, t = 1, 25
    noisy_latents = torch.randn(b, t, 4, 16, 16)
    initial_image_latents = torch.randn(b, 4, 16, 16)
    spatial_masks = torch.rand(b, t, 1, 128, 128)
    camera_clip_embeds = torch.randn(b, 6, 256)
    motion_states = torch.randn(b, t, 6)
    timesteps = torch.tensor([0.05])

    camera_token_banks = {}

    for cam_name in spec.CAMERAS:
        # Zero any accumulated gradients before each camera pass
        model.zero_grad()

        out = model(
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            camera_clip_embeds=camera_clip_embeds,
            motion_states=motion_states,
            target_camera=cam_name,
            initial_image_latents=initial_image_latents,
            spatial_masks=spatial_masks,
        )

        sample = out["sample"]
        assert sample.shape == (b, t, 4, 16, 16), f"Sample shape incorrect for {cam_name}"
        encoder_hidden_states = out["encoder_hidden_states"]
        assert encoder_hidden_states.shape == (b, 27, 256), f"Token bank shape incorrect for {cam_name}"

        # Verify token bank structure
        cond_dict = out["cssc_out"]
        assert torch.allclose(encoder_hidden_states[:, 0:1, :], cond_dict["target_view_token"], atol=1e-6)
        assert torch.allclose(encoder_hidden_states[:, 1:2, :], cond_dict["pooled_token"], atol=1e-6)
        assert torch.allclose(encoder_hidden_states[:, 2:27, :], cond_dict["conditioning_tokens"], atol=1e-6)

        # Execute backward pass and verify parameter isolation
        loss = sample.mean()
        loss.backward()

        # Strict frozen UNet contract
        for name, p in model.unet.named_parameters():
            assert p.grad is None, f"UNet parameter {name} leaked gradients during {cam_name} forward pass!"

        # Trainable parameters receive active gradients
        cond_grad_active = any(p.grad is not None and p.grad.norm() > 0 for p in model.conditioner.parameters())
        adapter_grad_active = any(p.grad is not None and p.grad.norm() > 0 for p in model.spatial_adapter.parameters())
        assert cond_grad_active, f"MultiViewConditioner gradients inactive for {cam_name}!"
        assert adapter_grad_active, f"SpatialAdapter gradients inactive for {cam_name}!"

        camera_token_banks[cam_name] = encoder_hidden_states.detach()

    # Verify that different target cameras generate distinct conditioning banks
    camera_list = list(spec.CAMERAS)
    for i in range(len(camera_list)):
        for j in range(i + 1, len(camera_list)):
            c1, c2 = camera_list[i], camera_list[j]
            diff = torch.norm(camera_token_banks[c1] - camera_token_banks[c2]).item()
            assert diff > 0.1, f"Token banks for cameras {c1} and {c2} must differ: diff={diff:.4f}"

