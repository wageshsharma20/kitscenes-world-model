"""Unit tests for KITScenes SVD-XT Spatial Conditioning Adapter & Ablation Framework."""

from __future__ import annotations

import numpy as np
import torch

from kitscenes import camera_projection, spec
from kitscenes.conditioning_ablation import (
    ConditioningAblationFramework,
    ConditioningMode,
)
from kitscenes.spatial_adapter import (
    MockSVDUNet,
    SpatialAdapter,
    SpatialAdapterConfig,
    SpatialConditionedSVDModel,
)


def compute_lateral_bias(tensor: torch.Tensor) -> float:
    """Compute normalized horizontal Center-of-Mass across width dimension."""
    w = tensor.shape[-1]
    coords = torch.linspace(-1.0, 1.0, w, device=tensor.device)
    intensity = tensor.abs().sum(dim=tuple(range(tensor.ndim - 1)))
    total = float(intensity.sum().item())
    if total < 1e-7:
        return 0.0
    return float((intensity * coords).sum().item() / total)


def test_spatial_adapter_shapes_and_modes():
    """Verify adapter forward pass shapes with 5D, 4D, and different conditioning modes."""
    # 1. Latent addition mode (default)
    cfg_add = SpatialAdapterConfig(
        adapter_channels=(16, 32, 64, 128),
        latent_channels=4,
        conditioning_type="latent_addition",
    )
    adapter_add = SpatialAdapter(cfg_add)

    # 5D input: [B=2, T=5, C=1, H=64, W=64] -> output [B=2, T=5, C_lat=4, H/8=8, W/8=8]
    mask_5d = torch.randn(2, 5, 1, 64, 64)
    out_5d = adapter_add(mask_5d)
    assert out_5d.latent_conditioning.shape == (2, 5, 4, 8, 8)
    assert out_5d.down_block_residuals is None

    # 4D input: [B=2, C=1, H=64, W=64] with num_frames=10 -> [B=2, T=10, 4, 8, 8]
    mask_4d = torch.randn(2, 1, 64, 64)
    out_4d = adapter_add(mask_4d, num_frames=10)
    assert out_4d.latent_conditioning.shape == (2, 10, 4, 8, 8)

    # Explicit target latent shape interpolation (e.g. matching 71x109)
    out_interp = adapter_add(mask_5d, target_latent_shape=(10, 12))
    assert out_interp.latent_conditioning.shape == (2, 5, 4, 10, 12)

    # 2. Residual injection mode (ControlNet-style)
    cfg_res = SpatialAdapterConfig(
        adapter_channels=(16, 32, 64, 128),
        latent_channels=4,
        down_block_channels=(32, 64, 128, 256),
        conditioning_type="residual_injection",
    )
    adapter_res = SpatialAdapter(cfg_res)
    out_res = adapter_res(mask_5d)
    assert out_res.down_block_residuals is not None
    assert len(out_res.down_block_residuals) == 4
    # Multi-scale down-block channels: [32, 64, 128, 256]
    expected_channels = [32, 64, 128, 256]
    for _idx, (res_i, exp_ch) in enumerate(zip(out_res.down_block_residuals, expected_channels)):
        assert res_i.shape[0] == 2
        assert res_i.shape[1] == 5
        assert res_i.shape[2] == exp_ch


def test_zero_conv_initialization_and_identity():
    """Verify that at initialization, adapter output is strictly zero and leaves latents unchanged."""
    adapter = SpatialAdapter()
    mask = torch.randn(2, 4, 1, 64, 64)

    out = adapter(mask)
    # Latent conditioning must be exactly 0.0
    assert torch.all(out.latent_conditioning == 0.0)

    # Latent addition preserves exact noisy latents
    noisy_latents = torch.randn(2, 4, 4, 8, 8)
    applied = adapter.apply_to_latent(noisy_latents, mask)
    assert torch.equal(applied, noisy_latents)


def test_parameter_budget_compliance():
    """Verify that adapter parameter count strictly satisfies the < 5-10M parameter limit."""
    # Standard latent addition configuration
    adapter = SpatialAdapter(SpatialAdapterConfig(conditioning_type="latent_addition"))
    count = adapter.parameter_count
    # Should be well under 5-10M
    assert count < 5_000_000
    assert count > 10_000

    # Residual injection configuration
    adapter_res = SpatialAdapter(SpatialAdapterConfig(conditioning_type="residual_injection"))
    count_res = adapter_res.parameter_count
    assert count_res < 10_000_000


def test_backbone_freezing_and_gradient_isolation():
    """Verify that SVD backbone parameters remain frozen and only adapter parameters receive gradients."""
    unet = MockSVDUNet(in_channels=8, out_channels=4)
    model = SpatialConditionedSVDModel(unet=unet)
    model.freeze_backbone()

    audit = model.count_parameters()
    assert audit["adapter_budget_compliant"] is True
    assert audit["frozen_backbone_params"] > 0
    assert audit["trainable_adapter_params"] > 0

    # Verify requires_grad status
    for name, param in model.unet.named_parameters():
        assert not param.requires_grad, f"UNet parameter {name} requires_grad must be False"

    for name, param in model.spatial_adapter.named_parameters():
        assert param.requires_grad, f"Adapter parameter {name} requires_grad must be True"

    # Forward + backward pass
    target = torch.randn(2, 4, 4, 16, 16)
    initial = torch.randn(2, 4, 16, 16)
    mask = torch.randn(2, 4, 1, 32, 32)

    loss, _ = model.compute_latent_loss(target, initial, mask, loss_type="latent_mse")
    loss.backward()

    # Backbone parameters must have NO gradient
    for name, param in model.unet.named_parameters():
        assert param.grad is None, f"UNet parameter {name} should have None gradient"

    # Adapter parameters must have active gradients
    adapter_has_grad = False
    for _name, param in model.spatial_adapter.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            adapter_has_grad = True
    assert adapter_has_grad, "Spatial adapter received no gradients during backward pass!"


def test_4_way_conditioning_ablation_contracts():
    """Verify all 4 conditioning modes generate valid masks and enforce contracts."""
    adapter = SpatialAdapter()
    framework = ConditioningAblationFramework(spatial_adapter=adapter)

    # Sample scenario
    sample = {
        "scenario_id": "0000450",
        "driving_instruction": "turn left",
        "scenario_type": "4 intersection",
        "past": [[-10.0 + i * 0.5, 0.0] for i in range(21)],
        "expert_like": [[float(i), float(i * 0.4)] for i in range(1, 26)],
    }

    ablation = framework.run_4way_ablation(sample, target_latent_shape=(16, 24))
    assert len(ablation) == 4

    # Mode 0: NO_SPATIAL_CONDITIONING
    m0 = ablation["NO_SPATIAL_CONDITIONING"]
    assert m0.mode == ConditioningMode.NO_SPATIAL_CONDITIONING
    assert m0.adapter_output is not None
    assert torch.all(m0.adapter_output.latent_conditioning == 0.0)

    # Mode 1: OPTION_B1_ORACLE
    m1 = ablation["OPTION_B1_ORACLE"]
    assert m1.mode == ConditioningMode.OPTION_B1_ORACLE
    assert m1.spatial_mask is not None
    assert m1.spatial_mask.shape == (1, 25, 1, spec.HEIGHT, spec.WIDTH)
    assert m1.spatial_mask.max() > 0.0

    # Mode 2: OPTION_B1_PREDICTED
    m2 = ablation["OPTION_B1_PREDICTED"]
    assert m2.mode == ConditioningMode.OPTION_B1_PREDICTED
    assert m2.spatial_mask is not None
    assert m2.spatial_mask.shape == (1, 25, 1, spec.HEIGHT, spec.WIDTH)
    assert "l2_to_gt" in m2.metadata

    # Mode 3: OPTION_B2_ORACLE
    m3 = ablation["OPTION_B2_ORACLE"]
    assert m3.mode == ConditioningMode.OPTION_B2_ORACLE
    assert m3.spatial_mask is not None
    assert m3.spatial_mask.shape == (1, 25, 1, spec.HEIGHT, spec.WIDTH)
    assert m3.spatial_mask.max() > 0.0


def test_motion_steerability_lateral_bias():
    """Verify that left-turning vs right-turning trajectory masks measurably redirect spatial flow."""
    engine = camera_projection.CameraProjectionEngine(width=876, height=568)

    # Left-turning trajectory (y > 0) vs Right-turning trajectory (y < 0)
    x_pts = np.linspace(1.5, 30.0, 25)
    t_frac = np.linspace(0.0, 1.0, 25)
    left_traj = np.column_stack([x_pts, 15.0 * (t_frac**1.5)])
    right_traj = np.column_stack([x_pts, -15.0 * (t_frac**1.5)])

    m_left_np = engine.render_initial_frame_temporal_masks("front", left_traj, channel_dim=True)
    m_right_np = engine.render_initial_frame_temporal_masks("front", right_traj, channel_dim=True)

    m_left = torch.from_numpy(m_left_np).unsqueeze(0).float()
    m_right = torch.from_numpy(m_right_np).unsqueeze(0).float()

    bias_left = compute_lateral_bias(m_left)
    bias_right = compute_lateral_bias(m_right)

    # In front camera: left turn curves to image left (bias < 0); right turn curves to image right (bias > 0)
    assert bias_left < 0.0, f"Left mask bias should be negative, got {bias_left}"
    assert bias_right > 0.0, f"Right mask bias should be positive, got {bias_right}"
    steer_delta = bias_right - bias_left
    assert steer_delta > 0.25, f"Steerability delta should be > 0.25, got {steer_delta}"


def test_small_overfit_convergence():
    """Verify that optimizer on spatial adapter strictly decreases loss on a small batch."""
    torch.manual_seed(42)
    cfg = SpatialAdapterConfig(adapter_channels=(16, 32, 64, 128))
    model = SpatialConditionedSVDModel(adapter_config=cfg)
    model.freeze_backbone()

    optimizer = torch.optim.AdamW(model.spatial_adapter.parameters(), lr=1e-3)

    target = torch.randn(2, 4, 4, 16, 16)
    initial = torch.randn(2, 4, 16, 16)
    mask = torch.randn(2, 4, 1, 32, 32)
    noise = torch.randn_like(target)

    losses = []
    for _step in range(15):
        optimizer.zero_grad()
        loss, _ = model.compute_latent_loss(target, initial, mask, noise=noise, loss_type="latent_mse")
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Final loss must strictly decrease relative to initial loss
    assert losses[-1] < losses[0], f"Loss failed to decrease: initial {losses[0]}, final {losses[-1]}"
