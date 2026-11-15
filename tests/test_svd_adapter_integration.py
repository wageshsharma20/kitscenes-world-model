"""Unit tests for SVD-XT UNet Spatial Adapter integration and training contracts."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
import torch.nn as nn

from kitscenes import spec
from kitscenes.spatial_adapter import (
    MockSVDUNet,
    SpatialAdapterConfig,
    SpatialConditionedSVDModel,
    ZeroConv2d,
    ZeroConv1d,
)


def test_zero_conv_strict_identity():
    """Verify that ZeroConv2d and ZeroConv1d are strictly initialized to zero."""
    conv2d = ZeroConv2d(in_channels=128, out_channels=4, kernel_size=3)
    x = torch.randn(2, 128, 32, 32)
    out = conv2d(x)
    assert torch.all(out == 0.0), "ZeroConv2d output must be identically zero at init"
    assert torch.all(conv2d.conv.weight == 0.0), "ZeroConv2d weight must be all zeros"
    assert torch.all(conv2d.conv.bias == 0.0), "ZeroConv2d bias must be all zeros"

    conv1d = ZeroConv1d(in_channels=64, out_channels=64, kernel_size=3)
    t_in = torch.randn(2, 64, 25)
    t_out = conv1d(t_in)
    assert torch.all(t_out == 0.0), "ZeroConv1d output must be identically zero at init"


def test_adapter_parameter_budget_under_10m():
    """Verify SpatialAdapter satisfies the strict < 10M parameter budget."""
    cfg = SpatialAdapterConfig(
        adapter_channels=(32, 64, 128, 256),
        latent_channels=4,
        conditioning_type="latent_addition",
        use_temporal_conv=True,
        zero_conv_init=True,
    )
    unet = MockSVDUNet(in_channels=8, out_channels=4)
    model = SpatialConditionedSVDModel(unet=unet, adapter_config=cfg)

    audit = model.count_parameters()
    print(f"Trainable adapter parameters: {audit['trainable_adapter_params']:,}")
    assert audit["adapter_budget_compliant"] is True
    assert audit["trainable_adapter_params"] < 10_000_000
    assert audit["trainable_adapter_params"] > 50_000


def test_strict_gradient_isolation():
    """Verify that UNet backbone receives zero gradients while adapter receives non-zero gradients."""
    cfg = SpatialAdapterConfig(
        adapter_channels=(32, 64, 128, 256),
        latent_channels=4,
        conditioning_type="latent_addition",
    )
    unet = MockSVDUNet(in_channels=8, out_channels=4)
    model = SpatialConditionedSVDModel(unet=unet, adapter_config=cfg)
    model.freeze_backbone()

    # Create dummy batch
    target = torch.randn(1, spec.FRAME_COUNT, 4, 72, 128)
    initial = torch.randn(1, 1, 4, 72, 128)
    mask = torch.rand(1, spec.FRAME_COUNT, 1, 568, 876)

    loss, metrics = model.compute_latent_loss(
        target_latents=target,
        initial_image_latents=initial,
        spatial_masks=mask,
        loss_type="latent_mse",
    )
    loss.backward()

    # UNet backbone must have NO gradients
    for name, param in model.unet.named_parameters():
        assert param.grad is None, f"Backbone parameter '{name}' received a gradient!"

    # Spatial adapter must receive gradients
    has_grad = False
    for name, param in model.spatial_adapter.named_parameters():
        if param.requires_grad and param.grad is not None:
            if param.grad.abs().sum() > 0:
                has_grad = True
                break

    assert has_grad, "Spatial adapter failed to receive gradients!"


def test_zero_step_prior_preservation():
    """Verify that at step 0, adding spatial masks produces identical noisy latents."""
    cfg = SpatialAdapterConfig(
        adapter_channels=(32, 64, 128, 256),
        latent_channels=4,
        conditioning_type="latent_addition",
        zero_conv_init=True,
    )
    unet = MockSVDUNet(in_channels=8, out_channels=4)
    model = SpatialConditionedSVDModel(unet=unet, adapter_config=cfg)

    noisy_latents = torch.randn(1, spec.FRAME_COUNT, 4, 72, 128)
    mask = torch.ones(1, spec.FRAME_COUNT, 1, 568, 876)

    conditioned = model.spatial_adapter.apply_to_latent(noisy_latents, mask)
    diff = (conditioned - noisy_latents).abs().max().item()

    assert diff == 0.0, f"At step 0, conditioned latents must match unconditioned exactly! Diff: {diff}"
