"""Unit tests for SVD-XT UNet Spatial Adapter integration and training contracts."""

from __future__ import annotations

import torch

from kitscenes import spec
from kitscenes.spatial_adapter import (
    MockSVDUNet,
    SpatialAdapterConfig,
    SpatialConditionedSVDModel,
    ZeroConv1d,
    ZeroConv2d,
    create_default_euler_scheduler,
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
    for _name, param in model.spatial_adapter.named_parameters():
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


def test_iterative_generation_from_pure_noise():
    """Verify generate_iterative starts from pure Gaussian noise and runs Euler diffusion loop."""
    unet = MockSVDUNet(in_channels=8, out_channels=4)
    model = SpatialConditionedSVDModel(unet=unet)
    scheduler = create_default_euler_scheduler()

    b, t, c, h_lat, w_lat = 1, 25, 4, 72, 128
    initial_latents = torch.randn(b, 1, c, h_lat, w_lat)
    mask = torch.rand(b, t, 1, 568, 876)

    # Pure Gaussian noise completely decoupled from ground-truth future latents
    torch.manual_seed(42)
    pure_noise = torch.randn(b, t, c, h_lat, w_lat)

    # Baseline unconditioned iterative generation
    gen_baseline = model.generate_iterative(
        initial_image_latents=initial_latents,
        spatial_masks=None,
        scheduler=scheduler,
        num_inference_steps=5,
        num_frames=t,
        noise=pure_noise.clone(),
    )

    # Conditioned iterative generation
    gen_cond = model.generate_iterative(
        initial_image_latents=initial_latents,
        spatial_masks=mask,
        scheduler=scheduler,
        num_inference_steps=5,
        num_frames=t,
        noise=pure_noise.clone(),
    )

    assert gen_baseline.shape == (b, t, c, h_lat, w_lat)
    assert gen_cond.shape == (b, t, c, h_lat, w_lat)
    assert torch.isfinite(gen_baseline).all(), "Baseline generated latents must be finite!"
    assert torch.isfinite(gen_cond).all(), "Conditioned generated latents must be finite!"
    # Verify decoupled: output is not identical to input pure noise
    assert not torch.allclose(gen_baseline, pure_noise)


def test_unet_spatio_temporal_kwargs_construction_and_forward():
    """Verify genuine UNet kwargs (encoder_hidden_states, added_time_ids) are properly passed."""
    unet = MockSVDUNet(in_channels=8, out_channels=4)
    model = SpatialConditionedSVDModel(unet=unet)

    b, t, c, h_lat, w_lat = 1, 25, 4, 72, 128
    target = torch.randn(b, t, c, h_lat, w_lat)
    initial = torch.randn(b, 1, c, h_lat, w_lat)
    mask = torch.rand(b, t, 1, 568, 876)

    # Real CLIP embedding [B, 1, 1024] and micro-conditioning IDs [B, 3]
    real_clip_embeds = torch.randn(b, 1, 1024)
    fps_micro = float(spec.FPS - 1)  # fps - 1 = 4.0
    motion_bucket = 127.0
    aug_level = 0.0
    real_added_time_ids = torch.tensor([[fps_micro, motion_bucket, aug_level]])

    # Test in compute_latent_loss
    loss, metrics = model.compute_latent_loss(
        target_latents=target,
        initial_image_latents=initial,
        spatial_masks=mask,
        timesteps=torch.tensor([500]),
        encoder_hidden_states=real_clip_embeds,
        added_time_ids=real_added_time_ids,
        loss_type="latent_mse",
    )
    assert loss.item() >= 0.0
    assert unet.last_forward_kwargs["encoder_hidden_states"] is real_clip_embeds
    assert unet.last_forward_kwargs["added_time_ids"] is real_added_time_ids
    assert unet.last_forward_kwargs["sample_shape"] == (b, t, 8, h_lat, w_lat)

    # Test in generate_iterative
    scheduler = create_default_euler_scheduler()
    gen = model.generate_iterative(
        initial_image_latents=initial,
        spatial_masks=mask,
        scheduler=scheduler,
        num_inference_steps=3,
        num_frames=t,
        encoder_hidden_states=real_clip_embeds,
        added_time_ids=real_added_time_ids,
    )
    assert gen.shape == (b, t, c, h_lat, w_lat)
    assert unet.last_forward_kwargs["encoder_hidden_states"] is real_clip_embeds
    assert unet.last_forward_kwargs["added_time_ids"] is real_added_time_ids


def test_diffusion_timesteps_schedule_uniform_sampling():
    """Verify diffusion noise schedule supports timesteps t ~ U(0, 1000) for v-prediction and eps."""
    unet = MockSVDUNet(in_channels=8, out_channels=4)
    model = SpatialConditionedSVDModel(unet=unet)

    b, t, c, h_lat, w_lat = 2, 25, 4, 72, 128
    target = torch.randn(b, t, c, h_lat, w_lat)
    initial = torch.randn(b, 1, c, h_lat, w_lat)

    # Sample random timesteps t ~ U(0, 1000)
    torch.manual_seed(123)
    t_uniform = torch.randint(0, 1000, (b,))

    # Test v-prediction
    loss_v, metrics_v = model.compute_latent_loss(
        target_latents=target,
        initial_image_latents=initial,
        timesteps=t_uniform,
        loss_type="v_mse",
        prediction_type="v_prediction",
    )
    assert torch.isfinite(loss_v)
    assert metrics_v["prediction_type"] == "v_prediction"

    # Test epsilon-prediction
    loss_eps, metrics_eps = model.compute_latent_loss(
        target_latents=target,
        initial_image_latents=initial,
        timesteps=t_uniform,
        loss_type="noise_mse",
        prediction_type="epsilon",
    )
    assert torch.isfinite(loss_eps)
    assert metrics_eps["prediction_type"] == "epsilon"
