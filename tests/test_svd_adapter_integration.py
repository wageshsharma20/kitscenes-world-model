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

    # Test in generate_iterative (single-pass guidance=1.0)
    scheduler = create_default_euler_scheduler()
    gen = model.generate_iterative(
        initial_image_latents=initial,
        spatial_masks=mask,
        scheduler=scheduler,
        num_inference_steps=3,
        num_frames=t,
        min_guidance_scale=1.0,
        max_guidance_scale=1.0,
        encoder_hidden_states=real_clip_embeds,
        added_time_ids=real_added_time_ids,
    )
    assert gen.shape == (b, t, c, h_lat, w_lat)
    assert unet.last_forward_kwargs["encoder_hidden_states"] is real_clip_embeds
    assert unet.last_forward_kwargs["added_time_ids"] is real_added_time_ids

    # Test in generate_iterative with frame-wise CFG (max_guidance_scale > 1.0)
    gen_cfg = model.generate_iterative(
        initial_image_latents=initial,
        spatial_masks=mask,
        scheduler=scheduler,
        num_inference_steps=2,
        num_frames=t,
        min_guidance_scale=1.0,
        max_guidance_scale=3.0,
        encoder_hidden_states=real_clip_embeds,
        added_time_ids=real_added_time_ids,
    )
    assert gen_cfg.shape == (b, t, c, h_lat, w_lat)
    assert torch.equal(unet.last_forward_kwargs["encoder_hidden_states"][b:], real_clip_embeds)
    assert torch.equal(unet.last_forward_kwargs["encoder_hidden_states"][:b], torch.zeros_like(real_clip_embeds))


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


def test_sigma_c_noise_roundtrip_and_scheduler_consistency():
    """Verify sigma <-> c_noise mathematical roundtrip and official scheduler consistency."""
    from kitscenes.spatial_adapter import (
        c_noise_to_sigma,
        create_default_euler_scheduler,
        sigma_to_c_noise,
    )

    sigmas = torch.tensor([0.002, 0.05, 1.0, 15.0, 700.0])
    c_noise = sigma_to_c_noise(sigmas)
    sigmas_rec = c_noise_to_sigma(c_noise)
    assert torch.allclose(sigmas, sigmas_rec, rtol=1e-5)

    sched = create_default_euler_scheduler()
    sched.set_timesteps(25)
    expected_c_noise = 0.25 * torch.log(sched.sigmas[:25])
    assert torch.allclose(sched.timesteps[:25], expected_c_noise, atol=1e-4)


def test_svd_noise_parameterization_and_v_target_recovery():
    """Verify continuous SVD noise tensors and analytical recovery of clean latents and epsilon."""
    from kitscenes.spatial_adapter import compute_svd_noise_tensors

    b, t, c, h, w = 2, 5, 4, 16, 16
    clean_z0 = torch.randn(b, t, c, h, w)
    noise_eps = torch.randn(b, t, c, h, w)
    sigmas = torch.tensor([0.5, 12.0])

    z_sigma, model_input, v_target, c_out, c_skip = compute_svd_noise_tensors(
        clean_latents=clean_z0,
        noise=noise_eps,
        sigmas=sigmas,
    )

    # Analytical check 1: z_0 recovery from v_target
    z0_hat = v_target * c_out + z_sigma * c_skip
    assert torch.allclose(z0_hat, clean_z0, atol=1e-5), "Analytical z_0 recovery failed!"

    # Analytical check 2: epsilon recovery
    sig_reshaped = sigmas.view(-1, 1, 1, 1, 1)
    denom = torch.sqrt(1.0 + sig_reshaped**2)
    eps_hat = v_target / denom + z_sigma * (sig_reshaped / (1.0 + sig_reshaped**2))
    assert torch.allclose(eps_hat, noise_eps, atol=1e-5), "Analytical epsilon recovery failed!"


def test_log_normal_training_noise_sampling():
    """Verify training noise distribution log(sigma) ~ Normal(1.0, 1.6^2)."""
    from kitscenes.spatial_adapter import sample_training_noise_levels

    torch.manual_seed(42)
    sigmas, c_noise = sample_training_noise_levels(10_000, loc=1.0, scale=1.6)

    log_sigma = torch.log(sigmas)
    assert abs(log_sigma.mean().item() - 1.0) < 0.05
    assert abs(log_sigma.std().item() - 1.6) < 0.05
    assert torch.allclose(c_noise, 0.25 * log_sigma, atol=1e-5)


def test_frame_wise_cfg_generation_mock():
    """Verify frame-wise CFG schedule produces valid latents on MockSVDUNet."""
    unet = MockSVDUNet(in_channels=8, out_channels=4)
    model = SpatialConditionedSVDModel(unet=unet)
    scheduler = create_default_euler_scheduler()

    b, t, c, h, w = 1, 5, 4, 16, 16
    init_latents = torch.randn(b, 1, c, h, w)
    noise = torch.randn(b, t, c, h, w)

    latents_cfg = model.generate_iterative(
        initial_image_latents=init_latents,
        spatial_masks=None,
        scheduler=scheduler,
        num_inference_steps=2,
        num_frames=t,
        latent_shape=(h, w),
        min_guidance_scale=1.0,
        max_guidance_scale=3.0,
        noise=noise,
    )
    assert latents_cfg.shape == (b, t, c, h, w)
    assert torch.isfinite(latents_cfg).all()


def test_checkpoint_backbone_mismatch_rejection(tmp_path, monkeypatch):
    """Verify evaluator rejects checkpoint trained on mock backbone when requested backbone is svd-xt."""
    import pytest
    from scripts.eval_pilot_comparison import run_pilot_evaluation

    mock_ckpt_path = tmp_path / "mock_adapter.pt"
    torch.save(
        {
            "spatial_adapter_state_dict": {},
            "metadata": {"backbone": "mock"},
        },
        mock_ckpt_path,
    )

    # Mock CUDA availability to test checkpoint backbone validation on svd-xt
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(ValueError, match="Checkpoint backbone mismatch"):
        run_pilot_evaluation(
            model_checkpoint=mock_ckpt_path,
            latents_dir=tmp_path,
            observed_t0_dir=tmp_path,
            future_videos_dir=tmp_path,
            trajectories_file=tmp_path,
            output_renders_dir=tmp_path,
            report_out_path=tmp_path / "report.json",
            backbone="svd-xt",
            device="cuda:0",
        )


def test_meta_device_real_unet_structure_inspection():
    """Verify genuine 1.5B SVD-XT UNet architecture can be inspected on meta device with 0 RAM."""
    from diffusers import UNetSpatioTemporalConditionModel

    config = UNetSpatioTemporalConditionModel.load_config(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        subfolder="unet",
    )
    assert config["in_channels"] == 8
    assert config["out_channels"] == 4
    assert config["addition_time_embed_dim"] == 256
    assert config["cross_attention_dim"] == 1024

    with torch.device("meta"):
        meta_unet = UNetSpatioTemporalConditionModel.from_config(config)

    assert meta_unet.conv_in.weight.device.type == "meta"
    assert meta_unet.conv_in.in_channels == 8


def test_scheduler_fallback_config_strict_match():
    """Verify that create_default_euler_scheduler matches official SVD-XT scheduler config attributes."""
    from kitscenes.spatial_adapter import create_default_euler_scheduler

    sched = create_default_euler_scheduler()
    assert sched.config.sigma_min == 0.002
    assert sched.config.sigma_max == 700.0
    assert sched.config.prediction_type == "v_prediction"
    assert sched.config.timestep_spacing == "leading"
    assert sched.config.steps_offset == 1
    assert sched.config.rescale_betas_zero_snr is False
    assert sched.config.final_sigmas_type == "zero"
    assert sched.config.use_karras_sigmas is True


def test_cfg_dual_batching_tensor_ordering():
    """Verify CFG dual-batching strictly orders unconditional (zeros) first, conditional second."""
    from kitscenes.spatial_adapter import SpatialConditionedSVDModel, SpatialAdapterConfig, MockSVDUNet

    cfg = SpatialAdapterConfig()
    unet = MockSVDUNet(in_channels=8, out_channels=4)
    model = SpatialConditionedSVDModel(unet=unet, adapter_config=cfg)

    b, t, c, h, w = 1, 2, 4, 16, 16
    initial_image_latents = torch.randn(b, c, h, w)
    encoder_hidden_states = torch.randn(b, 1, 1024)
    added_time_ids = torch.tensor([[4.0, 127.0, 0.02]])

    captured_inputs = {}

    def mock_forward(sample, timestep, encoder_hidden_states=None, added_time_ids=None):
        captured_inputs["sample"] = sample
        captured_inputs["encoder_hidden_states"] = encoder_hidden_states
        captured_inputs["added_time_ids"] = added_time_ids
        return type("Output", (), {"sample": torch.zeros_like(sample[:, :, :4])})()

    unet.forward = mock_forward

    _ = model.generate_iterative(
        initial_image_latents=initial_image_latents,
        num_inference_steps=1,
        num_frames=t,
        min_guidance_scale=1.0,
        max_guidance_scale=3.0,  # Enable CFG
        encoder_hidden_states=encoder_hidden_states,
        added_time_ids=added_time_ids,
    )

    sample = captured_inputs["sample"]
    enc = captured_inputs["encoder_hidden_states"]
    time_ids = captured_inputs["added_time_ids"]

    # In CFG, batch size is 2 * b
    assert sample.shape[0] == 2 * b
    assert enc.shape[0] == 2 * b
    assert time_ids.shape[0] == 2 * b

    # Sample conditioning channels [:, :, 4:8] correspond to cond_latents_batch
    uncond_cond_latents = sample[:b, :, 4:8]
    cond_cond_latents = sample[b:, :, 4:8]

    # Unconditional slice MUST be zero
    assert torch.all(uncond_cond_latents == 0.0), "CFG unconditional cond latents slice must be zero!"

    # Conditional slice MUST match initial_image_latents repeated across time
    for frame_idx in range(t):
        assert torch.allclose(cond_cond_latents[:, frame_idx], initial_image_latents), "CFG conditional cond latents slice must match initial image latents!"

    # Unconditional encoder states MUST be zero
    assert torch.all(enc[:b] == 0.0), "CFG unconditional CLIP embeddings slice must be zero!"
    assert torch.allclose(enc[b:], encoder_hidden_states), "CFG conditional CLIP embeddings slice must match input!"


def test_svd_v_prediction_loss_weighting_equivalence():
    """Verify analytical equivalence between SNR-weighted latent loss and v-prediction loss."""
    from kitscenes.spatial_adapter import SpatialConditionedSVDModel, SpatialAdapterConfig, MockSVDUNet

    cfg = SpatialAdapterConfig()
    unet = MockSVDUNet(in_channels=8, out_channels=4)
    model = SpatialConditionedSVDModel(unet=unet, adapter_config=cfg)

    b, t, c, h, w = 1, 2, 4, 8, 8
    target_latents = torch.randn(b, t, c, h, w)
    initial_latents = torch.randn(b, c, h, w)

    # Test across multiple noise levels sigma
    for sig_val in [0.05, 0.5, 1.0, 15.0, 100.0]:
        sigmas = torch.tensor([sig_val])
        loss_v, metrics = model.compute_latent_loss(
            target_latents=target_latents,
            initial_image_latents=initial_latents,
            sigmas=sigmas,
            loss_type="v_mse",
        )

        # Inside the same forward pass, loss_v and weighted_latent_mse must match analytically
        assert abs(metrics["loss_v"] - metrics["weighted_latent_mse"]) < 1e-4, (
            f"Equivalence failed at sigma={sig_val}: v_loss={metrics['loss_v']}, weighted={metrics['weighted_latent_mse']}"
        )


def test_svd_xt_train_fails_loud_on_cpu_mps(tmp_path, monkeypatch):
    """Verify that svd-xt execution on CPU or MPS fails loud immediately with RuntimeError on training."""
    import pytest
    from scripts.train_svd_spatial_adapter import train_pilot_adapter, validate_device_and_dtype

    # 1. When CUDA is unavailable: CPU and MPS must fail loud immediately
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is strictly required for backbone='svd-xt'"):
        train_pilot_adapter(
            latents_dir=tmp_path,
            data_path=tmp_path / "train_trajectories.jsonl",
            output_model_path=tmp_path / "model.pt",
            report_out_path=tmp_path / "report.json",
            backbone="svd-xt",
            device="cpu",
        )

    with pytest.raises(RuntimeError, match="CUDA is strictly required for backbone='svd-xt'"):
        validate_device_and_dtype("mps", "svd-xt")

    # 2. When CUDA is available: non-cuda device string must fail loud
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(RuntimeError, match="Invalid device 'cpu' for backbone='svd-xt'"):
        validate_device_and_dtype("cpu", "svd-xt")

    with pytest.raises(RuntimeError, match="Invalid device 'mps' for backbone='svd-xt'"):
        validate_device_and_dtype("mps", "svd-xt")

    # 3. CUDA device index out of bounds must fail loud
    with pytest.raises(RuntimeError, match="CUDA device index 5 is out of bounds"):
        validate_device_and_dtype("cuda:5", "svd-xt")


def test_validate_device_and_dtype_cuda_device_strings_float16(monkeypatch):
    """Verify that device strings like 'cuda:0' correctly resolve to float16 without falling back to float32."""
    from scripts.train_svd_spatial_adapter import validate_device_and_dtype

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    # Test cuda:0 on svd-xt
    dev0, dt0 = validate_device_and_dtype("cuda:0", "svd-xt")
    assert dev0 == torch.device("cuda:0")
    assert dt0 == torch.float16, f"Expected float16 for 'cuda:0' on svd-xt, got {dt0}"

    # Test bare cuda on svd-xt
    dev_bare, dt_bare = validate_device_and_dtype("cuda", "svd-xt")
    assert dev_bare == torch.device("cuda")
    assert dt_bare == torch.float16

    # Test cuda:2 on svd-xt
    dev2, dt2 = validate_device_and_dtype("cuda:2", "svd-xt")
    assert dev2 == torch.device("cuda:2")
    assert dt2 == torch.float16

    # Test mock backbone on cuda:0 (must stay float32)
    dev_mock_cuda, dt_mock_cuda = validate_device_and_dtype("cuda:0", "mock")
    assert dev_mock_cuda == torch.device("cuda:0")
    assert dt_mock_cuda == torch.float32

    # Test mock backbone on cpu (must stay float32)
    dev_mock_cpu, dt_mock_cpu = validate_device_and_dtype("cpu", "mock")
    assert dev_mock_cpu == torch.device("cpu")
    assert dt_mock_cpu == torch.float32


def test_train_pilot_adapter_updates_metadata_and_fresh_noise(tmp_path):
    """Verify train_pilot_adapter records required optimizer updates and provenance metadata."""
    import json
    from scripts.train_svd_spatial_adapter import train_pilot_adapter

    # Set up minimal mock data
    latents_dir = tmp_path / "latents"
    latents_dir.mkdir()
    train_ids = ["0000400", "0000401"]
    val_ids = ["0000406"]

    b, t, c, h, w = 1, 25, 4, 72, 128
    for sid in train_ids + val_ids:
        torch.save(
            {
                "scenario_id": sid,
                "target_latents": torch.randn(b, t, c, h, w),
                "initial_latents": torch.randn(b, 1, c, h, w),
            },
            latents_dir / f"{sid}.pt",
        )

    traj_file = tmp_path / "trajectories.jsonl"
    with open(traj_file, "w") as f:
        for sid in train_ids + val_ids:
            line = {
                "scenario_id": sid,
                "past": [[-float(i), 0.0] for i in range(10, 0, -1)],
                "driving_instruction": "drive straight",
                "scenario_type": "straight",
                "expert_like": [[float(i), 0.0] for i in range(25)],
            }
            f.write(json.dumps(line) + "\n")

    out_model = tmp_path / "model.pt"
    out_report = tmp_path / "report.json"

    report = train_pilot_adapter(
        latents_dir=latents_dir,
        data_path=traj_file,
        output_model_path=out_model,
        report_out_path=out_report,
        backbone="mock",
        conditioning_source="knn",
        iterations=2,
        lr=1e-4,
        device="cpu",
        train_ids=train_ids,
        val_ids=val_ids,
    )

    # Verify provenance metadata
    meta = report["metadata"]
    assert meta["iterations"] == 2
    assert meta["train_sequences_count"] == 2
    assert meta["optimizer_updates_per_iteration"] == 2
    assert meta["total_optimizer_updates"] == 4
    assert meta["device"] == "cpu"
    assert meta["dtype"] == "torch.float32"
    assert meta["backbone"] == "mock"
    assert meta["model_revision"] == "main"

    # Verify saved checkpoint metadata matches
    ckpt = torch.load(out_model, map_location="cpu")
    ckpt_meta = ckpt["metadata"]
    assert ckpt_meta["iterations"] == 2
    assert ckpt_meta["train_sequences_count"] == 2
    assert ckpt_meta["optimizer_updates_per_iteration"] == 2
    assert ckpt_meta["total_optimizer_updates"] == 4
    assert ckpt_meta["device"] == "cpu"
    assert ckpt_meta["dtype"] == "torch.float32"
    assert ckpt_meta["backbone"] == "mock"
    assert ckpt_meta["model_revision"] == "main"
    assert ckpt_meta["val_sigma_grid"] == [1.0, 15.0, 70.0]
    assert ckpt_meta["val_noise_realizations"] == 2


def test_smoke_test_svd_gpu_fails_loud_on_non_cuda(monkeypatch):
    """Verify that smoke_test_svd_gpu strictly rejects CPU and MPS execution with expected RuntimeError."""
    import pytest
    from scripts.smoke_test_svd_gpu import (
        CUDA_REQUIRED_ERROR_MSG,
        main,
        run_svd_gpu_smoke_test,
        validate_cuda_device,
    )

    # 1. Direct device validation for CPU and MPS
    with pytest.raises(RuntimeError) as exc_info:
        validate_cuda_device("cpu")
    assert CUDA_REQUIRED_ERROR_MSG in str(exc_info.value)

    with pytest.raises(RuntimeError) as exc_info:
        validate_cuda_device("mps")
    assert CUDA_REQUIRED_ERROR_MSG in str(exc_info.value)

    # 2. When torch.cuda.is_available() is False, any device (including cuda:0) must fail loud
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError) as exc_info:
        validate_cuda_device("cuda:0")
    assert CUDA_REQUIRED_ERROR_MSG in str(exc_info.value)

    # 3. Direct runner invocation on CPU/MPS must raise expected RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        run_svd_gpu_smoke_test(device="cpu")
    assert CUDA_REQUIRED_ERROR_MSG in str(exc_info.value)

    with pytest.raises(RuntimeError) as exc_info:
        run_svd_gpu_smoke_test(device="mps")
    assert CUDA_REQUIRED_ERROR_MSG in str(exc_info.value)

    # 4. CLI main() entrypoint with --device cpu and --device mps
    with pytest.raises(RuntimeError) as exc_info:
        main(["--device", "cpu"])
    assert CUDA_REQUIRED_ERROR_MSG in str(exc_info.value)

    with pytest.raises(RuntimeError) as exc_info:
        main(["--device", "mps"])
    assert CUDA_REQUIRED_ERROR_MSG in str(exc_info.value)

    # 5. Device index validation when CUDA is available
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="Invalid CUDA device index 5"):
        validate_cuda_device("cuda:5")

