"""Unit tests for Epona autonomous driving world model integration.

Tests verify:
1. Output shapes: exactly (25, 568, 876, 3), dtype uint8.
2. FP16 dtype selection when on CUDA (Tesla T4 hardware optimization).
3. Temporal patch size 6: internal generation of 30 frames and truncation to 25 frames.
4. Spatial scale 32 divisibility and target resolution resizing.
5. Trajectory conditioning kinematics and token encoding.
6. Robust error handling for missing conditioning frames and unaligned waypoints.
7. Integration with WorldModelPipeline and batch packaging.
"""

import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from kitscenes import epona_engine, spec, video_engine


def make_dummy_image(height: int = 568, width: int = 876) -> np.ndarray:
    """Create a synthetic RGB test image."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[: height // 2, :, 0] = 70
    arr[: height // 2, :, 1] = 130
    arr[: height // 2, :, 2] = 200
    arr[height // 2 :, :, :] = 50
    return arr


def make_straight_waypoints(speed: float = 10.0) -> list[list[float]]:
    """Create 25 straight-line future waypoints at 5 Hz (0.2s step)."""
    step_m = speed * spec.WAYPOINT_STEP_S
    return [[step_m * (k + 1), 0.0] for k in range(spec.WAYPOINT_COUNT)]


def test_epona_camera_video_output_shape_and_dtype():
    """Verify output video matches challenge specification: (25, 568, 876, 3) uint8."""
    model = epona_engine.EponaWorldModel(mock_mode=True, device="cpu")
    cond_frame = make_dummy_image()
    waypoints = make_straight_waypoints(speed=12.0)

    video = model.generate_camera_video(
        conditioning_frame=cond_frame,
        waypoints=waypoints,
        camera="front",
    )

    assert isinstance(video, np.ndarray)
    assert video.shape == (spec.FRAME_COUNT, spec.HEIGHT, spec.WIDTH, 3)
    assert video.dtype == np.uint8
    assert video.min() >= 0
    assert video.max() <= 255


def test_epona_generate_scenario_surround_views():
    """Verify generate_scenario generates valid videos for all 6 surround cameras."""
    model = epona_engine.EponaWorldModel(mock_mode=True, device="cpu")
    dummy_img = make_dummy_image()
    cond_images = {cam: dummy_img for cam in spec.CAMERAS}
    waypoints = make_straight_waypoints(speed=8.0)

    results = model.generate_scenario(
        scenario_id="0000042",
        trajectory=waypoints,
        weather="clear",
        conditioning_images=cond_images,
    )

    assert set(results.keys()) == set(spec.CAMERAS)
    for _cam_name, cam_video in results.items():
        assert cam_video.shape == (spec.FRAME_COUNT, spec.HEIGHT, spec.WIDTH, 3)
        assert cam_video.dtype == np.uint8


def test_optimal_torch_dtype_selection():
    """Verify FP16 precision is selected for CUDA (Tesla T4) and FP32 for CPU."""
    # CUDA requires FP16 for Tesla T4 Turing tensor cores
    cuda_dtype = epona_engine.get_optimal_torch_dtype("cuda")
    assert cuda_dtype == torch.float16

    cuda0_dtype = epona_engine.get_optimal_torch_dtype("cuda:0")
    assert cuda0_dtype == torch.float16

    # CPU uses FP32
    cpu_dtype = epona_engine.get_optimal_torch_dtype("cpu")
    assert cpu_dtype == torch.float32


def test_convert_weights_to_fp16():
    """Verify bfloat16 tensors in checkpoints are properly converted to float16."""
    dummy_state_dict = {
        "weight_bf16": torch.randn(4, 4, dtype=torch.bfloat16),
        "weight_fp32": torch.randn(4, 4, dtype=torch.float32),
        "bias": torch.zeros(4, dtype=torch.bfloat16),
        "submodule": {
            "inner_weight": torch.ones(2, 2, dtype=torch.bfloat16),
        },
    }

    converted = epona_engine.convert_weights_to_fp16(dummy_state_dict)

    assert converted["weight_bf16"].dtype == torch.float16
    assert converted["weight_fp32"].dtype == torch.float32
    assert converted["bias"].dtype == torch.float16
    assert converted["submodule"]["inner_weight"].dtype == torch.float16


def test_epona_cuda_fp16_initialization():
    """Verify EponaWorldModel selects FP16 and device properly when CUDA is requested."""
    with patch("torch.cuda.is_available", return_value=True):
        model = epona_engine.EponaWorldModel(device="cuda", mock_mode=True)
        assert model.torch_dtype == torch.float16
        assert model.device == "cuda"


def test_temporal_patch_size_and_truncation():
    """Verify that Epona generates 30 frames (5 blocks of 6) and truncates to 25."""
    model = epona_engine.EponaWorldModel(mock_mode=True, device="cpu")
    cond_frame = make_dummy_image()
    waypoints = make_straight_waypoints()

    output = model.generate_camera_video(cond_frame, waypoints, camera="front")

    # Internal raw generation was 30 frames
    assert model.last_raw_frame_count == 30
    # Truncated to 25 frames
    assert model.last_truncated_frame_count == 25
    assert output.shape[0] == 25


def test_spatial_scale_divisibility_by_32():
    """Verify generator resolution divisibility by 32."""
    # Divisible by 32: 288 % 32 == 0, 448 % 32 == 0
    model1 = epona_engine.EponaWorldModel(resolution=(288, 448), mock_mode=True)
    assert model1.resolution == (288, 448)

    # 512 % 32 == 0, 768 % 32 == 0
    model2 = epona_engine.EponaWorldModel(resolution=(512, 768), mock_mode=True)
    assert model2.resolution == (512, 768)

    # Invalid resolution not divisible by 32
    with pytest.raises(ValueError, match="divisible by 32"):
        epona_engine.EponaWorldModel(resolution=(500, 700), mock_mode=True)


def test_trajectory_kinematics_derivation_and_padding():
    """Verify trajectory kinematics derivation from 25 waypoints extrapolated to 30."""
    waypoints = make_straight_waypoints(speed=10.0)
    kin = epona_engine.derive_trajectory_kinematics(waypoints, pad_to_count=30)

    # Length must be padded to 30 steps
    assert kin.poses.shape == (30, 2)
    assert kin.yaws.shape == (30, 1)
    assert kin.speeds.shape == (30,)
    assert kin.yaw_rates.shape == (30,)
    assert kin.features.shape == (30, 5)

    # Straight line speeds ~ 10 m/s
    assert np.allclose(kin.speeds, 10.0, atol=1e-3)
    assert np.allclose(kin.yaw_rates, 0.0, atol=1e-3)


def test_trajectory_encoder_module():
    """Verify PyTorch TrajectoryEncoder creates step and chunk tokens."""
    encoder = epona_engine.TrajectoryEncoder(input_dim=5, embed_dim=128, temporal_patch_size=6)
    dummy_features = torch.randn(2, 30, 5)

    step_tokens, chunk_tokens = encoder(dummy_features)

    assert step_tokens.shape == (2, 30, 128)
    assert chunk_tokens.shape == (2, 5, 128)  # 30 / 6 = 5 chunks


def test_dcae_encoder_and_decoder_shapes():
    """Verify 32x downsampling in DCAEEncoder and 32x upsampling in DCAEDecoder."""
    encoder = epona_engine.DCAEEncoder(in_channels=3, latent_dim=32)
    decoder = epona_engine.DCAEDecoder(latent_dim=32, out_channels=3)

    # Input divisible by 32: (1, 3, 64, 64)
    x = torch.randn(1, 3, 64, 64)
    z = encoder(x)
    assert z.shape == (1, 32, 2, 2)  # 64 / 32 = 2

    x_rec = decoder(z)
    assert x_rec.shape == (1, 3, 64, 64)


def test_vis_dit_autoregressive_chunks():
    """Verify VisDiT predicts 30 latent frames in 5 chunks of 6 frames."""
    dit = epona_engine.VisDiT(latent_dim=32, embed_dim=128, temporal_patch_size=6, num_chunks=5)
    init_latent = torch.randn(1, 32, 4, 4)
    chunk_tokens = torch.randn(1, 5, 128)

    out_latents = dit(init_latent, chunk_tokens)
    assert out_latents.shape == (1, 30, 32, 4, 4)


def test_error_missing_conditioning_frame():
    """Verify error is raised when conditioning frame is missing or None."""
    model = epona_engine.EponaWorldModel(mock_mode=True)
    waypoints = make_straight_waypoints()

    with pytest.raises(ValueError, match="conditioning_frame cannot be None"):
        model.generate_camera_video(conditioning_frame=None, waypoints=waypoints)


def test_error_missing_scenario_conditioning_images():
    """Verify error is raised when conditioning_images dictionary is empty or incomplete."""
    model = epona_engine.EponaWorldModel(mock_mode=True)
    waypoints = make_straight_waypoints()

    # Empty conditioning images
    with pytest.raises(ValueError, match="observed t=0 frame"):
        model.generate_scenario(scenario_id="0000001", trajectory=waypoints, conditioning_images={})

    # Missing some cameras
    partial_cond = {"front": make_dummy_image()}
    with pytest.raises(ValueError, match="rear"):
        model.generate_scenario(
            scenario_id="0000001", trajectory=waypoints, conditioning_images=partial_cond
        )


def test_error_unaligned_waypoints():
    """Verify error is raised when waypoints are not exactly (25, 2) or have non-finite values."""
    model = epona_engine.EponaWorldModel(mock_mode=True)
    cond = make_dummy_image()

    # 24 waypoints instead of 25
    bad_count_waypoints = [[1.0, 0.0] for _ in range(24)]
    with pytest.raises(ValueError, match="unaligned waypoints"):
        model.generate_camera_video(cond, bad_count_waypoints)

    # Wrong dimension: (25, 3)
    bad_dim_waypoints = [[1.0, 0.0, 0.0] for _ in range(25)]
    with pytest.raises(ValueError, match="unaligned waypoints"):
        model.generate_camera_video(cond, bad_dim_waypoints)

    # Non-finite values
    nan_waypoints = make_straight_waypoints()
    nan_waypoints[5][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        model.generate_camera_video(cond, nan_waypoints)


def test_world_model_pipeline_with_epona_backbone(tmp_path):
    """Verify WorldModelPipeline creates and encodes videos using backbone='epona'."""
    pipeline = video_engine.WorldModelPipeline(backbone="epona", device="cpu")
    assert pipeline.backbone_name == "epona"
    assert isinstance(pipeline.generator, epona_engine.EponaWorldModel)

    dummy_img = make_dummy_image()
    cond_images = {cam: dummy_img for cam in spec.CAMERAS}
    waypoints = make_straight_waypoints(speed=10.0)

    encoded = pipeline.generate_and_save_scenario(
        scenario_id="0000005",
        trajectory=waypoints,
        output_dir=tmp_path / "renders",
        conditioning_images=cond_images,
    )

    assert set(encoded.keys()) == set(spec.CAMERAS)
    for _cam, video_file in encoded.items():
        assert video_file.exists()
        assert video_file.stat().st_size > 0
        info = video_engine.media.probe_video(video_file)
        assert info.frame_count == spec.FRAME_COUNT
        assert info.width == spec.WIDTH
        assert info.height == spec.HEIGHT
        assert info.fps == pytest.approx(spec.FPS, abs=spec.FPS_TOLERANCE)


def test_batch_generate_and_package_epona_refuses_without_conditioning(tmp_path):
    """Verify batch_generate_and_package refuses to run 'epona' without conditioning images."""
    traj_file = tmp_path / "trajectories.jsonl"
    traj_data = {
        "scenario_id": "0000000",
        "future_trajectory": make_straight_waypoints(),
    }
    traj_file.write_text(json.dumps(traj_data) + "\n")

    with pytest.raises(ValueError, match="epona backbone needs the observed t=0 frame"):
        video_engine.batch_generate_and_package(
            trajectories_file=traj_file,
            output_dir=tmp_path / "renders",
            backbone="epona",
            scenario_ids=["0000000"],
            allow_partial=True,
        )
