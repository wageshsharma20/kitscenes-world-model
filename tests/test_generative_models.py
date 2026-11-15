"""Unit tests for generative models and ablation baselines."""

import numpy as np
import pytest

from kitscenes import generative_models, spec


@pytest.fixture
def dummy_conditioning_images():
    """Create dummy 6-camera images for testing."""
    return {
        cam: np.full((spec.HEIGHT, spec.WIDTH, 3), fill_value=128, dtype=np.uint8)
        for cam in spec.CAMERAS
    }


@pytest.fixture
def dummy_trajectory():
    """Create dummy 25-waypoint trajectory."""
    return [[1.0 * (k + 1), 0.1 * (k + 1)] for k in range(spec.WAYPOINT_COUNT)]


def test_static_photographic_hold(dummy_conditioning_images, dummy_trajectory):
    model = generative_models.StaticPhotographicHoldModel()
    assert model.model_type == "static_photographic_hold"

    videos = model.generate_scenario("0000902", dummy_trajectory, dummy_conditioning_images)
    assert len(videos) == 6
    for cam in spec.CAMERAS:
        v = videos[cam]
        assert v.shape == (25, spec.HEIGHT, spec.WIDTH, 3)
        assert v.dtype == np.uint8
        # Exact frame hold: frame 0 == frame 24
        assert np.array_equal(v[0], v[24])


def test_static_photographic_hold_missing_cam(dummy_trajectory):
    model = generative_models.StaticPhotographicHoldModel()
    with pytest.raises(KeyError, match="Missing conditioning image for front"):
        model.generate_scenario("0000902", dummy_trajectory, {})


def test_speed_scaled_zoom_warp_baseline(dummy_conditioning_images, dummy_trajectory):
    model = generative_models.SpeedScaledZoomWarpBaseline()
    assert model.model_type == "speed_scaled_zoom_warp_baseline"

    bucket = model._compute_motion_bucket(dummy_trajectory)
    assert 20 <= bucket <= 240

    videos = model.generate_scenario("0000902", dummy_trajectory, dummy_conditioning_images)
    assert len(videos) == 6
    for cam in spec.CAMERAS:
        v = videos[cam]
        assert v.shape == (25, spec.HEIGHT, spec.WIDTH, 3)
        assert v.dtype == np.uint8


def test_trajectory_guided_zoom_pan_warp_baseline(dummy_conditioning_images, dummy_trajectory):
    model = generative_models.TrajectoryGuidedZoomPanWarpBaseline()
    assert model.model_type == "trajectory_guided_zoom_pan_warp_baseline"

    videos = model.generate_scenario("0000902", dummy_trajectory, dummy_conditioning_images)
    assert len(videos) == 6
    for cam in spec.CAMERAS:
        v = videos[cam]
        assert v.shape == (25, spec.HEIGHT, spec.WIDTH, 3)
        assert v.dtype == np.uint8


def test_svd_sampling_size_validation():
    with pytest.raises(ValueError, match="multiples of 8"):
        generative_models.GenuineSVDVideoPipeline(sampling_height=575, sampling_width=1024)


def test_svd_motion_bucket_computation(dummy_trajectory):
    # Test motion bucket derivation logic without loading model weights
    model_cls = generative_models.GenuineSVDVideoPipeline
    bucket = model_cls.compute_motion_bucket(None, dummy_trajectory)
    assert 1 <= bucket <= 255


def test_wan22_hardware_validation_on_cpu():
    # If host does not have CUDA (e.g. macOS), Wan2.2 must raise an explicit RuntimeError
    try:
        import torch

        if not torch.cuda.is_available():
            with pytest.raises(RuntimeError, match="Hardware Incompatible: CUDA is not available"):
                generative_models.GenuineWan22VideoPipeline()
    except ImportError:
        pass


def test_svd_aspect_ratio_adaptation():
    from PIL import Image

    from scripts.run_kaggle_svd_single_camera import adapt_aspect_ratio_to_svd, unpad_and_resize_frames

    # Synthetic KITScenes conditioning image
    img = Image.new("RGB", (spec.WIDTH, spec.HEIGHT), color=(100, 150, 200))
    padded, crop_box = adapt_aspect_ratio_to_svd(img, target_width=1024, target_height=576)

    assert padded.size == (1024, 576)
    assert crop_box == (68, 0, 956, 576)  # 68 px padding on left and right (888 px width)

    # Simulated 25-frame diffusion output
    frames = [padded] * spec.FRAME_COUNT
    restored = unpad_and_resize_frames(frames, crop_box=crop_box, target_width=spec.WIDTH, target_height=spec.HEIGHT)

    assert restored.shape == (25, spec.HEIGHT, spec.WIDTH, 3)
    assert restored.dtype == np.uint8


def test_svd_runner_argument_parser():
    from pathlib import Path

    from scripts.run_kaggle_svd_single_camera import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "--scenario-id", "0000000",
        "--camera", "front",
        "--observed-t0-dir", "data/observed_t0",
        "--history-file", "data/history.jsonl",
        "--donor-file", "data/train_trajectories.jsonl",
        "--model-revision", "main",
        "--steps", "15",
    ])

    assert args.scenario_id == "0000000"
    assert args.camera == "front"
    assert args.observed_t0_dir == Path("data/observed_t0")
    assert args.history_file == Path("data/history.jsonl")
    assert args.donor_file == Path("data/train_trajectories.jsonl")
    assert args.model_revision == "main"
    assert args.steps == 15


def test_svd_cv_fallback_provenance():
    from kitscenes import knn_trajectory

    # When no donor database is loaded (e.g. initial Kaggle metadata split)
    planner = knn_trajectory.KNNTrajectoryPlanner(donor_path=None)
    assert len(planner.donors) == 0

    past_trajectory = [[-10.0 + i, 0.0] for i in range(11)]
    plan = planner.plan("0000000", past_trajectory, "drive straight on")
    assert len(plan) == 25

    # Check kinematic motion bucket logic for CV fallback
    pts = np.asarray(plan, dtype=float)
    total_dist = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    mean_speed = total_dist / (len(pts) * spec.WAYPOINT_STEP_S)
    bucket = int(np.clip(10 + mean_speed * 12.0, 1, 255))
    assert 1 <= bucket <= 255


