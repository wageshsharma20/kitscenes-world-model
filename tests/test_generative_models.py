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
