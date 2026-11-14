"""Tests for the photographic warp backbone: real frames, geometry from the plan."""

import numpy as np
import pytest

from kitscenes import spec, video_engine


def gradient_frame(seed: int = 0) -> np.ndarray:
    ys, xs = np.mgrid[0 : spec.HEIGHT, 0 : spec.WIDTH]
    frame = np.zeros((spec.HEIGHT, spec.WIDTH, 3), dtype=np.uint8)
    frame[..., 0] = (xs * 255 // spec.WIDTH).astype(np.uint8)
    frame[..., 1] = (ys * 255 // spec.HEIGHT).astype(np.uint8)
    frame[..., 2] = seed
    return frame


@pytest.fixture
def frames():
    return {cam: gradient_frame(i) for i, cam in enumerate(spec.CAMERAS)}


def forward_waypoints(speed_mps: float = 10.0):
    step = speed_mps * spec.WAYPOINT_STEP_S
    return [[step * (k + 1), 0.0] for k in range(spec.WAYPOINT_COUNT)]


def test_plan_profile_is_monotone_and_starts_at_rest():
    distances, yaws = video_engine.PhotographicWarpModel.plan_profile(forward_waypoints())

    assert distances[0] == 0.0
    assert np.all(np.diff(distances) > 0)
    assert np.allclose(yaws, 0.0)


def test_still_plan_reproduces_the_source_frame(frames):
    model = video_engine.PhotographicWarpModel()
    still = [[0.0, 0.0] for _ in range(spec.WAYPOINT_COUNT)]

    video = model.generate_scenario("0000000", still, conditioning_images=frames)

    assert video["front"].shape == (spec.FRAME_COUNT, spec.HEIGHT, spec.WIDTH, 3)
    assert np.array_equal(video["front"][0], frames["front"])
    # A zero-displacement homography still round-trips through float32 sampling.
    assert np.abs(video["front"][-1].astype(int) - frames["front"].astype(int)).max() <= 1


def test_forward_motion_moves_the_ground_plane(frames):
    model = video_engine.PhotographicWarpModel()

    video = model.generate_scenario(
        "0000000", forward_waypoints(), conditioning_images=frames
    )

    change = np.mean(np.abs(video["front"][-1].astype(float) - frames["front"].astype(float)))
    assert change > 1.0


def test_ground_plane_expands_towards_the_bottom(frames):
    """Driving forward makes road markings stream out of the horizon and down the frame."""
    model = video_engine.PhotographicWarpModel()
    distances = np.array([0.0] + [1.0] * (spec.FRAME_COUNT - 1))
    frame = np.zeros((spec.HEIGHT, spec.WIDTH, 3), dtype=np.uint8)
    frame[420:460, :, :] = 255  # a band on the road about 7 m ahead of the camera

    video = model.render_camera("front", frame, distances, np.zeros(spec.FRAME_COUNT))

    def band_top(frames: np.ndarray) -> int:
        bright = np.where(frames[:, :, 0].mean(axis=1) > 127)[0]
        assert bright.size, "the band left the frame entirely"
        return int(bright.min())

    assert band_top(video[-1]) >= band_top(video[0]) + 10


def test_missing_camera_is_refused():
    model = video_engine.PhotographicWarpModel()

    with pytest.raises(ValueError, match="rear"):
        model.generate_scenario(
            "0000000",
            forward_waypoints(),
            conditioning_images={"front": gradient_frame()},
        )


def test_photo_warp_is_a_pipeline_backbone():
    pipeline = video_engine.WorldModelPipeline(backbone="photo_warp")

    assert isinstance(pipeline.generator, video_engine.PhotographicWarpModel)
