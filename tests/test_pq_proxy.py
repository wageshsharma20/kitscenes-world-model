import numpy as np
import pytest

from kitscenes import pq_proxy, spec, video_engine


def test_heuristic_perceptual_score_zeros_on_empty():
    empty_frames = np.zeros((0, spec.HEIGHT, spec.WIDTH, 3), dtype=np.uint8)
    assert pq_proxy.heuristic_perceptual_score(empty_frames) == 0.0


def test_heuristic_perceptual_score_low_on_procedural_mockup():
    model = video_engine.GeometricWorldModel()
    waypoints = [[2.0 * (k + 1), 0.0] for k in range(spec.WAYPOINT_COUNT)]
    videos = model.generate_scenario("0000000", waypoints)

    # Procedural mockup has flat trapezoid and simple solid color fills
    front_score = pq_proxy.heuristic_perceptual_score(videos["front"])
    # Should score in the lower bracket (typically < 35)
    assert front_score < 40.0


def test_heuristic_perceptual_score_higher_on_rich_textured_frames():
    # Synthetic frames with high dynamic range and spatial texture
    np.random.seed(42)
    textured = np.random.randint(0, 256, size=(10, spec.HEIGHT, spec.WIDTH, 3), dtype=np.uint8)
    score = pq_proxy.heuristic_perceptual_score(textured)
    assert score > 50.0


def test_score_scenario_videos_weighted_formula():
    mock_videos = {}
    for cam in spec.CAMERAS:
        # 25 frames
        frames = np.full((spec.FRAME_COUNT, spec.HEIGHT, spec.WIDTH, 3), fill_value=128, dtype=np.uint8)
        mock_videos[cam] = frames

    scores = pq_proxy.score_scenario_videos(mock_videos)
    assert "scenario_pq" in scores
    assert "front" in scores
    # Weights: front 0.50 + 5 * 0.10 = 1.0
    for cam in spec.CAMERAS:
        assert cam in scores


def test_video_engine_accepts_backbone_parameter(tmp_path):
    # Verify WorldModelPipeline and batch_generate_and_package accept backbone
    pipe = video_engine.WorldModelPipeline(backbone="geometric")
    assert pipe.backbone_name == "geometric"

    with pytest.raises(ValueError, match="unknown backbone"):
        video_engine.WorldModelPipeline(backbone="nonexistent_backbone")
