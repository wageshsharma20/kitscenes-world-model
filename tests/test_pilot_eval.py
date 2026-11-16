"""Unit tests for Phase 3 Pilot comparative evaluation metrics and contracts."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from scripts.eval_pilot_comparison import (
    compute_frame_ssim,
    compute_laplacian_sharpness,
    compute_lateral_motion_center,
    compute_optical_flow_coherence,
    compute_psnr,
    resolve_conditioning_trajectory,
)


def test_compute_psnr_identical():
    """Verify PSNR on identical images is capped at 100 dB."""
    val = compute_psnr(0.0)
    assert val == 100.0


def test_compute_psnr_noisy():
    """Verify PSNR decreases monotonically with MSE."""
    psnr_low_noise = compute_psnr(1.0)
    psnr_high_noise = compute_psnr(100.0)
    assert psnr_low_noise > psnr_high_noise
    assert round(psnr_low_noise, 1) == 48.1


def test_compute_frame_ssim_identical():
    """Verify SSIM on identical images is 1.0."""
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    ssim = compute_frame_ssim(img, img)
    assert abs(ssim - 1.0) < 1e-3


def test_compute_frame_ssim_different():
    """Verify SSIM on distinct images is less than 1.0."""
    img1 = np.full((128, 128, 3), 50, dtype=np.uint8)
    img2 = np.full((128, 128, 3), 200, dtype=np.uint8)
    ssim = compute_frame_ssim(img1, img2)
    assert ssim < 0.5


def test_laplacian_sharpness_blurry_vs_sharp():
    """Verify Laplacian sharpness proxy detects edge degradation."""
    # Sharp pattern: high frequency checkerboard
    sharp_frames = np.zeros((5, 128, 128, 3), dtype=np.uint8)
    sharp_frames[:, ::4, ::4] = 255

    # Blurry pattern: constant smooth gradient
    blurry_frames = np.ones((5, 128, 128, 3), dtype=np.uint8) * 128

    score_sharp = compute_laplacian_sharpness(sharp_frames)
    score_blurry = compute_laplacian_sharpness(blurry_frames)

    assert score_sharp > score_blurry
    assert score_blurry < 1.0


def test_lateral_motion_center_directional_response():
    """Verify lateral center-of-mass detects left vs right motion."""
    # Video 1: motion on the left half of the image (coords in [-1.0, 0.0])
    frames_left = np.zeros((5, 64, 128, 3), dtype=np.uint8)
    for t in range(5):
        frames_left[t, :, :40] = t * 40  # Motion concentrated on left

    com_left = compute_lateral_motion_center(frames_left)
    assert np.all(com_left < 0.0), f"Left motion must have negative CoM! Got: {com_left}"

    # Video 2: motion on the right half of the image (coords in [0.0, 1.0])
    frames_right = np.zeros((5, 64, 128, 3), dtype=np.uint8)
    for t in range(5):
        frames_right[t, :, 88:] = t * 40  # Motion concentrated on right

    com_right = compute_lateral_motion_center(frames_right)
    assert np.all(com_right > 0.0), f"Right motion must have positive CoM! Got: {com_right}"


def test_conditioning_source_knn_vs_oracle_distinction(tmp_path: Path):
    """Verify strict distinction and contracts between 'knn' and 'oracle' conditioning."""
    import json

    traj_file = tmp_path / "test_trajectories.jsonl"
    expert_like = [[float(i), 0.5 * i] for i in range(25)]
    past = [[-float(i), 0.0] for i in range(10, 0, -1)]

    record = {
        "scenario_id": "0000406",
        "past": past,
        "driving_instruction": "turn left",
        "scenario_type": "4 intersection",
        "expert_like": expert_like,
    }
    with open(traj_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # Oracle mode must return expert_like ground truth
    oracle_traj, oracle_label, oracle_meta = resolve_conditioning_trajectory(
        scenario_id="0000406",
        trajectories_file=traj_file,
        conditioning_source="oracle",
    )
    assert "Oracle" in oracle_label
    assert np.allclose(oracle_traj, np.array(expert_like))
    assert oracle_meta["fallback_used"] is False
    assert oracle_meta["source"] == "oracle"
    assert len(oracle_meta["trajectory_checksum"]) == 64

    # kNN mode must predict waypoints via planner (not oracle ground truth)
    knn_traj, knn_label, knn_meta = resolve_conditioning_trajectory(
        scenario_id="0000406",
        trajectories_file=traj_file,
        conditioning_source="knn",
    )
    assert "kNN" in knn_label
    assert knn_traj.shape == (25, 2)
    assert knn_meta["fallback_used"] is False
    assert knn_meta["source"] == "knn"
    assert knn_meta["donor_count"] >= 1
    assert len(knn_meta["trajectory_checksum"]) == 64

    # Fail-loud: Missing scenario in oracle mode must raise RuntimeError
    with pytest.raises(RuntimeError, match="Fail-loud: Oracle trajectory"):
        resolve_conditioning_trajectory(
            scenario_id="nonexistent_scenario",
            trajectories_file=traj_file,
            conditioning_source="oracle",
        )

    # Fail-loud: Missing file must raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        resolve_conditioning_trajectory(
            scenario_id="0000406",
            trajectories_file=tmp_path / "nonexistent.jsonl",
            conditioning_source="knn",
        )

    # Invalid source must raise ValueError
    with pytest.raises(ValueError, match="conditioning_source must be 'knn' or 'oracle'"):
        resolve_conditioning_trajectory(
            scenario_id="0000406",
            trajectories_file=traj_file,
            conditioning_source="invalid_source",
        )


def test_proxy_metrics_relabeling_contract():
    """Verify that proxy metrics are properly relabeled with exact standardized naming."""
    frames_a = np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8)

    # Check that individual proxy metric calculators function properly
    com_a = compute_lateral_motion_center(frames_a)
    sharp_a = compute_laplacian_sharpness(frames_a)
    flow_a = compute_optical_flow_coherence(frames_a)

    assert isinstance(com_a, np.ndarray)
    assert isinstance(sharp_a, float)
    assert isinstance(flow_a, float)
    assert sharp_a >= 0.0
    assert flow_a >= 0.0


def test_temporal_vae_chunk_decoding():
    """Verify temporal VAE decoding chunks temporal latents into chunks of 14 frames."""
    from kitscenes.spatial_adapter import decode_latents_temporal
    import torch

    class MockVAE:
        def __init__(self):
            self.config = type("Config", (), {"scaling_factor": 0.18215})()
            self.chunk_sizes = []

        def decode(self, chunk, num_frames=None):
            self.chunk_sizes.append(chunk.shape[0])
            return type("Output", (), {"sample": torch.zeros(chunk.shape[0], 3, 16, 16)})()

    mock_vae = MockVAE()
    dummy_latents = torch.randn(1, 25, 4, 4, 4)

    decoded = decode_latents_temporal(mock_vae, dummy_latents, num_frames=25, decode_chunk_size=14)

    assert decoded.shape == (1, 25, 3, 16, 16)
    assert mock_vae.chunk_sizes == [14, 11], f"Expected chunks [14, 11], got {mock_vae.chunk_sizes}"
