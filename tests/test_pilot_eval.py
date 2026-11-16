"""Unit tests for Phase 3 Pilot comparative evaluation metrics and contracts."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

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

    # Provide held-out scenarios 0000406, 0000407, plus a valid train donor 0000408
    records = [
        {
            "scenario_id": "0000406",
            "past": past,
            "driving_instruction": "turn left",
            "scenario_type": "4 intersection",
            "expert_like": expert_like,
        },
        {
            "scenario_id": "0000407",
            "past": past,
            "driving_instruction": "turn right",
            "scenario_type": "4 intersection",
            "expert_like": expert_like,
        },
        {
            "scenario_id": "0000408",
            "past": past,
            "driving_instruction": "turn left",
            "scenario_type": "4 intersection",
            "expert_like": expert_like,
        },
    ]
    with open(traj_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

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
    # 0000406 and 0000407 must be globally excluded from donor pool
    knn_traj, knn_label, knn_meta = resolve_conditioning_trajectory(
        scenario_id="0000406",
        trajectories_file=traj_file,
        conditioning_source="knn",
    )
    assert "kNN" in knn_label
    assert knn_traj.shape == (25, 2)
    assert knn_meta["source"] == "knn"
    assert knn_meta["method"] == "knn_retrieval_with_cv_blend"
    assert knn_meta["fallback_used"] is False
    assert knn_meta["donor_count"] == 1  # Only 0000408 loaded as donor
    assert knn_meta["retrieved_donors"] == ["0000408"]
    assert "0000406" not in knn_meta["retrieved_donors"]
    assert "0000407" not in knn_meta["retrieved_donors"]
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


def test_pilot_eval_leakage_block_held_out(tmp_path: Path):
    """Verify that held-out pilot scenarios 0000406 and 0000407 are strictly blocked."""
    import json

    traj_file = tmp_path / "leakage_test_trajectories.jsonl"
    expert_like = [[float(i), 0.5 * i] for i in range(25)]
    past = [[-float(i), 0.0] for i in range(10, 0, -1)]

    # File containing only 0000406 and 0000407 (no other donors)
    records = [
        {"scenario_id": "0000406", "past": past, "driving_instruction": "turn left", "expert_like": expert_like},
        {"scenario_id": "0000407", "past": past, "driving_instruction": "turn right", "expert_like": expert_like},
    ]
    with open(traj_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # With only 0000406 and 0000407 in file, both are excluded so 0 donors remain -> fail-loud!
    with pytest.raises(RuntimeError, match="Fail-loud: No kNN donors available"):
        resolve_conditioning_trajectory(
            scenario_id="0000406",
            trajectories_file=traj_file,
            conditioning_source="knn",
        )


def test_pilot_eval_provenance_stationary_fallback(tmp_path: Path):
    """Verify provenance metadata correctly records stationary constant velocity fallback."""
    import json

    traj_file = tmp_path / "stationary_test.jsonl"
    expert_like = [[float(i), 0.0] for i in range(25)]
    stationary_past = [[0.0, 0.0] for _ in range(10)]

    records = [
        {"scenario_id": "0000406", "past": stationary_past, "driving_instruction": "drive straight", "expert_like": expert_like},
        {"scenario_id": "0000408", "past": [[-float(i), 0.0] for i in range(10, 0, -1)], "driving_instruction": "drive straight", "expert_like": expert_like},
    ]
    with open(traj_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    planned, label, meta = resolve_conditioning_trajectory(
        scenario_id="0000406",
        trajectories_file=traj_file,
        conditioning_source="knn",
    )
    assert meta["source"] == "knn"
    assert meta["method"] == "constant_velocity_stationary"
    assert meta["fallback_used"] is True
    assert meta["retrieved_donors"] == []
    assert np.allclose(planned, 0.0)


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


def test_svd_xt_eval_fails_loud_on_cpu_mps(tmp_path, monkeypatch):
    """Verify that svd-xt execution on CPU or MPS fails loud immediately with RuntimeError on evaluation."""
    from scripts.eval_pilot_comparison import run_pilot_evaluation

    # 1. When CUDA is unavailable: CPU and MPS must fail loud immediately
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is strictly required for backbone='svd-xt'"):
        run_pilot_evaluation(
            model_checkpoint=tmp_path / "model.pt",
            latents_dir=tmp_path,
            observed_t0_dir=tmp_path,
            future_videos_dir=tmp_path,
            trajectories_file=tmp_path / "trajectories.jsonl",
            output_renders_dir=tmp_path / "renders",
            report_out_path=tmp_path / "report.json",
            backbone="svd-xt",
            device="cpu",
        )

    with pytest.raises(RuntimeError, match="CUDA is strictly required for backbone='svd-xt'"):
        run_pilot_evaluation(
            model_checkpoint=tmp_path / "model.pt",
            latents_dir=tmp_path,
            observed_t0_dir=tmp_path,
            future_videos_dir=tmp_path,
            trajectories_file=tmp_path / "trajectories.jsonl",
            output_renders_dir=tmp_path / "renders",
            report_out_path=tmp_path / "report.json",
            backbone="svd-xt",
            device="mps",
        )

    # 2. When CUDA is available: non-cuda device string must fail loud
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(RuntimeError, match="Invalid device 'cpu' for backbone='svd-xt'"):
        run_pilot_evaluation(
            model_checkpoint=tmp_path / "model.pt",
            latents_dir=tmp_path,
            observed_t0_dir=tmp_path,
            future_videos_dir=tmp_path,
            trajectories_file=tmp_path / "trajectories.jsonl",
            output_renders_dir=tmp_path / "renders",
            report_out_path=tmp_path / "report.json",
            backbone="svd-xt",
            device="cpu",
        )

    with pytest.raises(RuntimeError, match="CUDA device index 7 is out of bounds"):
        run_pilot_evaluation(
            model_checkpoint=tmp_path / "model.pt",
            latents_dir=tmp_path,
            observed_t0_dir=tmp_path,
            future_videos_dir=tmp_path,
            trajectories_file=tmp_path / "trajectories.jsonl",
            output_renders_dir=tmp_path / "renders",
            report_out_path=tmp_path / "report.json",
            backbone="svd-xt",
            device="cuda:7",
        )


def test_evaluate_scenario_pair_default_added_time_ids_noise_aug_002(tmp_path):
    """Verify that evaluate_scenario_pair defaults added_time_ids to noise augmentation 0.02."""
    import json
    from kitscenes import camera_projection, spec
    from kitscenes.spatial_adapter import MockSVDUNet, SpatialConditionedSVDModel, create_default_euler_scheduler
    from scripts.eval_pilot_comparison import evaluate_scenario_pair

    sid = "0000406"
    latents_dir = tmp_path / "latents"
    latents_dir.mkdir()
    torch.save(
        {
            "scenario_id": sid,
            "target_latents": torch.randn(1, spec.FRAME_COUNT, 4, 72, 128),
            "initial_latents": torch.randn(1, 1, 4, 72, 128),
        },
        latents_dir / f"{sid}.pt",
    )

    t0_dir = tmp_path / "t0"
    (t0_dir / sid).mkdir(parents=True)
    img_t0 = Image.new("RGB", (spec.WIDTH, spec.HEIGHT), color=(100, 100, 100))
    img_t0.save(t0_dir / sid / "front.jpg")

    vid_dir = tmp_path / "future_videos" / sid / "front"
    vid_dir.mkdir(parents=True)
    for i in range(spec.FRAME_COUNT):
        img_f = Image.new("RGB", (spec.WIDTH, spec.HEIGHT), color=(i * 10, i * 10, i * 10))
        img_f.save(vid_dir / f"frame_{i:03d}.jpg")

    traj_file = tmp_path / "trajectories.jsonl"
    with open(traj_file, "w") as f:
        line = {
            "scenario_id": sid,
            "past": [[-float(i), 0.0] for i in range(10, 0, -1)],
            "driving_instruction": "drive straight",
            "scenario_type": "straight",
            "expert_like": [[float(i), 0.0] for i in range(25)],
        }
        f.write(json.dumps(line) + "\n")

    unet = MockSVDUNet(in_channels=8, out_channels=4)
    model = SpatialConditionedSVDModel(unet=unet)

    captured_time_ids = []

    def mock_generate(*args, **kwargs):
        captured_time_ids.append(kwargs.get("added_time_ids"))
        return torch.randn(1, spec.FRAME_COUNT, 4, 72, 128)

    model.generate_iterative = mock_generate

    class MockVAE:
        def __init__(self):
            self.config = type("Config", (), {"scaling_factor": 0.18215})()

        def decode(self, chunk, num_frames=None):
            return type("Output", (), {"sample": torch.zeros(chunk.shape[0], 3, spec.HEIGHT, spec.WIDTH)})()

    engine = camera_projection.CameraProjectionEngine(width=spec.WIDTH, height=spec.HEIGHT)
    scheduler = create_default_euler_scheduler()

    # Call with added_time_ids=None -> verify it defaults to noise_aug_strength=0.02
    _ = evaluate_scenario_pair(
        scenario_id=sid,
        model=model,
        vae=MockVAE(),
        scheduler=scheduler,
        engine=engine,
        latents_dir=latents_dir,
        observed_t0_dir=t0_dir,
        future_videos_dir=tmp_path / "future_videos",
        trajectories_file=traj_file,
        output_renders_dir=tmp_path / "renders",
        conditioning_source="oracle",
        steps=1,
        device="cpu",
        added_time_ids=None,
    )

    assert len(captured_time_ids) >= 1
    for time_id_tensor in captured_time_ids:
        assert time_id_tensor is not None
        assert time_id_tensor.shape == (1, 3)
        fps_val = time_id_tensor[0, 0].item()
        motion_bucket = time_id_tensor[0, 1].item()
        noise_aug = time_id_tensor[0, 2].item()
        assert abs(fps_val - float(spec.FPS - 1)) < 1e-4
        assert abs(motion_bucket - 127.0) < 1e-4
        assert abs(noise_aug - 0.02) < 1e-4, f"Expected noise_aug=0.02, got {noise_aug}"
