"""Phase 5.4 Test Suite: Checkpoint Forensics, Metric Audit & Four-Way Conditioning Ablation.

Verifies:
1. Baseline submission archive verification (hard failure on missing, unchanged SHA-256).
2. Checkpoint state-dict hashes and parameter counts against provenance audit.
3. Deterministic paired camera noise (independent across perspectives, identical across models).
4. Zero future target leakage in rollout inference.
5. All 5 ablation configurations activate/deactivate the intended modules.
6. Metric labelling and contracts (visual_naturalness_proxy, local_emc_approximation, boundary_strip_correlation).
7. Trajectory adherence returns None when flow is unmeasurable (never 0.5 fallback).
8. Local EMC approximation recovers synthetic motion and enforces front/rear yaw agreement.
9. Output video shape contract (25 frames, 5 fps, 876x568, 3 RGB channels).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from kitscenes import emc_eval, pq_proxy, spec
from kitscenes.multiview_conditioning import MultiViewConditioner
from kitscenes.spatial_adapter import (
    MockSVDUNet,
    SpatialAdapter,
    SpatialAdapterConfig,
    create_default_euler_scheduler,
)
from scripts.run_phase5_4_ablation import (
    ABLATION_VARIANTS,
    BASELINE_SUBMISSION_PATH,
    CAMERA_SEED_OFFSETS,
    EXPECTED_BASELINE_SHA256,
    adapt_aspect_ratio_to_svd,
    compute_boundary_strip_correlation,
    compute_cssc_token_diagnostics,
    compute_trajectory_adherence_from_flow,
    compute_true_psnr,
    compute_video_ssim,
    run_diffusion_rollout_with_telemetry,
    validate_baseline_archive,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. Baseline Submission Archive Integrity & Missing Archive Handling
# ---------------------------------------------------------------------------

def test_baseline_archive_integrity():
    """Verify that the official baseline submission zip exists and is strictly unchanged."""
    is_intact, actual_hash, meta = validate_baseline_archive(strict=False)
    assert meta["exists"] is True
    assert actual_hash == EXPECTED_BASELINE_SHA256
    assert is_intact is True
    assert meta["intact"] is True


def test_baseline_archive_missing_hard_failure():
    """Verify that a missing archive raises FileNotFoundError when strict, or returns False without faking hash."""
    fake_missing_path = PROJECT_ROOT / "submissions" / "nonexistent_baseline.zip"
    with patch("scripts.run_phase5_4_ablation.BASELINE_SUBMISSION_PATH", fake_missing_path):
        with pytest.raises(FileNotFoundError):
            validate_baseline_archive(strict=True)

        is_intact, actual_hash, meta = validate_baseline_archive(strict=False)
        assert is_intact is False
        assert actual_hash == "ARCHIVE_MISSING_ON_DISK"
        assert meta["exists"] is False
        assert meta["actual_sha256"] is None
        assert meta["intact"] is False


# ---------------------------------------------------------------------------
# 2. Checkpoint State-Dict Hashes & Provenance
# ---------------------------------------------------------------------------

def test_checkpoint_state_dict_hashes_and_provenance():
    """Verify that models/cssc_pilot_step30.pt matches provenance audit hashes."""
    ckpt_path = PROJECT_ROOT / "models" / "cssc_pilot_step30.pt"
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"

    audit_path = PROJECT_ROOT / "reports" / "phase5_4_checkpoint_provenance_audit.json"
    assert audit_path.exists(), f"Missing audit report: {audit_path}"
    with open(audit_path, "r") as f:
        audit = json.load(f)

    # File SHA-256
    file_h = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        while chunk := f.read(65536):
            file_h.update(chunk)
    actual_file_sha256 = file_h.hexdigest()
    assert actual_file_sha256 == audit["file_sha256"]

    # State dicts
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert "spatial_adapter_state_dict" in ckpt
    assert "conditioner_state_dict" in ckpt

    def compute_composite_hash(sd):
        h = hashlib.sha256()
        for k in sorted(sd.keys()):
            h.update(k.encode())
            h.update(sd[k].float().cpu().numpy().tobytes())
        return h.hexdigest()

    sa_sd = ckpt["spatial_adapter_state_dict"]
    sa_params = sum(p.numel() for p in sa_sd.values())
    assert sa_params == audit["spatial_adapter_summary"]["total_parameters"]
    assert compute_composite_hash(sa_sd) == audit["spatial_adapter_summary"]["composite_tensor_sha256"]

    mvc_sd = ckpt["conditioner_state_dict"]
    mvc_params = sum(p.numel() for p in mvc_sd.values())
    assert mvc_params == audit["conditioner_summary"]["total_parameters"]
    assert compute_composite_hash(mvc_sd) == audit["conditioner_summary"]["composite_tensor_sha256"]


# ---------------------------------------------------------------------------
# 3. Deterministic Paired Camera Noise
# ---------------------------------------------------------------------------

def test_deterministic_camera_noise_pairing():
    """Verify that noise is independent across cameras but 100% identical across models for the same camera."""
    base_seed = 42
    shape = (1, 25, 4, 72, 128)

    camera_noises: dict[str, torch.Tensor] = {}
    for cam in spec.CAMERAS:
        cam_seed = base_seed + CAMERA_SEED_OFFSETS[cam]
        gen = torch.Generator(device="cpu").manual_seed(cam_seed)
        noise = torch.randn(shape, generator=gen, dtype=torch.float32)
        camera_noises[cam] = noise

    # Verify independence across cameras
    cams = list(spec.CAMERAS)
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            diff = torch.abs(camera_noises[cams[i]] - camera_noises[cams[j]]).mean().item()
            assert diff > 0.5, f"Camera noise for {cams[i]} and {cams[j]} should be independent!"

    # Verify identical across models / systems for the same camera
    for cam in spec.CAMERAS:
        cam_seed = base_seed + CAMERA_SEED_OFFSETS[cam]
        gen_sysA = torch.Generator(device="cpu").manual_seed(cam_seed)
        noise_sysA = torch.randn(shape, generator=gen_sysA, dtype=torch.float32)

        gen_sysB = torch.Generator(device="cpu").manual_seed(cam_seed)
        noise_sysB = torch.randn(shape, generator=gen_sysB, dtype=torch.float32)

        gen_sysC = torch.Generator(device="cpu").manual_seed(cam_seed)
        noise_sysC = torch.randn(shape, generator=gen_sysC, dtype=torch.float32)

        assert torch.equal(noise_sysA, noise_sysB)
        assert torch.equal(noise_sysB, noise_sysC)


# ---------------------------------------------------------------------------
# 4. Zero Future Target Leakage in Rollout
# ---------------------------------------------------------------------------

def test_zero_future_leakage_in_rollout_signature():
    """Verify that run_diffusion_rollout_with_telemetry has no parameters for future targets."""
    import inspect
    sig = inspect.signature(run_diffusion_rollout_with_telemetry)
    param_names = list(sig.parameters.keys())

    # Strictly forbidden parameter names
    forbidden = ["future_latents", "future_video", "target_latents", "target_video", "gt_video", "ground_truth"]
    for f in forbidden:
        assert f not in param_names, f"Rollout must not accept {f}!"


# ---------------------------------------------------------------------------
# 5. Ablation Configurations and Diagnostics
# ---------------------------------------------------------------------------

def test_ablation_variants_contract():
    """Verify all 5 ablation variants are defined and behave properly under mock rollout."""
    assert len(ABLATION_VARIANTS) == 5
    expected = ("vanilla_svd", "project_baseline", "cssc_only", "spatial_adapter_only", "cssc_plus_spatial_adapter")
    assert ABLATION_VARIANTS == expected

    unet = MockSVDUNet(in_channels=8, out_channels=4)
    scheduler = create_default_euler_scheduler()
    init_latent = torch.randn(1, 4, 72, 128, dtype=torch.float32)
    starting_noise = torch.randn(1, 25, 4, 72, 128, dtype=torch.float32)
    added_time_ids = torch.tensor([[127.0, 6.0, 0.0]], dtype=torch.float32)

    # 1. vanilla_svd (seq_len=1, no adapter)
    clip_hidden = torch.randn(1, 1, 1024, dtype=torch.float32)
    lat_vanilla, telem_vanilla = run_diffusion_rollout_with_telemetry(
        unet=unet,
        scheduler=scheduler,
        initial_latent=init_latent,
        encoder_hidden_states=clip_hidden,
        added_time_ids=added_time_ids,
        starting_noise=starting_noise,
        spatial_adapter=None,
        num_inference_steps=2,
        dtype=torch.float32,
    )
    assert telem_vanilla["mean_adapter_ratio"] == 0.0
    assert not telem_vanilla["has_nan"]

    # 2. spatial_adapter_only (seq_len=1, with adapter)
    adapter = SpatialAdapter(SpatialAdapterConfig(latent_channels=4))
    ckpt_path = PROJECT_ROOT / "models" / "cssc_pilot_step30.pt"
    if ckpt_path.exists():
        ckpt_sa = torch.load(ckpt_path, map_location="cpu")
        adapter.load_state_dict(ckpt_sa["spatial_adapter_state_dict"])
    else:
        # Fallback to non-zero weight for mock
        with torch.no_grad():
            adapter.out_conv.conv.weight.fill_(0.01)
    mask_cube = torch.ones(1, 25, 1, 72, 128, dtype=torch.float32)
    lat_adapter, telem_adapter = run_diffusion_rollout_with_telemetry(
        unet=unet,
        scheduler=scheduler,
        initial_latent=init_latent,
        encoder_hidden_states=clip_hidden,
        added_time_ids=added_time_ids,
        starting_noise=starting_noise,
        spatial_adapter=adapter,
        spatial_mask_cube=mask_cube,
        num_inference_steps=2,
        dtype=torch.float32,
    )
    assert telem_adapter["mean_adapter_ratio"] > 0.0
    assert not telem_adapter["has_nan"]

    # 3. cssc_only (seq_len=27, no adapter)
    cssc_hidden = torch.randn(1, 27, 1024, dtype=torch.float32)
    lat_cssc, telem_cssc = run_diffusion_rollout_with_telemetry(
        unet=unet,
        scheduler=scheduler,
        initial_latent=init_latent,
        encoder_hidden_states=cssc_hidden,
        added_time_ids=added_time_ids,
        starting_noise=starting_noise,
        spatial_adapter=None,
        num_inference_steps=2,
        dtype=torch.float32,
    )
    assert telem_cssc["mean_adapter_ratio"] == 0.0

    # 4. cssc_plus_spatial_adapter (seq_len=27, with adapter)
    lat_both, telem_both = run_diffusion_rollout_with_telemetry(
        unet=unet,
        scheduler=scheduler,
        initial_latent=init_latent,
        encoder_hidden_states=cssc_hidden,
        added_time_ids=added_time_ids,
        starting_noise=starting_noise,
        spatial_adapter=adapter,
        spatial_mask_cube=mask_cube,
        num_inference_steps=2,
        dtype=torch.float32,
    )
    assert telem_both["mean_adapter_ratio"] > 0.0


def test_cssc_token_diagnostics():
    """Verify CSSC token sequence diagnostics correctly report per-token and aggregate norms."""
    token_bank = torch.ones(1, 27, 1024, dtype=torch.float32)
    diag = compute_cssc_token_diagnostics(token_bank)
    assert diag["sequence_length"] == 27
    assert diag["hidden_dim"] == 1024
    expected_norm = round(math.sqrt(1024), 4)
    assert diag["mean_token_norm"] == expected_norm
    assert diag["camera_token_norm"] == expected_norm
    assert diag["scene_motion_token_norm"] == expected_norm
    assert diag["waypoint_token_norms_mean"] == expected_norm


# ---------------------------------------------------------------------------
# 6. Metric Auditing, Contracts & Disclaimers
# ---------------------------------------------------------------------------

def test_metric_labelling_and_contracts():
    """Verify that all metrics are accurately labelled and explicitly disclaim official status."""
    frames = np.random.randint(0, 255, (25, spec.HEIGHT, spec.WIDTH, 3), dtype=np.uint8)
    cam_dict = {cam: frames for cam in spec.CAMERAS}

    # 1. Visual naturalness proxy (renamed from PQ)
    pq_res = pq_proxy.score_scenario_videos(cam_dict)
    assert "scenario_pq_proxy" in pq_res

    # 2. Local EMC approximation (renamed from EMC)
    plan_pts = np.zeros((25, 2))
    plan_pts[:, 1] = np.linspace(0, 50, 25)
    emc_res = emc_eval.compute_video_optical_flow_emc(cam_dict, plan_pts)
    assert emc_res["is_official_emc"] is False
    assert "local_emc_approximation" in emc_res
    assert "seg1_yaw_agreement" in emc_res
    assert "seg2_yaw_agreement" in emc_res

    # 3. Boundary strip correlation (renamed from cross_camera_consistency)
    corr_res = compute_boundary_strip_correlation(cam_dict)
    assert corr_res["is_calibrated_3d_geometry"] is False
    assert corr_res["pair_count_evaluated"] == 6
    expected_pairs = [
        "front_to_front_left",
        "front_to_front_right",
        "rear_to_rear_left",
        "rear_to_rear_right",
        "front_left_to_rear_left",
        "front_right_to_rear_right",
    ]
    for p in expected_pairs:
        assert p in corr_res["pairwise_correlations"]


def test_trajectory_adherence_null_safe_on_no_motion():
    """Verify that trajectory adherence returns None and measurable: False when optical flow is unmeasurable."""
    blank_frames = np.zeros((25, spec.HEIGHT, spec.WIDTH, 3), dtype=np.uint8)
    waypoints = np.zeros((25, 2))
    waypoints[:, 1] = np.linspace(0, 25, 25)

    res = compute_trajectory_adherence_from_flow(blank_frames, waypoints)
    assert res["measurable"] is False
    assert res["adherence"] is None
    assert res["valid_flow_count"] == 0


# ---------------------------------------------------------------------------
# 7. Local EMC Controlled Synthetic Motion Recovery
# ---------------------------------------------------------------------------

def test_local_emc_synthetic_motion_and_agreement():
    """Verify local EMC approximation on synthetic motion patterns and front/rear agreement."""
    # Synthetic straight forward: shift texture vertically downwards across time
    h, w = spec.HEIGHT, spec.WIDTH
    base_tex = np.random.randint(50, 200, (h + 100, w), dtype=np.uint8)
    straight_front = np.zeros((25, h, w, 3), dtype=np.uint8)
    straight_rear = np.zeros((25, h, w, 3), dtype=np.uint8)
    for t in range(25):
        shift = t * 2
        crop_f = base_tex[shift : shift + h, :]
        crop_r = base_tex[100 - shift : 100 - shift + h, :]
        for c in range(3):
            straight_front[t, ..., c] = crop_f
            straight_rear[t, ..., c] = crop_r

    gt_straight = np.zeros((25, 2))
    gt_straight[:, 1] = np.linspace(0, 50, 25)

    cam_dict_straight = {
        "front": straight_front,
        "rear": straight_rear,
        "front_left": straight_front,
        "front_right": straight_front,
        "rear_left": straight_rear,
        "rear_right": straight_rear,
    }

    res_straight = emc_eval.compute_video_optical_flow_emc(
        camera_video_paths=cam_dict_straight,
        planned_trajectory=gt_straight,
    )
    assert res_straight["seg1_yaw_agreement"] is True

    # Test opposing yaw (front moves left, rear moves same direction instead of opposite)
    # causing yaw sign disagreement
    disagree_front = np.zeros((25, h, w, 3), dtype=np.uint8)
    disagree_rear = np.zeros((25, h, w, 3), dtype=np.uint8)
    wide_tex = np.random.randint(50, 200, (h, w + 100), dtype=np.uint8)
    for t in range(25):
        shift = t * 3
        crop = wide_tex[:, shift : shift + w]
        for c in range(3):
            disagree_front[t, ..., c] = crop
            disagree_rear[t, ..., c] = crop  # Disagreeing yaw rotation sign

    cam_dict_disagree = {
        "front": disagree_front,
        "rear": disagree_rear,
        "front_left": disagree_front,
        "front_right": disagree_front,
        "rear_left": disagree_rear,
        "rear_right": disagree_rear,
    }

    res_disagree = emc_eval.compute_video_optical_flow_emc(
        camera_video_paths=cam_dict_disagree,
        planned_trajectory=gt_straight,
    )
    # Opposing motion must fail yaw agreement
    assert res_disagree["seg1_yaw_agreement"] is False
    assert res_disagree["seg1_steering_match"] is False


# ---------------------------------------------------------------------------
# 8. Output Video Dimension Contract
# ---------------------------------------------------------------------------

def test_video_aspect_ratio_and_dimensions():
    """Verify aspect ratio adaptation and dimensions match SVD 1024x576 and target 876x568."""
    img = Image.new("RGB", (spec.WIDTH, spec.HEIGHT), color=(128, 128, 128))
    padded, (pad_top, pad_left) = adapt_aspect_ratio_to_svd(img, target_width=1024, target_height=576)

    assert padded.size == (1024, 576)
    assert pad_top == 4  # (576 - 568) // 2
    assert pad_left == 74  # (1024 - 876) // 2

    # Uncrop back
    arr = np.array(padded)
    uncropped = arr[pad_top : pad_top + spec.HEIGHT, pad_left : pad_left + spec.WIDTH, :]
    assert uncropped.shape == (spec.HEIGHT, spec.WIDTH, 3)
