"""Unit and integration tests for Phase 5.5 Head-to-Head Candidate Evaluation."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from kitscenes import knn_trajectory, spec
from scripts.run_candidate_head_to_head import (
    CANDIDATES,
    CAMERA_SEED_OFFSETS,
    DEFAULT_HEAD_TO_HEAD_SCENARIOS,
    compute_true_psnr,
    compute_video_ssim,
    compute_boundary_strip_correlation,
    compute_trajectory_adherence_from_flow,
    run_prediction_stage,
    run_evaluation_stage,
    validate_baseline_archive,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_candidates_contract():
    """Verify that only the 3 top candidates are evaluated and SpatialAdapter is excluded."""
    assert CANDIDATES == ("vanilla_svd", "project_baseline", "cssc_only")
    assert "spatial_adapter_only" not in CANDIDATES
    assert "cssc_plus_spatial_adapter" not in CANDIDATES


def test_camera_seed_offsets():
    """Verify that each camera has a distinct deterministic seed offset."""
    for cam in spec.CAMERAS:
        assert cam in CAMERA_SEED_OFFSETS
    offsets = list(CAMERA_SEED_OFFSETS.values())
    assert len(offsets) == len(set(offsets))


def test_baseline_archive_integrity():
    """Verify baseline submission archive SHA-256."""
    intact, actual_hash, meta = validate_baseline_archive(strict=False)
    assert meta["exists"] is True
    assert intact is True
    assert actual_hash == "26119a957a37123db3f7dac5a74fab50efb7051c8e7312f1cad1c384343ef10e"


def test_knn_self_exclusion_on_validation_split():
    """Verify that kNN trajectory planner never retrieves the evaluated validation scenario."""
    planner = knn_trajectory.KNNTrajectoryPlanner()
    for sid in DEFAULT_HEAD_TO_HEAD_SCENARIOS:
        pts, prov = planner.plan_with_provenance(
            scenario_id=sid,
            past_trajectory=[[0.0, 0.0], [0.0, 1.0]],
            driving_instruction="drive straight on",
            scenario_type="regular",
        )
        assert sid not in prov["retrieved_donors"], f"Leakage detected: {sid} in retrieved donors!"


def test_mock_two_stage_head_to_head_run():
    """Execute decoupled Stage A (Prediction) and Stage B (Evaluation) mock rollout."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        out_dir = tmppath / "renders"
        manifest_path = tmppath / "prediction_manifest.json"
        rep_json = tmppath / "reports" / "phase5_5.json"
        rep_md = tmppath / "reports" / "phase5_5.md"

        # Stage A: Prediction (no ground truth required)
        manifest = run_prediction_stage(
            scenarios=["0000900"],
            checkpoint_path=PROJECT_ROOT / "models" / "cssc_pilot_step30.pt",
            device="cpu",
            mock=True,
            steps=1,
            base_seed=42,
            output_dir=out_dir,
            manifest_path=manifest_path,
        )

        assert manifest_path.exists()
        assert "0000900" in manifest["scenarios"]
        assert manifest["stage"] == "prediction"
        assert manifest["inference_mathematics"]["fps_micro_conditioning"] == "fps - 1 = 4"

        # Stage B: Evaluation (requires ground truth for scenario 0000900)
        results = run_evaluation_stage(
            manifest_path=manifest_path,
            val_fut_dir=PROJECT_ROOT / "data" / "val_future_videos",
            val_meta_path=PROJECT_ROOT / "data" / "val_manifest.jsonl",
            report_json=rep_json,
            report_md=rep_md,
        )

        assert rep_json.exists()
        assert rep_md.exists()
        assert "0000900" in results["scenarios"]
        s_data = results["scenarios"]["0000900"]
        for cand in CANDIDATES:
            assert cand in s_data["candidates"]
            cand_eval = s_data["candidates"][cand]
            assert "mean_psnr" in cand_eval
            assert "mean_ssim" in cand_eval
            assert "boundary_strip_correlation" in cand_eval
            assert "local_emc_approximation" in cand_eval
            assert cand_eval["is_official_pq"] is False

        assert "aggregate_statistics" in results
        assert "pairwise_comparisons" in results
        assert "cssc_only_vs_project_baseline" in results["pairwise_comparisons"]


def test_missing_ground_truth_raises_explicit_error():
    """Verify that incomplete or missing ground truth raises RuntimeError instead of silent black padding."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        fake_manifest = tmppath / "manifest.json"
        fake_fut_dir = tmppath / "fake_fut_dir"
        fake_fut_dir.mkdir(parents=True)

        manifest_data = {
            "scenarios_evaluated": ["0000900"],
            "candidates": ["vanilla_svd"],
            "checkpoint_path": "models/cssc_pilot_step30.pt",
            "checkpoint_sha256": "9e65c397...",
            "checkpoint_training_step": 6,
            "device": "cpu",
            "mock_mode": True,
            "denoising_steps": 1,
            "base_seed": 42,
            "scenarios": {
                "0000900": {
                    "scenario_id": "0000900",
                    "instruction": "drive straight on",
                    "planned_waypoints": [[0.0, 0.0]] * 25,
                    "planned_mean_speed": 10.0,
                    "trajectory_motion_bucket": 127,
                    "video_paths": {"vanilla_svd": {c: str(tmppath / f"{c}.mp4") for c in spec.CAMERAS}},
                }
            }
        }
        with open(fake_manifest, "w") as f:
            json.dump(manifest_data, f)

        # Should raise RuntimeError because ground truth is missing
        with pytest.raises(RuntimeError) as exc_info:
            run_evaluation_stage(
                manifest_path=fake_manifest,
                val_fut_dir=fake_fut_dir,
                val_meta_path=PROJECT_ROOT / "data" / "val_manifest.jsonl",
                report_json=tmppath / "rep.json",
                report_md=tmppath / "rep.md",
            )
        assert "Silent black padding is prohibited" in str(exc_info.value)
