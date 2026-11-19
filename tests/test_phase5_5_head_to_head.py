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
    run_head_to_head_evaluation,
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


def test_mock_head_to_head_run():
    """Execute a complete mock rollout of Phase 5.5 to verify dataflow, report generation and contract compliance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        out_dir = tmppath / "renders"
        rep_json = tmppath / "reports" / "phase5_5.json"
        rep_md = tmppath / "reports" / "phase5_5.md"

        results = run_head_to_head_evaluation(
            scenarios=["0000900"],
            checkpoint_path=PROJECT_ROOT / "models" / "cssc_pilot_step30.pt",
            device="cpu",
            mock=True,
            steps=1,
            base_seed=42,
            extract_missing_gt=False,
            output_dir=out_dir,
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
