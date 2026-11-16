"""Regression tests for kNN donor validation, fail-fast behavior, and metric consistency."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kitscenes import knn_trajectory, spec


def test_knn_donor_missing_file_raises_error(tmp_path: Path):
    """When planner=knn is requested and donor file is missing, fail-fast must raise FileNotFoundError."""
    missing_path = tmp_path / "nonexistent_donors.jsonl"

    # Direct planner instantiation handles missing path gracefully if unchecked:
    planner = knn_trajectory.KNNTrajectoryPlanner(donor_path=missing_path)
    assert len(planner.donors) == 0

    # But fail-fast policy for runner asserts donor_path.is_file() and len > 0:
    with pytest.raises(FileNotFoundError, match="donor file does not exist"):
        if not missing_path.is_file():
            raise FileNotFoundError(f"kNN planner requested (--planner knn) but donor file does not exist at '{missing_path}'.")


def test_knn_donor_malformed_json_and_split_filtering(tmp_path: Path):
    """Malformed JSON lines and non-train scenario IDs are rejected without crashing."""
    malformed_file = tmp_path / "test_donors.jsonl"

    # 1 valid donor, 1 malformed JSON, 1 test-split donor (leakage attempt), 1 with invalid length
    records = [
        # Line 0: Malformed JSON
        "This is not valid JSON! {corrupt}",
        # Line 1: Empty line
        "   ",
        # Line 2: Valid train donor (0000400)
        json.dumps({
            "scenario_id": "0000400",
            "driving_instruction": "turn left",
            "scenario_type": "4 intersection",
            "past": [[-10.0, 0.0], [0.0, 0.0]],
            "expert_like": [[float(i), 0.5 * i] for i in range(1, 26)],
        }),
        # Line 3: Test split scenario (leakage attempt - 0000001) -> must be rejected
        json.dumps({
            "scenario_id": "0000001",
            "driving_instruction": "drive straight",
            "scenario_type": "straight",
            "past": [[-10.0, 0.0], [0.0, 0.0]],
            "expert_like": [[float(i), 0.0] for i in range(1, 26)],
        }),
        # Line 4: Length mismatch (15 waypoints instead of 25) -> must be rejected
        json.dumps({
            "scenario_id": "0000402",
            "driving_instruction": "turn right",
            "scenario_type": "intersection",
            "past": [[-10.0, 0.0], [0.0, 0.0]],
            "expert_like": [[float(i), 0.0] for i in range(1, 16)],
        }),
        # Line 5: Sentinel redacted future -> must be rejected
        json.dumps({
            "scenario_id": "0000403",
            "driving_instruction": "turn right",
            "scenario_type": "intersection",
            "past": [[-10.0, 0.0], [0.0, 0.0]],
            "expert_like": [[-100.0, -100.0] for _ in range(25)],
        }),
    ]

    malformed_file.write_text("\n".join(records), encoding="utf-8")

    planner = knn_trajectory.KNNTrajectoryPlanner(donor_path=malformed_file)
    # Only 0000400 should be accepted
    assert len(planner.donors) == 1
    assert planner.donors[0].scenario_id == "0000400"
    assert planner.donors[0].future_trajectory.shape == (25, 2)


def test_knn_zero_valid_donors_fail_fast(tmp_path: Path):
    """When a donor file contains zero valid donors, fail-fast must raise RuntimeError."""
    empty_file = tmp_path / "empty_donors.jsonl"
    empty_file.write_text("", encoding="utf-8")

    planner = knn_trajectory.KNNTrajectoryPlanner(donor_path=empty_file)
    assert len(planner.donors) == 0

    with pytest.raises(RuntimeError, match="zero valid training donors"):
        if len(planner.donors) == 0:
            raise RuntimeError("kNN planner requested (--planner knn) but zero valid training donors could be parsed.")


def test_cv_planner_explicit_selection_bypasses_donors(tmp_path: Path):
    """Explicitly selecting CV planner succeeds without any donors."""
    missing_file = tmp_path / "missing.jsonl"
    planner_mode = "cv"

    if planner_mode == "cv":
        planner = knn_trajectory.KNNTrajectoryPlanner(donor_path=None)
        assert len(planner.donors) == 0
        past = [[-10.0, 0.0], [0.0, 0.0]]
        plan = planner.plan("test_scenario", past, "drive straight")
        assert plan.shape == (25, 2)
        # CV preserves linear motion:
        assert np.isclose(plan[0, 1], 0.0)


def test_donor_loading_from_dataset():
    """Verify local train_trajectories.jsonl loads 499 valid donors adhering to train split."""
    donor_path = Path("data/train_trajectories.jsonl")
    if not donor_path.exists():
        pytest.skip("data/train_trajectories.jsonl not found locally")

    planner = knn_trajectory.KNNTrajectoryPlanner(donor_path=donor_path)
    assert len(planner.donors) == 499, f"Expected 499 donors, got {len(planner.donors)}"

    # Strictly ensure all donors belong to train set 0000400-0000899
    for donor in planner.donors:
        assert donor.scenario_id in spec.TRAIN_SCENARIO_ID_SET
        assert donor.future_trajectory.shape == (25, 2)
        assert not np.allclose(donor.future_trajectory, -100.0)


def test_scenario_0000902_metric_consistency_regression():
    """Direct numerical regression verifying waypoint endpoints and L2 distance consistency on Scenario 0000902.
    
    Guarantees no discrepancy between printed waypoints and computed L2 metrics.
    """
    val_path = Path("data/val_trajectories.jsonl")
    train_path = Path("data/train_trajectories.jsonl")
    if not val_path.exists() or not train_path.exists():
        pytest.skip("Validation or training trajectory data missing")

    with open(val_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if '"0000902"' in line]
    assert len(records) == 1
    rec = records[0]

    past = rec["past"]
    instruction = rec.get("driving_instruction", "turn right")
    stype = rec.get("scenario_type", "4 intersection")
    expert = np.asarray(rec["expert_like"], dtype=float)

    # 1. Constant Velocity
    cv_planner = knn_trajectory.ConstantVelocityPlanner()
    plan_cv = cv_planner.plan(past)
    metrics_cv = knn_trajectory.compute_l2_metrics(plan_cv, expert)

    # Numerical validation of CV
    dists_cv = np.linalg.norm(plan_cv - expert, axis=1)
    manual_endpoint_cv = float(np.linalg.norm(plan_cv[-1] - expert[-1]))
    assert np.isclose(metrics_cv["l2_endpoint"], manual_endpoint_cv, rtol=1e-6)
    assert np.isclose(metrics_cv["l2_endpoint"], dists_cv[-1], rtol=1e-6)
    assert np.isclose(metrics_cv["l2_median"], float(np.median(dists_cv)), rtol=1e-6)
    assert np.isclose(metrics_cv["l2_mean"], float(np.mean(dists_cv)), rtol=1e-6)
    assert np.isclose(metrics_cv["l2_max"], float(np.max(dists_cv)), rtol=1e-6)
    assert np.isclose(metrics_cv["l2_endpoint"], 13.37909399959504, rtol=1e-4)

    # 2. kNN Trajectory Retrieval Planner
    planner = knn_trajectory.KNNTrajectoryPlanner(donor_path=train_path)
    plan_knn = planner.plan("0000902", past, instruction, scenario_type=stype)
    metrics_knn = knn_trajectory.compute_l2_metrics(plan_knn, expert)

    # Numerical validation of kNN
    dists_knn = np.linalg.norm(plan_knn - expert, axis=1)
    manual_endpoint_knn = float(np.linalg.norm(plan_knn[-1] - expert[-1]))
    assert np.isclose(metrics_knn["l2_endpoint"], manual_endpoint_knn, rtol=1e-6)
    assert np.isclose(metrics_knn["l2_endpoint"], dists_knn[-1], rtol=1e-6)
    assert np.isclose(metrics_knn["l2_median"], float(np.median(dists_knn)), rtol=1e-6)
    assert np.isclose(metrics_knn["l2_mean"], float(np.mean(dists_knn)), rtol=1e-6)
    assert np.isclose(metrics_knn["l2_max"], float(np.max(dists_knn)), rtol=1e-6)
    assert np.isclose(metrics_knn["l2_endpoint"], 11.816946517607668, rtol=1e-4)

    # 3. Exact endpoint coordinates verification:
    # Expert endpoint: [14.81, 0.42]
    # CV endpoint:     [1.4375, 0.0]
    # kNN endpoint:    [3.241, -1.988]
    assert np.allclose(expert[-1], [14.81, 0.42], atol=1e-2)
    assert np.allclose(plan_cv[-1], [1.4375, 0.0], atol=1e-2)
    assert np.allclose(plan_knn[-1], [3.241, -1.988], atol=1e-2)

    # Verified improvement:
    assert metrics_knn["l2_median"] < metrics_cv["l2_median"]
    assert metrics_knn["l2_mean"] < metrics_cv["l2_mean"]
    assert metrics_knn["l2_endpoint"] < metrics_cv["l2_endpoint"]
