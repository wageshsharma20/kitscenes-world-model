"""Tests for kNN/CV Trajectory Engine and evaluation metrics."""

import math
from pathlib import Path

import numpy as np
import pytest

from kitscenes import spec
from kitscenes.knn_trajectory import (
    ConstantVelocityPlanner,
    KNNTrajectoryPlanner,
    ManeuverClass,
    classify_instruction,
    compute_l2_metrics,
    compute_mms_score,
)


def _make_straight_history(speed: float = 10.0, n_points: int = 21) -> np.ndarray:
    """Generate synthetic straight-line past trajectory."""
    dt = 0.2
    history = np.empty((n_points, 2), dtype=float)
    for i in range(n_points):
        t = (i - (n_points - 1)) * dt
        history[i] = [speed * t, 0.0]
    return history


def test_classify_instruction():
    assert classify_instruction("drive straight on") == ManeuverClass.STRAIGHT
    assert classify_instruction("cruise along road") == ManeuverClass.STRAIGHT
    assert classify_instruction("turn left at light") == ManeuverClass.TURN_LEFT
    assert classify_instruction("turn right at intersection") == ManeuverClass.TURN_RIGHT
    assert classify_instruction("use left lane") == ManeuverClass.LANE_CHANGE_LEFT
    assert classify_instruction("use right lane") == ManeuverClass.LANE_CHANGE_RIGHT
    assert classify_instruction("use middle lane") == ManeuverClass.LANE_CHANGE_MIDDLE
    assert classify_instruction("overtake truck driving on the right") == ManeuverClass.OVERTAKE
    assert classify_instruction("u-turn") == ManeuverClass.U_TURN
    assert classify_instruction("stop at red light") == ManeuverClass.STOP


def test_constant_velocity_planner():
    planner = ConstantVelocityPlanner()
    hist = _make_straight_history(speed=12.0)
    future = planner.plan(hist)

    assert future.shape == (25, 2)
    # At 12 m/s, step = 0.2s -> delta_x = 2.4m per step
    expected_step_x = 12.0 * 0.2
    assert future[0, 0] == pytest.approx(expected_step_x, abs=1e-3)
    assert future[0, 1] == pytest.approx(0.0, abs=1e-3)
    assert future[-1, 0] == pytest.approx(expected_step_x * 25, abs=1e-3)

    # Jerk is strictly zero for pure constant velocity
    diff3 = np.linalg.norm(np.diff(future, n=3, axis=0), axis=1)
    assert np.all(diff3 < 1e-6)


def test_constant_velocity_planner_stationary():
    planner = ConstantVelocityPlanner()
    hist = np.zeros((21, 2), dtype=float)
    future = planner.plan(hist)

    assert future.shape == (25, 2)
    assert np.allclose(future, 0.0)


def test_knn_planner_donor_loading():
    train_path = Path("data/train_trajectories.jsonl")
    if not train_path.exists():
        pytest.skip("train_trajectories.jsonl not found")

    planner = KNNTrajectoryPlanner(donor_path=train_path)
    # Scenario 0000762 has only 18 waypoints in training dataset, leaving 499 full donors
    assert len(planner.donors) == 499

    # Ensure all donors strictly adhere to train split
    for donor in planner.donors:
        assert donor.scenario_id in spec.TRAIN_SCENARIO_ID_SET
        assert donor.future_trajectory.shape == (25, 2)
        assert donor.p0.shape == (2,)


def test_knn_planner_plan_execution():
    train_path = Path("data/train_trajectories.jsonl")
    if not train_path.exists():
        pytest.skip("train_trajectories.jsonl not found")

    planner = KNNTrajectoryPlanner(donor_path=train_path)
    hist = _make_straight_history(speed=15.0)

    # Test right turn
    traj_turn = planner.plan("0000902", hist, "turn right", k=5)
    assert traj_turn.shape == (25, 2)
    assert np.all(np.isfinite(traj_turn))

    # Waypoint 0 heading direction check
    v_init = np.array([1.0, 0.0])
    v_wp0 = traj_turn[0] - hist[-1]
    cos_sim = float(np.dot(v_init, v_wp0) / (np.linalg.norm(v_init) * np.linalg.norm(v_wp0)))
    assert cos_sim >= 0.85

    # Speeds must be under max speed limit
    steps = np.linalg.norm(np.diff(traj_turn, axis=0), axis=1)
    speeds = steps / 0.2
    from kitscenes.knn_trajectory import MAX_SPEED
    assert np.all(speeds <= MAX_SPEED + 1.0)


def test_compute_l2_metrics():
    a = np.zeros((25, 2))
    b = np.ones((25, 2)) * 3.0  # dist = sqrt(9+9) = sqrt(18) = 4.2426...
    metrics = compute_l2_metrics(a, b)

    expected_dist = math.sqrt(18.0)
    assert metrics["l2_mean"] == pytest.approx(expected_dist, abs=1e-3)
    assert metrics["l2_median"] == pytest.approx(expected_dist, abs=1e-3)
    assert metrics["l2_endpoint"] == pytest.approx(expected_dist, abs=1e-3)


def test_compute_mms_score():
    # Exact match gives base score 10
    ref = np.zeros((25, 2))
    for i in range(25):
        ref[i] = [(i + 1) * 2.0, 0.0]

    score_perfect = compute_mms_score(ref, ref)
    assert score_perfect == 10.0

    # Direction mismatch at waypoint 0 gives 0.0
    wrong_dir = ref.copy()
    wrong_dir[0] = [-2.0, 0.0]  # opposite direction
    score_wrong_dir = compute_mms_score(wrong_dir, ref)
    assert score_wrong_dir == 0.0

    # Unmatched straight trajectory when reference turns left gets 3.5 floor
    ref_turn = np.zeros((25, 2))
    heading = 0.0
    x, y = 0.0, 0.0
    for i in range(25):
        heading += 0.08  # turning left
        x += 2.0 * math.cos(heading)
        y += 2.0 * math.sin(heading)
        ref_turn[i] = [x, y]

    pred_straight = np.zeros((25, 2))
    for i in range(25):
        pred_straight[i] = [(i + 1) * 2.0, 0.0]

    score_unmatched = compute_mms_score(pred_straight, ref_turn)
    assert score_unmatched == 3.5
