import json
import math

import numpy as np
import pytest

from kitscenes import spec, trajectories

STEP = spec.WAYPOINT_STEP_S


def history(speed=10.0, yaw_rate=0.0, count=8):
    """Odometry for a vehicle driving at ``speed`` with a constant yaw rate."""
    points, x, y, heading = [], 0.0, 0.0, 0.0
    for _ in range(count):
        x += speed * STEP * math.cos(heading)
        y += speed * STEP * math.sin(heading)
        points.append([x, y])
        heading += yaw_rate * STEP
    return points


def test_returns_one_waypoint_per_timestep():
    future = trajectories.plan_trajectory(history())

    assert future.shape == (spec.WAYPOINT_COUNT, 2)
    assert np.all(np.isfinite(future))


def test_continues_from_the_last_observed_position():
    past = np.asarray(history(speed=10.0))
    future = trajectories.plan_trajectory(past)

    step = np.linalg.norm(future[0] - past[-1])
    assert step == pytest.approx(10.0 * STEP, rel=0.05)


def test_cruises_straight_when_the_history_is_straight():
    future = trajectories.plan_trajectory(history(speed=12.0))

    assert np.allclose(future[:, 1], 0.0, atol=1e-6)
    spacing = np.linalg.norm(np.diff(future, axis=0), axis=1)
    assert np.allclose(spacing, 12.0 * STEP, atol=0.05)


def test_turns_in_the_direction_of_the_observed_yaw():
    future = trajectories.plan_trajectory(history(yaw_rate=0.2))

    assert future[:, 1].min() > 0.0


def test_yaw_rate_decays_so_the_path_does_not_circle():
    future = trajectories.plan_trajectory(history(yaw_rate=0.4))
    early = np.diff(future[:5], axis=0)
    late = np.diff(future[-5:], axis=0)

    def turn(deltas):
        return abs(np.diff(np.arctan2(deltas[:, 1], deltas[:, 0])).mean())

    assert turn(late) < turn(early)


def test_handles_a_stopped_vehicle():
    past = [[0.0, 0.0] for _ in range(6)]
    future = trajectories.plan_trajectory(past)

    assert np.all(np.isfinite(future))
    assert np.allclose(future, 0.0)


def test_jerk_stays_low():
    future = trajectories.plan_trajectory(history(speed=14.0, yaw_rate=0.1))
    velocity = np.diff(future, axis=0) / STEP
    acceleration = np.diff(velocity, axis=0) / STEP
    jerk = np.linalg.norm(np.diff(acceleration, axis=0) / STEP, axis=1)

    assert jerk.max() < 1.0


def test_rejects_short_history():
    with pytest.raises(ValueError, match="at least four"):
        trajectories.plan_trajectory([[0.0, 0.0], [1.0, 0.0]])


def test_rejects_non_finite_history():
    past = history()
    past[2] = [float("inf"), 0.0]

    with pytest.raises(ValueError, match="non-finite"):
        trajectories.plan_trajectory(past)


def test_plan_file_round_trips_through_the_validator(tmp_path):
    source = tmp_path / "history.jsonl"
    with open(source, "w", encoding="utf-8") as handle:
        for scenario_id in spec.SCENARIO_IDS:
            handle.write(
                json.dumps({"scenario_id": scenario_id, "past_trajectory": history()})
                + "\n"
            )

    output = tmp_path / "trajectories.jsonl"
    written = trajectories.plan_file(source, output)

    assert written == spec.SCENARIO_COUNT
    report = trajectories.inspect_trajectories(output)
    assert report.ok, report.errors
    assert report.warnings == []


def test_plan_file_rejects_unknown_scenarios(tmp_path):
    source = tmp_path / "history.jsonl"
    source.write_text(
        json.dumps({"scenario_id": "0000999", "past_trajectory": history()}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown scenario_id"):
        trajectories.plan_file(source, tmp_path / "out.jsonl")
