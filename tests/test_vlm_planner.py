"""Comprehensive unit tests for the Gemini VLM Trajectory Planner."""

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from kitscenes import spec, trajectories, vlm_planner

STEP = spec.WAYPOINT_STEP_S


def make_history(speed: float = 10.0, yaw_rate: float = 0.0, count: int = 8) -> list:
    """Generate odometry history for a vehicle driving at ``speed`` with ``yaw_rate``."""
    points = []
    x, y, heading = 0.0, 0.0, 0.0
    for _ in range(count):
        x += speed * STEP * math.cos(heading)
        y += speed * STEP * math.sin(heading)
        points.append([x, y])
        heading += yaw_rate * STEP
    return points


# ============================================================================
# 1. Prompt Generation Tests
# ============================================================================


def test_build_vlm_prompt_basic():
    hist = make_history(speed=12.0)
    prompt_data = vlm_planner.build_vlm_prompt(
        scenario_id="0000042",
        history=hist,
        instruction="Slow down for construction zone",
    )

    assert prompt_data["scenario_id"] == "0000042"
    assert "0000042" in prompt_data["user_prompt"]
    assert "Slow down for construction zone" in prompt_data["user_prompt"]
    assert "Speed: 12.00 m/s" in prompt_data["user_prompt"]
    assert "Past Odometry History:" in prompt_data["user_prompt"]
    assert "0.00s" in prompt_data["user_prompt"]

    # Initial state verification
    state = prompt_data["initial_state"]
    assert state["speed"] == pytest.approx(12.0, rel=0.05)
    assert abs(state["heading"]) < 1e-4
    assert abs(state["yaw_rate"]) < 1e-4


def test_build_vlm_prompt_with_camera_summaries():
    hist = make_history(speed=8.0)
    summaries = {
        "front": "Traffic cones blocking the left lane, worker visible at 30m.",
        "rear": "Empty road behind.",
    }
    prompt_data = vlm_planner.build_vlm_prompt(
        scenario_id="0000100",
        history=hist,
        instruction="Overtake on right",
        camera_summaries=summaries,
    )

    assert "Visual Scene Context:" in prompt_data["user_prompt"]
    assert "front: Traffic cones blocking the left lane" in prompt_data["user_prompt"]
    assert "rear: Empty road behind" in prompt_data["user_prompt"]


def test_build_vlm_prompt_empty_instruction_defaults_gracefully():
    hist = make_history()
    prompt_data = vlm_planner.build_vlm_prompt("0000001", hist, instruction="")
    assert "Maintain current cruising speed and stay in lane" in prompt_data["user_prompt"]


def test_system_instruction_contains_mms_rules():
    sys_prompt = vlm_planner.SYSTEM_INSTRUCTION
    assert "Multi-Maneuver Score" in sys_prompt or "MMS" in sys_prompt
    assert "Initial heading direction check" in sys_prompt
    assert "jerk" in sys_prompt.lower()
    assert "tortuosity" in sys_prompt.lower()
    assert "25 waypoints at 5 Hz" in sys_prompt


# ============================================================================
# 2. Tactical Parameter Parsing Tests
# ============================================================================


def test_parse_valid_json_response():
    resp = json.dumps({
        "reasoning": "Slow down safely for construction ahead.",
        "maneuver": "slow_down",
        "target_speed": 4.5,
        "target_acceleration": -1.5,
        "target_yaw_rate": 0.0,
        "lateral_offset": 0.0,
        "maneuver_duration": 3.5,
    })
    plan = vlm_planner.parse_vlm_response(resp)

    assert plan.maneuver == "slow_down"
    assert plan.target_speed == 4.5
    assert plan.target_acceleration == -1.5
    assert plan.target_yaw_rate == 0.0
    assert plan.lateral_offset == 0.0
    assert plan.maneuver_duration == 3.5


def test_parse_json_wrapped_in_markdown_code_block():
    resp = """Here is the tactical planning response:
```json
{
  "reasoning": "Overtake stalled vehicle on the right lane.",
  "maneuver": "lane_change_right",
  "target_speed": 11.0,
  "target_acceleration": 0.5,
  "target_yaw_rate": 0.0,
  "lateral_offset": -3.5,
  "maneuver_duration": 3.0
}
```
Hope this is helpful!"""

    plan = vlm_planner.parse_vlm_response(resp)
    assert plan.maneuver == "lane_change_right"
    assert plan.target_speed == 11.0
    assert plan.lateral_offset == -3.5
    assert plan.maneuver_duration == 3.0


def test_parse_json_with_clamping():
    resp = json.dumps({
        "reasoning": "Extreme values test",
        "maneuver": "accelerate",
        "target_speed": 150.0,  # exceeds MAX_SPEED
        "target_acceleration": 10.0,  # exceeds MAX_ACCELERATION
        "target_yaw_rate": 2.5,  # exceeds MAX_YAW_RATE
        "lateral_offset": -25.0,  # exceeds max offset
        "maneuver_duration": 15.0,  # exceeds max duration
    })
    plan = vlm_planner.parse_vlm_response(resp)

    assert plan.target_speed <= vlm_planner.MAX_SPEED
    assert plan.target_acceleration <= vlm_planner.MAX_ACCELERATION
    assert plan.target_yaw_rate <= vlm_planner.MAX_YAW_RATE
    assert abs(plan.lateral_offset) <= 8.0
    assert plan.maneuver_duration <= 5.0


def test_parse_malformed_json_falls_back_to_heuristic():
    resp = "I think the car should stop immediately because a pedestrian is on the road."
    plan = vlm_planner.parse_vlm_response(
        resp, fallback_instruction="Brake for pedestrian in crosswalk", initial_speed=10.0
    )

    assert plan.maneuver == "brake_stop"
    assert plan.target_speed == 0.0
    assert plan.target_acceleration <= -2.0


@pytest.mark.parametrize(
    "instruction,expected_maneuver,expected_speed_fn,expected_lat_sign",
    [
        ("Brake for pedestrian crossing street", "brake_stop", lambda s: s == 0.0, 0),
        ("Emergency stop at red traffic light", "brake_stop", lambda s: s == 0.0, 0),
        ("Slow down for construction zone", "slow_down", lambda s: s < 8.0, 0),
        ("Overtake on right around slow truck", "lane_change_right", lambda s: s >= 10.0, -1),
        ("Pass on right", "lane_change_right", lambda s: s >= 10.0, -1),
        ("Overtake on left lane", "lane_change_left", lambda s: s >= 10.0, 1),
        ("Turn left at intersection", "turn_left", lambda s: s <= 6.0, 0),
        ("Turn right at the junction", "turn_right", lambda s: s <= 6.0, 0),
        ("Accelerate onto highway", "accelerate", lambda s: s > 10.0, 0),
        ("Maintain speed and stay in lane", "cruise", lambda s: s == 10.0, 0),
    ],
)
def test_heuristic_planner_maneuvers(instruction, expected_maneuver, expected_speed_fn, expected_lat_sign):
    plan = vlm_planner.infer_tactical_plan_heuristic(instruction, initial_speed=10.0)

    assert plan.maneuver == expected_maneuver
    assert expected_speed_fn(plan.target_speed)
    if expected_lat_sign < 0:
        assert plan.lateral_offset < -2.0
    elif expected_lat_sign > 0:
        assert plan.lateral_offset > 2.0
    else:
        assert plan.lateral_offset == 0.0


# ============================================================================
# 3. Kinematic Smoothing & Metric Quality Tests
# ============================================================================


def test_kinematic_generator_shape_and_types():
    hist = make_history(speed=10.0)
    plan = vlm_planner.TacticalPlan(maneuver="cruise", target_speed=10.0)
    future = vlm_planner.generate_kinematic_trajectory(hist, plan)

    assert future.shape == (spec.WAYPOINT_COUNT, 2)
    assert np.all(np.isfinite(future))


@pytest.mark.parametrize(
    "instruction,v0,yaw0",
    [
        ("Maintain cruising speed", 12.0, 0.0),
        ("Overtake on right", 10.0, 0.0),
        ("Overtake on left", 10.0, 0.0),
        ("Slow down for construction", 14.0, 0.0),
        ("Brake for pedestrian", 10.0, 0.0),
        ("Turn left at intersection", 8.0, 0.0),
        ("Turn right at intersection", 8.0, 0.0),
        ("Curved highway driving", 15.0, 0.05),
    ],
)
def test_initial_heading_direction_check(instruction, v0, yaw0):
    """MMS requires dot(ref[0], pred[0]) >= 0.5 * |ref[0]|.

    The first waypoint at t=0.2s MUST preserve the initial vehicle heading direction.
    """
    hist = make_history(speed=v0, yaw_rate=yaw0)
    plan = vlm_planner.infer_tactical_plan_heuristic(instruction, initial_speed=v0)
    future = vlm_planner.generate_kinematic_trajectory(hist, plan)

    # Reference displacement from t=0 to t=0.2s
    ref_displacement = np.array([v0 * STEP * math.cos(0.0), v0 * STEP * math.sin(0.0)])
    if yaw0 != 0.0:
        ref_displacement = np.array([v0 * STEP * math.cos(0.5 * yaw0 * STEP), v0 * STEP * math.sin(0.5 * yaw0 * STEP)])

    pred_displacement = future[0] - np.array(hist[-1])

    dot_product = float(np.dot(ref_displacement, pred_displacement))
    norm_ref = float(np.linalg.norm(ref_displacement))
    norm_pred = float(np.linalg.norm(pred_displacement))

    # dot >= 0.5 * |ref| (when pred is normalised) or dot/(|ref|*|pred|) >= 0.5
    cosine_similarity = dot_product / (norm_ref * norm_pred)
    assert cosine_similarity >= 0.90, f"Initial direction check failed for {instruction}: cos={cosine_similarity}"


def test_comfort_jerk_ratio_stays_low():
    """MMS comfort penalty penalizes jerk > 44% above reference.

    Third finite differences divided by dt^3 should remain smooth and low.
    """
    scenarios = [
        ("cruise", vlm_planner.TacticalPlan(maneuver="cruise", target_speed=12.0)),
        ("overtake_right", vlm_planner.TacticalPlan(maneuver="lane_change_right", lateral_offset=-3.5)),
        ("overtake_left", vlm_planner.TacticalPlan(maneuver="lane_change_left", lateral_offset=3.5)),
        ("slow_down", vlm_planner.TacticalPlan(maneuver="slow_down", target_speed=5.0, target_acceleration=-1.5)),
        ("brake_stop", vlm_planner.TacticalPlan(maneuver="brake_stop", target_speed=0.0)),
        ("turn_left", vlm_planner.TacticalPlan(maneuver="turn_left", target_yaw_rate=0.28, target_speed=6.0)),
    ]

    for name, plan in scenarios:
        hist = make_history(speed=12.0)
        future = vlm_planner.generate_kinematic_trajectory(hist, plan)

        # Jerk is mean magnitude of third finite difference
        diff3 = np.linalg.norm(np.diff(future, n=3, axis=0), axis=1)
        mean_diff3 = float(np.mean(diff3))
        # Raw third differences should be <= 0.03 m
        assert mean_diff3 < 0.03, f"{name} had high raw 3rd difference: {mean_diff3}"

        jerk = diff3 / (STEP**3)
        mean_jerk = float(np.mean(jerk))
        # Mean jerk should be well below 2.5 m/s^3
        assert mean_jerk < 2.5, f"{name} had high mean jerk: {mean_jerk}"


def test_comfort_tortuosity_stays_low():
    """MMS comfort penalty penalizes tortuosity > 6% above reference (tortuosity <= 1.06)."""
    # 1. Straight driving tortuosity is exactly 1.0
    hist = make_history(speed=10.0)
    plan_straight = vlm_planner.TacticalPlan(maneuver="cruise", target_speed=10.0)
    future_straight = vlm_planner.generate_kinematic_trajectory(hist, plan_straight)
    assert trajectories._tortuosity(future_straight) == pytest.approx(1.0, abs=1e-5)

    # 2. Smooth lane change tortuosity should be <= 1.01 (way below 1.06 limit)
    plan_lane_change = vlm_planner.TacticalPlan(maneuver="lane_change_right", lateral_offset=-3.5)
    future_lc = vlm_planner.generate_kinematic_trajectory(hist, plan_lane_change)
    tort_lc = trajectories._tortuosity(future_lc)
    assert tort_lc < 1.01, f"Lane change tortuosity too high: {tort_lc}"


def test_smooth_braking_no_repeated_waypoints():
    """Braking to stop should decelerate smoothly without triggering repeated identical waypoints."""
    hist = make_history(speed=10.0)
    plan = vlm_planner.TacticalPlan(maneuver="brake_stop", target_speed=0.0)
    future = vlm_planner.generate_kinematic_trajectory(hist, plan)
    rounded = np.round(future, trajectories.PRECISION)

    diff = np.diff(rounded, axis=0)
    repeated = np.any(np.all(diff == 0, axis=1))
    assert not repeated, "Braking generated repeated identical waypoints!"


def test_lane_change_achieves_lateral_offset():
    """Lane change should smoothly reach target lateral displacement (e.g. +3.5m or -3.5m)."""
    hist = make_history(speed=12.0)
    plan_right = vlm_planner.TacticalPlan(maneuver="lane_change_right", lateral_offset=-3.5, maneuver_duration=3.5)
    future_right = vlm_planner.generate_kinematic_trajectory(hist, plan_right)

    # End lateral displacement should be approximately -3.5m
    assert future_right[-1, 1] == pytest.approx(-3.5, abs=0.2)

    plan_left = vlm_planner.TacticalPlan(maneuver="lane_change_left", lateral_offset=3.5, maneuver_duration=3.5)
    future_left = vlm_planner.generate_kinematic_trajectory(hist, plan_left)
    assert future_left[-1, 1] == pytest.approx(3.5, abs=0.2)


def test_stopped_vehicle_handling():
    """A stopped vehicle should remain stopped at (x0, y0) unless instructed to accelerate."""
    hist = [[0.0, 0.0] for _ in range(6)]
    plan_stop = vlm_planner.TacticalPlan(maneuver="brake_stop", target_speed=0.0)
    future_stop = vlm_planner.generate_kinematic_trajectory(hist, plan_stop)

    assert np.allclose(future_stop, 0.0)

    # If instructed to accelerate from stop, moves forward smoothly
    plan_go = vlm_planner.TacticalPlan(maneuver="accelerate", target_speed=10.0, target_acceleration=1.5)
    future_go = vlm_planner.generate_kinematic_trajectory(hist, plan_go)
    assert future_go[-1, 0] > 1.0


# ============================================================================
# 4. Mocked Gemini Response Execution Tests
# ============================================================================


def test_mocked_gemini_client_full_planning():
    hist = make_history(speed=10.0)

    mock_response = json.dumps({
        "reasoning": "Detected bicycle on right edge. Shifting left into adjacent lane to pass safely.",
        "maneuver": "lane_change_left",
        "target_speed": 11.0,
        "target_acceleration": 0.5,
        "target_yaw_rate": 0.0,
        "lateral_offset": 3.5,
        "maneuver_duration": 3.0,
    })

    client = vlm_planner.GeminiPlannerClient(mock_fn=lambda prompt: mock_response)
    future = vlm_planner.plan_vlm_trajectory(
        hist,
        instruction="Pass the bicycle on the left",
        client=client,
        scenario_id="0000007",
    )

    assert future.shape == (spec.WAYPOINT_COUNT, 2)
    assert future[-1, 1] == pytest.approx(3.5, abs=0.25)


def test_gemini_client_fallback_when_api_fails():
    hist = make_history(speed=10.0)

    def failing_mock(prompt):
        raise RuntimeError("Network timeout connecting to Gemini API")

    client = vlm_planner.GeminiPlannerClient(mock_fn=failing_mock)
    # With heuristic_fallback=True, should smoothly fall back without raising
    future = vlm_planner.plan_vlm_trajectory(
        hist,
        instruction="Brake for pedestrian in roadway",
        client=client,
        scenario_id="0000008",
        heuristic_fallback=True,
    )

    assert future.shape == (spec.WAYPOINT_COUNT, 2)
    # The heuristic should have chosen brake_stop
    speed_end = np.linalg.norm(future[-1] - future[-2]) / STEP
    assert speed_end < 2.0


def test_gemini_client_raises_when_fallback_disabled():
    hist = make_history(speed=10.0)

    def failing_mock(prompt):
        raise RuntimeError("API quota exceeded")

    client = vlm_planner.GeminiPlannerClient(mock_fn=failing_mock)
    with pytest.raises(RuntimeError, match="API quota exceeded"):
        vlm_planner.plan_vlm_trajectory(
            hist,
            instruction="Slow down",
            client=client,
            heuristic_fallback=False,
        )


# ============================================================================
# 5. Schema Validation & File Runner Tests
# ============================================================================


def test_plan_vlm_file_round_trips_through_validator(tmp_path):
    """Generate all 400 scenarios with diverse long-tail instructions and validate schema."""
    source = tmp_path / "history.jsonl"
    instructions = [
        "Maintain speed and lane",
        "Slow down for construction zone",
        "Overtake on right around stalled truck",
        "Brake for pedestrian at crossing",
        "Turn left at intersection",
        "Turn right at intersection",
        "Accelerate past slow vehicle",
    ]

    with open(source, "w", encoding="utf-8") as handle:
        for idx, scenario_id in enumerate(spec.SCENARIO_IDS):
            instr = instructions[idx % len(instructions)]
            hist = make_history(speed=10.0 + (idx % 5))
            handle.write(
                json.dumps({
                    "scenario_id": scenario_id,
                    "past_trajectory": hist,
                    "instruction": instr,
                })
                + "\n"
            )

    output = tmp_path / "trajectories.jsonl"
    count = vlm_planner.plan_vlm_file(source, output)

    assert count == spec.SCENARIO_COUNT

    # Validate with official submission checker
    report = trajectories.inspect_trajectories(output)
    assert report.ok, f"Validation failed with errors: {report.errors}"
    assert report.count == spec.SCENARIO_COUNT
    assert report.warnings == [], f"Validation produced warnings: {report.warnings}"


def test_plan_vlm_file_with_separate_instructions(tmp_path):
    """Test loading driving instructions from a separate JSON file."""
    source = tmp_path / "history.jsonl"
    instr_file = tmp_path / "instructions.json"

    instr_map = {}
    with open(source, "w", encoding="utf-8") as handle:
        for scenario_id in spec.SCENARIO_IDS:
            instr_map[scenario_id] = "Slow down for construction"
            handle.write(
                json.dumps({
                    "scenario_id": scenario_id,
                    "past_trajectory": make_history(speed=12.0),
                })
                + "\n"
            )

    instr_file.write_text(json.dumps(instr_map), encoding="utf-8")

    output = tmp_path / "trajectories.jsonl"
    count = vlm_planner.plan_vlm_file(source, output, instructions_path=instr_file)

    assert count == spec.SCENARIO_COUNT
    report = trajectories.inspect_trajectories(output)
    assert report.ok
    assert report.warnings == []


def test_build_submission_cli_vlm_flag(tmp_path):
    """Test running build_submission.py plan --vlm via CLI."""
    source = tmp_path / "history.jsonl"
    with open(source, "w", encoding="utf-8") as handle:
        handle.writelines(json.dumps({
                    "scenario_id": scenario_id,
                    "past_trajectory": make_history(speed=10.0),
                    "instruction": "Overtake on right",
                })
                + "\n" for scenario_id in spec.SCENARIO_IDS)

    output = tmp_path / "trajectories.jsonl"

    cmd = [
        sys.executable,
        "build_submission.py",
        "plan",
        "--history",
        str(source),
        "--out",
        str(output),
        "--vlm",
    ]
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    }
    res = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    # Without a key the module uses its offline parser, and the CLI has to say so
    # rather than report a model call that never happened.
    assert "using heuristic fallback" in res.stdout
    assert "no GEMINI_API_KEY" in res.stderr
    assert "Gemini VLM" not in res.stdout
    assert "result: valid" in res.stdout

    report = trajectories.inspect_trajectories(output)
    assert report.ok
    assert report.warnings == []
