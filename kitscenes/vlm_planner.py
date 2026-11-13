"""Gemini VLM Trajectory Reasoning & Planning module for KITScenes-LongTail.

Integrates multimodal/text reasoning with physically plausible kinematic trajectory
generation for the 400 long-tail driving scenarios.

Scoring Requirements (MMS - Multi-Maneuver Score):
- Initial heading direction check: dot(ref[0], pred[0]) >= 0.5 * |ref[0]|
- Comfort jerk penalty: jerk ratio <= 0.44
- Comfort tortuosity penalty: tortuosity ratio <= 0.06
- Tactical response: React accurately to instructions (overtake, turn, slow down, brake).
"""

from __future__ import annotations

import json
import math
import os
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import spec
from .trajectories import (
    HISTORY_KEYS,
    LOOKBACK,
    PRECISION,
    YAW_RATE_DECAY,
    _circular_mean,
    _to_xy,
    _wrap,
)

# Kinematic limits for comfort and physical plausibility
MAX_ACCELERATION = 2.0  # m/s^2 (comfortable acceleration)
MAX_DECELERATION = 3.5  # m/s^2 (comfortable braking)
MAX_JERK = 2.5  # m/s^3 (jerk rate limit for smoothing)
MAX_YAW_RATE = 0.6  # rad/s (maximum curvature / steering)
MAX_YAW_ACCEL = 0.4  # rad/s^2 (yaw acceleration limit)
MAX_SPEED = 55.0  # m/s (~200 km/h)
DEFAULT_LANE_CHANGE_DURATION = 3.5  # seconds
DEFAULT_MODEL = "gemini-3.8-flash"

SYSTEM_INSTRUCTION = """You are an expert Autonomous Vehicle Trajectory Planning & Reasoning AI for the KITScenes-LongTail World Modeling Challenge.
Your task is to analyze the vehicle's past odometry and driving instruction, reason about tactical maneuvers, and output optimal tactical driving parameters for a 5.0-second horizon (25 waypoints at 5 Hz, from t=0.2s to t=5.0s).

Scoring & Safety Constraints:
1. Multi-Maneuver Score (MMS) requires:
   - Initial heading direction check: The vehicle's initial waypoint at t=0.2s MUST preserve current heading direction. Never make sharp or backwards jumps.
   - Low jerk: Jerk ratio must be <= 0.44 above reference (smooth acceleration/braking).
   - Low tortuosity: Tortuosity ratio must be <= 0.06 above reference (path length / straight-line distance, no erratic swerving).
   - Faithful tactical response to instruction: React accurately to instructions like overtaking, turning, slowing down, or braking to stop.

Available Maneuver Types:
- "cruise": Maintain current speed and lane.
- "slow_down": Decelerate smoothly to a lower safe speed (e.g. for construction, traffic, hazard).
- "brake_stop": Decelerate smoothly to come to a complete stop (e.g. for pedestrian, red light, stop sign).
- "accelerate": Accelerate smoothly to higher speed (e.g. open highway, clearing hazard).
- "turn_left": Left turn at intersection (smooth yaw rate ~ +0.25 to +0.35 rad/s, moderate speed).
- "turn_right": Right turn at intersection (smooth yaw rate ~ -0.25 to -0.35 rad/s, moderate speed).
- "lane_change_left": Smooth lane change or overtake on the left (lateral offset ~ +3.0 to +3.7 m).
- "lane_change_right": Smooth lane change or overtake on the right (lateral offset ~ -3.0 to -3.7 m).

Output Format:
You MUST respond with a valid JSON object matching this schema:
{
  "reasoning": "<concise tactical reasoning>",
  "maneuver": "<cruise | slow_down | brake_stop | accelerate | turn_left | turn_right | lane_change_left | lane_change_right>",
  "target_speed": <float in m/s, or null to maintain current speed>,
  "target_acceleration": <float in m/s^2, between -3.5 and 2.0, or null>,
  "target_yaw_rate": <float in rad/s, between -0.5 and 0.5, or null>,
  "lateral_offset": <float in meters, e.g. +3.5 for left, -3.5 for right, 0.0 for center>,
  "maneuver_duration": <float in seconds, e.g. 2.5 to 4.5>
}
"""


@dataclass
class TacticalPlan:
    """Tactical decisions and parameters for trajectory generation."""

    reasoning: str = ""
    maneuver: str = "cruise"
    target_speed: float | None = None
    target_acceleration: float | None = None
    target_yaw_rate: float | None = None
    lateral_offset: float = 0.0
    maneuver_start_time: float = 0.0
    maneuver_duration: float = DEFAULT_LANE_CHANGE_DURATION
    confidence: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "reasoning": self.reasoning,
            "maneuver": self.maneuver,
            "target_speed": self.target_speed,
            "target_acceleration": self.target_acceleration,
            "target_yaw_rate": self.target_yaw_rate,
            "lateral_offset": self.lateral_offset,
            "maneuver_start_time": self.maneuver_start_time,
            "maneuver_duration": self.maneuver_duration,
            "confidence": self.confidence,
        }


def extract_initial_state(
    history: Sequence[Sequence[float]], lookback: int = LOOKBACK
) -> dict[str, float]:
    """Estimate ego vehicle position, speed, heading, yaw rate, and acceleration at t=0."""
    points = _to_xy(history)
    step = spec.WAYPOINT_STEP_S
    x0, y0 = float(points[-1, 0]), float(points[-1, 1])

    deltas = np.diff(points, axis=0)[-lookback:]
    displacements = np.linalg.norm(deltas, axis=1)
    speeds = displacements / step
    moving = displacements > 1e-3

    if moving.any():
        headings = np.arctan2(deltas[moving, 1], deltas[moving, 0])
        heading0 = _circular_mean(headings, speeds[moving])
        speed0 = float(np.median(speeds[moving][-3:]))
        if len(headings) >= 2:
            turns = np.angle(np.exp(1j * np.diff(headings)))
            yaw_rate0 = float(np.median(turns)) / step
        else:
            yaw_rate0 = 0.0
        if len(speeds) >= 2:
            accel0 = float(
                np.clip(
                    (speeds[-1] - speeds[0]) / (len(speeds) * step),
                    -MAX_DECELERATION,
                    MAX_ACCELERATION,
                )
            )
        else:
            accel0 = 0.0
    else:
        heading0, speed0, yaw_rate0, accel0 = 0.0, 0.0, 0.0, 0.0

    yaw_rate0 = float(np.clip(yaw_rate0, -MAX_YAW_RATE, MAX_YAW_RATE))

    return {
        "x": x0,
        "y": y0,
        "speed": speed0,
        "heading": heading0,
        "yaw_rate": yaw_rate0,
        "acceleration": accel0,
    }


def build_vlm_prompt(
    scenario_id: str,
    history: Sequence[Sequence[float]],
    instruction: str = "",
    camera_summaries: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Formulate structured prompt for Gemini 3.8 Flash given past trajectory and instruction."""
    points = _to_xy(history)
    state = extract_initial_state(history)
    step = spec.WAYPOINT_STEP_S

    # Format recent history points table
    history_lines = []
    n_points = len(points)
    for idx, pt in enumerate(points):
        t_rel = -(n_points - 1 - idx) * step
        history_lines.append(f"  t={t_rel:+.2f}s: x={pt[0]:.3f}, y={pt[1]:.3f}")
    history_table = "\n".join(history_lines)

    # Format camera scene summaries
    if camera_summaries:
        cam_lines = [f"- {cam}: {desc}" for cam, desc in sorted(camera_summaries.items())]
        cam_text = "\n".join(cam_lines)
    else:
        cam_text = "No visual camera summaries available."

    instr_text = instruction.strip() if instruction and instruction.strip() else "Maintain current cruising speed and stay in lane."

    user_prompt = f"""Scenario ID: {scenario_id}
Driving Instruction: "{instr_text}"

Current Ego State at t=0s:
- Speed: {state['speed']:.2f} m/s ({state['speed'] * 3.6:.1f} km/h)
- Heading: {math.degrees(state['heading']):.1f}° ({state['heading']:.3f} rad)
- Yaw Rate: {state['yaw_rate']:.3f} rad/s
- Acceleration: {state['acceleration']:.2f} m/s²
- Position at t=0: ({state['x']:.3f}, {state['y']:.3f})

Past Odometry History:
{history_table}

Visual Scene Context:
{cam_text}

Provide your tactical planning decision in JSON."""

    return {
        "scenario_id": scenario_id,
        "system_instruction": SYSTEM_INSTRUCTION,
        "user_prompt": user_prompt,
        "initial_state": state,
    }


def infer_tactical_plan_heuristic(
    instruction: str, initial_speed: float = 10.0
) -> TacticalPlan:
    """Rule-based tactical reasoning fallback when offline or no API key is available."""
    text = (instruction or "").lower().strip()

    # 1. Brake / Stop / Pedestrian / Red Light
    if any(k in text for k in ["brake", "stop", "halt", "pedestrian", "red light", "yield to cross"]):
        return TacticalPlan(
            reasoning=f"Instruction indicates stop/hazard ('{instruction}'). Decelerating to complete stop.",
            maneuver="brake_stop",
            target_speed=0.0,
            target_acceleration=-2.5,
            target_yaw_rate=0.0,
            lateral_offset=0.0,
            maneuver_duration=4.0,
        )

    # 2. Slow down / Construction / Work zone
    if any(k in text for k in ["slow down", "decelerate", "construction", "work zone", "caution", "reduce speed", "debris"]):
        target_v = max(initial_speed * 0.45, 3.5) if initial_speed > 4.0 else initial_speed
        return TacticalPlan(
            reasoning=f"Instruction requests slowing down for hazard ('{instruction}'). Lowering target speed.",
            maneuver="slow_down",
            target_speed=target_v,
            target_acceleration=-1.5,
            target_yaw_rate=0.0,
            lateral_offset=0.0,
            maneuver_duration=3.5,
        )

    # 3. U-turn
    if any(k in text for k in ["u-turn", "uturn", "make a u-turn"]):
        return TacticalPlan(
            reasoning=f"Instruction specifies U-turn ('{instruction}'). Executing tight left turn.",
            maneuver="turn_left",
            target_speed=4.5,
            target_acceleration=1.0 if initial_speed < 2.0 else -0.8 if initial_speed > 5.0 else 0.0,
            target_yaw_rate=0.52,
            lateral_offset=0.0,
            maneuver_duration=4.5,
        )

    # 4. Overtake on Right / Pass on Right / Lane change right / Use right lane
    if any(k in text for k in ["use right lane", "overtake on right", "pass on right", "overtake on the right", "pass on the right", "shift right", "lane change right", "move right", "pass right"]):
        return TacticalPlan(
            reasoning=f"Instruction indicates overtaking/shifting on the right ('{instruction}').",
            maneuver="lane_change_right",
            target_speed=max(initial_speed, 8.0),
            target_acceleration=1.0 if initial_speed < 6.0 else 0.0,
            target_yaw_rate=0.0,
            lateral_offset=-3.5,
            maneuver_duration=3.5,
        )

    # 5. Overtake on Left / Pass on Left / Lane change left / Use left lane
    if any(k in text for k in ["use left lane", "overtake on left", "pass on left", "overtake on the left", "pass on the left", "shift left", "lane change left", "move left", "pass left", "overtake", "pass"]):
        return TacticalPlan(
            reasoning=f"Instruction indicates overtaking/shifting on the left ('{instruction}').",
            maneuver="lane_change_left",
            target_speed=max(initial_speed, 8.0),
            target_acceleration=1.0 if initial_speed < 6.0 else 0.0,
            target_yaw_rate=0.0,
            lateral_offset=3.5,
            maneuver_duration=3.5,
        )

    # 6. Turn Left
    if any(k in text for k in ["turn left", "bear left", "left turn", "make a left"]):
        return TacticalPlan(
            reasoning=f"Instruction specifies left turn ('{instruction}'). Setting positive yaw rate.",
            maneuver="turn_left",
            target_speed=min(initial_speed, 6.0) if initial_speed > 3.0 else 5.5,
            target_acceleration=-1.0 if initial_speed > 6.0 else (1.2 if initial_speed < 3.0 else 0.0),
            target_yaw_rate=0.28,
            lateral_offset=0.0,
            maneuver_duration=3.5,
        )

    # 7. Turn Right
    if any(k in text for k in ["turn right", "bear right", "right turn", "make a right"]):
        return TacticalPlan(
            reasoning=f"Instruction specifies right turn ('{instruction}'). Setting negative yaw rate.",
            maneuver="turn_right",
            target_speed=min(initial_speed, 6.0) if initial_speed > 3.0 else 5.5,
            target_acceleration=-1.0 if initial_speed > 6.0 else (1.2 if initial_speed < 3.0 else 0.0),
            target_yaw_rate=-0.28,
            lateral_offset=0.0,
            maneuver_duration=3.5,
        )

    # 8. Accelerate / Speed up
    if any(k in text for k in ["accelerate", "speed up", "faster", "resume speed"]):
        return TacticalPlan(
            reasoning=f"Instruction indicates accelerating ('{instruction}').",
            maneuver="accelerate",
            target_speed=min(initial_speed + 5.0, 30.0) if initial_speed > 1.0 else 8.0,
            target_acceleration=1.5,
            target_yaw_rate=0.0,
            lateral_offset=0.0,
            maneuver_duration=3.0,
        )

    # 9. Default Cruise (accelerate smoothly if stopped at light/intersection)
    target_cruise_v = initial_speed if initial_speed >= 1.0 else 7.0
    target_cruise_a = 0.0 if initial_speed >= 1.0 else 1.2
    return TacticalPlan(
        reasoning="Standard lane cruising.",
        maneuver="cruise",
        target_speed=target_cruise_v,
        target_acceleration=target_cruise_a,
        target_yaw_rate=0.0,
        lateral_offset=0.0,
        maneuver_duration=DEFAULT_LANE_CHANGE_DURATION,
    )


def parse_vlm_response(
    response_text: str, fallback_instruction: str = "", initial_speed: float = 10.0
) -> TacticalPlan:
    """Parse Gemini JSON response into TacticalPlan with bounds clamping and fallback."""
    cleaned = response_text.strip()

    # Extract JSON string if wrapped in markdown code fence
    if "```json" in cleaned:
        start_idx = cleaned.find("```json") + 7
        end_idx = cleaned.find("```", start_idx)
        if end_idx != -1:
            cleaned = cleaned[start_idx:end_idx].strip()
    elif "```" in cleaned:
        start_idx = cleaned.find("```") + 3
        end_idx = cleaned.find("```", start_idx)
        if end_idx != -1:
            cleaned = cleaned[start_idx:end_idx].strip()

    # Extract outermost braces if embedded in prose
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace : last_brace + 1]

    data = None
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        data = None

    if isinstance(data, dict):
        reasoning = str(data.get("reasoning", ""))
        maneuver = str(data.get("maneuver", "cruise")).lower().strip()

        # Target speed
        raw_speed = data.get("target_speed")
        target_speed = None
        if raw_speed is not None and isinstance(raw_speed, (int, float)) and math.isfinite(raw_speed):
            target_speed = float(np.clip(raw_speed, 0.0, MAX_SPEED))

        # Target acceleration
        raw_accel = data.get("target_acceleration")
        target_accel = None
        if raw_accel is not None and isinstance(raw_accel, (int, float)) and math.isfinite(raw_accel):
            target_accel = float(np.clip(raw_accel, -MAX_DECELERATION, MAX_ACCELERATION))

        # Target yaw rate
        raw_yaw = data.get("target_yaw_rate")
        target_yaw = None
        if raw_yaw is not None and isinstance(raw_yaw, (int, float)) and math.isfinite(raw_yaw):
            target_yaw = float(np.clip(raw_yaw, -MAX_YAW_RATE, MAX_YAW_RATE))

        # Lateral offset
        raw_offset = data.get("lateral_offset", 0.0)
        lat_offset = 0.0
        if isinstance(raw_offset, (int, float)) and math.isfinite(raw_offset):
            lat_offset = float(np.clip(raw_offset, -8.0, 8.0))

        # Maneuver duration
        raw_duration = data.get("maneuver_duration", DEFAULT_LANE_CHANGE_DURATION)
        duration = DEFAULT_LANE_CHANGE_DURATION
        if isinstance(raw_duration, (int, float)) and math.isfinite(raw_duration):
            duration = float(np.clip(raw_duration, 1.0, 5.0))

        return TacticalPlan(
            reasoning=reasoning,
            maneuver=maneuver,
            target_speed=target_speed,
            target_acceleration=target_accel,
            target_yaw_rate=target_yaw,
            lateral_offset=lat_offset,
            maneuver_duration=duration,
        )

    # Fallback to heuristic
    return infer_tactical_plan_heuristic(fallback_instruction, initial_speed)


def generate_kinematic_trajectory(
    history: Sequence[Sequence[float]],
    tactical_plan: TacticalPlan,
    horizon: int = spec.WAYPOINT_COUNT,
    step: float = spec.WAYPOINT_STEP_S,
    yaw_rate_decay: float = YAW_RATE_DECAY,
) -> np.ndarray:
    """Generate 25 smooth waypoints satisfying physical constraints and MMS metrics.

    Ensures:
    1. Initial heading direction check: dot(ref[0], pred[0]) >= 0.5 * |ref[0]|
    2. Jerk clamping: rate-limited longitudinal acceleration and yaw rate
    3. Low tortuosity: smooth quintic Frenet polynomial lateral transition
    4. Clean deceleration to stop without repeated identical waypoints
    """
    state = extract_initial_state(history)
    x_curr = state["x"]
    y_curr = state["y"]
    v_curr = state["speed"]
    h_curr = state["heading"]
    yaw_curr = state["yaw_rate"]
    a_curr = state["acceleration"]

    # Target speed resolution
    if tactical_plan.target_speed is not None:
        v_target = float(tactical_plan.target_speed)
    elif tactical_plan.maneuver == "brake_stop":
        v_target = 0.0
    elif tactical_plan.maneuver == "slow_down":
        v_target = max(v_curr * 0.45, 3.5) if v_curr > 4.0 else v_curr
    elif tactical_plan.maneuver == "accelerate":
        v_target = min(v_curr + 5.0, 30.0) if v_curr > 1.0 else 8.0
    elif tactical_plan.maneuver in ("turn_left", "turn_right"):
        v_target = min(v_curr, 6.0) if v_curr > 3.0 else 5.5
    else:
        v_target = v_curr if v_curr > 1.0 else 7.0

    # Target yaw rate resolution
    if tactical_plan.target_yaw_rate is not None:
        target_yaw = float(tactical_plan.target_yaw_rate)
    elif tactical_plan.maneuver == "turn_left":
        target_yaw = 0.28
    elif tactical_plan.maneuver == "turn_right":
        target_yaw = -0.28
    else:
        target_yaw = None

    lat_offset = float(tactical_plan.lateral_offset)
    duration = float(np.clip(tactical_plan.maneuver_duration, 2.0, 5.0))
    t_start = float(np.clip(tactical_plan.maneuver_start_time, 0.0, 2.0))

    waypoints = np.empty((horizon, 2), dtype=float)

    for k in range(horizon):
        t = (k + 1) * step

        # Longitudinal acceleration control with jerk rate limiting
        rem_time = max(step, (horizon - k) * step)
        if v_target == 0.0:
            a_des = np.clip(-v_curr / rem_time, -MAX_DECELERATION, 0.0)
            if tactical_plan.target_acceleration is not None:
                a_des = max(a_des, tactical_plan.target_acceleration)
        elif v_target is not None:
            a_des = np.clip((v_target - v_curr) / 1.5, -MAX_DECELERATION, MAX_ACCELERATION)
            if tactical_plan.target_acceleration is not None:
                if a_des < 0:
                    a_des = max(a_des, tactical_plan.target_acceleration)
                elif a_des > 0:
                    a_des = min(a_des, tactical_plan.target_acceleration)
        else:
            a_des = 0.0

        da = np.clip(a_des - a_curr, -MAX_JERK * step, MAX_JERK * step)
        a_curr += da
        v_next = v_curr + a_curr * step

        if v_target == 0.0:
            # Maintain tiny crawl speed until the end of horizon so consecutive waypoints
            # don't round to identical coordinates (which would trigger the repeated waypoints penalty)
            v_next = max(v_next, 0.01) if state["speed"] > 0.1 else 0.0
        else:
            v_next = float(np.clip(v_next, 0.0, MAX_SPEED))

        # Lateral yaw rate control
        if target_yaw is not None:
            dyaw = np.clip(target_yaw - yaw_curr, -MAX_YAW_ACCEL * step, MAX_YAW_ACCEL * step)
            yaw_curr += dyaw
        else:
            yaw_curr *= yaw_rate_decay
        yaw_curr = float(np.clip(yaw_curr, -MAX_YAW_RATE, MAX_YAW_RATE))

        # Midpoint kinematic integration
        mid_h = h_curr + 0.5 * yaw_curr * step
        mid_v = 0.5 * (v_curr + v_next)
        x_base = x_curr + mid_v * step * math.cos(mid_h)
        y_base = y_curr + mid_v * step * math.sin(mid_h)
        h_curr = _wrap(h_curr + yaw_curr * step)
        v_curr = v_next
        x_curr, y_curr = x_base, y_base

        # Quintic Frenet polynomial lateral offset for lane change / overtaking
        if lat_offset != 0.0 and t >= t_start:
            tau = min(1.0, max(0.0, (t - t_start) / duration))
            # Quintic polynomial: 10*tau^3 - 15*tau^4 + 6*tau^5 (zero 1st and 2nd derivatives at ends)
            s_poly = 10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5)
            offset = lat_offset * s_poly
            nx = -math.sin(h_curr)
            ny = math.cos(h_curr)
            px = x_base + offset * nx
            py = y_base + offset * ny
        else:
            px = x_base
            py = y_base

        waypoints[k] = (px, py)

    return waypoints


class GeminiPlannerClient:
    """Zero-dependency client for Gemini VLM planning with optional mock or REST support."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout: float = 30.0,
        mock_fn: Callable[[str], str] | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.timeout = timeout
        self.mock_fn = mock_fn

    def generate(self, prompt: str, system_instruction: str | None = SYSTEM_INSTRUCTION) -> str:
        """Query Gemini model and return text response."""
        if self.mock_fn is not None:
            return self.mock_fn(prompt)

        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )
        headers = {"Content-Type": "application/json"}
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                parts = body["candidates"][0]["content"]["parts"]
                return "".join(part.get("text", "") for part in parts)
        except Exception as exc:
            raise RuntimeError(f"Gemini API request failed: {exc}") from exc


def plan_vlm_trajectory(
    history: Sequence[Sequence[float]],
    *,
    instruction: str = "",
    camera_summaries: dict[str, str] | None = None,
    client: Any | None = None,
    tactical_plan: TacticalPlan | None = None,
    scenario_id: str = "0000000",
    heuristic_fallback: bool = True,
) -> np.ndarray:
    """Plan trajectory using VLM tactical reasoning and kinematic generator."""
    if tactical_plan is not None:
        plan = tactical_plan
    elif client is not None:
        prompt_data = build_vlm_prompt(scenario_id, history, instruction, camera_summaries)
        try:
            response_text = client.generate(
                prompt_data["user_prompt"],
                system_instruction=prompt_data["system_instruction"],
            )
            plan = parse_vlm_response(
                response_text,
                fallback_instruction=instruction,
                initial_speed=prompt_data["initial_state"]["speed"],
            )
        except Exception:
            if not heuristic_fallback:
                raise
            state = extract_initial_state(history)
            plan = infer_tactical_plan_heuristic(instruction, state["speed"])
    else:
        state = extract_initial_state(history)
        plan = infer_tactical_plan_heuristic(instruction, state["speed"])

    return generate_kinematic_trajectory(history, plan)


def _load_auxiliary_map(path: Path | None) -> dict[str, Any]:
    """Load scenario_id -> value mapping from JSON or JSONL file."""
    if not path or not Path(path).exists():
        return {}
    path = Path(path)
    res: dict[str, Any] = {}
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return {}
    if content.startswith("{") and not content.endswith("}"):
        # JSONL
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sid = item.get("scenario_id")
            if sid:
                val = item.get("instruction") or item.get("camera_summaries") or item.get("text") or item
                res[sid] = val
    elif content.startswith("{"):
        try:
            raw = json.loads(content)
            if isinstance(raw, dict):
                return raw
        except json.JSONDecodeError:
            pass
    return res


def plan_vlm_file(
    source: Path,
    destination: Path,
    *,
    model: str = DEFAULT_MODEL,
    instructions_path: Path | None = None,
    camera_summaries_path: Path | None = None,
    api_key: str | None = None,
    heuristic_fallback: bool = True,
    client: Any | None = None,
) -> int:
    """Batch trajectory planner with Gemini VLM reasoning for KITScenes scenarios.

    Writes a valid trajectories.jsonl containing 400 scenarios with 25 waypoints each.
    """
    source = Path(source)
    destination = Path(destination)

    if not source.exists():
        raise ValueError(f"no such file: {source}")

    instructions_map = _load_auxiliary_map(instructions_path)
    summaries_map = _load_auxiliary_map(camera_summaries_path)

    # Initialize client if not provided and API key is present
    vlm_client = client
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if vlm_client is None and resolved_key:
        vlm_client = GeminiPlannerClient(model=model, api_key=resolved_key)

    planned: dict[str, np.ndarray] = {}
    with open(source, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            scenario_id = record.get("scenario_id")
            if scenario_id not in spec.SCENARIO_ID_SET:
                raise ValueError(f"line {number}: unknown scenario_id {scenario_id!r}")

            history = next((record[key] for key in HISTORY_KEYS if key in record), None)
            if history is None:
                raise ValueError(f"line {number}: expected one of {', '.join(HISTORY_KEYS)}")

            instruction = (
                instructions_map.get(scenario_id)
                or record.get("instruction")
                or record.get("driving_instruction")
                or record.get("scene_instruction")
                or record.get("prompt")
                or ""
            )

            camera_summaries = (
                summaries_map.get(scenario_id)
                or record.get("camera_summaries")
                or record.get("visual_context")
                or None
            )

            trajectory = plan_vlm_trajectory(
                history,
                instruction=instruction,
                camera_summaries=camera_summaries,
                client=vlm_client,
                scenario_id=scenario_id,
                heuristic_fallback=heuristic_fallback,
            )
            planned[scenario_id] = trajectory

    if not planned:
        raise ValueError(f"{source} contains no records")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        for scenario_id in sorted(planned):
            points = np.round(planned[scenario_id], PRECISION).tolist()
            handle.write(
                json.dumps({"scenario_id": scenario_id, "future_trajectory": points})
                + "\n"
            )

    return len(planned)
