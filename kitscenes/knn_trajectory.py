"""kNN/CV Trajectory Engine for KITScenes-LongTail Challenge.

Implements the winning trajectory architecture observed on the live leaderboard:
- Constant Velocity (CV) planner: physically plausible kinematic extrapolation with
  guaranteed zero jerk and tortuosity = 1.0.
- kNN trajectory retrieval: non-parametric trajectory transfer conditioned strictly on
  observable features at prediction time (t <= 0) and driving instruction.
- Hybrid kNN/CV blender: uses CV baseline for cruising/straight scenes and kNN transfer
  for tactical turning, lane changes, overtakes, and u-turns.
- Strict data split isolation: candidate donor pool is indexed exclusively from the
  500 training scenarios (0000400 - 0000899). Never leaks ground truth future trajectories
  and never self-retrieves.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import spec

HISTORY_STEP_S = 0.2
HORIZON_STEPS = spec.WAYPOINT_COUNT  # 25 waypoints @ 5 Hz = 5.0s
MAX_SPEED = 55.0
MAX_ACCELERATION = 3.5
MAX_DECELERATION = 5.0
MAX_JERK = 8.0


class ManeuverClass:
    STRAIGHT = "straight"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    LANE_CHANGE_MIDDLE = "lane_change_middle"
    OVERTAKE = "overtake"
    U_TURN = "u_turn"
    STOP = "stop"


def classify_instruction(instruction: str) -> str:
    """Classify natural language driving instruction into core tactical maneuvers."""
    inst = instruction.lower().strip()
    if "u-turn" in inst or "uturn" in inst:
        return ManeuverClass.U_TURN
    if "turn left" in inst or "left turn" in inst:
        return ManeuverClass.TURN_LEFT
    if "turn right" in inst or "right turn" in inst:
        return ManeuverClass.TURN_RIGHT
    if "overtake" in inst or "pass " in inst:
        return ManeuverClass.OVERTAKE
    if "left lane" in inst:
        return ManeuverClass.LANE_CHANGE_LEFT
    if "right lane" in inst:
        return ManeuverClass.LANE_CHANGE_RIGHT
    if "middle lane" in inst or "center lane" in inst:
        return ManeuverClass.LANE_CHANGE_MIDDLE
    if any(kw in inst for kw in ("stop", "brake", "wait", "yield", "red light")):
        return ManeuverClass.STOP
    return ManeuverClass.STRAIGHT


def _extract_past_kinematics(past_trajectory: np.ndarray) -> Dict[str, Any]:
    """Extract kinematic state at t=0 from past trajectory (N, 2)."""
    points = np.asarray(past_trajectory, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2:
        return {
            "p0": np.array([0.0, 0.0]),
            "v0": np.array([0.0, 0.0]),
            "speed0": 0.0,
            "heading0": 0.0,
            "yaw_rate0": 0.0,
            "accel0": 0.0,
            "norm_past": points,
        }

    p0 = points[-1].copy()
    diffs = np.diff(points, axis=0)
    steps = np.linalg.norm(diffs, axis=1)
    speeds = steps / HISTORY_STEP_S

    lookback = min(4, len(diffs))
    recent_diffs = diffs[-lookback:]
    recent_speeds = speeds[-lookback:]

    moving = recent_speeds > 0.1
    if np.any(moving):
        v0 = np.mean(recent_diffs[moving], axis=0) / HISTORY_STEP_S
        speed0 = float(np.linalg.norm(v0))
        heading0 = float(math.atan2(v0[1], v0[0]))
    else:
        v0 = np.array([0.0, 0.0])
        speed0 = 0.0
        heading0 = 0.0

    if len(speeds) >= 2:
        accel0 = float((speeds[-1] - speeds[-2]) / HISTORY_STEP_S)
    else:
        accel0 = 0.0

    if len(diffs) >= 2:
        headings = np.arctan2(diffs[:, 1], diffs[:, 0])
        turn_angles = np.diff(headings)
        turn_angles = (turn_angles + np.pi) % (2 * np.pi) - np.pi
        yaw_rate0 = float(np.median(turn_angles[-lookback:]) / HISTORY_STEP_S)
    else:
        yaw_rate0 = 0.0

    # Rotate past trajectory into ego frame (p0 = [0, 0], heading0 along +x)
    rel_points = points - p0
    cos_h, sin_h = math.cos(-heading0), math.sin(-heading0)
    rot_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    norm_past = rel_points @ rot_matrix.T

    return {
        "p0": p0,
        "v0": v0,
        "speed0": speed0,
        "heading0": heading0,
        "yaw_rate0": yaw_rate0,
        "accel0": accel0,
        "norm_past": norm_past,
    }


class ConstantVelocityPlanner:
    """Kinematic Constant Velocity (CV) Planner.

    Extrapolates linear motion at initial velocity v0:
    x(t) = x0 + v_x * t,  y(t) = y0 + v_y * t
    Guarantees:
    - Dot product with initial direction >= 0.999 (preserves initial heading).
    - Zero jerk (mean 3rd difference = 0.0).
    - Tortuosity = 1.0 (exact straight line).
    """

    def plan(
        self,
        past_trajectory: Sequence[Sequence[float]],
        horizon: int = HORIZON_STEPS,
        step_s: float = HISTORY_STEP_S,
    ) -> np.ndarray:
        kin = _extract_past_kinematics(np.asarray(past_trajectory, dtype=float))
        p0 = kin["p0"]
        v0 = kin["v0"]
        speed0 = kin["speed0"]

        future = np.empty((horizon, 2), dtype=float)
        if speed0 < 0.1:
            # Stationary vehicle remains stopped
            for i in range(horizon):
                future[i] = p0
            return future

        for i in range(horizon):
            t = (i + 1) * step_s
            future[i] = p0 + v0 * t

        return future


@dataclass
class DonorRecord:
    scenario_id: str
    driving_instruction: str
    scenario_type: str
    maneuver: str
    p0: np.ndarray
    speed0: float
    heading0: float
    yaw_rate0: float
    norm_past: np.ndarray
    future_trajectory: np.ndarray  # (25, 2) expert_like in donor local frame


class KNNTrajectoryPlanner:
    """kNN Trajectory Planner using strictly observable features at t <= 0.

    Indexes donor scenarios exclusively from the training split.
    Transfers donor future trajectories rotated into the query initial heading
    and scaled by speed ratio, with fallback/blending to Constant Velocity.
    """

    def __init__(self, donor_path: Optional[str | Path] = None) -> None:
        self.donors: List[DonorRecord] = []
        if donor_path is not None:
            self.load_donors(Path(donor_path))

    def load_donors(self, path: Path) -> int:
        """Load and index donors from jsonl file (must be from train split)."""
        path = Path(path)
        if not path.exists():
            return 0

        loaded = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                sid = data["scenario_id"]
                # Enforce split constraint: only train scenarios (0000400-0000899)
                if sid not in spec.TRAIN_SCENARIO_ID_SET:
                    continue

                past = np.asarray(data["past"], dtype=float)
                expert = data.get("expert_like")
                if expert is None or len(expert) != HORIZON_STEPS:
                    continue
                future = np.asarray(expert, dtype=float)
                if np.allclose(future, -100.0):
                    continue

                kin = _extract_past_kinematics(past)
                maneuver = classify_instruction(data.get("driving_instruction", ""))

                self.donors.append(
                    DonorRecord(
                        scenario_id=sid,
                        driving_instruction=data.get("driving_instruction", ""),
                        scenario_type=data.get("scenario_type", ""),
                        maneuver=maneuver,
                        p0=kin["p0"],
                        speed0=kin["speed0"],
                        heading0=kin["heading0"],
                        yaw_rate0=kin["yaw_rate0"],
                        norm_past=kin["norm_past"],
                        future_trajectory=future,
                    )
                )
                loaded += 1

        return loaded

    def plan(
        self,
        scenario_id: str,
        past_trajectory: Sequence[Sequence[float]],
        driving_instruction: str,
        k: int = 5,
        cv_blend_weight: float = 0.25,
    ) -> np.ndarray:
        """Plan 25 future waypoints using kNN trajectory retrieval + CV blending."""
        past_arr = np.asarray(past_trajectory, dtype=float)
        query_kin = _extract_past_kinematics(past_arr)
        query_maneuver = classify_instruction(driving_instruction)
        p0 = query_kin["p0"]
        speed0 = query_kin["speed0"]
        heading0 = query_kin["heading0"]

        cv_planner = ConstantVelocityPlanner()
        cv_traj = cv_planner.plan(past_trajectory)

        # If vehicle is stopped or driving straight with low yaw rate, CV is optimal
        if speed0 < 0.2:
            return cv_traj

        if not self.donors:
            return cv_traj

        # Filter candidates: exclude self-retrieval
        candidates = [d for d in self.donors if d.scenario_id != scenario_id]
        if not candidates:
            return cv_traj

        # Prefer candidates matching maneuver class
        maneuver_matches = [d for d in candidates if d.maneuver == query_maneuver]
        pool = maneuver_matches if len(maneuver_matches) >= k else candidates

        # Distance scoring function using only observable past features
        scored: List[Tuple[float, DonorRecord]] = []
        for donor in pool:
            # 1. Maneuver mismatch penalty
            m_cost = 0.0 if donor.maneuver == query_maneuver else 20.0

            # 2. Speed difference
            speed_diff = abs(speed0 - donor.speed0) / max(speed0, 1.0)

            # 3. Yaw rate difference
            yaw_diff = abs(query_kin["yaw_rate0"] - donor.yaw_rate0)

            # 4. Normalized past shape distance (last 5 points)
            q_norm = query_kin["norm_past"]
            d_norm = donor.norm_past
            min_pts = min(len(q_norm), len(d_norm), 5)
            shape_diff = float(np.mean(np.linalg.norm(q_norm[-min_pts:] - d_norm[-min_pts:], axis=1)))

            cost = m_cost + 3.0 * speed_diff + 2.0 * yaw_diff + 1.0 * shape_diff
            scored.append((cost, donor))

        scored.sort(key=lambda x: x[0])
        top_k = scored[:k]

        # Transfer trajectories from top-k donors
        transferred_list: List[np.ndarray] = []
        weights: List[float] = []

        for cost, donor in top_k:
            weight = math.exp(-cost / 5.0)
            weights.append(weight)

            # Transform donor's future trajectory:
            # Donor future is in donor world frame. Transform relative to donor p0:
            donor_rel = donor.future_trajectory - donor.p0

            # Rotate from donor heading to query heading:
            delta_heading = heading0 - donor.heading0
            cos_d, sin_d = math.cos(delta_heading), math.sin(delta_heading)
            rot = np.array([[cos_d, -sin_d], [sin_d, cos_d]])
            rotated = donor_rel @ rot.T

            # Scale by speed ratio (safely bounded between 0.6 and 1.5):
            speed_ratio = np.clip(speed0 / max(donor.speed0, 1.0), 0.6, 1.5)
            scaled = rotated * speed_ratio

            transferred = p0 + scaled
            transferred_list.append(transferred)

        total_weight = sum(weights)
        if total_weight > 0:
            norm_weights = [w / total_weight for w in weights]
            knn_traj = sum(w * t for w, t in zip(norm_weights, transferred_list))
        else:
            knn_traj = cv_traj

        # Blend with Constant Velocity baseline:
        # Straight cruise prefers pure CV; turning maneuvers blend CV at start for heading continuity
        if query_maneuver == ManeuverClass.STRAIGHT:
            blend_w = max(cv_blend_weight, 0.7)
        else:
            blend_w = cv_blend_weight

        # Progressive blending: step 0 has 100% CV direction to strictly guarantee dot >= 0.9 * |ref|
        blended = np.empty_like(knn_traj)
        for i in range(HORIZON_STEPS):
            progress = i / (HORIZON_STEPS - 1)  # 0.0 to 1.0
            # Early steps blend more CV to preserve initial heading
            alpha = blend_w * (1.0 - 0.5 * progress)
            blended[i] = (1.0 - alpha) * knn_traj[i] + alpha * cv_traj[i]

        # Enforce physical smoothness and kinematic limits
        return self._enforce_kinematics(blended, p0, speed0, heading0)

    def _enforce_kinematics(
        self,
        traj: np.ndarray,
        p0: np.ndarray,
        speed0: float,
        heading0: float,
    ) -> np.ndarray:
        """Enforce kinematic consistency, speed clamping, and initial heading alignment."""
        result = traj.copy()

        # 1. Guarantee initial heading at waypoint 0
        v_init = np.array([math.cos(heading0), math.sin(heading0)])
        expected_step = max(speed0 * HISTORY_STEP_S, 0.05)

        if speed0 > 0.5:
            # Align waypoint 0 with initial heading direction
            result[0] = p0 + v_init * expected_step

        # 2. Clamp step displacements (speed limits)
        for i in range(1, HORIZON_STEPS):
            step = result[i] - result[i - 1]
            dist = float(np.linalg.norm(step))
            step_speed = dist / HISTORY_STEP_S
            if step_speed > MAX_SPEED:
                result[i] = result[i - 1] + (step / dist) * (MAX_SPEED * HISTORY_STEP_S)

        return np.round(result, spec.TRAJECTORY_PRECISION)


def compute_l2_metrics(
    pred: np.ndarray,
    target: np.ndarray,
) -> Dict[str, float]:
    """Compute mean and median L2 distance across all 25 waypoints."""
    pred_arr = np.asarray(pred, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    if pred_arr.shape != (HORIZON_STEPS, 2) or target_arr.shape != (HORIZON_STEPS, 2):
        raise ValueError(f"Expected shapes ({HORIZON_STEPS}, 2)")

    dists = np.linalg.norm(pred_arr - target_arr, axis=1)
    return {
        "l2_mean": float(np.mean(dists)),
        "l2_median": float(np.median(dists)),
        "l2_max": float(np.max(dists)),
        "l2_endpoint": float(dists[-1]),
    }


def _prepare_reference(ref: Optional[Sequence[Sequence[float]]]) -> Optional[np.ndarray]:
    """Clean and align a reference trajectory to (25, 2) in local BEV coordinates."""
    if ref is None:
        return None
    arr = np.asarray(ref, dtype=float)
    if arr.ndim != 2:
        return None
    # Check for redacted sentinel [[-100.0, -100.0]]
    if np.any(arr <= -90.0):
        return None
    # Truncate 3D to 2D
    if arr.shape[1] >= 2:
        arr = arr[:, :2]
    else:
        return None
    # Align to 25 points
    if arr.shape[0] == HORIZON_STEPS:
        return arr
    if arr.shape[0] > HORIZON_STEPS:
        return arr[:HORIZON_STEPS]
    return None


def compute_mms_score(
    pred: np.ndarray,
    expert_like: np.ndarray,
    wrong_speed: Optional[np.ndarray] = None,
    neglect_instruction: Optional[np.ndarray] = None,
    off_road: Optional[np.ndarray] = None,
    crash: Optional[np.ndarray] = None,
) -> float:
    """Compute local MMS (Multi-Maneuver Score) against reference trajectories.

    MMS Logic:
    - Base score 10.0 if closest to expert_like with initial heading alignment.
    - Base score 7.0 if closest to wrong_speed.
    - Base score 3.5 if unmatched.
    - Base score 0.0 if direction mismatch on step 0 or closer to crash/off_road.
    - Comfort penalties:
      - Jerk penalty (up to 44% penalty for erratic 3rd differences).
      - Tortuosity penalty (up to 6% penalty for tortuosity > 1.06).
    """
    pred_arr = np.asarray(pred, dtype=float)
    expert_arr = _prepare_reference(expert_like)
    if expert_arr is None:
        return 3.5

    # 1. Heading check at waypoint 0: dot(ref[0], pred[0]) >= 0.5 * |ref[0]|
    ref_d0 = expert_arr[0]
    pred_d0 = pred_arr[0]
    norm_ref = np.linalg.norm(ref_d0)
    norm_pred = np.linalg.norm(pred_d0)

    if norm_ref > 0.1 and norm_pred > 0.1:
        cos_sim = float(np.dot(ref_d0, pred_d0) / (norm_ref * norm_pred))
        if cos_sim < 0.5:
            return 0.0

    # 2. Check distances to all reference classes
    d_expert = float(np.mean(np.linalg.norm(pred_arr - expert_arr, axis=1)))

    ref_wrong = _prepare_reference(wrong_speed)
    d_wrong = float(np.mean(np.linalg.norm(pred_arr - ref_wrong, axis=1))) if ref_wrong is not None else float("inf")

    ref_neglect = _prepare_reference(neglect_instruction)
    d_neglect = float(np.mean(np.linalg.norm(pred_arr - ref_neglect, axis=1))) if ref_neglect is not None else float("inf")

    ref_offroad = _prepare_reference(off_road)
    d_offroad = float(np.mean(np.linalg.norm(pred_arr - ref_offroad, axis=1))) if ref_offroad is not None else float("inf")

    ref_crash = _prepare_reference(crash)
    d_crash = float(np.mean(np.linalg.norm(pred_arr - ref_crash, axis=1))) if ref_crash is not None else float("inf")

    # Determine base score
    if d_expert <= 8.0 and d_expert <= min(d_wrong, d_neglect, d_offroad, d_crash):
        base_score = spec.MMS_BASE_EXPERT  # 10.0
    elif d_wrong <= 8.0 and d_wrong < d_expert:
        base_score = spec.MMS_BASE_WRONG_SPEED  # 7.0
    elif d_crash <= 5.0 or d_offroad <= 5.0:
        base_score = 0.0
    else:
        base_score = spec.MMS_BASE_UNMATCHED  # 3.5

    # 3. Comfort penalties (jerk & tortuosity)
    diff3 = np.linalg.norm(np.diff(pred_arr, n=3, axis=0), axis=1)
    mean_jerk = float(np.mean(diff3)) / (HISTORY_STEP_S**3)
    jerk_excess = max(0.0, (mean_jerk - 2.5) / 5.0)
    jerk_penalty = min(spec.MMS_JERK_PENALTY_RATIO, jerk_excess * spec.MMS_JERK_PENALTY_RATIO)

    length = float(np.sum(np.linalg.norm(np.diff(pred_arr, axis=0), axis=1)))
    chord = float(np.linalg.norm(pred_arr[-1] - pred_arr[0]))
    tortuosity = length / chord if chord > 0.1 else 1.0
    tort_excess = max(0.0, (tortuosity - 1.06) / 0.5)
    tort_penalty = min(spec.MMS_TORTUOSITY_PENALTY_RATIO, tort_excess * spec.MMS_TORTUOSITY_PENALTY_RATIO)

    total_penalty = jerk_penalty + tort_penalty
    final_score = max(0.0, base_score * (1.0 - total_penalty))
    return float(round(final_score, 2))
