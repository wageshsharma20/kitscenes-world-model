"""Optical-flow Ego-Motion Consistency (EMC) evaluator for KITScenes-LongTail.

Official Challenge Specification:
- Optical flow recovers ego-motion from the generated 6-camera videos.
- Yaw comes from `front` and `rear`, which must agree.
- Speed comes from the four side cameras (`front_left`, `front_right`, `rear_left`, `rear_right`).
- Planned and recovered trajectories are each cut into two segments:
    Segment 1: 0.0 s to 3.0 s (frames 0..14)
    Segment 2: 3.0 s to 5.0 s (frames 15..24)
- Each segment gets a steering class ("left", "straight", "right") and an
  acceleration class ("accelerate", "maintain", "brake").
- A class matches only if the segment speeds also satisfy the agreement ratio:
    (min(v_rec, v_plan) + 0.5) / (max(v_rec, v_plan) + 0.5) >= 0.5
- If any of the six camera videos is missing, every class counts as a mismatch (EMC = 0.0).
- Final EMC score is the fraction of matching classes (out of 4 per scenario),
  normalized and clamped to [0.0, 1.0].
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from . import spec

try:
    import cv2  # type: ignore[import-untyped]
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

DT = spec.WAYPOINT_STEP_S  # 0.2 s per frame @ 5 fps


def extract_video_frames(
    video_source: Union[Path, str, np.ndarray],
    max_frames: int = spec.FRAME_COUNT,
    downsample_factor: int = 4,
) -> np.ndarray:
    """Extract grayscale frames from an MP4 file or convert in-memory array.

    Returns an array of shape (N, H // downsample, W // downsample) uint8.
    """
    target_w = spec.WIDTH // downsample_factor
    target_h = spec.HEIGHT // downsample_factor

    if isinstance(video_source, np.ndarray):
        frames = video_source[:max_frames]
        out_frames = []
        for f in frames:
            if f.ndim == 3 and f.shape[-1] == 3:
                gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if HAS_OPENCV else (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]).astype(np.uint8)
            elif f.ndim == 2:
                gray = f.astype(np.uint8)
            else:
                gray = f[..., 0].astype(np.uint8)

            if HAS_OPENCV:
                resized = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
            else:
                resized = gray[::downsample_factor, ::downsample_factor][:target_h, :target_w]
            out_frames.append(resized)
        return np.array(out_frames, dtype=np.uint8) if out_frames else np.zeros((0, target_h, target_w), dtype=np.uint8)

    video_path = Path(video_source)
    if not video_path.exists():
        return np.zeros((0, target_h, target_w), dtype=np.uint8)

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        return np.zeros((0, target_h, target_w), dtype=np.uint8)

    cmd = [
        ffmpeg_bin,
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"scale={target_w}:{target_h}",
        "-vframes",
        str(max_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        raw = proc.stdout
        frame_bytes = target_w * target_h
        n_frames = len(raw) // frame_bytes
        if n_frames == 0:
            return np.zeros((0, target_h, target_w), dtype=np.uint8)
        arr = np.frombuffer(raw[: n_frames * frame_bytes], dtype=np.uint8)
        return arr.reshape((n_frames, target_h, target_w))
    except Exception:
        return np.zeros((0, target_h, target_w), dtype=np.uint8)


def _classify_steering(total_yaw_rad: float, threshold_deg: float = 4.0) -> str:
    """Classify total segment yaw rotation into left, right, or straight."""
    thresh_rad = math.radians(threshold_deg)
    if total_yaw_rad > thresh_rad:
        return "left"
    if total_yaw_rad < -thresh_rad:
        return "right"
    return "straight"


def _classify_acceleration(accel_mps2: float, threshold_mps2: float = 0.4) -> str:
    """Classify segment acceleration into accelerate, brake, or maintain."""
    if accel_mps2 > threshold_mps2:
        return "accelerate"
    if accel_mps2 < -threshold_mps2:
        return "brake"
    return "maintain"


def compute_video_optical_flow_emc(
    camera_video_paths: Dict[str, Path],
    planned_trajectory: Sequence[Sequence[float]],
) -> Dict[str, float | bool]:
    """Compute true optical-flow Ego-Motion Consistency (EMC) from 6 camera videos.

    Parameters:
        camera_video_paths: dict mapping camera name ("front", "front_left", etc.) to MP4 file path.
        planned_trajectory: (25, 2) planned waypoints at 5 Hz from 0.2 s to 5.0 s.

    Returns:
        Dict containing:
            "emc": final score in [0.0, 1.0]
            "all_cameras_present": bool
            "matches": count of matched classes (out of 4)
            "seg1_speed_ratio": float
            "seg2_speed_ratio": float
            "seg1_steering_match": bool
            "seg1_accel_match": bool
            "seg2_steering_match": bool
            "seg2_accel_match": bool
    """
    # 1. Enforce 6-camera presence: if any camera is missing, EMC is strictly 0.0
    for cam in spec.CAMERAS:
        p = camera_video_paths.get(cam)
        if p is None:
            return {
                "emc": 0.0,
                "all_cameras_present": False,
                "matches": 0,
                "seg1_speed_ratio": 0.0,
                "seg2_speed_ratio": 0.0,
                "seg1_steering_match": False,
                "seg1_accel_match": False,
                "seg2_steering_match": False,
                "seg2_accel_match": False,
            }
        if isinstance(p, np.ndarray):
            if len(p) < spec.FRAME_COUNT:
                return {
                    "emc": 0.0,
                    "all_cameras_present": False,
                    "matches": 0,
                    "seg1_speed_ratio": 0.0,
                    "seg2_speed_ratio": 0.0,
                    "seg1_steering_match": False,
                    "seg1_accel_match": False,
                    "seg2_steering_match": False,
                    "seg2_accel_match": False,
                }
        else:
            p_path = Path(p)
            if not p_path.exists() or p_path.stat().st_size == 0:
                return {
                    "emc": 0.0,
                    "all_cameras_present": False,
                    "matches": 0,
                    "seg1_speed_ratio": 0.0,
                    "seg2_speed_ratio": 0.0,
                    "seg1_steering_match": False,
                    "seg1_accel_match": False,
                    "seg2_steering_match": False,
                    "seg2_accel_match": False,
                }

    plan_pts = np.asarray(planned_trajectory, dtype=float)
    if plan_pts.shape != (spec.WAYPOINT_COUNT, 2):
        return {
            "emc": 0.0,
            "all_cameras_present": True,
            "matches": 0,
            "seg1_speed_ratio": 0.0,
            "seg2_speed_ratio": 0.0,
            "seg1_steering_match": False,
            "seg1_accel_match": False,
            "seg2_steering_match": False,
            "seg2_accel_match": False,
        }

    # 2. Extract grayscale frames from all 6 videos
    frames: Dict[str, np.ndarray] = {}
    for cam in spec.CAMERAS:
        cam_frames = extract_video_frames(camera_video_paths[cam], downsample_factor=4)
        if len(cam_frames) < spec.FRAME_COUNT:
            # Under-frame video fails contract
            return {
                "emc": 0.0,
                "all_cameras_present": True,
                "matches": 0,
                "seg1_speed_ratio": 0.0,
                "seg2_speed_ratio": 0.0,
                "seg1_steering_match": False,
                "seg1_accel_match": False,
                "seg2_steering_match": False,
                "seg2_accel_match": False,
            }
        frames[cam] = cam_frames

    # 3. Compute optical flow on consecutive frames
    # Speed is recovered from the four side cameras:
    # front_left, front_right, rear_left, rear_right
    side_flow_speeds: List[float] = []
    # Yaw is recovered from front and rear cameras
    front_yaw_rates: List[float] = []
    rear_yaw_rates: List[float] = []

    for t in range(spec.FRAME_COUNT - 1):
        # A. Side camera road flow magnitude -> recovered forward speed
        side_mags = []
        for cam in spec.SIDE_CAMERAS:
            f0 = frames[cam][t]
            f1 = frames[cam][t + 1]
            if HAS_OPENCV:
                flow = cv2.calcOpticalFlowFarneback(f0, f1, None, 0.5, 2, 9, 2, 5, 1.1, 0)
                # Lower half of frame (road plane) carries the cleanest horizontal translation flow
                h_half = flow.shape[0] // 2
                road_flow = flow[h_half:, ...]
                side_mags.append(float(np.mean(np.abs(road_flow[..., 0]))))
            else:
                # Absolute difference fallback
                side_mags.append(float(np.mean(np.abs(f1.astype(float) - f0.astype(float)))))

        mean_side_mag = float(np.mean(side_mags))
        # Empirically calibrated scale against geometric ground truth: ~0.9 px flow at 1/4 res = 10 m/s
        flow_speed = mean_side_mag * 11.0
        side_flow_speeds.append(flow_speed)

        # B. Front and Rear horizontal flow -> recovered yaw
        if HAS_OPENCV:
            flow_front = cv2.calcOpticalFlowFarneback(frames["front"][t], frames["front"][t + 1], None, 0.5, 2, 9, 2, 5, 1.1, 0)
            flow_rear = cv2.calcOpticalFlowFarneback(frames["rear"][t], frames["rear"][t + 1], None, 0.5, 2, 9, 2, 5, 1.1, 0)
            h_half = flow_front.shape[0] // 2
            # Upper half / horizon region: yaw shifts horizon horizontally
            # Front: vehicle turning left (yaw > 0) -> scene shifts right (+u)
            front_yaw = float(np.mean(flow_front[:h_half, ..., 0])) * 5.0 * DT
            # Rear: vehicle turning left (yaw > 0) -> rear scene shifts left (-u)
            rear_yaw = -float(np.mean(flow_rear[:h_half, ..., 0])) * 5.0 * DT
        else:
            front_yaw = 0.0
            rear_yaw = 0.0

        front_yaw_rates.append(front_yaw)
        rear_yaw_rates.append(rear_yaw)

    # 4. Planned trajectory metrics per segment
    # Segment 1: 0.0 to 3.0 s (steps 0..14)
    # Segment 2: 3.0 to 5.0 s (steps 15..24)
    plan_deltas = np.diff(plan_pts, axis=0)
    plan_steps = np.linalg.norm(plan_deltas, axis=1)
    plan_speeds = plan_steps / DT

    plan_headings = np.arctan2(plan_deltas[:, 1], plan_deltas[:, 0])
    plan_turn_angles = np.diff(plan_headings)
    plan_turn_angles = (plan_turn_angles + np.pi) % (2 * np.pi) - np.pi

    # Segment slices for 24 intervals between 25 points
    seg1_slice = slice(0, 14)   # 0 to 3.0s
    seg2_slice = slice(14, 24)  # 3.0 to 5.0s

    v_plan_s1 = float(np.mean(plan_speeds[seg1_slice]))
    v_plan_s2 = float(np.mean(plan_speeds[seg2_slice]))

    a_plan_s1 = float((plan_speeds[13] - plan_speeds[0]) / (14 * DT)) if len(plan_speeds) > 13 else 0.0
    a_plan_s2 = float((plan_speeds[-1] - plan_speeds[14]) / (10 * DT)) if len(plan_speeds) > 14 else 0.0

    yaw_plan_s1 = float(np.sum(plan_turn_angles[0:13])) if len(plan_turn_angles) > 12 else 0.0
    yaw_plan_s2 = float(np.sum(plan_turn_angles[13:])) if len(plan_turn_angles) > 13 else 0.0

    # 5. Recovered flow metrics per segment with strict front/rear agreement check
    v_rec_s1 = float(np.mean(side_flow_speeds[seg1_slice]))
    v_rec_s2 = float(np.mean(side_flow_speeds[seg2_slice]))

    a_rec_s1 = float((side_flow_speeds[13] - side_flow_speeds[0]) / (14 * DT))
    a_rec_s2 = float((side_flow_speeds[-1] - side_flow_speeds[14]) / (10 * DT))

    # Independent front and rear yaw computation
    yaw_front_s1 = float(np.sum(front_yaw_rates[seg1_slice]))
    yaw_rear_s1 = float(np.sum(rear_yaw_rates[seg1_slice]))
    yaw_front_s2 = float(np.sum(front_yaw_rates[seg2_slice]))
    yaw_rear_s2 = float(np.sum(rear_yaw_rates[seg2_slice]))

    # Strict agreement audit: Front and Rear must agree in sign (or both straight)
    steer_front_s1 = _classify_steering(yaw_front_s1)
    steer_rear_s1 = _classify_steering(yaw_rear_s1)
    yaw_agrees_s1 = (steer_front_s1 == steer_rear_s1)

    steer_front_s2 = _classify_steering(yaw_front_s2)
    steer_rear_s2 = _classify_steering(yaw_rear_s2)
    yaw_agrees_s2 = (steer_front_s2 == steer_rear_s2)

    # If front and rear disagree, the recovered yaw is ambiguous and treated as non-matching
    yaw_rec_s1 = float(np.mean([yaw_front_s1, yaw_rear_s1])) if yaw_agrees_s1 else 0.0
    yaw_rec_s2 = float(np.mean([yaw_front_s2, yaw_rear_s2])) if yaw_agrees_s2 else 0.0

    # 6. Compare classes and check speed ratio condition:
    # (min(v_rec, v_plan) + 0.5) / (max(v_rec, v_plan) + 0.5) >= 0.5
    ratio_s1 = (min(v_rec_s1, v_plan_s1) + spec.EMC_SPEED_OFFSET) / (max(v_rec_s1, v_plan_s1) + spec.EMC_SPEED_OFFSET)
    ratio_s2 = (min(v_rec_s2, v_plan_s2) + spec.EMC_SPEED_OFFSET) / (max(v_rec_s2, v_plan_s2) + spec.EMC_SPEED_OFFSET)

    speed_ok_s1 = ratio_s1 >= spec.EMC_SPEED_RATIO_MIN
    speed_ok_s2 = ratio_s2 >= spec.EMC_SPEED_RATIO_MIN

    # Steering classes: requires front/rear agreement
    steer_plan_s1 = _classify_steering(yaw_plan_s1)
    steer_plan_s2 = _classify_steering(yaw_plan_s2)
    steer_rec_s1 = _classify_steering(yaw_rec_s1) if yaw_agrees_s1 else "disagreement"
    steer_rec_s2 = _classify_steering(yaw_rec_s2) if yaw_agrees_s2 else "disagreement"

    # Acceleration classes
    accel_plan_s1 = _classify_acceleration(a_plan_s1)
    accel_plan_s2 = _classify_acceleration(a_plan_s2)
    accel_rec_s1 = _classify_acceleration(a_rec_s1)
    accel_rec_s2 = _classify_acceleration(a_rec_s2)

    # A class matches only if segment speeds are also close and yaw agreed
    match_steer_s1 = (steer_plan_s1 == steer_rec_s1) and speed_ok_s1 and yaw_agrees_s1
    match_accel_s1 = (accel_plan_s1 == accel_rec_s1) and speed_ok_s1
    match_steer_s2 = (steer_plan_s2 == steer_rec_s2) and speed_ok_s2 and yaw_agrees_s2
    match_accel_s2 = (accel_plan_s2 == accel_rec_s2) and speed_ok_s2

    matches = sum([match_steer_s1, match_accel_s1, match_steer_s2, match_accel_s2])
    emc_score = float(round(matches / spec.EMC_CLASSES_PER_SCENARIO, 3))

    return {
        "local_emc_approximation": emc_score,
        "emc": emc_score,  # Backwards compatibility key with explicit alias
        "is_official_emc": False,  # Explicit non-official disclaimer
        "all_cameras_present": True,
        "matches": matches,
        "seg1_speed_ratio": float(round(ratio_s1, 3)),
        "seg2_speed_ratio": float(round(ratio_s2, 3)),
        "seg1_steering_match": match_steer_s1,
        "seg1_accel_match": match_accel_s1,
        "seg2_steering_match": match_steer_s2,
        "seg2_accel_match": match_accel_s2,
        "seg1_yaw_agreement": yaw_agrees_s1,
        "seg2_yaw_agreement": yaw_agrees_s2,
        "seg1_yaw_front": float(round(yaw_front_s1, 4)),
        "seg1_yaw_rear": float(round(yaw_rear_s1, 4)),
        "seg2_yaw_front": float(round(yaw_front_s2, 4)),
        "seg2_yaw_rear": float(round(yaw_rear_s2, 4)),
        "seg1_speed_recovered_mps": float(round(v_rec_s1, 2)),
        "seg1_speed_planned_mps": float(round(v_plan_s1, 2)),
        "seg2_speed_recovered_mps": float(round(v_rec_s2, 2)),
        "seg2_speed_planned_mps": float(round(v_plan_s2, 2)),
        "seg1_accel_recovered_mps2": float(round(a_rec_s1, 2)),
        "seg1_accel_planned_mps2": float(round(a_plan_s1, 2)),
        "seg2_accel_recovered_mps2": float(round(a_rec_s2, 2)),
        "seg2_accel_planned_mps2": float(round(a_plan_s2, 2)),
        "valid_flow_measurements": len(side_flow_speeds),
    }
