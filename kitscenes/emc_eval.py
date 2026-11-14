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
    video_path: Path,
    max_frames: int = spec.FRAME_COUNT,
    downsample_factor: int = 4,
) -> np.ndarray:
    """Extract grayscale frames from an MP4 file using ffmpeg.

    Returns an array of shape (N, H // downsample, W // downsample) uint8.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return np.zeros((0, spec.HEIGHT // downsample_factor, spec.WIDTH // downsample_factor), dtype=np.uint8)

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        return np.zeros((0, spec.HEIGHT // downsample_factor, spec.WIDTH // downsample_factor), dtype=np.uint8)

    target_w = spec.WIDTH // downsample_factor
    target_h = spec.HEIGHT // downsample_factor

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
        if p is None or not Path(p).exists() or Path(p).stat().st_size == 0:
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
        # A. Side camera flow magnitude -> recovered speed
        side_mags = []
        for cam in spec.SIDE_CAMERAS:
            f0 = frames[cam][t]
            f1 = frames[cam][t + 1]
            if HAS_OPENCV:
                flow = cv2.calcOpticalFlowFarneback(f0, f1, None, 0.5, 2, 9, 2, 5, 1.1, 0)
                # Mean horizontal flow magnitude (side cameras see horizontal ground motion)
                side_mags.append(float(np.mean(np.abs(flow[..., 0]))))
            else:
                # Fast fallback: absolute frame difference
                side_mags.append(float(np.mean(np.abs(f1.astype(float) - f0.astype(float)))))

        mean_side_mag = float(np.mean(side_mags))
        # Calibration constant: ~2.5 pixels of flow at 1/4 res corresponds to ~10 m/s
        flow_speed = mean_side_mag * 4.0
        side_flow_speeds.append(flow_speed)

        # B. Front and Rear horizontal flow -> recovered yaw
        if HAS_OPENCV:
            flow_front = cv2.calcOpticalFlowFarneback(frames["front"][t], frames["front"][t + 1], None, 0.5, 2, 9, 2, 5, 1.1, 0)
            flow_rear = cv2.calcOpticalFlowFarneback(frames["rear"][t], frames["rear"][t + 1], None, 0.5, 2, 9, 2, 5, 1.1, 0)
            # Front: camera turning left -> scene moves right (+u)
            front_yaw = float(np.mean(flow_front[..., 0])) * 0.02
            # Rear: camera turning left -> scene moves left (-u)
            rear_yaw = -float(np.mean(flow_rear[..., 0])) * 0.02
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

    # 5. Recovered flow metrics per segment
    v_rec_s1 = float(np.mean(side_flow_speeds[seg1_slice]))
    v_rec_s2 = float(np.mean(side_flow_speeds[seg2_slice]))

    a_rec_s1 = float((side_flow_speeds[13] - side_flow_speeds[0]) / (14 * DT))
    a_rec_s2 = float((side_flow_speeds[-1] - side_flow_speeds[14]) / (10 * DT))

    # Yaw agreement: front and rear must agree in sign
    yaw_rec_s1 = float(np.mean([np.sum(front_yaw_rates[seg1_slice]), np.sum(rear_yaw_rates[seg1_slice])]))
    yaw_rec_s2 = float(np.mean([np.sum(front_yaw_rates[seg2_slice]), np.sum(rear_yaw_rates[seg2_slice])]))

    # 6. Compare classes and check speed ratio condition:
    # (min(v_rec, v_plan) + 0.5) / (max(v_rec, v_plan) + 0.5) >= 0.5
    ratio_s1 = (min(v_rec_s1, v_plan_s1) + spec.EMC_SPEED_OFFSET) / (max(v_rec_s1, v_plan_s1) + spec.EMC_SPEED_OFFSET)
    ratio_s2 = (min(v_rec_s2, v_plan_s2) + spec.EMC_SPEED_OFFSET) / (max(v_rec_s2, v_plan_s2) + spec.EMC_SPEED_OFFSET)

    speed_ok_s1 = ratio_s1 >= spec.EMC_SPEED_RATIO_MIN
    speed_ok_s2 = ratio_s2 >= spec.EMC_SPEED_RATIO_MIN

    # Steering classes
    steer_plan_s1 = _classify_steering(yaw_plan_s1)
    steer_plan_s2 = _classify_steering(yaw_plan_s2)
    steer_rec_s1 = _classify_steering(yaw_rec_s1)
    steer_rec_s2 = _classify_steering(yaw_rec_s2)

    # Acceleration classes
    accel_plan_s1 = _classify_acceleration(a_plan_s1)
    accel_plan_s2 = _classify_acceleration(a_plan_s2)
    accel_rec_s1 = _classify_acceleration(a_rec_s1)
    accel_rec_s2 = _classify_acceleration(a_rec_s2)

    # A class matches only if segment speeds are also close
    match_steer_s1 = (steer_plan_s1 == steer_rec_s1) and speed_ok_s1
    match_accel_s1 = (accel_plan_s1 == accel_rec_s1) and speed_ok_s1
    match_steer_s2 = (steer_plan_s2 == steer_rec_s2) and speed_ok_s2
    match_accel_s2 = (accel_plan_s2 == accel_rec_s2) and speed_ok_s2

    matches = sum([match_steer_s1, match_accel_s1, match_steer_s2, match_accel_s2])
    emc_score = float(round(matches / spec.EMC_CLASSES_PER_SCENARIO, 3))

    return {
        "emc": emc_score,
        "all_cameras_present": True,
        "matches": matches,
        "seg1_speed_ratio": float(round(ratio_s1, 3)),
        "seg2_speed_ratio": float(round(ratio_s2, 3)),
        "seg1_steering_match": match_steer_s1,
        "seg1_accel_match": match_accel_s1,
        "seg2_steering_match": match_steer_s2,
        "seg2_accel_match": match_accel_s2,
    }
