"""Tests for true optical-flow Ego-Motion Consistency (EMC) evaluator."""

from pathlib import Path
import numpy as np

from kitscenes import spec
from kitscenes.emc_eval import (
    _classify_acceleration,
    _classify_steering,
    compute_video_optical_flow_emc,
)


def test_classify_steering():
    assert _classify_steering(0.0) == "straight"
    assert _classify_steering(0.15) == "left"
    assert _classify_steering(-0.15) == "right"


def test_classify_acceleration():
    assert _classify_acceleration(0.0) == "maintain"
    assert _classify_acceleration(1.2) == "accelerate"
    assert _classify_acceleration(-1.2) == "brake"


def test_emc_missing_camera_returns_zero():
    # Only front camera provided; 5 cameras missing
    cam_paths = {"front": Path("dummy_front.mp4")}
    plan = np.zeros((25, 2))
    for i in range(25):
        plan[i] = [(i + 1) * 2.0, 0.0]

    res = compute_video_optical_flow_emc(cam_paths, plan)
    assert res["emc"] == 0.0
    assert res["all_cameras_present"] is False
    assert res["matches"] == 0


def test_emc_empty_or_nonexistent_paths():
    cam_paths = {cam: Path(f"nonexistent_{cam}.mp4") for cam in spec.CAMERAS}
    plan = np.zeros((25, 2))
    res = compute_video_optical_flow_emc(cam_paths, plan)
    assert res["emc"] == 0.0
    assert res["all_cameras_present"] is False
