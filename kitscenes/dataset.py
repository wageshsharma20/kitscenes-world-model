"""Spatial-Conditioned World Model Dataset & Paired Tuple Extractor.

Extracts paired training tuples conforming to the KITScenes-LongTail specification:
  - Initial camera RGB frame (t = 0.0s)
  - 25-frame future ground truth trajectory (5 Hz, 5.0 seconds)
  - Option A: Static Route Corridor Mask [B, 1, 568, 876]
  - Option B: Time-Indexed Spatial Conditioning Maps [B, 25, 1, 568, 876]
  - Initial speed v0 and derived motion bucket

Enforces strict partition isolation: no validation scenarios (0000900-0000999) can ever
enter the training dataset.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from kitscenes import camera_projection, spec, video_engine


def calculate_kinematics_from_past(past: np.ndarray) -> Tuple[float, float]:
    """Derive initial forward velocity v0 (m/s) and yaw rate w0 (rad/s) from past waypoints.

    Past array has shape (21, 2) at 10 Hz (dt = 0.1s), running from t = -2.0s to 0.0s.
    """
    if len(past) < 2:
        return 0.0, 0.0

    # Use the last 5 steps (-0.4s to 0.0s) for low-noise finite difference
    window = min(5, len(past))
    dx = past[-1, 0] - past[-window, 0]
    dy = past[-1, 1] - past[-window, 1]
    dt = (window - 1) * 0.1
    speed = float(math.hypot(dx, dy) / max(dt, 1e-4))

    # Yaw rate estimation
    if len(past) >= 3:
        headings = []
        for i in range(-window, -1):
            ddx = past[i + 1, 0] - past[i, 0]
            ddy = past[i + 1, 1] - past[i, 1]
            if math.hypot(ddx, ddy) > 0.05:
                headings.append(math.atan2(ddy, ddx))
        if len(headings) >= 2:
            dyaw = video_engine.wrap_angle(headings[-1] - headings[0])
            yaw_rate = float(dyaw / max(dt, 1e-4))
        else:
            yaw_rate = 0.0
    else:
        yaw_rate = 0.0

    return speed, yaw_rate


class SpatialConditionedWorldModelDataset(Dataset):
    """PyTorch Dataset yielding paired multi-modal training tuples for spatial world models."""

    def __init__(
        self,
        trajectories_file: Union[str, Path] = "data/train_trajectories.jsonl",
        observed_t0_dir: Optional[Union[str, Path]] = "data/observed_t0",
        camera_name: Optional[str] = "front",
        split: str = "train",
        projection_engine: Optional[camera_projection.CameraProjectionEngine] = None,
        num_frames: int = spec.FRAME_COUNT,
    ):
        self.trajectories_file = Path(trajectories_file)
        self.observed_t0_dir = Path(observed_t0_dir) if observed_t0_dir else None
        self.camera_name = camera_name  # If None, extracts all 6 cameras
        self.split = split
        self.num_frames = num_frames
        self.engine = projection_engine or camera_projection.CameraProjectionEngine()

        if not self.trajectories_file.exists():
            raise FileNotFoundError(f"Trajectories file not found: {self.trajectories_file}")

        # Load and validate records
        self.records: List[Dict[str, Any]] = []
        with open(self.trajectories_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                sid = str(data["scenario_id"])

                # STRICT PARTITION INTEGRITY CHECK
                if self.split == "train":
                    # Validation scenarios are strictly 0000900-0000999
                    if 900 <= int(sid) <= 999:
                        raise ValueError(
                            f"CRITICAL DATA LEAKAGE DETECTED at line {line_no}: "
                            f"Validation scenario {sid} is present in training dataset!"
                        )

                self.records.append(data)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        sid = rec["scenario_id"]
        instruction = rec.get("driving_instruction", "")
        scenario_type = rec.get("scenario_type", "")

        gt_traj = np.asarray(rec["expert_like"], dtype=np.float32)  # (25, 2)
        past_traj = np.asarray(rec.get("past", [[0.0, 0.0]]), dtype=np.float32)  # (21, 2)

        v0, w0 = calculate_kinematics_from_past(past_traj)
        motion_bucket = video_engine.derive_motion_bucket(v0)

        # 1. Option A & Option B Spatial Conditioning Masks
        if self.camera_name is not None:
            # Single camera
            static_mask = self.engine.render_static_route_mask(
                self.camera_name, gt_traj, channel_dim=True
            )  # (1, 568, 876)
            time_indexed_masks = self.engine.render_time_indexed_spatial_masks(
                self.camera_name, gt_traj, num_frames=self.num_frames, channel_dim=True
            )  # (25, 1, 568, 876)
        else:
            # All 6 cameras stacked: shape (6, 1, 568, 876) and (6, 25, 1, 568, 876)
            static_dict = self.engine.render_static_route_masks_all_cameras(gt_traj)
            time_dict = self.engine.render_time_indexed_spatial_masks_all_cameras(
                gt_traj, num_frames=self.num_frames
            )
            static_mask = np.stack([static_dict[c] for c in spec.CAMERAS], axis=0)
            time_indexed_masks = np.stack([time_dict[c] for c in spec.CAMERAS], axis=0)

        # 2. RGB image loading (if available on disk)
        rgb_tensor = None
        has_rgb = False
        if self.observed_t0_dir:
            scene_dir = self.observed_t0_dir / sid
            if scene_dir.exists():
                if self.camera_name is not None:
                    img_path = scene_dir / f"{self.camera_name}.jpg"
                    if img_path.exists():
                        im = Image.open(img_path).convert("RGB")
                        arr = np.asarray(im, dtype=np.float32) / 255.0  # (568, 876, 3)
                        rgb_tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, 568, 876)
                        has_rgb = True
                else:
                    # All 6 cameras
                    cams_list = []
                    found_any = False
                    for c in spec.CAMERAS:
                        img_path = scene_dir / f"{c}.jpg"
                        if img_path.exists():
                            im = Image.open(img_path).convert("RGB")
                            arr = np.asarray(im, dtype=np.float32) / 255.0
                            cams_list.append(torch.from_numpy(arr).permute(2, 0, 1))
                            found_any = True
                        else:
                            cams_list.append(torch.zeros((3, self.engine.height, self.engine.width), dtype=torch.float32))
                    rgb_tensor = torch.stack(cams_list, dim=0)  # (6, 3, 568, 876)
                    has_rgb = found_any

        if rgb_tensor is None:
            if self.camera_name is not None:
                rgb_tensor = torch.zeros((3, self.engine.height, self.engine.width), dtype=torch.float32)
            else:
                rgb_tensor = torch.zeros((6, 3, self.engine.height, self.engine.width), dtype=torch.float32)

        return {
            "scenario_id": sid,
            "driving_instruction": instruction,
            "scenario_type": scenario_type,
            "gt_trajectory": torch.from_numpy(gt_traj),
            "past_trajectory": torch.from_numpy(past_traj),
            "v0": float(v0),
            "w0": float(w0),
            "motion_bucket": int(motion_bucket),
            "static_route_mask": torch.from_numpy(static_mask),
            "time_indexed_masks": torch.from_numpy(time_indexed_masks),
            "initial_rgb": rgb_tensor,
            "has_rgb": has_rgb,
        }
