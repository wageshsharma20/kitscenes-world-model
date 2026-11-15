"""Spatial-Conditioned World Model Dataset & Paired Tuple Extractor.

Extracts complete paired training tuples conforming to the KITScenes-LongTail specification:
  - Initial camera RGB frame (t = 0.0s): [B, 3, 568, 876] or [B, 6, 3, 568, 876]
  - Future RGB video target (25 frames @ 5 fps): [B, 25, 3, 568, 876] or [B, 6, 25, 3, 568, 876]
  - Ground-truth trajectory (5 Hz, 5.0s): [B, 25, 2]
  - Option A: Static Route Corridor Mask: [B, 1, 568, 876] or [B, 6, 1, 568, 876]
  - Option B.1: Route-Initial-Frame Temporal Masks: [B, 25, 1, 568, 876] or [B, 6, 25, 1, 568, 876]
  - Option B.2: Future-Camera-Frame Spatial Masks: [B, 25, 1, 568, 876] or [B, 6, 25, 1, 568, 876]
  - Derived kinematics (v0, w0) and SVD motion bucket
  - Sensor calibration metadata (provisional nominal model)

Enforces strict partition isolation: no validation scenarios (0000900-0000999) can ever
enter the training dataset.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
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

    window = min(5, len(past))
    dx = past[-1, 0] - past[-window, 0]
    dy = past[-1, 1] - past[-window, 1]
    dt = (window - 1) * 0.1
    speed = float(math.hypot(dx, dy) / max(dt, 1e-4))

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


def load_future_video_tensor(
    video_source_dir: Path,
    scenario_id: str,
    camera_name: str,
    num_frames: int = spec.FRAME_COUNT,
    width: int = spec.WIDTH,
    height: int = spec.HEIGHT,
) -> Optional[torch.Tensor]:
    """Load a 25-frame future ground-truth video target from disk.

    Supports:
      1. MP4 container: <video_source_dir>/<scenario_id>/<camera_name>.mp4
      2. Frame sequence directory: <video_source_dir>/<scenario_id>/<camera_name>/
    """
    scene_dir = video_source_dir / scenario_id
    if not scene_dir.exists():
        return None

    # Check for MP4 file
    mp4_path = scene_dir / f"{camera_name}.mp4"
    if mp4_path.exists():
        cap = cv2.VideoCapture(str(mp4_path))
        frames = []
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR -> RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if rgb.shape[1] != width or rgb.shape[0] != height:
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
        cap.release()

        if len(frames) == num_frames:
            arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (25, H, W, 3)
            tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)       # (25, 3, H, W)
            return tensor

    # Check for frame image folder
    frame_dir = scene_dir / camera_name
    if frame_dir.is_dir():
        img_files = sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        if len(img_files) >= num_frames:
            frames = []
            for p in img_files[:num_frames]:
                im = Image.open(p).convert("RGB")
                if im.size != (width, height):
                    im = im.resize((width, height), Image.Resampling.LANCZOS)
                arr = np.asarray(im, dtype=np.float32) / 255.0
                frames.append(arr)
            arr_stacked = np.stack(frames, axis=0)  # (25, H, W, 3)
            tensor = torch.from_numpy(arr_stacked).permute(0, 3, 1, 2)  # (25, 3, H, W)
            return tensor

    return None


class SpatialConditionedWorldModelDataset(Dataset):
    """PyTorch Dataset yielding complete paired multi-modal training tuples for world models."""

    def __init__(
        self,
        trajectories_file: Union[str, Path] = "data/train_trajectories.jsonl",
        observed_t0_dir: Optional[Union[str, Path]] = "data/observed_t0",
        future_video_dir: Optional[Union[str, Path]] = None,
        camera_name: Optional[str] = "front",
        split: str = "train",
        require_future_video: bool = False,
        require_initial_rgb: bool = False,
        projection_engine: Optional[camera_projection.CameraProjectionEngine] = None,
        num_frames: int = spec.FRAME_COUNT,
    ):
        self.trajectories_file = Path(trajectories_file)
        self.observed_t0_dir = Path(observed_t0_dir) if observed_t0_dir else None
        self.future_video_dir = Path(future_video_dir) if future_video_dir else None
        self.camera_name = camera_name  # None extracts all 6 cameras
        self.split = split
        self.require_future_video = require_future_video
        self.require_initial_rgb = require_initial_rgb
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

        gt_raw = np.asarray(rec["expert_like"], dtype=np.float32)
        if gt_raw.shape[0] < self.num_frames:
            alphas = np.linspace(0.0, 1.0, self.num_frames)
            orig_alphas = np.linspace(0.0, 1.0, max(gt_raw.shape[0], 1))
            gt_traj = np.column_stack([
                np.interp(alphas, orig_alphas, gt_raw[:, 0]),
                np.interp(alphas, orig_alphas, gt_raw[:, 1]),
            ]).astype(np.float32)
        else:
            gt_traj = gt_raw[: self.num_frames]
        past_traj = np.asarray(rec.get("past", [[0.0, 0.0]]), dtype=np.float32)  # (21, 2)

        v0, w0 = calculate_kinematics_from_past(past_traj)
        motion_bucket = video_engine.derive_motion_bucket(v0)

        # 1. Spatial Conditioning Masks: Option A, Option B.1, Option B.2
        if self.camera_name is not None:
            # Single camera
            static_mask = self.engine.render_static_route_mask(
                self.camera_name, gt_traj, channel_dim=True
            )  # (1, 568, 876)
            b1_masks = self.engine.render_initial_frame_temporal_masks(
                self.camera_name, gt_traj, num_frames=self.num_frames, channel_dim=True
            )  # (25, 1, 568, 876)
            b2_masks = self.engine.render_future_camera_spatial_masks(
                self.camera_name, gt_traj, num_frames=self.num_frames, channel_dim=True
            )  # (25, 1, 568, 876)
        else:
            # All 6 cameras stacked
            static_dict = self.engine.render_static_route_masks_all_cameras(gt_traj)
            b1_dict = {c: self.engine.render_initial_frame_temporal_masks(c, gt_traj, num_frames=self.num_frames, channel_dim=True) for c in spec.CAMERAS}
            b2_dict = {c: self.engine.render_future_camera_spatial_masks(c, gt_traj, num_frames=self.num_frames, channel_dim=True) for c in spec.CAMERAS}

            static_mask = np.stack([static_dict[c] for c in spec.CAMERAS], axis=0)  # (6, 1, 568, 876)
            b1_masks = np.stack([b1_dict[c] for c in spec.CAMERAS], axis=0)        # (6, 25, 1, 568, 876)
            b2_masks = np.stack([b2_dict[c] for c in spec.CAMERAS], axis=0)        # (6, 25, 1, 568, 876)

        # 2. Initial RGB Frame (t=0.0s)
        rgb_tensor = None
        has_initial_rgb = False
        if self.observed_t0_dir:
            scene_dir = self.observed_t0_dir / sid
            if scene_dir.exists():
                if self.camera_name is not None:
                    img_path = scene_dir / f"{self.camera_name}.jpg"
                    if img_path.exists():
                        im = Image.open(img_path).convert("RGB")
                        arr = np.asarray(im, dtype=np.float32) / 255.0
                        rgb_tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, 568, 876)
                        has_initial_rgb = True
                else:
                    cams_list = []
                    all_found = True
                    for c in spec.CAMERAS:
                        img_path = scene_dir / f"{c}.jpg"
                        if img_path.exists():
                            im = Image.open(img_path).convert("RGB")
                            arr = np.asarray(im, dtype=np.float32) / 255.0
                            cams_list.append(torch.from_numpy(arr).permute(2, 0, 1))
                        else:
                            all_found = False
                            break
                    if all_found:
                        rgb_tensor = torch.stack(cams_list, dim=0)  # (6, 3, 568, 876)
                        has_initial_rgb = True

        if self.require_initial_rgb and not has_initial_rgb:
            raise FileNotFoundError(
                f"Missing required initial RGB frame for scenario {sid} in {self.observed_t0_dir}"
            )

        # 3. Future Ground-Truth Video Targets (25 frames @ 5 fps)
        future_video = None
        has_future_video = False
        if self.future_video_dir:
            if self.camera_name is not None:
                future_video = load_future_video_tensor(
                    self.future_video_dir,
                    sid,
                    self.camera_name,
                    num_frames=self.num_frames,
                    width=self.engine.width,
                    height=self.engine.height,
                )
                has_future_video = future_video is not None
            else:
                cams_vids = []
                all_vids_found = True
                for c in spec.CAMERAS:
                    v_cam = load_future_video_tensor(
                        self.future_video_dir,
                        sid,
                        c,
                        num_frames=self.num_frames,
                        width=self.engine.width,
                        height=self.engine.height,
                    )
                    if v_cam is not None:
                        cams_vids.append(v_cam)
                    else:
                        all_vids_found = False
                        break
                if all_vids_found:
                    future_video = torch.stack(cams_vids, dim=0)  # (6, 25, 3, 568, 876)
                    has_future_video = True

        if self.require_future_video and not has_future_video:
            raise FileNotFoundError(
                f"Missing required future ground-truth video target for scenario {sid} in {self.future_video_dir}"
            )

        # Calibration dictionary
        if self.camera_name is not None:
            cam = self.engine.rig[self.camera_name]
            calib = {
                "fx": cam.fx,
                "fy": cam.fy,
                "cx": cam.cx,
                "cy": cam.cy,
                "intrinsic_matrix": cam.intrinsic_matrix,
                "position_m": cam.position_m,
                "yaw_deg": cam.yaw_deg,
                "is_provisional": cam.is_provisional,
            }
        else:
            calib = {
                "is_provisional": self.engine.rig.is_provisional,
                "cameras": {c: self.engine.rig[c].intrinsic_matrix for c in spec.CAMERAS},
            }

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
            "initial_frame_temporal_masks": torch.from_numpy(b1_masks),
            "future_camera_spatial_masks": torch.from_numpy(b2_masks),
            "time_indexed_masks": torch.from_numpy(b2_masks),  # Default Option B alias
            "initial_rgb": rgb_tensor,
            "has_initial_rgb": has_initial_rgb,
            "future_video": future_video,
            "has_future_video": has_future_video,
            "calibration": calib,
        }


def collate_spatial_dataset(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collator for SpatialConditionedWorldModelDataset."""
    collated = {
        "scenario_id": [d["scenario_id"] for d in batch],
        "driving_instruction": [d["driving_instruction"] for d in batch],
        "scenario_type": [d["scenario_type"] for d in batch],
        "gt_trajectory": torch.stack([d["gt_trajectory"] for d in batch], dim=0),
        "past_trajectory": torch.stack([d["past_trajectory"] for d in batch], dim=0),
        "v0": torch.tensor([d["v0"] for d in batch], dtype=torch.float32),
        "w0": torch.tensor([d["w0"] for d in batch], dtype=torch.float32),
        "motion_bucket": torch.tensor([d["motion_bucket"] for d in batch], dtype=torch.long),
        "static_route_mask": torch.stack([d["static_route_mask"] for d in batch], dim=0),
        "initial_frame_temporal_masks": torch.stack([d["initial_frame_temporal_masks"] for d in batch], dim=0),
        "future_camera_spatial_masks": torch.stack([d["future_camera_spatial_masks"] for d in batch], dim=0),
        "time_indexed_masks": torch.stack([d["time_indexed_masks"] for d in batch], dim=0),
        "has_initial_rgb": [d["has_initial_rgb"] for d in batch],
        "has_future_video": [d["has_future_video"] for d in batch],
        "calibration": [d["calibration"] for d in batch],
    }

    # Stack initial RGB if present across all samples
    if all(d["has_initial_rgb"] for d in batch) and batch[0]["initial_rgb"] is not None:
        collated["initial_rgb"] = torch.stack([d["initial_rgb"] for d in batch], dim=0)
    else:
        collated["initial_rgb"] = None

    # Stack future video if present across all samples
    if all(d["has_future_video"] for d in batch) and batch[0]["future_video"] is not None:
        collated["future_video"] = torch.stack([d["future_video"] for d in batch], dim=0)
    else:
        collated["future_video"] = None

    return collated
