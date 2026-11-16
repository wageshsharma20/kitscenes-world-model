"""Unit tests for SpatialConditionedWorldModelDataset and spatial conditioning tensors."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from kitscenes import dataset, spec


def test_partition_integrity_blocks_validation_scenarios(tmp_path):
    """Ensure any validation scenario (0000900-0000999) raises an immediate ValueError in train split."""
    bad_data = [
        {
            "scenario_id": "0000950",
            "driving_instruction": "drive straight on",
            "scenario_type": "1 heavy rain",
            "past": [[0.0, 0.0]],
            "expert_like": [[float(i), 0.0] for i in range(1, 26)],
        }
    ]
    jsonl_path = tmp_path / "leaked_train.jsonl"
    with open(jsonl_path, "w") as f:
        for r in bad_data:
            f.write(json.dumps(r) + "\n")

    with pytest.raises(ValueError, match="CRITICAL DATA LEAKAGE DETECTED"):
        dataset.SpatialConditionedWorldModelDataset(
            trajectories_file=jsonl_path,
            split="train",
        )


def test_dataset_tensor_shapes_and_kinematics(tmp_path):
    """Verify single-camera and 6-camera tensor shapes, kinematics, and motion bucket derivation."""
    # Synthetic clean training donor (ID: 0000450)
    # Past moving at 10 m/s: 21 points from x=-20.0 to 0.0 at 0.1s intervals
    past_pts = [[-20.0 + i * 1.0, 0.0] for i in range(21)]
    # Future straight: 25 points from x=2.0 to 50.0 at 0.2s intervals
    fut_pts = [[2.0 + i * 2.0, 0.0] for i in range(25)]

    sample_record = {
        "scenario_id": "0000450",
        "driving_instruction": "drive straight on",
        "scenario_type": "0 specifically selected",
        "past": past_pts,
        "expert_like": fut_pts,
    }
    jsonl_path = tmp_path / "valid_train.jsonl"
    with open(jsonl_path, "w") as f:
        f.write(json.dumps(sample_record) + "\n")

    # 1. Single Camera ("front")
    ds_front = dataset.SpatialConditionedWorldModelDataset(
        trajectories_file=jsonl_path,
        camera_name="front",
        split="train",
    )
    assert len(ds_front) == 1
    sample = ds_front[0]

    assert sample["scenario_id"] == "0000450"
    assert sample["gt_trajectory"].shape == (25, 2)
    assert sample["past_trajectory"].shape == (21, 2)
    assert np.isclose(sample["v0"], 10.0, atol=0.2)
    assert sample["motion_bucket"] == int(np.clip(10.0 * 12.0 + 10.0, 1, 255))

    # Option A: (1, 568, 876)
    assert sample["static_route_mask"].shape == (1, spec.HEIGHT, spec.WIDTH)
    assert sample["static_route_mask"].dtype == torch.float32

    # Option B: (25, 1, 568, 876)
    assert sample["time_indexed_masks"].shape == (25, 1, spec.HEIGHT, spec.WIDTH)
    assert sample["time_indexed_masks"].dtype == torch.float32

    # 2. Multi-camera ("all" 6 cameras)
    ds_all = dataset.SpatialConditionedWorldModelDataset(
        trajectories_file=jsonl_path,
        camera_name=None,
        split="train",
    )
    sample_all = ds_all[0]
    # Option A (6 cameras): (6, 1, 568, 876)
    assert sample_all["static_route_mask"].shape == (6, 1, spec.HEIGHT, spec.WIDTH)
    # Option B (6 cameras): (6, 25, 1, 568, 876)
    assert sample_all["future_camera_spatial_masks"].shape == (6, 25, 1, spec.HEIGHT, spec.WIDTH)
    assert sample_all["initial_frame_temporal_masks"].shape == (6, 25, 1, spec.HEIGHT, spec.WIDTH)


def test_future_video_loading_and_strict_assertion(tmp_path):
    """Verify future video target loading, shape verification, and failure on missing video."""
    # Create synthetic trajectory record
    sample_record = {
        "scenario_id": "0000405",
        "driving_instruction": "drive straight on",
        "scenario_type": "0 specifically selected",
        "past": [[-10.0 + i * 0.5, 0.0] for i in range(21)],
        "expert_like": [[float(i), 0.0] for i in range(1, 26)],
    }
    jsonl_path = tmp_path / "train_with_video.jsonl"
    with open(jsonl_path, "w") as f:
        f.write(json.dumps(sample_record) + "\n")

    video_dir = tmp_path / "future_videos"
    scene_dir = video_dir / "0000405" / "front"
    scene_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create 25 synthetic frame images
    for i in range(25):
        im = Image.new("RGB", (64, 64), color=(i * 10, 100, 200))
        im.save(scene_dir / f"frame_{i:02d}.jpg")

    # Load dataset with future_video_dir
    ds = dataset.SpatialConditionedWorldModelDataset(
        trajectories_file=jsonl_path,
        future_video_dir=video_dir,
        camera_name="front",
        split="train",
        require_future_video=True,
    )
    sample = ds[0]

    assert sample["has_future_video"] is True
    assert sample["future_video"] is not None
    assert sample["future_video"].shape == (25, 3, spec.HEIGHT, spec.WIDTH)
    assert sample["future_video"].dtype == torch.float32

    # 2. Strict assertion: when future video is missing and require_future_video=True -> raises FileNotFoundError
    ds_missing = dataset.SpatialConditionedWorldModelDataset(
        trajectories_file=jsonl_path,
        future_video_dir=tmp_path / "non_existent_videos",
        camera_name="front",
        split="train",
        require_future_video=True,
    )
    with pytest.raises(FileNotFoundError, match="Missing required future ground-truth video target"):
        _ = ds_missing[0]


def test_dataset_handles_variable_length_trajectories(tmp_path: Path):
    """Test that trajectories with fewer than 25 points (e.g. 18 points) are smoothly interpolated."""
    sample_record = {
        "scenario_id": "0000762",
        "driving_instruction": "turn left",
        "scenario_type": "4 intersection",
        "past": [[-10.0 + i * 0.5, 0.0] for i in range(21)],
        "expert_like": [[float(i), 0.0] for i in range(1, 19)],  # 18 points
    }
    jsonl_path = tmp_path / "short_traj.jsonl"
    with open(jsonl_path, "w") as f:
        f.write(json.dumps(sample_record) + "\n")

    ds = dataset.SpatialConditionedWorldModelDataset(
        trajectories_file=jsonl_path,
        camera_name="front",
        split="train",
    )
    sample = ds[0]
    assert sample["gt_trajectory"].shape == (25, 2)
    assert sample["initial_frame_temporal_masks"].shape == (25, 1, spec.HEIGHT, spec.WIDTH)
    assert sample["future_camera_spatial_masks"].shape == (25, 1, spec.HEIGHT, spec.WIDTH)


