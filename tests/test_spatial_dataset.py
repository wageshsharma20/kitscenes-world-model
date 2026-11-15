"""Unit tests for SpatialConditionedWorldModelDataset and spatial conditioning tensors."""

from __future__ import annotations

import tempfile
import json
from pathlib import Path
import numpy as np
import pytest
import torch

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
    assert sample_all["time_indexed_masks"].shape == (6, 25, 1, spec.HEIGHT, spec.WIDTH)
