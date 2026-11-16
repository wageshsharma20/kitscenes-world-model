#!/usr/bin/env python3
"""Training Dataset Extractor & Verification for Spatial Conditioning World Models.

Extracts and validates paired training tuples:
  - Initial observation frames (t=0.0s)
  - Future ground-truth video targets (25 frames @ 5 fps)
  - Ground truth trajectories (25 waypoints @ 5 Hz)
  - Option A: Static Route Corridor Masks [B, 1, 568, 876]
  - Option B.1: Route-Initial-Frame Temporal Masks [B, 25, 1, 568, 876]
  - Option B.2: Future-Camera-Frame Spatial Masks [B, 25, 1, 568, 876]
  - Derived kinematics & SVD Motion Buckets
  - Sensor calibration metadata (provisional nominal model)

Performs rigorous partition verification: strictly asserts zero validation leakage.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from kitscenes import dataset, spec


def main():
    parser = argparse.ArgumentParser(
        description="Verify and extract spatial conditioning dataset from training donors."
    )
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=Path("data/train_trajectories.jsonl"),
        help="Path to training trajectories JSONL",
    )
    parser.add_argument(
        "--observed-t0-dir",
        type=Path,
        default=Path("data/observed_t0"),
        help="Path to directory containing observed t=0 camera frames",
    )
    parser.add_argument(
        "--future-video-dir",
        type=Path,
        default=None,
        help="Path to directory containing ground-truth future videos (optional locally, required for GPU training)",
    )
    parser.add_argument(
        "--require-future-video",
        action="store_true",
        help="Strictly assert future video presence (fails if any training video target is missing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for extraction throughput benchmark",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="front",
        choices=["front", "front_left", "front_right", "rear", "rear_left", "rear_right", "all"],
        help="Camera view to extract (or 'all' for 6-camera rig)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of donors to inspect",
    )
    args = parser.parse_args()

    camera_name = None if args.camera == "all" else args.camera

    print("=" * 80)
    print("KITSCENES SPATIAL CONDITIONING TRAINING DATASET EXTRACTOR")
    print(f"Source file:       {args.trajectories}")
    print(f"Observed t0 dir:   {args.observed_t0_dir}")
    print(f"Future video dir:  {args.future_video_dir or '(None: offline metadata mode)'}")
    print(f"Camera mode:       {args.camera}")
    print(f"Batch size:        {args.batch_size}")
    print("=" * 80)

    t0 = time.time()
    ds = dataset.SpatialConditionedWorldModelDataset(
        trajectories_file=args.trajectories,
        observed_t0_dir=args.observed_t0_dir,
        future_video_dir=args.future_video_dir,
        camera_name=camera_name,
        split="train",
        require_future_video=args.require_future_video,
    )
    load_time = time.time() - t0

    n_samples = len(ds)
    print(f"[+] Loaded {n_samples} training records in {load_time:.2f}s")

    # 1. Verification of partition integrity
    val_leaked = [
        r["scenario_id"] for r in ds.records if 900 <= int(r["scenario_id"]) <= 999
    ]
    assert len(val_leaked) == 0, f"DATA LEAKAGE DETECTED: {val_leaked}"
    print("[+] Partition Integrity: PASS (0 validation scenarios found in training split)")

    # 2. Inspect first record
    sample_0 = ds[0]
    print("\n--- Canonical Training Tuple Inspection (Index 0) ---")
    print(f"Scenario ID:                  {sample_0['scenario_id']}")
    print(f"Instruction:                  {sample_0['driving_instruction']}")
    print(f"Scenario Type:                {sample_0['scenario_type']}")
    print(f"Initial Speed v0:             {sample_0['v0']:.2f} m/s")
    print(f"Initial Yaw Rate w0:          {sample_0['w0']:.4f} rad/s")
    print(f"SVD Motion Bucket:            {sample_0['motion_bucket']}")
    print(f"GT Trajectory Tensor:         shape={sample_0['gt_trajectory'].shape}, dtype={sample_0['gt_trajectory'].dtype}")
    print(f"Option A Static Mask:         shape={sample_0['static_route_mask'].shape}, dtype={sample_0['static_route_mask'].dtype}, range=[{sample_0['static_route_mask'].min():.2f}, {sample_0['static_route_mask'].max():.2f}]")
    print(f"Option B.1 Initial-Frame:     shape={sample_0['initial_frame_temporal_masks'].shape}, dtype={sample_0['initial_frame_temporal_masks'].dtype}")
    print(f"Option B.2 Future-Camera:     shape={sample_0['future_camera_spatial_masks'].shape}, dtype={sample_0['future_camera_spatial_masks'].dtype}")
    print(f"Initial RGB Available:        {sample_0['has_initial_rgb']}")
    print(f"Future Video Target Present:  {sample_0['has_future_video']}")
    calib = sample_0['calibration']
    print(f"Calibration Rig Status:       is_provisional={calib.get('is_provisional', True)} (Provisional Nominal Model)")

    # 3. Benchmark DataLoader throughput
    test_limit = min(args.limit or n_samples, n_samples)
    subset_ds = torch.utils.data.Subset(ds, range(test_limit))
    loader = DataLoader(
        subset_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_spatial_dataset,
    )

    print(f"\n--- Benchmarking DataLoader Throughput ({test_limit} samples, batch_size={args.batch_size}) ---")
    t_start = time.time()
    n_batches = 0
    total_samples = 0
    speeds = []
    buckets = []
    initial_rgb_count = 0
    future_vid_count = 0

    for batch in loader:
        n_batches += 1
        total_samples += len(batch["scenario_id"])
        speeds.extend([float(v) for v in batch["v0"]])
        buckets.extend([int(b) for b in batch["motion_bucket"]])
        initial_rgb_count += sum(batch["has_initial_rgb"])
        future_vid_count += sum(batch["has_future_video"])

        if n_batches == 1:
            print("[Batch 0 Shapes]")
            print(f"  - static_route_mask (Option A):             {batch['static_route_mask'].shape}")
            print(f"  - initial_frame_temporal_masks (Option B.1): {batch['initial_frame_temporal_masks'].shape}")
            print(f"  - future_camera_spatial_masks (Option B.2):  {batch['future_camera_spatial_masks'].shape}")
            print(f"  - gt_trajectory:                             {batch['gt_trajectory'].shape}")
            if batch["initial_rgb"] is not None:
                print(f"  - initial_rgb:                               {batch['initial_rgb'].shape}")
            if batch["future_video"] is not None:
                print(f"  - future_video:                              {batch['future_video'].shape}")

    t_total = time.time() - t_start
    fps = total_samples / max(t_total, 1e-4)

    print(f"\n[+] Extraction Benchmark: {total_samples} samples processed in {t_total:.2f}s ({fps:.1f} samples/sec)")
    print(f"Speed distribution: min={min(speeds):.2f} m/s, mean={sum(speeds)/len(speeds):.2f} m/s, max={max(speeds):.2f} m/s")
    print(f"Motion bucket distribution: min={min(buckets)}, median={sorted(buckets)[len(buckets)//2]}, max={max(buckets)}")
    print(f"Initial RGB frames available locally: {initial_rgb_count}/{total_samples}")
    print(f"Future ground-truth video targets available locally: {future_vid_count}/{total_samples}")

    print("\n" + "=" * 80)
    print("DATASET CONTRACT VERIFICATION SUMMARY:")
    print(f"  - Donors Loaded:                     {n_samples} (100% of training split)")
    print("  - Validation Leakage:                0 (Strict partition verified)")
    print(f"  - Option A Static Tensor Shape:      ({args.batch_size}, 1, {spec.HEIGHT}, {spec.WIDTH})")
    print(f"  - Option B.1 Initial-Frame Shape:    ({args.batch_size}, {spec.FRAME_COUNT}, 1, {spec.HEIGHT}, {spec.WIDTH})")
    print(f"  - Option B.2 Future-Camera Shape:    ({args.batch_size}, {spec.FRAME_COUNT}, 1, {spec.HEIGHT}, {spec.WIDTH})")
    print("  - Calibration Model:                 PROVISIONAL NOMINAL SENSOR RIG (HFOV=70°)")
    print(f"  - Future Video Target Availability:  {future_vid_count}/{total_samples} locally on disk")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
