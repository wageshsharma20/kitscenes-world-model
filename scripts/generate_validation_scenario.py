#!/usr/bin/env python3
"""Generate a complete 6-camera validation scenario using PhotographicWarpModel and evaluate motion & PQ."""

import json
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitscenes import emc_eval, media, pq_proxy, spec, video_engine


def main():
    scenario_id = "0000900"
    val_t0_dir = Path("data/val_observed_t0") / scenario_id
    val_traj_file = Path("data/val_trajectories.jsonl")

    if not val_t0_dir.exists():
        print(f"Error: {val_t0_dir} does not exist", file=sys.stderr)
        return 1

    # Read trajectory
    trajectory = None
    with open(val_traj_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["scenario_id"] == scenario_id:
                trajectory = rec["expert_like"]
                break

    if trajectory is None:
        print(f"Error: scenario {scenario_id} not found in {val_traj_file}", file=sys.stderr)
        return 1

    print("=" * 70)
    print(f"Generating 6-Camera Validation Scenario {scenario_id} (PhotographicWarp)")
    print(f"Conditioning Source: {val_t0_dir} (Genuine Validation Split)")
    print(f"Trajectory Length:   {len(trajectory)} waypoints (5.0s at 5 Hz)")
    print(f"First Waypoint:      {trajectory[0]} (t=0.2s)")
    print(f"Final Waypoint:      {trajectory[-1]} (t=5.0s)")
    print("=" * 70)

    # 1. Load conditioning images
    cond_images = video_engine.load_conditioning_images(val_t0_dir.parent, scenario_id)

    # 2. PhotographicWarp Generation
    out_dir_warp = Path("renders/validation_photo_warp") / scenario_id
    out_dir_warp.mkdir(parents=True, exist_ok=True)

    pipeline_warp = video_engine.WorldModelPipeline(backbone="photo_warp")
    t0 = time.time()
    warp_paths = pipeline_warp.generate_and_save_scenario(
        scenario_id=scenario_id,
        trajectory=trajectory,
        output_dir=out_dir_warp.parent,
        conditioning_images=cond_images,
    )
    warp_time = time.time() - t0
    print(f"\n[+] Rendered & encoded 6 photo_warp videos in {warp_time:.2f}s:")
    for cam, p in warp_paths.items():
        info = media.probe_video(p)
        print(f"    - {cam:12s}: {p.name} ({p.stat().st_size / 1024:.1f} KB, "
              f"{info.width}x{info.height}, {info.frame_count} frames, {info.fps:.1f} fps)")

    # 3. Static Hold Generation
    out_dir_hold = Path("renders/validation_static_hold") / scenario_id
    out_dir_hold.mkdir(parents=True, exist_ok=True)

    encoder = video_engine.VideoEncoder(crf=24)
    hold_paths = {}
    for cam in spec.CAMERAS:
        raw_frame = video_engine._decode_frame(cond_images[cam])
        held_frames = np.repeat(raw_frame[None, ...], spec.FRAME_COUNT, axis=0)
        p = out_dir_hold / f"{cam}.mp4"
        encoder.encode(held_frames, p, validate=True)
        hold_paths[cam] = p

    # 4. Optical Flow & Motion Evaluation (EMC)
    print("\n" + "=" * 70)
    print("EGO-MOTION CONSISTENCY (EMC) & OPTICAL FLOW EVALUATION")
    print("=" * 70)

    warp_cam_paths = {cam: out_dir_warp / f"{cam}.mp4" for cam in spec.CAMERAS}
    hold_cam_paths = {cam: out_dir_hold / f"{cam}.mp4" for cam in spec.CAMERAS}

    emc_warp = emc_eval.compute_video_optical_flow_emc(warp_cam_paths, trajectory)
    emc_hold = emc_eval.compute_video_optical_flow_emc(hold_cam_paths, trajectory)

    print("\n--- PhotographicWarpModel (0000900) ---")
    print(f"Overall EMC Score:        {emc_warp['emc']:.3f} (matches: {emc_warp['matches']}/4)")
    print(f"Segment 1 Speed Ratio:    {emc_warp['seg1_speed_ratio']:.3f} (accel match: {emc_warp['seg1_accel_match']})")
    print(f"Segment 2 Speed Ratio:    {emc_warp['seg2_speed_ratio']:.3f} (accel match: {emc_warp['seg2_accel_match']})")
    print(f"Segment 1 Steering Match: {emc_warp['seg1_steering_match']}")
    print(f"Segment 2 Steering Match: {emc_warp['seg2_steering_match']}")

    print("\n--- Static Hold Baseline (0000900) ---")
    print(f"Overall EMC Score:        {emc_hold['emc']:.3f} (matches: {emc_hold['matches']}/4)")
    print(f"Segment 1 Speed Ratio:    {emc_hold['seg1_speed_ratio']:.3f}")
    print(f"Segment 2 Speed Ratio:    {emc_hold['seg2_speed_ratio']:.3f}")

    # 5. PQ Proxy Evaluation
    print("\n" + "=" * 70)
    print("PERCEPTUAL QUALITY (PQ) PROXY COMPARISON")
    print("=" * 70)

    pq_warp = pq_proxy.score_scenario_videos(warp_cam_paths)
    pq_hold = pq_proxy.score_scenario_videos(hold_cam_paths)

    print(f"PhotographicWarpModel PQ: {pq_warp['scenario_pq']:.2f} (Front: {pq_warp['front']:.2f})")
    print(f"Static Hold Baseline PQ:  {pq_hold['scenario_pq']:.2f} (Front: {pq_hold['front']:.2f})")

    report_path = Path("reports/validation_scenario_0000900_eval.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "scenario_id": scenario_id,
        "split": "validation",
        "warp": {
            "emc": emc_warp,
            "pq": pq_warp,
            "generation_time_s": warp_time,
            "video_paths": {cam: str(p) for cam, p in warp_paths.items()},
        },
        "static_hold": {
            "emc": emc_hold,
            "pq": pq_hold,
            "video_paths": {cam: str(p) for cam, p in hold_paths.items()},
        },
    }
    report_path.write_text(json.dumps(report_data, indent=2))
    print(f"\nSaved evaluation report into: {report_path}")
    return 0


if __name__ == "__main__":
    import numpy as np
    sys.exit(main())
