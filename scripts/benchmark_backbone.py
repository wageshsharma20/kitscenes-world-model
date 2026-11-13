#!/usr/bin/env python3
"""Benchmark video generation throughput, latency, and VRAM footprint.

Runs a single scenario (6 camera views @ 25 frames, 5 fps, 876x568) through the selected
backbone, measures timing, and computes projected GPU hours for all 400 test scenarios (2,400 videos)
against Kaggle's 30-hour weekly quota.

Usage:
  python scripts/benchmark_backbone.py --backbone geometric
  python scripts/benchmark_backbone.py --backbone svd --device cuda
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitscenes import spec, video_engine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark video generation throughput and calculate total GPU hours.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backbone",
        choices=["geometric", "svd"],
        default="geometric",
        help="Model backbone to benchmark",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--cameras",
        type=int,
        default=6,
        help="Number of camera views to render for the benchmark scenario (1-6)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_renders"),
        help="Directory to store benchmark video files",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    print("=" * 65)
    print("KITScenes-LongTail World Model Throughput Benchmark")
    print(f"Backbone: {args.backbone} | Device: {args.device} | Cameras: {args.cameras}")
    print("=" * 65)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Trajectory creation for benchmark scenario
    waypoints = [[2.0 * (k + 1), 0.0] for k in range(spec.WAYPOINT_COUNT)]
    scenario_id = "0000000"

    # 2. VRAM tracking initialization
    has_cuda = False
    try:
        import torch
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            has_cuda = True
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    except ImportError:
        pass

    # 3. Model setup
    print("\n[*] Initializing pipeline...")
    init_start = time.perf_counter()
    try:
        pipeline = video_engine.WorldModelPipeline(
            backbone=args.backbone, device=args.device
        )
    except Exception as exc:
        print(f"[X] Pipeline initialization failed: {exc}", file=sys.stderr)
        return 1
    init_elapsed = time.perf_counter() - init_start
    print(f"[✓] Pipeline initialized in {init_elapsed:.2f}s")

    # 4. Scenario execution
    print(f"\n[*] Generating benchmark scenario ({args.cameras} camera view(s), 25 frames each)...")
    gen_start = time.perf_counter()
    try:
        # For SVD, if conditioning images are required, provide placeholder test images
        conditioning = None
        if args.backbone == "svd":
            import numpy as np
            conditioning = {
                cam: np.zeros((spec.HEIGHT, spec.WIDTH, 3), dtype=np.uint8)
                for cam in spec.CAMERAS
            }
            print("[!] SVD timing uses blank conditioning frames: latency is representative, "
                  "output quality is not", file=sys.stderr)

        encoded_paths = pipeline.generate_and_save_scenario(
            scenario_id=scenario_id,
            trajectory=waypoints,
            output_dir=args.output_dir,
            conditioning_images=conditioning,
        )
    except Exception as exc:
        print(f"[X] Generation failed: {exc}", file=sys.stderr)
        return 2

    if has_cuda:
        import torch
        torch.cuda.synchronize()
    gen_elapsed = time.perf_counter() - gen_start

    # 5. Throughput & capacity math
    num_videos = len(encoded_paths)
    time_per_video = gen_elapsed / num_videos if num_videos else 0.0
    time_per_full_scenario = time_per_video * 6.0  # 6 cameras
    total_projected_seconds = time_per_full_scenario * spec.SCENARIO_COUNT
    total_projected_hours = total_projected_seconds / 3600.0

    print("\n" + "-" * 65)
    print("BENCHMARK RESULTS")
    print("-" * 65)
    print(f"Wall-clock generation time ({num_videos} videos): {gen_elapsed:.2f}s")
    print(f"Average time per video (25 frames):       {time_per_video:.2f}s ({25.0 / time_per_video:.1f} fps generation speed)")
    print(f"Time per 6-camera scenario:               {time_per_full_scenario:.2f}s")

    if has_cuda:
        import torch
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak GPU VRAM allocated:                  {peak_vram_mb:.1f} MiB")

    print("\n" + "=" * 65)
    print("PROJECTED RUN FOR ALL 400 SCENARIOS (2,400 VIDEOS)")
    print("=" * 65)
    unit = "GPU-hours" if has_cuda else "CPU-hours (no accelerator in use)"
    print(f"Total projected compute time:             {total_projected_hours:.2f} {unit}")
    print("Kaggle weekly free quota:                 30.00 GPU-hours")

    if not has_cuda:
        print(
            "\n[!] Verdict withheld: this run used no GPU, so the number describes the "
            "procedural renderer, not a diffusion backbone. A CPU figure does not transfer "
            "to SVD; run --backbone svd --device cuda on the target hardware before planning "
            "a 2,400-video run."
        )
    elif total_projected_hours <= 15.0:
        print("\n[✓] FEASIBILITY: EXCELLENT. Easily fits within Kaggle's weekly quota with room for experiments.")
    elif total_projected_hours <= 28.0:
        print("\n[!] FEASIBILITY: TIGHT. Fits within Kaggle's 30h limit, but requires careful batching (9h session limit).")
    else:
        print("\n[X] FEASIBILITY: EXCEEDS KAGGLE QUOTA. Requires cloud GPU instances (A100 / H100 via RunPod/Lambda).")

    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
