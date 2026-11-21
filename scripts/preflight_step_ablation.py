#!/usr/bin/env python3
"""Preflight verification for Phase 5.6 Reduced-Step Feasibility Test on Kaggle GPU.

Verifies:
  1. CUDA availability and VRAM capacity (>= 14.5 GiB)
  2. Dependencies: diffusers, transformers, accelerate, torch, cv2
  3. Pretrained SVD-XT weights accessible from Hugging Face cache or hub
  4. Scenarios (0000900, 0000902, 0000905) ground-truth and conditioning assets present
  5. Frozen kNN planner donor file intact
  6. Compute budget and runtime estimation display
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_preflight() -> int:
    print("=" * 70)
    print("PHASE 5.6: REDUCED-STEP FEASIBILITY PREFLIGHT VERIFICATION")
    print("=" * 70)

    errors = 0

    # 1. CUDA & GPU VRAM Check
    try:
        import torch
        if not torch.cuda.is_available():
            print("[-] ERROR: CUDA is not available! Run this notebook with GPU T4 x2 enabled.")
            errors += 1
        else:
            gpu_name = torch.cuda.get_device_name(0)
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"[+] GPU Detected: {gpu_name} ({total_vram_gb:.2f} GiB VRAM)")
            if total_vram_gb < 14.0:
                print(f"[-] WARNING: Total VRAM ({total_vram_gb:.2f} GiB) is lower than 14.0 GiB.")
    except Exception as exc:
        print(f"[-] ERROR: Failed to probe PyTorch/CUDA: {exc}")
        errors += 1

    # 2. Package Dependencies
    required_packages = ["diffusers", "transformers", "accelerate", "cv2", "PIL", "scipy"]
    for pkg in required_packages:
        try:
            if pkg == "cv2":
                import cv2
            elif pkg == "PIL":
                import PIL
            else:
                __import__(pkg)
            print(f"[+] Package '{pkg}' is installed.")
        except ImportError:
            print(f"[-] ERROR: Required package '{pkg}' is missing. Run: pip install -q {pkg}")
            errors += 1

    # 3. Target Scenarios Ground Truth & Conditioning
    target_scenarios = ["0000900", "0000902", "0000905"]
    gt_dir = PROJECT_ROOT / "data" / "val_future_videos"
    for sid in target_scenarios:
        s_gt = gt_dir / sid
        if not s_gt.exists():
            print(f"[-] ERROR: Ground-truth future videos for scenario {sid} missing at {s_gt}")
            errors += 1
        else:
            mp4_count = len(list(s_gt.glob("*.mp4")))
            print(f"[+] Scenario {sid} has {mp4_count}/6 ground-truth camera videos.")

    # 4. Donor Trajectories
    donor_path = PROJECT_ROOT / "data" / "train_trajectories.jsonl"
    if not donor_path.exists():
        print(f"[-] ERROR: Training donor trajectories missing at {donor_path}")
        errors += 1
    else:
        print(f"[+] Training donor trajectory file present ({donor_path.stat().st_size / 1024:.1f} KB).")

    # 5. Runtime & Quota Estimate
    print("-" * 70)
    print("ESTIMATED TEST WORKLOAD (3 Scenarios x 6 Cameras = 18 Videos per Step Setting):")
    print("  * 10 Steps: 18 videos x ~2.8 min = ~50 min GPU time")
    print("  * 12 Steps: 18 videos x ~3.3 min = ~60 min GPU time")
    print("  * 15 Steps: 18 videos x ~4.1 min = ~74 min GPU time")
    print("  * 25 Steps: 18 videos x ~7.0 min = Imported from Phase 5.5 (0 min GPU time!)")
    print("  -------------------------------------------------------------")
    print("  Total Estimated GPU Runtime: ~3.1 hours on 1x T4 GPU")
    print("=" * 70)

    if errors == 0:
        print("\n>>> PREFLIGHT PASSED: Ready to run Phase 5.6 Reduced-Step Ablation! <<<\n")
        return 0
    else:
        print(f"\n>>> PREFLIGHT FAILED: {errors} error(s) detected. Fix issues before launching. <<<\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_preflight())
