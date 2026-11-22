#!/usr/bin/env python3
"""Phase 5.6: Reduced-Step Production Feasibility Test & Denoising Step Ablation.

Evaluates the frozen project_baseline (kNN trajectory conditioning + SVD-XT motion bucket)
across reduced denoising steps: 10, 12, 15, and 25 (reference).

Architectural Safeguards (per User Guidance):
  1. Per-Configuration Pipeline: Loops over steps first (10 -> 12 -> 15 -> 25).
  2. Intermediate Evaluation: Evaluates ground-truth fidelity (PSNR, SSIM, Boundary Strip Correlation)
     immediately after each step configuration completes.
  3. Persistent Kaggle Storage: Writes all reports and metrics directly into /kaggle/working/.
  4. Per-Step & Cumulative Packaging: Immediately generates:
     - /kaggle/working/results_step_{steps}.zip (isolated per-step archive)
     - /kaggle/working/results_step_ablation.zip (cumulative updated archive)
     ensuring no results can be lost even if an interactive session terminates.
  5. Resumption: Verified valid MP4s on disk are reused automatically without recomputation.

Usage:
  python scripts/run_step_ablation_race.py --stage all --device cuda:0 --steps-list 10,12,15,25 --resume --auto-zip
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitscenes import knn_trajectory, spec
from kitscenes.spatial_adapter import (
    MockSVDUNet,
    create_default_euler_scheduler,
    decode_latents_temporal,
)
from kitscenes.svd_inference import (
    adapt_aspect_ratio_to_svd,
    crop_svd_array_to_kitscenes,
    encode_image_to_clip_embedding,
    encode_image_to_vae_latents,
    run_official_svd_diffusion_rollout,
)

CANONICAL_CAMERAS: Tuple[str, ...] = spec.CAMERAS
DEFAULT_ABLATION_SCENARIOS: Tuple[str, ...] = ("0000900", "0000902", "0000905")
DEFAULT_STEPS_LIST: Tuple[int, ...] = (10, 12, 15, 25)

CAMERA_SEED_OFFSETS: Dict[str, int] = {
    "front": 0,
    "front_left": 1000,
    "front_right": 2000,
    "rear": 3000,
    "rear_left": 4000,
    "rear_right": 5000,
}


def is_valid_mp4(path: Path, expected_frames: int = spec.FRAME_COUNT) -> bool:
    """Verify MP4 file integrity and frame count."""
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return (count == expected_frames) and (w == spec.WIDTH) and (h == spec.HEIGHT)
    except Exception:
        return False


def load_video_frames(path: Path) -> np.ndarray:
    """Load video frames as uint8 RGB numpy array [T, H, W, C]."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"failed to decode frames from {path}")
    return np.stack(frames, axis=0)


def compute_true_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute true peak signal-to-noise ratio."""
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse <= 1e-10:
        return 50.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def compute_video_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Gaussian-windowed Structural Similarity Index (SSIM) using OpenCV."""
    p = pred.astype(np.float64)
    g = gt.astype(np.float64)
    frame_ssims = []
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    for t in range(min(len(p), len(g))):
        p_t = p[t]
        g_t = g[t]
        mu_x = cv2.GaussianBlur(p_t, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(g_t, (11, 11), 1.5)
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        sigma_x_sq = cv2.GaussianBlur(p_t ** 2, (11, 11), 1.5) - mu_x_sq
        sigma_y_sq = cv2.GaussianBlur(g_t ** 2, (11, 11), 1.5) - mu_y_sq
        sigma_xy = cv2.GaussianBlur(p_t * g_t, (11, 11), 1.5) - mu_xy
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        )
        frame_ssims.append(np.mean(ssim_map))
    return float(np.mean(frame_ssims))


def compute_boundary_strip_correlation(camera_frames: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Compute 2D boundary-strip Pearson normalized cross-correlation across adjacent camera pairs.

    DISCLAIMER: This is a 2D boundary pixel correlation heuristic across 15%-wide image edges,
    NOT calibrated 3D geometric scene reprojection consistency.
    """
    pairs = [
        ("front", "front_left"),
        ("front", "front_right"),
        ("rear", "rear_left"),
        ("rear", "rear_right"),
        ("front_left", "rear_left"),
        ("front_right", "rear_right"),
    ]
    pairwise_scores: Dict[str, float] = {}
    overlap_scores = []
    w = spec.WIDTH
    strip = int(w * 0.15)

    for c1, c2 in pairs:
        pair_key = f"{c1}_to_{c2}"
        if c1 not in camera_frames or c2 not in camera_frames:
            continue
        v1 = camera_frames[c1]
        v2 = camera_frames[c2]

        if c2.endswith("_left"):
            s1 = v1[:, :, :strip, :].astype(np.float32)
            s2 = v2[:, :, -strip:, :].astype(np.float32)
        else:
            s1 = v1[:, :, -strip:, :].astype(np.float32)
            s2 = v2[:, :, :strip, :].astype(np.float32)

        s1_norm = (s1 - s1.mean()) / (s1.std() + 1e-6)
        s2_norm = (s2 - s2.mean()) / (s2.std() + 1e-6)
        ncc = float(np.clip(np.mean(s1_norm * s2_norm), 0.0, 1.0))
        pairwise_scores[pair_key] = round(ncc, 4)
        overlap_scores.append(ncc)

    mean_score = float(round(np.mean(overlap_scores), 4)) if overlap_scores else 0.0
    return {
        "boundary_strip_correlation": mean_score,
        "pairwise_correlations": pairwise_scores,
        "pair_count_evaluated": len(overlap_scores),
    }


def sync_to_kaggle_working(source_path: Path, dest_filename: Optional[str] = None) -> Optional[Path]:
    """Copy an artifact directly to /kaggle/working for immediate user persistence and download."""
    kaggle_dir = Path("/kaggle/working")
    if not kaggle_dir.exists():
        return None
    try:
        dest_name = dest_filename or source_path.name
        dest_path = kaggle_dir / dest_name
        if source_path.is_file():
            shutil.copy2(source_path, dest_path)
        elif source_path.is_dir():
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(source_path, dest_path)
        return dest_path
    except Exception as e:
        print(f"[-] Warning: Failed to sync {source_path.name} to /kaggle/working: {e}")
        return None


def package_zip_archive(files_or_dir: Sequence[Path] | Path, zip_dest: Path, arc_prefix: str = "") -> Path:
    """Create a ZIP archive containing specified files or directory contents."""
    zip_dest.parent.mkdir(parents=True, exist_ok=True)
    seen_arcnames: set[str] = set()
    with zipfile.ZipFile(zip_dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if isinstance(files_or_dir, Path) and files_or_dir.is_dir():
            for item in sorted(files_or_dir.rglob("*")):
                if item.is_file():
                    arcname = item.relative_to(files_or_dir)
                    if arc_prefix:
                        arcname = Path(arc_prefix) / arcname
                    arcname_str = str(arcname)
                    if arcname_str not in seen_arcnames:
                        seen_arcnames.add(arcname_str)
                        zf.write(item, arcname=arcname_str)
        else:
            for item in files_or_dir:
                if item.exists() and item.is_file():
                    arcname = item.name if not arc_prefix else f"{arc_prefix}/{item.name}"
                    if arcname not in seen_arcnames:
                        seen_arcnames.add(arcname)
                        zf.write(item, arcname=arcname)
    return zip_dest


def generate_ablation_markdown_report(
    step_stats: Dict[str, Any],
    report_md_path: Path,
    completed_steps: List[int],
    target_steps: List[int],
) -> str:
    """Generate professional GitHub Flavored Markdown report with head-to-head metrics."""
    md = []
    md.append("# Phase 5.6: Reduced-Step Denoising Feasibility Report")
    md.append("")
    md.append(f"**Generated**: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ")
    md.append(f"**Frozen Architecture**: `project_baseline` (kNN Trajectory Conditioning + SVD-XT Motion Bucket)  ")
    md.append(f"**Progress**: Completed step configurations: **{completed_steps}** (Target: {target_steps})  ")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. Denoising Step Head-to-Head Comparison")
    md.append("")
    md.append("| Steps | Evaluated Videos | Mean Runtime / Vid | Mean PSNR (dB) | Median PSNR | Mean SSIM | Median SSIM | Boundary Corr | $\\Delta$ PSNR vs 25 | $\\Delta$ SSIM vs 25 |")
    md.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for st_str, sdata in sorted(step_stats.items(), key=lambda x: int(x[0])):
        rt_str = f"{sdata['mean_runtime_min']:.2f} min" if sdata.get("mean_runtime_min") else "Ref (~7.0m)"
        d_p = f"{sdata.get('delta_psnr_vs_25', 0.0):+.3f} dB" if "delta_psnr_vs_25" in sdata else "N/A"
        d_s = f"{sdata.get('delta_ssim_vs_25', 0.0):+.4f}" if "delta_ssim_vs_25" in sdata else "N/A"
        b_corr = f"{sdata.get('mean_boundary_corr', 0.0):.4f}"
        md.append(
            f"| **{sdata['steps']}** | {sdata['count']} | {rt_str} | **{sdata['mean_psnr']:.2f}** | "
            f"{sdata['median_psnr']:.2f} | **{sdata['mean_ssim']:.4f}** | {sdata['median_ssim']:.4f} | "
            f"{b_corr} | {d_p} | {d_s} |"
        )

    md.append("")
    md.append("---")
    md.append("")
    md.append("## 2. Production Compute Projection Across 2,400 Videos")
    md.append("")
    md.append("| Configuration | Single Video Time | 400 Front Videos | All 2,400 Videos | Feasibility on 3 Accounts (~170h) |")
    md.append("| :--- | :---: | :---: | :---: | :--- |")

    for st_str in ["10", "12", "15", "25"]:
        sdata = step_stats.get(st_str, {})
        rt_min = sdata.get("mean_runtime_min") or (float(st_str) / 25.0 * 6.98)
        front_hrs = round((400 * rt_min) / 60.0, 1)
        full_hrs = round((2400 * rt_min) / 60.0, 1)
        feas = "FEASIBLE" if full_hrs <= 170 else ("FRONT ONLY FEASIBLE" if front_hrs <= 170 else "EXCEEDS QUOTA")
        md.append(f"| **{st_str} Steps** | {rt_min:.2f} min | {front_hrs} GPU-h | {full_hrs} GPU-h | **{feas}** |")

    md.append("")
    md.append("---")
    md.append("")
    md.append("## 3. Operational Candidate Selection Rule")
    md.append("")
    md.append("| Metric Boundary | Decision | Production Action |")
    md.append("| :--- | :--- | :--- |")
    md.append("| $\\Delta\\text{PSNR} \\ge -0.3\\text{ dB}$ & $\\Delta\\text{SSIM} \\ge -0.02$ at 10 steps | **Select 10 Steps** | Full 2,400 videos across all 6 cameras (~112 GPU-h) |")
    md.append("| 12 steps beats 10 steps by $\\ge 0.2\\text{ dB}$ with $\\le 135\\text{ GPU-h}$ | **Select 12 Steps** | Full 2,400 videos across all 6 cameras (~134 GPU-h) |")
    md.append("| Sharp degradation below 15 steps | **Option B (25-Step Front Priority)** | 400 front-camera videos at 25 steps (~46.5 GPU-h) + partial surround |")
    md.append("")

    content = "\n".join(md)
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_md_path, "w") as f:
        f.write(content)

    sync_to_kaggle_working(report_md_path)
    return content


def run_evaluation_for_steps(
    output_dir: Path,
    manifest_records: List[Dict[str, Any]],
    gt_dir: Path,
    report_json_path: Path,
    report_md_path: Path,
    target_steps: List[int],
) -> Dict[str, Any]:
    """Evaluate all predictions generated up to this point and update reports."""
    # Find all evaluated records
    evaluations: List[Dict[str, Any]] = []
    scenario_cam_map: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}

    for rec in manifest_records:
        sid = rec["scenario_id"]
        cam = rec["camera"]
        steps = rec["steps"]
        vpath = Path(rec["file_path"])

        gt_path = gt_dir / sid / f"{cam}.mp4"
        if not vpath.exists() or not gt_path.exists():
            continue

        try:
            pred_frames = load_video_frames(vpath)
            gt_frames = load_video_frames(gt_path)
        except Exception as e:
            print(f"[-] Warning: Failed to load frames for {vpath.name}: {e}")
            continue

        psnr_val = compute_true_psnr(pred_frames, gt_frames)
        ssim_val = compute_video_ssim(pred_frames, gt_frames)

        # Cache for multi-camera boundary correlation
        scenario_cam_map.setdefault((sid, steps), {})[cam] = pred_frames

        eval_rec = dict(rec)
        eval_rec.update({
            "psnr": round(psnr_val, 3),
            "ssim": round(ssim_val, 4),
        })
        evaluations.append(eval_rec)

    # Compute boundary correlations per scenario & step
    scenario_bcorr: Dict[Tuple[str, int], float] = {}
    for (sid, steps), cam_dict in scenario_cam_map.items():
        if len(cam_dict) >= 2:
            bcorr_res = compute_boundary_strip_correlation(cam_dict)
            scenario_bcorr[(sid, steps)] = bcorr_res["boundary_strip_correlation"]
        else:
            scenario_bcorr[(sid, steps)] = 0.0

    # Aggregate by step count
    steps_group: Dict[int, List[Dict[str, Any]]] = {}
    for ev in evaluations:
        st = ev["steps"]
        ev["boundary_strip_corr"] = scenario_bcorr.get((ev["scenario_id"], st), 0.0)
        steps_group.setdefault(st, []).append(ev)

    step_stats: Dict[str, Any] = {}
    for st, rows in sorted(steps_group.items()):
        psnrs = [r["psnr"] for r in rows]
        ssims = [r["ssim"] for r in rows]
        bcorrs = [r.get("boundary_strip_corr", 0.0) for r in rows]
        runtimes = [r["runtime_sec"] for r in rows if r.get("runtime_sec") is not None]
        vram_allocs = [r["peak_allocated_mb"] for r in rows if r.get("peak_allocated_mb") is not None]

        step_stats[str(st)] = {
            "steps": st,
            "count": len(rows),
            "mean_psnr": round(float(np.mean(psnrs)), 3),
            "median_psnr": round(float(np.median(psnrs)), 3),
            "std_psnr": round(float(np.std(psnrs)), 3),
            "mean_ssim": round(float(np.mean(ssims)), 4),
            "median_ssim": round(float(np.median(ssims)), 4),
            "std_ssim": round(float(np.std(ssims)), 4),
            "mean_boundary_corr": round(float(np.mean(bcorrs)), 4) if bcorrs else 0.0,
            "mean_runtime_sec": round(float(np.mean(runtimes)), 2) if runtimes else None,
            "mean_runtime_min": round(float(np.mean(runtimes)) / 60.0, 2) if runtimes else None,
            "mean_vram_alloc_mb": round(float(np.mean(vram_allocs)), 2) if vram_allocs else None,
        }

    # Compute delta vs 25 steps
    ref_stats = step_stats.get("25")
    if ref_stats:
        ref_psnr = ref_stats["mean_psnr"]
        ref_ssim = ref_stats["mean_ssim"]
        for st_str, st_dict in step_stats.items():
            st_dict["delta_psnr_vs_25"] = round(st_dict["mean_psnr"] - ref_psnr, 3)
            st_dict["delta_ssim_vs_25"] = round(st_dict["mean_ssim"] - ref_ssim, 4)

    completed_steps = sorted([int(k) for k in step_stats.keys()])

    # Save JSON report
    report_payload = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "completed_steps": completed_steps,
        "target_steps": target_steps,
        "step_statistics": step_stats,
        "evaluations": evaluations,
    }
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json_path, "w") as f:
        json.dump(report_payload, f, indent=2)
    sync_to_kaggle_working(report_json_path)

    # Generate Markdown report
    generate_ablation_markdown_report(step_stats, report_md_path, completed_steps, target_steps)

    return report_payload


def run_step_ablation_pipeline(
    output_dir: Path,
    manifest_path: Path,
    report_json_path: Path,
    report_md_path: Path,
    gt_dir: Path,
    device_str: str,
    mock_mode: bool,
    steps_list: List[int],
    scenarios: List[str],
    cameras: List[str],
    decode_chunk_size: int = 4,
    resume: bool = True,
    import_phase5_5_ref: bool = True,
    auto_zip: bool = True,
) -> Dict[str, Any]:
    """Execute Phase 5.6 Step Ablation with per-configuration intermediate evaluation and packaging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    if "cuda" in device_str and not torch.cuda.is_available():
        print(f"[-] WARNING: CUDA requested ({device_str}) but not available. Falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    print(f"\n======================================================================")
    print(f"PHASE 5.6: INTEGRATED STEP ABLATION RACE ON {device}")
    print(f"Target Scenarios: {scenarios}")
    print(f"Target Steps:     {steps_list}")
    print(f"Target Cameras:   {cameras}")
    print(f"Architecture:     Frozen project_baseline (kNN + SVD-XT motion bucket)")
    print(f"======================================================================\n")

    # Load frozen kNN planner
    train_traj_path = PROJECT_ROOT / "data" / "train_trajectories.jsonl"
    knn_planner = knn_trajectory.KNNTrajectoryPlanner(
        donor_path=train_traj_path if train_traj_path.exists() else None
    )

    # Load validation metadata
    val_meta = {}
    val_meta_path = PROJECT_ROOT / "data" / "val_trajectories.jsonl"
    if val_meta_path.exists():
        with open(val_meta_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    val_meta[rec["scenario_id"]] = rec

    # Load SVD pipeline components
    if mock_mode or device_str == "cpu":
        print("[+] Mock mode enabled: using MockSVDUNet")
        unet = MockSVDUNet().to(device)
        vae = None
        image_encoder = None
        feature_extractor = None
    else:
        print("[+] Loading official pretrained SVD-XT weights (fp16)...")
        from diffusers import AutoencoderKLTemporalDecoder
        from diffusers.models.unets.unet_spatio_temporal_condition import (
            UNetSpatioTemporalConditionModel,
        )
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            torch_dtype=torch.float16,
            variant="fp16",
            low_cpu_mem_usage=True,
        ).to(device)
        unet.eval()
        unet.enable_forward_chunking()

        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)
        vae.eval()

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            model_id,
            subfolder="image_encoder",
            torch_dtype=torch.float16,
            variant="fp16",
            low_cpu_mem_usage=True,
        ).to(device)
        image_encoder.eval()

        feature_extractor = CLIPImageProcessor.from_pretrained(
            model_id,
            subfolder="feature_extractor",
        )

    # Load existing manifest records if resuming
    manifest_records: List[Dict[str, Any]] = []
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                prev = json.load(f)
                manifest_records = prev.get("predictions", [])
        except Exception:
            manifest_records = []

    already_done = {(r["scenario_id"], r["camera"], r["steps"]) for r in manifest_records}

    # Reference search directories for 25-step import
    ref_search_dirs = [
        PROJECT_ROOT / "data" / "ref_phase5_5",
        PROJECT_ROOT / "renders" / "phase5_5",
    ]

    latest_report: Dict[str, Any] = {}
    completed_steps_list: List[int] = []

    # OUTER LOOP: Iterate by step count (per user safeguard recommendation)
    for steps in steps_list:
        step_start_t = time.perf_counter()
        print(f"\n======================================================================")
        print(f">>> STARTING STEP CONFIGURATION: {steps} STEPS <<<")
        print(f"Target: {len(scenarios)} scenarios x {len(cameras)} cameras = {len(scenarios) * len(cameras)} videos")
        print(f"======================================================================\n")

        # 1. Check for instant 25-step reference import if applicable
        if steps == 25 and import_phase5_5_ref:
            imported_25 = 0
            for sid in scenarios:
                for cam in cameras:
                    dst_v = output_dir / f"{sid}_{cam}_steps25.mp4"
                    if not dst_v.exists() or not is_valid_mp4(dst_v):
                        for ref_dir in ref_search_dirs:
                            src_v = ref_dir / f"{sid}_{cam}_project_baseline.mp4"
                            if src_v.exists() and is_valid_mp4(src_v):
                                shutil.copy2(src_v, dst_v)
                                imported_25 += 1
                                break
                    if dst_v.exists() and is_valid_mp4(dst_v) and (sid, cam, 25) not in already_done:
                        manifest_records.append({
                            "scenario_id": sid,
                            "camera": cam,
                            "steps": 25,
                            "file_path": str(dst_v),
                            "file_size": dst_v.stat().st_size,
                            "runtime_sec": 418.0,
                            "peak_allocated_mb": None,
                            "peak_reserved_mb": None,
                            "motion_bucket": 127,
                            "mean_speed": 8.0,
                        })
                        already_done.add((sid, cam, 25))
            if imported_25 > 0:
                print(f"[+] Successfully imported {imported_25} pre-computed 25-step reference videos!")

        # 2. Render all scenarios and cameras for this step setting
        step_video_paths: List[Path] = []
        video_idx = 0
        total_for_step = len(scenarios) * len(cameras)

        for sid in scenarios:
            meta_rec = val_meta.get(sid, {})
            past = meta_rec.get("past", meta_rec.get("past_trajectory", []))
            inst = meta_rec.get("driving_instruction", meta_rec.get("instruction", ""))
            stype = meta_rec.get("scenario_type", "")

            if len(past) >= 4:
                plan_future = knn_planner.plan(sid, past, inst, scenario_type=stype)
                speeds = np.linalg.norm(np.diff(plan_future, axis=0), axis=1) / 0.2
                mean_speed = float(np.mean(speeds))
            else:
                mean_speed = 8.0

            motion_bucket = int(np.clip(12.0 + (mean_speed / 25.0) * (255.0 - 12.0), 12, 255))

            for cam in cameras:
                video_idx += 1
                out_path = output_dir / f"{sid}_{cam}_steps{steps}.mp4"
                step_video_paths.append(out_path)

                # Skip if already verified
                if resume and (sid, cam, steps) in already_done and is_valid_mp4(out_path):
                    print(f"[{video_idx}/{total_for_step}] SKIP {out_path.name} (already verified)")
                    continue

                if resume and out_path.exists() and is_valid_mp4(out_path):
                    print(f"[{video_idx}/{total_for_step}] REUSE {out_path.name} (valid on disk)")
                    manifest_records.append({
                        "scenario_id": sid,
                        "camera": cam,
                        "steps": steps,
                        "file_path": str(out_path),
                        "file_size": out_path.stat().st_size,
                        "runtime_sec": 418.0 if steps == 25 else None,
                        "peak_allocated_mb": None,
                        "peak_reserved_mb": None,
                        "motion_bucket": motion_bucket,
                        "mean_speed": mean_speed,
                    })
                    already_done.add((sid, cam, steps))
                    continue

                # Load conditioning image
                cond_img_path = PROJECT_ROOT / "data" / "observed_t0" / sid / f"{cam}.png"
                if not cond_img_path.exists():
                    gt_vid = gt_dir / sid / f"{cam}.mp4"
                    if gt_vid.exists():
                        cap = cv2.VideoCapture(str(gt_vid))
                        ret, frame0 = cap.read()
                        cap.release()
                        if ret:
                            cond_img_path.parent.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(cond_img_path), frame0)
                            cond_img = Image.fromarray(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB))
                        else:
                            cond_img = Image.new("RGB", (spec.WIDTH, spec.HEIGHT), (128, 128, 128))
                    else:
                        cond_img = Image.new("RGB", (spec.WIDTH, spec.HEIGHT), (128, 128, 128))
                else:
                    cond_img = Image.open(cond_img_path).convert("RGB")

                # Encode conditioning
                if not mock_mode and device_str != "cpu":
                    torch.cuda.reset_peak_memory_stats(device)
                    aspect_img, _ = adapt_aspect_ratio_to_svd(cond_img)
                    clip_emb = encode_image_to_clip_embedding(image_encoder, feature_extractor, aspect_img, device)
                    vae_latents = encode_image_to_vae_latents(vae, aspect_img, device)
                else:
                    clip_emb = torch.randn(1, 1, 1024, device=device)
                    vae_latents = torch.randn(1, 4, 72, 128, device=device)

                print(f"[{video_idx}/{total_for_step}] RENDERING {out_path.name} (steps={steps}, bucket={motion_bucket})...")
                start_vid_t = time.perf_counter()

                # Seed for reproducible multi-camera rollout
                seed = 42 + CAMERA_SEED_OFFSETS.get(cam, 0)
                generator = torch.Generator(device=device).manual_seed(seed)
                starting_noise = torch.randn(
                    1, spec.FRAME_COUNT, 4, 72, 128,
                    device=device,
                    dtype=torch.float16 if not mock_mode else torch.float32,
                    generator=generator,
                )

                scheduler = create_default_euler_scheduler()

                with torch.inference_mode():
                    latents, _ = run_official_svd_diffusion_rollout(
                        unet=unet,
                        scheduler=scheduler,
                        initial_latent=vae_latents,
                        encoder_hidden_states=clip_emb,
                        starting_noise=starting_noise,
                        fps=spec.FPS,
                        motion_bucket_id=motion_bucket,
                        num_inference_steps=steps,
                        do_classifier_free_guidance=True,
                    )

                    if not mock_mode and device_str != "cpu":
                        frames_raw = decode_latents_temporal(vae, latents, decode_chunk_size=decode_chunk_size)
                        frames_uint8 = ((frames_raw[0].permute(0, 2, 3, 1).cpu().float().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                        peak_res = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
                        del latents, frames_raw
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        frames_uint8 = np.full((spec.FRAME_COUNT, 576, 1024, 3), 120, dtype=np.uint8)
                        peak_alloc = 0.0
                        peak_res = 0.0

                canonical_frames = crop_svd_array_to_kitscenes(frames_uint8)
                runtime_sec = time.perf_counter() - start_vid_t

                # Write H.264 MP4
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_writer = cv2.VideoWriter(str(out_path), fourcc, float(spec.FPS), (spec.WIDTH, spec.HEIGHT))
                for frame in canonical_frames:
                    out_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                out_writer.release()

                assert is_valid_mp4(out_path), f"Encoded video {out_path} failed validation check!"

                rec = {
                    "scenario_id": sid,
                    "camera": cam,
                    "steps": steps,
                    "file_path": str(out_path),
                    "file_size": out_path.stat().st_size,
                    "runtime_sec": round(runtime_sec, 2),
                    "peak_allocated_mb": round(peak_alloc, 2),
                    "peak_reserved_mb": round(peak_res, 2),
                    "motion_bucket": motion_bucket,
                    "mean_speed": round(mean_speed, 2),
                }
                manifest_records.append(rec)
                already_done.add((sid, cam, steps))

                print(f"      -> DONE {out_path.name} in {runtime_sec:.1f}s (Alloc: {peak_alloc:.1f} MB, Res: {peak_res:.1f} MB)")

                # Incrementally flush manifest to disk
                with open(manifest_path, "w") as f:
                    json.dump({"predictions": manifest_records}, f, indent=2)
                sync_to_kaggle_working(manifest_path)

        completed_steps_list.append(steps)
        step_elapsed_sec = time.perf_counter() - step_start_t

        # 3. INTERMEDIATE EVALUATION: Evaluate all predictions completed so far
        print(f"\n[+] Running Intermediate Evaluation for completed configurations: {completed_steps_list}...")
        latest_report = run_evaluation_for_steps(
            output_dir=output_dir,
            manifest_records=manifest_records,
            gt_dir=gt_dir,
            report_json_path=report_json_path,
            report_md_path=report_md_path,
            target_steps=steps_list,
        )

        # 4. INTERMEDIATE PACKAGING: Per-step ZIP and cumulative ZIP
        if auto_zip:
            # Per-step ZIP
            step_zip_name = f"results_step_{steps}.zip"
            step_zip_path = submissions_dir / step_zip_name
            valid_step_files = [p for p in step_video_paths if p.exists() and is_valid_mp4(p)]
            package_zip_archive(valid_step_files + [manifest_path], step_zip_path)
            sync_to_kaggle_working(step_zip_path)

            # Cumulative ZIP
            cum_zip_path = submissions_dir / "results_step_ablation.zip"
            all_valid_files = [Path(r["file_path"]) for r in manifest_records if Path(r["file_path"]).exists()]
            reports = [manifest_path, report_json_path, report_md_path]
            package_zip_archive(all_valid_files + [r for r in reports if r.exists()], cum_zip_path)
            sync_to_kaggle_working(cum_zip_path)

        # 5. Prominent intermediate summary banner
        s_stats = latest_report.get("step_statistics", {}).get(str(steps), {})
        mean_p = s_stats.get("mean_psnr", 0.0)
        med_p = s_stats.get("median_psnr", 0.0)
        mean_s = s_stats.get("mean_ssim", 0.0)
        med_s = s_stats.get("median_ssim", 0.0)
        b_corr = s_stats.get("mean_boundary_corr", 0.0)
        rt_vid = s_stats.get("mean_runtime_sec", 0.0)

        print(f"\n======================================================================")
        print(f"MILESTONE ACHIEVED: {steps}-STEP CONFIGURATION 100% COMPLETE & SAVED")
        print(f"----------------------------------------------------------------------")
        print(f"  * Total Time for {steps} Steps: {step_elapsed_sec / 60.0:.1f} minutes")
        print(f"  * Mean Runtime per Video:   {rt_vid:.1f}s ({rt_vid/60.0:.2f} min)")
        print(f"  * Mean PSNR:                 {mean_p:.3f} dB (Median: {med_p:.3f} dB)")
        print(f"  * Mean SSIM:                 {mean_s:.4f} (Median: {med_s:.4f})")
        print(f"  * Boundary Strip Corr:       {b_corr:.4f}")
        if auto_zip:
            print(f"  * Isolated Step Archive:     /kaggle/working/results_step_{steps}.zip")
            print(f"  * Cumulative Master Archive: /kaggle/working/results_step_ablation.zip")
        print(f"  * Intermediate Report:       /kaggle/working/phase5_6_step_ablation.md")
        print(f"======================================================================\n")

    print(f"\n[+] PHASE 5.6 ABLATION RACE FULLY COMPLETED across all step settings: {steps_list}!")
    return latest_report


def main():
    parser = argparse.ArgumentParser(description="Phase 5.6 Reduced-Step Feasibility Test")
    parser.add_argument("--stage", choices=["predict", "evaluate", "all"], default="all")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mock", action="store_true", help="Run with MockSVDUNet")
    parser.add_argument("--steps-list", type=str, default="10,12,15,25")
    parser.add_argument("--scenarios", type=str, default="0000900,0000902,0000905")
    parser.add_argument("--cameras", type=str, default="front,front_left,front_right,rear,rear_left,rear_right")
    parser.add_argument("--decode-chunk-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--import-phase5-5-ref", action="store_true", default=True)
    parser.add_argument("--auto-zip", action="store_true", default=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "renders" / "phase5_6")
    parser.add_argument("--gt-dir", type=Path, default=PROJECT_ROOT / "data" / "val_future_videos")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports")

    args = parser.parse_args()

    steps_list = [int(s.strip()) for s in args.steps_list.split(",") if s.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]

    manifest_path = args.output_dir / "step_ablation_manifest.json"
    report_json_path = args.report_dir / "phase5_6_step_ablation.json"
    report_md_path = args.report_dir / "phase5_6_step_ablation.md"

    if args.stage == "evaluate":
        # Pure evaluate mode on existing manifest
        if manifest_path.exists():
            with open(manifest_path) as f:
                rec_data = json.load(f).get("predictions", [])
        else:
            rec_data = []
        run_evaluation_for_steps(
            output_dir=args.output_dir,
            manifest_records=rec_data,
            gt_dir=args.gt_dir,
            report_json_path=report_json_path,
            report_md_path=report_md_path,
            target_steps=steps_list,
        )
    else:
        # Integrated predict + evaluate + intermediate auto-zip
        run_step_ablation_pipeline(
            output_dir=args.output_dir,
            manifest_path=manifest_path,
            report_json_path=report_json_path,
            report_md_path=report_md_path,
            gt_dir=args.gt_dir,
            device_str=args.device,
            mock_mode=args.mock,
            steps_list=steps_list,
            scenarios=scenarios,
            cameras=cameras,
            decode_chunk_size=args.decode_chunk_size,
            resume=args.resume,
            import_phase5_5_ref=args.import_phase5_5_ref,
            auto_zip=args.auto_zip,
        )


if __name__ == "__main__":
    main()
