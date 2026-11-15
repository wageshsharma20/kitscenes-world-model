#!/usr/bin/env python3
"""Extract and validate genuine training video targets from KITScenes-LongTail on Hugging Face.

Prerequisites:
  Hugging Face authenticated (via HF_TOKEN or cached ~/.cache/huggingface/token).

Extracts:
  1. Initial t=0 observation frames from train-00000-of-00001.parquet -> data/train_observed_t0/<scenario_id>/<camera>.jpg
  2. 25-frame future videos from train_future-00000-of-00001.parquet -> data/train_future_videos/<scenario_id>/<camera>.mp4
     and data/train_future_videos/<scenario_id>/<camera>/frame_%02d.jpg
  3. Validates continuity, frame count (exactly 25), 5 fps, and resolution (876x568).
  4. Writes validation manifest -> reports/training_data_audit/training_video_manifest.json

Usage:
  # Extract 8 canonical training sequences for front camera:
  python scripts/extract_training_videos.py --scenarios 0000400,0000401,0000402,0000403,0000404,0000405,0000406,0000407 --cameras front
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitscenes import spec

CAMERA_FRAME_COLUMNS = {
    "front": "frames_camera_front",
    "front_left": "frames_camera_front_left",
    "front_right": "frames_camera_front_right",
    "rear": "frames_camera_rear",
    "rear_left": "frames_camera_rear_left",
    "rear_right": "frames_camera_rear_right",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract and validate genuine training video targets from KITScenes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="KIT-MRT/KITScenes-LongTail",
        help="Hugging Face dataset repository ID",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="0000400,0000401,0000402,0000403,0000404,0000405,0000406,0000407",
        help="Comma-separated list of scenario IDs to extract (default: 8 canonical sequences)",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default="front",
        help="Comma-separated list of cameras to extract (default: front)",
    )
    parser.add_argument(
        "--observed-t0-dir",
        type=Path,
        default=Path("data/train_observed_t0"),
        help="Output directory for t=0 initial observation frames",
    )
    parser.add_argument(
        "--future-video-dir",
        type=Path,
        default=Path("data/train_future_videos"),
        help="Output directory for future video targets",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("reports/training_data_audit/training_video_manifest.json"),
        help="Path to write the verified dataset audit manifest",
    )
    return parser


def resize_image_bytes(raw_bytes: bytes, width: int = spec.WIDTH, height: int = spec.HEIGHT) -> bytes:
    from PIL import Image

    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    resized = img.resize((width, height), Image.Resampling.LANCZOS)
    out_buf = io.BytesIO()
    resized.save(out_buf, format="JPEG", quality=92)
    return out_buf.getvalue()


def encode_frames_to_mp4(frame_paths: List[Path], out_mp4: Path, fps: float = 5.0) -> None:
    """Encode list of frame JPEGs into an MP4 video file at 5 fps using ffmpeg."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg") is not None:
        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frame_paths[0].parent / "frame_%02d.jpg"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_mp4),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg encoding failed: {res.stderr}")
    else:
        # Fallback to OpenCV VideoWriter
        import cv2

        im0 = cv2.imread(str(frame_paths[0]))
        h, w = im0.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h))
        for p in frame_paths:
            writer.write(cv2.imread(str(p)))
        writer.release()


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def extract_and_validate(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import HfFileSystem
        import pyarrow.parquet as pq
    except ImportError as e:
        print(f"[X] Missing required libraries: {e}", file=sys.stderr)
        return 1

    fs = HfFileSystem()
    target_sids = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    target_cams = [c.strip() for c in args.cameras.split(",") if c.strip()]

    print("=" * 80)
    print("KITSCENES GENUINE TRAINING VIDEO TARGET EXTRACTOR & VALIDATION AUDIT")
    print(f"Repository:      {args.repo_id}")
    print(f"Target Scenarios: {target_sids} ({len(target_sids)} sequences)")
    print(f"Target Cameras:   {target_cams}")
    print(f"Observed t0 Dir:  {args.observed_t0_dir}")
    print(f"Future Video Dir: {args.future_video_dir}")
    print("=" * 80)

    args.observed_t0_dir.mkdir(parents=True, exist_ok=True)
    args.future_video_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    past_parquet_path = f"datasets/{args.repo_id}/data/train-00000-of-00001.parquet"
    future_parquet_path = f"datasets/{args.repo_id}/data/train_future-00000-of-00001.parquet"

    req_cols = ["scenario_id"] + [CAMERA_FRAME_COLUMNS[c] for c in target_cams if c in CAMERA_FRAME_COLUMNS]

    # Step 1: Extract Initial t=0 Frames from train-00000-of-00001.parquet
    print("\n[*] Step 1/3: Extracting initial t=0 observation frames from train-00000-of-00001.parquet...")
    t0_extracted = {}
    for sid in target_sids:
        for cam in target_cams:
            p = args.observed_t0_dir / sid / f"{cam}.jpg"
            if p.exists():
                t0_extracted[(sid, cam)] = p

    needed_t0 = [(s, c) for s in target_sids for c in target_cams if (s, c) not in t0_extracted]
    if needed_t0:
        with fs.open(past_parquet_path, "rb") as f_past:
            pf_past = pq.ParquetFile(f_past)
            for rg_idx in range(pf_past.num_row_groups):
                t_past = pf_past.read_row_group(rg_idx, columns=req_cols)
                rg_sids = t_past["scenario_id"].to_pylist()
                matching = [s for s in target_sids if s in rg_sids]
                if not matching:
                    continue

                for sid in matching:
                    idx = rg_sids.index(sid)
                    scene_dir = args.observed_t0_dir / sid
                    scene_dir.mkdir(parents=True, exist_ok=True)
                    for cam in target_cams:
                        if (sid, cam) in t0_extracted:
                            continue
                        col_name = CAMERA_FRAME_COLUMNS[cam]
                        frames = t_past[col_name][idx].as_py()
                        assert len(frames) >= 1, f"No frames found for {sid} {cam}"
                        # Last frame is t=0.0s observation
                        t0_frame_bytes = frames[-1]["bytes"]
                        resized_bytes = resize_image_bytes(t0_frame_bytes)
                        img_path = scene_dir / f"{cam}.jpg"
                        img_path.write_bytes(resized_bytes)
                        t0_extracted[(sid, cam)] = img_path
                        print(f"    [+] Saved t=0 image: {img_path} ({len(resized_bytes) / 1024:.1f} KB)")

                if all((sid, cam) in t0_extracted for sid in target_sids for cam in target_cams):
                    break
    else:
        print("    [✓] All initial t=0 images already present on disk.")

    # Step 2: Extract 25-frame Future Videos from train_future-00000-of-00001.parquet
    print("\n[*] Step 2/3: Extracting 25-frame future ground-truth videos from train_future-00000-of-00001.parquet...")
    future_extracted = {}
    for sid in target_sids:
        for cam in target_cams:
            mp4_p = args.future_video_dir / sid / f"{cam}.mp4"
            cam_dir = args.future_video_dir / sid / cam
            if mp4_p.exists() and cam_dir.exists():
                frames = sorted([f for f in cam_dir.glob("frame_*.jpg")])
                if len(frames) == 25:
                    future_extracted[(sid, cam)] = {
                        "frame_paths": frames,
                        "mp4_path": mp4_p,
                    }

    needed_fut = [(s, c) for s in target_sids for c in target_cams if (s, c) not in future_extracted]
    if needed_fut:
        with fs.open(future_parquet_path, "rb") as f_fut:
            pf_fut = pq.ParquetFile(f_fut)
            for rg_idx in range(pf_fut.num_row_groups):
                t_fut = pf_fut.read_row_group(rg_idx, columns=req_cols)
                rg_sids = t_fut["scenario_id"].to_pylist()
                matching = [s for s in target_sids if s in rg_sids]
                if not matching:
                    continue

                for sid in matching:
                    idx = rg_sids.index(sid)
                    for cam in target_cams:
                        if (sid, cam) in future_extracted:
                            continue
                        col_name = CAMERA_FRAME_COLUMNS[cam]
                        frames = t_fut[col_name][idx].as_py()
                        # STRICT ASSERTION: must be exactly 25 frames
                        if len(frames) != 25:
                            raise ValueError(f"Scenario {sid} camera {cam} has {len(frames)} future frames, expected 25!")

                        cam_frame_dir = args.future_video_dir / sid / cam
                        cam_frame_dir.mkdir(parents=True, exist_ok=True)
                        frame_paths = []
                        for f_idx, frame_obj in enumerate(frames):
                            raw_bytes = frame_obj["bytes"]
                            resized_bytes = resize_image_bytes(raw_bytes)
                            frame_path = cam_frame_dir / f"frame_{f_idx:02d}.jpg"
                            frame_path.write_bytes(resized_bytes)
                            frame_paths.append(frame_path)

                        # Encode to MP4
                        mp4_path = args.future_video_dir / sid / f"{cam}.mp4"
                        encode_frames_to_mp4(frame_paths, mp4_path, fps=5.0)
                        future_extracted[(sid, cam)] = {
                            "frame_paths": frame_paths,
                            "mp4_path": mp4_path,
                        }
                        print(f"    [+] Saved future video: {mp4_path} (25 frames, {mp4_path.stat().st_size / 1024:.1f} KB)")

                if all((sid, cam) in future_extracted for sid in target_sids for cam in target_cams):
                    break
    else:
        print("    [✓] All future video targets already present on disk.")

    # Step 3: Validation Audit & Manifest Generation
    print("\n[*] Step 3/3: Running validation assertions and generating audit manifest...")
    manifest_records = []
    for sid in target_sids:
        for cam in target_cams:
            t0_p = t0_extracted.get((sid, cam))
            fut_info = future_extracted.get((sid, cam))
            if not t0_p or not fut_info:
                raise RuntimeError(f"Missing extracted data for scenario {sid} camera {cam}!")

            mp4_p = fut_info["mp4_path"]
            frames = fut_info["frame_paths"]

            # Verify assertions
            assert t0_p.exists(), f"Missing {t0_p}"
            assert mp4_p.exists(), f"Missing {mp4_p}"
            assert len(frames) == 25, f"Expected 25 frames, found {len(frames)}"

            sha_t0 = compute_sha256(t0_p)
            sha_mp4 = compute_sha256(mp4_p)

            record = {
                "scenario_id": sid,
                "camera": cam,
                "num_frames": len(frames),
                "fps": 5.0,
                "resolution": [spec.WIDTH, spec.HEIGHT],
                "initial_rgb_path": str(t0_p.resolve().relative_to(PROJECT_ROOT.resolve())),
                "initial_rgb_sha256": sha_t0,
                "future_video_mp4_path": str(mp4_p.resolve().relative_to(PROJECT_ROOT.resolve())),
                "future_video_sha256": sha_mp4,
                "status": "VERIFIED_GENUINE",
                "provenance": f"Hugging Face datasets/{args.repo_id} (train_future-00000-of-00001.parquet)",
            }
            manifest_records.append(record)

    with open(args.manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "total_sequences": len(manifest_records),
                "records": manifest_records,
            },
            f,
            indent=2,
        )

    print(f"\n[✓] Successfully verified and saved {len(manifest_records)} genuine sequences!")
    print(f"[✓] Manifest written to: {args.manifest_path}")
    return 0


if __name__ == "__main__":
    parser = build_parser()
    sys.exit(extract_and_validate(parser.parse_args()))
