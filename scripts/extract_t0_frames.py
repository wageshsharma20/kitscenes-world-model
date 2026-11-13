#!/usr/bin/env python3
"""Extract observed t=0 conditioning frames for all 6 camera views from the KITScenes test split.

In KITScenes-LongTail, past camera observations run from t = -4.0s to 0.0s (21 frames at 5 Hz).
The frame at index -1 (t = 0.0s) is the exact observation conditioning image required by
generative multi-view world models (Stable Video Diffusion, Vista, Drive-WM) to generate
physically consistent future driving videos.

Supports two operational modes:
1. Local Parquet: Reads directly from local parquet file (fastest, ideal for Kaggle).
2. Remote Streaming: Streams over HfFileSystem via HTTP range requests without downloading
   the full 32 GB parquet file.

Outputs:
  <output_dir>/<scenario_id>/<camera_name>.jpg (downscaled to 876x568 by default)

Usage:
  # On Kaggle (with local parquet file):
  python scripts/extract_t0_frames.py --parquet data/test-00000-of-00001.parquet

  # Over remote stream for 2 scenarios:
  python scripts/extract_t0_frames.py --token hf_... --limit 2
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

# Ensure project root is in sys.path
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
        description="Extract observed t=0 camera frames from KITScenes-LongTail dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        help="Path to local parquet file (e.g. data/test-00000-of-00001.parquet)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default="",
        help="Hugging Face access token (defaults to HF_TOKEN env var)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="KIT-MRT/KITScenes-LongTail",
        help="Hugging Face dataset repository ID",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/observed_t0"),
        help="Directory to save extracted t=0 camera images",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        help="Comma-separated list of scenario IDs to extract (e.g. 0000000,0000001)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of scenarios to extract",
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        default=True,
        help="Downscale frames from native 3504x2272 to challenge resolution 876x568",
    )
    parser.add_argument(
        "--no-resize",
        action="store_false",
        dest="resize",
        help="Keep raw native resolution (3504x2272)",
    )
    return parser


def resize_image_bytes(
    raw_bytes: bytes, width: int = spec.WIDTH, height: int = spec.HEIGHT
) -> bytes:
    """Resize JPEG/PNG image bytes to (width, height) using Pillow or ffmpeg fallback."""
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        resized = img.resize((width, height), Image.Resampling.LANCZOS)
        out_buf = io.BytesIO()
        resized.save(out_buf, format="JPEG", quality=92)
        return out_buf.getvalue()
    except ImportError:
        # Fallback to ffmpeg
        if not shutil.which("ffmpeg"):
            return raw_bytes
        res = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "-i",
                "-",
                "-vf",
                f"scale={width}:{height}",
                "-q:v",
                "2",
                "-f",
                "image2",
                "-",
            ],
            input=raw_bytes,
            capture_output=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout:
            return res.stdout
        return raw_bytes


def extract_scenario_t0(
    row: dict[str, Any],
    output_dir: Path,
    resize: bool = True,
) -> dict[str, Path]:
    """Extract and save t=0 frames for all 6 cameras for a single row dict."""
    scenario_id = str(row.get("scenario_id", "")).zfill(7)
    scenario_dir = output_dir / scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, Path] = {}

    for cam_name, col_name in CAMERA_FRAME_COLUMNS.items():
        frames_list = row.get(col_name)
        if not frames_list or not isinstance(frames_list, list):
            continue

        # Last frame in past sequence is t=0.0s
        t0_entry = frames_list[-1]
        img_bytes = (
            t0_entry.get("bytes") if isinstance(t0_entry, dict) else getattr(t0_entry, "bytes", b"")
        )

        if not img_bytes:
            continue

        if resize:
            img_bytes = resize_image_bytes(img_bytes)

        out_path = scenario_dir / f"{cam_name}.jpg"
        out_path.write_bytes(img_bytes)
        saved_paths[cam_name] = out_path

    return saved_paths


def process_parquet_file(
    parquet_source: Any,
    output_dir: Path,
    target_scenarios: set[str] | None = None,
    limit: int | None = None,
    resize: bool = True,
) -> dict[str, dict[str, Path]]:
    """Process a ParquetFile instance (local or remote fsspec stream) and extract t=0 frames."""
    import pyarrow.parquet as pq

    if isinstance(parquet_source, (str, Path)):
        pf = pq.ParquetFile(str(parquet_source))
    else:
        pf = pq.ParquetFile(parquet_source)

    all_extracted: dict[str, dict[str, Path]] = {}
    total_rgs = pf.num_row_groups

    print(f"[*] Processing {total_rgs} row group(s)")

    for rg_idx in range(total_rgs):
        if limit and len(all_extracted) >= limit:
            break

        # 1. Read scenario IDs in this row group first
        sid_table = pf.read_row_group(rg_idx, columns=["scenario_id"])
        sids = [str(r["scenario_id"]).zfill(7) for r in sid_table.to_pylist()]

        needed_indices = []
        for i, sid in enumerate(sids):
            if target_scenarios and sid not in target_scenarios:
                continue
            if limit and (len(all_extracted) + len(needed_indices)) >= limit:
                break
            needed_indices.append(i)

        if not needed_indices:
            continue

        for i in needed_indices:
            sid = sids[i]
            if sid not in all_extracted:
                all_extracted[sid] = {}
                (output_dir / sid).mkdir(parents=True, exist_ok=True)

        # 2. Read each camera column sequentially (contiguous byte range)
        for cam_name, col_name in CAMERA_FRAME_COLUMNS.items():
            if col_name not in pf.schema_arrow.names:
                continue
            cam_table = pf.read_row_group(rg_idx, columns=[col_name])
            cam_rows = cam_table.to_pylist()
            for i in needed_indices:
                sid = sids[i]
                frames_list = cam_rows[i].get(col_name)
                if not frames_list or not isinstance(frames_list, list):
                    continue
                t0_entry = frames_list[-1]
                img_bytes = (
                    t0_entry.get("bytes")
                    if isinstance(t0_entry, dict)
                    else getattr(t0_entry, "bytes", b"")
                )
                if not img_bytes:
                    continue
                if resize:
                    img_bytes = resize_image_bytes(img_bytes)
                out_path = output_dir / sid / f"{cam_name}.jpg"
                out_path.write_bytes(img_bytes)
                all_extracted[sid][cam_name] = out_path

        print(
            f"    Progress: [{rg_idx + 1:2d}/{total_rgs:2d}] row groups | "
            f"{len(all_extracted)} scenario(s) extracted",
            flush=True,
        )

    return all_extracted


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_scenarios = None
    if args.scenarios:
        target_scenarios = {s.strip().zfill(7) for s in args.scenarios.split(",") if s.strip()}

    print("=" * 65)
    print("KITScenes-LongTail Observed t=0 Frame Extraction")
    print(f"Output Directory: {args.output_dir}")
    print(f"Resolution:       {'876x568 (downscaled)' if args.resize else '3504x2272 (native)'}")
    print("=" * 65)

    start_time = time.time()

    # Mode 1: Local Parquet File
    if args.parquet and Path(args.parquet).exists():
        print(f"[*] Reading from local Parquet file: {args.parquet}")
        results = process_parquet_file(
            parquet_source=args.parquet,
            output_dir=args.output_dir,
            target_scenarios=target_scenarios,
            limit=args.limit,
            resize=args.resize,
        )
    else:
        # Mode 2: Remote Streaming
        token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            print(
                "[X] Error: HF token is required for remote streaming access.",
                file=sys.stderr,
            )
            print("    Pass --token hf_... or set HF_TOKEN environment variable.", file=sys.stderr)
            print("    Alternatively, provide a local file via --parquet <path>.", file=sys.stderr)
            return 1

        try:
            from huggingface_hub import HfFileSystem
        except ImportError:
            print("[X] huggingface_hub is required for streaming.", file=sys.stderr)
            return 2

        fs = HfFileSystem(token=token)
        remote_uri = f"datasets/{args.repo_id}/data/{args.split}-00000-of-00001.parquet"
        print(f"[*] Streaming remote Parquet via range requests: {remote_uri}")

        with fs.open(remote_uri, "rb") as remote_file:
            results = process_parquet_file(
                parquet_source=remote_file,
                output_dir=args.output_dir,
                target_scenarios=target_scenarios,
                limit=args.limit,
                resize=args.resize,
            )

    elapsed = time.time() - start_time
    total_imgs = sum(len(cams) for cams in results.values())
    print("\n" + "=" * 65)
    print("EXTRACTION COMPLETE")
    print("=" * 65)
    print(f"Extracted {len(results)} scenarios ({total_imgs} camera frames) in {elapsed:.1f}s.")
    print(f"Saved into: {args.output_dir.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
