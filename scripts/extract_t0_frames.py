#!/usr/bin/env python3
"""Extract observed t=0 conditioning frames for camera views from the KITScenes test split.

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

  # Over remote stream for single front camera on scenario 0000000:
  python scripts/extract_t0_frames.py --token hf_... --scenarios 0000000 --cameras front
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
        "--cameras",
        type=str,
        default=None,
        help="Comma-separated list of cameras to extract (e.g. front or front,rear; default: all 6 cameras)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of scenarios to extract",
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        dest="resize",
        help="downscale frames from native 3504x2272 to challenge resolution 876x568 (default)",
    )
    parser.add_argument(
        "--no-resize",
        action="store_false",
        dest="resize",
        help="keep the native 3504x2272 resolution; without this flag frames are "
        "downscaled to the 876x568 contract size",
    )
    parser.set_defaults(resize=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-extract scenarios that already have images on disk",
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
                "-i",
                "pipe:0",
                "-vf",
                f"scale={width}:{height}:flags=lanczos",
                "-f",
                "image2",
                "-q:v",
                "2",
                "pipe:1",
            ],
            input=raw_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if res.returncode == 0 and res.stdout:
            return res.stdout
        return raw_bytes


def extract_scenario_t0(
    row: dict[str, Any],
    output_dir: Path,
    target_cameras: set[str] | None = None,
    resize: bool = True,
) -> dict[str, Path]:
    """Extract and save t=0 frames for requested cameras for a single row dict."""
    scenario_id = str(row.get("scenario_id", "")).zfill(7)
    scenario_dir = output_dir / scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, Path] = {}
    cams_to_process = target_cameras if target_cameras else set(CAMERA_FRAME_COLUMNS.keys())

    for cam_name in cams_to_process:
        col_name = CAMERA_FRAME_COLUMNS.get(cam_name)
        if not col_name:
            continue
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
    target_cameras: set[str] | None = None,
    limit: int | None = None,
    resize: bool = True,
    force: bool = False,
    return_stats: bool = False,
) -> dict[str, dict[str, Path]] | tuple[dict[str, dict[str, Path]], dict[str, Any]]:
    """Process a ParquetFile instance (local or remote fsspec stream) and extract t=0 frames.

    Reading a column chunk transfers every frame of that row group, not just the last one.
    To minimize network transfer:
    - We only read columns for requested cameras (e.g. 'frames_camera_front' when --cameras front).
    - We skip row groups that contain no target scenarios.
    - We stop scanning row groups early once all target scenarios have been extracted.
    - We log precise compressed byte sizes of the transferred column chunks.
    """
    import pyarrow.parquet as pq

    if isinstance(parquet_source, (str, Path)):
        pf = pq.ParquetFile(str(parquet_source))
    else:
        pf = pq.ParquetFile(parquet_source)

    all_extracted: dict[str, dict[str, Path]] = {}
    total_rgs = pf.num_row_groups

    cams_to_extract = target_cameras if target_cameras else set(CAMERA_FRAME_COLUMNS.keys())
    selected_cols = [
        col_name
        for cam_name, col_name in CAMERA_FRAME_COLUMNS.items()
        if cam_name in cams_to_extract and col_name in pf.schema_arrow.names
    ]

    if not selected_cols:
        raise RuntimeError(
            f"No camera frame columns found matching requested cameras {cams_to_extract} in parquet schema. "
            f"Available: {pf.schema_arrow.names}"
        )

    # Parquet Row-Group Transfer Accounting
    transfer_stats: dict[str, Any] = {
        "total_row_groups": total_rgs,
        "row_groups_scanned": 0,
        "matching_row_groups": [],
        "scenarios_in_matching_groups": [],
        "compressed_bytes_by_column": {col: 0 for col in ["scenario_id"] + selected_cols},
        "estimated_compressed_column_chunk_bytes": 0,
    }

    print(f"[*] Processing parquet with {total_rgs} row group(s)")
    print(f"[*] Selected camera column(s) to read: {selected_cols}")

    for rg_idx in range(total_rgs):
        if limit and len(all_extracted) >= limit:
            break

        transfer_stats["row_groups_scanned"] += 1
        rg_meta = pf.metadata.row_group(rg_idx)

        # Fast pre-filtering: read only scenario_id column to check if target scenarios exist
        if target_scenarios:
            # Account for the scenario_id column chunk read to inspect this row group
            scen_chunk_bytes = sum(
                rg_meta.column(i).total_compressed_size
                for i in range(rg_meta.num_columns)
                if rg_meta.column(i).path_in_schema == "scenario_id"
                or rg_meta.column(i).path_in_schema.startswith("scenario_id.")
            )
            transfer_stats["compressed_bytes_by_column"]["scenario_id"] += scen_chunk_bytes
            transfer_stats["estimated_compressed_column_chunk_bytes"] += scen_chunk_bytes

            rg_meta_table = pf.read_row_group(rg_idx, columns=["scenario_id"])
            rg_sids_list = [str(s).zfill(7) for s in rg_meta_table["scenario_id"].to_pylist()]
            rg_sids = set(rg_sids_list)
            matching_sids = rg_sids.intersection(target_scenarios)

            # If no targets in this row group, or all already extracted, skip entire row group
            if not matching_sids or (
                not force
                and all(
                    all((output_dir / sid / f"{cam}.jpg").exists() for cam in cams_to_extract)
                    for sid in matching_sids
                )
            ):
                continue

            # Found matching row group: record transfer statistics for the camera columns read
            transfer_stats["matching_row_groups"].append(rg_idx)
            transfer_stats["scenarios_in_matching_groups"].append(rg_meta.num_rows)

            # Read only the selected camera columns without re-reading scenario_id
            for col_name in selected_cols:
                col_chunk_bytes = sum(
                    rg_meta.column(i).total_compressed_size
                    for i in range(rg_meta.num_columns)
                    if rg_meta.column(i).path_in_schema == col_name
                    or rg_meta.column(i).path_in_schema.startswith(f"{col_name}.")
                )
                transfer_stats["compressed_bytes_by_column"][col_name] += col_chunk_bytes
                transfer_stats["estimated_compressed_column_chunk_bytes"] += col_chunk_bytes

            cam_table = pf.read_row_group(rg_idx, columns=selected_cols)
            rows = cam_table.to_pylist()
            # Attach scenario_id from the previously read rg_meta_table
            for i, row in enumerate(rows):
                row["scenario_id"] = rg_sids_list[i]
        else:
            # No scenario filter: read scenario_id and all selected camera columns
            transfer_stats["matching_row_groups"].append(rg_idx)
            transfer_stats["scenarios_in_matching_groups"].append(rg_meta.num_rows)

            read_cols = ["scenario_id"] + selected_cols
            for col_name in read_cols:
                col_chunk_bytes = sum(
                    rg_meta.column(i).total_compressed_size
                    for i in range(rg_meta.num_columns)
                    if rg_meta.column(i).path_in_schema == col_name
                    or rg_meta.column(i).path_in_schema.startswith(f"{col_name}.")
                )
                transfer_stats["compressed_bytes_by_column"][col_name] += col_chunk_bytes
                transfer_stats["estimated_compressed_column_chunk_bytes"] += col_chunk_bytes

            table = pf.read_row_group(rg_idx, columns=read_cols)
            rows = table.to_pylist()

        for row in rows:
            if limit and len(all_extracted) >= limit:
                break

            sid = str(row["scenario_id"]).zfill(7)
            if target_scenarios and sid not in target_scenarios:
                continue
            if sid in all_extracted:
                continue

            destination = output_dir / sid
            if not force and all(
                (destination / f"{camera}.jpg").exists() for camera in cams_to_extract
            ):
                continue

            saved = extract_scenario_t0(
                row, output_dir, target_cameras=cams_to_extract, resize=resize
            )
            if saved:
                all_extracted[sid] = saved

        print(
            f"    Progress: [{rg_idx + 1:2d}/{total_rgs:2d}] row groups | "
            f"{len(all_extracted)} scenario(s) extracted",
            flush=True,
        )

        # Early stopping: once every requested scenario has been extracted, stop scanning
        if target_scenarios and target_scenarios.issubset(all_extracted.keys()):
            print(
                f"[*] All {len(target_scenarios)} target scenario(s) extracted. "
                f"Stopping row-group scan at row group {rg_idx + 1}/{total_rgs}."
            )
            break

    if return_stats:
        return all_extracted, transfer_stats
    return all_extracted


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_scenarios = None
    if args.scenarios:
        target_scenarios = {s.strip().zfill(7) for s in args.scenarios.split(",") if s.strip()}

    target_cameras = None
    if args.cameras:
        target_cameras = {c.strip() for c in args.cameras.split(",") if c.strip()}
        invalid = target_cameras - set(spec.CAMERAS)
        if invalid:
            print(f"[X] Error: Invalid camera name(s): {invalid}. Valid cameras: {spec.CAMERAS}", file=sys.stderr)
            return 1

    print("=" * 65)
    print("KITScenes-LongTail Observed t=0 Frame Extraction")
    print(f"Output Directory: {args.output_dir}")
    print(f"Cameras:          {sorted(target_cameras) if target_cameras else 'all 6 surround cameras'}")
    print(f"Resolution:       {'876x568 (downscaled)' if args.resize else '3504x2272 (native)'}")
    print("=" * 65)

    start_time = time.time()

    # Mode 1: Local Parquet File
    if args.parquet and Path(args.parquet).exists():
        print(f"[*] Reading from local Parquet file: {args.parquet}")
        results, stats = process_parquet_file(
            parquet_source=args.parquet,
            output_dir=args.output_dir,
            target_scenarios=target_scenarios,
            target_cameras=target_cameras,
            limit=args.limit,
            resize=args.resize,
            force=args.force,
            return_stats=True,
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
            results, stats = process_parquet_file(
                parquet_source=remote_file,
                output_dir=args.output_dir,
                target_scenarios=target_scenarios,
                target_cameras=target_cameras,
                limit=args.limit,
                resize=args.resize,
                force=args.force,
                return_stats=True,
            )

    elapsed = time.time() - start_time
    total_imgs = sum(len(cams) for cams in results.values())

    print("\n" + "=" * 65)
    print("PARQUET ROW-GROUP TRANSFER ACCOUNTING")
    print("=" * 65)
    print(f"Total Row Groups in File:       {stats['total_row_groups']}")
    print(f"Row Groups Scanned:             {stats['row_groups_scanned']}")
    print(f"Matching Row Groups Read:       {stats['matching_row_groups']}")
    print(f"Scenarios in Matching Groups:   {stats['scenarios_in_matching_groups']} (total: {sum(stats['scenarios_in_matching_groups'])})")
    print(f"Target Cameras Requested:       {sorted(target_cameras) if target_cameras else list(CAMERA_FRAME_COLUMNS.keys())}")
    print("Estimated Compressed Column Chunk Bytes (by Column):")
    for col, nbytes in stats["compressed_bytes_by_column"].items():
        mb = nbytes / (1024 * 1024)
        print(f"  - {col:30s}: {mb:8.2f} MB ({nbytes:,} bytes)")
    total_mb = stats["estimated_compressed_column_chunk_bytes"] / (1024 * 1024)
    print(f"Estimated Column Chunk Total:   {total_mb:8.2f} MB ({stats['estimated_compressed_column_chunk_bytes']:,} bytes)")
    print("Note: Represents selected Parquet column-chunk payload sizes read,")
    print("      not measured HTTP network traffic (headers, TLS, range buffering).")
    print("=" * 65)

    # Check for missing scenarios
    if target_scenarios:
        missing = target_scenarios - set(results.keys())
        # Also check if already existing on disk
        cams_to_check = target_cameras if target_cameras else set(CAMERA_FRAME_COLUMNS.keys())
        missing_on_disk = {
            sid for sid in missing
            if not all((args.output_dir / sid / f"{c}.jpg").exists() for c in cams_to_check)
        }
        if missing_on_disk:
            print(f"[X] Error: Target scenario(s) not found in dataset or on disk: {sorted(missing_on_disk)}", file=sys.stderr)
            return 3

    print(f"\nExtracted {len(results)} scenario(s) ({total_imgs} camera frame(s)) in {elapsed:.1f}s.")
    print(f"Saved into: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
