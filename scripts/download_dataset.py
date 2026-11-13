#!/usr/bin/env python3
"""Download and inspect the gated KITScenes-LongTail dataset from Hugging Face.

Prerequisites:
  1. Accept terms at https://huggingface.co/datasets/KIT-MRT/KITScenes-LongTail
  2. Set HF_TOKEN environment variable or pass --token

Usage:
  # Stream metadata and past trajectories only (fast, < 5 MiB on disk, no 32 GB download):
  python scripts/download_dataset.py --token hf_... --metadata-only

  # Full download of multi-view video parquet files:
  python scripts/download_dataset.py --token hf_...
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download KITScenes-LongTail test metadata and scenario files from Hugging Face.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--token",
        type=str,
        default="",
        help="Hugging Face user access token (defaults to HF_TOKEN env var)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory to save downloaded dataset files",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "test_raw", "val", "train"],
        help="Dataset split to download",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="stream scenario metadata, instructions, and past trajectories without downloading 30+ GB video frames",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list the files and their sizes without downloading anything",
    )
    return parser


def space_check(required_bytes: int, free_bytes: int, margin: float = 0.9):
    """Return an error string when ``required_bytes`` will not fit in ``free_bytes``."""
    if required_bytes > free_bytes * margin:
        return (
            f"this download needs {required_bytes / 1024**3:.1f} GiB but only "
            f"{free_bytes / 1024**3:.1f} GiB is free; free space or download on another "
            f"machine, or use --metadata-only to stream non-frame metadata, and see --dry-run for the per-file breakdown"
        )
    return None


def stream_split_metadata(
    repo_id: str,
    matching_files: list[str],
    token: str,
    output_dir: Path,
    split_name: str = "test",
) -> list[Path]:
    """Stream non-frame metadata columns from remote parquet files via HfFileSystem.

    Extracts scenario metadata, instructions, and past trajectories across all row
    groups without downloading the multi-gigabyte video frame columns.
    """
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem
    except ImportError as exc:
        raise RuntimeError(f"Missing required streaming dependency: {exc}") from exc

    fs = HfFileSystem(token=token)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = output_dir / f"{split_name}_metadata.jsonl"
    hist_path = output_dir / "history.jsonl"
    inst_path = output_dir / "instructions.jsonl"

    all_scenarios: dict[str, dict] = {}
    target_columns = [
        "scenario_id",
        "driving_instruction",
        "scenario_type",
        "trajectory",
        "reasoning",
    ]

    for rfile in matching_files:
        hf_uri = f"datasets/{repo_id}/{rfile}"
        print(f"[*] Streaming metadata from: {hf_uri} ...")
        with fs.open(hf_uri, "rb") as remote_file:
            pf = pq.ParquetFile(remote_file)
            available = [c for c in target_columns if c in pf.schema_arrow.names]
            total_rgs = pf.num_row_groups
            total_rows = pf.metadata.num_rows
            print(f"    Available metadata columns: {available}")
            print(f"    Total rows: {total_rows} across {total_rgs} row group(s)")

            for rg_idx in range(total_rgs):
                rg = pf.read_row_group(rg_idx, columns=available)
                rows = rg.to_pylist()
                for row in rows:
                    sid = row.get("scenario_id")
                    if sid:
                        all_scenarios[sid] = row
                print(
                    f"    Progress: [{rg_idx + 1:2d}/{total_rgs:2d}] row groups read "
                    f"({len(all_scenarios)} scenarios)",
                    flush=True,
                )

    sorted_sids = sorted(all_scenarios.keys())
    print(f"\n[✓] Successfully collected metadata for {len(sorted_sids)} scenarios.")

    with open(meta_path, "w", encoding="utf-8") as f_meta, open(
        hist_path, "w", encoding="utf-8"
    ) as f_hist, open(inst_path, "w", encoding="utf-8") as f_inst:
        for sid in sorted_sids:
            row = all_scenarios[sid]
            f_meta.write(json.dumps(row) + "\n")

            traj_data = row.get("trajectory") or {}
            past = traj_data.get("past") if isinstance(traj_data, dict) else None
            instr = row.get("driving_instruction") or ""
            stype = row.get("scenario_type") or ""

            hist_rec = {
                "scenario_id": sid,
                "past_trajectory": past or [],
                "driving_instruction": instr,
                "scenario_type": stype,
            }
            f_hist.write(json.dumps(hist_rec) + "\n")
            f_inst.write(json.dumps({"scenario_id": sid, "instruction": instr}) + "\n")

    return [meta_path, hist_path, inst_path]


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("[X] Error: Hugging Face token is required for gated dataset access.", file=sys.stderr)
        print("    1. Create a read token at: https://huggingface.co/settings/tokens", file=sys.stderr)
        print("    2. Pass --token hf_... or run export HF_TOKEN='hf_...'", file=sys.stderr)
        return 1

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print("[X] huggingface_hub is not installed in the current environment.", file=sys.stderr)
        return 2

    repo_id = "KIT-MRT/KITScenes-LongTail"
    print("=" * 65)
    print(f"Connecting to gated dataset: {repo_id}")
    print("=" * 65)

    # 1. Verify access
    try:
        listing = HfApi().repo_info(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            files_metadata=True,
        )
        files = [sibling.rfilename for sibling in listing.siblings]
        sizes = {sibling.rfilename: sibling.size or 0 for sibling in listing.siblings}
        print(f"[✓] Token accepted! Found {len(files)} files in repository.")
    except Exception as exc:
        print(f"[X] Failed to access repository: {exc}", file=sys.stderr)
        print(
            "    Make sure you accepted the terms on: https://huggingface.co/datasets/KIT-MRT/KITScenes-LongTail",
            file=sys.stderr,
        )
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Identify target files for split
    target_pattern = f"data/{args.split}-"
    matching_files = [f for f in files if f.startswith(target_pattern) and f.endswith(".parquet")]
    if not matching_files:
        print(
            f"[!] No parquet files found matching pattern '{target_pattern}'. Available files:",
            file=sys.stderr,
        )
        for f in files[:10]:
            print(f"    - {f}", file=sys.stderr)
        return 3

    required = sum(sizes.get(name, 0) for name in matching_files)
    print(
        f"\n[*] Split '{args.split}': {len(matching_files)} file(s), "
        f"{required / 1024**3:.1f} GiB full payload on disk"
    )
    for name in matching_files:
        print(f"    {sizes.get(name, 0) / 1024**3:7.1f} GiB  {name}")

    free = shutil.disk_usage(args.output_dir.anchor or ".").free

    # 3. Handle --metadata-only streaming
    if args.metadata_only:
        print("\n[*] Mode: --metadata-only (streaming non-frame columns via HTTP range requests)")
        print("    Payload transferred: ~150-500 KiB (skips downloading 30+ GiB video frames)")
        if args.dry_run:
            print(f"[*] Dry run: {free / 1024**3:.1f} GiB free, nothing downloaded or streamed.")
            return 0

        try:
            out_files = stream_split_metadata(
                repo_id=repo_id,
                matching_files=matching_files,
                token=token,
                output_dir=args.output_dir,
                split_name=args.split,
            )
            print("\n" + "=" * 65)
            print("STREAMING METADATA COMPLETE")
            print("=" * 65)
            for p in out_files:
                size_kb = p.stat().st_size / 1024
                print(f"Saved: {p} ({size_kb:.1f} KiB)")
            return 0
        except Exception as exc:
            print(f"[X] Metadata streaming failed: {exc}", file=sys.stderr)
            return 4

    # 4. Handle full download
    if args.dry_run:
        print(f"\n[*] Dry run: {free / 1024**3:.1f} GiB free, nothing downloaded")
        problem = space_check(required, free)
        if problem:
            print(f"[!] this would not fit: {problem}", file=sys.stderr)
            return 3
        return 0

    problem = space_check(required, free)
    if problem:
        print(f"[X] Refusing to download: {problem}", file=sys.stderr)
        return 3

    print(f"\n[*] Downloading split '{args.split}' ({len(matching_files)} file(s))...")
    downloaded = []
    for rfile in matching_files:
        print(f"    Downloading {rfile}...")
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=rfile,
            token=token,
            local_dir=args.output_dir,
        )
        downloaded.append(Path(local_path))
        print(f"    [✓] Saved to: {local_path}")

    print("\n" + "=" * 65)
    print("DOWNLOAD COMPLETE")
    print("=" * 65)
    for p in downloaded:
        size_mb = p.stat().st_size / (1024**2)
        print(f"File: {p.name} ({size_mb:.2f} MiB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
