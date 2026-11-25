#!/usr/bin/env python3
"""Extract high-resolution t=0 and multi-frame past observations from KITScenes-LongTail.

Designed to run in a free Kaggle CPU session (0.00 GPU hours) to extract:
1. Test split (all 400 scenarios x 6 cameras = 2,400 cameras):
   - Native full-resolution t=0 conditioning frame (<sid>/<cam>.jpg, raw sensor JPEG bytes)
   - Downscaled 876x568 past context frames at t = -2.0s, -1.0s, -0.4s, -0.2s
   - Test metadata JSONL (scenario_id, driving_instruction, scenario_type, past trajectory)
2. Validation calibration split (diverse scenarios across all 7 challenge slice types):
   - Native t=0 frames and past context frames
   - Ground-truth 25-frame @ 5fps H.264 future videos (876x568) from val_future
   - Validation metadata JSONL (with ground-truth future trajectories)

Optional --pack-tar bundles output into 3 sequential .tar files so Kaggle Notebook Output
stores thousands of frames cleanly with fast attach speeds in GPU sessions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitscenes import spec

CAMERA_FRAME_COLUMNS: Dict[str, str] = {
    "front": "frames_camera_front",
    "front_left": "frames_camera_front_left",
    "front_right": "frames_camera_front_right",
    "rear": "frames_camera_rear",
    "rear_left": "frames_camera_rear_left",
    "rear_right": "frames_camera_rear_right",
}

DEFAULT_PAST_OFFSETS: Tuple[Tuple[int, str], ...] = (
    (-21, "tm40"),
    (-16, "tm30"),
    (-11, "tm20"),
    (-6, "tm10"),
    (-3, "tm04"),
    (-2, "tm02"),
)

DEFAULT_VAL_CALIB_SCENARIOS: Tuple[str, ...] = (
    # 0 specifically selected
    "0000905", "0000907", "0000908",
    # 1 heavy rain
    "0000903", "0000912", "0000915",
    # 2 construction zone
    "0000904", "0000918", "0000934",
    # 3 overtake or lane change
    "0000901", "0000914", "0000916",
    # 4 intersection
    "0000902", "0000913", "0000920",
    # 5 nighttime
    "0000906", "0000926", "0000962",
    # 6 snow & wintry mix
    "0000900", "0000909", "0000910",
)


def _extract_bytes(frame_obj: Any) -> bytes:
    if isinstance(frame_obj, dict):
        return bytes(frame_obj.get("bytes") or b"")
    return bytes(getattr(frame_obj, "bytes", b"") or b"")


def resize_jpeg_bytes(
    raw_bytes: bytes,
    width: int = spec.WIDTH,
    height: int = spec.HEIGHT,
    quality: int = 92,
) -> bytes:
    """Downscale encoded image bytes cleanly with Lanczos resampling."""
    from PIL import Image

    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def encode_frames_h264(raw_frames: Sequence[bytes], out_mp4: Path, fps: float = float(spec.FPS)) -> None:
    """Decode raw frame bytes, resize to 876x568, and encode H.264 MP4 via ffmpeg."""
    from PIL import Image
    import numpy as np

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg binary is required to encode validation GT future videos.")

    cmd = [
        ffmpeg_bin,
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{spec.WIDTH}x{spec.HEIGHT}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-crf",
        "14",
        "-pix_fmt",
        "yuv420p",
        str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for rb in raw_frames:
        img = Image.open(io.BytesIO(rb)).convert("RGB")
        if img.size != (spec.WIDTH, spec.HEIGHT):
            img = img.resize((spec.WIDTH, spec.HEIGHT), Image.Resampling.LANCZOS)
        proc.stdin.write(np.asarray(img, dtype=np.uint8).tobytes())
    proc.stdin.close()
    stderr_bytes = proc.stderr.read() if proc.stderr else b""
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed ({ret}) writing {out_mp4}: {stderr_bytes.decode(errors='replace')}")


def _process_observed_row_group(
    table_dict: Dict[str, List[Any]],
    t0_dir: Path,
    past_dir: Path,
    past_offsets: Sequence[Tuple[int, str]],
    target_scenarios: Optional[Set[str]],
) -> List[Dict[str, Any]]:
    """Write native t=0 JPEG and resized past context JPEGs for one row group."""
    sids = [str(s).zfill(7) for s in table_dict["scenario_id"]]
    n_rows = len(sids)
    meta_records: List[Dict[str, Any]] = []

    for row_idx in range(n_rows):
        sid = sids[row_idx]
        if target_scenarios is not None and sid not in target_scenarios:
            continue

        t0_scene = t0_dir / sid
        past_scene = past_dir / sid
        t0_scene.mkdir(parents=True, exist_ok=True)
        past_scene.mkdir(parents=True, exist_ok=True)

        for cam, col_name in CAMERA_FRAME_COLUMNS.items():
            frames = table_dict[col_name][row_idx]
            if not frames:
                raise RuntimeError(f"Empty frame list for {sid}/{cam}")

            t0_out = t0_scene / f"{cam}.jpg"
            if not t0_out.exists() or t0_out.stat().st_size < 1000:
                t0_bytes = _extract_bytes(frames[-1])
                if not t0_bytes:
                    raise RuntimeError(f"Empty t=0 frame bytes for {sid}/{cam}")
                t0_out.write_bytes(t0_bytes)

            for offset, tag in past_offsets:
                if abs(offset) > len(frames):
                    continue
                p_out = past_scene / f"{cam}_{tag}.jpg"
                if not p_out.exists() or p_out.stat().st_size < 1000:
                    raw_b = _extract_bytes(frames[offset])
                    if raw_b:
                        p_out.write_bytes(resize_jpeg_bytes(raw_b, spec.WIDTH, spec.HEIGHT, quality=92))

        rec = {
            "scenario_id": sid,
            "driving_instruction": table_dict["driving_instruction"][row_idx]
            if "driving_instruction" in table_dict
            else "",
            "scenario_type": table_dict["scenario_type"][row_idx]
            if "scenario_type" in table_dict
            else "",
            "trajectory": table_dict["trajectory"][row_idx]
            if "trajectory" in table_dict
            else None,
        }
        meta_records.append(rec)

    return meta_records


def _worker_rg_from_file(args: Tuple[str, int, List[str], str, str, Sequence[Tuple[int, str]], Optional[Set[str]]]) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet_path, rg_idx, cols, t0_dir_s, past_dir_s, past_offsets, target_scenarios = args
    pf = pq.ParquetFile(parquet_path)
    tbl = pf.read_row_group(rg_idx, columns=cols)
    return _process_observed_row_group(
        tbl.to_pydict(),
        Path(t0_dir_s),
        Path(past_dir_s),
        past_offsets,
        target_scenarios,
    )


def _worker_rg_from_hf(args: Tuple[str, str, int, List[str], str, str, Sequence[Tuple[int, str]], Optional[Set[str]]]) -> List[Dict[str, Any]]:
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    remote_uri, token, rg_idx, cols, t0_dir_s, past_dir_s, past_offsets, target_scenarios = args
    fs = HfFileSystem(token=token)
    with fs.open(remote_uri, "rb") as f:
        pf = pq.ParquetFile(f)
        tbl = pf.read_row_group(rg_idx, columns=cols)
    return _process_observed_row_group(
        tbl.to_pydict(),
        Path(t0_dir_s),
        Path(past_dir_s),
        past_offsets,
        target_scenarios,
    )


def _scenario_complete(
    t0_dir: Path,
    past_dir: Path,
    sid: str,
    past_offsets: Sequence[Tuple[int, str]],
) -> bool:
    for cam in spec.CAMERAS:
        if not (t0_dir / sid / f"{cam}.jpg").exists():
            return False
        for _, tag in past_offsets:
            if not (past_dir / sid / f"{cam}_{tag}.jpg").exists():
                return False
    return True


def extract_observed_split(
    repo_id: str,
    split: str,
    token: str,
    out_root: Path,
    target_scenarios: Optional[Set[str]] = None,
    workers: int = 4,
    download_first: bool = True,
    past_offsets: Sequence[Tuple[int, str]] = DEFAULT_PAST_OFFSETS,
) -> List[Dict[str, Any]]:
    """Extract native t=0 frames + resized past frames for ``split``."""
    from huggingface_hub import HfFileSystem, hf_hub_download
    import pyarrow.parquet as pq

    t0_dir = out_root / f"{split}_t0_native"
    past_dir = out_root / f"{split}_past"
    meta_path = out_root / f"{split}_metadata.jsonl"
    t0_dir.mkdir(parents=True, exist_ok=True)
    past_dir.mkdir(parents=True, exist_ok=True)

    rel_path = f"data/{split}-00000-of-00001.parquet"
    remote_uri = f"datasets/{repo_id}/{rel_path}"

    fs = HfFileSystem(token=token)
    pending_rgs: List[int] = []
    with fs.open(remote_uri, "rb") as f:
        pf_meta = pq.ParquetFile(f)
        num_rgs = pf_meta.num_row_groups
        available_cols = set(pf_meta.schema_arrow.names)

        cols = ["scenario_id"] + [CAMERA_FRAME_COLUMNS[c] for c in spec.CAMERAS]
        for extra in ("driving_instruction", "scenario_type", "trajectory"):
            if extra in available_cols:
                cols.append(extra)

        for rg_idx in range(num_rgs):
            sid_col = [
                str(s).zfill(7)
                for s in pf_meta.read_row_group(rg_idx, columns=["scenario_id"])["scenario_id"].to_pylist()
            ]
            relevant = [s for s in sid_col if target_scenarios is None or s in target_scenarios]
            if relevant and not all(_scenario_complete(t0_dir, past_dir, s, past_offsets) for s in relevant):
                pending_rgs.append(rg_idx)

    if not pending_rgs:
        print(f"[+] {split}: all target observations already extracted on disk.")
        if meta_path.exists():
            return [json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()]
        return []

    disk_free_gb = shutil.disk_usage(out_root).free / (1024 ** 3)
    print(f"[*] {split}: {len(pending_rgs)} row groups pending | free disk = {disk_free_gb:.1f} GiB")

    local_parquet: Optional[Path] = None
    temp_dl_dir: Optional[Path] = None
    all_meta: List[Dict[str, Any]] = []
    t_start = time.time()

    try:
        # Use fast full-file download only when extracting all/most row groups and disk permits
        if download_first and len(pending_rgs) >= 15 and disk_free_gb >= 42.0:
            temp_dl_dir = out_root / "_tmp_parquet"
            temp_dl_dir.mkdir(parents=True, exist_ok=True)
            print(f"[*] Downloading {rel_path} into {temp_dl_dir} (disk free {disk_free_gb:.1f} GiB >= 42 GiB)...")
            dl_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=rel_path,
                    token=token,
                    local_dir=str(temp_dl_dir),
                )
            )
            local_parquet = dl_path
            print(f"[+] Downloaded {local_parquet} in {time.time() - t_start:.1f}s")

        if local_parquet is not None:
            tasks = [
                (str(local_parquet), rg, cols, str(t0_dir), str(past_dir), past_offsets, target_scenarios)
                for rg in pending_rgs
            ]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_worker_rg_from_file, t): t[1] for t in tasks}
                for done_idx, fut in enumerate(as_completed(futures), start=1):
                    rg = futures[fut]
                    recs = fut.result()
                    all_meta.extend(recs)
                    print(f"    [{split}] RG {rg:02d} done ({done_idx}/{len(tasks)}, {len(recs)} scenarios) | elapsed={time.time() - t_start:.1f}s", flush=True)
        else:
            tasks = [
                (remote_uri, token, rg, cols, str(t0_dir), str(past_dir), past_offsets, target_scenarios)
                for rg in pending_rgs
            ]
            with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
                futures = {pool.submit(_worker_rg_from_hf, t): t[2] for t in tasks}
                for done_idx, fut in enumerate(as_completed(futures), start=1):
                    rg = futures[fut]
                    recs = fut.result()
                    all_meta.extend(recs)
                    print(f"    [{split}] RG {rg:02d} streamed ({done_idx}/{len(tasks)}, {len(recs)} scenarios) | elapsed={time.time() - t_start:.1f}s", flush=True)
    finally:
        if temp_dl_dir is not None and temp_dl_dir.exists():
            shutil.rmtree(temp_dl_dir, ignore_errors=True)

    # Merge with existing metadata JSONL if present
    by_sid: Dict[str, Dict[str, Any]] = {}
    if meta_path.exists():
        for line in meta_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                by_sid[str(r["scenario_id"]).zfill(7)] = r
    for r in all_meta:
        by_sid[str(r["scenario_id"]).zfill(7)] = r

    sorted_recs = [by_sid[k] for k in sorted(by_sid)]
    with open(meta_path, "w") as f:
        for r in sorted_recs:
            f.write(json.dumps(r) + "\n")

    print(f"[+] Saved {len(sorted_recs)} metadata records to {meta_path}")
    return sorted_recs


def extract_val_future_videos(
    repo_id: str,
    token: str,
    out_root: Path,
    target_scenarios: Set[str],
    workers: int = 4,
) -> int:
    """Stream val_future parquet row groups and encode exact 876x568 5fps H.264 GT MP4s."""
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    fut_dir = out_root / "val_future_videos"
    fut_dir.mkdir(parents=True, exist_ok=True)
    remote_uri = f"datasets/{repo_id}/data/val_future-00000-of-00001.parquet"
    cols = ["scenario_id"] + [CAMERA_FRAME_COLUMNS[c] for c in spec.CAMERAS]

    fs = HfFileSystem(token=token)
    with fs.open(remote_uri, "rb") as f:
        pf = pq.ParquetFile(f)
        num_rgs = pf.num_row_groups
        needed_rgs: List[int] = []
        for rg_idx in range(num_rgs):
            sids = [str(s).zfill(7) for s in pf.read_row_group(rg_idx, columns=["scenario_id"])["scenario_id"].to_pylist()]
            if any(s in target_scenarios for s in sids):
                # Check if already done
                all_done = True
                for s in sids:
                    if s in target_scenarios:
                        for cam in spec.CAMERAS:
                            if not (fut_dir / s / f"{cam}.mp4").exists():
                                all_done = False
                                break
                if not all_done:
                    needed_rgs.append(rg_idx)

    if not needed_rgs:
        print("[+] All target val_future GT videos already encoded.")
        return 0

    encoded_count = 0
    t0 = time.time()
    for rg_idx in needed_rgs:
        with fs.open(remote_uri, "rb") as f:
            pf = pq.ParquetFile(f)
            tbl = pf.read_row_group(rg_idx, columns=cols).to_pydict()

        encode_jobs: List[Tuple[List[bytes], Path]] = []
        sids = [str(s).zfill(7) for s in tbl["scenario_id"]]
        for i, sid in enumerate(sids):
            if sid not in target_scenarios:
                continue
            for cam, col in CAMERA_FRAME_COLUMNS.items():
                mp4_out = fut_dir / sid / f"{cam}.mp4"
                if mp4_out.exists() and mp4_out.stat().st_size > 1000:
                    continue
                frames = tbl[col][i]
                if len(frames) != spec.FRAME_COUNT:
                    raise ValueError(f"Expected 25 frames for {sid}/{cam}, got {len(frames)}")
                raw_list = [_extract_bytes(fr) for fr in frames]
                encode_jobs.append((raw_list, mp4_out))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(encode_frames_h264, raw_l, mp4_p) for raw_l, mp4_p in encode_jobs]
            for fut in as_completed(futs):
                fut.result()
                encoded_count += 1
        print(f"    [val_future] RG {rg_idx:02d} done -> encoded {encoded_count} GT MP4s ({time.time() - t0:.1f}s)", flush=True)

    return encoded_count


def pack_tar(src_dir: Path, tar_path: Path, arc_root: str) -> None:
    """Bundle directory into uncompressed tar for fast sequential Kaggle read."""
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tf:
        tf.add(src_dir, arcname=arc_root)
    size_mib = tar_path.stat().st_size / (1024 ** 2)
    print(f"[+] Packed {tar_path} ({size_mib:.1f} MiB)")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract native t=0 + past frames and val GT videos.")
    parser.add_argument("--token", type=str, default="", help="HF token (defaults to HF_TOKEN env var)")
    parser.add_argument("--repo-id", type=str, default="KIT-MRT/KITScenes-LongTail")
    parser.add_argument("--out-root", type=Path, default=Path("/kaggle/working/kitscenes_cache"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-download-first", action="store_true", help="Always stream row groups instead of downloading full parquet")
    parser.add_argument("--extract-test", action="store_true", default=False)
    parser.add_argument("--extract-val", action="store_true", default=False)
    parser.add_argument("--test-scenarios", type=str, default="", help="Optional comma-separated test scenarios subset")
    parser.add_argument("--val-scenarios", type=str, default=",".join(DEFAULT_VAL_CALIB_SCENARIOS))
    parser.add_argument("--pack-tar", action="store_true", default=False, help="Bundle cache folders into 3 .tar archives in out-root.parent")
    parser.add_argument("--tar-dir", type=Path, default=Path("/kaggle/working"))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    if not token:
        print("[X] HF_TOKEN required (--token or HF_TOKEN env var)", file=sys.stderr)
        return 1

    args.out_root.mkdir(parents=True, exist_ok=True)

    if args.extract_val:
        val_sids = {s.strip().zfill(7) for s in args.val_scenarios.split(",") if s.strip()}
        extract_observed_split(
            repo_id=args.repo_id,
            split="val",
            token=token,
            out_root=args.out_root,
            target_scenarios=val_sids,
            workers=args.workers,
            download_first=False,
        )
        extract_val_future_videos(
            repo_id=args.repo_id,
            token=token,
            out_root=args.out_root,
            target_scenarios=val_sids,
            workers=args.workers,
        )

    if args.extract_test:
        test_sids: Optional[Set[str]] = None
        if args.test_scenarios.strip():
            test_sids = {s.strip().zfill(7) for s in args.test_scenarios.split(",") if s.strip()}
        extract_observed_split(
            repo_id=args.repo_id,
            split="test",
            token=token,
            out_root=args.out_root,
            target_scenarios=test_sids,
            workers=args.workers,
            download_first=not args.no_download_first,
        )

    if args.pack_tar:
        args.tar_dir.mkdir(parents=True, exist_ok=True)
        if (args.out_root / "test_t0_native").exists():
            pack_tar(args.out_root / "test_t0_native", args.tar_dir / "test_t0_native.tar", "test_t0_native")
        if (args.out_root / "test_past").exists():
            pack_tar(args.out_root / "test_past", args.tar_dir / "test_past.tar", "test_past")
        if (args.out_root / "test_metadata.jsonl").exists():
            shutil.copy2(args.out_root / "test_metadata.jsonl", args.tar_dir / "test_metadata.jsonl")

        val_bundle_tmp = args.out_root / "_val_bundle"
        if (args.out_root / "val_t0_native").exists():
            val_bundle_tmp.mkdir(exist_ok=True)
            for sub in ("val_t0_native", "val_past", "val_future_videos"):
                p = args.out_root / sub
                if p.exists():
                    if (val_bundle_tmp / sub).exists():
                        shutil.rmtree(val_bundle_tmp / sub)
                    shutil.move(str(p), str(val_bundle_tmp / sub))
            if (args.out_root / "val_metadata.jsonl").exists():
                shutil.copy2(args.out_root / "val_metadata.jsonl", val_bundle_tmp / "val_metadata.jsonl")
            pack_tar(val_bundle_tmp, args.tar_dir / "val_calib.tar", "val_calib")

        # Cleanup loose dirs to keep Kaggle working directory compact (<3.5 GB)
        shutil.rmtree(args.out_root, ignore_errors=True)
        print("[+] Cleaned up loose extracted files; preserved compact .tar archives.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
