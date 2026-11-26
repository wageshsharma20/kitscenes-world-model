#!/usr/bin/env python3
"""Dual-GPU production renderer for all 400 KITScenes test scenarios (2,400 MP4 videos).

Renders calibrated full-FoV 3D multi-view + multi-frame reprojection videos at native
3488x2272 -> 876x568 @ 5fps H.264, supporting parallel multi-GPU workers (`--gpus 0,1`),
monocular Depth-Anything-V2 foreground elevation, LaMa hole completion, and verified
ZIP archive assembly (`<= 3 GiB`) via `kitscenes.archive.build_archive`.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
import tarfile
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitscenes import archive, spec
from kitscenes.depth_reproj_world_model import Depth3DFullFoVWorldModel
from scripts.eval_world_model_qalign import build_depth_fn, build_lama_fn, encode_rgb_h264


def load_test_inputs(
    traj_path: Path,
    history_path: Path,
) -> List[Tuple[str, List[List[float]], Optional[List[List[float]]]]]:
    """Merge future trajectory plan + past odometry for all 400 test scenarios."""
    past_by_sid: Dict[str, List[List[float]]] = {}
    if history_path.exists():
        with open(history_path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                sid = str(r["scenario_id"]).zfill(7)
                past = r.get("past_trajectory") or r.get("trajectory", {}).get("past")
                if past:
                    past_by_sid[sid] = past

    items: List[Tuple[str, List[List[float]], Optional[List[List[float]]]]] = []
    with open(traj_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            sid = str(r["scenario_id"]).zfill(7)
            fut = r.get("future_trajectory") or r.get("trajectory")
            if not fut or len(fut) != spec.WAYPOINT_COUNT:
                raise ValueError(f"Expected 25 waypoints for {sid}, got {len(fut or [])}")
            items.append((sid, fut, past_by_sid.get(sid)))
    items.sort(key=lambda x: x[0])
    return items


def unpack_if_needed(src_dir: Optional[Path], tar_path: Optional[Path], dst_parent: Path, expected_sub: str) -> Path:
    if src_dir is not None and src_dir.exists():
        return src_dir
    if tar_path is None or not tar_path.exists():
        raise FileNotFoundError(f"Missing directory {src_dir} and archive {tar_path}")
    target = dst_parent / expected_sub
    if target.exists() and any(target.iterdir()):
        return target
    dst_parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] Extracting {tar_path} -> {dst_parent} ...", flush=True)
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(dst_parent)
    if not target.exists():
        raise FileNotFoundError(f"Expected unpacked subdirectory {target} inside {tar_path}")
    return target


def _gpu_worker(
    worker_id: int,
    gpu_idx: Optional[int],
    tasks: List[Tuple[str, List[List[float]], Optional[List[List[float]]]]],
    t0_dir: Path,
    past_dir: Optional[Path],
    out_videos_dir: Path,
    cfg: Dict[str, object],
) -> Tuple[int, float]:
    if gpu_idx is not None and torch.cuda.is_available():
        device = f"cuda:{gpu_idx}"
        torch.cuda.set_device(gpu_idx)
    else:
        device = "cpu"

    depth_fn = build_depth_fn(str(cfg["depth_model"]), device=device) if cfg["use_depth"] else None
    lama_fn = build_lama_fn(device=device) if cfg["use_lama"] else None
    model = Depth3DFullFoVWorldModel(
        device=device,
        render_scale=int(cfg["render_scale"]),
        depth_fn=depth_fn,
        inpaint_fn=lama_fn,
        crf_sharpen=float(cfg["crf_sharpen"]),
    )

    t_start = time.time()
    rendered = 0
    for idx, (sid, fut_traj, past_traj) in enumerate(tasks, start=1):
        scen_dir = out_videos_dir / sid
        if all((scen_dir / f"{c}.mp4").exists() for c in spec.CAMERAS):
            rendered += 1
            continue

        cond = {c: t0_dir / sid / f"{c}.jpg" for c in spec.CAMERAS}
        past_imgs: Dict[Tuple[str, str], Path] = {}
        if past_dir is not None:
            for cam in spec.CAMERAS:
                for tag in ("tm02", "tm04", "tm10", "tm20", "tm30", "tm40"):
                    pp = past_dir / sid / f"{cam}_{tag}.jpg"
                    if pp.exists():
                        past_imgs[(cam, tag)] = pp

        gen = model.generate_scenario(
            sid,
            fut_traj,
            cond,
            past_images=past_imgs if past_imgs else None,
            past_trajectory=past_traj,
        )
        scen_dir.mkdir(parents=True, exist_ok=True)
        for cam in spec.CAMERAS:
            encode_rgb_h264(gen[cam], scen_dir / f"{cam}.mp4", crf=int(cfg["crf"]))
        rendered += 1

        if idx % 10 == 0 or idx == len(tasks):
            elapsed = time.time() - t_start
            rate = idx / max(elapsed, 1e-3)
            print(
                f"  [Worker {worker_id} ({device})] {idx:03d}/{len(tasks)} ({sid}) "
                f"| {elapsed:.1f}s | {rate:.2f} scen/s",
                flush=True,
            )

    return rendered, time.time() - t_start


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate all 2,400 production KITScenes videos cleanly.")
    parser.add_argument("--trajectories", type=Path, default=PROJECT_ROOT / "submissions" / "trajectories.jsonl")
    parser.add_argument("--history", type=Path, default=PROJECT_ROOT / "data" / "history.jsonl")
    parser.add_argument("--t0-dir", type=Path, default=None)
    parser.add_argument("--t0-tar", type=Path, default=None)
    parser.add_argument("--past-dir", type=Path, default=None)
    parser.add_argument("--past-tar", type=Path, default=None)
    parser.add_argument("--unpack-dir", type=Path, default=Path("/kaggle/working/unpacked_test"))
    parser.add_argument("--out-videos-dir", type=Path, default=Path("/kaggle/working/production_videos"))
    parser.add_argument("--out-zip", type=Path, default=Path("/kaggle/working/videos.zip"))
    parser.add_argument("--depth-model", type=str, default="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf")
    parser.add_argument("--no-depth", action="store_true", default=False)
    parser.add_argument("--no-lama", action="store_true", default=False)
    parser.add_argument("--render-scale", type=int, default=2)
    parser.add_argument("--crf-sharpen", type=float, default=0.0)
    parser.add_argument("--crf", type=int, default=14)
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated CUDA device indices (or empty for CPU)")
    parser.add_argument("--scenarios", type=str, default="", help="Optional comma-separated scenario IDs subset")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--pack-zip", action="store_true", default=False)
    parser.add_argument("--allow-partial", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    t0_dir = unpack_if_needed(args.t0_dir, args.t0_tar, args.unpack_dir, "test_t0_native")
    past_dir: Optional[Path] = None
    if args.past_dir is not None or (args.past_tar is not None and args.past_tar.exists()):
        past_dir = unpack_if_needed(args.past_dir, args.past_tar, args.unpack_dir, "test_past")

    items = load_test_inputs(args.trajectories, args.history)
    if args.scenarios.strip():
        want = {s.strip().zfill(7) for s in args.scenarios.split(",") if s.strip()}
        items = [it for it in items if it[0] in want]
    if args.limit > 0:
        items = items[: args.limit]
    if not items:
        raise RuntimeError("No scenarios selected to render.")

    n_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",") if g.strip() and int(g.strip()) < n_cuda]

    cfg: Dict[str, object] = {
        "depth_model": args.depth_model,
        "use_depth": not args.no_depth,
        "use_lama": not args.no_lama,
        "render_scale": args.render_scale,
        "crf_sharpen": args.crf_sharpen,
        "crf": args.crf,
    }

    t0 = time.time()
    print(f"[*] Rendering {len(items)} scenarios | GPUs={gpu_ids or ['cpu']} | cfg={cfg}", flush=True)

    if len(gpu_ids) <= 1:
        gpu_sel = gpu_ids[0] if gpu_ids else None
        count, elapsed = _gpu_worker(0, gpu_sel, items, t0_dir, past_dir, args.out_videos_dir, cfg)
        print(f"[+] Single-worker complete: {count} scenarios in {elapsed:.1f}s", flush=True)
    else:
        shards: List[List[Tuple[str, List[List[float]], Optional[List[List[float]]]]]] = [
            items[i :: len(gpu_ids)] for i in range(len(gpu_ids))
        ]
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(gpu_ids)) as pool:
            jobs = [
                pool.apply_async(
                    _gpu_worker,
                    (wid, gpu_ids[wid], shards[wid], t0_dir, past_dir, args.out_videos_dir, cfg),
                )
                for wid in range(len(gpu_ids))
            ]
            for job in jobs:
                job.get()
        print(f"[+] Parallel Dual-GPU render complete in {time.time() - t0:.1f}s", flush=True)

    if args.pack_zip:
        print(f"[*] Validating and packing {args.out_zip} ...", flush=True)
        report = archive.build_archive(
            args.out_videos_dir,
            output=args.out_zip,
            allow_partial=args.allow_partial,
            check_media=True,
        )
        for line in report.summary_lines():
            print(f"    {line}", flush=True)
        if not report.ok:
            for err in report.errors[:20]:
                print(f"[!] Archive error: {err}", file=sys.stderr, flush=True)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
