#!/usr/bin/env python3
"""Ground-truth A/B of renderer changes on validation scenarios (CPU or GPU).

The official score is Q-Align on the five 1 fps frames (0, 5, 10, 15, 20) of each clip,
which we cannot run on a laptop. This script measures the properties Q-Align reacts to
against the real future video that the validation split ships with, so a renderer change
can be accepted or rejected without spending a submission slot.

Reported per frame index, rendered vs ground truth:

  psnr     pixel agreement with the real future (geometry check)
  sharp    mean |Laplacian| of luma; both blur and resampling aliasing move this
  hf       energy above 1/4 Nyquist, normalised by the GT frame's own energy
  sat      mean HSV saturation; the submission drifted 19 -> 33 on construction footage
  holes    fraction of the frame the renderer had to synthesise instead of sample

Usage:
  python scripts/ab_render_metrics.py --scenario 0000903 --chain both
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitscenes import spec  # noqa: E402
from kitscenes.depth_reproj_world_model import Depth3DFullFoVWorldModel  # noqa: E402

SCORED_INDICES = tuple(range(0, spec.FRAME_COUNT, int(spec.FPS)))


def decode_clip(path: Path) -> np.ndarray:
    """Decode every frame of a clip to (T, HEIGHT, WIDTH, 3) uint8 RGB."""
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-vf", f"scale={spec.WIDTH}:{spec.HEIGHT}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        capture_output=True,
        check=False,
    )
    frame_bytes = spec.WIDTH * spec.HEIGHT * 3
    if result.returncode != 0 or len(result.stdout) < frame_bytes:
        raise RuntimeError(f"could not decode {path}: {result.stderr.decode()[:200]}")
    count = len(result.stdout) // frame_bytes
    data = np.frombuffer(result.stdout[: count * frame_bytes], dtype=np.uint8)
    return data.reshape(count, spec.HEIGHT, spec.WIDTH, 3)


def frame_metrics(render: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    """Compare one rendered frame against the real frame at the same time step."""
    r = render.astype(np.float32)
    t = truth.astype(np.float32)
    mse = float(np.mean((r - t) ** 2))
    psnr = 99.0 if mse <= 1e-9 else float(10.0 * np.log10(255.0**2 / mse))

    r_l = cv2.cvtColor(render, cv2.COLOR_RGB2GRAY).astype(np.float32)
    t_l = cv2.cvtColor(truth, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sharp_r = float(np.abs(cv2.Laplacian(r_l, cv2.CV_32F)).mean())
    sharp_t = float(np.abs(cv2.Laplacian(t_l, cv2.CV_32F)).mean())

    def high_freq(plane: np.ndarray) -> float:
        blur = cv2.GaussianBlur(plane, (0, 0), 1.0)
        return float(np.abs(plane - blur).mean())

    hf_r, hf_t = high_freq(r_l), high_freq(t_l)

    sat_r = float(cv2.cvtColor(render, cv2.COLOR_RGB2HSV)[..., 1].mean())
    sat_t = float(cv2.cvtColor(truth, cv2.COLOR_RGB2HSV)[..., 1].mean())

    return {
        "psnr": psnr,
        "sharp": sharp_r / max(sharp_t, 1e-6),
        "hf": hf_r / max(hf_t, 1e-6),
        "sat": sat_r / max(sat_t, 1e-6),
        "sat_abs": sat_r,
    }


def load_inputs(sid: str, split: str) -> tuple:
    obs = PROJECT_ROOT / "data" / f"{split}_observed_t0" / sid
    traj_path = PROJECT_ROOT / "data" / ("trajectories.jsonl" if split == "test" else f"{split}_trajectories.jsonl")
    future: Dict[str, List[List[float]]] = {}
    past: Dict[str, List[List[float]]] = {}
    if traj_path.exists():
        with open(traj_path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if str(record["scenario_id"]).zfill(7) != sid:
                    continue
                # The validation split stores the driven future as "expert_like"; the test
                # predictions carry "future_trajectory".
                future = (
                    record.get("future_trajectory")
                    or record.get("future")
                    or record.get("expert_like")
                    or []
                )
                past = record.get("past_trajectory") or record.get("past") or []
    if not future:
        raise SystemExit(f"no future trajectory for {sid} in {traj_path}")
    conditioning = {cam: obs / f"{cam}.jpg" for cam in spec.CAMERAS}
    missing = [str(p) for p in conditioning.values() if not p.exists()]
    if missing:
        raise SystemExit(f"missing conditioning frames: {missing}")
    past_images = {}
    for cam in spec.CAMERAS:
        for tag in ("tm02", "tm04", "tm10", "tm20", "tm30", "tm40"):
            candidate = obs / f"{cam}_{tag}.jpg"
            if candidate.exists():
                past_images[(cam, tag)] = candidate
    return conditioning, future, past or None, (past_images or None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True, help="Seven-digit scenario id")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--chain", default="both", choices=("both", "on", "off"))
    parser.add_argument("--render-scale", type=int, default=1, help="Renderer supersampling factor")
    parser.add_argument("--gpu", action="store_true", help="Render on CUDA if available")
    args = parser.parse_args(argv)

    sid = args.scenario.zfill(7)
    conditioning, future, past, past_images = load_inputs(sid, args.split)
    truth_path = PROJECT_ROOT / "data" / f"{args.split}_future_videos" / sid / "front.mp4"
    truth = decode_clip(truth_path) if truth_path.exists() else None
    if truth is None:
        print(f"warning: no ground-truth clip at {truth_path}; reporting render-only stats")

    variants = {"off": False, "on": True}
    if args.chain != "both":
        variants = {args.chain: args.chain == "on"}

    for name, chain in variants.items():
        model = Depth3DFullFoVWorldModel(
            device="cuda" if args.gpu else "cpu",
            render_scale=args.render_scale,
            chain_warps=chain,
        )
        started = time.time()
        videos = model.generate_scenario(
            sid, future, conditioning, past_images=past_images, past_trajectory=past
        )
        elapsed = time.time() - started
        holes = np.array([model.last_hole_fraction[cam] for cam in spec.CAMERAS], dtype=np.float64)
        print(f"\n=== {sid} chain={name} render_scale={args.render_scale} ({elapsed:.1f}s) ===")
        print(f"hole fraction: mean={holes.mean():.4f} worst_frame={holes.max():.4f}")
        for cam in ("front",):
            clip = videos[cam]
            rows = []
            for index in SCORED_INDICES:
                row = {"frame": index}
                if truth is not None:
                    row.update(frame_metrics(clip[index], truth[index]))
                rows.append(row)
            fields = [key for key in rows[0] if key != "frame"]
            header = "frame  " + "  ".join(f"{key:>7s}" for key in fields)
            print(f"[{cam}] {header}")
            for row in rows:
                cells = "  ".join(f"{row[key]:7.3f}" for key in row if key != "frame")
                print(f"{row['frame']:5d}  {cells}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
