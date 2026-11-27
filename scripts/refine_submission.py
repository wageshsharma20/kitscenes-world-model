#!/usr/bin/env python3
"""Apply photometric refinement to already-rendered clips and re-encode them.

Geometry, motion and frame count are untouched: the pass only stabilises exposure and
white balance, restores acutance and removes the colour cast a stretched warp leaves.
Repack afterwards with scripts/build_submission.py package.

Usage:
  python scripts/refine_submission.py --renders renders/ --out-dir refined/ \\
      --anchor-dir data/observed_t0 --jobs 8
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitscenes import refine, spec  # noqa: E402
from kitscenes.video_engine import VideoEncoder  # noqa: E402


def decode_clip(path: Path) -> np.ndarray:
    """Decode every frame of a clip to (T, 568, 876, 3) uint8."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"scale={spec.WIDTH}:{spec.HEIGHT}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
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


def load_anchor(anchor_dir: Optional[Path], scenario_id: str, camera: str) -> Optional[np.ndarray]:
    if anchor_dir is None:
        return None
    path = anchor_dir / scenario_id / f"{camera}.jpg"
    if not path.exists():
        path = anchor_dir / scenario_id / f"{camera}.png"
    if not path.exists():
        return None
    from kitscenes.video_engine import _decode_frame

    return _decode_frame(path)


def refine_file(
    source: Path,
    destination: Path,
    anchor: Optional[np.ndarray],
    encoder_crf: int,
    sharpen: float,
    equalize: float,
) -> Path:
    frames = decode_clip(source)
    if frames.shape[0] != spec.FRAME_COUNT:
        raise ValueError(f"{source} has {frames.shape[0]} frames, expected {spec.FRAME_COUNT}")
    refined = refine.refine_clip(
        frames, anchor=anchor, sharpen_amount=sharpen, equalize_strength=equalize
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    VideoEncoder(crf=encoder_crf).encode(refined, destination, validate=True)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Photometrically refine rendered surround clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--renders", type=Path, required=True, help="Directory of <id>/<camera>.mp4")
    parser.add_argument("--out-dir", type=Path, required=True, help="Where refined clips go")
    parser.add_argument("--anchor-dir", type=Path, default=None, help="Observed t=0 frames")
    parser.add_argument("--scenarios", type=str, default=None, help="Comma-separated ids")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N scenarios")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel workers")
    parser.add_argument("--crf", type=int, default=18, help="H.264 CRF of the refined clips")
    parser.add_argument("--sharpen", type=float, default=0.30, help="Unsharp amount")
    parser.add_argument("--equalize", type=float, default=0.5, help="Grey-world correction")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    targets: List[str]
    if args.scenarios:
        targets = [part.strip().zfill(7) for part in args.scenarios.split(",") if part.strip()]
    else:
        targets = sorted({path.name for path in args.renders.iterdir() if path.is_dir()})
    if args.limit:
        targets = targets[: args.limit]
    if not targets:
        print(f"error: no scenario directories under {args.renders}", file=sys.stderr)
        return 1

    jobs = []
    for scenario_id in targets:
        for camera in spec.CAMERAS:
            source = args.renders / scenario_id / f"{camera}.mp4"
            if not source.exists():
                print(f"error: missing {source}", file=sys.stderr)
                return 1
            jobs.append((source, args.out_dir / scenario_id / f"{camera}.mp4", scenario_id, camera))

    print(f"refining {len(jobs)} clips from {len(targets)} scenario(s) with {args.jobs} worker(s)")

    def run(job) -> str:
        source, destination, scenario_id, camera = job
        anchor = load_anchor(args.anchor_dir, scenario_id, camera)
        refine_file(source, destination, anchor, args.crf, args.sharpen, args.equalize)
        return f"{scenario_id}/{camera}"

    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for index, _result in enumerate(pool.map(run, jobs), start=1):
            done += 1
            if index % 50 == 0 or index == len(jobs):
                print(f"  {index}/{len(jobs)} clips refined")

    print(f"done: {done} clips -> {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
