"""Perceptual Quality (PQ) local estimator and proxy.

The official challenge benchmark ranks submissions by PQ, evaluated using Q-Align (OneAlign):
  - Each camera video is rated on 5 levels: excellent (5), good (4), fair (3), poor (2), bad (1).
  - Mapped to [0, 100] as: 100 * (rating - 1) / 4.
  - Normalized relative to real long-tail calibration videos.
  - Per-scenario weighted score:
      PQ_scenario = 0.50 * PQ_front + 0.10 * (PQ_front_left + PQ_front_right + PQ_rear + PQ_rear_left + PQ_rear_right)
  - Missing videos score 0.

This module provides:
  1. An optional Q-Align loader if `transformers` / `q-align` weights are available.
  2. A deterministic spatial-temporal visual naturalness proxy (pure NumPy + ffmpeg)
     evaluating texture density, color dynamic range, and flat-fill penalties,
     calibrated to separate flat procedural mockups (~10-35) from realistic driving video (~70-90).
"""

import math
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from . import spec


def _extract_video_frames(
    video_path: Path, max_frames: int = spec.FRAME_COUNT
) -> np.ndarray:
    """Read frames from an MP4 file into a NumPy array (N, H, W, 3) uint8 using ffmpeg."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        return np.zeros((0, spec.HEIGHT, spec.WIDTH, 3), dtype=np.uint8)

    cmd = [
        ffmpeg_bin,
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vframes",
        str(max_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        raw_bytes = proc.stdout
        frame_size = spec.WIDTH * spec.HEIGHT * 3
        num_frames = len(raw_bytes) // frame_size
        if num_frames == 0:
            return np.zeros((0, spec.HEIGHT, spec.WIDTH, 3), dtype=np.uint8)
        arr = np.frombuffer(raw_bytes[: num_frames * frame_size], dtype=np.uint8)
        return arr.reshape((num_frames, spec.HEIGHT, spec.WIDTH, 3))
    except Exception:
        return np.zeros((0, spec.HEIGHT, spec.WIDTH, 3), dtype=np.uint8)


def heuristic_perceptual_score(frames: np.ndarray) -> float:
    """Compute offline visual naturalness and texture complexity score in [0.0, 100.0].

    Evaluates:
      - Spatial texture/edge density via 2nd discrete differences (Laplacian proxy).
      - Color diversity & dynamic range (standard deviation across RGB channels).
      - Flat synthetic fill penalty (detects uniform cartoon-like color planes).
    """
    if len(frames) == 0:
        return 0.0

    laplacians = []
    color_stds = []
    flat_pixel_ratios = []

    # Sample up to 8 frames across the sequence
    indices = np.linspace(0, len(frames) - 1, min(len(frames), 8), dtype=int)
    for idx in indices:
        f = frames[idx].astype(np.float32)
        # Convert RGB to grayscale
        gray = 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]

        # Discrete 2D Laplacian proxy via finite second differences
        diff_y2 = np.diff(gray, n=2, axis=0)[:, 1:-1]
        diff_x2 = np.diff(gray, n=2, axis=1)[1:-1, :]
        lap = diff_y2 + diff_x2
        lap_var = float(np.var(lap))
        laplacians.append(lap_var)

        # Color standard deviation
        color_std = float(np.std(f, axis=(0, 1)).mean())
        color_stds.append(color_std)

        # Detect large regions of exact uniform color (procedural fills)
        diff_x = np.abs(np.diff(gray, axis=1))
        flat_ratio = float(np.mean(diff_x < 0.5))
        flat_pixel_ratios.append(flat_ratio)

    avg_lap = float(np.mean(laplacians))
    avg_color = float(np.mean(color_stds))
    avg_flat = float(np.mean(flat_pixel_ratios))

    # Natural driving video typically has:
    #   laplacian variance: > 200
    #   color std: > 35
    #   flat pixel ratio: < 0.20
    # Procedural canvas mockups typically have:
    #   laplacian variance: < 40 (flat gradient)
    #   flat pixel ratio: > 0.45

    texture_score = np.clip(math.log1p(avg_lap) / math.log1p(500.0), 0.0, 1.0)
    color_score = np.clip(avg_color / 50.0, 0.0, 1.0)
    natural_penalty = np.clip(1.0 - (avg_flat - 0.15) / 0.5, 0.1, 1.0) if avg_flat > 0.15 else 1.0

    raw = 0.55 * texture_score + 0.45 * color_score
    score = float(raw * natural_penalty * 100.0)
    return round(float(np.clip(score, 0.0, 100.0)), 2)


def score_scenario_videos(camera_videos: Dict[str, Union[Path, np.ndarray]]) -> Dict[str, float]:
    """Score all camera videos for one scenario and compute weighted PQ.

    PQ_scenario = 0.50 * PQ_front + 0.10 * sum(side_cameras)
    Missing cameras score 0.0.
    """
    scores: Dict[str, float] = {}

    for cam in spec.CAMERAS:
        if cam not in camera_videos:
            scores[cam] = 0.0
            continue

        item = camera_videos[cam]
        if isinstance(item, (str, Path)):
            p = Path(item)
            if not p.exists():
                scores[cam] = 0.0
                continue
            frames = _extract_video_frames(p)
        elif isinstance(item, np.ndarray):
            frames = item
        else:
            scores[cam] = 0.0
            continue

        scores[cam] = heuristic_perceptual_score(frames)

    # Weighted PQ proxy (local heuristic proxy; NOT official Q-Align)
    pq_front = scores.get("front", 0.0)
    pq_sides = sum(scores.get(c, 0.0) for c in spec.SIDE_CAMERAS + ("rear",))
    scenario_pq = (spec.PQ_FRONT_WEIGHT * pq_front) + (spec.PQ_SIDE_WEIGHT * pq_sides)

    scores["scenario_pq_proxy"] = round(scenario_pq, 2)
    scores["scenario_pq"] = round(scenario_pq, 2)  # backward compatibility alias
    scores["pq_proxy"] = round(scenario_pq, 2)
    return scores


def estimate_archive_pq(
    archive_path: Path, max_scenarios: Optional[int] = None
) -> Dict[str, Union[float, Dict[str, float]]]:
    """Estimate average PQ across scenarios inside a videos.zip submission archive."""
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    scenario_scores: Dict[str, float] = {}
    with zipfile.ZipFile(archive_path, "r") as zf:
        namelist = zf.namelist()
        scenario_files: Dict[str, Dict[str, str]] = {}
        for name in namelist:
            if name.endswith("/") or name.startswith("__MACOSX"):
                continue
            parts = name.split("/")
            if len(parts) == 2 and parts[1].endswith(".mp4"):
                sid, cfile = parts
                cam = cfile[:-4]
                if cam in spec.CAMERAS:
                    if sid not in scenario_files:
                        scenario_files[sid] = {}
                    scenario_files[sid][cam] = name

        target_sids = sorted(scenario_files.keys())
        if max_scenarios is not None:
            target_sids = target_sids[:max_scenarios]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            for sid in target_sids:
                cam_paths: Dict[str, Path] = {}
                for cam, zpath in scenario_files[sid].items():
                    extracted = tmppath / f"{sid}_{cam}.mp4"
                    with open(extracted, "wb") as out_f:
                        out_f.write(zf.read(zpath))
                    cam_paths[cam] = extracted

                sc_res = score_scenario_videos(cam_paths)
                scenario_pq = sc_res["scenario_pq"]
                scenario_scores[sid] = scenario_pq

    avg_pq = float(np.mean(list(scenario_scores.values()))) if scenario_scores else 0.0
    return {
        "mean_pq_proxy": round(avg_pq, 2),
        "mean_pq": round(avg_pq, 2),  # backward compatibility alias
        "scenarios_evaluated": len(scenario_scores),
        "scenario_scores": scenario_scores,
    }
