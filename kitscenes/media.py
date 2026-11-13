"""Video inspection through ffprobe.

The server does not decode submitted videos during upload validation; a scenario
whose video breaks the encoding contract simply scores 0 for PQ and EMC. These checks
exist to catch that locally, before the 3 GB upload and the 20-30 minute wait.
"""

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import List, Sequence, Tuple

from . import spec

STREAM_FIELDS = (
    "stream=codec_name,width,height,nb_frames,avg_frame_rate,r_frame_rate:format=duration"
)


class ProbeError(RuntimeError):
    """ffprobe is unavailable, or the file could not be read as a video."""


@dataclass(frozen=True)
class VideoInfo:
    codec: str
    width: int
    height: int
    frame_count: int
    fps: float

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0


def _ffprobe() -> str:
    binary = shutil.which("ffprobe")
    if binary is None:
        raise ProbeError(
            "ffprobe is not on PATH; install ffmpeg or pass --skip-media"
        )
    return binary


def _parse_rate(value: str) -> float:
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _run(args: Sequence[str]) -> dict:
    completed = subprocess.run(args, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        message = detail[-1] if detail else f"exit code {completed.returncode}"
        raise ProbeError(message)
    try:
        return json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise ProbeError(f"unreadable ffprobe output ({exc})") from exc


def _count_frames(path: Path) -> int:
    payload = _run(
        [
            _ffprobe(),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "json",
            str(path),
        ]
    )
    streams = payload.get("streams") or []
    if not streams:
        raise ProbeError(f"no video stream in {path.name}")
    return int(streams[0].get("nb_read_frames") or 0)


def probe_video(path: Path) -> VideoInfo:
    """Read codec, geometry, frame count and frame rate of the first video stream."""
    path = Path(path)
    payload = _run(
        [
            _ffprobe(),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            STREAM_FIELDS,
            "-of",
            "json",
            str(path),
        ]
    )
    streams = payload.get("streams") or []
    if not streams:
        raise ProbeError(f"no video stream in {path.name}")

    stream = streams[0]
    raw_frames = str(stream.get("nb_frames") or "")
    frame_count = int(raw_frames) if raw_frames.isdigit() else _count_frames(path)
    fps = _parse_rate(str(stream.get("avg_frame_rate") or "0/1"))
    if fps <= 0:
        fps = _parse_rate(str(stream.get("r_frame_rate") or "0/1"))

    return VideoInfo(
        codec=str(stream.get("codec_name") or ""),
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        frame_count=frame_count,
        fps=fps,
    )


def check_video(info: VideoInfo) -> Tuple[List[str], List[str]]:
    """Return ``(errors, warnings)`` for one video.

    Errors mark a video the scoring pipeline cannot use: those scenarios score 0 for
    both PQ and EMC. Warnings mark videos that will be scored but are likely to lose
    points, since EMC recovers ego-motion from the camera geometry.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if info.codec not in spec.ACCEPTED_CODECS:
        errors.append(f"codec is '{info.codec or 'unknown'}', expected H.264")
    if info.frame_count != spec.FRAME_COUNT:
        errors.append(
            f"{info.frame_count} frames, expected exactly {spec.FRAME_COUNT}"
        )
    if not math.isclose(info.fps, spec.FPS, rel_tol=0.0, abs_tol=spec.FPS_TOLERANCE):
        errors.append(f"{info.fps:g} fps, expected {spec.FPS:g} fps")

    if info.height:
        drift = abs(info.aspect_ratio - spec.ASPECT_RATIO) / spec.ASPECT_RATIO
        if drift > spec.ASPECT_RATIO_TOLERANCE:
            warnings.append(
                f"aspect ratio {info.aspect_ratio:.3f} differs from "
                f"{spec.ASPECT_RATIO:.3f} (876x568); EMC recovers ego-motion from "
                f"the camera geometry"
            )
        elif (info.width, info.height) != (spec.WIDTH, spec.HEIGHT):
            warnings.append(
                f"{info.width}x{info.height} instead of {spec.WIDTH}x{spec.HEIGHT}"
            )

    return errors, warnings


def describe(info: VideoInfo) -> str:
    """One-line summary, used in progress output and reports."""
    return (
        f"{info.width}x{info.height} {info.codec} "
        f"{info.frame_count}f @ {info.fps:g}fps"
    )
