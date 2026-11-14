"""Multi-view retrieval-assisted world model engine for KITScenes-LongTail.

This module provides a full 6-camera retrieval-assisted world model pipeline that
leverages real future driving footage from donor training/validation scenarios while
strictly enforcing:
1. Strict Single-Donor Policy:
   For any target scenario, exactly ONE donor training scenario is selected across all
   6 surround cameras (front, front_left, front_right, rear, rear_left, rear_right).
   Selecting different donors per camera destroys multi-view coherence and degrades EMC.
2. Multimodal Matching:
   Computes composite similarity score:
     score = w_visual * sim_visual + w_kinematics * sim_kin + w_instruction * sim_text + w_type * match_type
   evaluating visual features of observed t=0 frames, trajectory kinematics (mean speed,
   yaw change, delta trajectory curvature, acceleration), driving instructions, and
   scenario type metadata.
3. Seam Smoothing & Motion Alignment:
   Computes optical flow / motion alignment between the target's observed t=0 frame and
   the donor's future video, warping and cross-dissolving the first 3-5 frames into the
   donor trajectory to eliminate seam artifacts and visual pops.
"""

from __future__ import annotations

import io
import hashlib
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import spec

# Optional OpenCV support
try:
    import cv2  # type: ignore[import-untyped]
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# Optional Pillow support
try:
    from PIL import Image  # type: ignore[import-untyped]
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class SingleDonorViolationError(ValueError):
    """Raised when cameras within a scenario are assigned different donor scenarios."""
    pass


# ---------------------------------------------------------------------------
# Image / Video IO Utilities
# ---------------------------------------------------------------------------

def read_image_to_numpy(
    source: Union[np.ndarray, Path, str, bytes, Any],
    width: int = spec.WIDTH,
    height: int = spec.HEIGHT,
) -> np.ndarray:
    """Convert an image source into an (H, W, 3) uint8 numpy array."""
    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.shape[0] != height or arr.shape[1] != width:
            return resize_array(arr, width, height)
        return arr.astype(np.uint8)

    if HAS_PIL and hasattr(source, "convert"):
        # Source is already a PIL Image
        rgb = source.convert("RGB")
        if rgb.size != (width, height):
            rgb = rgb.resize((width, height), Image.Resampling.BILINEAR)
        return np.array(rgb, dtype=np.uint8)

    # If it's a file path
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"image file not found: {p}")
        raw_bytes = p.read_bytes()
    elif isinstance(source, bytes):
        raw_bytes = source
    else:
        raise TypeError(f"unsupported image source type: {type(source)}")

    if HAS_PIL:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.Resampling.BILINEAR)
        return np.array(img, dtype=np.uint8)

    # Pure fallback using ffmpeg
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        proc = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-i",
                "-",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-",
            ],
            input=raw_bytes,
            stdout=subprocess.PIPE,
            check=True,
        )
        return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(height, width, 3)

    raise RuntimeError("neither Pillow nor ffmpeg is available to decode image bytes")


def resize_array(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an (H, W, C) numpy array to target (height, width, C)."""
    h, w, c = arr.shape
    if (h, w) == (height, width):
        return arr

    if HAS_OPENCV:
        return cv2.resize(arr, (width, height), interpolation=cv2.INTER_LINEAR)

    if HAS_PIL:
        img = Image.fromarray(arr)
        resized = img.resize((width, height), Image.Resampling.BILINEAR)
        return np.array(resized, dtype=np.uint8)

    # Pure-NumPy bilinear resizing fallback
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    map_x = grid_x * (float(w) / float(width))
    map_y = grid_y * (float(h) / float(height))
    return sample_bilinear(arr, map_x, map_y)


def decode_video_to_frames(
    video_source: Union[np.ndarray, Path, str],
    width: int = spec.WIDTH,
    height: int = spec.HEIGHT,
    frame_count: int = spec.FRAME_COUNT,
) -> np.ndarray:
    """Decode a video source into (frame_count, height, width, 3) uint8 numpy array."""
    if isinstance(video_source, np.ndarray):
        arr = video_source
        if arr.shape == (frame_count, height, width, 3):
            return arr.astype(np.uint8)
        if arr.ndim == 4:
            # Need resizing or frame count adjustment
            resized_frames = []
            for i in range(min(frame_count, len(arr))):
                resized_frames.append(resize_array(arr[i], width, height))
            while len(resized_frames) < frame_count:
                resized_frames.append(resized_frames[-1].copy())
            return np.stack(resized_frames, axis=0).astype(np.uint8)
        raise ValueError(f"unexpected video array shape {arr.shape}")

    path = Path(video_source)
    if not path.exists():
        raise FileNotFoundError(f"video file not found: {path}")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg binary is required to decode video files")

    proc = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    raw = proc.stdout
    expected_bytes_per_frame = width * height * 3
    total_frames = len(raw) // expected_bytes_per_frame
    if total_frames == 0:
        raise ValueError(f"no video frames decoded from {path}")

    frames = np.frombuffer(
        raw[: frame_count * expected_bytes_per_frame], dtype=np.uint8
    ).reshape(-1, height, width, 3)

    if len(frames) < frame_count:
        # Pad with last frame if needed
        pad_count = frame_count - len(frames)
        last_frame = frames[-1:] if len(frames) > 0 else np.zeros((1, height, width, 3), dtype=np.uint8)
        frames = np.concatenate([frames] + [last_frame] * pad_count, axis=0)

    return frames[:frame_count]


# ---------------------------------------------------------------------------
# Bilinear Sampling & Motion Alignment
# ---------------------------------------------------------------------------

def sample_bilinear(img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Sample an (H, W, C) image at floating point coordinates (map_x, map_y).

    Coordinates are clamped to boundaries to prevent artifacts.
    """
    h, w = img.shape[:2]
    c = img.shape[2] if img.ndim == 3 else 1

    x0 = np.floor(map_x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(map_y).astype(np.int32)
    y1 = y0 + 1

    x0_c = np.clip(x0, 0, w - 1)
    x1_c = np.clip(x1, 0, w - 1)
    y0_c = np.clip(y0, 0, h - 1)
    y1_c = np.clip(y1, 0, h - 1)

    wx = (map_x - x0)
    wy = (map_y - y0)
    if c > 1:
        wx = wx[..., None]
        wy = wy[..., None]

    ia = img[y0_c, x0_c]
    ib = img[y0_c, x1_c]
    ic = img[y1_c, x0_c]
    id_ = img[y1_c, x1_c]

    top = (1.0 - wx) * ia + wx * ib
    bot = (1.0 - wx) * ic + wx * id_
    out = (1.0 - wy) * top + wy * bot
    return np.clip(out, 0, 255).astype(np.uint8)


def estimate_optical_flow(
    target_img: np.ndarray, donor_img: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute dense optical flow displacement field (u, v) from donor to target.

    Positive u: donor coordinate needs +u to reach target.
    Returns:
        (u, v): each of shape (H, W), dtype float32.
    """
    h, w = target_img.shape[:2]

    if HAS_OPENCV:
        gray_target = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)
        gray_donor = cv2.cvtColor(donor_img, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray_donor,
            gray_target,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        return flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32)

    # Robust multi-band pure-NumPy flow estimator
    lum_t = (
        0.299 * target_img[..., 0] + 0.587 * target_img[..., 1] + 0.114 * target_img[..., 2]
    ).astype(np.float32)
    lum_d = (
        0.299 * donor_img[..., 0] + 0.587 * donor_img[..., 1] + 0.114 * donor_img[..., 2]
    ).astype(np.float32)

    bands = [
        (0, int(0.35 * h)),                 # sky & distant features
        (int(0.35 * h), int(0.60 * h)),     # horizon & vanishing region
        (int(0.60 * h), int(0.85 * h)),     # mid-road & roadside
        (int(0.85 * h), h),                 # near hood & asphalt
    ]

    u_field = np.zeros((h, w), dtype=np.float32)
    v_field = np.zeros((h, w), dtype=np.float32)

    for y0, y1 in bands:
        sub_t = lum_t[y0:y1, :]
        sub_d = lum_d[y0:y1, :]

        # Subsample for speed
        step_y = max(1, (y1 - y0) // 16)
        step_x = max(1, w // 32)
        s_t = sub_t[::step_y, ::step_x]
        s_d = sub_d[::step_y, ::step_x]

        best_err = float("inf")
        best_dy, best_dx = 0, 0
        for dy in range(-2, 3):
            for dx in range(-3, 4):
                shifted = np.roll(np.roll(s_d, dy, axis=0), dx, axis=1)
                err = float(np.mean(np.abs(s_t - shifted)))
                if err < best_err:
                    best_err = err
                    best_dy = dy * step_y
                    best_dx = dx * step_x

        u_field[y0:y1, :] = float(best_dx)
        v_field[y0:y1, :] = float(best_dy)

    return u_field, v_field


def smooth_seam_transition(
    target_t0: np.ndarray,
    donor_frames: np.ndarray,
    blend_frames: int = 4,
) -> np.ndarray:
    """Seamlessly blend target t=0 frame into donor future video over the first K frames.

    Args:
        target_t0: (H, W, 3) target frame observed at t=0.
        donor_frames: (25, H, W, 3) donor future video (t = 0.2s to 5.0s).
        blend_frames: number of transition frames (3 to 5 frames).

    Returns:
        (25, H, W, 3) seamless uint8 video array.
    """
    h, w, c = target_t0.shape
    num_frames = len(donor_frames)
    blend_count = max(1, min(blend_frames, num_frames, 6))

    donor_anchor = donor_frames[0]
    u_flow, v_flow = estimate_optical_flow(target_t0, donor_anchor)

    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)

    # Compute color statistics for lighting / exposure matching
    mean_t = target_t0.mean(axis=(0, 1), keepdims=True)
    std_t = np.maximum(target_t0.std(axis=(0, 1), keepdims=True), 1.0)
    mean_d = donor_anchor.mean(axis=(0, 1), keepdims=True)
    std_d = np.maximum(donor_anchor.std(axis=(0, 1), keepdims=True), 1.0)

    result = donor_frames.copy()

    for k in range(blend_count):
        # Progress parameter in (0, 1)
        s = float(k + 1) / float(blend_count + 1)
        # Smoothstep Hermite polynomial: 3*s^2 - 2*s^3
        donor_weight = 3.0 * (s**2) - 2.0 * (s**3)
        target_weight = 1.0 - donor_weight

        # Motion warp target toward donor geometry
        warp_alpha = 1.0 - s
        map_x = grid_x + warp_alpha * u_flow
        map_y = grid_y + warp_alpha * v_flow

        if HAS_OPENCV:
            warped_target = cv2.remap(
                target_t0, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            )
        else:
            warped_target = sample_bilinear(target_t0, map_x, map_y)

        # Color balance target towards donor
        warped_normalized = (warped_target.astype(np.float32) - mean_t) / std_t
        adjusted_target = warped_normalized * (
            (1.0 - s) * std_t + s * std_d
        ) + ((1.0 - s) * mean_t + s * mean_d)
        adjusted_target = np.clip(adjusted_target, 0, 255)

        # Cross-dissolve
        donor_curr = donor_frames[k].astype(np.float32)
        blended = target_weight * adjusted_target + donor_weight * donor_curr
        result[k] = np.clip(blended, 0, 255).astype(np.uint8)

    return result


# ---------------------------------------------------------------------------
# Multimodal Feature Extractors
# ---------------------------------------------------------------------------

class VisualFeatureExtractor:
    """Extracts lightweight normalized visual descriptors from t=0 camera observations."""

    def __init__(self, feature_dim: int = 156):
        self.feature_dim = feature_dim

    def extract_single_frame(self, frame: np.ndarray) -> np.ndarray:
        """Extract a 156-dim normalized spatial-chromatic-gradient descriptor."""
        if frame.ndim == 4:
            frame = frame[0]
        h, w, c = frame.shape
        bh, bw = max(1, h // 32), max(1, w // 32)
        trimmed = frame[: bh * 32, : bw * 32, :]
        down = (
            trimmed.reshape(32, bh, 32, bw, c)
            .mean(axis=(1, 3))
            .astype(np.float32)
            / 255.0
        )

        # 4x4 spatial blocks of means and standard deviations
        blocks = down.reshape(4, 8, 4, 8, 3)
        grid_means = blocks.mean(axis=(1, 3)).flatten()
        grid_stds = blocks.std(axis=(1, 3)).flatten()

        # Directional gradients with boundary padding
        dx = np.abs(down[:, 1:, :] - down[:, :-1, :]).mean(axis=-1)
        dy = np.abs(down[1:, :, :] - down[:-1, :, :]).mean(axis=-1)
        dx_p = np.pad(dx, ((0, 0), (0, 1)))
        dy_p = np.pad(dy, ((0, 1), (0, 0)))
        dx_pool = dx_p.reshape(4, 8, 4, 8).mean(axis=(1, 3)).flatten()
        dy_pool = dy_p.reshape(4, 8, 4, 8).mean(axis=(1, 3)).flatten()

        # Color histograms (8 bins per channel in [0, 1])
        hist_r, _ = np.histogram(down[..., 0], bins=8, range=(0.0, 1.0))
        hist_g, _ = np.histogram(down[..., 1], bins=8, range=(0.0, 1.0))
        hist_b, _ = np.histogram(down[..., 2], bins=8, range=(0.0, 1.0))
        hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist /= hist_sum

        # Luminance stats
        lum = 0.299 * down[..., 0] + 0.587 * down[..., 1] + 0.114 * down[..., 2]
        lum_stats = np.array(
            [lum.mean(), lum.std(), np.percentile(lum, 10), np.percentile(lum, 90)],
            dtype=np.float32,
        )

        feat = np.concatenate([grid_means, grid_stds, dx_pool, dy_pool, hist, lum_stats])
        norm = float(np.linalg.norm(feat))
        if norm > 1e-6:
            feat /= norm
        return feat

    def extract(
        self,
        observations: Union[np.ndarray, Dict[str, Any], Path, str, bytes],
    ) -> np.ndarray:
        """Extract multi-camera or single-camera normalized visual embedding."""
        if isinstance(observations, dict):
            # Surround rig camera weights: front and rear carry primary heading cues
            cam_weights = {
                "front": 0.40,
                "rear": 0.20,
                "front_left": 0.10,
                "front_right": 0.10,
                "rear_left": 0.10,
                "rear_right": 0.10,
            }
            sub_feats: List[np.ndarray] = []
            weights: List[float] = []
            for cam, wt in cam_weights.items():
                if cam in observations:
                    arr = read_image_to_numpy(observations[cam])
                    sub_feats.append(self.extract_single_frame(arr))
                    weights.append(wt)
            if not sub_feats:
                return np.zeros(self.feature_dim, dtype=np.float32)
            w_arr = np.array(weights, dtype=np.float32)
            w_arr /= w_arr.sum()
            combined = sum(f * w for f, w in zip(sub_feats, w_arr))
            norm = float(np.linalg.norm(combined))
            return combined / norm if norm > 1e-6 else combined

        arr = read_image_to_numpy(observations)
        return self.extract_single_frame(arr)

    def similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine similarity in [0, 1]."""
        n1 = float(np.linalg.norm(feat1))
        n2 = float(np.linalg.norm(feat2))
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.5  # Neutral fallback when visual feature is missing
        sim = float(np.dot(feat1, feat2) / (n1 * n2))
        return max(0.0, min(1.0, sim))


class KinematicsFeatureExtractor:
    """Extracts speed, yaw, curvature, and acceleration features from 25 waypoints."""

    def __init__(self, dt: float = spec.WAYPOINT_STEP_S):
        self.dt = dt

    def extract(self, trajectory: Sequence[Sequence[float]]) -> Dict[str, Any]:
        """Analyze 25 (x, y) coordinates into kinematic feature metrics."""
        pts = np.asarray(trajectory, dtype=np.float64)
        if pts.shape != (spec.WAYPOINT_COUNT, 2):
            raise ValueError(
                f"expected ({spec.WAYPOINT_COUNT}, 2) trajectory waypoints, got {pts.shape}"
            )

        # Sequence including origin (0, 0) at t=0
        all_pts = np.vstack([[0.0, 0.0], pts])
        deltas = np.diff(all_pts, axis=0)
        displacements = np.linalg.norm(deltas, axis=1)
        speeds = displacements / self.dt

        # Heading determination
        headings = np.zeros(spec.WAYPOINT_COUNT, dtype=np.float64)
        curr_heading = 0.0
        for i in range(spec.WAYPOINT_COUNT):
            if displacements[i] > 1e-4:
                curr_heading = math.atan2(deltas[i, 1], deltas[i, 0])
            headings[i] = curr_heading

        all_headings = np.concatenate([[0.0], headings])
        heading_diffs = np.zeros(spec.WAYPOINT_COUNT, dtype=np.float64)
        for i in range(spec.WAYPOINT_COUNT):
            diff = (all_headings[i + 1] - all_headings[i] + math.pi) % (2.0 * math.pi) - math.pi
            heading_diffs[i] = diff
        yaw_rates = heading_diffs / self.dt

        # Curvatures kappa = |yaw_rate| / max(speed, 0.5)
        curvatures = np.abs(yaw_rates) / np.maximum(speeds, 0.5)

        # Accelerations
        all_speeds = np.concatenate([[speeds[0]], speeds])
        accelerations = np.diff(all_speeds) / self.dt

        total_yaw_change = float(
            (headings[-1] - 0.0 + math.pi) % (2.0 * math.pi) - math.pi
        )

        return {
            "mean_speed": float(np.mean(speeds)),
            "yaw_change": total_yaw_change,
            "mean_curvature": float(np.mean(curvatures)),
            "delta_curvature": float(np.ptp(curvatures)),
            "mean_accel": float(np.mean(accelerations)),
            "speeds": speeds,
            "headings": headings,
            "waypoints": pts,
        }

    def similarity(self, k1: Dict[str, Any], k2: Dict[str, Any]) -> float:
        """Compute kinematic agreement similarity in [0, 1]."""
        # Speed similarity
        speed_diff = abs(k1["mean_speed"] - k2["mean_speed"])
        sim_speed = math.exp(-speed_diff / 5.0)

        # Yaw change similarity (signed yaw angle agreement)
        yaw_diff = abs(k1["yaw_change"] - k2["yaw_change"])
        sim_yaw = math.exp(-yaw_diff / 0.35)

        # Curvature similarity
        curv_diff = abs(k1["mean_curvature"] - k2["mean_curvature"])
        sim_curv = math.exp(-curv_diff / 0.06)

        # Trajectory coordinates Euclidean displacement
        pts1 = k1["waypoints"]
        pts2 = k2["waypoints"]
        traj_dist = float(np.mean(np.linalg.norm(pts1 - pts2, axis=1)))
        sim_traj = math.exp(-traj_dist / 6.0)

        # Acceleration similarity
        acc_diff = abs(k1["mean_accel"] - k2["mean_accel"])
        sim_acc = math.exp(-acc_diff / 2.0)

        sim_kin = (
            0.30 * sim_speed
            + 0.30 * sim_yaw
            + 0.15 * sim_curv
            + 0.15 * sim_traj
            + 0.10 * sim_acc
        )
        return max(0.0, min(1.0, float(sim_kin)))


class InstructionTextExtractor:
    """Semantic driving instruction embedding and intent similarity matcher."""

    DRIVING_KEYWORDS: List[str] = [
        "straight", "ahead", "forward", "continue",
        "left", "right", "turn", "steer", "swerve",
        "overtake", "pass", "lane", "change", "shift", "merge",
        "stop", "halt", "brake", "slow", "decelerate",
        "accelerate", "speed", "fast", "u-turn", "roundabout",
        "intersection", "junction", "traffic", "light", "signal",
        "pedestrian", "truck", "bus", "car", "vehicle",
        "rain", "night", "highway", "urban", "curve", "bend",
    ]

    def __init__(self, embed_dim: int = 64):
        self.embed_dim = embed_dim
        self.keyword_map = {kw: i for i, kw in enumerate(self.DRIVING_KEYWORDS)}

    def embed(self, text: str) -> np.ndarray:
        """Embed driving instruction into normalized semantic vector."""
        vec = np.zeros(self.embed_dim, dtype=np.float32)
        words = re.findall(r"\b[a-z0-9\-]+\b", (text or "").lower())
        if not words:
            return vec

        for w in words:
            if w in self.keyword_map:
                vec[self.keyword_map[w]] += 2.0
            else:
                # Use deterministic SHA-256 hash instead of built-in hash() which is process-randomized
                bucket = int(hashlib.sha256(w.encode("utf-8")).hexdigest(), 16)
                idx = len(self.DRIVING_KEYWORDS) + (bucket % (self.embed_dim - len(self.DRIVING_KEYWORDS)))
                vec[idx] += 1.0

        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            vec /= norm
        return vec

    def similarity(self, text1: str, text2: str) -> float:
        """Compute semantic instruction similarity in [0, 1] with directional contrast."""
        t1 = (text1 or "").strip().lower()
        t2 = (text2 or "").strip().lower()

        if not t1 and not t2:
            return 1.0
        if not t1 or not t2:
            return 0.5

        if t1 == t2:
            return 1.0

        v1 = self.embed(t1)
        v2 = self.embed(t2)
        base_sim = float(np.dot(v1, v2))

        # Directional contrast penalty: opposing lateral maneuvers (left vs right)
        w1 = set(re.findall(r"\b[a-z]+\b", t1))
        w2 = set(re.findall(r"\b[a-z]+\b", t2))
        has_l1, has_r1 = "left" in w1, "right" in w1
        has_l2, has_r2 = "left" in w2, "right" in w2

        penalty = 1.0
        if (has_l1 and not has_r1 and has_r2 and not has_l2) or (
            has_r1 and not has_l1 and has_l2 and not has_r2
        ):
            penalty = 0.35

        return max(0.0, min(1.0, base_sim * penalty))


class ScenarioTypeMatcher:
    """Matches scenario type metadata (e.g. '5 nighttime', 'intersection', 'overtake')."""

    LIGHTING_TAGS = frozenset({"night", "nighttime", "dark", "dusk", "day", "daytime", "sun", "clear"})
    WEATHER_TAGS = frozenset({"rain", "rainy", "wet", "fog", "foggy", "snow"})
    ROAD_TAGS = frozenset({"intersection", "junction", "overtake", "highway", "urban", "curve"})

    def match(self, type1: str, type2: str) -> float:
        """Compute scenario type match score in [0, 1]."""
        t1 = (type1 or "").strip().lower()
        t2 = (type2 or "").strip().lower()

        if t1 == t2:
            return 1.0

        # Unconstrained wildcard scenarios in KITScenes (e.g. '0 specifically selected')
        if "specifically selected" in t1 or "specifically selected" in t2 or not t1 or not t2:
            return 0.70

        words1 = set(re.findall(r"[a-z]+", t1))
        words2 = set(re.findall(r"[a-z]+", t2))

        # Check lighting conflict (night vs day)
        night1 = bool(words1 & {"night", "nighttime", "dark"})
        night2 = bool(words2 & {"night", "nighttime", "dark"})
        if night1 != night2 and (night1 or night2):
            return 0.10

        # Jaccard overlap on recognized domain tokens
        union = words1 | words2
        if not union:
            return 0.5
        overlap = words1 & words2
        return float(len(overlap) / len(union))


# ---------------------------------------------------------------------------
# Data Models: Target & Donor Scenarios
# ---------------------------------------------------------------------------

@dataclass
class TargetScenario:
    """Target scenario requiring video generation."""
    scenario_id: str
    future_trajectory: Sequence[Sequence[float]]  # 25 [x, y] coordinates
    driving_instruction: str = ""
    scenario_type: str = ""
    observed_t0_frames: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DonorScenario:
    """Training or validation donor scenario providing real future video footage."""
    scenario_id: str
    future_trajectory: Sequence[Sequence[float]]  # 25 [x, y] coordinates
    driving_instruction: str = ""
    scenario_type: str = ""
    observed_t0_frames: Optional[Dict[str, Any]] = None
    future_videos: Optional[Dict[str, Any]] = None  # map camera -> array (25, 568, 876, 3) or path
    metadata: Optional[Dict[str, Any]] = None
    visual_features: Optional[np.ndarray] = None
    kinematic_features: Optional[Dict[str, Any]] = None
    text_features: Optional[np.ndarray] = None


@dataclass
class RankedDonor:
    """Ranked donor scenario with similarity scores."""
    donor: DonorScenario
    composite_score: float
    similarity_breakdown: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Multimodal Ranker
# ---------------------------------------------------------------------------

class MultimodalRanker:
    """Ranks donor scenarios for a target scenario using composite multimodal scoring."""

    def __init__(
        self,
        w_visual: float = 0.30,
        w_kinematics: float = 0.35,
        w_instruction: float = 0.20,
        w_type: float = 0.15,
    ):
        total_w = w_visual + w_kinematics + w_instruction + w_type
        if total_w <= 0:
            raise ValueError("sum of ranker weights must be strictly positive")
        self.w_visual = w_visual / total_w
        self.w_kinematics = w_kinematics / total_w
        self.w_instruction = w_instruction / total_w
        self.w_type = w_type / total_w

        self.visual_extractor = VisualFeatureExtractor()
        self.kinematics_extractor = KinematicsFeatureExtractor()
        self.text_extractor = InstructionTextExtractor()
        self.type_matcher = ScenarioTypeMatcher()

    def score_pair(self, target: TargetScenario, donor: DonorScenario) -> RankedDonor:
        """Score single (target, donor) pair across all 4 multimodal criteria."""
        # 1. Visual similarity
        sim_vis = 0.5
        if target.observed_t0_frames and donor.observed_t0_frames:
            target_feat = self.visual_extractor.extract(target.observed_t0_frames)
            donor_feat = donor.visual_features
            if donor_feat is None:
                donor_feat = self.visual_extractor.extract(donor.observed_t0_frames)
            sim_vis = self.visual_extractor.similarity(target_feat, donor_feat)
        elif donor.visual_features is not None and target.observed_t0_frames:
            target_feat = self.visual_extractor.extract(target.observed_t0_frames)
            sim_vis = self.visual_extractor.similarity(target_feat, donor.visual_features)

        # 2. Kinematics similarity
        target_kin = self.kinematics_extractor.extract(target.future_trajectory)
        donor_kin = donor.kinematic_features
        if donor_kin is None:
            donor_kin = self.kinematics_extractor.extract(donor.future_trajectory)
        sim_kin = self.kinematics_extractor.similarity(target_kin, donor_kin)

        # 3. Instruction similarity
        sim_text = self.text_extractor.similarity(
            target.driving_instruction, donor.driving_instruction
        )

        # 4. Scenario type match
        match_type = self.type_matcher.match(target.scenario_type, donor.scenario_type)

        composite = (
            self.w_visual * sim_vis
            + self.w_kinematics * sim_kin
            + self.w_instruction * sim_text
            + self.w_type * match_type
        )

        breakdown = {
            "sim_visual": sim_vis,
            "sim_kinematics": sim_kin,
            "sim_instruction": sim_text,
            "sim_type": match_type,
        }

        return RankedDonor(donor=donor, composite_score=composite, similarity_breakdown=breakdown)

    def rank(
        self,
        target: TargetScenario,
        candidates: Sequence[DonorScenario],
        top_k: int = 1,
    ) -> List[RankedDonor]:
        """Rank donor candidates descending by composite similarity score."""
        if not candidates:
            raise ValueError("candidates sequence is empty")

        ranked = [self.score_pair(target, d) for d in candidates]
        ranked.sort(key=lambda r: r.composite_score, reverse=True)
        return ranked[:top_k]

    def select_best(
        self,
        target: TargetScenario,
        candidates: Sequence[DonorScenario],
    ) -> RankedDonor:
        """Select single highest-scoring donor candidate."""
        results = self.rank(target, candidates, top_k=1)
        return results[0]


# ---------------------------------------------------------------------------
# Donor Index
# ---------------------------------------------------------------------------

class DonorIndex:
    """Manages the indexed database of donor training scenarios."""

    def __init__(self, donors: Optional[Sequence[DonorScenario]] = None):
        self._donors: Dict[str, DonorScenario] = {}
        if donors:
            for d in donors:
                self.add_donor(d)

    def add_donor(self, donor: DonorScenario) -> None:
        """Add a donor scenario to index and pre-compute cached descriptors."""
        if donor.kinematic_features is None:
            donor.kinematic_features = KinematicsFeatureExtractor().extract(
                donor.future_trajectory
            )
        if donor.text_features is None:
            donor.text_features = InstructionTextExtractor().embed(donor.driving_instruction)
        if donor.visual_features is None and donor.observed_t0_frames:
            try:
                donor.visual_features = VisualFeatureExtractor().extract(
                    donor.observed_t0_frames
                )
            except Exception:
                pass
        self._donors[donor.scenario_id] = donor

    def get_donor(self, scenario_id: str) -> DonorScenario:
        """Look up donor by ID."""
        return self._donors[scenario_id]

    def __len__(self) -> int:
        return len(self._donors)

    def __iter__(self):
        return iter(self._donors.values())

    def values(self) -> Sequence[DonorScenario]:
        return list(self._donors.values())

    @classmethod
    def create_default_bank(cls) -> DonorIndex:
        """Create a diverse default bank of canonical driving maneuver donor scenarios."""
        index = cls()

        def make_trajectory(
            speed_mps: float, yaw_rate_rps: float, curve_type: str = "straight"
        ) -> List[List[float]]:
            dt = spec.WAYPOINT_STEP_S
            waypoints = []
            x, y, heading = 0.0, 0.0, 0.0
            for k in range(spec.WAYPOINT_COUNT):
                if curve_type == "straight":
                    w = 0.0
                elif curve_type == "steady_turn":
                    w = yaw_rate_rps
                elif curve_type == "overtake_s_curve":
                    # S-curve: steer left for first 1.5s, steer right for next 1.5s, then straight
                    t = (k + 1) * dt
                    if t < 1.6:
                        w = yaw_rate_rps
                    elif t < 3.2:
                        w = -yaw_rate_rps
                    else:
                        w = 0.0
                elif curve_type == "decelerate":
                    speed_mps = max(0.0, speed_mps - 0.5 * dt * 10.0)
                    w = 0.0
                else:
                    w = yaw_rate_rps

                heading += w * dt
                x += speed_mps * dt * math.cos(heading)
                y += speed_mps * dt * math.sin(heading)
                waypoints.append([round(x, 3), round(y, 3)])
            return waypoints

        # 1. Straight cruising highway (12 m/s)
        index.add_donor(
            DonorScenario(
                scenario_id="donor_highway_straight",
                future_trajectory=make_trajectory(12.0, 0.0, "straight"),
                driving_instruction="drive straight on",
                scenario_type="highway",
            )
        )
        # 2. Straight urban cruise (6 m/s)
        index.add_donor(
            DonorScenario(
                scenario_id="donor_urban_straight",
                future_trajectory=make_trajectory(6.0, 0.0, "straight"),
                driving_instruction="keep in lane and follow traffic ahead",
                scenario_type="urban",
            )
        )
        # 3. Left turn intersection (8 m/s, +0.20 rad/s)
        index.add_donor(
            DonorScenario(
                scenario_id="donor_left_turn_intersection",
                future_trajectory=make_trajectory(8.0, 0.20, "steady_turn"),
                driving_instruction="turn left at the intersection",
                scenario_type="intersection",
            )
        )
        # 4. Right turn intersection (7 m/s, -0.22 rad/s)
        index.add_donor(
            DonorScenario(
                scenario_id="donor_right_turn_intersection",
                future_trajectory=make_trajectory(7.0, -0.22, "steady_turn"),
                driving_instruction="turn right at the intersection",
                scenario_type="intersection",
            )
        )
        # 5. Overtake maneuver (14 m/s S-curve)
        index.add_donor(
            DonorScenario(
                scenario_id="donor_overtake_truck",
                future_trajectory=make_trajectory(14.0, 0.15, "overtake_s_curve"),
                driving_instruction="overtake truck driving on the right",
                scenario_type="overtake",
            )
        )
        # 6. Nighttime straight cruise (13 m/s)
        index.add_donor(
            DonorScenario(
                scenario_id="donor_nighttime_straight",
                future_trajectory=make_trajectory(13.0, 0.0, "straight"),
                driving_instruction="drive straight on at night",
                scenario_type="5 nighttime",
            )
        )
        # 7. Deceleration to stop at traffic light
        index.add_donor(
            DonorScenario(
                scenario_id="donor_stop_traffic_light",
                future_trajectory=make_trajectory(10.0, 0.0, "decelerate"),
                driving_instruction="slow down and stop at traffic light",
                scenario_type="intersection",
            )
        )
        # 8. Rain curve
        index.add_donor(
            DonorScenario(
                scenario_id="donor_rain_curve",
                future_trajectory=make_trajectory(9.0, 0.10, "steady_turn"),
                driving_instruction="follow curved road carefully in rain",
                scenario_type="rain",
            )
        )

        return index

    @classmethod
    def load_from_jsonl(
        cls,
        metadata_path: Union[Path, str],
        video_dir: Optional[Union[Path, str]] = None,
        t0_dir: Optional[Union[Path, str]] = None,
        limit: Optional[int] = None,
    ) -> DonorIndex:
        """Load donor scenarios from a KITScenes metadata JSONL file."""
        metadata_path = Path(metadata_path)
        index = cls()
        count = 0

        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                sc_id = str(data.get("scenario_id", f"{count:07d}")).zfill(7)

                # Trajectory: look for expert_like or future_trajectory
                traj_info = data.get("trajectory", {})
                future_pts = traj_info.get("expert_like") or data.get("future_trajectory")
                if not future_pts or np.allclose(np.array(future_pts), -100.0):
                    # Redacted test split; cannot serve as donor future
                    continue

                instruction = data.get("driving_instruction") or data.get("instruction", "")
                sc_type = data.get("scenario_type", "")

                t0_dict = None
                if t0_dir is not None:
                    sc_t0 = Path(t0_dir) / sc_id
                    if sc_t0.exists():
                        t0_dict = {}
                        for cam in spec.CAMERAS:
                            img_p = sc_t0 / f"{cam}.jpg"
                            if not img_p.exists():
                                img_p = sc_t0 / f"{cam}.png"
                            if img_p.exists():
                                t0_dict[cam] = img_p

                vid_dict = None
                if video_dir is not None:
                    sc_vids = Path(video_dir) / sc_id
                    if sc_vids.exists():
                        vid_dict = {}
                        for cam in spec.CAMERAS:
                            vid_p = sc_vids / f"{cam}.mp4"
                            if vid_p.exists():
                                vid_dict[cam] = vid_p

                donor = DonorScenario(
                    scenario_id=sc_id,
                    future_trajectory=future_pts,
                    driving_instruction=instruction,
                    scenario_type=sc_type,
                    observed_t0_frames=t0_dict,
                    future_videos=vid_dict,
                )
                index.add_donor(donor)
                count += 1
                if limit and count >= limit:
                    break

        return index


# ---------------------------------------------------------------------------
# Aligned Video Synthesizer
# ---------------------------------------------------------------------------

class AlignedVideoSynthesizer:
    """Generates 6-camera surround video footage strictly sourced from ONE donor.

    Applies motion alignment and temporal smoothing across all 6 surround cameras
    to eliminate seam artifacts between the target t=0 frame and the donor's future video.
    """

    def __init__(self, blend_frames: int = 4):
        self.blend_frames = blend_frames
        self._cached_renderer = None

    def _get_fallback_renderer(self):
        """Lazy load geometric renderer to synthesize donor video when not pre-rendered."""
        if self._cached_renderer is None:
            from .video_engine import GeometricWorldModel
            self._cached_renderer = GeometricWorldModel()
        return self._cached_renderer

    def synthesize(
        self,
        target: TargetScenario,
        donor: DonorScenario,
        blend_frames: Optional[int] = None,
        camera_donors: Optional[Dict[str, DonorScenario]] = None,
    ) -> Dict[str, np.ndarray]:
        """Synthesize all 6 camera surround views for target using single chosen donor.

        Args:
            target: Target driving scenario.
            donor: The selected donor scenario supplying all 6 views.
            blend_frames: Number of transition frames for seam smoothing (3-5 frames).
            camera_donors: Validation check parameter. If passed, verifies that every
                camera view points to the EXACT same donor scenario. Any mismatch raises
                SingleDonorViolationError.

        Returns:
            Dict mapping each camera to (25, 568, 876, 3) uint8 numpy array.
        """
        # Strict Single-Donor Rule Enforcement:
        if camera_donors is not None:
            donor_ids = {cam: d.scenario_id for cam, d in camera_donors.items()}
            distinct_donors = set(donor_ids.values())
            if len(distinct_donors) > 1:
                raise SingleDonorViolationError(
                    f"Single-Donor Policy violation: cameras were assigned {len(distinct_donors)} "
                    f"different donors: {donor_ids}. Exactly ONE donor scenario must supply all "
                    f"6 surround camera views to guarantee 360° spatial-temporal coherence."
                )

        k_blend = blend_frames if blend_frames is not None else self.blend_frames

        # Fetch donor's 6 synchronized future camera videos
        donor_videos: Dict[str, np.ndarray] = {}
        if donor.future_videos:
            for cam in spec.CAMERAS:
                if cam in donor.future_videos:
                    donor_videos[cam] = decode_video_to_frames(donor.future_videos[cam])

        # If donor videos are not pre-rendered on disk, render them via geometric engine
        missing_cams = [cam for cam in spec.CAMERAS if cam not in donor_videos]
        if missing_cams:
            renderer = self._get_fallback_renderer()
            weather = "night" if "night" in donor.scenario_type.lower() else "clear"
            synth_vids = renderer.generate_scenario(
                donor.scenario_id, donor.future_trajectory, weather=weather
            )
            for cam in missing_cams:
                donor_videos[cam] = synth_vids[cam]

        # Apply motion-aligned seam smoothing per camera
        results: Dict[str, np.ndarray] = {}
        target_t0 = target.observed_t0_frames or {}

        for cam in spec.CAMERAS:
            donor_footage = donor_videos[cam]  # (25, 568, 876, 3)
            if cam in target_t0:
                t0_arr = read_image_to_numpy(target_t0[cam])
                smoothed_frames = smooth_seam_transition(
                    target_t0=t0_arr,
                    donor_frames=donor_footage,
                    blend_frames=k_blend,
                )
                results[cam] = smoothed_frames
            else:
                # No target t=0 frame supplied for this camera; use donor future footage
                results[cam] = donor_footage

            # Enforce exact output contract
            assert results[cam].shape == (spec.FRAME_COUNT, spec.HEIGHT, spec.WIDTH, 3)
            assert results[cam].dtype == np.uint8

        return results


# ---------------------------------------------------------------------------
# Retrieval World Model Generator (Pipeline Compatible)
# ---------------------------------------------------------------------------

class RetrievalWorldModel:
    """Retrieval-assisted world model implementing the standard generator contract.

    Compatible with WorldModelPipeline(backbone='retrieval') and batch generation.
    """

    def __init__(
        self,
        donor_index: Optional[DonorIndex] = None,
        ranker: Optional[MultimodalRanker] = None,
        synthesizer: Optional[AlignedVideoSynthesizer] = None,
        blend_frames: int = 4,
    ):
        self.donor_index = donor_index or DonorIndex.create_default_bank()
        self.ranker = ranker or MultimodalRanker()
        self.synthesizer = synthesizer or AlignedVideoSynthesizer(blend_frames=blend_frames)
        self.last_retrieval_result: Optional[RankedDonor] = None

    def generate_scenario(
        self,
        scenario_id: str,
        trajectory: Sequence[Sequence[float]],
        weather: str = "clear",
        conditioning_images: Optional[Dict[str, Any]] = None,
        driving_instruction: Optional[str] = None,
        scenario_type: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Generate all 6 camera views using the best-matching single donor scenario."""
        # Infer instruction / type if not explicitly supplied
        instruction = driving_instruction or ""
        sc_type = scenario_type or ("5 nighttime" if weather == "night" else "")

        target = TargetScenario(
            scenario_id=str(scenario_id).zfill(7),
            future_trajectory=trajectory,
            driving_instruction=instruction,
            scenario_type=sc_type,
            observed_t0_frames=conditioning_images,
        )

        # Multimodal retrieval: select exactly ONE best donor
        best_match = self.ranker.select_best(target, self.donor_index.values())
        self.last_retrieval_result = best_match

        # Synthesize 6 surround camera views strictly from this single donor
        return self.synthesizer.synthesize(target, best_match.donor)
