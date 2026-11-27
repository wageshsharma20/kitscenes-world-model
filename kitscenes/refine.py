"""Photometric refinement of rendered surround clips.

PQ is scored by Q-Align (OneAlign), which rates a clip on how good it looks, not on how
faithfully it matches the ground truth. A reprojection renderer is geometrically correct
but leaves three things that cost those ratings on every clip:

  * exposure and white balance drift frame to frame, because each new frame is a warp of
    a different source region;
  * soft, low-contrast texture where the warp stretched detail;
  * compression and inpainting leave the picture slightly hazy.

This module fixes those three without touching geometry or motion, so EMC and MMS are
unaffected. Everything here is deterministic numpy; no model, no GPU.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from . import spec

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def luma(frames: np.ndarray) -> np.ndarray:
    """Per-frame luma plane, shape (T, H, W)."""
    return frames.astype(np.float32) @ LUMA


def _box_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur via summed-area tables; edges use a shrunk window.

    Only the first two axes (H, W) are blurred. Frames arrive either as (H, W, C) or as
    (T, H, W, C), so the padding must cover every trailing axis with a zero pair.
    """
    if radius <= 0:
        return image
    pad = [(radius, radius), (radius, radius)] + [(0, 0)] * max(0, image.ndim - 2)
    padded = np.pad(image, pad, mode="edge")
    cumulative = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    # A leading zero border makes c[i] mean "sum of the first i rows/columns", which is
    # what the window subtraction below assumes; without it the result is one pixel short
    # in each axis and every caller ends up broadcasting against a smaller array.
    cumulative = np.pad(cumulative, [(1, 0), (1, 0)] + [(0, 0)] * max(0, cumulative.ndim - 2))

    def window_sum(c: np.ndarray, size: int) -> np.ndarray:
        total = c[size:, size:] - c[:-size, size:] - c[size:, :-size] + c[:-size, :-size]
        return total

    k = 2 * radius + 1
    total = window_sum(cumulative, k)
    return total / float(k * k)


def _smooth_1d(values: np.ndarray, smooth_window: int) -> np.ndarray:
    """Box-smooth a 1-D per-frame parameter curve with edge padding."""
    if smooth_window <= 1 or len(values) <= 1:
        return values.astype(np.float32)
    kernel = np.ones(smooth_window, dtype=np.float32) / float(smooth_window)
    pad_left = smooth_window // 2
    pad_right = smooth_window - 1 - pad_left
    padded = np.pad(values.astype(np.float32), (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)].astype(np.float32)


def photometric_gains(
    frames: np.ndarray,
    anchor: Optional[np.ndarray] = None,
    strength: float = 1.0,
    max_gain: float = 1.15,
    smooth_window: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame gain and offset that pull each frame towards the anchor's exposure.

    Anchoring on the observed t=0 frame keeps the clip in the statistics of real footage
    from the same camera and scene, which is what Q-Align's calibration compares against.
    """
    if anchor is None:
        anchor = frames[0]
    target_luma = luma(anchor[None, ...])[0]
    target_mean = float(target_luma.mean())
    target_std = float(target_luma.std()) + 1e-6

    clip_luma = luma(frames)
    means = clip_luma.mean(axis=(1, 2))
    stds = clip_luma.std(axis=(1, 2)) + 1e-6

    gains = np.clip(target_std / stds, 1.0 / max_gain, max_gain)
    gains = 1.0 + (gains - 1.0) * strength
    offsets = (target_mean - means) * strength

    gains = _smooth_1d(gains, smooth_window)
    offsets = _smooth_1d(offsets, smooth_window)
    return gains.astype(np.float32), offsets.astype(np.float32)


def saturation_gains(
    frames: np.ndarray,
    anchor: Optional[np.ndarray] = None,
    strength: float = 1.0,
    max_scale: float = 1.25,
    smooth_window: int = 5,
) -> np.ndarray:
    """Per-frame chroma gain anchored to `t=0` to prevent warp/compression saturation drift."""
    if anchor is None:
        anchor = frames[0]
    anchor_f = anchor.astype(np.float32)
    anchor_y = (anchor_f @ LUMA)[..., None]
    anchor_chroma = float(np.sqrt(np.mean(np.sum((anchor_f - anchor_y) ** 2, axis=-1)))) + 1e-3

    data_f = frames.astype(np.float32)
    clip_y = (data_f @ LUMA)[..., None]
    clip_chroma = np.sqrt(np.mean(np.sum((data_f - clip_y) ** 2, axis=-1), axis=(1, 2))) + 1e-3

    gains = np.clip(anchor_chroma / clip_chroma, 1.0 / max_scale, max_scale)
    gains = 1.0 + (gains - 1.0) * strength
    return _smooth_1d(gains, smooth_window)


def apply_saturation(frames: np.ndarray, sat_gains: np.ndarray) -> np.ndarray:
    """Scale opponent chroma around per-pixel luma without shifting exposure."""
    data = frames.astype(np.float32)
    y = (data @ LUMA)[..., None]
    out = y + (data - y) * sat_gains[:, None, None, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_photometric(frames: np.ndarray, gains: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Apply per-frame gain/offset in float, clipping back into uint8."""
    data = frames.astype(np.float32)
    data = data * gains[:, None, None, None] + offsets[:, None, None, None]
    return np.clip(data, 0, 255).astype(np.uint8)


def unsharp(frames: np.ndarray, radius: int = 2, amount: float = 0.30) -> np.ndarray:
    """Moderate unsharp mask: restores the acutance the warp sampled away."""
    if amount <= 0:
        return frames
    data = frames.astype(np.float32)
    blurred = _box_blur(data, radius)
    sharpened = data + amount * (data - blurred)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def equalize_channels(frames: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Neutralise the colour cast a stretched warp leaves on asphalt and sky.

    Each frame is pulled towards the grey-world assumption only partially: a full
    correction would wash out genuinely tinted scenes (sodium-lit night, sunset).
    """
    if strength <= 0:
        return frames
    data = frames.astype(np.float32)
    channel_means = data.mean(axis=(1, 2), keepdims=True)
    grey = channel_means.mean(axis=3, keepdims=True)
    correction = (grey / np.maximum(channel_means, 1e-3)) ** strength
    return np.clip(data * correction, 0, 255).astype(np.uint8)


def refine_clip(
    frames: np.ndarray,
    anchor: Optional[np.ndarray] = None,
    sharpen_amount: float = 0.30,
    equalize_strength: float = 0.5,
    photometric_strength: float = 1.0,
    saturation_strength: float = 0.0,
) -> np.ndarray:
    """Run the full refinement chain on one camera clip."""
    if frames.ndim != 4 or frames.shape[0] != spec.FRAME_COUNT:
        raise ValueError(
            f"expected {spec.FRAME_COUNT} frames of shape (T, H, W, 3), got {frames.shape}"
        )
    gains, offsets = photometric_gains(frames, anchor=anchor, strength=photometric_strength)
    out = apply_photometric(frames, gains, offsets)
    if saturation_strength > 0:
        sat_g = saturation_gains(out, anchor=anchor, strength=saturation_strength)
        out = apply_saturation(out, sat_g)
    out = equalize_channels(out, strength=equalize_strength)
    out = unsharp(out, radius=2, amount=sharpen_amount)
    out[0] = frames[0] if anchor is None else anchor
    return out


def summarize(frames: Sequence[np.ndarray]) -> dict:
    """Clip statistics, for before/after comparison without a quality model."""
    clip = np.stack(frames)
    plane = luma(clip)
    diffs = np.abs(np.diff(plane, axis=0))
    return {
        "mean_luma": float(plane.mean()),
        "std_luma": float(plane.std()),
        "temporal_flicker": float(diffs.mean()),
        "sharpness": float(np.abs(np.diff(plane, axis=1)).mean()),
    }
