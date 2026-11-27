"""Tests for photometric refinement of rendered clips."""

import numpy as np
import pytest

from kitscenes import refine, spec


def _clip(multiplier=1.0, offset=0.0, frames=spec.FRAME_COUNT):
    rng = np.random.default_rng(3)
    base = rng.integers(60, 190, size=(spec.HEIGHT, spec.WIDTH, 3), dtype=np.uint8)
    clip = np.empty((frames, spec.HEIGHT, spec.WIDTH, 3), dtype=np.uint8)
    for index in range(frames):
        clip[index] = np.clip(base * multiplier + offset, 0, 255).astype(np.uint8)
    return clip


def test_photometric_gains_are_neutral_when_frames_share_the_anchor():
    clip = _clip()
    gains, offsets = refine.photometric_gains(clip, anchor=clip[0], smooth_window=1)
    assert np.allclose(gains, 1.0, atol=1e-5)
    assert np.allclose(offsets, 0.0, atol=1e-3)


def test_photometric_gains_lift_a_dark_frame_towards_the_anchor():
    clip = _clip()
    clip[5:] = (clip[5:] * 0.5).astype(np.uint8)
    gains, offsets = refine.photometric_gains(clip, anchor=clip[0], smooth_window=1)
    assert gains[10] > 1.0
    assert offsets[10] > 0.0
    # Gains are bounded so a near-black frame cannot be amplified into noise.
    assert gains.max() <= 1.15 + 1e-6


def test_refine_clip_preserves_the_observed_anchor_frame():
    clip = _clip()
    anchor = clip[0].copy()
    out = refine.refine_clip(clip, anchor=anchor)
    assert out.shape == clip.shape
    assert out.dtype == np.uint8
    assert np.array_equal(out[0], anchor)


def test_refine_clip_rejects_wrong_frame_count():
    clip = _clip(frames=spec.FRAME_COUNT - 1)
    with pytest.raises(ValueError, match="expected 25 frames"):
        refine.refine_clip(clip)


def test_unsharp_raises_acutance_and_equalize_neutralises_a_cast():
    clip = _clip()
    sharpened = refine.unsharp(clip, radius=2, amount=0.30)
    before = float(np.abs(np.diff(refine.luma(clip), axis=1)).mean())
    after = float(np.abs(np.diff(refine.luma(sharpened), axis=1)).mean())
    assert after > before

    blue = clip.copy()
    blue[..., 2] = np.clip(blue[..., 2].astype(np.int16) + 40, 0, 255).astype(np.uint8)
    corrected = refine.equalize_channels(blue, strength=1.0)
    cast_before = float(np.abs(blue[..., 2].mean() - blue[..., 0].mean()))
    cast_after = float(np.abs(corrected[..., 2].mean() - corrected[..., 0].mean()))
    assert cast_after < cast_before


def test_box_blur_keeps_a_flat_field_flat():
    """A wrong window normalisation would scale a constant image; this pins it down."""
    flat = np.full((4, spec.HEIGHT, spec.WIDTH, 3), 90, dtype=np.uint8)
    out = refine.unsharp(flat, radius=2, amount=0.5)
    assert out.shape == flat.shape
    assert np.array_equal(out, flat)


def test_summarize_reports_clip_statistics():
    stats = refine.summarize(list(_clip()))
    assert set(stats) == {"mean_luma", "std_luma", "temporal_flicker", "sharpness"}
    assert all(np.isfinite(value) for value in stats.values())


def test_saturation_anchoring_restores_chroma_without_shifting_luma():
    rng = np.random.default_rng(7)
    base = rng.integers(50, 200, size=(spec.HEIGHT, spec.WIDTH, 3), dtype=np.uint8)
    clip = np.repeat(base[None, ...], spec.FRAME_COUNT, axis=0)
    # Boost chroma on frames 1..24 while keeping roughly neutral anchor chroma at frame 0
    clip_f = clip.astype(np.float32)
    y = (clip_f @ refine.LUMA)[..., None]
    clip_f[1:] = np.clip(y[1:] + 1.35 * (clip_f[1:] - y[1:]), 0, 255)
    drifted = clip_f.astype(np.uint8)

    sat_g = refine.saturation_gains(drifted, anchor=drifted[0], smooth_window=1)
    assert sat_g[0] == pytest.approx(1.0, abs=1e-4)
    assert sat_g[10] < 0.90

    out = refine.refine_clip(drifted, anchor=drifted[0], sharpen_amount=0.0, equalize_strength=0.0, saturation_strength=1.0)
    anchor_c = float(np.sqrt(np.mean(np.sum((drifted[0].astype(np.float32) - y[0]) ** 2, axis=-1))))
    before_c = float(np.sqrt(np.mean(np.sum((drifted[10].astype(np.float32) - y[10]) ** 2, axis=-1))))
    out_y10 = (out[10].astype(np.float32) @ refine.LUMA)[..., None]
    after_c = float(np.sqrt(np.mean(np.sum((out[10].astype(np.float32) - out_y10) ** 2, axis=-1))))
    assert abs(after_c - anchor_c) < abs(before_c - anchor_c)
    assert float(refine.luma(out[10:11]).mean()) == pytest.approx(float(refine.luma(drifted[:1]).mean()), abs=2.0)
