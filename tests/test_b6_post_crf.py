"""Unit tests for Stage B.6 luma-safe sharpen, preset encoder helpers, and size projection."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kitscenes import spec
from scripts.eval_b6_post_crf import _repack_variant_under_budget, apply_luma_safe_unsharp, project_2400_gib
from scripts.eval_world_model_qalign import _load_predicted_trajectories


def test_load_predicted_trajectories_accepts_stage_b6_kwargs() -> None:
    """Verify Stage B.6 keyword invocation matches _load_predicted_trajectories signature."""
    import inspect

    sig = inspect.signature(_load_predicted_trajectories)
    bound = sig.bind("ridge", train_path=Path("t.jsonl"), val_path=Path("v.jsonl"), pred_path=None)
    assert bound.arguments["traj_source"] == "ridge"


def test_repack_reuses_existing_under_budget_clips(tmp_path: Path) -> None:
    """When all clips already exist on disk under <=2.92 GiB, _repack_variant_under_budget skips re-encoding."""
    src_entries = []
    v_dir = tmp_path / "videos" / "cached_var" / "0000900"
    v_dir.mkdir(parents=True)
    for cam in spec.CAMERAS:
        mp4 = v_dir / f"{cam}.mp4"
        mp4.write_bytes(b"x" * 2048)
        src_entries.append(
            {
                "scenario_id": "0000900",
                "scenario_type": "0 specifically selected",
                "camera": cam,
                "mp4_path": str(tmp_path / "nonexistent" / f"{cam}.mp4"),
                "psnr": 20.0,
                "emc_local": 0.01,
            }
        )

    entries, final_gib, pair = _repack_variant_under_budget(
        src_entries, tmp_path, "cached_var", n_scenarios=21, crf_schedule=[(15, 19)]
    )
    assert len(entries) == 6
    assert final_gib < 2.92
    assert pair == (15, 19)


def test_apply_luma_safe_unsharp_skips_dark_night_frames() -> None:
    """Dark night frames (mean luma < 45) must remain untouched; bright frames get crisper edges."""
    rng = np.random.default_rng(11)
    dark = rng.integers(5, 18, size=(3, 64, 64, 3), dtype=np.uint8)
    bright = rng.integers(80, 160, size=(3, 64, 64, 3), dtype=np.uint8)
    mixed = np.concatenate([dark, bright], axis=0)

    out = apply_luma_safe_unsharp(mixed, extra_amount=0.30, min_luma=45.0)
    assert np.array_equal(out[:3], dark)
    assert not np.array_equal(out[3:], bright)


def test_project_2400_gib_scales_linearly_to_400_scenarios(tmp_path: Path) -> None:
    """Projected 2,400-video GiB and per-camera breakdown scale by 400 / n_scenarios."""
    scen_dir = tmp_path / "0000900"
    scen_dir.mkdir()
    per_file_bytes = 1024 * 1024
    for cam in spec.CAMERAS:
        (scen_dir / f"{cam}.mp4").write_bytes(b"x" * per_file_bytes)

    total_gib, by_cam = project_2400_gib(tmp_path, n_scenarios=1)
    expected_total = (6 * per_file_bytes * 400.0) / float(1024**3)
    assert abs(total_gib - expected_total) < 1e-6
    expected_cam = round((per_file_bytes * 400.0) / float(1024**3), 3)
    assert all(val == expected_cam for val in by_cam.values())
