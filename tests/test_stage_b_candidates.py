"""Tests for Stage B candidates (A-E), Jacobian mip filtering, and Stage C reporting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from kitscenes import spec
from kitscenes.depth_reproj_world_model import Depth3DFullFoVWorldModel
from scripts import eval_world_model_qalign as eq


def test_stage_b_variant_specs_explicit_configs():
    for name in ("cand_a_baseline", "cand_b_scale1", "cand_c_adaptive", "cand_d_photometric", "cand_e_selective"):
        assert name in eq.VARIANT_SPECS

    assert eq.VARIANT_SPECS["cand_a_baseline"]["render_scale"] == 2
    assert eq.VARIANT_SPECS["cand_a_baseline"]["adaptive_filter"] is False
    assert eq.VARIANT_SPECS["cand_b_scale1"]["render_scale"] == 1
    assert eq.VARIANT_SPECS["cand_b_scale1"]["adaptive_filter"] is False
    assert eq.VARIANT_SPECS["cand_c_adaptive"]["render_scale"] == 1
    assert eq.VARIANT_SPECS["cand_c_adaptive"]["adaptive_filter"] is True
    assert eq.VARIANT_SPECS["cand_c_adaptive"]["photometric_refine"] is False
    assert eq.VARIANT_SPECS["cand_d_photometric"]["photometric_refine"] is True
    assert eq.VARIANT_SPECS["cand_e_selective"]["use_lama"] is True
    assert int(eq.VARIANT_SPECS["cand_e_selective"]["min_lama_hole_pixels"]) > 0


def test_jacobian_lod_identity_vs_minification():
    model = Depth3DFullFoVWorldModel(render_scale=1, adaptive_filter=True)
    h, w = model.render_h, model.render_w

    # 1. Identity warp: du/dx = 1, dv/dy = 1 -> lod == 0 across interior
    us = torch.arange(w, dtype=torch.float32).view(1, -1).expand(h, w)
    vs = torch.arange(h, dtype=torch.float32).view(-1, 1).expand(h, w)
    in_b = torch.ones((h, w), dtype=torch.bool)
    lod_id = model._compute_jacobian_lod(us, vs, in_b, None, h, w)
    assert float(lod_id[2:-2, 2:-2].max().item()) == pytest.approx(0.0, abs=1e-5)

    # 2. 2.5x minification warp: du/dx = 2.5, dv/dy = 2.5 -> lod = log2(2.5) > 1.2
    lod_min = model._compute_jacobian_lod(us * 2.5, vs * 2.5, in_b, None, h, w)
    assert float(lod_min[4:-4, 4:-4].mean().item()) > 1.2

    # High-frequency checkerboard should be attenuated only under minification, never on identity
    yy, xx = np.mgrid[0:h, 0:w]
    checker = (((yy + xx) % 2) * 255).astype(np.float32) / 255.0
    src_img = torch.from_numpy(np.stack([checker, checker, checker], axis=0)).unsqueeze(0)
    u_norm = (us / max(w - 1, 1)) * 2.0 - 1.0
    v_norm = (vs / max(h - 1, 1)) * 2.0 - 1.0
    grid_id = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)
    out_id = model._sample_with_optional_mips(src_img, grid_id, us, vs, in_b, None)
    assert torch.allclose(out_id[:, 4:-4, 4:-4], src_img[0, :, 4:-4, 4:-4], atol=1e-3)

    u_min = (us * 2.2).clamp(0, w - 1)
    v_min = (vs * 2.2).clamp(0, h - 1)
    in_min = (us * 2.2 < w - 1) & (vs * 2.2 < h - 1)
    grid_min = torch.stack([(u_min / max(w - 1, 1)) * 2 - 1, (v_min / max(h - 1, 1)) * 2 - 1], dim=-1).unsqueeze(0)

    model_unfiltered = Depth3DFullFoVWorldModel(render_scale=1, adaptive_filter=False)
    out_alias = model_unfiltered._sample_with_optional_mips(src_img, grid_min, u_min, v_min, in_min, None)
    out_prefilt = model._sample_with_optional_mips(src_img, grid_min, u_min, v_min, in_min, None)
    # Filtered minification has strictly lower aliasing oscillation variance in valid interior
    roi = (slice(None), slice(10, 120), slice(10, 180))
    assert float(out_prefilt[roi].std()) < float(out_alias[roi].std())


def test_photometric_refine_flag_runs_inside_world_model():
    rng = np.random.default_rng(19)
    base = rng.integers(40, 210, size=(spec.HEIGHT, spec.WIDTH, 3), dtype=np.uint8)
    cond = {cam: base for cam in spec.CAMERAS}
    traj = [[0.8 * (k + 1), 0.1 * (k + 1)] for k in range(spec.WAYPOINT_COUNT)]

    model = Depth3DFullFoVWorldModel(
        render_scale=1,
        crf_sharpen=0.24,
        adaptive_filter=True,
        photometric_refine=True,
    )
    out = model.generate_scenario("0000902", traj, cond, cameras=["front"])
    clip = out["front"]
    assert clip.shape == (spec.FRAME_COUNT, spec.HEIGHT, spec.WIDTH, 3)
    assert clip.dtype == np.uint8


def test_stratified_load_and_stage_c_summary(tmp_path: Path):
    val_root = tmp_path / "val_calib"
    traj_path = tmp_path / "val_trajectories.jsonl"
    lines = []
    # 14 scenarios across all 7 slices
    for i in range(14):
        sid = f"{900 + i:07d}"
        stype = eq.SLICE_ORDER[i % len(eq.SLICE_ORDER)]
        (val_root / "val_t0_native" / sid).mkdir(parents=True, exist_ok=True)
        (val_root / "val_t0_native" / sid / "front.jpg").write_bytes(b"x")
        (val_root / "val_future_videos" / sid).mkdir(parents=True, exist_ok=True)
        (val_root / "val_future_videos" / sid / "front.mp4").write_bytes(b"y")
        lines.append(json.dumps({"scenario_id": sid, "scenario_type": stype, "expert_like": [[0.0, 0.0]] * 25}))
    traj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    picked = eq.load_val_records(traj_path, val_root, limit=7, stratified=True)
    assert len(picked) == 7
    assert {r["scenario_type"] for r in picked} == set(eq.SLICE_ORDER)

    # Verify Stage C worst_25_videos + paired_vs_baseline score aggregation from cached scores
    out_dir = tmp_path / "calib_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "crf": 14,
        "variants": ["cand_a_baseline", "cand_b_scale1"],
        "variant_configs": {
            "cand_a_baseline": eq.VARIANT_SPECS["cand_a_baseline"],
            "cand_b_scale1": eq.VARIANT_SPECS["cand_b_scale1"],
        },
        "scenarios": [{"scenario_id": f"{900 + i:07d}", "scenario_type": eq.SLICE_ORDER[i % 7]} for i in range(5)],
        "entries": [],
    }
    cached = {}
    for var, bonus in (("gt", 0.0), ("cand_a_baseline", -0.35), ("cand_b_scale1", -0.05)):
        for i in range(5):
            sid = f"{900 + i:07d}"
            stype = eq.SLICE_ORDER[i % 7]
            for cam in spec.CAMERAS:
                key = f"{var}__{sid}__{cam}"
                q_val = max(1.0, 4.6 + bonus - 0.02 * i)
                entry = {
                    "key": key,
                    "variant": var,
                    "scenario_id": sid,
                    "scenario_type": stype,
                    "camera": cam,
                    "mp4_path": str(tmp_path / "unused.mp4"),
                    "psnr": 20.0,
                    "emc_local": 0.5,
                    "q_raw": q_val,
                    "level_probs": {"excellent": 0.8, "good": 0.15, "fair": 0.05, "poor": 0.0, "bad": 0.0},
                }
                manifest["entries"].append(entry)
                cached[key] = entry
    (out_dir / "render_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (out_dir / "qalign_scores.json").write_text(json.dumps(cached), encoding="utf-8")

    rc = eq.main(["--stage", "score", "--out-dir", str(out_dir), "--baseline-variant", "cand_a_baseline"])
    assert rc == 0
    summary = json.loads((out_dir / "calibration_summary.json").read_text(encoding="utf-8"))
    assert len(summary["cand_b_scale1"]["worst_25_videos"]) == 25
    assert summary["cand_b_scale1"]["paired_vs_baseline"]["mean_delta"] > 0
    assert summary["cand_b_scale1"]["paired_vs_baseline"]["scenarios_improved"] == 5
    assert summary["cand_b_scale1"]["config"]["render_scale"] == 1
