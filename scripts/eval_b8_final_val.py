#!/usr/bin/env python3
"""Stage B.8: End-to-end single-pass validation of ego-luma fix + RealESRGAN (1.0/1.0) + (12,15) CRF."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import json
from pathlib import Path
import sys
import time
from typing import Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitscenes import spec
from kitscenes.depth_reproj_world_model import Depth3DFullFoVWorldModel
from kitscenes.emc_eval import compute_video_optical_flow_emc
from kitscenes.esrgan import build_realesrgan_blend_fn, encode_rgb_h264_preset
from scripts.eval_b6_post_crf import _repack_variant_under_budget, project_2400_gib, reencode_mp4_preset
from scripts.eval_world_model_qalign import (
    _load_predicted_trajectories,
    build_lama_fn,
    compute_psnr,
    load_val_records,
    read_video_rgb,
)


def _render_single_pass_egofix_esrgan(
    b5_dir: Path,
    b8_dir: Path,
    weights_dir: Path,
    val_root: Path,
    pred_trajs: Dict[str, List[List[float]]],
    front_crf: int = 12,
    surr_crf: int = 15,
    preset: str = "slow",
) -> tuple[List[Dict[str, object]], float, tuple[int, int]]:
    """Render 100% end-to-end single-pass pipeline matching `run_production_reproj.py`."""
    vname = "b8_egofix_esr100_c12_15"
    v_dir = b8_dir / "videos" / vname
    v_dir.mkdir(parents=True, exist_ok=True)

    records = load_val_records(val_root, limit=30)
    lama_fn = build_lama_fn(device="cuda:0")
    esr_fn = build_realesrgan_blend_fn(
        weights_dir,
        device="cuda:0",
        blend_alpha=1.00,
        min_luma=35.0,
        night_blend_alpha=1.00,
    )
    model = Depth3DFullFoVWorldModel(
        device="cuda:0",
        render_scale=2,
        depth_fn=None,
        inpaint_fn=lama_fn,
        crf_sharpen=0.24,
        min_lama_hole_pixels=1200,
        min_lama_luma=45.0,
        ego_static_min_luma=45.0,
        preserve_native_t0=False,
        adaptive_filter=False,
        photometric_refine=False,
    )

    t0_base = val_root / "val_t0_native"
    past_base = val_root / "val_past"
    fut_base = val_root / "val_future_videos"
    entries: List[Dict[str, object]] = []
    t_v = time.time()

    for idx, rec in enumerate(records, start=1):
        sid = str(rec["scenario_id"])
        stype = str(rec["scenario_type"])
        fut_traj = pred_trajs.get(sid) or rec["future_trajectory"]
        cond = {c: t0_base / sid / f"{c}.jpg" for c in spec.CAMERAS}
        past_imgs = {}
        for cam in spec.CAMERAS:
            for tag in ("tm02", "tm04", "tm10", "tm20", "tm30", "tm40"):
                pp = past_base / sid / f"{cam}_{tag}.jpg"
                if pp.exists():
                    past_imgs[(cam, tag)] = pp

        t_s = time.time()
        gen = model.generate_scenario(
            sid,
            fut_traj,
            cond,
            past_images=past_imgs if past_imgs else None,
            past_trajectory=rec.get("past_trajectory"),
        )
        scen_paths: Dict[str, Path] = {}
        scen_psnr: Dict[str, float] = {}
        for cam in spec.CAMERAS:
            frames = esr_fn(gen[cam]) if esr_fn is not None else gen[cam]
            mp4 = v_dir / sid / f"{cam}.mp4"
            cam_crf = front_crf if cam == "front" else surr_crf
            encode_rgb_h264_preset(frames, mp4, crf=cam_crf, preset=preset)
            scen_paths[cam] = mp4
            gt_mp4 = fut_base / sid / f"{cam}.mp4"
            gt_arr = read_video_rgb(gt_mp4)
            scen_psnr[cam] = round(compute_psnr(gt_arr, frames), 3)

        emc_val = None
        try:
            emc_val = round(float(compute_video_optical_flow_emc(scen_paths, fut_traj).get("emc", 0.0)), 4)
        except Exception:
            emc_val = None

        for cam in spec.CAMERAS:
            mp4 = scen_paths[cam]
            entries.append(
                {
                    "key": f"{vname}__{sid}__{cam}",
                    "variant": vname,
                    "scenario_id": sid,
                    "scenario_type": stype,
                    "camera": cam,
                    "mp4_path": str(mp4),
                    "size_bytes": mp4.stat().st_size,
                    "psnr": scen_psnr[cam],
                    "emc_local": emc_val,
                }
            )
        mean_psnr = sum(scen_psnr.values()) / len(scen_psnr)
        print(
            f"  [{idx:02d}/{len(records)}] {vname} {sid} ({stype[:16]:16s}) | "
            f"PSNR={mean_psnr:.2f} (front={scen_psnr['front']:.2f}, emc={emc_val}) | {time.time() - t_s:.1f}s",
            flush=True,
        )

    proj_gib, cam_gib = project_2400_gib(v_dir, len(records))
    chosen = (front_crf, surr_crf)
    print(f"  [{vname}] initial crf={chosen} -> proj={proj_gib:.3f} GiB | {cam_gib}", flush=True)
    if proj_gib > 2.92:
        for f_c, s_c in ((13, 16), (14, 17)):
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(
                    pool.map(
                        lambda e, fc=f_c, sc=s_c: reencode_mp4_preset(
                            Path(str(e["mp4_path"])),
                            Path(str(e["mp4_path"])),
                            crf=(fc if e["camera"] == "front" else sc),
                            preset=preset,
                        ),
                        entries,
                    )
                )
            for e in entries:
                e["size_bytes"] = Path(str(e["mp4_path"])).stat().st_size
            proj_gib, cam_gib = project_2400_gib(v_dir, len(records))
            chosen = (f_c, s_c)
            print(f"  [{vname}] clamped crf={chosen} -> proj={proj_gib:.3f} GiB | {cam_gib}", flush=True)
            if proj_gib <= 2.92:
                break

    del model, lama_fn, esr_fn
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[+] {vname} finished in {time.time() - t_v:.1f}s -> {proj_gib:.3f} GiB", flush=True)
    return entries, proj_gib, chosen


def main() -> int:
    b5_dir = Path("/content/stage_b5_calib")
    b6_dir = Path("/content/stage_b6_calib")
    b7_dir = Path("/content/stage_b7_calib")
    b8_dir = Path("/content/stage_b8_calib")
    b8_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = b6_dir / "weights"

    m5 = json.loads((b5_dir / "render_manifest.json").read_text())
    m7 = json.loads((b7_dir / "render_manifest.json").read_text())
    s7 = json.loads((b7_dir / "qalign_scores.json").read_text())
    scenarios = m5["scenarios"]
    n_scen = len(scenarios)
    val_root = Path(str(m5["val_root"]))

    entries_b5: Dict[str, List[Dict[str, object]]] = {}
    for e in m5["entries"]:
        entries_b5.setdefault(str(e["variant"]), []).append(e)
    entries_b7: Dict[str, List[Dict[str, object]]] = {}
    for e in m7["entries"]:
        entries_b7.setdefault(str(e["variant"]), []).append(e)

    val_traj_p = next(
        (p for p in (val_root / "val_trajectories.jsonl", PROJECT_ROOT / "data" / "val_trajectories.jsonl") if p.exists()),
        None,
    )
    train_traj_p = next(
        (p for p in (val_root / "train_trajectories.jsonl", PROJECT_ROOT / "data" / "train_trajectories.jsonl") if p.exists()),
        PROJECT_ROOT / "data" / "train_trajectories.jsonl",
    )
    pred_trajs: Dict[str, List[List[float]]] = {}
    if val_traj_p is not None:
        pred_trajs = _load_predicted_trajectories("ridge", train_path=train_traj_p, val_path=val_traj_p, pred_path=None)

    # 1. Peak alpha=1.00/1.00 + higher bitrate (12, 15) on cached cand_f_lumagate
    esr_fn = build_realesrgan_blend_fn(
        weights_dir, device="cuda:0", blend_alpha=1.00, min_luma=35.0, night_blend_alpha=1.00
    )
    f100_entries, f100_gib, f100_crf = _repack_variant_under_budget(
        entries_b5["cand_f_lumagate"],
        b8_dir,
        "b8_f_esr100_wdn100_c12_15",
        n_scen,
        crf_schedule=[(12, 15), (13, 15), (13, 16), (14, 17)],
        preset="slow",
        frame_transform=esr_fn,
        pred_trajs=pred_trajs,
    )
    del esr_fn
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Full single-pass end-to-end validation of fixed ego_static_min_luma=45 + ESRGAN (1.0/1.0) + (12, 15)
    egofix_entries, ego_gib, ego_crf = _render_single_pass_egofix_esrgan(
        b5_dir, b8_dir, weights_dir, val_root, pred_trajs, front_crf=12, surr_crf=15, preset="slow"
    )

    all_entries = (
        entries_b5["gt"]
        + entries_b5["cand_f_lumagate"]
        + entries_b7["b7_f_esr75_wdn"]
        + f100_entries
        + egofix_entries
    )
    v_cfgs = {
        k: v
        for k, v in m7.get("variant_configs", {}).items()
        if k in ("cand_f_lumagate", "b7_f_esr75_wdn")
    }
    v_cfgs["b8_f_esr100_wdn100_c12_15"] = {
        "src": "cand_f_lumagate",
        "day_alpha": 1.00,
        "night_wdn_alpha": 1.00,
        "preset": "slow",
        "crf": list(f100_crf),
        "proj_gib": round(f100_gib, 3),
    }
    v_cfgs["b8_egofix_esr100_c12_15"] = {
        "src": "single_pass_egofix45",
        "day_alpha": 1.00,
        "night_wdn_alpha": 1.00,
        "ego_static_min_luma": 45.0,
        "min_lama_luma": 45.0,
        "preset": "slow",
        "crf": list(ego_crf),
        "proj_gib": round(ego_gib, 3),
    }
    b8_manifest = {
        "variants": [
            "cand_f_lumagate",
            "b7_f_esr75_wdn",
            "b8_f_esr100_wdn100_c12_15",
            "b8_egofix_esr100_c12_15",
        ],
        "variant_configs": v_cfgs,
        "scenarios": scenarios,
        "entries": all_entries,
    }
    (b8_dir / "render_manifest.json").write_text(json.dumps(b8_manifest, indent=2))
    seed_scores = {
        k: v
        for k, v in s7.items()
        if k.startswith(("gt__", "cand_f_lumagate__", "b7_f_esr75_wdn__"))
    }
    (b8_dir / "qalign_scores.json").write_text(json.dumps(seed_scores, indent=2))
    print(f"[+] Wrote Stage B.8 manifest ({len(all_entries)} entries) & seeded {len(seed_scores)} scores", flush=True)
    print("STAGE_B8_PREP_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
