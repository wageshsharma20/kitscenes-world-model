#!/usr/bin/env python3
"""Stage B.7: High-alpha RealESRGAN + WDN night denoiser + full <=2.92 GiB bitrate optimization."""

from __future__ import annotations

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

from scripts.eval_b6_post_crf import _repack_variant_under_budget, build_realesrgan_blend_fn
from scripts.eval_world_model_qalign import _load_predicted_trajectories


def main() -> int:
    b5_dir = Path("/content/stage_b5_calib")
    b6_dir = Path("/content/stage_b6_calib")
    b7_dir = Path("/content/stage_b7_calib")
    b7_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = b6_dir / "weights"

    m5 = json.loads((b5_dir / "render_manifest.json").read_text())
    m6 = json.loads((b6_dir / "render_manifest.json").read_text())
    s6 = json.loads((b6_dir / "qalign_scores.json").read_text())
    scenarios = m5["scenarios"]
    n_scen = len(scenarios)

    entries_b5: Dict[str, List[Dict[str, object]]] = {}
    for e in m5["entries"]:
        entries_b5.setdefault(str(e["variant"]), []).append(e)
    entries_b6: Dict[str, List[Dict[str, object]]] = {}
    for e in m6["entries"]:
        entries_b6.setdefault(str(e["variant"]), []).append(e)

    pred_trajs: Dict[str, List[List[float]]] = {}
    val_root = Path(str(m5.get("val_root", "")))
    val_traj_p = next(
        (p for p in (val_root / "val_trajectories.jsonl", PROJECT_ROOT / "data" / "val_trajectories.jsonl") if p.exists()),
        None,
    )
    train_traj_p = next(
        (p for p in (val_root / "train_trajectories.jsonl", PROJECT_ROOT / "data" / "train_trajectories.jsonl") if p.exists()),
        PROJECT_ROOT / "data" / "train_trajectories.jsonl",
    )
    if val_traj_p is not None:
        pred_trajs = _load_predicted_trajectories(
            "ridge", train_path=train_traj_p, val_path=val_traj_p, pred_path=None
        )

    crf_sched = [(14, 17), (14, 18), (15, 18), (15, 19)]
    t0 = time.time()

    specs = [
        ("b7_k_esr75_c14_17", "cand_k_native", 0.75, 0.0),
        ("b7_k_esr100_c14_17", "cand_k_native", 1.00, 0.0),
        ("b7_f_esr75_wdn", "cand_f_lumagate", 0.75, 0.65),
    ]
    new_entries: List[Dict[str, object]] = []
    v_cfgs: Dict[str, object] = {
        k: v for k, v in m6.get("variant_configs", {}).items() if k in ("cand_f_lumagate", "b6_esrgan45_asym")
    }
    for vname, base_v, day_alpha, night_alpha in specs:
        fn = build_realesrgan_blend_fn(
            weights_dir,
            device="cuda:0",
            blend_alpha=day_alpha,
            min_luma=35.0,
            night_blend_alpha=night_alpha,
        )
        v_entries, v_gib, v_crf = _repack_variant_under_budget(
            entries_b5[base_v],
            b7_dir,
            vname,
            n_scen,
            crf_schedule=crf_sched,
            preset="slow",
            frame_transform=fn,
            pred_trajs=pred_trajs,
        )
        new_entries.extend(v_entries)
        v_cfgs[vname] = {
            "src": base_v,
            "day_alpha": day_alpha,
            "night_wdn_alpha": night_alpha,
            "preset": "slow",
            "crf": list(v_crf),
            "proj_gib": round(v_gib, 3),
        }
        del fn
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[+] Built 3 Stage B.7 <=2.92 GiB variants in {time.time() - t0:.1f}s", flush=True)

    all_entries = (
        entries_b5["gt"]
        + entries_b5["cand_f_lumagate"]
        + entries_b6["b6_esrgan45_asym"]
        + new_entries
    )
    b7_manifest = {
        "variants": [
            "cand_f_lumagate",
            "b6_esrgan45_asym",
            "b7_k_esr75_c14_17",
            "b7_k_esr100_c14_17",
            "b7_f_esr75_wdn",
        ],
        "variant_configs": v_cfgs,
        "scenarios": scenarios,
        "entries": all_entries,
    }
    (b7_dir / "render_manifest.json").write_text(json.dumps(b7_manifest, indent=2))
    seed_scores = {
        k: v
        for k, v in s6.items()
        if k.startswith(("gt__", "cand_f_lumagate__", "b6_esrgan45_asym__"))
    }
    (b7_dir / "qalign_scores.json").write_text(json.dumps(seed_scores, indent=2))
    print(f"[+] Wrote Stage B.7 manifest ({len(all_entries)} entries) & seeded {len(seed_scores)} scores", flush=True)
    print("STAGE_B7_PREP_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
