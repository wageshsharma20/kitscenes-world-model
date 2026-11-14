#!/usr/bin/env python3
"""Comprehensive 10-Scenario Benchmark across 4 Architectural Experiments.

Experiments:
  Experiment A: Static Hold Baseline + Constant Velocity (CV) Trajectory
  Experiment B: PhotoWarp Baseline    + Constant Velocity (CV) Trajectory
  Experiment C: PhotoWarp Baseline    + kNN/CV Trajectory Planner
  Experiment D: Single-Donor Retrieval World Model + kNN/CV Trajectory Planner

Evaluates on the exact 10 canonical validation scenarios defined in spec.VALIDATION_10_IDS:
  0000900 (snow & wintry mix, straight)
  0000901 (lane change right)
  0000902 (intersection right turn)
  0000903 (heavy rain, straight)
  0000904 (construction zone, straight)
  0000905 (specifically selected, right turn)
  0000906 (nighttime, lane change left)
  0000911 (specifically selected, left turn)
  0000919 (overtake truck driving on the right)
  0000984 (intersection, u-turn)

Outputs:
  reports/val_10_comprehensive_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitscenes import emc_eval, knn_trajectory, pq_proxy, spec, video_engine
from kitscenes.retrieval_engine import AlignedVideoSynthesizer, DonorIndex, MultimodalRanker, TargetScenario


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 10-scenario comprehensive benchmark comparing 4 world model experiments."
    )
    parser.add_argument(
        "--val-trajectories",
        type=Path,
        default=Path("data/val_trajectories.jsonl"),
        help="Path to genuine validation trajectories jsonl file",
    )
    parser.add_argument(
        "--train-trajectories",
        type=Path,
        default=Path("data/train_trajectories.jsonl"),
        help="Path to genuine train trajectories donor pool",
    )
    parser.add_argument(
        "--observed-t0-dir",
        type=Path,
        default=Path("data/val_observed_t0"),
        help="Directory containing extracted t=0 camera frames for validation scenarios",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("renders/val_10_eval"),
        help="Base directory to save generated scenario videos",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("reports/val_10_comprehensive_benchmark.json"),
        help="Output JSON evaluation report path",
    )
    return parser


def load_val_records(path: Path, target_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Load metadata and ground-truth trajectories for target validation scenarios."""
    target_set = set(target_ids)
    records: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = str(rec["scenario_id"]).zfill(7)
            if sid in target_set:
                records[sid] = rec
    return records


def render_static_hold(
    scenario_id: str,
    cond_images: Dict[str, Path],
    output_scenario_dir: Path,
) -> Dict[str, Path]:
    """Render 6 cameras holding the static t=0 frame for 25 frames."""
    output_scenario_dir.mkdir(parents=True, exist_ok=True)
    encoder = video_engine.VideoEncoder(crf=24)
    paths: Dict[str, Path] = {}
    for cam in spec.CAMERAS:
        out_p = output_scenario_dir / f"{cam}.mp4"
        raw_frame = video_engine._decode_frame(cond_images[cam])
        held_frames = np.repeat(raw_frame[None, ...], spec.FRAME_COUNT, axis=0)
        encoder.encode(held_frames, out_p, validate=True)
        paths[cam] = out_p
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    print("=" * 75)
    print("KITScenes-LongTail 10-Scenario Comprehensive World Modeling Benchmark")
    print("=" * 75)
    print(f"Validation Trajectories: {args.val_trajectories}")
    print(f"Observed t=0 Source:     {args.observed_t0_dir}")
    print(f"Target Scenarios:        {list(spec.VALIDATION_10_IDS)}")
    print("=" * 75)

    # 1. Load validation records
    val_records = load_val_records(args.val_trajectories, spec.VALIDATION_10_IDS)
    missing_records = [sid for sid in spec.VALIDATION_10_IDS if sid not in val_records]
    if missing_records:
        raise ValueError(f"Missing validation records for scenarios: {missing_records}")

    # 2. Check conditioning image presence
    for sid in spec.VALIDATION_10_IDS:
        s_dir = args.observed_t0_dir / sid
        if not s_dir.exists():
            raise FileNotFoundError(f"Missing conditioning folder: {s_dir}")
        for cam in spec.CAMERAS:
            if not (s_dir / f"{cam}.jpg").exists() and not (s_dir / f"{cam}.png").exists():
                raise FileNotFoundError(f"Missing {cam} frame for scenario {sid} in {s_dir}")

    # 3. Initialize Planners and World Model Pipelines
    print("\n[*] Initializing Planners and World Model Generators...")
    cv_planner = knn_trajectory.ConstantVelocityPlanner()
    knn_planner = knn_trajectory.KNNTrajectoryPlanner(donor_path=args.train_trajectories)
    print(f"    Loaded {len(knn_planner.donors)} donor trajectories from {args.train_trajectories}")

    photowarp_model = video_engine.PhotographicWarpModel()
    encoder = video_engine.VideoEncoder(crf=24)
    retrieval_synthesizer = AlignedVideoSynthesizer()
    multimodal_ranker = MultimodalRanker()
    donor_bank = DonorIndex.create_default_bank()

    experiments = ["exp_a_static_cv", "exp_b_warp_cv", "exp_c_warp_knn", "exp_d_retrieval_knn"]
    results: Dict[str, Dict[str, Any]] = {exp: {"scenarios": {}, "mean": {}} for exp in experiments}

    start_total_time = time.time()

    for idx, sid in enumerate(spec.VALIDATION_10_IDS):
        rec = val_records[sid]
        past = rec["past"]
        expert = np.asarray(rec["expert_like"], dtype=float)
        instruction = rec.get("driving_instruction", "")
        stype = rec.get("scenario_type", "")
        cond_images = video_engine.load_conditioning_images(args.observed_t0_dir, sid)

        print(f"\n[{idx + 1:2d}/10] Evaluating Scenario {sid} | {stype} | Instruction: '{instruction}'")

        # --- A. Plan Trajectories ---
        # 1. CV Plan
        plan_cv = cv_planner.plan(past)
        l2_cv = knn_trajectory.compute_l2_metrics(plan_cv, expert)
        mms_cv = knn_trajectory.compute_mms_score(
            plan_cv,
            expert,
            wrong_speed=rec.get("wrong_speed"),
            neglect_instruction=rec.get("neglect_instruction"),
            off_road=rec.get("off_road"),
            crash=rec.get("crash"),
        )

        # 2. kNN/CV Plan
        plan_knn = knn_planner.plan(sid, past, instruction)
        l2_knn = knn_trajectory.compute_l2_metrics(plan_knn, expert)
        mms_knn = knn_trajectory.compute_mms_score(
            plan_knn,
            expert,
            wrong_speed=rec.get("wrong_speed"),
            neglect_instruction=rec.get("neglect_instruction"),
            off_road=rec.get("off_road"),
            crash=rec.get("crash"),
        )

        print(f"    Traj CV:     MMS={mms_cv:.2f} | L2 mean={l2_cv['l2_mean']:.2f}m | L2 med={l2_cv['l2_median']:.2f}m")
        print(f"    Traj kNN/CV: MMS={mms_knn:.2f} | L2 mean={l2_knn['l2_mean']:.2f}m | L2 med={l2_knn['l2_median']:.2f}m")

        # --- B. Experiment A: Static Hold + CV ---
        dir_exp_a = args.output_dir / "exp_a_static_cv" / sid
        t0 = time.time()
        paths_a = render_static_hold(sid, cond_images, dir_exp_a)
        time_a = time.time() - t0
        emc_a = emc_eval.compute_video_optical_flow_emc(paths_a, plan_cv)
        pq_a = pq_proxy.score_scenario_videos(paths_a)
        comp_a = emc_a["emc"] * mms_cv
        results["exp_a_static_cv"]["scenarios"][sid] = {
            "emc": emc_a["emc"],
            "mms": mms_cv,
            "emc_x_mms": comp_a,
            "pq": pq_a["scenario_pq"],
            "pq_front": pq_a["front"],
            "l2_mean": l2_cv["l2_mean"],
            "l2_median": l2_cv["l2_median"],
            "runtime_s": time_a,
        }

        # --- C. Experiment B: PhotoWarp + CV ---
        dir_exp_b = args.output_dir / "exp_b_warp_cv" / sid
        dir_exp_b.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        vids_b = photowarp_model.generate_scenario(sid, plan_cv, conditioning_images=cond_images)
        paths_b = {}
        for cam in spec.CAMERAS:
            out_p = dir_exp_b / f"{cam}.mp4"
            encoder.encode(vids_b[cam], out_p, validate=True)
            paths_b[cam] = out_p
        time_b = time.time() - t0
        emc_b = emc_eval.compute_video_optical_flow_emc(paths_b, plan_cv)
        pq_b = pq_proxy.score_scenario_videos(paths_b)
        comp_b = emc_b["emc"] * mms_cv
        results["exp_b_warp_cv"]["scenarios"][sid] = {
            "emc": emc_b["emc"],
            "mms": mms_cv,
            "emc_x_mms": comp_b,
            "pq": pq_b["scenario_pq"],
            "pq_front": pq_b["front"],
            "l2_mean": l2_cv["l2_mean"],
            "l2_median": l2_cv["l2_median"],
            "runtime_s": time_b,
        }

        # --- D. Experiment C: PhotoWarp + kNN/CV ---
        dir_exp_c = args.output_dir / "exp_c_warp_knn" / sid
        dir_exp_c.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        vids_c = photowarp_model.generate_scenario(sid, plan_knn, conditioning_images=cond_images)
        paths_c = {}
        for cam in spec.CAMERAS:
            out_p = dir_exp_c / f"{cam}.mp4"
            encoder.encode(vids_c[cam], out_p, validate=True)
            paths_c[cam] = out_p
        time_c = time.time() - t0
        emc_c = emc_eval.compute_video_optical_flow_emc(paths_c, plan_knn)
        pq_c = pq_proxy.score_scenario_videos(paths_c)
        comp_c = emc_c["emc"] * mms_knn
        results["exp_c_warp_knn"]["scenarios"][sid] = {
            "emc": emc_c["emc"],
            "mms": mms_knn,
            "emc_x_mms": comp_c,
            "pq": pq_c["scenario_pq"],
            "pq_front": pq_c["front"],
            "l2_mean": l2_knn["l2_mean"],
            "l2_median": l2_knn["l2_median"],
            "runtime_s": time_c,
        }

        # --- E. Experiment D: Single-Donor Retrieval + kNN/CV ---
        dir_exp_d = args.output_dir / "exp_d_retrieval_knn" / sid
        dir_exp_d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        target_obj = TargetScenario(
            scenario_id=sid,
            future_trajectory=plan_knn,
            driving_instruction=instruction,
            scenario_type=stype,
            observed_t0_frames=cond_images,
        )
        ranked_donors = multimodal_ranker.rank(target_obj, donor_bank.values(), top_k=1)
        best_donor = ranked_donors[0].donor
        vids_d = retrieval_synthesizer.synthesize(target_obj, best_donor)
        paths_d = {}
        for cam in spec.CAMERAS:
            out_p = dir_exp_d / f"{cam}.mp4"
            encoder.encode(vids_d[cam], out_p, validate=True)
            paths_d[cam] = out_p
        time_d = time.time() - t0
        emc_d = emc_eval.compute_video_optical_flow_emc(paths_d, plan_knn)
        pq_d = pq_proxy.score_scenario_videos(paths_d)
        comp_d = emc_d["emc"] * mms_knn
        results["exp_d_retrieval_knn"]["scenarios"][sid] = {
            "emc": emc_d["emc"],
            "mms": mms_knn,
            "emc_x_mms": comp_d,
            "pq": pq_d["scenario_pq"],
            "pq_front": pq_d["front"],
            "l2_mean": l2_knn["l2_mean"],
            "l2_median": l2_knn["l2_median"],
            "donor_selected": best_donor.scenario_id,
            "runtime_s": time_d,
        }

        print(f"      Exp A (Static + CV):     PQ_proxy={pq_a['scenario_pq']:5.2f} | EMC={emc_a['emc']:.3f} | EMCxMMS={comp_a:.3f}")
        print(f"      Exp B (Warp + CV):       PQ_proxy={pq_b['scenario_pq']:5.2f} | EMC={emc_b['emc']:.3f} | EMCxMMS={comp_b:.3f}")
        print(f"      Exp C (Warp + kNN/CV):   PQ_proxy={pq_c['scenario_pq']:5.2f} | EMC={emc_c['emc']:.3f} | EMCxMMS={comp_c:.3f}")
        print(f"      Exp D (Retrieval + kNN): PQ_proxy={pq_d['scenario_pq']:5.2f} | EMC={emc_d['emc']:.3f} | EMCxMMS={comp_d:.3f}")

    # Compute overall summary metrics across the 10 scenarios
    total_duration = time.time() - start_total_time

    for exp in experiments:
        scs = results[exp]["scenarios"]
        results[exp]["mean"] = {
            "pq": round(float(np.mean([s["pq"] for s in scs.values()])), 2),
            "pq_front": round(float(np.mean([s["pq_front"] for s in scs.values()])), 2),
            "emc": round(float(np.mean([s["emc"] for s in scs.values()])), 3),
            "mms": round(float(np.mean([s["mms"] for s in scs.values()])), 2),
            "emc_x_mms": round(float(np.mean([s["emc_x_mms"] for s in scs.values()])), 3),
            "l2_mean": round(float(np.mean([s["l2_mean"] for s in scs.values()])), 2),
            "l2_median": round(float(np.mean([s["l2_median"] for s in scs.values()])), 2),
            "avg_runtime_s": round(float(np.mean([s["runtime_s"] for s in scs.values()])), 2),
        }

    # Save detailed JSON report
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 75)
    print(f"BENCHMARK COMPLETE (Total Runtime: {total_duration:.1f}s)")
    print(f"Report saved into: {args.report_out}")
    print("=" * 75)

    print("\nSUMMARY TABLE ACROSS 10 VALIDATION SCENARIOS (PQ_proxy is local heuristic proxy):")
    print(f"{'Experiment':<26s} | {'PQ_proxy':<8s} | {'EMC':<6s} | {'MMS':<6s} | {'EMCxMMS':<8s} | {'L2 (med)':<8s} | {'Runtime'}")
    print("-" * 75)
    for exp in experiments:
        m = results[exp]["mean"]
        print(f"{exp:<26s} | {m['pq']:8.2f} | {m['emc']:6.3f} | {m['mms']:6.2f} | {m['emc_x_mms']:8.3f} | {m['l2_median']:7.2f}m | {m['avg_runtime_s']:.1f}s")
    print("-" * 75)

    return 0


if __name__ == "__main__":
    sys.exit(main())
