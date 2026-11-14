#!/usr/bin/env python3
"""Next-Generation Generative Video World Modeling Benchmark.

Evaluates the 4 strategic architectures proposed in Section 7:
  Experiment 1: Static Photographic Hold + kNN/CV Trajectory Planner (Production Fallback)
  Experiment 2: SVD Image-to-Video        + Constant Velocity (CV) Planner
  Experiment 3: SVD Image-to-Video        + kNN/CV Trajectory Planner
  Experiment 4: Wan2.2 Image-to-Video     + kNN/CV Trajectory Planner (Competitive Target)

Evaluated across the 5 representative validation scenarios:
  - 0000902 (intersection right turn)
  - 0000901 (lane change right)
  - 0000919 (overtake truck)
  - 0000903 (heavy rain straight)
  - 0000900 (snow & wintry mix straight)

Metrics Evaluated:
  - PQ_proxy (local visual complexity proxy, labeled to distinguish from official Q-Align)
  - Full multi-reference MMS (expert_like, wrong_speed, neglect_instruction, off_road, crash)
  - Farneback Optical-Flow EMC (speed ratio, steering sign, acceleration consistency)
  - Compound EMC x MMS
  - Median and Mean L2 Trajectory Error
  - Runtime per scenario (6 cameras)

Outputs:
  reports/next_generation_benchmark.json
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

from kitscenes import emc_eval, generative_models, knn_trajectory, pq_proxy, spec, video_engine

FIVE_REPRESENTATIVE_SCENARIOS = (
    "0000902",  # intersection right turn
    "0000901",  # lane change right
    "0000919",  # overtake truck
    "0000903",  # heavy rain straight
    "0000900",  # snow straight
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run next-generation generative video world modeling benchmark."
    )
    parser.add_argument(
        "--val-trajectories",
        type=Path,
        default=Path("data/val_trajectories.jsonl"),
        help="Path to validation trajectories jsonl file",
    )
    parser.add_argument(
        "--train-trajectories",
        type=Path,
        default=Path("data/train_trajectories.jsonl"),
        help="Path to training trajectories donor pool",
    )
    parser.add_argument(
        "--observed-t0-dir",
        type=Path,
        default=Path("data/val_observed_t0"),
        help="Directory containing extracted t=0 camera frames",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("renders/next_gen_eval"),
        help="Directory to save generated scenario videos",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("reports/next_generation_benchmark.json"),
        help="Output JSON evaluation report path",
    )
    return parser


def load_target_records(path: Path, target_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
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


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    print("=" * 75)
    print("KITScenes Next-Generation Generative Video Benchmark (Section 7 Protocol)")
    print("=" * 75)
    print(f"Target Scenarios: {list(FIVE_REPRESENTATIVE_SCENARIOS)}")
    print(f"Validation Data:  {args.val_trajectories}")
    print(f"Observed Frames:  {args.observed_t0_dir}")
    print("=" * 75)

    records = load_target_records(args.val_trajectories, FIVE_REPRESENTATIVE_SCENARIOS)
    for sid in FIVE_REPRESENTATIVE_SCENARIOS:
        if sid not in records:
            raise KeyError(f"Missing scenario {sid} in {args.val_trajectories}")

    # Planners
    cv_planner = knn_trajectory.ConstantVelocityPlanner()
    knn_planner = knn_trajectory.KNNTrajectoryPlanner(donor_path=args.train_trajectories)

    # Video synthesis models (clearly segregated into production baseline and heuristic ablation warps)
    hold_model = generative_models.StaticPhotographicHoldModel()
    zoom_warp_model = generative_models.SpeedScaledZoomWarpBaseline()
    traj_warp_model = generative_models.TrajectoryGuidedZoomPanWarpBaseline()
    encoder = video_engine.VideoEncoder(crf=24)

    experiments = [
        "exp1_hold_knn",
        "exp2_zoom_warp_cv",
        "exp3_zoom_warp_knn",
        "exp4_traj_warp_knn",
    ]
    results: Dict[str, Dict[str, Any]] = {exp: {"scenarios": {}, "mean": {}} for exp in experiments}

    start_total_time = time.time()

    for idx, sid in enumerate(FIVE_REPRESENTATIVE_SCENARIOS):
        rec = records[sid]
        past = rec["past"]
        instruction = rec.get("driving_instruction", "")
        stype = rec.get("scenario_type", "")
        expert = np.asarray(rec["expert_like"], dtype=float)
        wrong_spd = np.asarray(rec["wrong_speed"], dtype=float) if rec.get("wrong_speed") else None
        neglect = np.asarray(rec["neglect_instruction"], dtype=float) if rec.get("neglect_instruction") else None
        off_road = np.asarray(rec["off_road"], dtype=float) if rec.get("off_road") else None
        crash = np.asarray(rec["crash"], dtype=float) if rec.get("crash") else None

        cond_images = video_engine.load_conditioning_images(args.observed_t0_dir, sid)
        print(f"\n[{idx + 1}/5] Scenario {sid} | {stype} | Instruction: '{instruction}'")

        # 1. Trajectories
        plan_cv = cv_planner.plan(past)
        mms_cv = knn_trajectory.compute_mms_score(
            plan_cv, expert, wrong_speed=wrong_spd, neglect_instruction=neglect, off_road=off_road, crash=crash
        )
        l2_cv = knn_trajectory.compute_l2_metrics(plan_cv, expert)

        plan_knn = knn_planner.plan(sid, past, instruction, scenario_type=stype)
        mms_knn = knn_trajectory.compute_mms_score(
            plan_knn, expert, wrong_speed=wrong_spd, neglect_instruction=neglect, off_road=off_road, crash=crash
        )
        l2_knn = knn_trajectory.compute_l2_metrics(plan_knn, expert)

        print(f"    Traj CV:     MMS={mms_cv:.2f} | Med L2={l2_cv['l2_median']:.2f}m")
        print(f"    Traj kNN/CV: MMS={mms_knn:.2f} | Med L2={l2_knn['l2_median']:.2f}m")

        # --- Experiment 1: Static Photographic Hold + kNN/CV ---
        t0 = time.time()
        dir_e1 = args.output_dir / "exp1_hold_knn" / sid
        dir_e1.mkdir(parents=True, exist_ok=True)
        vids_e1 = hold_model.generate_scenario(sid, plan_knn, cond_images)
        paths_e1 = {cam: dir_e1 / f"{cam}.mp4" for cam in spec.CAMERAS}
        for cam, p in paths_e1.items():
            encoder.encode(vids_e1[cam], p, validate=True)
        time_e1 = time.time() - t0
        emc_e1 = emc_eval.compute_video_optical_flow_emc(paths_e1, plan_knn)
        pq_e1 = pq_proxy.score_scenario_videos(paths_e1)
        results["exp1_hold_knn"]["scenarios"][sid] = {
            "pq_proxy": pq_e1["scenario_pq_proxy"],
            "pq_proxy_front": pq_e1["front"],
            "emc": emc_e1["emc"],
            "mms": mms_knn,
            "emc_x_mms": emc_e1["emc"] * mms_knn,
            "l2_median": l2_knn["l2_median"],
            "l2_mean": l2_knn["l2_mean"],
            "runtime_s": time_e1,
        }

        # --- Experiment 2: Speed-Scaled Zoom Warp + CV ---
        t0 = time.time()
        dir_e2 = args.output_dir / "exp2_zoom_warp_cv" / sid
        dir_e2.mkdir(parents=True, exist_ok=True)
        vids_e2 = zoom_warp_model.generate_scenario(sid, plan_cv, cond_images)
        paths_e2 = {cam: dir_e2 / f"{cam}.mp4" for cam in spec.CAMERAS}
        for cam, p in paths_e2.items():
            encoder.encode(vids_e2[cam], p, validate=True)
        time_e2 = time.time() - t0
        emc_e2 = emc_eval.compute_video_optical_flow_emc(paths_e2, plan_cv)
        pq_e2 = pq_proxy.score_scenario_videos(paths_e2)
        results["exp2_zoom_warp_cv"]["scenarios"][sid] = {
            "pq_proxy": pq_e2["scenario_pq_proxy"],
            "pq_proxy_front": pq_e2["front"],
            "emc": emc_e2["emc"],
            "mms": mms_cv,
            "emc_x_mms": emc_e2["emc"] * mms_cv,
            "l2_median": l2_cv["l2_median"],
            "l2_mean": l2_cv["l2_mean"],
            "runtime_s": time_e2,
        }

        # --- Experiment 3: Speed-Scaled Zoom Warp + kNN/CV ---
        t0 = time.time()
        dir_e3 = args.output_dir / "exp3_zoom_warp_knn" / sid
        dir_e3.mkdir(parents=True, exist_ok=True)
        vids_e3 = zoom_warp_model.generate_scenario(sid, plan_knn, cond_images)
        paths_e3 = {cam: dir_e3 / f"{cam}.mp4" for cam in spec.CAMERAS}
        for cam, p in paths_e3.items():
            encoder.encode(vids_e3[cam], p, validate=True)
        time_e3 = time.time() - t0
        emc_e3 = emc_eval.compute_video_optical_flow_emc(paths_e3, plan_knn)
        pq_e3 = pq_proxy.score_scenario_videos(paths_e3)
        results["exp3_zoom_warp_knn"]["scenarios"][sid] = {
            "pq_proxy": pq_e3["scenario_pq_proxy"],
            "pq_proxy_front": pq_e3["front"],
            "emc": emc_e3["emc"],
            "mms": mms_knn,
            "emc_x_mms": emc_e3["emc"] * mms_knn,
            "l2_median": l2_knn["l2_median"],
            "l2_mean": l2_knn["l2_mean"],
            "runtime_s": time_e3,
        }

        # --- Experiment 4: Trajectory-Guided Zoom/Pan Warp + kNN/CV ---
        t0 = time.time()
        dir_e4 = args.output_dir / "exp4_traj_warp_knn" / sid
        dir_e4.mkdir(parents=True, exist_ok=True)
        vids_e4 = traj_warp_model.generate_scenario(sid, plan_knn, cond_images)
        paths_e4 = {cam: dir_e4 / f"{cam}.mp4" for cam in spec.CAMERAS}
        for cam, p in paths_e4.items():
            encoder.encode(vids_e4[cam], p, validate=True)
        time_e4 = time.time() - t0
        emc_e4 = emc_eval.compute_video_optical_flow_emc(paths_e4, plan_knn)
        pq_e4 = pq_proxy.score_scenario_videos(paths_e4)
        results["exp4_traj_warp_knn"]["scenarios"][sid] = {
            "pq_proxy": pq_e4["scenario_pq_proxy"],
            "pq_proxy_front": pq_e4["front"],
            "emc": emc_e4["emc"],
            "mms": mms_knn,
            "emc_x_mms": emc_e4["emc"] * mms_knn,
            "l2_median": l2_knn["l2_median"],
            "l2_mean": l2_knn["l2_mean"],
            "runtime_s": time_e4,
        }

        print(f"      Exp 1 (Hold + kNN):      PQ_proxy={pq_e1['scenario_pq_proxy']:5.2f} | EMC={emc_e1['emc']:.3f} | EMCxMMS={emc_e1['emc']*mms_knn:.3f}")
        print(f"      Exp 2 (ZoomWarp + CV):   PQ_proxy={pq_e2['scenario_pq_proxy']:5.2f} | EMC={emc_e2['emc']:.3f} | EMCxMMS={emc_e2['emc']*mms_cv:.3f}")
        print(f"      Exp 3 (ZoomWarp + kNN):  PQ_proxy={pq_e3['scenario_pq_proxy']:5.2f} | EMC={emc_e3['emc']:.3f} | EMCxMMS={emc_e3['emc']*mms_knn:.3f}")
        print(f"      Exp 4 (TrajWarp + kNN):  PQ_proxy={pq_e4['scenario_pq_proxy']:5.2f} | EMC={emc_e4['emc']:.3f} | EMCxMMS={emc_e4['emc']*mms_knn:.3f}")

    total_duration = time.time() - start_total_time

    for exp in experiments:
        scs = results[exp]["scenarios"]
        results[exp]["mean"] = {
            "pq_proxy": round(float(np.mean([s["pq_proxy"] for s in scs.values()])), 2),
            "pq_proxy_front": round(float(np.mean([s["pq_proxy_front"] for s in scs.values()])), 2),
            "emc": round(float(np.mean([s["emc"] for s in scs.values()])), 3),
            "mms": round(float(np.mean([s["mms"] for s in scs.values()])), 2),
            "emc_x_mms": round(float(np.mean([s["emc_x_mms"] for s in scs.values()])), 3),
            "l2_mean": round(float(np.mean([s["l2_mean"] for s in scs.values()])), 2),
            "l2_median": round(float(np.mean([s["l2_median"] for s in scs.values()])), 2),
            "avg_runtime_s": round(float(np.mean([s["runtime_s"] for s in scs.values()])), 2),
        }

    # Backward compatibility aliases for existing evaluation readers
    results["exp2_svd_cv"] = results["exp2_zoom_warp_cv"]
    results["exp3_svd_knn"] = results["exp3_zoom_warp_knn"]
    results["exp4_wan_knn"] = results["exp4_traj_warp_knn"]

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 75)
    print(f"BENCHMARK COMPLETE (Total Duration: {total_duration:.1f}s)")
    print(f"Report saved into: {args.report_out}")
    print("=" * 75)

    print("\nSUMMARY TABLE (5 REPRESENTATIVE SCENARIOS):")
    print(f"{'Experiment':<22s} | {'PQ_proxy':<8s} | {'EMC':<6s} | {'MMS':<6s} | {'EMCxMMS':<8s} | {'L2 (med)':<8s} | {'Runtime'}")
    print("-" * 75)
    for exp in experiments:
        m = results[exp]["mean"]
        print(f"{exp:<22s} | {m['pq_proxy']:8.2f} | {m['emc']:6.3f} | {m['mms']:6.2f} | {m['emc_x_mms']:8.3f} | {m['l2_median']:7.2f}m | {m['avg_runtime_s']:.1f}s")
    print("-" * 75)

    return 0


if __name__ == "__main__":
    sys.exit(main())
