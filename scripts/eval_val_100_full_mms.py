#!/usr/bin/env python3
"""Evaluate Trajectory Planners on All 100 Validation Scenarios with Full Multi-Reference MMS.

Evaluates:
  1. Constant Velocity (CV) Planner
  2. Enhanced kNN/CV Trajectory Planner

Uses the official challenge multi-reference MMS metric passing:
  - expert_like (10.0 pts)
  - wrong_speed (7.0 pts)
  - neglect_instruction (4.0 pts)
  - off_road (1.0 pt)
  - crash (0.0 pts)
  - initial heading cosine check (< 0.5 -> 0.0 pts)
  - unmatched floor (3.5 pts) with comfort penalties (jerk & tortuosity)

Outputs:
  reports/val_100_full_mms_evaluation.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitscenes import knn_trajectory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Trajectory Planners on All 100 Validation Scenarios using full multi-reference MMS."
    )
    parser.add_argument(
        "--val-trajectories",
        type=Path,
        default=Path("data/val_trajectories.jsonl"),
        help="Path to genuine validation trajectories jsonl file (100 scenarios)",
    )
    parser.add_argument(
        "--train-trajectories",
        type=Path,
        default=Path("data/train_trajectories.jsonl"),
        help="Path to genuine train trajectories donor pool (500 scenarios)",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("reports/val_100_full_mms_evaluation.json"),
        help="Output JSON evaluation report path",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    print("=" * 75)
    print("KITScenes-LongTail Full 100-Scenario Trajectory Evaluation (Multi-Ref MMS)")
    print("=" * 75)
    print(f"Validation Records: {args.val_trajectories}")
    print(f"Donor Pool:         {args.train_trajectories}")
    print("=" * 75)

    # Load validation records
    val_records: List[Dict[str, Any]] = []
    with open(args.val_trajectories, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_records.append(json.loads(line))

    print(f"[*] Loaded {len(val_records)} validation scenarios")

    # Initialize planners
    cv_planner = knn_trajectory.ConstantVelocityPlanner()
    knn_planner = knn_trajectory.KNNTrajectoryPlanner(donor_path=args.train_trajectories)
    print(f"[*] Loaded {len(knn_planner.donors)} donor trajectories into kNN planner")

    cv_results = []
    knn_results = []
    per_maneuver = defaultdict(lambda: {"cv_mms": [], "knn_mms": [], "cv_l2": [], "knn_l2": []})
    per_scenario_type = defaultdict(lambda: {"cv_mms": [], "knn_mms": [], "cv_l2": [], "knn_l2": []})

    start_time = time.time()

    for _idx, rec in enumerate(val_records):
        sid = rec["scenario_id"]
        past = rec["past"]
        instruction = rec.get("driving_instruction", "")
        stype = rec.get("scenario_type", "unknown")
        expert = np.asarray(rec["expert_like"], dtype=float)
        wrong_spd = np.asarray(rec["wrong_speed"], dtype=float) if rec.get("wrong_speed") else None
        neglect = np.asarray(rec["neglect_instruction"], dtype=float) if rec.get("neglect_instruction") else None
        off_road = np.asarray(rec["off_road"], dtype=float) if rec.get("off_road") else None
        crash = np.asarray(rec["crash"], dtype=float) if rec.get("crash") else None

        maneuver = knn_trajectory.classify_instruction(instruction)

        # Plan with CV
        plan_cv = cv_planner.plan(past)
        mms_cv = knn_trajectory.compute_mms_score(
            plan_cv,
            expert,
            wrong_speed=wrong_spd,
            neglect_instruction=neglect,
            off_road=off_road,
            crash=crash,
        )
        l2_cv = knn_trajectory.compute_l2_metrics(plan_cv, expert)

        # Plan with kNN/CV
        plan_knn = knn_planner.plan(sid, past, instruction, scenario_type=stype)
        mms_knn = knn_trajectory.compute_mms_score(
            plan_knn,
            expert,
            wrong_speed=wrong_spd,
            neglect_instruction=neglect,
            off_road=off_road,
            crash=crash,
        )
        l2_knn = knn_trajectory.compute_l2_metrics(plan_knn, expert)

        res_cv = {
            "scenario_id": sid,
            "maneuver": maneuver,
            "scenario_type": stype,
            "mms": mms_cv,
            "l2_mean": l2_cv["l2_mean"],
            "l2_median": l2_cv["l2_median"],
        }
        res_knn = {
            "scenario_id": sid,
            "maneuver": maneuver,
            "scenario_type": stype,
            "mms": mms_knn,
            "l2_mean": l2_knn["l2_mean"],
            "l2_median": l2_knn["l2_median"],
        }

        cv_results.append(res_cv)
        knn_results.append(res_knn)

        per_maneuver[maneuver]["cv_mms"].append(mms_cv)
        per_maneuver[maneuver]["knn_mms"].append(mms_knn)
        per_maneuver[maneuver]["cv_l2"].append(l2_cv["l2_median"])
        per_maneuver[maneuver]["knn_l2"].append(l2_knn["l2_median"])

        per_scenario_type[stype]["cv_mms"].append(mms_cv)
        per_scenario_type[stype]["knn_mms"].append(mms_knn)
        per_scenario_type[stype]["cv_l2"].append(l2_cv["l2_median"])
        per_scenario_type[stype]["knn_l2"].append(l2_knn["l2_median"])

    elapsed = time.time() - start_time

    # Compute overall summary
    summary = {
        "num_scenarios": len(val_records),
        "cv_planner": {
            "mean_mms": round(float(np.mean([r["mms"] for r in cv_results])), 2),
            "median_mms": round(float(np.median([r["mms"] for r in cv_results])), 2),
            "mean_l2": round(float(np.mean([r["l2_mean"] for r in cv_results])), 2),
            "median_l2": round(float(np.median([r["l2_median"] for r in cv_results])), 2),
        },
        "knn_cv_planner": {
            "mean_mms": round(float(np.mean([r["mms"] for r in knn_results])), 2),
            "median_mms": round(float(np.median([r["mms"] for r in knn_results])), 2),
            "mean_l2": round(float(np.mean([r["l2_mean"] for r in knn_results])), 2),
            "median_l2": round(float(np.median([r["l2_median"] for r in knn_results])), 2),
        },
        "per_maneuver": {},
        "per_scenario_type": {},
        "evaluation_duration_s": round(elapsed, 2),
    }

    for m, vals in sorted(per_maneuver.items()):
        summary["per_maneuver"][m] = {
            "count": len(vals["cv_mms"]),
            "cv_mms": round(float(np.mean(vals["cv_mms"])), 2),
            "knn_mms": round(float(np.mean(vals["knn_mms"])), 2),
            "cv_median_l2": round(float(np.median(vals["cv_l2"])), 2),
            "knn_median_l2": round(float(np.median(vals["knn_l2"])), 2),
        }

    for st, vals in sorted(per_scenario_type.items()):
        summary["per_scenario_type"][st] = {
            "count": len(vals["cv_mms"]),
            "cv_mms": round(float(np.mean(vals["cv_mms"])), 2),
            "knn_mms": round(float(np.mean(vals["knn_mms"])), 2),
            "cv_median_l2": round(float(np.median(vals["cv_l2"])), 2),
            "knn_median_l2": round(float(np.median(vals["knn_l2"])), 2),
        }

    # Save output report
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "summary": summary,
        "scenarios_cv": cv_results,
        "scenarios_knn": knn_results,
    }
    args.report_out.write_text(json.dumps(report_data, indent=2))

    print("\n" + "=" * 75)
    print("EVALUATION COMPLETE")
    print(f"Report saved into: {args.report_out}")
    print("=" * 75)
    print(f"{'Metric':<25s} | {'CV Planner':<12s} | {'kNN/CV Planner':<14s} | {'Delta'}")
    print("-" * 75)
    mms_delta = summary["knn_cv_planner"]["mean_mms"] - summary["cv_planner"]["mean_mms"]
    l2_delta = summary["knn_cv_planner"]["median_l2"] - summary["cv_planner"]["median_l2"]
    print(f"{'Mean MMS (multi-ref)':<25s} | {summary['cv_planner']['mean_mms']:<12.2f} | {summary['knn_cv_planner']['mean_mms']:<14.2f} | {mms_delta:+.2f}")
    print(f"{'Median L2 error':<25s} | {summary['cv_planner']['median_l2']:<11.2f}m | {summary['knn_cv_planner']['median_l2']:<13.2f}m | {l2_delta:+.2f}m")
    print(f"{'Mean L2 error':<25s} | {summary['cv_planner']['mean_l2']:<11.2f}m | {summary['knn_cv_planner']['mean_l2']:<13.2f}m | {summary['knn_cv_planner']['mean_l2'] - summary['cv_planner']['mean_l2']:+.2f}m")
    print("-" * 75)

    print("\nPER-MANEUVER BREAKDOWN:")
    print(f"{'Maneuver':<22s} | {'Count':<5s} | {'CV MMS':<8s} | {'kNN MMS':<8s} | {'CV Med L2':<10s} | {'kNN Med L2'}")
    print("-" * 75)
    for m, d in summary["per_maneuver"].items():
        print(f"{m:<22s} | {d['count']:<5d} | {d['cv_mms']:<8.2f} | {d['knn_mms']:<8.2f} | {d['cv_median_l2']:<9.2f}m | {d['knn_median_l2']:.2f}m")
    print("-" * 75)

    return 0


if __name__ == "__main__":
    sys.exit(main())
