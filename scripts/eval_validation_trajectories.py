"""Evaluate Constant Velocity (CV) vs kNN Trajectory Planner on all 100 Validation Scenarios.

Uses the official KITScenes MMS metric (velocity-scaled longitudinal/lateral matching,
direction checks, comfort penalties) and L2 trajectory displacement metrics against
real ground-truth expert trajectories and negative references.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np

from kitscenes.knn_trajectory import (
    ConstantVelocityPlanner,
    KNNTrajectoryPlanner,
    classify_instruction,
    compute_l2_metrics,
    compute_mms_score,
)


def evaluate_all_100_val(
    val_path: Path = Path("data/val_trajectories.jsonl"),
    train_path: Path = Path("data/train_trajectories.jsonl"),
) -> dict:
    if not val_path.exists():
        raise FileNotFoundError(f"Missing {val_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}")

    # Load 100 validation records
    val_records = []
    with open(val_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_records.append(json.loads(line))

    print(f"Loaded {len(val_records)} genuine validation scenarios from {val_path}")

    cv_planner = ConstantVelocityPlanner()
    knn_planner = KNNTrajectoryPlanner(donor_path=train_path)

    cv_scores = {"mms": [], "l2_mean": [], "l2_median": []}
    knn_scores = {"mms": [], "l2_mean": [], "l2_median": []}

    by_type = defaultdict(lambda: {"cv_mms": [], "knn_mms": [], "cv_l2_med": [], "knn_l2_med": []})
    by_maneuver = defaultdict(lambda: {"cv_mms": [], "knn_mms": [], "cv_l2_med": [], "knn_l2_med": []})

    scenario_results = []

    for rec in val_records:
        sid = rec["scenario_id"]
        past = rec["past"]
        expert = rec["expert_like"]
        inst = rec.get("driving_instruction", "")
        stype = rec.get("scenario_type", "")
        maneuver = classify_instruction(inst)

        # 1. Constant Velocity
        plan_cv = cv_planner.plan(past)
        mms_cv = compute_mms_score(
            plan_cv,
            expert,
            wrong_speed=rec.get("wrong_speed"),
            neglect_instruction=rec.get("neglect_instruction"),
            off_road=rec.get("off_road"),
            crash=rec.get("crash"),
        )
        l2_cv = compute_l2_metrics(plan_cv, expert)

        # 2. kNN / CV Planner
        plan_knn = knn_planner.plan(sid, past, inst)
        mms_knn = compute_mms_score(
            plan_knn,
            expert,
            wrong_speed=rec.get("wrong_speed"),
            neglect_instruction=rec.get("neglect_instruction"),
            off_road=rec.get("off_road"),
            crash=rec.get("crash"),
        )
        l2_knn = compute_l2_metrics(plan_knn, expert)

        cv_scores["mms"].append(mms_cv)
        cv_scores["l2_mean"].append(l2_cv["l2_mean"])
        cv_scores["l2_median"].append(l2_cv["l2_median"])

        knn_scores["mms"].append(mms_knn)
        knn_scores["l2_mean"].append(l2_knn["l2_mean"])
        knn_scores["l2_median"].append(l2_knn["l2_median"])

        by_type[stype]["cv_mms"].append(mms_cv)
        by_type[stype]["knn_mms"].append(mms_knn)
        by_type[stype]["cv_l2_med"].append(l2_cv["l2_median"])
        by_type[stype]["knn_l2_med"].append(l2_knn["l2_median"])

        by_maneuver[maneuver]["cv_mms"].append(mms_cv)
        by_maneuver[maneuver]["knn_mms"].append(mms_knn)
        by_maneuver[maneuver]["cv_l2_med"].append(l2_cv["l2_median"])
        by_maneuver[maneuver]["knn_l2_med"].append(l2_knn["l2_median"])

        scenario_results.append({
            "scenario_id": sid,
            "scenario_type": stype,
            "maneuver": maneuver,
            "driving_instruction": inst,
            "cv_mms": round(mms_cv, 2),
            "knn_mms": round(mms_knn, 2),
            "cv_l2_mean": round(l2_cv["l2_mean"], 2),
            "knn_l2_mean": round(l2_knn["l2_mean"], 2),
            "cv_l2_median": round(l2_cv["l2_median"], 2),
            "knn_l2_median": round(l2_knn["l2_median"], 2),
        })

    # Summary
    summary = {
        "num_scenarios": len(val_records),
        "cv": {
            "mms_mean": float(np.mean(cv_scores["mms"])),
            "mms_median": float(np.median(cv_scores["mms"])),
            "l2_mean": float(np.mean(cv_scores["l2_mean"])),
            "l2_median": float(np.median(cv_scores["l2_median"])),
        },
        "knn": {
            "mms_mean": float(np.mean(knn_scores["mms"])),
            "mms_median": float(np.median(knn_scores["mms"])),
            "l2_mean": float(np.mean(knn_scores["l2_mean"])),
            "l2_median": float(np.median(knn_scores["l2_median"])),
        },
        "by_type": {
            k: {
                "count": len(v["cv_mms"]),
                "cv_mms": float(np.mean(v["cv_mms"])),
                "knn_mms": float(np.mean(v["knn_mms"])),
                "cv_l2_med": float(np.mean(v["cv_l2_med"])),
                "knn_l2_med": float(np.mean(v["knn_l2_med"])),
            }
            for k, v in by_type.items()
        },
        "by_maneuver": {
            k: {
                "count": len(v["cv_mms"]),
                "cv_mms": float(np.mean(v["cv_mms"])),
                "knn_mms": float(np.mean(v["knn_mms"])),
                "cv_l2_med": float(np.mean(v["cv_l2_med"])),
                "knn_l2_med": float(np.mean(v["knn_l2_med"])),
            }
            for k, v in by_maneuver.items()
        },
        "scenarios": scenario_results,
    }

    out_path = Path("reports/val_100_trajectories_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved full 100-scenario evaluation report to: {out_path}\n")

    # Print clean formatted tables
    print("=" * 95)
    print("OVERALL 100-SCENARIO EVALUATION SUMMARY")
    print("=" * 95)
    print(f"{'Metric':<25} | {'Constant Velocity (CV)':<30} | {'kNN / CV Planner':<30}")
    print("-" * 95)
    print(f"{'Mean MMS':<25} | {summary['cv']['mms_mean']:<30.2f} | {summary['knn']['mms_mean']:<30.2f}")
    print(f"{'Median MMS':<25} | {summary['cv']['mms_median']:<30.2f} | {summary['knn']['mms_median']:<30.2f}")
    print(f"{'Mean L2 (m)':<25} | {summary['cv']['l2_mean']:<30.2f} | {summary['knn']['l2_mean']:<30.2f}")
    print(f"{'Median L2 (m)':<25} | {summary['cv']['l2_median']:<30.2f} | {summary['knn']['l2_median']:<30.2f}")
    print("=" * 95)

    print("\nBREAKDOWN BY MANEUVER CLASS (100 VALIDATION SCENARIOS):")
    print(f"{'Maneuver':<20} | {'Count':<6} | {'CV MMS':<8} | {'KNN MMS':<8} | {'CV L2(med)':<12} | {'KNN L2(med)':<12}")
    print("-" * 75)
    for m, d in sorted(summary["by_maneuver"].items(), key=lambda x: -x[1]["count"]):
        print(f"{m:<20} | {d['count']:<6} | {d['cv_mms']:<8.2f} | {d['knn_mms']:<8.2f} | {d['cv_l2_med']:^12.2f} | {d['knn_l2_med']:^12.2f}")
    print("-" * 75)

    print("\nBREAKDOWN BY SCENARIO TYPE (100 VALIDATION SCENARIOS):")
    print(f"{'Scenario Type':<26} | {'Count':<6} | {'CV MMS':<8} | {'KNN MMS':<8} | {'CV L2(med)':<12} | {'KNN L2(med)':<12}")
    print("-" * 81)
    for t, d in sorted(summary["by_type"].items(), key=lambda x: -x[1]["count"]):
        print(f"{t:<26} | {d['count']:<6} | {d['cv_mms']:<8.2f} | {d['knn_mms']:<8.2f} | {d['cv_l2_med']:^12.2f} | {d['knn_l2_med']:^12.2f}")
    print("=" * 81)

    return summary


if __name__ == "__main__":
    evaluate_all_100_val()
