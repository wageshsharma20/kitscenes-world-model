#!/usr/bin/env python3
"""Rigorous paired evaluation script comparing Constant Velocity (CV) against kNN Trajectory Retrieval.

Evaluates on the official KITScenes-LongTail validation split (N=100 scenarios).
Outputs paired CSV results, bootstrap 95% confidence intervals, and hypothesis tests.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitscenes import knn_trajectory, spec

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"


def run_paired_evaluation(
    val_path: Path = DATA_DIR / "val_trajectories.jsonl",
    train_path: Path = DATA_DIR / "train_trajectories.jsonl",
    bootstrap_samples: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    assert val_path.is_file(), f"Validation file missing: {val_path}"
    assert train_path.is_file(), f"Training donor file missing: {train_path}"

    with open(val_path, "r", encoding="utf-8") as f:
        val_records = [json.loads(line) for line in f if line.strip()]

    cv_planner = knn_trajectory.ConstantVelocityPlanner()
    knn_planner = knn_trajectory.KNNTrajectoryPlanner(donor_path=train_path)

    rows: List[Dict[str, Any]] = []
    cv_means, knn_means = [], []
    cv_ends, knn_ends = [], []
    cv_meds, knn_meds = [], []

    for rec in val_records:
        sid = rec["scenario_id"]
        past = rec["past"]
        inst = rec.get("driving_instruction", "drive straight on")
        stype = rec.get("scenario_type", "")
        expert = np.asarray(rec["expert_like"], dtype=float)

        plan_cv = cv_planner.plan(past)
        m_cv = knn_trajectory.compute_l2_metrics(plan_cv, expert)

        plan_knn = knn_planner.plan(sid, past, inst, scenario_type=stype)
        m_knn = knn_trajectory.compute_l2_metrics(plan_knn, expert)

        gt_dist = float(np.linalg.norm(expert[-1]))
        cv_dist = float(np.linalg.norm(plan_cv[-1]))
        knn_dist = float(np.linalg.norm(plan_knn[-1]))

        diff_mean = m_cv["l2_mean"] - m_knn["l2_mean"]
        diff_end = m_cv["l2_endpoint"] - m_knn["l2_endpoint"]

        rows.append({
            "scenario_id": sid,
            "driving_instruction": inst,
            "scenario_type": stype,
            "gt_distance_m": round(gt_dist, 2),
            "cv_predicted_dist_m": round(cv_dist, 2),
            "knn_predicted_dist_m": round(knn_dist, 2),
            "cv_mean_l2_m": round(m_cv["l2_mean"], 3),
            "knn_mean_l2_m": round(m_knn["l2_mean"], 3),
            "diff_mean_l2_m": round(diff_mean, 3),
            "cv_endpoint_l2_m": round(m_cv["l2_endpoint"], 3),
            "knn_endpoint_l2_m": round(m_knn["l2_endpoint"], 3),
            "diff_endpoint_l2_m": round(diff_end, 3),
            "knn_won_mean": bool(diff_mean > 0),
            "knn_won_endpoint": bool(diff_end > 0),
        })

        cv_means.append(m_cv["l2_mean"])
        knn_means.append(m_knn["l2_mean"])
        cv_ends.append(m_cv["l2_endpoint"])
        knn_ends.append(m_knn["l2_endpoint"])
        cv_meds.append(m_cv["l2_median"])
        knn_meds.append(m_knn["l2_median"])

    # Save paired CSV
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "validation_paired_100_scenarios.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    cv_means = np.array(cv_means)
    knn_means = np.array(knn_means)
    cv_ends = np.array(cv_ends)
    knn_ends = np.array(knn_ends)
    n = len(rows)

    # Paired differences
    diff_mean_arr = cv_means - knn_means
    diff_end_arr = cv_ends - knn_ends

    # Win rate
    win_mean = int(np.sum(diff_mean_arr > 0))
    win_end = int(np.sum(diff_end_arr > 0))

    # Bootstrap resampling preserving pairing (same resample index sampled simultaneously)
    rng = np.random.default_rng(seed)
    boot_mean_diff = np.empty(bootstrap_samples)
    boot_mean_rel = np.empty(bootstrap_samples)
    boot_end_diff = np.empty(bootstrap_samples)
    boot_end_rel = np.empty(bootstrap_samples)

    for b in range(bootstrap_samples):
        idx = rng.choice(n, size=n, replace=True)
        c_m, k_m = cv_means[idx], knn_means[idx]
        mean_c = np.mean(c_m)
        mean_k = np.mean(k_m)
        boot_mean_diff[b] = mean_c - mean_k
        boot_mean_rel[b] = (mean_c - mean_k) / mean_c * 100.0

        c_e, k_e = cv_ends[idx], knn_ends[idx]
        end_c = np.mean(c_e)
        end_k = np.mean(k_e)
        boot_end_diff[b] = end_c - end_k
        boot_end_rel[b] = (end_c - end_k) / end_c * 100.0

    ci_mean_diff = np.percentile(boot_mean_diff, [2.5, 97.5])
    ci_mean_rel = np.percentile(boot_mean_rel, [2.5, 97.5])
    ci_end_diff = np.percentile(boot_end_diff, [2.5, 97.5])
    ci_end_rel = np.percentile(boot_end_rel, [2.5, 97.5])

    # Paired t-test
    t_stat_m = float(np.mean(diff_mean_arr) / (np.std(diff_mean_arr, ddof=1) / math.sqrt(n)))
    p_t_m = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat_m) / math.sqrt(2.0))))

    t_stat_e = float(np.mean(diff_end_arr) / (np.std(diff_end_arr, ddof=1) / math.sqrt(n)))
    p_t_e = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat_e) / math.sqrt(2.0))))

    # Wilcoxon signed-rank test
    def _wilcoxon(d: np.ndarray):
        nz = d[d != 0]
        nn = len(nz)
        ranks = np.argsort(np.argsort(np.abs(nz))) + 1.0
        w_pos = np.sum(ranks[nz > 0])
        w_neg = np.sum(ranks[nz < 0])
        w = min(w_pos, w_neg)
        mean_w = nn * (nn + 1) / 4.0
        sd_w = math.sqrt(nn * (nn + 1) * (2 * nn + 1) / 24.0)
        z = (w - mean_w) / sd_w
        pval = 2.0 * 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        return w_pos, w_neg, z, pval

    w_pos_m, w_neg_m, z_m, p_w_m = _wilcoxon(diff_mean_arr)
    w_pos_e, w_neg_e, z_e, p_w_e = _wilcoxon(diff_end_arr)

    return {
        "n": n,
        "csv_path": str(csv_path),
        "cv_mean_l2": float(np.mean(cv_means)),
        "knn_mean_l2": float(np.mean(knn_means)),
        "abs_mean_reduction_m": float(np.mean(diff_mean_arr)),
        "rel_mean_reduction_pct": float((np.mean(cv_means) - np.mean(knn_means)) / np.mean(cv_means) * 100.0),
        "ci_mean_rel_pct": [float(ci_mean_rel[0]), float(ci_mean_rel[1])],
        "ci_mean_diff_m": [float(ci_mean_diff[0]), float(ci_mean_diff[1])],
        "cv_end_l2": float(np.mean(cv_ends)),
        "knn_end_l2": float(np.mean(knn_ends)),
        "abs_end_reduction_m": float(np.mean(diff_end_arr)),
        "rel_end_reduction_pct": float((np.mean(cv_ends) - np.mean(knn_ends)) / np.mean(cv_ends) * 100.0),
        "ci_end_rel_pct": [float(ci_end_rel[0]), float(ci_end_rel[1])],
        "ci_end_diff_m": [float(ci_end_diff[0]), float(ci_end_diff[1])],
        "win_mean_count": win_mean,
        "win_end_count": win_end,
        "p_paired_t_mean": p_t_m,
        "p_wilcoxon_mean": p_w_m,
        "p_paired_t_end": p_t_e,
        "p_wilcoxon_end": p_w_e,
    }


if __name__ == "__main__":
    out = run_paired_evaluation()
    print("Paired Evaluation Completed:")
    print(json.dumps(out, indent=2))
