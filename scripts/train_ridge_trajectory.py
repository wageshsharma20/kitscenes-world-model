#!/usr/bin/env python3
"""Train a smooth polynomial-basis Ridge trajectory planner on train_trajectories.jsonl.

Strictly fits feature normalization and Ridge weights on the 500 training split
scenarios (using 5-fold Train CV to select regularization lambda), evaluates MMS and L2
on the 100 validation scenarios, and writes a validated 400-scenario test JSONL plan.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitscenes import spec
from kitscenes.knn_trajectory import (
    classify_instruction,
    compute_l2_metrics,
    compute_mms_score,
)
from kitscenes.trajectories import inspect_trajectories

MANEUVERS: Tuple[str, ...] = (
    "straight",
    "turn_left",
    "turn_right",
    "lane_change_left",
    "lane_change_right",
    "lane_change_middle",
    "overtake",
    "u_turn",
    "stop",
)

SCENARIO_TYPES: Tuple[str, ...] = (
    "0 specifically selected",
    "1 heavy rain",
    "2 construction zone",
    "3 overtake or lane change",
    "4 intersection",
    "5 nighttime",
    "6 snow & wintry mix",
)

KEYWORDS: Tuple[str, ...] = (
    "left",
    "right",
    "straight",
    "slow",
    "fast",
    "stop",
    "keep",
    "follow",
    "park",
    "merge",
    "u-turn",
)


def extract_kinematic_features(
    past_points: Sequence[Sequence[float]],
    instruction: str,
    scenario_type: str,
) -> np.ndarray:
    """Extract causal t<=0 kinematic, curvature, and instruction interaction features."""
    past = np.asarray(past_points, dtype=np.float64)
    diffs = np.diff(past, axis=0) / 0.2
    spds = np.linalg.norm(diffs, axis=1)

    vx_1, vy_1 = float(diffs[-1, 0]), float(diffs[-1, 1])
    v_3 = np.mean(diffs[-min(3, len(diffs)) :], axis=0)
    spd_3 = float(np.linalg.norm(v_3))
    v_mid = np.mean(diffs[-min(8, len(diffs)) : -min(4, len(diffs) - 1)], axis=0) if len(diffs) >= 8 else v_3
    spd_mid = float(np.linalg.norm(v_mid))
    v_far = np.mean(diffs[: min(5, len(diffs))], axis=0)
    spd_far = float(np.linalg.norm(v_far))

    acc_1 = (float(spds[-1]) - float(spds[-min(4, len(spds))])) / 0.6
    acc_long = (spd_3 - spd_far) / 3.0
    # Compute yaw strictly where speed > 0.8 m/s so stationary GPS jitter never spikes dyaw
    low_spd = spds < 0.8
    yaws = np.where(low_spd, 0.0, np.arctan2(diffs[:, 1], diffs[:, 0]))
    if np.all(low_spd[-min(6, len(low_spd)) :]):
        dyaw_1 = 0.0
    else:
        dyaw_1 = float(np.clip((float(yaws[-1]) - float(yaws[-min(5, len(yaws))])) / 0.8, -1.0, 1.0))
    if np.all(low_spd):
        dyaw_long = 0.0
    else:
        dyaw_long = float(np.clip((float(np.mean(yaws[-min(3, len(yaws)) :])) - float(np.mean(yaws[: min(5, len(yaws))]))) / 3.0, -0.8, 0.8))
    lat_vel = float(v_3[1]) if spd_3 >= 0.8 else 0.0
    lat_acc = float(diffs[-1, 1] - diffs[-min(6, len(diffs)), 1]) if spd_3 >= 0.8 else 0.0

    maneuver = classify_instruction(instruction)
    oh_m = [1.0 if maneuver == mv else 0.0 for mv in MANEUVERS]
    oh_s = [1.0 if scenario_type == sv else 0.0 for sv in SCENARIO_TYPES]
    inst_l = instruction.lower()
    kw = [float(w in inst_l) for w in KEYWORDS]

    idx_mid = max(0, len(past) - 11)
    idx_qtr = max(0, len(past) - 6)
    base = [
        1.0,
        float(spds[-1]),
        spd_3,
        spd_mid,
        spd_far,
        float(np.sqrt(max(spd_3, 0.0))),
        acc_1,
        acc_long,
        vx_1,
        vy_1,
        float(v_3[0]),
        lat_vel,
        lat_acc,
        dyaw_1,
        dyaw_long,
        spd_3 * dyaw_1,
        spd_3 * acc_1,
        float(past[idx_qtr, 0]),
        float(past[idx_qtr, 1]),
        float(past[idx_mid, 0]),
        float(past[idx_mid, 1]),
        float(past[0, 0]),
        float(past[0, 1]),
    ]
    interactions: List[float] = []
    for k in range(len(MANEUVERS)):
        interactions.extend([spd_3 * oh_m[k], acc_1 * oh_m[k], dyaw_1 * oh_m[k], spd_3 * spd_3 * oh_m[k]])
    return np.asarray(base + oh_m + oh_s + kw + interactions, dtype=np.float64)


def smooth_projection_matrix() -> np.ndarray:
    """Orthogonal projection onto quartic zero-intercept temporal basis [t, t^2, t^3, t^4]."""
    ts = np.arange(1, spec.WAYPOINT_COUNT + 1, dtype=np.float64) * 0.2
    basis = np.column_stack([ts, ts**2, ts**3, ts**4])
    return basis @ np.linalg.pinv(basis)


def fit_ridge(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x_tr.mean(axis=0)
    sd = x_tr.std(axis=0) + 1e-6
    x_norm = (x_tr - mu) / sd
    x_norm[:, 0] = 1.0
    reg = np.eye(x_tr.shape[1]) * lam
    reg[0, 0] = 0.0
    weights = np.linalg.solve(x_norm.T @ x_norm + reg, x_norm.T @ y_tr)
    return weights, mu, sd


def predict_ridge(
    x: np.ndarray,
    weights: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    p_smooth: np.ndarray,
) -> np.ndarray:
    x_norm = np.clip((x - mu) / sd, -4.0, 4.0)
    x_norm[:, 0] = 1.0
    raw = (x_norm @ weights).reshape(-1, spec.WAYPOINT_COUNT, 2)
    smoothed = np.einsum("ij,njk->nik", p_smooth, raw)
    # Enforce monotonic forward longitudinal travel (avoid negative reverse jumps near stationary)
    smoothed[:, :, 0] = np.maximum(smoothed[:, :, 0], 0.0)
    # Enforce physical speed bound (<= 38 m/s) and unique monotonic steps after 3-decimal rounding
    for i in range(smoothed.shape[0]):
        pts = smoothed[i]
        # Scale down only if peak step speed exceeds 38 m/s (137 km/h)
        step_d = np.linalg.norm(np.diff(np.vstack([np.zeros((1, 2)), pts]), axis=0), axis=1)
        max_spd = float(np.max(step_d) / 0.2)
        if max_spd > 38.0:
            pts *= 38.0 / max_spd
        # Ensure minimum 0.02m/step monotonic forward floor so rounded steps never repeat identically
        dx = np.diff(np.concatenate([[0.0], pts[:, 0]]))
        if np.any(np.abs(dx) < 2e-3) and np.max(np.abs(pts)) >= 0.5:
            for j in range(len(pts)):
                prev_x = 0.0 if j == 0 else pts[j - 1, 0]
                if abs(pts[j, 0] - prev_x) < 2e-3 and abs(pts[j, 1] - (0.0 if j == 0 else pts[j - 1, 1])) < 2e-3:
                    pts[j, 0] = prev_x + 2e-3
        elif np.max(np.abs(pts)) < 0.05:
            pts[:, 0] = np.arange(1, spec.WAYPOINT_COUNT + 1, dtype=np.float64) * 2e-3
    return smoothed


def evaluate_mms(recs: Sequence[Dict[str, Any]], preds: np.ndarray) -> Tuple[float, float, float]:
    mms_vals, l2_vals = [], []
    for i, r in enumerate(recs):
        mms_vals.append(
            compute_mms_score(
                preds[i],
                np.asarray(r["expert_like"], dtype=np.float64),
                wrong_speed=np.asarray(r["wrong_speed"]) if r.get("wrong_speed") else None,
                neglect_instruction=np.asarray(r["neglect_instruction"]) if r.get("neglect_instruction") else None,
                off_road=np.asarray(r["off_road"]) if r.get("off_road") else None,
                crash=np.asarray(r["crash"]) if r.get("crash") else None,
            )
        )
        l2_vals.append(compute_l2_metrics(preds[i], np.asarray(r["expert_like"]))["l2_mean"])
    return float(np.mean(mms_vals)), float(np.median(l2_vals)), float(np.mean(l2_vals))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fit and validate polynomial-basis Ridge trajectory planner.")
    parser.add_argument("--train", type=Path, default=PROJECT_ROOT / "data" / "train_trajectories.jsonl")
    parser.add_argument("--val", type=Path, default=PROJECT_ROOT / "data" / "val_trajectories.jsonl")
    parser.add_argument("--history", type=Path, default=PROJECT_ROOT / "data" / "history.jsonl")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "submissions" / "trajectories_ridge.jsonl")
    args = parser.parse_args(argv)

    with open(args.train) as f:
        tr_all = [json.loads(line) for line in f if line.strip()]
    tr = [
        r
        for r in tr_all
        if r.get("expert_like")
        and len(r["expert_like"]) == spec.WAYPOINT_COUNT
        and not np.any(np.asarray(r["expert_like"]) <= -90.0)
    ]
    with open(args.val) as f:
        va = [json.loads(line) for line in f if line.strip()]

    x_tr = np.vstack([extract_kinematic_features(r["past"], r.get("driving_instruction", ""), r.get("scenario_type", "")) for r in tr])
    y_tr = np.vstack([np.asarray(r["expert_like"], dtype=np.float64).reshape(-1) for r in tr])
    x_va = np.vstack([extract_kinematic_features(r["past"], r.get("driving_instruction", ""), r.get("scenario_type", "")) for r in va])
    p_smooth = smooth_projection_matrix()

    best_lam = 1.0
    best_cv = -1.0
    idx = np.arange(len(tr))
    for lam in (0.3, 0.5, 1.0, 2.5, 5.0, 10.0):
        cv_preds = np.zeros((len(tr), spec.WAYPOINT_COUNT, 2), dtype=np.float64)
        for fold in range(5):
            te_m = (idx % 5) == fold
            tr_m = ~te_m
            w_f, mu_f, sd_f = fit_ridge(x_tr[tr_m], y_tr[tr_m], lam)
            cv_preds[te_m] = predict_ridge(x_tr[te_m], w_f, mu_f, sd_f, p_smooth)
        cv_mms, cv_l2m, _ = evaluate_mms(tr, cv_preds)
        print(f"5-fold Train CV lam={lam:4.1f}: MMS={cv_mms:.3f} | L2_med={cv_l2m:.3f}")
        if cv_mms > best_cv:
            best_cv = cv_mms
            best_lam = lam

    w, mu, sd = fit_ridge(x_tr, y_tr, best_lam)
    va_preds = predict_ridge(x_va, w, mu, sd, p_smooth)
    va_mms, va_l2med, va_l2mean = evaluate_mms(va, va_preds)
    print(f"\nSelected lam={best_lam} -> Val MMS={va_mms:.3f} | L2_med={va_l2med:.3f} | L2_mean={va_l2mean:.3f}")

    by_m: Dict[str, List[float]] = defaultdict(list)
    for i, r in enumerate(va):
        mv = classify_instruction(r.get("driving_instruction", ""))
        m_val = compute_mms_score(
            va_preds[i],
            np.asarray(r["expert_like"], dtype=np.float64),
            wrong_speed=np.asarray(r["wrong_speed"]) if r.get("wrong_speed") else None,
            neglect_instruction=np.asarray(r["neglect_instruction"]) if r.get("neglect_instruction") else None,
            off_road=np.asarray(r["off_road"]) if r.get("off_road") else None,
            crash=np.asarray(r["crash"]) if r.get("crash") else None,
        )
        by_m[mv].append(m_val)
    for mv in sorted(by_m):
        print(f"  {mv:<20s} (n={len(by_m[mv]):2d}): MMS={np.mean(by_m[mv]):.2f}")

    with open(args.history) as f:
        hist = [json.loads(line) for line in f if line.strip()]
    hist.sort(key=lambda r: str(r["scenario_id"]).zfill(7))
    x_te = np.vstack([extract_kinematic_features(r["past_trajectory"], r.get("driving_instruction", ""), r.get("scenario_type", "")) for r in hist])
    te_preds = predict_ridge(x_te, w, mu, sd, p_smooth)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, r in enumerate(hist):
            pts = [[round(float(x), spec.TRAJECTORY_PRECISION), round(float(y), spec.TRAJECTORY_PRECISION)] for x, y in te_preds[i]]
            rec = {"scenario_id": str(r["scenario_id"]).zfill(7), "future_trajectory": pts}
            f.write(json.dumps(rec) + "\n")

    rep = inspect_trajectories(args.out)
    print(f"\nValidated {args.out}: ok={rep.ok}, errors={rep.errors}, warnings={rep.warnings}")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
