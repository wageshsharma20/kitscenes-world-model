import json
import hashlib
import numpy as np
from pathlib import Path
from scipy import stats

report_path = Path("reports/submission_archive_provenance_audit.json")
md_path = Path("reports/submission_archive_provenance_audit.md")

# 1. Archive Provenance Audit
zip_path = Path("submissions/videos.zip")
hasher = hashlib.sha256()
with open(zip_path, "rb") as f:
    while chunk := f.read(1024 * 1024):
        hasher.update(chunk)
zip_sha256 = hasher.hexdigest()
zip_size = zip_path.stat().st_size

# 2. Statistical Qualification Investigation
with open("reports/phase5_5_2_final_audit_and_freeze.json") as f:
    eval_data = json.load(f)

scenarios = ['0000900', '0000901', '0000902', '0000903', '0000904', '0000905', '0000906', '0000911', '0000919', '0000984']
b_psnr = [eval_data['per_scenario'][s]['project_baseline']['mean_psnr_common'] for s in scenarios]
v_psnr = [eval_data['per_scenario'][s]['vanilla_svd']['mean_psnr_common'] for s in scenarios]
c_psnr = [eval_data['per_scenario'][s]['cssc_only']['mean_psnr_common'] for s in scenarios]

diff_bv = np.array(b_psnr) - np.array(v_psnr)
diff_bc = np.array(b_psnr) - np.array(c_psnr)

mean_bv = float(np.mean(diff_bv))
std_bv = float(np.std(diff_bv, ddof=1))
se_bv = float(std_bv / np.sqrt(len(diff_bv)))
t_stat_bv = float(mean_bv / se_bv)
p_two_bv = float(2 * (1 - stats.t.cdf(abs(t_stat_bv), df=9)))
p_one_bv = float(1 - stats.t.cdf(t_stat_bv, df=9))
t_ci_95_bv = [float(mean_bv - stats.t.ppf(0.975, df=9) * se_bv), float(mean_bv + stats.t.ppf(0.975, df=9) * se_bv)]
w_res_bv = stats.wilcoxon(diff_bv)

# Bootstrap methods
res_perc = stats.bootstrap((diff_bv,), np.mean, confidence_level=0.95, n_resamples=10000, method='percentile', random_state=42)
res_bca = stats.bootstrap((diff_bv,), np.mean, confidence_level=0.95, n_resamples=10000, method='BCa', random_state=42)

payload = {
    "archive_audit": {
        "path": str(zip_path),
        "sha256": zip_sha256,
        "size_bytes": zip_size,
        "size_mib": round(zip_size / (1024 * 1024), 2),
        "generator_backbone": "GeometricWorldModel (procedural raster canvas)",
        "is_svd_diffusion": False,
        "diffusion_denoising_steps": 0,
        "average_video_size_kib": round((zip_size / 2400) / 1024, 2),
        "expected_svd_archive_size_gib": "0.5 to 1.2 GiB",
        "format_validation_status": "VALID (0 errors, 0 warnings)",
        "competition_pq_verdict": "Low perceptual quality expected on Q-Align (~10-25 PQ) due to procedural geometric line rendering"
    },
    "trajectories_audit": {
        "original_file": "submissions/trajectories_vlm_fallback.jsonl",
        "original_generator": "kitscenes.vlm_planner.plan_vlm_trajectory (Gemini VLM heuristic fallback, Nov 14)",
        "updated_file": "submissions/trajectories.jsonl",
        "updated_generator": "kitscenes.knn_trajectory.KNNTrajectoryPlanner (k=5 training donors, constant velocity blend)",
        "scenario_count": 400,
        "validation_status": "VALID (0 errors, 1 stationary warning)"
    },
    "statistical_qualification": {
        "n_scenarios": 10,
        "degrees_of_freedom": 9,
        "baseline_minus_vanilla_psnr": {
            "differences": [round(float(x), 4) for x in diff_bv],
            "mean_delta_psnr": mean_bv,
            "std_delta_psnr": std_bv,
            "se_delta_psnr": se_bv,
            "t_statistic": t_stat_bv,
            "p_value_two_tailed": p_two_bv,
            "p_value_one_tailed": p_one_bv,
            "parametric_student_t_95_ci": t_ci_95_bv,
            "wilcoxon_w": float(w_res_bv.statistic),
            "wilcoxon_p": float(w_res_bv.pvalue),
            "percentile_bootstrap_95_ci": [float(res_perc.confidence_interval.low), float(res_perc.confidence_interval.high)],
            "bca_bootstrap_95_ci": [float(res_bca.confidence_interval.low), float(res_bca.confidence_interval.high)],
            "win_rate": "7/10 (70%)",
            "rigorous_conclusion": "The +0.186 dB mean PSNR improvement is an operational engineering advantage with a 70% scenario win rate, but it is NOT statistically significant at the conventional alpha = 0.05 level (two-tailed p = 0.1097, Student-t 95% CI includes zero [-0.051, +0.422] dB)."
        }
    }
}

with open(report_path, "w") as f:
    json.dump(payload, f, indent=2)

# Write Markdown report
md = []
md.append("# Provenance Audit & Statistical Qualification Report")
md.append("")
md.append("**Date**: November 21, 2026  ")
md.append("**Challenge**: NeurIPS 2026 KITScenes LongTail World Modeling Challenge  ")
md.append("")
md.append("---")
md.append("")
md.append("## 1. Submission ZIP (`submissions/videos.zip`) Provenance Audit")
md.append("")
md.append("A thorough inspection of `submissions/videos.zip` (SHA-256 `26119a957a37123db3f7dac5a74fab50efb7051c8e7312f1cad1c384343ef10e`) reveals:")
md.append("")
md.append("- **Archive Size**: **85.4 MiB** across 2,400 videos (~37 KiB per video).")
md.append("- **Generating Backbone**: **`GeometricWorldModel`** (`kitscenes/video_engine.py`, procedural canvas generator).")
md.append("- **Diffusion Status**: **0 diffusion steps**. It is **not** Stable Video Diffusion (SVD).")
md.append("- **Visual Content**: 2D procedural geometric road canvas (sky gradient RGB `[65, 120, 195]`, asphalt polygon RGB `[58, 60, 63]`, solid edge lines, dashed yellow centerline).")
md.append("- **Format Compliance**: **100% Valid** (0 errors, 0 warnings under `validate_submission.py`). Meets all container, resolution ($876 \\times 568$), frame rate (5 fps), and frame count (25) rules.")
md.append("- **Perceptual Quality (PQ) Assessment**: As documented in `kitscenes/video_engine.py` line 626, procedural canvas rendering will score low on Q-Align (~10-25 PQ) compared to diffusion models. The archive serves as an emergency format-compliant fallback.")
md.append("")
md.append("---")
md.append("")
md.append("## 2. Trajectories (`submissions/trajectories.jsonl`) Provenance Audit & Correction")
md.append("")
md.append("- **Original File State**: Generated on Nov 14, 2026 by `kitscenes.vlm_planner.plan_vlm_trajectory` (Gemini VLM heuristic tactical planner).")
md.append("- **Audit Finding**: Did not match the frozen `KNNTrajectoryPlanner` ($k=5$ donor retrieval).")
md.append("- **Action Taken**: Replaced `submissions/trajectories.jsonl` with 100% verified output from `KNNTrajectoryPlanner` across all 400 test scenarios (backed up original to `submissions/trajectories_vlm_fallback.jsonl`).")
md.append("- **Validation**: Passed `validate_submission.py` with **0 errors, 400/400 valid scenario records**.")
md.append("")
md.append("---")
md.append("")
md.append("## 3. Statistical Qualification: Bootstrap CI vs. Paired t-Test Investigation")
md.append("")
md.append("Across the 10 evaluated scenarios:")
md.append("- Differences $\\Delta = \\text{Baseline} - \\text{Vanilla}$: `[-0.0097, +0.0147, +0.3461, +0.0496, -0.1472, +0.9464, +0.1295, +0.2252, -0.1463, +0.4483] dB`")
md.append("- **Mean $\\Delta$ PSNR**: **+0.1857 dB** ($s = 0.3308\\text{ dB}$, $\\text{SE} = 0.1046\\text{ dB}$)")
md.append("- **t-Statistic**: $t = 1.7746$ ($df = 9$)")
md.append("- **Two-Tailed p-Value**: **$p = 0.1097$** (does not reach $\\alpha = 0.05$)")
md.append("- **One-Tailed p-Value**: $p = 0.0548$ (borderline)")
md.append("- **Parametric Student-$t$ 95% CI**: **`[-0.0510, +0.4223] dB`** (includes zero)")
md.append("- **Wilcoxon Signed-Rank Test**: $W = 12.0$, $p = 0.1309$")
md.append("- **Non-Parametric Percentile Bootstrap 95% CI**: `[+0.0110, +0.3990] dB`")
md.append("")
md.append("### Root Cause of CI Disagreement")
md.append("1. **Small Sample Degrees of Freedom**: With $N=10$, Student's $t$ distribution requires a heavy critical multiplier ($t_{0.975, 9} = 2.262$), widening the confidence interval to include negative values down to $-0.051\\text{ dB}$.")
md.append("2. **Bootstrap Skewness**: In a small sample where 7/10 points are positive and one point is an outlier (+0.946 dB in `0000905`), unstudentized percentile bootstrap underestimates tail variance, yielding a strictly positive interval.")
md.append("3. **Conclusion**: The user's qualification is mathematically confirmed: **The +0.186 dB improvement is a consistent operational engineering advantage (70% win rate), but NOT a statistically proven rejection of the null hypothesis at $\\alpha = 0.05$**.")
md.append("")

with open(md_path, "w") as f:
    f.write("\n".join(md))

print("Audit script completed successfully.")
