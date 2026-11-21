# Provenance Audit & Statistical Qualification Report

**Date**: November 21, 2026  
**Challenge**: NeurIPS 2026 KITScenes LongTail World Modeling Challenge  

---

## 1. Submission ZIP (`submissions/videos.zip`) Provenance Audit

A thorough inspection of `submissions/videos.zip` (SHA-256 `26119a957a37123db3f7dac5a74fab50efb7051c8e7312f1cad1c384343ef10e`) reveals:

- **Archive Size**: **85.4 MiB** across 2,400 videos (~37 KiB per video).
- **Generating Backbone**: **`GeometricWorldModel`** (`kitscenes/video_engine.py`, procedural canvas generator).
- **Diffusion Status**: **0 diffusion steps**. It is **not** Stable Video Diffusion (SVD).
- **Visual Content**: 2D procedural geometric road canvas (sky gradient RGB `[65, 120, 195]`, asphalt polygon RGB `[58, 60, 63]`, solid edge lines, dashed yellow centerline).
- **Format Compliance**: **100% Valid** (0 errors, 0 warnings under `validate_submission.py`). Meets all container, resolution ($876 \times 568$), frame rate (5 fps), and frame count (25) rules.
- **Perceptual Quality (PQ) Assessment**: As documented in `kitscenes/video_engine.py` line 626, procedural canvas rendering will score low on Q-Align (~10-25 PQ) compared to diffusion models. The archive serves as an emergency format-compliant fallback.

---

## 2. Trajectories (`submissions/trajectories.jsonl`) Provenance Audit & Correction

- **Original File State**: Generated on Nov 14, 2026 by `kitscenes.vlm_planner.plan_vlm_trajectory` (Gemini VLM heuristic tactical planner).
- **Audit Finding**: Did not match the frozen `KNNTrajectoryPlanner` ($k=5$ donor retrieval).
- **Action Taken**: Replaced `submissions/trajectories.jsonl` with 100% verified output from `KNNTrajectoryPlanner` across all 400 test scenarios (backed up original to `submissions/trajectories_vlm_fallback.jsonl`).
- **Validation**: Passed `validate_submission.py` with **0 errors, 400/400 valid scenario records**.

---

## 3. Statistical Qualification: Bootstrap CI vs. Paired t-Test Investigation

Across the 10 evaluated scenarios:
- Differences $\Delta = \text{Baseline} - \text{Vanilla}$: `[-0.0097, +0.0147, +0.3461, +0.0496, -0.1472, +0.9464, +0.1295, +0.2252, -0.1463, +0.4483] dB`
- **Mean $\Delta$ PSNR**: **+0.1857 dB** ($s = 0.3308\text{ dB}$, $\text{SE} = 0.1046\text{ dB}$)
- **t-Statistic**: $t = 1.7746$ ($df = 9$)
- **Two-Tailed p-Value**: **$p = 0.1097$** (does not reach $\alpha = 0.05$)
- **One-Tailed p-Value**: $p = 0.0548$ (borderline)
- **Parametric Student-$t$ 95% CI**: **`[-0.0510, +0.4223] dB`** (includes zero)
- **Wilcoxon Signed-Rank Test**: $W = 12.0$, $p = 0.1309$
- **Non-Parametric Percentile Bootstrap 95% CI**: `[+0.0110, +0.3990] dB`

### Root Cause of CI Disagreement
1. **Small Sample Degrees of Freedom**: With $N=10$, Student's $t$ distribution requires a heavy critical multiplier ($t_{0.975, 9} = 2.262$), widening the confidence interval to include negative values down to $-0.051\text{ dB}$.
2. **Bootstrap Skewness**: In a small sample where 7/10 points are positive and one point is an outlier (+0.946 dB in `0000905`), unstudentized percentile bootstrap underestimates tail variance, yielding a strictly positive interval.
3. **Conclusion**: The user's qualification is mathematically confirmed: **The +0.186 dB improvement is a consistent operational engineering advantage (70% win rate), but NOT a statistically proven rejection of the null hypothesis at $\alpha = 0.05$**.
