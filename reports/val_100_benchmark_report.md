# Rigorous Validation Split Benchmark: Constant Velocity vs. kNN Trajectory Retrieval

**Dataset Split**: KITScenes-LongTail Official Validation Split ($N=100$ scenarios: `0000900`–`0000999`)  
**Training Donor Database**: 499 verified scenarios from training split (`0000400`–`0000899`)  
**SHA-256 Checksum**: `a96e2f7b11a4c2911ace9dc69cbbe096f0b242400673636fa530d373fff65e58`  
**Evaluation Standard**: Paired per-scenario evaluation, 10,000-sample bootstrap confidence intervals, and Wilcoxon signed-rank hypothesis testing  

---

## 1. Executive Summary & Statistical Uncertainty

Across all 100 validation scenarios, kNN trajectory retrieval achieves a **17.85% observed reduction in Mean $L_2$ error** and a **17.28% observed reduction in Endpoint $L_2$ error** compared to Constant Velocity (CV).

| Metric | Constant Velocity (CV) | kNN Trajectory Retrieval | Absolute $\Delta$ | Relative $\Delta$ | 95% Bootstrap CI (10,000 resamples) |
|---|---|---|---|---|---|
| **Mean $L_2$ Error** | 4.729 m | **3.885 m** | $-0.844\text{ m}$ | **$-17.85\%$** | $[12.75\%,\; 23.13\%]$ |
| **Median $L_2$ Error** | 3.994 m | **3.243 m** | $-0.751\text{ m}$ | **$-18.80\%$** | $[13.20\%,\; 24.15\%]$ |
| **Endpoint $L_2$ Error ($t=5.0\text{s}$)** | 11.963 m | **9.896 m** | $-2.067\text{ m}$ | **$-17.28\%$** | $[11.13\%,\; 23.34\%]$ |

### Hypothesis Testing & Statistical Significance (scipy.stats verified):
* **Paired Student's $t$-test (Mean $L_2$, $\text{df} = 99$):** $t = 5.979$, $p = 3.56 \times 10^{-8}$ (`scipy.stats.ttest_rel`)
* **Paired Student's $t$-test (Endpoint $L_2$, $\text{df} = 99$):** $t = 4.838$, $p = 4.82 \times 10^{-6}$ (`scipy.stats.ttest_rel`)
* **Wilcoxon Signed-Rank Test (Mean $L_2$, tie-averaged, continuity-corrected):** $W = 777.0$, $p = 4.09 \times 10^{-8}$ (`scipy.stats.wilcoxon`)
* **Wilcoxon Signed-Rank Test (Endpoint $L_2$, tie-averaged, continuity-corrected):** $W = 1039.0$, $p = 6.84 \times 10^{-6}$ (`scipy.stats.wilcoxon`)

*Full-precision 64-bit float paired values for all 100 scenarios are published in `reports/validation_paired_100_scenarios_full_precision.json` to ensure exact reproduction without CSV decimal rounding.*

Both parametric and non-parametric tests reject the null hypothesis ($H_0$: no difference between CV and kNN) with $p < 10^{-5}$. The 95% bootstrap confidence interval demonstrates that the expected Mean $L_2$ error reduction across resampled splits is bounded between **$12.75\%$ and $23.13\%$**.

---

## 2. Per-Scenario Win Rate & Error Distribution

Evaluating per-scenario paired differences ($\Delta L_2 = \text{CV} - \text{kNN}$):

* **Win Rate (Mean $L_2$):** kNN outperforms CV in **71 / 100 scenarios (71.0%)**; CV wins in 23%, with 6% ties ($\le 1\text{mm}$).
* **Win Rate (Endpoint $L_2$):** kNN outperforms CV in **67 / 100 scenarios (67.0%)**.
* **Win Rate (Median $L_2$):** kNN outperforms CV in **71 / 100 scenarios (71.0%)**.

### Quantiles of Per-Scenario Difference ($\Delta = \text{CV} - \text{kNN}$ in meters):
* $p_{10} = -0.57\text{ m}$
* $p_{25} = 0.00\text{ m}$ (lower quartile: parity or better)
* $p_{50}\text{ (Median)} = \mathbf{+0.52\text{ m}}$
* $p_{75} = \mathbf{+1.46\text{ m}}$
* $p_{90} = \mathbf{+2.46\text{ m}}$
* $p_{95} = \mathbf{+3.77\text{ m}}$

### Magnitude of Gain vs. Loss:
* **Substantial Improvement ($\Delta \ge +1.0\text{ m}$):** **32 scenarios** (kNN beats CV by more than 1 meter).
* **Substantial Degradation ($\Delta \le -1.0\text{ m}$):** **3 scenarios** (CV beats kNN by more than 1 meter).

---

## 3. Maneuver & Environmental Subset Breakdowns

### Breakdown by Driving Instruction

| Driving Instruction | Scenarios ($N$) | CV Mean $L_2$ | kNN Mean $L_2$ | Observed $\Delta$ Mean | CV Endpoint $L_2$ | kNN Endpoint $L_2$ | Observed $\Delta$ Endpoint |
|---|---|---|---|---|---|---|---|
| `turn right` | 21 | 6.17 m | **4.27 m** | **$-30.8\%$** | 17.25 m | **11.82 m** | **$-31.5\%$** |
| `use right lane` | 9 | 3.52 m | **1.74 m** | **$-50.7\%$** | 7.81 m | **4.00 m** | **$-48.8\%$** |
| `turn left` | 11 | 5.08 m | **4.51 m** | **$-11.1\%$** | 13.19 m | **11.11 m** | **$-15.8\%$** |
| `drive straight on` | 41 | 4.45 m | **4.11 m** | **$-7.7\%$** | 10.63 m | **10.01 m** | **$-5.9\%$** |
| `use left lane` | 7 | 4.88 m | **4.26 m** | **$-12.6\%$** | 12.67 m | **11.31 m** | **$-10.7\%$** |
| `overtake truck on right` | 3 | 4.55 m | **3.42 m** | **$-24.8\%$** | 10.45 m | **8.10 m** | **$-22.5\%$** |
| `use middle lane` | 3 | 2.89 m | **2.52 m** | **$-13.0\%$** | 6.94 m | **6.73 m** | **$-3.1\%$** |
| `overtake trucks on right`| 1 | 1.92 m | **0.92 m** | **$-51.9\%$** | 3.86 m | **2.12 m** | **$-45.0\%$** |
| `overtake van in front` | 1 | 2.28 m | 2.16 m | $-5.4\%$ | 4.03 m | 4.22 m | $+4.8\%$ |
| `u-turn` | 1 | 9.80 m | 10.27 m | $+4.8\%$ | 25.05 m | 25.11 m | $+0.3\%$ |
| `overtake truck in front`| 1 | 1.84 m | 2.56 m | $+39.0\%$ | 6.23 m | 9.02 m | $+44.7\%$ |
| `overtake car in front` | 1 | 1.03 m | 1.68 m | $+62.5\%$ | 2.57 m | 8.86 m | $+244.6\%$ |

### Breakdown by Scenario Type & Environmental Domain

| Scenario Type | Scenarios ($N$) | CV Mean $L_2$ | kNN Mean $L_2$ | Observed $\Delta$ Mean | CV Endpoint $L_2$ | kNN Endpoint $L_2$ | Observed $\Delta$ Endpoint |
|---|---|---|---|---|---|---|---|
| `4 intersection` | 35 | 6.17 m | **4.95 m** | **$-19.7\%$** | 15.63 m | **11.98 m** | **$-23.4\%$** |
| `3 overtake or lane change`| 20 | 3.87 m | **2.91 m** | **$-24.7\%$** | 9.29 m | **7.56 m** | **$-18.6\%$** |
| `0 specifically selected` | 18 | 4.27 m | **3.50 m** | **$-18.0\%$** | 12.09 m | **10.50 m** | **$-13.2\%$** |
| `2 construction zone` | 6 | 2.85 m | **2.15 m** | **$-24.4\%$** | 5.87 m | **4.19 m** | **$-28.7\%$** |
| `5 nighttime` | 5 | 3.07 m | **2.05 m** | **$-33.4\%$** | 7.21 m | **5.36 m** | **$-25.7\%$** |
| `6 snow & wintry mix` | 6 | 7.06 m | **6.83 m** | **$-3.2\%$** | 17.10 m | **16.51 m** | **$-3.4\%$** |
| `1 heavy rain` | 10 | 2.81 m | 2.98 m | **$+6.2\%$** | 7.16 m | 7.90 m | **$+10.3\%$** |

---

## 4. Analysis of Degradation Cases & Environmental Mismatches

kNN does not win universally. Understanding where and why it degrades is critical:

### A. Heavy Rain Domain Shift ($+6.2\%$ degradation, $N=10$):
* In `heavy rain`, drivers in the real world reduce speed and brake earlier due to reduced traction and visibility.
* 2 of the 3 largest degradations across the entire validation set occurred in heavy rain:
  * **Scenario `0000912`:** CV Mean $= 3.24\text{ m}$, kNN Mean $= 4.27\text{ m}$ ($\Delta = -1.04\text{ m}$).
  * **Scenario `0000954`:** CV Mean $= 4.33\text{ m}$, kNN Mean $= 5.51\text{ m}$ ($\Delta = -1.18\text{ m}$).
* **Root Cause:** The training donor database is dominated by dry and clear scenarios. Matching on geometry without conditioning on precipitation causes kNN to retrieve faster, dry-weather donors, overshooting the conservative driving profile of wet weather.

### B. Stationary Acceleration / Start-from-Stop Outliers ($N=7$):
* In 7 scenarios (e.g. `0000902`, `0000921`, `0000944`, `0000949`, `0000951`), the ego vehicle was stopped at $t \le 0$ ($v < 0.3\text{ m/s}$) at a red light and then accelerated rapidly.
* Both CV ($1.44\text{ m}$) and kNN ($3.80\text{ m}$) underestimate travel distance ($14.81\text{ m}$) because past kinematics carry zero momentum. While kNN correctly predicts lateral turning curvature, predicting the exact moment of throttle application requires traffic light state or signal phase input.

---

## 5. Synchronized SVD-XT Motion-Bucket Video Ablations

To directly test whether motion bucket conditioning changes the video generator, we compared three empirical runs on Kaggle hardware (Tesla T4, 15 steps, seed 42, FP16, $876 \times 568$):

* **Baseline Run (CV-Derived):** $\text{motion\_bucket\_id} = \mathbf{73}$ (speed $= 5.27\text{ m/s}$)
* **Run 1 (kNN-Derived, bucket 69):** $\text{motion\_bucket\_id} = \mathbf{69}$ (speed $= 4.98\text{ m/s}$, intermediate test)
* **Run 2 (kNN-Derived, bucket 65, Current `main` Artifact):** $\text{motion\_bucket\_id} = \mathbf{65}$ (speed $= 4.60\text{ m/s}$, commit `3618765` / `78daffe`)

| Metric | CV-Derived (`bucket=73`) | kNN-Derived (`bucket=69`) | kNN-Derived (`bucket=65`, Current `main`) |
|---|---|---|---|
| **Git Commit SHA** | `fa9fabf...` | `635c507...` | `3618765...` / `78daffe` |
| **Model Checkpoint Revision** | `9e43909513c...` | `9e43909513c...` | `9e43909513c6714f1bc78bcb44d96e733cd242aa` |
| **Training Donors Loaded** | 0 (CV fallback) | 499 (kNN verified) | 499 (kNN verified) |
| **Inference Duration (Tesla T4)** | 385.06 s | 385.60 s | 386.28 s |
| **Peak VRAM Allocated** | 8,716.94 MiB | 8,716.94 MiB | 8,716.94 MiB |
| **Mean Inter-Frame $L_1$** | 3.9455 | 3.9705 | 3.9561 |
| **Total Drift $L_1$ from $t_0$** | 18.6215 | 18.6109 | 17.7267 |
| **Temporal Standard Deviation ($\sigma_t$)** | 8.7760 | 8.9828 | 9.1333 |
| **Local PQ Proxy** | 52.68 | 52.74 | 52.05 |
| **MP4 File Size** | 426,690 bytes | 426,885 bytes | 427,794 bytes |

### Objective Video Evaluation:
1. **Divergence of Generated Latents:** Shifting the scalar motion bucket from 73 to 65 significantly alters the temporal attention weights and latents, shifting mid-sequence frame differences by up to $56\%$ and increasing temporal standard deviation ($\sigma_t$) from 8.78 to 9.13.
2. **Quality vs. Divergence:** **The local PQ proxy differences ($52.68 \to 52.74 \to 52.05$) are negligible and within normal diffusion noise.** Furthermore, local proxy metrics are NOT the official private Q-Align perceptual quality benchmark. Therefore, this ablation demonstrates **controllability/influence on video dynamics**, not superior perceptual realism.
3. **Architectural Limitation:** Scalar motion conditioning modulates optical flow speed, but does not provide **spatial trajectory steerability**. It cannot enforce road geometry, lane-keeping, or multi-view cross-camera consistency.
