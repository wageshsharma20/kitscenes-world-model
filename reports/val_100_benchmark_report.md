# Full Validation Split Benchmark: Constant Velocity vs. kNN Trajectory Retrieval

**Dataset Split**: KITScenes-LongTail Official Validation Split ($N=100$ scenarios: `0000900`–`0000999`)  
**Training Donors**: 499 parsed and validated scenarios from the official training split (`0000400`–`0000899`)  
**Benchmark Date**: November 15, 2026  

---

## 1. Executive Summary & Aggregate Metrics

Across all 100 validation scenarios, the kNN Trajectory Retrieval Planner systematically outperforms the Constant Velocity (CV) baseline across every spatial error metric:

| Metric | Constant Velocity (CV) | kNN Trajectory Retrieval | Absolute $\Delta$ | Relative Improvement |
|---|---|---|---|---|
| **Mean $L_2$ Error** | 4.729 m | **3.885 m** | $-0.844\text{ m}$ | **$-17.85\%$** |
| **Median $L_2$ Error** | 3.994 m | **3.243 m** | $-0.751\text{ m}$ | **$-18.80\%$** |
| **Endpoint $L_2$ Error ($t=5.0\text{s}$)** | 11.963 m | **9.896 m** | $-2.067\text{ m}$ | **$-17.28\%$** |

---

## 2. Granular Breakdown by Driving Maneuver

Categorizing the 100 validation scenarios by their natural language driving instructions demonstrates where kNN provides massive advantages and where physical limitations arise:

| Driving Instruction | Scenarios ($N$) | CV Mean $L_2$ | kNN Mean $L_2$ | $\Delta$ Mean | CV Endpoint $L_2$ | kNN Endpoint $L_2$ | $\Delta$ Endpoint |
|---|---|---|---|---|---|---|---|
| `turn right` | 21 | 6.17 m | **4.27 m** | **$-30.8\%$** | 17.25 m | **11.82 m** | **$-31.5\%$** |
| `use right lane` | 9 | 3.52 m | **1.74 m** | **$-50.7\%$** | 7.81 m | **4.00 m** | **$-48.8\%$** |
| `turn left` | 11 | 5.08 m | **4.51 m** | **$-11.1\%$** | 13.19 m | **11.11 m** | **$-15.8\%$** |
| `drive straight on` | 41 | 4.45 m | **4.11 m** | **$-7.7\%$** | 10.63 m | **10.01 m** | **$-5.9\%$** |
| `use left lane` | 7 | 4.88 m | **4.26 m** | **$-12.6\%$** | 12.67 m | **11.31 m** | **$-10.7\%$** |
| `overtake truck on right` | 3 | 4.55 m | **3.42 m** | **$-24.8\%$** | 10.45 m | **8.10 m** | **$-22.5\%$** |
| `use middle lane` | 3 | 2.89 m | **2.52 m** | **$-13.0\%$** | 6.94 m | **6.73 m** | **$-3.1\%$** |
| `overtake trucks on right`| 1 | 1.92 m | **0.92 m** | **$-51.9\%$** | 3.86 m | **2.12 m** | **$-45.0\%$** |
| `overtake van in front` | 1 | 2.28 m | **2.16 m** | **$-5.4\%$** | 4.03 m | 4.22 m | $+4.8\%$ |
| `u-turn` | 1 | 9.80 m | 10.27 m | $+4.8\%$ | 25.05 m | 25.11 m | $+0.3\%$ |
| `overtake truck in front`| 1 | 1.84 m | 2.56 m | $+39.0\%$ | 6.23 m | 9.02 m | $+44.7\%$ |
| `overtake car in front` | 1 | 1.03 m | 1.68 m | $+62.5\%$ | 2.57 m | 8.86 m | $+244.6\%$ |

### Key Takeaways on Maneuvers:
1. **Curved and Lane-Change Maneuvers**: On `turn right` ($N=21$) and `use right lane` ($N=9$), kNN cuts error by **30.8% to 50.7%**. Linear extrapolation (CV) fails completely when yaw angles exceed $30^\circ$, whereas retrieved donor trajectories guide the ego-vehicle along realistic road curvature.
2. **Straight Driving Preservation**: On `drive straight on` ($N=41$), kNN still improves by **7.7%** over CV because it models subtle longitudinal deceleration/braking patterns observed in historical training donors.
3. **Outlier Overtake Failures ($N=2$)**: In 2 single-instance overtaking scenarios, CV performed better because the ego vehicle in the ground truth delayed pulling out into the passing lane, while kNN immediately initiated an aggressive lateral lane change.

---

## 3. Breakdown by Scenario Environmental Type

| Scenario Type | Scenarios ($N$) | CV Mean $L_2$ | kNN Mean $L_2$ | $\Delta$ Mean | CV Endpoint $L_2$ | kNN Endpoint $L_2$ | $\Delta$ Endpoint |
|---|---|---|---|---|---|---|---|
| `4 intersection` | 35 | 6.17 m | **4.95 m** | **$-19.7\%$** | 15.63 m | **11.98 m** | **$-23.4\%$** |
| `3 overtake / lane change`| 20 | 3.87 m | **2.91 m** | **$-24.7\%$** | 9.29 m | **7.56 m** | **$-18.6\%$** |
| `0 specifically selected` | 18 | 4.27 m | **3.50 m** | **$-18.0\%$** | 12.09 m | **10.50 m** | **$-13.2\%$** |
| `2 construction zone` | 6 | 2.85 m | **2.15 m** | **$-24.4\%$** | 5.87 m | **4.19 m** | **$-28.7\%$** |
| `5 nighttime` | 5 | 3.07 m | **2.05 m** | **$-33.4\%$** | 7.21 m | **5.36 m** | **$-25.7\%$** |
| `6 snow & wintry mix` | 6 | 7.06 m | **6.83 m** | **$-3.2\%$** | 17.10 m | **16.51 m** | **$-3.4\%$** |
| `1 heavy rain` | 10 | 2.81 m | 2.98 m | $+6.2\%$ | 7.16 m | 7.90 m | $+10.3\%$ |

---

## 4. Distance Profile & The "Start-from-Stop" Underestimation Dilemma

### Macro-Level Distribution:
Across all 100 scenarios, kNN does **not** systematically underestimate vehicle travel:
* **Ground Truth Mean Distance**: $60.08\text{ m}$ vs. kNN: **$59.35\text{ m}$** ($-1.2\%$)
* **Ground Truth Median Distance**: $47.20\text{ m}$ vs. kNN: **$47.23\text{ m}$** ($\mathbf{+0.06\%}$)
* **Constant Velocity Median Distance**: $50.39\text{ m}$ ($+6.8\%$ overshoot)

### The 7 Outlier Underestimations:
In exactly 7 out of 100 scenarios (including Scenario `0000902`, `0000921`, `0000944`, `0000949`, and `0000951`), predicted distance is $< 50\%$ of ground truth:
* **Root Cause**: The ego vehicle is stationary or creeping at $t \le 0$ ($v < 0.3\text{ m/s}$) waiting at a traffic light or stop sign.
* In the ground truth, the light turns green at $t=0.5\text{s}$, and the vehicle accelerates briskly across the intersection ($v \to 8\text{ m/s}$), traversing $14.82\text{ m}$.
* Constant velocity assumes the vehicle will remain at $0.28\text{ m/s}$ for the next 5 seconds ($1.44\text{ m}$).
* kNN retrieves donors matching the stationary entry state ($v \le 0.3\text{ m/s}$), producing $3.80\text{ m}$.
* Without visual traffic light state tokens or high-level scene intent, purely kinematic planners cannot predict the exact moment a stationary vehicle accelerates from zero. However, kNN correctly predicted the **right turn geometry** ($y = -1.99\text{ m}$ vs. CV's blind $y = 0.00\text{ m}$).

---

## 5. Visual Generation Ablation: CV vs. kNN Motion Bucket Conditioning

To evaluate whether the kNN planner actually alters the generative video output (rather than merely producing a separate trajectory file), an identical SVD-XT inference pipeline was executed under both conditioning regimes:

* **Fixed Baseline**: Scenario `0000000`, camera `front`, 15 denoising steps on Nvidia Tesla T4 GPU, 25 frames @ 5 fps, resolution $876 \times 568$, fixed seed `42`, FP16 precision.
* **Ablated Variable**: Motion bucket ID derived from planned trajectory kinematics:
  * **Run A (CV-Derived)**: Mean speed $5.27\text{ m/s} \implies \text{motion\_bucket\_id} = \mathbf{73}$
  * **Run B (kNN-Derived)**: Mean speed $4.98\text{ m/s} \implies \text{motion\_bucket\_id} = \mathbf{69}$

| Metric | CV-Derived (`bucket=73`) | kNN-Derived (`bucket=69`) | Absolute $\Delta$ | Relative $\Delta$ |
|---|---|---|---|---|
| **Mean Consecutive Step Diff $L_1$** | 3.9455 | 3.9705 | $+0.0250$ | $+0.63\%$ |
| **Total Drift $L_1$ from $t_0$** | 18.6215 | 18.6109 | $-0.0106$ | $-0.06\%$ |
| **Temporal Standard Deviation ($\sigma_t$)** | 8.7760 | 8.9828 | $+0.2068$ | $+2.36\%$ |
| **Local PQ Proxy** | 52.68 | 52.74 | $+0.06$ | $+0.11\%$ |
| **Step 1 Inter-Frame $L_1$** | 4.3341 | 5.0179 | $+0.6838$ | $+15.78\%$ |
| **Step 2 Inter-Frame $L_1$** | 6.3298 | 5.7741 | $-0.5557$ | $-8.78\%$ |
| **Step 9 Inter-Frame $L_1$** | 4.0965 | 4.4294 | $+0.3329$ | $+8.13\%$ |
| **Step 12 Inter-Frame $L_1$** | 3.8116 | 4.2027 | $+0.3911$ | $+10.26\%$ |
| **Output File Size** | 426,690 bytes | 426,885 bytes | $+195$ bytes | $+0.05\%$ |

### Ablation Findings:
1. **Definite Pixel & Latent Divergence**: Shifting the motion bucket by just 4 units alters latent denoising trajectories across every frame, producing up to **15.78%** difference in inter-frame pixel variation ($L_1$).
2. **Coupling Reality**: While the motion bucket alters video dynamics, it is a **scalar global speed prior**, not a directional trajectory constraint. SVD-XT does not turn left or right purely based on motion bucket ID; spatial steerability remains an open research frontier requiring spatial cross-attention or ControlNet conditioning.
