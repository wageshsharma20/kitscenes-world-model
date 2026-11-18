# Phase 5.4: Challenge Metric Audit & Diagnostic Correction Report

- **Date/Time (UTC):** 2026-11-18T08:50:00Z
- **Status:** `METRIC_AUDIT_COMPLETE_AND_STANDARDIZED`
- **Purpose:** Rigorous review and algorithmic correction of all evaluation metrics used in the KITScenes-LongTail World Model project, addressing five critical misinterpretations identified in Phase 5.3.

---

## 1. Perceptual Quality (PQ) Audit

### Original Defect
In Phase 5.2 and 5.3 reports, scores labeled "PQ Proxy" or "PQ" (e.g. `2.11` for Baseline, `0.31` for Conditioned) were presented alongside official metrics, risking the interpretation that they measured official competition PQ.

### Ground Truth Challenge Specification
- Official competition PQ is computed using **Q-Align (OneAlign)**.
- Each camera video is rated on 5 discrete perceptual levels: `bad` (1), `poor` (2), `fair` (3), `good` (4), `excellent` (5).
- Mapped to $[0, 100]$ as: $100 \times \frac{\text{rating} - 1}{4}$.
- Normalized relative to real long-tail calibration driving videos.
- Per-scenario weighted score:
  $$\text{PQ}_{\text{scenario}} = 0.50 \cdot \text{PQ}_{\text{front}} + 0.10 \sum_{c \in \{\text{fl}, \text{fr}, \text{r}, \text{rl}, \text{rr}\}} \text{PQ}_c$$

### Metric Correction
1. The project's heuristic (`kitscenes/pq_proxy.py`) computes a spatial-temporal texture and contrast score from Laplacian variance, RGB standard deviation, and flat-fill penalty.
2. It has been renamed to **`visual_naturalness_proxy`**.
3. It carries an explicit boolean flag: `"is_official_pq": false`.
4. It is formally disclaimed: **Never to be reported as official PQ or used to claim competition ranking improvements.**

---

## 2. Ego-Motion Coherence (EMC) Audit

### Original Defect
The previous evaluator computed optical-flow yaw from the front and rear cameras and blindly averaged them without checking whether they agreed in sign or magnitude.

### Ground Truth Challenge Specification
- Optical flow recovers ego-motion from the 6 surround cameras.
- **Yaw must come from `front` and `rear`, and they MUST agree.**
- Forward speed is recovered from the four side cameras (`front_left`, `front_right`, `rear_left`, `rear_right`).
- Trajectories are divided into two temporal intervals:
  - Segment 1: $0.0\,\text{s}$ to $3.0\,\text{s}$ (frames $0 \dots 14$)
  - Segment 2: $3.0\,\text{s}$ to $5.0\,\text{s}$ (frames $15 \dots 24$)
- Each segment receives a steering class (`left`, `straight`, `right`) and acceleration class (`accelerate`, `maintain`, `brake`).
- A class matches only if the segment speeds also satisfy the agreement ratio:
  $$\frac{\min(v_{\text{rec}}, v_{\text{plan}}) + 0.5}{\max(v_{\text{rec}}, v_{\text{plan}}) + 0.5} \ge 0.5$$

### Metric Correction
1. Front-camera yaw ($\text{yaw}_{\text{front}}$) and rear-camera yaw ($\text{yaw}_{\text{rear}}$) are now computed **independently**.
2. **Strict Agreement Check**: If $\text{steer}(\text{yaw}_{\text{front}}) \neq \text{steer}(\text{yaw}_{\text{rear}})$, the recovered steering is marked as `disagreement` and the steering class match evaluates to `False`.
3. The evaluator records per-segment recovered speed, planned speed, recovered acceleration, planned acceleration, and individual front/rear yaw estimates.
4. The metric is renamed to **`local_emc_approximation`** with `"is_official_emc": false` until validated against the organizer's exact evaluation code.

---

## 3. Cross-Camera Consistency Audit

### Original Defect
`compute_cross_camera_overlap_consistency()` compared 15%-wide boundary pixel strips using normalized cross-correlation across 4 camera pairs, despite docstrings listing 6 pairs, and was presented as "geometric overlap consistency".

### Metric Correction
1. Renamed to **`boundary_strip_correlation`** with `"is_calibrated_3d_geometry": false`.
2. Expanded to evaluate **all 6 adjacent camera relationships**:
   - `("front", "front_left")`
   - `("front", "front_right")`
   - `("rear", "rear_left")`
   - `("rear", "rear_right")`
   - `("front_left", "rear_left")`
   - `("front_right", "rear_right")`
3. Clearly documented: This is a 2D boundary pixel correlation heuristic across image borders, NOT a 3D calibrated scene reprojection check.

---

## 4. Trajectory Adherence Audit

### Original Defect
When Farneback optical flow returned near-zero horizontal and vertical displacement ($|u| \le 0.01$, $|v| \le 0.01$), the function returned a default fallback score of `0.5`. This caused all 6 conditioned views in Phase 5.3 to report exactly `0.5`, masquerading an absence of measurable motion as neutral adherence.

### Metric Correction
1. Fallback `0.5` removed entirely.
2. When flow is below threshold or unavailable, the function returns:
   - `"adherence": null`
   - `"measurable": false`
   - `"valid_flow_count": 0`
3. Planned heading and recovered flow heading are reported separately in radians.

---

## 5. Maneuver Match Score (MMS) Audit

### Original Defect
MMS was reported in video evaluation tables, creating the false impression that it reflected generated video improvements.

### Metric Correction
1. MMS is evaluated strictly on the **kNN planned trajectory waypoints** against the expert-like demonstration.
2. Because both the vanilla baseline and conditioned model used the exact same kNN trajectory for scenario `0000900`, both received an identical MMS of `2.50`.
3. Marked with `"mms_evaluates_video": false`. It is categorized as a trajectory-planning metric, not a video-synthesis quality metric.
