# Phase 5.4: Controlled Component Ablation & Metric Audit Report

- **Date/Time (UTC):** 2026-11-19T02:36:04.433111+00:00
- **Evaluation Status:** `✅ ABLATION_COMPLETE_AND_STANDARDIZED`
- **Device:** `cuda:0` (Mock Mode: `False`)
- **Denoising Steps:** 1 steps (EulerDiscreteScheduler)
- **Base Seed:** 42 (Deterministic camera seed offsets applied)
- **Evaluated Checkpoint:** `cssc_pilot_step30.pt` (SHA-256: `9e65c397e4059999...`, internal training step: 6 — early checkpoint from git history)
- **Baseline Submission Archive:** `ARCHIVE_MISSING_ON_DISK` (Intact: `False`)

---

## 1. Executive Summary & Experimental Controls

This benchmark evaluates three reference systems and isolates the impact of each conditioning component on identical observed initial images and camera-paired deterministic noise:

### Three Reference Systems:
- **System A (`vanilla_svd`)**: Pretrained frozen SVD-XT (single-camera CLIP token, default motion bucket 127).
- **System B (`project_baseline`)**: Existing best project baseline (kNN trajectory planner + trajectory-derived SVD motion bucket).
- **System C (`cssc_plus_spatial_adapter`)**: Full CSSC 27-token bank + SpatialAdapter from checkpoint.

### Component Ablation:
- **`cssc_only`**: Isolates the 27-token multi-view conditioning bank (SpatialAdapter disabled).
- **`spatial_adapter_only`**: Isolates the spatial projection adapter (vanilla single-camera CLIP token).

---

## 2. Quantitative Performance Comparison (`scenario 0000900`)

### Four-Way Component Ablation Summary
| Configuration | Mean PSNR (dB) | Mean SSIM | Divergence ($L_1$) | Mean Flow | Boundary Strip Corr | Local EMC Approx | Visual Naturalness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `vanilla_svd` | **14.94** | **0.6861** | 0.00 | 0.0005 | 0.8593 | 0.750 | 2.94 |
| `project_baseline` | **15.63** | **0.7103** | 7.31 | 0.0016 | 0.8232 | 0.500 | 3.04 |
| `cssc_only` | **15.15** | **0.7135** | 10.29 | 0.0013 | 0.8740 | 0.750 | 1.98 |
| `spatial_adapter_only` | **11.32** | **0.5172** | 49.72 | 0.0023 | 0.9579 | 1.000 | 2.47 |
| `cssc_plus_spatial_adapter` | **11.39** | **0.5158** | 51.60 | 0.0018 | 0.9635 | 1.000 | 2.12 |

---

## 3. Deep Diagnostic Telemetry & Residual Norm Analysis

### CSSC 27-Token Bank Norms
- **Mean Token Norm:** `31.9959`
- **Std Token Norm:** `0.0011`
- **Camera Token Norm ($i=0$):** `31.9998`
- **Scene/Motion Token Norm ($i=1$):** `31.9993`
- **Waypoint Token Norms Mean ($i=2..26$):** `31.9956`

### SpatialAdapter Residual Magnitude
- **Mean Residual / Latent Ratio:** `0.1009`
- **Max Residual / Latent Ratio:** `0.1009`
- **NaN / Inf Detections:** `False`

---

## 4. Metric Audit Compliance

1. **Visual Naturalness Proxy:** Computed from spatial-temporal gradient texture and flat-fill penalty. **Explicitly disclaimed as non-official PQ.**
2. **Local EMC Approximation:** Computed using independent front and rear optical flow with strict sign agreement check.
3. **Boundary Strip Correlation:** Computed across all 6 adjacent camera frustum boundaries (15% image-edge strips).
4. **Trajectory Adherence:** Returns `null` when optical flow is unmeasurable; never defaults to `0.5`.
5. **MMS:** Evaluates submitted trajectory waypoints (`3.50`), not video frames.

---

## 5. Architectural Findings, Qualifications & Candidate Strategy

### 5.1 Root-Cause Forensic Diagnosis
The SpatialAdapter injected a residual perturbation with RMS magnitude approximately **10.09%** of the latent representation ($\|\Delta z_{\text{adapter}}\|_{\text{RMS}} / \|z_t\|_{\text{RMS}} \approx 0.1009$).
This high-frequency additive perturbation severely disrupted the pretrained SVD-XT diffusion trajectory, causing an immediate collapse of ~3.8 dB PSNR and ~0.17–0.20 SSIM across all 6 camera views.

### 5.2 Scientific Qualifications
- **Experiment Scope:** The SpatialAdapter degradation is **strongly isolated in this experiment (on validation scenario `0000900`)**. Because this was measured on a single scenario, further multi-scenario evaluation is required before making a global dataset-wide claim.
- **Checkpoint Provenance:** As verified in `reports/phase5_4_checkpoint_provenance_audit.md`, the evaluated checkpoint is the git-tracked Step-6 artifact (`9e65c397...`). All findings strictly characterize this checkpoint; no claims of 30-step convergence are made.

### 5.3 Candidate Strategy for Submission
1. **Freeze SpatialAdapter:** The current SpatialAdapter implementation is removed from the submission pipeline and frozen for future research.
2. **Two Serious Submission Candidates Identified:**
   - **Candidate 1: Project Baseline (`project_baseline`)** — Trajectory planner + trajectory-derived motion bucket. Strongest PSNR (`15.63 dB`).
   - **Candidate 2: CSSC-Only (`cssc_only`)** — 27-token MultiViewConditioner bank with SpatialAdapter disabled. Highest SSIM (`0.7135`) and +0.21 dB PSNR over Vanilla SVD.
3. **Phase 5.5 Execution:** A dedicated multi-scenario head-to-head evaluation across 8+ diverse official validation scenarios will decide between Candidate 1 and Candidate 2 before final test split video generation.

