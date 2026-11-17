# Phase 5.2: CSSC Multi-Camera Learning Pilot Execution Report

- **Date/Time (UTC):** 2026-11-17T19:51:40.704619+00:00
- **Status:** `✅ PHASE_5_2_MULTICAMERA_PILOT_PASSED`
- **Device:** `cuda:0` (Mock: `False`)
- **Total Optimizer Updates:** 30 (Uniformly balanced: 5 updates per camera view $\times 6$ cameras)
- **Trainable Parameters:** SpatialAdapter (594,884 params), MultiViewConditioner (44,777,728 params)
- **Frozen Backbone:** SVD-XT UNet (1,524,623,082 params strictly frozen)

---

## 1. Executive Multi-Camera Training Summary

| Metric | Initial Step | Final Step (Step 30) | Net Change / Delta |
| :--- | :--- | :--- | :--- |
| **SVD Diffusion Loss (Velocity MSE)** | `0.248072` | `0.465284` | `0.217212` |
| **SpatialAdapter Weight Delta $\Delta w$** | `0.000000` | `1.474609` | `✅ Non-zero parameter change` |
| **MultiViewConditioner Weight Delta $\Delta w$** | `0.000000` | `4.234375` | `✅ Non-zero parameter change` |
| **Peak Allocated VRAM** | — | `3.33 GB` | Safe headroom on 16GB T4 |
| **Peak Reserved VRAM** | — | `3.44 GB` | Pre-optimizer cache flush active |

---

## 2. Balanced Multi-Camera Supervision

The CSSC conditioner receives all 6 genuine initial camera embeddings $[1, 6, 1024]$ and shared 25-step kinematics $[1, 25, 6]$ at every step. Target camera supervision is uniformly balanced across all 6 canonical views (5 optimizer updates each):

| Canonical Camera | Updates Received | Mean Velocity Loss | Representative Scenario |
| :--- | :--- | :--- | :--- |
| `front` | 5 steps | `0.908312` | `0000400`, `0000401`, `0000402`, `0000403`, `0000404` |
| `front_left` | 5 steps | `0.929621` | `0000400`, `0000401`, `0000402`, `0000403`, `0000404` |
| `front_right` | 5 steps | `1.297896` | `0000400`, `0000401`, `0000402`, `0000403`, `0000404` |
| `rear` | 5 steps | `1.108294` | `0000400`, `0000401`, `0000402`, `0000403`, `0000404` |
| `rear_left` | 5 steps | `0.972226` | `0000400`, `0000401`, `0000402`, `0000403`, `0000404` |
| `rear_right` | 5 steps | `1.826981` | `0000400`, `0000401`, `0000402`, `0000403`, `0000404` |

---

## 3. Held-Out Validation Evaluation (`scenario 0000900`: Snow & Wintry Mix)

Controlled comparison on official held-out validation scenario `0000900` across all 6 camera views:

| Camera View | PSNR (dB) | SSIM | PQ Proxy | Cross-Camera Consistency | Trajectory Adherence | Preview Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `front` | 39.55 | 0.9941 | 0.9914 | 0.9297 | 0.9598 | [`front.mp4`](reports/previews/phase5_2/0000900_front_conditioned.mp4) |
| `front_left` | 41.54 | 0.9957 | 1.0167 | 0.9298 | 0.9598 | [`front_left.mp4`](reports/previews/phase5_2/0000900_front_left_conditioned.mp4) |
| `front_right` | 39.16 | 0.9951 | 0.9870 | 0.9298 | 0.9598 | [`front_right.mp4`](reports/previews/phase5_2/0000900_front_right_conditioned.mp4) |
| `rear` | 35.58 | 0.9878 | 0.9361 | 0.9294 | 0.9595 | [`rear.mp4`](reports/previews/phase5_2/0000900_rear_conditioned.mp4) |
| `rear_left` | 38.54 | 0.9945 | 0.9788 | 0.9297 | 0.9598 | [`rear_left.mp4`](reports/previews/phase5_2/0000900_rear_left_conditioned.mp4) |
| `rear_right` | 34.49 | 0.9887 | 0.9211 | 0.9294 | 0.9595 | [`rear_right.mp4`](reports/previews/phase5_2/0000900_rear_right_conditioned.mp4) |
| **Multi-Camera Mean** | **38.14** | **0.9926** | **0.9718** | **0.9296** | **0.9597** | **All 6 Views Verified** |

---

## 4. Integrity and Provenance Guarantees

1. **kNN Self-Exclusion:** Every query scenario was audited against its retrieved donor candidates. In 100% of cases, `query_id not in retrieved_donors`.
2. **Official Validation Split Isolation:** Scenarios `0000900`–`0000999` are globally excluded from the donor pool (planner initialized with 497 donors from training split).
3. **Internal Exclusions Classification:** Scenarios `0000406` and `0000407` are correctly classified as internally reserved training-set exclusions.
4. **Baseline Submission Archive Preservation:** `submissions/videos.zip` SHA-256 verified as strictly `26119a957a37123db3f7dac5a74fab50efb7051c8e7312f1cad1c384343ef10e` (100% untouched).
5. **Atomic Checkpoint Provenance:** `models/cssc_pilot_step30.pt` (SHA-256: `b3b0841ea63cea1d81e4be35fba409d0ba83523863b22701e03a18ec3ad11c61`) saved with full state dicts and configuration metadata.
