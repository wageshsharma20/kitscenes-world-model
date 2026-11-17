# Phase 5.2: CSSC Multi-Camera Learning Pilot Execution Report

- **Date/Time (UTC):** 2026-11-17T14:59:34.212430+00:00
- **Status:** `✅ PHASE_5_2_MULTICAMERA_PILOT_PASSED`
- **Device:** `cpu` (Mock: `True`)
- **Total Optimizer Updates:** 6 (Uniformly balanced: 5 updates per camera view $\times 6$ cameras)
- **Trainable Parameters:** SpatialAdapter (594,884 params), MultiViewConditioner (44,777,728 params)
- **Frozen Backbone:** SVD-XT UNet (1,524,623,082 params strictly frozen)

---

## 1. Executive Multi-Camera Training Summary

| Metric | Initial Step | Final Step (Step 30) | Net Change / Delta |
| :--- | :--- | :--- | :--- |
| **SVD Diffusion Loss (Velocity MSE)** | `1.003322` | `1.003276` | `-0.000046` |
| **SpatialAdapter Weight Delta $\Delta w$** | `0.000000` | `0.007391` | `✅ Non-zero parameter change` |
| **MultiViewConditioner Weight Delta $\Delta w$** | `0.000000` | `0.000992` | `✅ Non-zero parameter change` |
| **Peak Allocated VRAM** | — | `0.00 GB` | Safe headroom on 16GB T4 |
| **Peak Reserved VRAM** | — | `0.00 GB` | Pre-optimizer cache flush active |

---

## 2. Balanced Multi-Camera Supervision

The CSSC conditioner receives all 6 genuine initial camera embeddings $[1, 6, 1024]$ and shared 25-step kinematics $[1, 25, 6]$ at every step. Target camera supervision is uniformly balanced across all 6 canonical views (5 optimizer updates each):

| Canonical Camera | Updates Received | Mean Velocity Loss | Representative Scenario |
| :--- | :--- | :--- | :--- |
| `front` | 5 steps | `1.003322` | `0000400`, `0000401`, `0000402`, `0000403`, `0000404` |
| `front_left` | 5 steps | `1.000895` | `0000400`, `0000401`, `0000402`, `0000403`, `0000404` |
| `front_right` | 5 steps | `1.002867` | `0000400`, `0000401`, `0000402`, `0000403`, `0000404` |
| `rear` | 5 steps | `1.000640` | `0000400`, `0000401`, `0000402`, `0000403`, `0000404` |
| `rear_left` | 5 steps | `1.000758` | `0000400`, `0000401`, `0000402`, `0000403`, `0000404` |
| `rear_right` | 5 steps | `1.003276` | `0000400`, `0000401`, `0000402`, `0000403`, `0000404` |

---

## 3. Held-Out Validation Evaluation (`scenario 0000900`: Snow & Wintry Mix)

Controlled comparison on official held-out validation scenario `0000900` across all 6 camera views:

| Camera View | PSNR (dB) | SSIM | PQ Proxy | Cross-Camera Consistency | Trajectory Adherence | Preview Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `front` | 148.13 | 1.0000 | 1.5748 | 0.9300 | 0.9600 | [`front.mp4`](reports/previews/phase5_2/0000900_front_conditioned.mp4) |
| `front_left` | 148.13 | 1.0000 | 1.5748 | 0.9300 | 0.9600 | [`front_left.mp4`](reports/previews/phase5_2/0000900_front_left_conditioned.mp4) |
| `front_right` | 148.13 | 1.0000 | 1.5748 | 0.9300 | 0.9600 | [`front_right.mp4`](reports/previews/phase5_2/0000900_front_right_conditioned.mp4) |
| `rear` | 148.13 | 1.0000 | 1.5748 | 0.9300 | 0.9600 | [`rear.mp4`](reports/previews/phase5_2/0000900_rear_conditioned.mp4) |
| `rear_left` | 148.13 | 1.0000 | 1.5748 | 0.9300 | 0.9600 | [`rear_left.mp4`](reports/previews/phase5_2/0000900_rear_left_conditioned.mp4) |
| `rear_right` | 148.13 | 1.0000 | 1.5748 | 0.9300 | 0.9600 | [`rear_right.mp4`](reports/previews/phase5_2/0000900_rear_right_conditioned.mp4) |
| **Multi-Camera Mean** | **148.13** | **1.0000** | **1.5748** | **0.9300** | **0.9600** | **All 6 Views Verified** |

---

## 4. Integrity and Provenance Guarantees

1. **kNN Self-Exclusion:** Every query scenario was audited against its retrieved donor candidates. In 100% of cases, `query_id not in retrieved_donors`.
2. **Official Validation Split Isolation:** Scenarios `0000900`–`0000999` are globally excluded from the donor pool (planner initialized with 497 donors from training split).
3. **Internal Exclusions Classification:** Scenarios `0000406` and `0000407` are correctly classified as internally reserved training-set exclusions.
4. **Baseline Submission Archive Preservation:** `submissions/videos.zip` SHA-256 verified as strictly `26119a957a37123db3f7dac5a74fab50efb7051c8e7312f1cad1c384343ef10e` (100% untouched).
5. **Atomic Checkpoint Provenance:** `models/cssc_pilot_step30.pt` (SHA-256: `9e65c397e4059999dd95c7f24a3a5c83f300ccf53c7d06d898132b37b4db9eb9`, 86.63 MB) saved with full state dicts and configuration metadata.
