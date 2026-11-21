# Phase 5.5: Multi-Scenario Head-to-Head Candidate Evaluation Report

- **Date/Time (UTC):** 2026-11-21T07:00:01.506149+00:00
- **Device:** `cuda:0` (Mock Mode: `False`)
- **Denoising Steps:** 25 steps (EulerDiscreteScheduler)
- **Base Seed:** 42 (Deterministic camera seed offsets applied)
- **Evaluated Checkpoint:** `/kaggle/working/repo/models/cssc_pilot_step30.pt` (SHA-256: `9e65c397e4059999...`, Step: 6)
- **SpatialAdapter Status:** `STRICTLY_DISABLED` (Residual perturbation bypassed)
- **Baseline Submission Archive:** `26119a957a37123db3f7dac5a74fab50efb7051c8e7312f1cad1c384343ef10e` (Intact: `True`)

---

## 1. Executive Summary & Selection Decision

This evaluation pits the two viable candidate architectures against each other and against Vanilla SVD across diverse official validation scenarios using official SVD-XT inference mathematics:
- **Candidate 1 (`project_baseline`)**: kNN trajectory planner + trajectory-derived SVD motion bucket.
- **Candidate 2 (`cssc_only`)**: MultiViewConditioner 27-token bank (camera + scene + waypoints), SpatialAdapter disabled.
- **Reference System (`vanilla_svd`)**: Pretrained frozen SVD-XT, default motion bucket 127.

### Aggregate Performance Summary
| Candidate | Mean PSNR (dB) | Mean SSIM | Mean Boundary Strip Corr | Mean Local EMC Approx |
| :--- | :--- | :--- | :--- | :--- |
| `vanilla_svd` | **18.04** (±3.89) | **0.6519** (±0.1432) | 0.4026 | 0.000 |
| `project_baseline` | **18.20** (±3.82) | **0.6373** (±0.1482) | 0.3967 | 0.000 |
| `cssc_only` | **18.10** (±4.07) | **0.6243** (±0.1451) | 0.4039 | 0.000 |

### Pairwise Head-to-Head Comparison
- **CSSC-Only vs Project Baseline:**
  - Mean $\Delta$ PSNR: `-0.10 dB` (Wins: 2/9)
  - Mean $\Delta$ SSIM: `-0.0130` (Wins: 2/9)
- **CSSC-Only vs Vanilla SVD:**
  - Mean $\Delta$ PSNR: `+0.06 dB` (Wins: 5/9)
  - Mean $\Delta$ SSIM: `-0.0276` (Wins: 1/9)
- **Project Baseline vs Vanilla SVD:**
  - Mean $\Delta$ PSNR: `+0.16 dB` (Wins: 6/9)
  - Mean $\Delta$ SSIM: `-0.0146` (Wins: 0/9)

---

## 2. Per-Scenario Performance Breakdown

| Scenario ID | Instruction | `vanilla_svd` PSNR/SSIM | `project_baseline` PSNR/SSIM | `cssc_only` PSNR/SSIM | Winning Architecture |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `0000900` | drive straight on | 20.83 / 0.8066 | 20.82 / 0.7887 | 20.18 / 0.7610 | **`project_baseline`** |
| `0000901` | drive straight on | 16.87 / 0.6753 | 16.89 / 0.6614 | 16.47 / 0.6346 | **`project_baseline`** |
| `0000902` | drive straight on | 15.71 / 0.5782 | 16.06 / 0.5693 | 15.83 / 0.5519 | **`project_baseline`** |
| `0000903` | drive straight on | 24.22 / 0.8575 | 24.27 / 0.8550 | 23.01 / 0.8240 | **`project_baseline`** |
| `0000904` | drive straight on | 13.81 / 0.4831 | 13.66 / 0.4546 | 13.62 / 0.4478 | **`project_baseline`** |
| `0000905` | drive straight on | 14.54 / 0.4983 | 15.49 / 0.4881 | 15.08 / 0.4775 | **`project_baseline`** |
| `0000906` | drive straight on | 24.42 / 0.8404 | 24.55 / 0.8379 | 26.56 / 0.8462 | **`cssc_only`** |
| `0000911` | drive straight on | 14.53 / 0.4955 | 14.75 / 0.4797 | 14.63 / 0.4717 | **`project_baseline`** |
| `0000919` | drive straight on | 17.44 / 0.6324 | 17.30 / 0.6008 | 17.50 / 0.6041 | **`cssc_only`** |

---

## 3. Metric Disclaimers & Experimental Guarantees

1. **Inference Mathematics Audit:** Initial noise is properly scaled by `scheduler.init_noise_sigma`. CFG schedule varies across frames, not steps. Unconditional image latents are strictly zero-conditioned. Added time IDs pass `fps - 1 = 4`.
2. **Strict Two-Stage Decoupling:** Stage A (prediction) has zero access to ground-truth future videos or future trajectories.
3. **Zero Silent Fallbacks:** Black-frame padding is strictly disabled; missing future videos trigger an explicit evaluation error. Missing reference trajectories produce null MMS.
4. **SpatialAdapter Frozen:** Bypassed completely across all runs. Zero latent energy perturbation.
