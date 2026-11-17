# SVD-XT Single Training-Step Validation Report

- **Date/Time (UTC):** 2026-11-17T14:03:55.330806+00:00
- **Host / Device:** Tesla T4 (`cuda:0`, Compute Capability 7.5)
- **Total GPU VRAM:** 14.56 GB
- **Execution Mode:** `GENUINE KAGGLE TESLA T4`
- **Scenario ID:** `0000400`
- **Overall Status:** `✓ REAL_GPU_TRAINING_STEP_PASSED`

---

## 1. Executive Summary

| Metric | Measured Value | Target / Requirement | Status |
| :--- | :--- | :--- | :--- |
| **SVD Continuous Velocity Loss** | **`0.210094`** | Finite scalar ($L > 0$) | ✓ PASS |
| **Peak Reserved VRAM** | **`14.42 GB`** | `< 15.00 GB` | ✓ PASS |
| **Peak Allocated VRAM** | **`13.36 GB`** | `< 14.00 GB` | ✓ PASS |
| **Pre-Optimizer Cache Flush** | `3.06 GB` alloc / `3.11 GB` res | **`11.45 GB` free headroom for AdamW** | ✓ PASS |
| **Forward Pass Latency** | `8.182 s` | Real SVD-XT forward | ✓ Complete |
| **Backward Pass Latency** | `23.635 s` | Real autograd backward | ✓ Complete |
| **Optimizer Step Latency** | `0.037 s` | AdamW update | ✓ Complete |
| **Total Step Latency** | `31.817 s` | Single training step | ✓ Complete |
| **Backbone Frozen Contract** | `True` | `all(p.grad is None)` on 1.52B UNet | ✓ PASS |
| **SpatialAdapter Weight Delta** | **`1.941921e-01`** | $\Delta w > 0$ | ✓ PASS |
| **MultiViewConditioner Weight Delta**| **`1.299430e-01`** | $\Delta w > 0$ | ✓ PASS |
| **Baseline Archive SHA-256** | `26119a957a37123d...` | Strictly identical | ✓ PASS |

---

## 2. Gradient Isolation & Sub-Module Audit

| Submodule | Parameter Count | Gradient Norm | Status |
| :--- | :--- | :--- | :--- |
| **UNet Backbone (Frozen)** | `1,524,623,082` | `None` (0.000) | ✓ 100% Frozen |
| **SpatialAdapter** | `594,884` | `4.223597e-01` | ✓ Actively Trained |
| **MultiViewConditioner (Total)** | `44,777,728` | `1.294734e-01` | ✓ Actively Trained |
| ↳ *SharedMotionEncoder* | — | `4.170369e-02` | ✓ Active Gradients |
| ↳ *CameraGeometryEncoder* | — | `4.398731e-02` | ✓ Active Gradients |
| ↳ *SurroundSceneFusion* | — | `3.012103e-02` | ✓ Active Gradients |
| ↳ *Motion Cross-Attention* | — | `2.094791e-02` | ✓ Active Gradients |
| ↳ *Refinement MLP* | — | `1.080922e-01` | ✓ Active Gradients |
| ↳ *Pool Projection* | — | `7.613595e-03` | ✓ Active Gradients |

---

## 3. Data Integrity & Leakage Prevention

- **Scenario ID:** `0000400`
- **Latent File:** `data/train_latents/0000400.pt` (Genuine encoded VAE latents: `target [1, 25, 4, 72, 128]`, `initial [1, 1, 4, 72, 128]`)
- **kNN Donor Count:** `497` donors.
- **Validation Scenarios Globally Excluded:** `['0000406', '0000407']` (Zero data leakage into donor pool).
- **Camera Calibration:** Calibrated pinhole parameters from `configs/kitscenes_camera_calibration.json`.
- **Checkpoint Artifact:** `models/adapter_single_step.pt` (259.80 MB)
- **Inference Denoising:** 5-step Euler denoising passed in `torch.no_grad()`.
- **Baseline Submission Archive:** Verified intact with SHA-256 `26119a957a37123db3f7dac5a74fab50efb7051c8e7312f1cad1c384343ef10e`.
