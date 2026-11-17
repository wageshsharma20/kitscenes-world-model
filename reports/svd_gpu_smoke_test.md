# SVD-XT GPU Smoke Test & Preflight Audit Report

- **Date/Time (UTC):** 2026-11-17 09:30:43 UTC
- **Host / Device:** Tesla T4 (`cuda:0`)
- **Total GPU VRAM:** 14.56 GB
- **CUDA Compute Capability:** 7.5
- **Overall Status:** `REAL_GPU_PASSED` — **FIT: PASSES** (< 15.0 GB)

---

## 1. Executive Summary

| Metric | Measured Value | Kaggle 16GB T4 Budget / Target | Compliance Status |
| :--- | :--- | :--- | :--- |
| **Peak Reserved VRAM** | **`14.34 GB`** | `< 15.00 GB` | ✓ PASS |
| **Peak Allocated VRAM** | **`13.35 GB`** | `< 14.00 GB` | ✓ PASS |
| **Baseline UNet + Adapter VRAM** | `2.92 GB` (Allocated) | `< 5.00 GB` | ✓ PASS |
| **Forward Pass Latency** | `8.724 s` | Single batch forward | ✓ Complete |
| **Backward Pass Latency** | `24.093 s` | Single batch autograd | ✓ Complete |
| **Total Step Latency** | `32.817 s` | Preflight forward+backward | ✓ Complete |
| **Backbone Parameters Frozen** | `True` (100% frozen) | `all(p.grad is None)` | ✓ PASS |
| **Adapter Gradients Active** | `True` (Active) | `any(p.grad.norm() > 0)` | ✓ PASS |
| **CSSC MultiViewConditioner** | `44,777,728` parameters | 27-token bank active | ✓ PASS |
| **CSSC Gradients Active** | `True` (Active) | `any(p.grad.norm() > 0)` | ✓ PASS |

> [!WARNING]
> **High Peak VRAM Detected**: Peak memory exceeds 13.5 GB threshold. Enabling `--enable-gradient-checkpointing` is strongly recommended for training to prevent CUDA OOM on Kaggle 16GB T4 instances.

---

## 2. Configuration & Parameter Audit

- **Backbone Model ID:** `stabilityai/stable-video-diffusion-img2vid-xt`
- **UNet Parameter Count:** `1,524,623,082` parameters
- **UNet Parameter `requires_grad`:** `False` (strictly frozen)
- **Spatial Adapter Parameter Count:** `594,884` parameters
- **Spatial Adapter `requires_grad`:** `True` (active gradient target)
- **CSSC Conditioning:** Enabled (27-token bank: target view + pooled summary + 25-step kinematics)
- **Gradient Checkpointing:** `Enabled`

---

## 3. Detailed VRAM Progression

| Execution Phase | Allocated VRAM (GB) | Reserved / Cached VRAM (GB) | Delta from Baseline |
| :--- | :--- | :--- | :--- |
| **Initial Baseline (Weights Loaded)** | `2.925 GB` | `2.932 GB` | `+0.000 GB` |
| **Post-Forward Pass** | `7.001 GB` | `11.041 GB` | `+4.076 GB` |
| **Peak During Backward Pass** | **`13.352 GB`** | **`14.344 GB`** | `+10.427 GB` |

---

## 4. Autograd Gradient Isolation Audit

- **UNet Parameter Audit:** Evaluated 1,524,623,082 backbone weights. Every single parameter retained `p.grad is None`. Zero gradient leakage observed.
- **Spatial Adapter Audit:** Evaluated 594,884 adapter weights. Verified `p.grad is not None` and `p.grad.norm() > 0`. Backpropagation successfully navigated through the frozen UNet back to the conditioning adapter.
- **CSSC MultiViewConditioner Audit:** Evaluated 44,777,728 weights. Verified `p.grad is not None` and `p.grad.norm() > 0`. Backpropagation successfully navigated through the frozen UNet cross-attention to the 27-token multi-view conditioning bank.

---

## 5. Preflight Conclusion & Deployment Guidance

1. **Kaggle T4 Suitability:** Fits within single 16GB T4 instance.
2. **Gradient Checkpointing Recommendation:** Recommended
3. **Execution Ready:** Model architecture and autograd computation graphs are verified and ready for training.
