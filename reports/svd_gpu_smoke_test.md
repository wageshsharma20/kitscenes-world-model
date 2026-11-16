# SVD-XT GPU Smoke Test & Preflight Audit Report

- **Date/Time (UTC):** 2026-11-16 19:27:10 UTC
- **Host / Device:** Mock CPU Surrogate (`cpu`)
- **Total GPU VRAM:** 0.00 GB
- **CUDA Compute Capability:** N/A
- **Overall Status:** `MOCK_ONLY` — **NOT MEASURED (Mock CPU Surrogate)**

---

## 1. Executive Summary

| Metric | Measured Value | Kaggle 16GB T4 Budget / Target | Compliance Status |
| :--- | :--- | :--- | :--- |
| **Peak Reserved VRAM** | **N/A (Mock CPU)** | `< 15.00 GB` | ⚠️ NOT MEASURED (Mock CPU) |
| **Peak Allocated VRAM** | **N/A (Mock CPU)** | `< 14.00 GB` | ⚠️ NOT MEASURED (Mock CPU) |
| **Baseline UNet + Adapter VRAM** | `0.00 GB` (Allocated) | `< 5.00 GB` | N/A |
| **Forward Pass Latency** | `0.200 s` | Single batch forward | ✅ Complete |
| **Backward Pass Latency** | `0.146 s` | Single batch autograd | ✅ Complete |
| **Total Step Latency** | `0.347 s` | Preflight forward+backward | ✅ Complete |
| **Backbone Parameters Frozen** | `True` (100% frozen) | `all(p.grad is None)` | ✅ PASS |
| **Adapter Gradients Active** | `True` (Active) | `any(p.grad.norm() > 0)` | ✅ PASS |
| **CSSC MultiViewConditioner** | `44,777,728` parameters | 27-token bank active | ✅ PASS |
| **CSSC Gradients Active** | `True` (Active) | `any(p.grad.norm() > 0)` | ✅ PASS |

> [!NOTE]
> **Mock Surrogate Mode**: This execution evaluated autograd graph connectivity and parameter freezing contracts on synthetic CPU tensors. GPU memory usage was **not measured** and must not be used to infer Kaggle 16GB T4 hardware compatibility.

---

## 2. Configuration & Parameter Audit

- **Backbone Model ID:** `mock-svd-unet`
- **UNet Parameter Count:** `12,740` parameters
- **UNet Parameter `requires_grad`:** `False` (strictly frozen)
- **Spatial Adapter Parameter Count:** `594,884` parameters
- **Spatial Adapter `requires_grad`:** `True` (active gradient target)
- **CSSC Conditioning:** Enabled (27-token bank: target view + pooled summary + 25-step kinematics)
- **Gradient Checkpointing:** `Enabled`

---

## 3. Detailed VRAM Profiling Progression

| Execution Phase | Allocated VRAM (GB) | Reserved / Cached VRAM (GB) | Delta from Baseline |
| :--- | :--- | :--- | :--- |
| **Initial Baseline (Weights Loaded)** | `0.000 GB` | `0.000 GB` | `+0.000 GB` |
| **Post-Forward Pass** | `0.000 GB` | `0.000 GB` | `+0.000 GB` |
| **Peak During Backward Pass** | **`0.000 GB`** | **`0.000 GB`** | `+0.000 GB` |

---

## 4. Autograd Gradient Isolation Audit

- **UNet Parameter Audit:** Evaluated 12,740 backbone weights. Every single parameter retained `p.grad is None`. Zero gradient leakage observed.
- **Spatial Adapter Audit:** Evaluated 594,884 adapter weights. Verified `p.grad is not None` and `p.grad.norm() > 0`. Backpropagation successfully navigated through the frozen UNet back to the conditioning adapter.
- **CSSC MultiViewConditioner Audit:** Evaluated 44,777,728 weights. Verified `p.grad is not None` and `p.grad.norm() > 0`. Backpropagation successfully navigated through the frozen UNet cross-attention to the 27-token multi-view conditioning bank.

---

## 5. Preflight Conclusion & Deployment Guidance

1. **Kaggle T4 Suitability:** NOT MEASURED (Executed on CPU mock surrogate; requires genuine NVIDIA T4 hardware).
2. **Gradient Checkpointing Recommendation:** NOT MEASURED (Requires actual GPU memory profiling on Kaggle T4).
3. **Execution Ready:** Model architecture and autograd computation graphs are verified and ready for training.
