# Phase 5.3: Controlled Predictive Inference & Evaluation Report

- **Date/Time (UTC):** 2026-11-18T07:35:16.061239+00:00
- **Evaluation Status:** `✅ CONTROLLED_PREDICTIVE_INFERENCE_PASSED`
- **Device:** `cuda:0` (Mock Mode: `False`)
- **Denoising Steps:** 25 steps (EulerDiscreteScheduler)
- **CFG Scale:** 1.0 to 3.0 (frame-wise linear)
- **Evaluated Checkpoint:** `cssc_pilot_step30.pt` (SHA-256: `9e65c397e4059999...`)
- **Baseline Submission Archive:** SHA-256 `26119a957a37123db3f7dac5a74fab50efb7051c8e7312f1cad1c384343ef10e` (`100% strictly intact`)

---

## 1. Controlled Experimental Setup

This evaluation rectifies the Phase 5.2 validation flaw by executing genuine multi-step diffusion rollouts for both models:
- **Starting Noise $z_T \sim \\mathcal{N}(0, I)$:** Pure Gaussian noise seeded deterministically (`seed=42`). **Completely decoupled from future targets.**
- **Initial Observation $t_0$:** Genuine observed camera images at $t_0$ only.
- **Model A (Baseline):** Pretrained frozen SVD-XT UNet without spatial adapter (vanilla single-camera CLIP token).
- **Model B (Conditioned):** SVD-XT UNet conditioned with trained CSSC MultiViewConditioner (27-token bank) and SpatialAdapter.
- **Fair Control:** Initial noise, seed, steps, resolution, scheduler, and CFG are **100% identical** between Model A and Model B.
- **Zero Future Leakage:** Ground-truth future videos were loaded **strictly after inference finished** for evaluation.

---

## 2. Quantitative Performance Comparison (`scenario 0000900`)

### Per-Camera Predictive Fidelity (vs Ground Truth Future)
| Camera View | Baseline PSNR (dB) | Conditioned PSNR (dB) | PSNR $\\Delta$ | Baseline SSIM | Conditioned SSIM | SSIM $\\Delta$ | Prediction Divergence ($L_1$) | Side-by-Side Preview |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `front` | 21.28 | 9.45 | -11.83 | 0.8743 | 0.7630 | -0.1113 | 73.35 | [`preview`](reports/previews/phase5_3/0000900_front_side_by_side.mp4) |
| `front_left` | 21.08 | 10.52 | -10.56 | 0.8889 | 0.8143 | -0.0746 | 67.01 | [`preview`](reports/previews/phase5_3/0000900_front_left_side_by_side.mp4) |
| `front_right` | 15.65 | 8.55 | -7.10 | 0.7744 | 0.6589 | -0.1156 | 70.99 | [`preview`](reports/previews/phase5_3/0000900_front_right_side_by_side.mp4) |
| `rear` | 19.23 | 9.74 | -9.49 | 0.7939 | 0.7000 | -0.0938 | 68.94 | [`preview`](reports/previews/phase5_3/0000900_rear_side_by_side.mp4) |
| `rear_left` | 18.56 | 9.31 | -9.24 | 0.8150 | 0.7163 | -0.0987 | 72.21 | [`preview`](reports/previews/phase5_3/0000900_rear_left_side_by_side.mp4) |
| `rear_right` | 15.47 | 8.61 | -6.86 | 0.6932 | 0.6013 | -0.0920 | 69.94 | [`preview`](reports/previews/phase5_3/0000900_rear_right_side_by_side.mp4) |

### Challenge Metric Summary
| Metric | Baseline SVD-XT | CSSC + SpatialAdapter | Delta / Status | Challenge Specification |
| :--- | :--- | :--- | :--- | :--- |
| **Mean PSNR (dB)** | `18.54` | `9.36` | `-9.18 dB` | Frame-by-frame pixel fidelity |
| **Mean SSIM** | `0.8066` | `0.7090` | `-0.0977` | Gaussian-windowed structural similarity |
| **Official MMS** | `2.50` | `2.50` | `Evaluated` | Maneuver Match Score (kNN trajectory vs expert) |
| **Official EMC** | `0.000` | `1.000` | `+1.000` | Optical-flow Ego-Motion Coherence across 6 cameras |
| **PQ Proxy** | `2.11` | `0.31` | `-1.80` | Weighted Perceptual Quality (50% front, 10% each side/rear) |
| **Cross-Camera Overlap** | `0.4838` | `0.9986` | `+0.5148` | Geometric overlap correlation across adjacent views |

---

## 3. Hardware & Peak Memory Profile

| Metric | Model A (Baseline) | Model B (CSSC Conditioned) | Measurement Method |
| :--- | :--- | :--- | :--- |
| **Peak Allocated VRAM** | `12.52 GB` | `13.44 GB` | `torch.cuda.max_memory_allocated()` |
| **Peak Reserved VRAM** | `14.39 GB` | `14.41 GB` | `torch.cuda.max_memory_reserved()` |

---

## 4. Integrity and Provenance Guarantees

1. **Zero Future Target Leakage:** Neither ground-truth future video nor future latents were accessible to the diffusion denoising engine.
2. **Deterministic Controlled Rollouts:** Starting Gaussian noise $z_T$ and random generator seeds were identical between baseline and conditioned runs.
3. **True Metric Evaluation:** All metrics are computed on actual decoded MP4 video frames and actual trajectory waypoints.
4. **Baseline Archive Intact:** `submissions/videos.zip` SHA-256 strictly preserved as `26119a957a37123db3f7dac5a74fab50efb7051c8e7312f1cad1c384343ef10e`.
