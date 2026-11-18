# Phase 5.2 Validation Audit & Correction Report

- **Date:** 2026-11-18
- **Auditor:** DeepMind Antigravity Engineering
- **Audit Target:** `reports/phase5_2_multicamera_pilot.md` & `reports/phase5_2_multicamera_pilot.json`
- **Verdict:** `⚠️ SECTION 3 VALIDATION METRICS FORMALLY INVALIDATED FOR PREDICTIVE PERFORMANCE`

---

## 1. Executive Summary

A rigorous code and methodology audit of Phase 5.2 (`train_svd_multicamera_pilot.py`, commit `cf146ee`) identified a critical evaluation flaw in the validation phase (Phase D). The reported metrics in Section 3 of the Phase 5.2 report (PSNR 38.14 dB, SSIM 0.9926, PQ Proxy 0.9718) **measured VAE latent reconstruction of known ground-truth future frames rather than future prediction by the diffusion world model**.

This document formally invalidates the predictive performance claims of Phase 5.2, explains the root causes, and delineates what Phase 5.2 legitimately established versus what remained unproven.

---

## 2. Identified Validation Flaws

### Flaw 1: Ground-Truth Latents Decoded Directly
In `train_svd_multicamera_pilot.py` (lines 753–757):
```python
with torch.no_grad():
    val_lat = all_scenario_latents[val_sid][cam].to(device=parsed_device, dtype=dtype)
    # Decode through VAE
    decoded_tensors = decode_latents_temporal(vae, val_lat, decode_chunk_size=8)
```
The script completely deleted the UNet (`del unet`) prior to Phase D and passed `all_scenario_latents[val_sid][cam]`—the VAE latents of the actual ground-truth future frames—directly into the VAE decoder. The trained SVD-XT UNet, SpatialAdapter, and MultiViewConditioner were never called to predict future latents from starting noise $z_T \sim \mathcal{N}(0, I)$.

### Flaw 2: Baseline Video Was Ground-Truth Video
In `train_svd_multicamera_pilot.py` (lines 767–770):
```python
# Write baseline comparison MP4 (unconditioned or smoothed prior)
writer_base = cv2.VideoWriter(str(base_preview_mp4), fourcc, 5.0, (876, 568))
for f in gt_frames:
    writer_base.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
writer_base.release()
```
`base_preview_mp4` was created by writing `gt_frames` (the ground-truth future frames) directly to disk. It was not the output of the unmodified pretrained SVD-XT baseline diffusion model.

### Flaw 3: Synthetic Formulas for Cross-Camera and Trajectory Metrics
In `train_svd_multicamera_pilot.py`:
- Cross-camera consistency was calculated via a synthetic affine formula from SSIM ($0.93 + \dots$).
- Trajectory adherence was similarly derived from a formula rather than vehicle kinematics or optical flow.
- The "PQ proxy" was an arbitrary harmonic mean of PSNR and SSIM, not the challenge's official Q-Align (OneAlign) metric.

### Flaw 4: Memory Reporting Recorded Post-Step Current, Not Peak
The reported VRAM values (3.33 GB allocated / 3.44 GB reserved) were sampled via `torch.cuda.memory_allocated()` after the step had finished and memory had been freed, rather than profiling peak allocation during the forward-backward pass via `torch.cuda.max_memory_allocated()`.

---

## 3. What Was Genuinely Accomplished in Phase 5.2

The invalidation of Phase 5.2's predictive validation metrics does **not** diminish the genuine engineering milestones verified in the training pipeline:
1. **Six-Camera Data Pipeline:** Successfully extracted, padded, and audited genuine $t_0$ observation images and 25-frame future video targets for all 6 canonical camera views across 5 training scenarios and 1 validation scenario.
2. **Six-View Conditioning Integration:** MultiViewConditioner received all 6 genuine initial camera embeddings $[1, 6, 1024]$, certified camera calibration matrices, and 25-step kinematics $[1, 25, 6]$.
3. **Balanced Multi-Camera Training Schedule:** Exactly 30 optimizer updates were executed on Kaggle Tesla T4 GPU, cycling uniformly (5 updates per camera view $\times 6$ cameras).
4. **Verified Parameter Updates:** The frozen 1.52B UNet contract was strictly preserved (`all(p.grad is None) == True`), while SpatialAdapter ($\Delta w = 1.4746$) and MultiViewConditioner ($\Delta w = 4.2344$) received active non-zero weight updates.
5. **kNN Provenance & Validation Isolation:** Scenarios `0000900`–`0000999` and internal exclusions `0000406`–`0000407` were strictly isolated from the donor pool, and query self-exclusion was verified in 100% of cases.
6. **Submission Archive Integrity:** `submissions/videos.zip` SHA-256 remained strictly `26119a957a37123db3f7dac5a74fab50efb7051c8e7312f1cad1c384343ef10e`.

---

## 4. Corrective Action: Phase 5.3 Mandate

To rectify these flaws, **Phase 5.3** has been established with the following mandatory specifications:
1. **Separate Inference Engine (`scripts/evaluate_cssc_checkpoint.py`):**
   - Must run iterative diffusion denoising starting from pure Gaussian noise $z_T \sim \mathcal{N}(0, I)$ (seeded with 42) using EulerDiscreteScheduler (25 steps) and linear CFG (1.0 to 3.0).
   - Ground-truth future videos and latents are strictly prohibited from being inputs or accessible to the inference function.
2. **Fair Controlled Baseline:**
   - Model A (Vanilla pretrained SVD-XT) and Model B (SVD-XT + CSSC & SpatialAdapter) must be evaluated from the exact same initial noise $z_T$, same seed, same $t_0$ images, same scheduler, and same CFG schedule.
3. **Real Metrics:**
   - True PSNR and true Gaussian-windowed SSIM calculated between generated frames and ground truth.
   - Official challenge metrics: MMS (`kitscenes.knn_trajectory.compute_mms_score`), EMC (`kitscenes.emc_eval.compute_video_optical_flow_emc`), and PQ (`kitscenes.pq_proxy.score_scenario_videos`).
   - True CUDA peak-memory profiling using `torch.cuda.reset_peak_memory_stats()` and `torch.cuda.max_memory_allocated()`.
