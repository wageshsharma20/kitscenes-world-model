# Phase 5.4: Provenance Reconciliation & Mathematical Audit Report

- **Date/Time (UTC):** 2026-11-19T05:10:00Z
- **Status:** `RECONCILIATION_COMPLETE_AND_STANDARDIZED`
- **Purpose:** Full forensic resolution of conflicting execution records identified between Phase 5.4 reports and user telemetry, and documentation of the Phase 5.5.1 mathematical sampler audit.

---

## 1. Executive Summary of Forensic Findings

A thorough audit of the committed historical records (`reports/phase5_4_ablation_and_evaluation.json`, `reports/phase5_4_ablation_and_evaluation.md`, and git history) establishes three clear explanations for the reported discrepancies:

| Dimension | Telemetry Summary Claim | Committed Report Record | Forensic Finding & Truth |
| :--- | :--- | :--- | :--- |
| **Denoising Steps** | Described multi-step rollout | `denoising_steps: 1` | On Kaggle GPU, Phase 5.4 was launched with `--steps 1` to verify memory and multi-camera tensor mechanics rapidly across 30 rollouts (5 variants × 6 cameras). All metrics in Phase 5.4 reflect 1 Euler denoising step. |
| **Baseline Archive Integrity** | Local archive intact | `ARCHIVE_MISSING_ON_DISK` | The script executed inside an ephemeral Kaggle container where the repo was cloned fresh from GitHub. Because `submissions/videos.zip` (1.4 GB) is gitignored, it was absent in `/kaggle/working`. Locally, the archive has remained present and intact (`26119a957a...`). |
| **Evaluated Checkpoint** | Expected 30 converged steps | Internal training step: `6` | Commit `cf146ee` pushed Markdown/JSON reports from the 30-step Kaggle run, but did **not** commit the 86.6 MB binary file. The Git repo retained the 6-step pilot checkpoint (`9e65c397...`). All Phase 5.3 and 5.4 evaluations ran on Step 6. |

---

## 2. Mathematical Sampler Defects Identified (Phase 5.5.1 Audit)

Code review against the official Hugging Face `diffusers.StableVideoDiffusionPipeline` identified four mathematical defects that affected all historical custom sampler runs (Phases 5.2 through 5.4):

### Defect 1: Missing Initial Noise Scaling (`init_noise_sigma`)
- **Problem:** Custom rollout initialized `latents = starting_noise.clone()`, omitting `scheduler.init_noise_sigma`.
- **Impact:** For `EulerDiscreteScheduler` with Karras sigmas, `init_noise_sigma` is `700.0007`. When `scale_model_input` scaled latents by `sqrt(sigma^2 + 1) \approx 700.0`, the UNet received latents with standard deviation $\approx 0.0014$ instead of $1.0$! The UNet was essentially fed near-zero inputs.
- **Correction:** `latents = starting_noise.clone() * scheduler.init_noise_sigma` produces unit variance inputs ($\approx 1.0$) at $t=0$, matching official SVD-XT training conditions.

### Defect 2: CFG Schedule Stepped Across Iterations instead of Video Frames
- **Problem:** Guidance scale was varied from 1.0 to 3.0 across the 25 denoising iterations ($i = 0 \dots 24$).
- **Impact:** Official SVD-XT applies CFG across the **25 video frames** (the temporal dimension) using `guidance_scale = torch.linspace(min_guidance, max_guidance, num_frames).view(1, 25, 1, 1, 1)`. Frame 0 receives minimum guidance (1.0) to preserve the observed start frame faithfully; Frame 24 receives maximum guidance (3.0) to extrapolate distant motion cleanly.
- **Correction:** Implemented official per-frame guidance tensor broadcast to `[1, 25, 1, 1, 1]`.

### Defect 3: Duplicate Image Latents in Unconditional CFG Branch
- **Problem:** `image_latents` was repeated identically into both halves of the CFG batch.
- **Impact:** The unconditional branch was not unconditional regarding image guidance. The CFG delta $(pred_{\text{cond}} - pred_{\text{uncond}})$ was computed with respect to an image-conditioned baseline, distorting extrapolation.
- **Correction:** Unconditional branch receives `torch.zeros_like(image_latents)`, while conditional branch receives actual observed-image VAE latents.

### Defect 4: FPS Micro-Conditioning Off-by-One
- **Problem:** Custom rollout passed `fps = 5` in added time IDs.
- **Impact:** SVD-XT was micro-conditioned on `fps - 1` during pretraining. Passing 5 conditioned the model as if it were 6 FPS.
- **Correction:** Added time IDs pass `fps - 1 = 4` for 5.0 FPS video generation.

---

## 3. Mathematical Verification & Identity Proof

In `tests/test_svd_inference_math_and_equivalence.py`, the corrected custom sampler (`run_official_svd_diffusion_rollout` in `kitscenes/svd_inference.py`) was compared directly against the official SVD pipeline denoising logic on identical noise, image latents, and conditioning states.

- **Maximum Absolute Difference:** `0.0000000` (Exact bitwise mathematical identity verified).
- **Test Suite Status:** 6 passed in 3.01s with 0 errors.

---

## 4. Preservation of Baseline Archive

- File path: `submissions/videos.zip`
- SHA-256: `26119a957a37123db3f7dac5a74fab50efb7051c8e7312f1cad1c384343ef10e`
- Size: `1,416,423,737 bytes` (1.32 GiB)
- Status: **STRICTLY PRESERVED & FULLY VERIFIED ON DISK**
