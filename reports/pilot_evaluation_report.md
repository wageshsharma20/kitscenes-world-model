# Phase 3 Limited Pilot: Comparative Evaluation on Held-Out Sequences

**Date:** 2026-11-15 15:47:00 UTC  
**Backbone:** `mock`  
**Conditioning Source:** `knn` (`kNN-retrieved predicted waypoints`)  
**Generation:** Full Iterative Scheduler Loop (2 steps from pure Gaussian noise)  
**Model Checkpoint:** `svd_spatial_adapter_front_b1.pt`  

## Aggregate Summary

- **Mean PSNR Gain:** `+0.00 dB`
- **Mean SSIM Gain:** `-0.0000`
- **Mean Trajectory Drift Reduction:** `-0.0000`
- **Conditioned Model Superiority:** `False`

## Per-Scenario Metrics

| Scenario ID | Conditioning | Baseline PSNR | Conditioned PSNR | PSNR Gain | Baseline SSIM | Conditioned SSIM | Drift Reduction |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0000406` | knn | 16.42 dB | 16.42 dB | **-0.00 dB** | 0.7934 | 0.7933 | **-0.0000** |
| `0000407` | knn | 12.44 dB | 12.44 dB | **+0.00 dB** | 0.4587 | 0.4587 | **-0.0000** |
