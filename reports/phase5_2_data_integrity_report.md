# Phase 5.2: Six-Camera Data Integrity & Provenance Audit Report

- **Date/Time (UTC):** 2026-11-17T14:50:02.626387+00:00
- **Status:** `✅ ALL_SIX_CAMERA_ASSETS_VERIFIED_GENUINE`
- **Canonical Camera Order:** `['front', 'front_left', 'front_right', 'rear', 'rear_left', 'rear_right']`
- **Total Scenarios Audited:** 6 (5 train, 1 official val)
- **Total Camera Streams:** 36 video/latent streams

---

## 1. Executive Data Quality & Verification Summary

| Integrity Contract | Verified Requirement | Status |
| :--- | :--- | :--- |
| **All Six Cameras Present** | Exact 6-camera rig: `front`, `front_left`, `front_right`, `rear`, `rear_left`, `rear_right` | ✅ PASS |
| **Identical Camera Ordering** | Strict `spec.CAMERAS` order enforced in data, models, and eval | ✅ PASS |
| **No Duplicate Camera Tensors** | Cross-camera mean absolute difference $\Delta > 0.05$ across all views | ✅ PASS |
| **Video Frame Count** | Exactly 25 frames @ 5.0 fps ($876 \times 568$) for all sequences | ✅ PASS |
| **Latent Dimensions** | Target latents $[1, 25, 4, 72, 128]$, Initial $[1, 1, 4, 72, 128]$ | ✅ PASS |
| **CLIP Vision Dimension** | Real CLIP embedding $[1, 1, 1024]$ per camera (no duplicated front) | ✅ PASS |
| **kNN Self-Exclusion** | Query scenario strictly excluded from its own donor pool (`query_id != donor_id`) | ✅ PASS |
| **Official Validation Split** | `0000900`–`0000999` strictly reserved for evaluation, blocked from donors | ✅ PASS |
| **Internal Exclusions** | `0000406` and `0000407` documented as internal training holdouts | ✅ PASS |

---

## 2. Training Subset Manifest (`configs/manifests/phase5_2_train_manifest.json`)

| Scenario ID | Maneuver Instruction | Scenario Type | kNN Method | Retrieved Donor IDs |
| :--- | :--- | :--- | :--- | :--- |
| `0000400` | turn left | 4 intersection | `knn_retrieval_with_cv_blend` | `['0000859', '0000760', '0000479', '0000637', '0000758']` |
| `0000401` | turn right | 4 intersection | `knn_retrieval_with_cv_blend` | `['0000653', '0000505', '0000781', '0000813', '0000436']` |
| `0000402` | use right lane | 3 overtake or lane change | `knn_retrieval_with_cv_blend` | `['0000460', '0000459', '0000419', '0000779', '0000722']` |
| `0000403` | drive straight on | 1 heavy rain | `knn_retrieval_with_cv_blend` | `['0000620', '0000702', '0000718', '0000457', '0000613']` |
| `0000404` | use left lane | 3 overtake or lane change | `knn_retrieval_with_cv_blend` | `['0000826', '0000475', '0000623', '0000730', '0000580']` |

---

## 3. Official Held-Out Validation Manifest (`configs/manifests/phase5_2_val_manifest.json`)

| Scenario ID | Maneuver Instruction | Scenario Type | kNN Method | Retrieved Donor IDs |
| :--- | :--- | :--- | :--- | :--- |
| `0000900` | drive straight on | 6 snow & wintry mix | `knn_retrieval_with_cv_blend` | `['0000868', '0000560', '0000894', '0000445', '0000454']` |

---

## 4. Checksums and Storage Policy

- Manifest file: `configs/manifests/phase5_2_train_manifest.json`
- Manifest file: `configs/manifests/phase5_2_val_manifest.json`
- All multi-camera bundles are tracked via SHA-256 hashes in the manifests.
