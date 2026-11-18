# Phase 5.4: Checkpoint Forensics & Provenance Audit Report

- **Audit Date (UTC):** 2026-11-18T08:45:30.523542+00:00
- **Status:** `AUDIT_COMPLETE_DISCREPANCY_EXPLAINED`
- **Current File Path:** `models/cssc_pilot_step30.pt`
- **Current File SHA-256:** `9e65c397e4059999dd95c7f24a3a5c83f300ccf53c7d06d898132b37b4db9eb9` (86.63 MB)

---

## 1. Checkpoint Hash Discrepancy Investigation

### The Reported Discrepancy
- **Phase 5.2 Training Report (`reports/phase5_2_multicamera_pilot.json`):** `b3b0841ea63cea1d81e4be35fba409d0ba83523863b22701e03a18ec3ad11c61`
- **Phase 5.3 Evaluation JSON (`reports/phase5_3_predictive_evaluation.json`):** `9e65c397e4059999dd95c7f24a3a5c83f300ccf53c7d06d898132b37b4db9eb9`

### Root-Cause Forensic Finding
1. In commit `f52161d`, a local 6-step pilot run produced a checkpoint with SHA-256 `9e65c397...`, which was committed to git as `models/cssc_pilot_step30.pt`.
2. In Phase 5.2 on Kaggle, the 30-update training ran on ephemeral Kaggle GPU disk, producing checkpoint `b3b0841e...`.
3. Commit `cf146ee` committed the Markdown and JSON reports from Kaggle, but did **not** commit the 86.6 MB binary file back into GitHub.
4. When Phase 5.3 cloned the GitHub repository fresh on Kaggle (`git clone ...`), it pulled the committed artifact `9e65c397...`.
5. Therefore, the evaluated checkpoint in Phase 5.3 was the committed step-6 checkpoint (`9e65c397...`), rather than the ephemeral Kaggle 30-update file (`b3b0841e...`).

---

## 2. Tensor-Level Checkpoint Hashes & Parameter Inventory

### Metadata Extracted from `models/cssc_pilot_step30.pt`
- **Internal Step Counter:** `6` (Step 6)
- **Saved Timestamp:** `2026-11-17T14:59:30.978267+00:00`
- **Dataset Version:** `KITScenes-LongTail Phase 5.2 Multi-Camera Pilot`
- **Training Scenarios:** `['0000400', '0000401', '0000402', '0000403', '0000404']`
- **Recorded Loss:** `1.003276`
- **Adapter Weight $\Delta$ Norm:** `0.007391`
- **Conditioner Weight $\Delta$ Norm:** `0.000992`
- **Baseline Submission Archive SHA-256:** `26119a957a37123db3f7dac5a74fab50efb7051c8e7312f1cad1c384343ef10e`

### Module Tensor Inventories
| Module | Tensor Count | Parameter Count | Composite Tensor SHA-256 |
| :--- | :--- | :--- | :--- |
| `spatial_adapter_state_dict` | 20 | 594,884 | `235897a5186077a349de5271f30705be0ff7a92f1f0c16cb0095295f4ca92af4` |
| `conditioner_state_dict` | 88 | 44,803,424 | `ec74a02730e569d558613546f30d6b278d6771a433ac46a1bb3d72d130dc2946` |

---

## 3. Preservation & Verification Guarantees

1. **Checkpoint Frozen:** The current artifact `models/cssc_pilot_step30.pt` is strictly frozen and preserved for the Phase 5.4 ablation.
2. **Deterministic Reproducibility:** Every tensor has been individually hashed and recorded in `reports/phase5_4_checkpoint_provenance_audit.json`.
3. **No Phantom Claims:** We explicitly document that evaluated checkpoint `9e65c397...` reflects early training (step 6), not 30 converged steps.
