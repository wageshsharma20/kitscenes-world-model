# 6-Camera Multi-View Spatial Conditioning Visual Audit & Calibration Report

### Autonomous Driving World Modeling Geometry, Line Clipping, & Spatial Representation Verification

*Generated on: 2026-11-15 09:27:29 UTC*

## 1. Camera Calibration Status & Provenance

> [!IMPORTANT]
> **PROVISIONAL NOMINAL SENSOR RIG MODEL:** The official KITScenes-LongTail dataset files do not publish explicit extrinsic transformation matrices or intrinsic camera parameter matrices. The geometry used in this pipeline is a provisional nominal model derived from standard autonomous surround sensor architectures (e.g. nuScenes/KITTI-360) and scaled to the vehicle wheelbase. It is an approximation and must not be described as certified exact calibration.

### Parameter Provenance Table:

| Parameter | Nominal Value | Calibration Status | Derivation / Source | Uncertainty & Impact |
|---|---|---|---|---|
| **Horizontal FOV** | $70.0^\circ$ | Provisional Nominal | Pinhole assumption matching $876 \times 568$ aspect ratio | $\pm 5^\circ$ uncertainty affects horizon focal scale $f_x, f_y$ by $\sim 8\%$. |
| **Focal Length ($f_x, f_y$)** | $625.534\text{ px}$ | Provisional Nominal | $f = \frac{W/2}{\tan(\text{HFOV}/2)}$ | Governs perspective convergence of ground lane boundaries. |
| **Principal Point ($c_x, c_y$)** | $(438.0, 284.0)$ | Provisional Nominal | Geometric canvas center (no optical decentering modeled) | Decentering could shift optical horizon by $\pm 5\text{ px}$. |
| **Camera Yaw Offsets** | $0^\circ, \pm 60^\circ, \pm 120^\circ, 180^\circ$ | Provisional Nominal | Standard 6-camera surround hexagonal rig | Determines boundary transition between front/side views. |
| **Camera Pitch & Roll** | $0.0^\circ, 0.0^\circ$ | Provisional Nominal | Flat ground plane baseline | Vehicle dynamic pitch under braking causes transient $\pm 2^\circ$ shift. |
| **Camera Height ($Z$)** | $1.40\text{ m}$ | Provisional Nominal | Vehicle roof/windshield mounting height | Controls distance at which road enters vertical FOV ($X \approx 4.58\text{ m}$). |
| **Near Clipping Plane ($z_c$)** | $0.50\text{ m}$ | Architectural | Near-plane guard preventing division-by-zero behind lens | Safely clips trajectories passing behind the camera. |

## 2. Line Clipping Implementation (3D Near-Plane + 2D Canvas Clipping)

To eliminate spurious cross-screen streak artifacts caused by waypoints straddling the camera plane, a two-stage clipping pipeline is enforced:

1. **3D Near-Plane Clipping (`clip_segment_3d_near_plane`):**
   - For segment $(P_1, P_2)$ in camera frame with $z_1 < 0.5\text{ m} \le z_2$:
     $$\alpha = \frac{0.5 - z_1}{z_2 - z_1} \in [0, 1],\quad P_{\text{clip}} = P_1 + \alpha(P_2 - P_1)$$
   - Segments with both $z_1, z_2 < 0.5\text{ m}$ are discarded immediately before projection.
2. **2D Canvas Clipping (`clip_segment_2d_canvas`):**
   - Implements Liang-Barsky parametric line clipping against canvas boundaries $[0, W-1] \times [0, H-1]$.
   - Replaces unbounded while-loops with strictly bounded $O(1)$ parameter interval intersections $[u_1, u_2]$.

## 3. Disambiguation: Option B.1 vs Option B.2 Semantics

Two distinct temporal conditioning representations have been implemented and compared side-by-side:

### Option B.1: Route-Initial-Frame Temporal Masks (`render_initial_frame_temporal_masks`)
- **Semantics:** Evaluates the entire future path from the stationary vantage point of the camera at $t = 0.0\text{s}$.
- **Appearance:** The camera remains static; the active beacon moves forward across the road surface toward the horizon as $t$ advances.
- **Use Case:** Initial-frame route guide for diffusion models where the latent representation encodes the stationary starting view.

### Option B.2: Future-Camera-Frame Spatial Masks (`render_future_camera_spatial_masks`)
- **Semantics:** Ego-motion aligned future camera projection. Reconstructs the vehicle pose $P_{\text{ego}}(t) = [X_t, Y_t]$ and yaw heading $\psi_t$ at every future timestep $t \in \{1..25\}$.
- **Appearance:** The camera translates and yaws synchronously with the moving vehicle. At future frame $t$, the camera observes the upcoming road corridor directly ahead of the car.
- **Use Case:** Physically aligned control maps for multi-view video diffusion where each conditioning frame corresponds to the instantaneous viewpoint of that video frame.

## 4. Scenario Audit Matrix & Camera Visibility Analysis

### Category A: Genuine Multi-Camera RGB Image Overlays

| Scenario ID | Maneuver Archetype | Instruction | $v_0$ (m/s) | Motion Bucket | Front Visibility | Front-Left | Rear / Sides | Contact Sheet | Temporal Comparison |
|---|---|---|---|---|---|---|---|---|---|
| `0000900` | Straight Highway (Snow / Wintry Mix) | `drive straight on` | 27.54 | 255 | 24/25 (96%) | 0/25 (0%) | RL:0 R:0 RR:0 FR:0 | [PNG](0000900_contact_sheet.png) | [B.1 vs B.2](0000900_temporal_comparison.png) |
| `0000901` | High-Speed Lane Change ('use right lane') | `use right lane` | 64.61 | 255 | 25/25 (100%) | 0/25 (0%) | RL:0 R:0 RR:0 FR:0 | [PNG](0000901_contact_sheet.png) | [B.1 vs B.2](0000901_temporal_comparison.png) |
| `0000902` | Stationary Launch & Right Turn | `turn right` | 0.57 | 16 | 11/25 (44%) | 0/25 (0%) | RL:0 R:0 RR:0 FR:0 | [PNG](0000902_contact_sheet.png) | [B.1 vs B.2](0000902_temporal_comparison.png) |
| `0000911` | Moderate Left Turn ('turn left') | `turn left` | 2.40 | 38 | 8/25 (32%) | 0/25 (0%) | RL:0 R:0 RR:0 FR:0 | [PNG](0000911_contact_sheet.png) | [B.1 vs B.2](0000911_temporal_comparison.png) |
| `0000919` | Curved Road & Overtake ('overtake truck on right') | `overtake truck driving on the right` | 20.53 | 255 | 23/25 (92%) | 0/25 (0%) | RL:0 R:0 RR:0 FR:0 | [PNG](0000919_contact_sheet.png) | [B.1 vs B.2](0000919_temporal_comparison.png) |
| `0000984` | Extreme 180° U-Turn | `u-turn` | 8.20 | 108 | 0/25 (0%) | 17/25 (68%) | RL:1 R:0 RR:0 FR:0 | [PNG](0000984_contact_sheet.png) | [B.1 vs B.2](0000984_temporal_comparison.png) |

### Category B: Synthetic Perspective Geometric Benchmarks

| Scenario ID | Maneuver Archetype | Instruction | $v_0$ (m/s) | Motion Bucket | Front Visibility | Front-Left | Rear / Sides | Contact Sheet | Temporal Comparison |
|---|---|---|---|---|---|---|---|---|---|
| `0000908` | Intersection Right Turn (High Speed) | `turn right` | 20.31 | 253 | 23/25 (92%) | 0/25 (0%) | RL:0 R:0 RR:0 FR:0 | [PNG](0000908_contact_sheet.png) | [B.1 vs B.2](0000908_temporal_comparison.png) |
| `0000970` | Sharp 90° Left Turn (Enters front_left) | `turn left` | 0.15 | 11 | 0/25 (0%) | 13/25 (52%) | RL:0 R:0 RR:0 FR:0 | [PNG](0000970_contact_sheet.png) | [B.1 vs B.2](0000970_temporal_comparison.png) |
| `0000949` | Stationary-to-Moving Launch | `drive straight on` | 0.12 | 11 | 10/25 (40%) | 0/25 (0%) | RL:0 R:0 RR:0 FR:1 | [PNG](0000949_contact_sheet.png) | [B.1 vs B.2](0000949_temporal_comparison.png) |
| `0000909` | Curved Road (Snow / Wintry Mix) | `drive straight on` | 21.75 | 255 | 23/25 (92%) | 0/25 (0%) | RL:0 R:0 RR:0 FR:0 | [PNG](0000909_contact_sheet.png) | [B.1 vs B.2](0000909_temporal_comparison.png) |

## 5. Physical & Optical Causes for Camera Occlusion

A critical question in multi-camera verification is why certain cameras have 0 visible projected waypoints:

1. **Rear Cameras (`rear`, `rear_left`, `rear_right`):**
   - In normal forward driving ($v > 0$), waypoints extend forward into the future ($X > 0$).
   - The rear camera faces $180^\circ$ backward. Forward points have negative optical depth $z_c < 0$ (behind camera lens), and are safely clipped by the 3D near-plane guard.

2. **Front Camera in Sharp Left Turn `0000970`:**
   - At $t=0.2\text{s}$, the car executes an immediate $90^\circ$ left turn ($Y \gg X$).
   - The trajectory immediately exits the front camera's $70^\circ$ horizontal FOV ($35^\circ$ half-cone) and sweeps directly into `front_left` (yaw $+60^\circ$), resulting in **13 visible points in `front_left`** and **0 in `front`**.

3. **Bumper Blind Zone in Stationary Starts (`0000902`, `0000949`):**
   - Because the camera is mounted $1.5\text{m}$ forward and $1.4\text{m}$ high, ground road points closer than $X \approx 4.58\text{m}$ project below the bottom edge of the image ($v > 568$).
   - In stationary launches, initial waypoints ($t < 1.0\text{s}$) lie under the bumper ($X < 2.0\text{m}$), and only enter the image canvas as the vehicle accelerates past $4.6\text{m}$.

## 6. Training Dataset Pairing & Future Video Target Audit

- **Training Donors Loaded:** Exactly 500 records (`0000400`–`0000899`).
- **Data Leakage Check:** 0 validation scenarios ($0000900 \le \text{ID} \le 0000999$) present in training split.
- **Future Video Target Audit:** The official KITScenes challenge hosts the full 30+ GB training video parquet files on Hugging Face. On this local environment, the 30+ GB remote video payload has not been downloaded, so 0/500 future video targets are present on disk.
- **Strict Assertion Safety Gate:** `SpatialConditionedWorldModelDataset(require_future_video=True)` strictly raises `FileNotFoundError` when future videos are missing, preventing any silent training on placeholder zero images.
