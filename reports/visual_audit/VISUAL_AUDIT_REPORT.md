# 6-Camera Multi-View Spatial Conditioning Visual Audit

### Autonomous Driving World Modeling Geometry & Spatial Representation Verification

*Generated on: 2026-11-15 09:09:10 UTC*

## 1. Executive Summary & Coordinate System Specification

This audit validates the exact coordinate transformation pipeline from planned 3D future waypoints into 2D pixel-space conditioning tensors across the full 6-camera surround rig ($876 \times 568$, $70^\circ$ HFOV):

- **Ego Vehicle Body Frame:** $X$: Forward, $Y$: Left, $Z$: Up (flat ground at $Z=0.0\text{m}$).
- **Camera Optical Frame:** $x$: Right, $y$: Down, $z$: Forward along optical axis.
- **Extrinsics Matrix:** $R_{\text{cam}\leftarrow\text{ego}} = \begin{bmatrix} \sin\theta & -\cos\theta & 0 \\ 0 & 0 & -1 \\ \cos\theta & \sin\theta & 0 \end{bmatrix}$, with $\det(R) = +1.000000$ (orthogonal rotation).
- **Pinhole Intrinsics:** $W=876, H=568, f_x = f_y \approx 625.534\text{ px}, c_x = 438.0, c_y = 284.0$.
- **Near-Plane Clipping:** $z_c > 0.5\text{m}$ (prevents division-by-zero behind lens).

## 2. Canonical Scenario Audit Matrix

| Scenario ID | Maneuver Archetype | Instruction | $v_0$ (m/s) | Motion Bucket | Visible Cameras & Counts | Genuine RGB? |
|---|---|---|---|---|---|---|
| `0000900` | Straight Highway (Snow / Wintry Mix) | `drive straight on` | 27.54 | 255 | **front** (24) | Yes (6 Cams) |
| `0000901` | Lane Change ('use right lane') | `use right lane` | 64.61 | 255 | **front** (25) | Yes (6 Cams) |
| `0000902` | Stationary Launch & Right Turn | `turn right` | 0.57 | 16 | **front** (11) | Yes (6 Cams) |
| `0000911` | Left Turn ('turn left') | `turn left` | 2.40 | 38 | **front** (8) | Yes (6 Cams) |
| `0000919` | Curved Road & Overtake | `overtake truck driving on the right` | 20.53 | 255 | **front** (23) | Yes (6 Cams) |
| `0000984` | Extreme 180° U-Turn | `u-turn` | 8.20 | 108 | **front_left** (17), **rear_left** (1) | Yes (6 Cams) |
| `0000908` | Right Turn (Intersection) | `turn right` | 20.31 | 253 | **front** (23) | Synthetic Perspective |
| `0000970` | Sharp Left Turn (Enters front_left) | `turn left` | 0.15 | 11 | **front_left** (13) | Synthetic Perspective |
| `0000949` | Stationary-to-Moving Launch | `drive straight on` | 0.12 | 11 | **front** (10), **front_right** (1) | Synthetic Perspective |
| `0000909` | Curved Road (Snow / Wintry Mix) | `drive straight on` | 21.75 | 255 | **front** (23) | Synthetic Perspective |

## 3. Spatial Representation Contracts

Two rigorous spatial conditioning representations are implemented and verified:

### Option A: Static Route Corridor Mask (`render_static_route_mask`)
- **Tensor Contract:** `[B, 1, 568, 876]` (single camera) or `[B, 6, 1, 568, 876]` (6-camera rig).
- **Representation:** Single-channel float32 map $\in [0.0, 1.0]$. Projects the full 5-second future corridor onto the road surface with depth-attenuated ribbon width and intensity falloff.
- **Sky Region Invariant:** Sky ($v < 284\text{ px}$) strictly receives zero mask intensity.

### Option B: Time-Indexed Spatial Conditioning Maps (`render_time_indexed_spatial_masks`)
- **Tensor Contract:** `[B, 25, 1, 568, 876]` (single camera) or `[B, 6, 25, 1, 568, 876]` (6-camera rig).
- **Representation:** 25-frame sequence matching future video timesteps $\tau_t = (t+1) \times 0.2\text{s}$. Each frame $t$ features an accentuated perspective beacon at $P(t)$, past trajectory trail $(0..t)$, and subtle future guide ribbon.
- **Temporal Motion Invariant:** For forward driving, the active beacon moves monotonically toward the horizon ($v$ decreases as $t$ increases).

## 4. Key Geometric Findings & Validation

1. **Turn Camera Transitions:**
   - Sharp left turn `0000970` has **13 points visible in `front_left`** and 0 in `front`. The trajectory completely exits the front camera's $70^\circ$ FOV and enters the left-facing camera.
   - U-turn `0000984` sweeps through **17 points in `front_left`** and transitions into **`rear_left`** (1 point), demonstrating cross-camera spatial continuity across the surround rig.
2. **Bumper Blind Zone Physics:**
   - Because camera mounts are at $Z=1.4\text{m}$ and $X=1.5\text{m}$, road points closer than $X \approx 4.6\text{m}$ fall below the camera's lower image boundary ($v > 568$).
   - In stationary launch scenarios (`0000902`, `0000905`, `0000949`), the vehicle starts from rest; early waypoints ($t < 1.0\text{s}$) lie under the bumper and smoothly emerge into view as speed increases.
3. **Optical Horizon Invariance:**
   - For all scenarios and all 6 cameras, road waypoints are strictly bounded by $v > c_y = 284\text{ px}$, confirming that ground elevation ($Z=0$) never projects into the sky hemisphere.
