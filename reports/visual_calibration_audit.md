# KITScenes Official Camera Calibration Visual Audit (60 Views)

**Calibration Provenance:** KITScenes-LongTail Dataset (KIT-MRT / Wagner et al., arXiv:2603.23607, 2026)
**Calibration Config:** [`configs/kitscenes_camera_calibration.json`](file:///Users/wageshsharma/.gemini/antigravity/scratch/kitscenes-world-model/configs/kitscenes_camera_calibration.json)
**Reference Coordinate Frame:** Primary Roof Sensor Frame (t=0 LiDAR scanning center; +X forward, +Y left, +Z up; nominal road plane Z ≈ -1.60m)
**Scenarios Audited:** 10 scenarios (0000900, 0000901, 0000902, 0000903, 0000904, 0000905, 0000906, 0000911, 0000919, 0000984)
**Total Multi-Camera Views:** 60 views
**Front Camera Waypoint Field-of-View Coverage:** 75.60%

> [!IMPORTANT]
> **Metric Distinction:** The waypoint visibility rate measures the percentage of 3D future waypoints that project inside the camera's image canvas boundaries `[0, W-1] x [0, H-1]` with positive optical depth $z_c > 0.5\text{ m}$. It quantifies geometric camera field-of-view coverage, NOT a photogrammetric 2D-3D reprojection residual. Distant horizon points (> 60m) or sharp lateral turns that exit the front canvas enter the side surround cameras (`front_left`/`front_right`).

## Camera Projection Verification Summary

| Camera Name | Nominal Yaw | Calibrated Optical Center (cx, cy) | Total Visible Waypoints Across 10 Scenarios |
|---|---|---|---|
| `front` | +0.0° | (443.27, 284.75) | 189 / 250 waypoints |
| `front_left` | +60.0° | (443.02, 282.75) | 19 / 250 waypoints |
| `front_right` | -60.0° | (439.26, 284.50) | 1 / 250 waypoints |
| `rear` | +180.0° | (443.27, 283.75) | 0 / 250 waypoints |
| `rear_left` | +120.0° | (444.28, 284.00) | 3 / 250 waypoints |
| `rear_right` | -120.0° | (441.01, 287.00) | 0 / 250 waypoints |

## Scenario Breakdown & Visual Overlays

### Scenario `0000900`
- **Instruction:** *"drive straight on"*
- **Scenario Type:** `6 snow & wintry mix`
- **6-Camera Contact Sheet:** [`0000900_6cam_calibration_contact_sheet.jpg`](file:///Users/wageshsharma/.gemini/antigravity/scratch/kitscenes-world-model/reports/visual_calibration_audit/0000900_6cam_calibration_contact_sheet.jpg)

| Camera | Visible / Total | Depth Range (m) | Canvas Range [u_min, u_max] x [v_min, v_max] | Ribbon Area (px) |
|---|---|---|---|---|
| `front` | 25 / 25 | [2.4, 57.3] | [418.5, 448.5] x [298.6, 553.8] | 3,733 |
| `front_left` | 0 / 25 | None | None x None | 0 |
| `front_right` | 0 / 25 | None | None x None | 0 |
| `rear` | 0 / 25 | None | None x None | 0 |
| `rear_left` | 0 / 25 | None | None x None | 0 |
| `rear_right` | 0 / 25 | None | None x None | 0 |

### Scenario `0000901`
- **Instruction:** *"use right lane"*
- **Scenario Type:** `3 overtake or lane change`
- **6-Camera Contact Sheet:** [`0000901_6cam_calibration_contact_sheet.jpg`](file:///Users/wageshsharma/.gemini/antigravity/scratch/kitscenes-world-model/reports/visual_calibration_audit/0000901_6cam_calibration_contact_sheet.jpg)

| Camera | Visible / Total | Depth Range (m) | Canvas Range [u_min, u_max] x [v_min, v_max] | Ribbon Area (px) |
|---|---|---|---|---|
| `front` | 25 / 25 | [6.1, 154.1] | [436.6, 448.9] x [291.4, 393.1] | 911 |
| `front_left` | 0 / 25 | None | None x None | 0 |
| `front_right` | 0 / 25 | None | None x None | 0 |
| `rear` | 0 / 25 | None | None x None | 0 |
| `rear_left` | 0 / 25 | None | None x None | 0 |
| `rear_right` | 0 / 25 | None | None x None | 0 |

### Scenario `0000902`
- **Instruction:** *"turn right"*
- **Scenario Type:** `4 intersection`
- **6-Camera Contact Sheet:** [`0000902_6cam_calibration_contact_sheet.jpg`](file:///Users/wageshsharma/.gemini/antigravity/scratch/kitscenes-world-model/reports/visual_calibration_audit/0000902_6cam_calibration_contact_sheet.jpg)

| Camera | Visible / Total | Depth Range (m) | Canvas Range [u_min, u_max] x [v_min, v_max] | Ribbon Area (px) |
|---|---|---|---|---|
| `front` | 15 / 25 | [2.3, 14.6] | [436.2, 439.0] x [331.8, 566.2] | 3,758 |
| `front_left` | 0 / 25 | None | None x None | 0 |
| `front_right` | 0 / 25 | None | None x None | 0 |
| `rear` | 0 / 25 | None | None x None | 0 |
| `rear_left` | 0 / 25 | None | None x None | 0 |
| `rear_right` | 0 / 25 | None | None x None | 0 |

### Scenario `0000903`
- **Instruction:** *"drive straight on"*
- **Scenario Type:** `1 heavy rain`
- **6-Camera Contact Sheet:** [`0000903_6cam_calibration_contact_sheet.jpg`](file:///Users/wageshsharma/.gemini/antigravity/scratch/kitscenes-world-model/reports/visual_calibration_audit/0000903_6cam_calibration_contact_sheet.jpg)

| Camera | Visible / Total | Depth Range (m) | Canvas Range [u_min, u_max] x [v_min, v_max] | Ribbon Area (px) |
|---|---|---|---|---|
| `front` | 25 / 25 | [3.8, 100.0] | [395.8, 437.2] x [293.4, 459.7] | 1,879 |
| `front_left` | 0 / 25 | None | None x None | 0 |
| `front_right` | 0 / 25 | None | None x None | 0 |
| `rear` | 0 / 25 | None | None x None | 0 |
| `rear_left` | 0 / 25 | None | None x None | 0 |
| `rear_right` | 0 / 25 | None | None x None | 0 |

### Scenario `0000904`
- **Instruction:** *"drive straight on"*
- **Scenario Type:** `2 construction zone`
- **6-Camera Contact Sheet:** [`0000904_6cam_calibration_contact_sheet.jpg`](file:///Users/wageshsharma/.gemini/antigravity/scratch/kitscenes-world-model/reports/visual_calibration_audit/0000904_6cam_calibration_contact_sheet.jpg)

| Camera | Visible / Total | Depth Range (m) | Canvas Range [u_min, u_max] x [v_min, v_max] | Ribbon Area (px) |
|---|---|---|---|---|
| `front` | 24 / 25 | [4.4, 55.1] | [430.9, 437.3] x [299.0, 434.2] | 3,783 |
| `front_left` | 0 / 25 | None | None x None | 0 |
| `front_right` | 0 / 25 | None | None x None | 0 |
| `rear` | 0 / 25 | None | None x None | 0 |
| `rear_left` | 0 / 25 | None | None x None | 0 |
| `rear_right` | 0 / 25 | None | None x None | 0 |

### Scenario `0000905`
- **Instruction:** *"turn right"*
- **Scenario Type:** `0 specifically selected`
- **6-Camera Contact Sheet:** [`0000905_6cam_calibration_contact_sheet.jpg`](file:///Users/wageshsharma/.gemini/antigravity/scratch/kitscenes-world-model/reports/visual_calibration_audit/0000905_6cam_calibration_contact_sheet.jpg)

| Camera | Visible / Total | Depth Range (m) | Canvas Range [u_min, u_max] x [v_min, v_max] | Ribbon Area (px) |
|---|---|---|---|---|
| `front` | 5 / 25 | [2.4, 4.6] | [536.2, 603.0] x [429.2, 556.3] | 2,953 |
| `front_left` | 0 / 25 | None | None x None | 0 |
| `front_right` | 1 / 25 | [3.6, 3.6] | [12.2, 12.2] x [465.4, 465.4] | 396 |
| `rear` | 0 / 25 | None | None x None | 0 |
| `rear_left` | 0 / 25 | None | None x None | 0 |
| `rear_right` | 0 / 25 | None | None x None | 0 |

### Scenario `0000906`
- **Instruction:** *"use left lane"*
- **Scenario Type:** `5 nighttime`
- **6-Camera Contact Sheet:** [`0000906_6cam_calibration_contact_sheet.jpg`](file:///Users/wageshsharma/.gemini/antigravity/scratch/kitscenes-world-model/reports/visual_calibration_audit/0000906_6cam_calibration_contact_sheet.jpg)

| Camera | Visible / Total | Depth Range (m) | Canvas Range [u_min, u_max] x [v_min, v_max] | Ribbon Area (px) |
|---|---|---|---|---|
| `front` | 25 / 25 | [6.1, 151.8] | [445.6, 449.0] x [291.5, 393.5] | 913 |
| `front_left` | 0 / 25 | None | None x None | 0 |
| `front_right` | 0 / 25 | None | None x None | 0 |
| `rear` | 0 / 25 | None | None x None | 0 |
| `rear_left` | 0 / 25 | None | None x None | 0 |
| `rear_right` | 0 / 25 | None | None x None | 0 |

### Scenario `0000911`
- **Instruction:** *"turn left"*
- **Scenario Type:** `0 specifically selected`
- **6-Camera Contact Sheet:** [`0000911_6cam_calibration_contact_sheet.jpg`](file:///Users/wageshsharma/.gemini/antigravity/scratch/kitscenes-world-model/reports/visual_calibration_audit/0000911_6cam_calibration_contact_sheet.jpg)

| Camera | Visible / Total | Depth Range (m) | Canvas Range [u_min, u_max] x [v_min, v_max] | Ribbon Area (px) |
|---|---|---|---|---|
| `front` | 12 / 25 | [2.3, 10.5] | [424.5, 561.7] x [350.0, 567.2] | 4,253 |
| `front_left` | 0 / 25 | None | None x None | 0 |
| `front_right` | 0 / 25 | None | None x None | 0 |
| `rear` | 0 / 25 | None | None x None | 0 |
| `rear_left` | 0 / 25 | None | None x None | 0 |
| `rear_right` | 0 / 25 | None | None x None | 0 |

### Scenario `0000919`
- **Instruction:** *"overtake truck driving on the right"*
- **Scenario Type:** `3 overtake or lane change`
- **6-Camera Contact Sheet:** [`0000919_6cam_calibration_contact_sheet.jpg`](file:///Users/wageshsharma/.gemini/antigravity/scratch/kitscenes-world-model/reports/visual_calibration_audit/0000919_6cam_calibration_contact_sheet.jpg)

| Camera | Visible / Total | Depth Range (m) | Canvas Range [u_min, u_max] x [v_min, v_max] | Ribbon Area (px) |
|---|---|---|---|---|
| `front` | 24 / 25 | [4.1, 57.8] | [433.7, 492.7] x [298.7, 447.3] | 4,155 |
| `front_left` | 0 / 25 | None | None x None | 0 |
| `front_right` | 0 / 25 | None | None x None | 0 |
| `rear` | 0 / 25 | None | None x None | 0 |
| `rear_left` | 0 / 25 | None | None x None | 0 |
| `rear_right` | 0 / 25 | None | None x None | 0 |

### Scenario `0000984`
- **Instruction:** *"u-turn"*
- **Scenario Type:** `4 intersection`
- **6-Camera Contact Sheet:** [`0000984_6cam_calibration_contact_sheet.jpg`](file:///Users/wageshsharma/.gemini/antigravity/scratch/kitscenes-world-model/reports/visual_calibration_audit/0000984_6cam_calibration_contact_sheet.jpg)

| Camera | Visible / Total | Depth Range (m) | Canvas Range [u_min, u_max] x [v_min, v_max] | Ribbon Area (px) |
|---|---|---|---|---|
| `front` | 9 / 25 | [2.5, 5.7] | [39.0, 335.9] x [399.9, 546.7] | 6,680 |
| `front_left` | 19 / 25 | [3.5, 11.3] | [185.4, 827.8] x [336.1, 469.3] | 9,041 |
| `front_right` | 0 / 25 | None | None x None | 0 |
| `rear` | 0 / 25 | None | None x None | 0 |
| `rear_left` | 3 / 25 | [9.3, 10.8] | [736.4, 836.4] x [339.8, 349.6] | 1,587 |
| `rear_right` | 0 / 25 | None | None x None | 0 |
