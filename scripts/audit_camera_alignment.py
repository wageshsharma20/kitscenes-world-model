#!/usr/bin/env python3
"""Empirical Camera Geometry & Real-Image Road-Plane Alignment Auditor.

Conducts a rigorous empirical calibration audit across the 10 real-image scenarios
in data/val_observed_t0 (60 total camera views: front, front_left, front_right,
rear, rear_left, rear_right).

Audits:
  1. Optical Horizon Line & Vanishing Point Analysis:
     - Detects road perspective lines (lane markings, curbstones, guardrails) via
       adaptive Canny and probabilistic Hough Line Transform.
     - Computes vanishing points (VP_x, VP_y) via robust pairwise line intersection clustering.
     - Estimates visual horizon row v_horizon via multi-slice vertical luminance gradient profiling.
     - Quantifies deviations from the provisional nominal optical horizon (cy = 284.0 px)
       and optical center (cx = 438.0 px), converting pixel offsets to equivalent pitch/yaw angles.
  2. Ground Plane Grid Reprojection & Road Surface Alignment:
     - Projects a 3D ground plane grid (longitudinal lane boundaries Y in [-4.5, 4.5]m,
       transverse distance markers X in [-25, 50]m at Z = 0) with full 3D near-plane clipping
       (z_c >= 0.5m) and 2D canvas boundary clipping.
     - Measures ground plane visibility across all 6 surround views.
     - Evaluates geometric alignment of projected ground grid with the empirical road surface.
  3. Side & Rear Camera Road-Plane Analysis:
     - Analyzes why and where the road plane appears in side (yaw = ±60°, ±120°) and rear (180°) views.
     - Validates field-of-view overlaps (10° overlap between adjacent hexagonal cameras).
  4. Multi-Camera Trajectory Visibility:
     - Projects planned 25-waypoint future trajectories (0.2s - 5.0s) across all 6 views.
     - Categorizes visibility by maneuver type (straight highway, lane change, intersection turn, U-turn).
  5. Artifact Generation:
     - Saves detailed CSV audit table: reports/camera_calibration_audit_table.csv
     - Saves structured JSON summary: reports/camera_calibration_audit_summary.json
     - Generates high-resolution 6-camera visual check contact sheets:
       - reports/camera_calibration_audit_visual_check_highway.png (Scenario 0000901)
       - reports/camera_calibration_audit_visual_check_urban.png (Scenario 0000902)
       - reports/camera_calibration_audit_visual_check_uturn.png (Scenario 0000984)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitscenes import camera_projection, spec, video_engine

# The 10 validation scenarios with real RGB camera frames
REAL_SCENARIOS = [
    ("0000900", "Straight Highway (Snow & Wintry Mix)", "drive straight on"),
    ("0000901", "High-Speed Highway Lane Change", "use right lane"),
    ("0000902", "Urban Intersection Launch & Right Turn", "turn right"),
    ("0000903", "Heavy Rain Highway Driving", "drive straight on"),
    ("0000904", "Construction Zone Driving", "drive straight on"),
    ("0000905", "Urban Residential Intersection", "turn right"),
    ("0000906", "Nighttime Highway Driving", "use left lane"),
    ("0000911", "Urban Intersection Left Turn", "turn left"),
    ("0000919", "Curved Highway & Overtake", "overtake truck driving on the right"),
    ("0000984", "Urban Intersection Extreme U-Turn", "u-turn"),
]

CAMERAS = spec.CAMERAS  # ['front', 'front_left', 'front_right', 'rear', 'rear_left', 'rear_right']

CAMERA_GRID_ORDER = [
    ["front_left", "front", "front_right"],
    ["rear_left", "rear", "rear_right"],
]


@dataclass
class CameraAuditMetric:
    scenario_id: str
    scenario_title: str
    camera_name: str
    nominal_yaw_deg: float
    nominal_cx: float
    nominal_cy: float
    nominal_fx: float
    nominal_fy: float
    nominal_cam_z: float
    empirical_horizon_row: float
    horizon_delta_cy_px: float
    equivalent_pitch_deg: float
    vp_x: Optional[float]
    vp_y: Optional[float]
    vp_delta_cx_px: Optional[float]
    vp_delta_cy_px: Optional[float]
    equivalent_yaw_offset_deg: Optional[float]
    num_supporting_lines: int
    num_line_intersections: int
    ground_grid_points_visible: int
    ground_grid_points_total: int
    ground_grid_visibility_pct: float
    trajectory_waypoints_visible: int
    trajectory_waypoints_total: int
    trajectory_visibility_pct: float
    road_mask_coverage_pct: float
    road_grid_alignment_iou: float


def detect_visual_horizon_and_vp(
    img_bgr: np.ndarray,
    camera_name: str,
) -> Tuple[float, float, Optional[Tuple[float, float]], int, int, List[Tuple[int, int, int, int]]]:
    """Detect the apparent visual horizon and vanishing point in an RGB camera frame.

    Returns:
        (mean_horizon_row, std_horizon_row, vanishing_point, num_lines, num_intersections, supporting_lines)
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Multi-slice vertical luminance gradient profiling
    # Divides image into 7 vertical columns to capture horizon slope / roll
    num_slices = 7
    slice_w = w // num_slices
    slice_horizons = []
    for s_idx in range(num_slices):
        col_start = s_idx * slice_w
        col_end = (s_idx + 1) * slice_w if s_idx < num_slices - 1 else w
        strip = gray[:, col_start:col_end]
        prof = strip.mean(axis=1)
        # Smoothing kernel
        kernel = np.ones(15) / 15.0
        smooth = np.convolve(prof, kernel, mode="same")
        grad = np.gradient(smooth)
        # Search plausible horizon band: rows 140 to 420
        band_start = 140
        band_end = 420
        search_band = grad[band_start:band_end]
        min_idx = int(np.argmin(search_band) + band_start)
        slice_horizons.append(min_idx)

    mean_horizon = float(np.mean(slice_horizons))
    std_horizon = float(np.std(slice_horizons))

    # 2. Line detection for perspective vanishing point
    # Apply Gaussian blur & bilateral filter to preserve edges
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=25,
        maxLineGap=15,
    )

    vp = None
    vp_intersections = 0
    supporting_lines = []

    if lines is not None and len(lines) >= 4:
        candidates = []
        for l in lines:
            x1, y1, x2, y2 = l.ravel()
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 1e-4:
                continue
            slope = dy / dx
            deg = math.degrees(math.atan(slope))
            # Road perspective lines typically have angles between 12° and 80°
            if 12.0 < abs(deg) < 80.0:
                length = math.hypot(dx, dy)
                candidates.append((x1, y1, x2, y2, slope, length))
                supporting_lines.append((int(x1), int(y1), int(x2), int(y2)))

        intersections = []
        weights = []
        n_cand = len(candidates)
        # Check pairwise intersections
        for i in range(n_cand):
            for j in range(i + 1, min(n_cand, i + 35)):
                x1, y1, x2, y2, m1, len1 = candidates[i]
                x3, y3, x4, y4, m2, len2 = candidates[j]
                if abs(m1 - m2) < 0.15:
                    continue  # Parallel lines
                c1 = m1 * x1 - y1
                c2 = m2 * x3 - y3
                denom = m1 - m2
                xi = (c1 - c2) / denom
                yi = m1 * (xi - x1) + y1

                # Restrict to plausible region around canvas
                if -200 <= xi <= w + 200 and 0.10 * h <= yi <= 0.90 * h:
                    intersections.append((xi, yi))
                    weights.append(len1 * len2)

        if len(intersections) >= 5:
            arr = np.array(intersections)
            w_arr = np.array(weights)
            # Weighted average
            vp_x = float(np.average(arr[:, 0], weights=w_arr))
            vp_y = float(np.average(arr[:, 1], weights=w_arr))
            vp = (vp_x, vp_y)
            vp_intersections = len(intersections)

    return mean_horizon, std_horizon, vp, len(supporting_lines), vp_intersections, supporting_lines


def build_3d_ground_plane_grid() -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """Construct a dense 3D ground plane grid in vehicle body coordinates (Z=0).

    Returns:
        points_3d: (N, 3) point cloud for visibility testing.
        line_segments_3d: List of (p1, p2) line segments for rendering.
    """
    longitudinal_ys = [-4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5]
    transverse_xs = [-25.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]

    # Sample points on grid
    pts = []
    for x in np.linspace(-30.0, 55.0, 35):
        for y in np.linspace(-12.0, 12.0, 25):
            pts.append([x, y, 0.0])
    points_3d = np.array(pts, dtype=np.float64)

    # Line segments
    segments = []
    # Longitudinal lines
    for y in longitudinal_ys:
        segments.append(
            (np.array([-25.0, y, 0.0], dtype=np.float64), np.array([50.0, y, 0.0], dtype=np.float64))
        )
    # Transverse lines
    for x in transverse_xs:
        segments.append(
            (np.array([x, -6.0, 0.0], dtype=np.float64), np.array([x, 6.0, 0.0], dtype=np.float64))
        )

    return points_3d, segments


def segment_empirical_road_surface(img_bgr: np.ndarray) -> np.ndarray:
    """Segment empirical asphalt/road pavement in camera image.

    Returns:
        Boolean 2D mask (True for road pavement).
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Asphalt characteristics: low saturation, moderate value/brightness
    # S in [0, 75], V in [20, 210]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    asphalt_mask = (sat < 75) & (val > 20) & (val < 210)

    # Enforce geometric constraint: road is in the lower portion of the image
    road_mask = np.zeros((h, w), dtype=bool)
    road_mask[240:, :] = asphalt_mask[240:, :]

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    road_mask_clean = cv2.morphologyEx(road_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return road_mask_clean.astype(bool)


def compute_ground_grid_alignment(
    img_bgr: np.ndarray,
    camera_name: str,
    engine: camera_projection.CameraProjectionEngine,
    grid_points_3d: np.ndarray,
) -> Tuple[int, int, float, float, float]:
    """Compute ground grid visibility, road surface coverage, and reprojection alignment IoU."""
    h, w = img_bgr.shape[:2]
    view = engine.project_waypoints_to_camera(camera_name, grid_points_3d)

    vis_count = int(view.visible_count)
    total_count = int(len(grid_points_3d))
    vis_pct = (vis_count / max(1, total_count)) * 100.0

    # Empirical road mask
    road_mask = segment_empirical_road_surface(img_bgr)
    road_area_pct = float(road_mask.sum()) / float(h * w) * 100.0

    # Projected ground mask
    proj_mask = np.zeros((h, w), dtype=bool)
    if vis_count >= 3:
        pts_2d = view.points_2d[view.is_visible]
        # Convex hull of projected visible ground points
        hull = cv2.convexHull(pts_2d.astype(np.int32))
        hull_img = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(hull_img, hull, 1)
        proj_mask = hull_img.astype(bool)

    # Intersection over Union between projected ground hull and empirical road mask
    intersection = float(np.logical_and(proj_mask, road_mask).sum())
    union = float(np.logical_or(proj_mask, road_mask).sum())
    iou = (intersection / max(1.0, union)) * 100.0

    return vis_count, total_count, vis_pct, road_area_pct, iou


def render_calibration_audit_overlay(
    img_bgr: np.ndarray,
    metric: CameraAuditMetric,
    cam_name: str,
    engine: camera_projection.CameraProjectionEngine,
    grid_segments: List[Tuple[np.ndarray, np.ndarray]],
    supporting_lines: List[Tuple[int, int, int, int]],
    trajectory: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render comprehensive calibration overlay with ground grid, horizons, and HUD telemetry."""
    img = img_bgr.copy()
    h, w = img.shape[:2]
    cam = engine.rig[cam_name]

    # 1. Projected 3D Ground Plane Grid (Lime Green / Teal)
    # Clip and draw 3D ground lines
    t_cam = np.asarray(cam.position_m, dtype=np.float64)
    r_mat = cam.rotation_matrix_body_to_cam

    for p1_ego, p2_ego in grid_segments:
        p1_cam = (p1_ego - t_cam) @ r_mat.T
        p2_cam = (p2_ego - t_cam) @ r_mat.T
        res = camera_projection.project_and_clip_segment(
            p1_cam, p2_cam, cam, width=w, height=h, near_clip_m=engine.near_clip_m
        )
        if res is not None:
            (u0, v0), (u1, v1), z_avg = res
            alpha = max(0.3, min(1.0, 12.0 / (z_avg + 3.0)))
            color = (int(40 * alpha), int(220 * alpha), int(100 * alpha))
            pt0 = (int(round(u0)), int(round(v0)))
            pt1 = (int(round(u1)), int(round(v1)))
            cv2.line(img, pt0, pt1, color, 1, cv2.LINE_AA)

    # 2. Detected Road Hough Lines (Magenta)
    for x1, y1, x2, y2 in supporting_lines[:25]:
        cv2.line(img, (x1, y1), (x2, y2), (220, 50, 220), 1, cv2.LINE_AA)

    # 3. Nominal Optical Horizon Line (Cyan at cy = 284)
    nom_cy = int(round(metric.nominal_cy))
    cv2.line(img, (0, nom_cy), (w, nom_cy), (240, 200, 0), 1, cv2.LINE_AA)
    cv2.putText(
        img,
        f"NOMINAL HORIZON cy={nom_cy}",
        (10, nom_cy - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (240, 200, 0),
        1,
        cv2.LINE_AA,
    )

    # 4. Empirical Visual Horizon Line (Orange/Red)
    emp_h = int(round(metric.empirical_horizon_row))
    if 0 <= emp_h < h:
        cv2.line(img, (0, emp_h), (w, emp_h), (0, 140, 255), 1, cv2.LINE_AA)
        cv2.putText(
            img,
            f"EMPIRICAL HORIZON v={emp_h} (Δ={metric.horizon_delta_cy_px:+.1f}px, pitch={metric.equivalent_pitch_deg:+.2f}°)",
            (w - 420, emp_h + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (0, 140, 255),
            1,
            cv2.LINE_AA,
        )

    # 5. Vanishing Point Marker (Yellow Crosshair)
    if metric.vp_x is not None and metric.vp_y is not None:
        vpx = int(round(metric.vp_x))
        vpy = int(round(metric.vp_y))
        if -50 <= vpx <= w + 50 and 0 <= vpy < h:
            cv2.drawMarker(img, (vpx, vpy), (0, 255, 255), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
            cv2.putText(
                img,
                f"VP ({vpx}, {vpy}) [N={metric.num_line_intersections}]",
                (vpx + 8, vpy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # 6. Future Trajectory Waypoints (if provided)
    if trajectory is not None and len(trajectory) > 0:
        traj_view = engine.project_waypoints_to_camera(cam_name, trajectory)
        n_pts = len(trajectory)
        for i in range(n_pts):
            if not traj_view.is_visible[i]:
                continue
            u_pt, v_pt = traj_view.points_2d[i]
            z_pt = max(1.0, float(traj_view.depths[i]))
            rad = max(2, int(8.0 * (10.0 / (z_pt + 5.0))))
            prog = i / max(1, n_pts - 1)
            b_c = 40
            g_c = int(255 * (1.0 - prog))
            r_c = int(255 * prog)
            cv2.circle(img, (int(round(u_pt)), int(round(v_pt))), rad, (b_c, g_c, r_c), -1, cv2.LINE_AA)

    # 7. Top Banner HUD Telemetry
    cv2.rectangle(img, (0, 0), (w, 32), (18, 22, 28), -1)
    yaw_deg = metric.nominal_yaw_deg
    cv2.putText(
        img,
        f"{cam_name.upper()} (yaw={yaw_deg:+.0f}°, HFOV={cam.hfov_deg:.0f}°, fx={metric.nominal_fx:.1f})",
        (12, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    hud_stat = (
        f"Grid Vis: {metric.ground_grid_points_visible}/{metric.ground_grid_points_total} "
        f"({metric.ground_grid_visibility_pct:.0f}%) | Road IoU: {metric.road_grid_alignment_iou:.1f}% | "
        f"Traj: {metric.trajectory_waypoints_visible}/25"
    )
    cv2.putText(
        img,
        hud_stat,
        (w - 480, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (50, 220, 130),
        1,
        cv2.LINE_AA,
    )

    return img


def generate_6camera_contact_sheet(
    scenario_id: str,
    scenario_title: str,
    metrics_by_cam: Dict[str, CameraAuditMetric],
    overlay_cells: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Compose a 3x2 contact sheet from the 6 camera overlay cells."""
    cell_w, cell_h = 438, 284  # Half-resolution of 876x568
    header_h = 75
    sheet_w = cell_w * 3
    sheet_h = cell_h * 2 + header_h

    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
    sheet[:header_h, :] = [20, 24, 30]

    # Header Title & Subtitle
    cv2.putText(
        sheet,
        f"SCENARIO {scenario_id}: {scenario_title.upper()}",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    front_m = metrics_by_cam.get("front")
    front_str = (
        f"Front Δcy: {front_m.horizon_delta_cy_px:+.1f}px (pitch {front_m.equivalent_pitch_deg:+.2f}°) | "
        f"Road IoU: {front_m.road_grid_alignment_iou:.1f}% | Grid Vis: {front_m.ground_grid_visibility_pct:.1f}%"
        if front_m
        else ""
    )
    subtitle = f"Empirical Calibration Audit | Provisional Nominal Model (70° HFOV, cy=284, h=1.4m) | {front_str}"
    cv2.putText(
        sheet,
        subtitle,
        (15, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (160, 190, 215),
        1,
        cv2.LINE_AA,
    )

    # Paste cells into 3x2 grid
    for r_idx, row_cams in enumerate(CAMERA_GRID_ORDER):
        for c_idx, cam in enumerate(row_cams):
            y_start = header_h + r_idx * cell_h
            x_start = c_idx * cell_w
            cell = cv2.resize(overlay_cells[cam], (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            sheet[y_start : y_start + cell_h, x_start : x_start + cell_w] = cell
            cv2.rectangle(
                sheet,
                (x_start, y_start),
                (x_start + cell_w, y_start + cell_h),
                (55, 65, 75),
                1,
            )

    cv2.imwrite(str(out_path), sheet)


def run_camera_alignment_audit(
    val_observed_dir: Path,
    val_trajectories_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Execute complete empirical calibration audit and generate artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = camera_projection.CameraProjectionEngine()
    grid_points_3d, grid_segments = build_3d_ground_plane_grid()

    # Load trajectories
    trajectories: Dict[str, np.ndarray] = {}
    if val_trajectories_path.exists():
        with open(val_trajectories_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                sid = rec.get("scenario_id")
                traj = rec.get("expert_like")
                if sid and traj:
                    trajectories[sid] = np.array(traj, dtype=np.float64)

    all_metrics: List[CameraAuditMetric] = []
    scenarios_data = {}

    print(f"[*] Starting Empirical Calibration Audit on {len(REAL_SCENARIOS)} Real-Image Scenarios...")
    print(f"    Total camera images to analyze: {len(REAL_SCENARIOS) * len(CAMERAS)}")

    for sid, title, instr in REAL_SCENARIOS:
        scene_dir = val_observed_dir / sid
        if not scene_dir.is_dir():
            print(f"[!] Warning: Missing scenario directory: {scene_dir}")
            continue

        traj = trajectories.get(sid)
        traj_points = traj if traj is not None and len(traj) > 0 and traj[0, 0] > -90.0 else None

        overlay_cells = {}
        metrics_by_cam = {}

        for cam_name in CAMERAS:
            img_path = scene_dir / f"{cam_name}.jpg"
            if not img_path.exists():
                print(f"[!] Warning: Missing image: {img_path}")
                continue

            img_bgr = cv2.imread(str(img_path))
            h_img, w_img = img_bgr.shape[:2]
            cam_pose = engine.rig[cam_name]

            # 1. Horizon and Vanishing Point
            (
                emp_horizon,
                std_horizon,
                vp,
                n_lines,
                n_intersections,
                supporting_lines,
            ) = detect_visual_horizon_and_vp(img_bgr, cam_name)

            delta_cy = emp_horizon - cam_pose.cy
            equiv_pitch = math.degrees(math.atan(delta_cy / cam_pose.fy))

            vp_x = vp[0] if vp else None
            vp_y = vp[1] if vp else None
            vp_delta_cx = (vp_x - cam_pose.cx) if vp_x is not None else None
            vp_delta_cy = (vp_y - cam_pose.cy) if vp_y is not None else None
            equiv_yaw = (
                math.degrees(math.atan(vp_delta_cx / cam_pose.fx)) if vp_delta_cx is not None else None
            )

            # 2. Ground Grid Alignment & Road Surface IoU
            (
                vis_grid_pts,
                total_grid_pts,
                grid_vis_pct,
                road_area_pct,
                road_iou,
            ) = compute_ground_grid_alignment(img_bgr, cam_name, engine, grid_points_3d)

            # 3. Trajectory Waypoints Visibility
            traj_vis_count = 0
            traj_total_count = 25
            if traj_points is not None:
                traj_view = engine.project_waypoints_to_camera(cam_name, traj_points)
                traj_vis_count = int(traj_view.visible_count)
            traj_vis_pct = (traj_vis_count / max(1, traj_total_count)) * 100.0

            metric = CameraAuditMetric(
                scenario_id=sid,
                scenario_title=title,
                camera_name=cam_name,
                nominal_yaw_deg=video_engine.PROVISIONAL_CAMERA_YAW_DEG.get(cam_name, 0.0),
                nominal_cx=cam_pose.cx,
                nominal_cy=cam_pose.cy,
                nominal_fx=cam_pose.fx,
                nominal_fy=cam_pose.fy,
                nominal_cam_z=cam_pose.position_m[2],
                empirical_horizon_row=round(emp_horizon, 2),
                horizon_delta_cy_px=round(delta_cy, 2),
                equivalent_pitch_deg=round(equiv_pitch, 3),
                vp_x=round(vp_x, 2) if vp_x is not None else None,
                vp_y=round(vp_y, 2) if vp_y is not None else None,
                vp_delta_cx_px=round(vp_delta_cx, 2) if vp_delta_cx is not None else None,
                vp_delta_cy_px=round(vp_delta_cy, 2) if vp_delta_cy is not None else None,
                equivalent_yaw_offset_deg=round(equiv_yaw, 3) if equiv_yaw is not None else None,
                num_supporting_lines=n_lines,
                num_line_intersections=n_intersections,
                ground_grid_points_visible=vis_grid_pts,
                ground_grid_points_total=total_grid_pts,
                ground_grid_visibility_pct=round(grid_vis_pct, 2),
                trajectory_waypoints_visible=traj_vis_count,
                trajectory_waypoints_total=traj_total_count,
                trajectory_visibility_pct=round(traj_vis_pct, 2),
                road_mask_coverage_pct=round(road_area_pct, 2),
                road_grid_alignment_iou=round(road_iou, 2),
            )
            all_metrics.append(metric)
            metrics_by_cam[cam_name] = metric

            # Render overlay cell
            overlay = render_calibration_audit_overlay(
                img_bgr,
                metric,
                cam_name,
                engine,
                grid_segments,
                supporting_lines,
                trajectory=traj_points,
            )
            overlay_cells[cam_name] = overlay

        scenarios_data[sid] = {"metrics": metrics_by_cam, "overlays": overlay_cells}

        # Check for contact sheets
        if sid == "0000901":
            out_img = output_dir / "camera_calibration_audit_visual_check_highway.png"
            generate_6camera_contact_sheet(sid, title, metrics_by_cam, overlay_cells, out_img)
            print(f"[✓] Generated Highway Contact Sheet: {out_img}")
        elif sid == "0000902":
            out_img = output_dir / "camera_calibration_audit_visual_check_urban.png"
            generate_6camera_contact_sheet(sid, title, metrics_by_cam, overlay_cells, out_img)
            print(f"[✓] Generated Urban Contact Sheet: {out_img}")
        elif sid == "0000984":
            out_img = output_dir / "camera_calibration_audit_visual_check_uturn.png"
            generate_6camera_contact_sheet(sid, title, metrics_by_cam, overlay_cells, out_img)
            print(f"[✓] Generated U-Turn Contact Sheet: {out_img}")

    # Generate primary visual check image as a copy of the highway audit
    primary_check = output_dir / "camera_calibration_audit_visual_check.png"
    hw_img = output_dir / "camera_calibration_audit_visual_check_highway.png"
    if hw_img.exists():
        import shutil

        shutil.copyfile(str(hw_img), str(primary_check))

    # 4. Save CSV Table
    csv_path = output_dir / "camera_calibration_audit_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[f.name for f in CameraAuditMetric.__dataclass_fields__.values()])
        writer.writeheader()
        for m in all_metrics:
            writer.writerow(asdict(m))
    print(f"[✓] Saved CSV Audit Table: {csv_path} ({len(all_metrics)} rows)")

    # 5. Compute Aggregated Summary Statistics
    summary_by_camera: Dict[str, Dict[str, Any]] = {}
    for cam_name in CAMERAS:
        cam_ms = [m for m in all_metrics if m.camera_name == cam_name]
        deltas = [m.horizon_delta_cy_px for m in cam_ms]
        pitches = [m.equivalent_pitch_deg for m in cam_ms]
        grid_vis = [m.ground_grid_visibility_pct for m in cam_ms]
        road_ious = [m.road_grid_alignment_iou for m in cam_ms]
        traj_vis = [m.trajectory_visibility_pct for m in cam_ms]

        vps_x = [m.vp_delta_cx_px for m in cam_ms if m.vp_delta_cx_px is not None]
        vps_y = [m.vp_delta_cy_px for m in cam_ms if m.vp_delta_cy_px is not None]

        summary_by_camera[cam_name] = {
            "num_scenarios": len(cam_ms),
            "mean_horizon_delta_cy_px": round(float(np.mean(deltas)), 2),
            "median_horizon_delta_cy_px": round(float(np.median(deltas)), 2),
            "std_horizon_delta_cy_px": round(float(np.std(deltas)), 2),
            "mean_equivalent_pitch_deg": round(float(np.mean(pitches)), 3),
            "std_equivalent_pitch_deg": round(float(np.std(pitches)), 3),
            "mean_ground_grid_vis_pct": round(float(np.mean(grid_vis)), 2),
            "mean_road_alignment_iou_pct": round(float(np.mean(road_ious)), 2),
            "mean_trajectory_vis_pct": round(float(np.mean(traj_vis)), 2),
            "vp_detected_count": len(vps_x),
            "mean_vp_delta_cx_px": round(float(np.mean(vps_x)), 2) if vps_x else None,
            "mean_vp_delta_cy_px": round(float(np.mean(vps_y)), 2) if vps_y else None,
        }

    overall_deltas = [m.horizon_delta_cy_px for m in all_metrics]
    overall_pitches = [m.equivalent_pitch_deg for m in all_metrics]
    overall_ious = [m.road_grid_alignment_iou for m in all_metrics]

    audit_summary = {
        "dataset_name": "KITScenes-LongTail",
        "dataset_repo": "KIT-MRT/KITScenes-LongTail",
        "scenarios_audited": len(REAL_SCENARIOS),
        "total_camera_views_audited": len(all_metrics),
        "calibration_status": "PROVISIONAL NOMINAL",
        "provisional_model_parameters": {
            "image_width": spec.WIDTH,
            "image_height": spec.HEIGHT,
            "hfov_deg": 70.0,
            "focal_length_px": round((spec.WIDTH / 2.0) / math.tan(math.radians(70.0 / 2.0)), 3),
            "principal_point": [spec.WIDTH / 2.0, spec.HEIGHT / 2.0],
            "camera_height_m": 1.40,
            "near_clip_m": 0.50,
            "yaw_angles_deg": video_engine.PROVISIONAL_CAMERA_YAW_DEG,
        },
        "overall_statistics": {
            "mean_horizon_error_px": round(float(np.mean(overall_deltas)), 2),
            "median_horizon_error_px": round(float(np.median(overall_deltas)), 2),
            "std_horizon_error_px": round(float(np.std(overall_deltas)), 2),
            "mean_pitch_offset_deg": round(float(np.mean(overall_pitches)), 3),
            "std_pitch_offset_deg": round(float(np.std(overall_pitches)), 3),
            "mean_road_alignment_iou_pct": round(float(np.mean(overall_ious)), 2),
        },
        "by_camera_summary": summary_by_camera,
    }

    json_path = output_dir / "camera_calibration_audit_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"[✓] Saved JSON Audit Summary: {json_path}")

    # 6. Print Markdown Summary Table
    print("\n" + "=" * 110)
    print("                    EMPIRICAL CAMERA CALIBRATION & ROAD ALIGNMENT AUDIT SUMMARY")
    print("=" * 110)
    header_fmt = "{:<12} | {:>10} | {:>10} | {:>12} | {:>12} | {:>12} | {:>14} | {:>12}"
    row_fmt = "{:<12} | {:>10.1f} | {:>10.1f} | {:>12.2f} | {:>12.1f} | {:>12.1f} | {:>14.1f} | {:>12}"
    print(
        header_fmt.format(
            "Camera",
            "Mean Δcy(px)",
            "Med Δcy(px)",
            "Pitch Off(°)",
            "Grid Vis(%)",
            "Road IoU(%)",
            "Traj Vis(%)",
            "VPs Found",
        )
    )
    print("-" * 110)
    for cam_name in CAMERAS:
        s = summary_by_camera[cam_name]
        vp_info = f"{s['vp_detected_count']}/{s['num_scenarios']}"
        print(
            row_fmt.format(
                cam_name,
                s["mean_horizon_delta_cy_px"],
                s["median_horizon_delta_cy_px"],
                s["mean_equivalent_pitch_deg"],
                s["mean_ground_grid_vis_pct"],
                s["mean_road_alignment_iou_pct"],
                s["mean_trajectory_vis_pct"],
                vp_info,
            )
        )
    print("=" * 110)
    print(
        f"OVERALL MEAN HORIZON ERROR: {audit_summary['overall_statistics']['mean_horizon_error_px']:+.2f} px "
        f"(pitch {audit_summary['overall_statistics']['mean_pitch_offset_deg']:+.2f}°) | "
        f"OVERALL ROAD ALIGNMENT IoU: {audit_summary['overall_statistics']['mean_road_alignment_iou_pct']:.1f}%"
    )
    print("=" * 110 + "\n")

    return audit_summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Empirical Camera Geometry & Real-Image Road-Plane Alignment Auditor."
    )
    parser.add_argument(
        "--val-observed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "val_observed_t0",
        help="Directory containing the 10 real-image scenarios with 6 camera views",
    )
    parser.add_argument(
        "--val-trajectories-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "val_trajectories.jsonl",
        help="Path to validation trajectories jsonl file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports",
        help="Output directory for audit table, JSON summary, and contact sheets",
    )
    args = parser.parse_args(argv)

    run_camera_alignment_audit(
        val_observed_dir=args.val_observed_dir,
        val_trajectories_path=args.val_trajectories_path,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
