"""Generate 60-view visual camera calibration audit across 10 genuine scenarios x 6 cameras.

Evaluates the official KITScenes-LongTail calibration (Wagner et al. 2026, pinhole model)
projected onto genuine observed t0 camera frames.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from kitscenes import camera_projection, spec


def run_visual_audit(
    trajectories_file: Path = Path("data/val_trajectories.jsonl"),
    observed_t0_dir: Path = Path("data/val_observed_t0"),
    output_dir: Path = Path("reports/visual_calibration_audit"),
    report_json_path: Path = Path("reports/visual_calibration_audit.json"),
    report_md_path: Path = Path("reports/visual_calibration_audit.md"),
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.parent.mkdir(parents=True, exist_ok=True)

    engine = camera_projection.CameraProjectionEngine()
    assert engine.rig.is_calibrated is True, "Rig must use official calibration!"

    # Load trajectories
    scenarios: Dict[str, Dict[str, Any]] = {}
    with open(trajectories_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            sid = item["scenario_id"]
            if (observed_t0_dir / sid).exists():
                scenarios[sid] = item

    scenario_ids = sorted(list(scenarios.keys()))[:10]
    print(f"[Audit] Running 60-view audit on {len(scenario_ids)} scenarios: {scenario_ids}")

    audit_results: Dict[str, Any] = {
        "calibration_provenance": {
            "source": engine.rig.calibration_source,
            "config_file": "configs/kitscenes_camera_calibration.json",
            "is_calibrated": True,
            "nominal_ground_z_ego": engine.rig.nominal_ground_z_ego,
            "cameras_audited": spec.CAMERAS,
        },
        "scenario_count": len(scenario_ids),
        "total_views_audited": len(scenario_ids) * len(spec.CAMERAS),
        "views": {},
        "summary": {
            "front_camera_visibility_rate": 0.0,
            "all_cameras_visibility_counts": {cam: 0 for cam in spec.CAMERAS},
            "total_masks_generated": 0,
        },
    }

    total_front_visible_pts = 0
    total_front_expected_pts = 0

    for sid in scenario_ids:
        item = scenarios[sid]
        traj = np.array(item["expert_like"][:spec.FRAME_COUNT], dtype=np.float64)
        instruction = item.get("driving_instruction", "N/A")
        scenario_type = item.get("scenario_type", "N/A")

        audit_results["views"][sid] = {
            "instruction": instruction,
            "scenario_type": scenario_type,
            "cameras": {},
        }

        # Multi-camera contact sheet: 2 rows x 3 columns of overlay images
        # Row 1: front_left, front, front_right
        # Row 2: rear_left, rear, rear_right
        grid_order = [
            ["front_left", "front", "front_right"],
            ["rear_left", "rear", "rear_right"],
        ]

        for cam_name in spec.CAMERAS:
            proj_view = engine.project_waypoints_to_camera(cam_name, traj)
            mask = engine.render_spatial_control_mask(cam_name, traj, ribbon_width_px=14)

            vis_pts = proj_view.points_2d[proj_view.is_visible]
            depths_vis = proj_view.depths[proj_view.is_visible]

            u_range = [float(vis_pts[:, 0].min()), float(vis_pts[:, 0].max())] if len(vis_pts) > 0 else None
            v_range = [float(vis_pts[:, 1].min()), float(vis_pts[:, 1].max())] if len(vis_pts) > 0 else None
            z_range = [float(depths_vis.min()), float(depths_vis.max())] if len(depths_vis) > 0 else None

            fg_pixel_count = int(np.sum(mask > 0.05))

            cam_record = {
                "visible_waypoints": int(proj_view.visible_count),
                "total_waypoints": len(traj),
                "u_range": u_range,
                "v_range": v_range,
                "depth_range_m": z_range,
                "fg_pixel_count": fg_pixel_count,
            }
            audit_results["views"][sid]["cameras"][cam_name] = cam_record
            audit_results["summary"]["all_cameras_visibility_counts"][cam_name] += proj_view.visible_count
            audit_results["summary"]["total_masks_generated"] += 1

            if cam_name == "front":
                total_front_visible_pts += proj_view.visible_count
                total_front_expected_pts += len(traj)

            # Load genuine observed image if present and overlay ribbon
            img_path = observed_t0_dir / sid / f"{cam_name}.jpg"
            if img_path.exists():
                orig_bgr = cv2.imread(str(img_path))
            else:
                orig_bgr = np.zeros((spec.HEIGHT, spec.WIDTH, 3), dtype=np.uint8)

            # Overlay mask in cyan/green: BGR (255, 200, 0) for route ribbon
            ribbon_color = np.array([255, 200, 0], dtype=np.uint8)  # Cyan-blue in BGR
            alpha = np.clip(mask[..., None] * 0.75, 0.0, 1.0)
            blended = (orig_bgr * (1.0 - alpha) + ribbon_color * alpha).astype(np.uint8)

            # Draw camera label and visibility count
            lbl = f"{cam_name.upper()} | Vis: {proj_view.visible_count}/25 | FG: {fg_pixel_count}px"
            cv2.rectangle(blended, (8, 8), (420, 36), (0, 0, 0), -1)
            cv2.putText(blended, lbl, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            out_img_path = output_dir / f"{sid}_{cam_name}_calib_overlay.jpg"
            cv2.imwrite(str(out_img_path), blended)

        # Assemble 2x3 contact sheet for this scenario
        row0_imgs = [cv2.imread(str(output_dir / f"{sid}_{c}_calib_overlay.jpg")) for c in grid_order[0]]
        row1_imgs = [cv2.imread(str(output_dir / f"{sid}_{c}_calib_overlay.jpg")) for c in grid_order[1]]
        # Resize to thumbnail width 438, height 284 for contact sheet
        thumb_w, thumb_h = 438, 284
        row0_thumbs = [cv2.resize(im, (thumb_w, thumb_h)) for im in row0_imgs]
        row1_thumbs = [cv2.resize(im, (thumb_w, thumb_h)) for im in row1_imgs]
        contact_sheet = np.vstack([np.hstack(row0_thumbs), np.hstack(row1_thumbs)])

        sheet_path = output_dir / f"{sid}_6cam_calibration_contact_sheet.jpg"
        cv2.imwrite(str(sheet_path), contact_sheet)
        audit_results["views"][sid]["contact_sheet"] = str(sheet_path)

    audit_results["summary"]["front_camera_visibility_rate"] = float(
        total_front_visible_pts / max(total_front_expected_pts, 1)
    )

    # Save JSON report
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    print(f"[Audit] Saved JSON audit report to {report_json_path}")

    # Generate Markdown summary report
    md_lines = [
        "# KITScenes Official Camera Calibration Visual Audit (60 Views)",
        "",
        f"**Calibration Provenance:** {audit_results['calibration_provenance']['source']}",
        f"**Calibration Config:** [`configs/kitscenes_camera_calibration.json`](file://{Path('configs/kitscenes_camera_calibration.json').resolve()})",
        f"**Scenarios Audited:** {len(scenario_ids)} scenarios ({', '.join(scenario_ids)})",
        f"**Total Multi-Camera Views:** {len(scenario_ids) * 6} views",
        f"**Front Camera Waypoint Visibility Rate:** {audit_results['summary']['front_camera_visibility_rate']*100:.2f}%",
        "",
        "## Camera Projection Verification Summary",
        "",
        "| Camera Name | Nominal Yaw | Calibrated Optical Center (cx, cy) | Total Visible Waypoints Across 10 Scenarios |",
        "|---|---|---|---|",
    ]

    for cam in spec.CAMERAS:
        c_obj = engine.rig[cam]
        vis_cnt = audit_results["summary"]["all_cameras_visibility_counts"][cam]
        md_lines.append(
            f"| `{cam}` | {c_obj.yaw_deg:+.1f}° | ({c_obj.cx:.2f}, {c_obj.cy:.2f}) | {vis_cnt} / 250 waypoints |"
        )

    md_lines.extend([
        "",
        "## Scenario Breakdown & Visual Overlays",
        "",
    ])

    for sid in scenario_ids:
        s_data = audit_results["views"][sid]
        md_lines.extend([
            f"### Scenario `{sid}`",
            f"- **Instruction:** *\"{s_data['instruction']}\"*",
            f"- **Scenario Type:** `{s_data['scenario_type']}`",
            f"- **6-Camera Contact Sheet:** [`{sid}_6cam_calibration_contact_sheet.jpg`](file://{(output_dir / f'{sid}_6cam_calibration_contact_sheet.jpg').resolve()})",
            "",
            "| Camera | Visible / Total | Depth Range (m) | Canvas Range [u_min, u_max] x [v_min, v_max] | Ribbon Area (px) |",
            "|---|---|---|---|---|",
        ])
        for cam in spec.CAMERAS:
            c_info = s_data["cameras"][cam]
            v_cnt = f"{c_info['visible_waypoints']} / {c_info['total_waypoints']}"
            z_r = f"[{c_info['depth_range_m'][0]:.1f}, {c_info['depth_range_m'][1]:.1f}]" if c_info['depth_range_m'] else "None"
            u_r = f"[{c_info['u_range'][0]:.1f}, {c_info['u_range'][1]:.1f}]" if c_info['u_range'] else "None"
            v_r = f"[{c_info['v_range'][0]:.1f}, {c_info['v_range'][1]:.1f}]" if c_info['v_range'] else "None"
            fg_px = f"{c_info['fg_pixel_count']:,}"
            md_lines.append(f"| `{cam}` | {v_cnt} | {z_r} | {u_r} x {v_r} | {fg_px} |")
        md_lines.append("")

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[Audit] Saved Markdown audit report to {report_md_path}")

    return audit_results


if __name__ == "__main__":
    run_visual_audit()
