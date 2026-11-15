"""Multi-Camera 3D-to-2D Projection & Spatial Conditioning Engine for KITScenes-LongTail.

Implements exact coordinate transformations from vehicle ego-body coordinates
(X: forward, Y: left, Z: up) into the 6 surround camera frames (front, front_left,
front_right, rear, rear_left, rear_right) and projects 3D waypoints onto 2D image planes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from kitscenes import spec
from kitscenes.video_engine import (
    CAMERA_YAW_DEG,
    CAMERA_YAW_RAD,
    DEFAULT_CAMERA_POSITIONS,
    CameraPose,
    CameraRig,
)


@dataclass(frozen=True)
class ProjectedCameraView:
    """Projected trajectory points and visibility metadata for a single camera view."""
    camera_name: str
    points_2d: np.ndarray        # (N, 2) pixel coordinates [u, v]
    depths: np.ndarray           # (N,) optical depth z_c in meters
    is_in_front: np.ndarray      # (N,) boolean mask: z_c > near_clip
    is_in_frame: np.ndarray      # (N,) boolean mask: inside [0, W) x [0, H)
    is_visible: np.ndarray       # (N,) boolean mask: is_in_front & is_in_frame
    visible_count: int


class CameraProjectionEngine:
    """Projects 3D planned waypoints into 2D pixel coordinates across all 6 surround cameras."""

    def __init__(
        self,
        rig: Optional[CameraRig] = None,
        width: int = spec.WIDTH,
        height: int = spec.HEIGHT,
        near_clip_m: float = 0.5,
    ):
        self.width = width
        self.height = height
        self.near_clip_m = near_clip_m
        self.rig = rig or CameraRig(width=width, height=height)

    def project_waypoints_to_camera(
        self,
        camera_name: str,
        waypoints_ego: np.ndarray,
        ground_z_m: float = 0.0,
    ) -> ProjectedCameraView:
        """Project ego-frame waypoints onto the specified camera's image canvas.

        Args:
            camera_name: One of spec.CAMERAS ('front', 'front_left', etc.)
            waypoints_ego: (N, 2) or (N, 3) array in vehicle body frame
                           (X: forward, Y: left, Z: up).
            ground_z_m: Elevation of road surface relative to vehicle origin (default: 0.0).

        Returns:
            ProjectedCameraView containing 2D points, optical depths, and visibility masks.
        """
        assert camera_name in self.rig.cameras, f"Unknown camera: {camera_name}"
        cam = self.rig.cameras[camera_name]

        pts = np.asarray(waypoints_ego, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        n = pts.shape[0]

        # Convert to 3D ego body points [X, Y, Z]
        if pts.shape[1] == 2:
            p_ego = np.column_stack([pts[:, 0], pts[:, 1], np.full(n, ground_z_m, dtype=np.float64)])
        else:
            p_ego = pts[:, :3]

        # Camera mounting position in ego body frame
        t_cam = np.asarray(cam.position_m, dtype=np.float64)

        # Vector from camera mount to point in ego body frame:
        p_rel = p_ego - t_cam[None, :]

        # Rotate into camera frame (x: right, y: down, z: forward along optical axis)
        # R_body_to_cam maps (X_b, Y_b, Z_b) -> (x_c, y_c, z_c)
        r_mat = cam.rotation_matrix_body_to_cam
        p_cam = p_rel @ r_mat.T

        xc = p_cam[:, 0]
        yc = p_cam[:, 1]
        zc = p_cam[:, 2]

        is_in_front = zc > self.near_clip_m

        # Safe division by depth: for points behind near plane, clamp depth to near_clip to avoid NaN
        safe_zc = np.where(is_in_front, zc, self.near_clip_m)

        # Pinhole projection: u = fx * (xc / zc) + cx, v = fy * (yc / zc) + cy
        u = cam.fx * (xc / safe_zc) + cam.cx
        v = cam.fy * (yc / safe_zc) + cam.cy
        points_2d = np.column_stack([u, v])

        # Bounds check on image canvas
        is_in_frame = (u >= 0.0) & (u < self.width) & (v >= 0.0) & (v < self.height)
        is_visible = is_in_front & is_in_frame

        return ProjectedCameraView(
            camera_name=camera_name,
            points_2d=points_2d,
            depths=zc,
            is_in_front=is_in_front,
            is_in_frame=is_in_frame,
            is_visible=is_visible,
            visible_count=int(np.sum(is_visible)),
        )

    def project_all_cameras(
        self,
        waypoints_ego: np.ndarray,
        ground_z_m: float = 0.0,
    ) -> Dict[str, ProjectedCameraView]:
        """Project waypoints across all 6 surround cameras simultaneously."""
        return {
            cam: self.project_waypoints_to_camera(cam, waypoints_ego, ground_z_m=ground_z_m)
            for cam in spec.CAMERAS
        }

    def render_spatial_control_mask(
        self,
        camera_name: str,
        waypoints_ego: np.ndarray,
        ribbon_width_px: int = 12,
        decay_with_depth: bool = True,
    ) -> np.ndarray:
        """Render a single-channel spatial conditioning mask (H, W) for ControlNet-SVD.

        Values range from 0.0 (background) to 1.0 (on-trajectory anchor).
        Ribbon width and intensity gracefully decay with distance to reflect perspective depth.
        """
        proj = self.project_waypoints_to_camera(camera_name, waypoints_ego)
        mask = np.zeros((self.height, self.width), dtype=np.float32)

        if proj.visible_count == 0:
            return mask

        # Rasterize visible consecutive segments onto mask
        pts = proj.points_2d
        vis = proj.is_visible
        depths = proj.depths

        # Simple anti-aliased polyline rasterization
        for i in range(len(pts) - 1):
            if not (vis[i] or vis[i + 1]):
                continue

            p1 = pts[i]
            p2 = pts[i + 1]
            z_avg = max(float((depths[i] + depths[i + 1]) / 2.0), 1.0)

            # Width in pixels scales inversely with depth
            w_px = max(1, int(ribbon_width_px * (10.0 / (z_avg + 5.0))))
            intensity = float(max(0.2, min(1.0, 15.0 / (z_avg + 5.0)))) if decay_with_depth else 1.0

            # Linear interpolation of segment
            seg_dist = float(np.linalg.norm(p2 - p1))
            n_steps = max(2, int(seg_dist * 2))
            alphas = np.linspace(0.0, 1.0, n_steps)

            for a in alphas:
                px = int(round(p1[0] * (1.0 - a) + p2[0] * a))
                py = int(round(p1[1] * (1.0 - a) + p2[1] * a))
                if 0 <= px < self.width and 0 <= py < self.height:
                    # Draw circle of radius w_px // 2
                    r = max(1, w_px // 2)
                    y_min = max(0, py - r)
                    y_max = min(self.height, py + r + 1)
                    x_min = max(0, px - r)
                    x_max = min(self.width, px + r + 1)
                    mask[y_min:y_max, x_min:x_max] = np.maximum(
                        mask[y_min:y_max, x_min:x_max], intensity
                    )

        return mask
