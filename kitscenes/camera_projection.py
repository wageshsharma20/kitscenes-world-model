"""Multi-Camera 3D-to-2D Projection & Spatial Conditioning Engine for KITScenes-LongTail.

Implements coordinate transformations from vehicle ego-body coordinates (X: forward,
Y: left, Z: up) into the 6 surround camera frames (front, front_left, front_right,
rear, rear_left, rear_right) and rasterizes spatial conditioning masks for world models.

CALIBRATION STATUS:
The official KITScenes-LongTail dataset files do not publish explicit extrinsic or
intrinsic calibration matrices. The rig parameters implemented here represent a
PROVISIONAL NOMINAL SENSOR RIG MODEL (70 deg HFOV pinhole, nominal surround yaw offsets,
and wheelbase-scaled mounting positions). They are subject to uncertainty and should
be treated as a provisional approximation rather than certified exact calibration.

ROBUST CLIPPING PIPELINE:
1. 3D Near-Plane Clipping: Line segments are clipped against z_c >= near_clip_m.
2. Perspective Pinhole Projection: Clipped 3D points are mapped to 2D pixel coordinates.
3. 2D Canvas Boundary Clipping: Segment endpoints are clipped against [0, W-1] x [0, H-1].
4. Subpixel Anti-Aliased Rasterization: Eliminates cross-screen streak artifacts.

SPATIAL CONDITIONING VARIANTS:
- Option A: Static Route Corridor Mask [B, 1, 568, 876]
- Option B.1: Route-Initial-Frame Temporal Masks [B, 25, 1, 568, 876] (viewed from t=0 mount)
- Option B.2: Future-Camera-Frame Spatial Masks [B, 25, 1, 568, 876] (ego-motion aligned: camera
  translates and yaws with the vehicle at each future timestep t in {1..25})
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from kitscenes import spec
from kitscenes.video_engine import (
    CameraPose,
    CameraRig,
)

# Cohen-Sutherland 2D outcodes
_INSIDE = 0
_LEFT = 1
_RIGHT = 2
_BOTTOM = 4
_TOP = 8


def clip_segment_3d_near_plane(
    p1_cam: np.ndarray,
    p2_cam: np.ndarray,
    near_clip_m: float = 0.5,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Clip a 3D line segment against the camera near clipping plane z_c >= near_clip_m.

    Args:
        p1_cam: (3,) array [x, y, z] in camera optical frame.
        p2_cam: (3,) array [x, y, z] in camera optical frame.
        near_clip_m: Minimum optical depth along z_c.

    Returns:
        Tuple (p1_clipped, p2_clipped) if at least part of the segment is in front
        of the near plane, else None.
    """
    z1 = float(p1_cam[2])
    z2 = float(p2_cam[2])

    if z1 < near_clip_m and z2 < near_clip_m:
        return None  # Completely behind near plane

    if z1 >= near_clip_m and z2 >= near_clip_m:
        return p1_cam.copy(), p2_cam.copy()  # Completely in front

    # One endpoint is behind near plane; compute exact planar intersection
    denom = z2 - z1
    if abs(denom) < 1e-9:
        return None
    alpha = (near_clip_m - z1) / denom

    p_inter = p1_cam + alpha * (p2_cam - p1_cam)
    p_inter[2] = near_clip_m

    if z1 < near_clip_m:
        return p_inter, p2_cam.copy()
    else:
        return p1_cam.copy(), p_inter


def clip_segment_2d_canvas(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: int = spec.WIDTH,
    height: int = spec.HEIGHT,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Clip a 2D line segment against the image canvas [0, width-1] x [0, height-1].

    Implements Liang-Barsky parametric line clipping algorithm (O(1), zero while-loop).
    """
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    p = [-dx, dx, -dy, dy]
    q = [float(x0), float(width - 1) - float(x0), float(y0), float(height - 1) - float(y0)]

    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0.0:
                return None  # Parallel and outside
        else:
            t = qi / pi
            if pi < 0.0:
                if t > u2:
                    return None
                if t > u1:
                    u1 = t
            else:  # pi > 0.0
                if t < u1:
                    return None
                if t < u2:
                    u2 = t
    if u1 > u2:
        return None
    return (float(x0 + u1 * dx), float(y0 + u1 * dy)), (float(x0 + u2 * dx), float(y0 + u2 * dy))


def project_and_clip_segment(
    p1_cam: np.ndarray,
    p2_cam: np.ndarray,
    cam: CameraPose,
    width: int = spec.WIDTH,
    height: int = spec.HEIGHT,
    near_clip_m: float = 0.5,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    """Clip 3D segment against near plane, project to 2D pinhole, and clip against canvas.

    Returns:
        Optional tuple: ((u0, v0), (u1, v1), avg_depth) or None if entirely clipped.
    """
    clipped_3d = clip_segment_3d_near_plane(p1_cam, p2_cam, near_clip_m=near_clip_m)
    if clipped_3d is None:
        return None
    pt1_3d, pt2_3d = clipped_3d

    # Pinhole projection
    z1 = max(float(pt1_3d[2]), near_clip_m)
    z2 = max(float(pt2_3d[2]), near_clip_m)

    u1 = cam.fx * (float(pt1_3d[0]) / z1) + cam.cx
    v1 = cam.fy * (float(pt1_3d[1]) / z1) + cam.cy
    u2 = cam.fx * (float(pt2_3d[0]) / z2) + cam.cx
    v2 = cam.fy * (float(pt2_3d[1]) / z2) + cam.cy

    clipped_2d = clip_segment_2d_canvas(u1, v1, u2, v2, width=width, height=height)
    if clipped_2d is None:
        return None

    (u1_c, v1_c), (u2_c, v2_c) = clipped_2d
    z_avg = (z1 + z2) / 2.0
    return (u1_c, v1_c), (u2_c, v2_c), z_avg


def compute_trajectory_headings(waypoints: np.ndarray) -> np.ndarray:
    """Compute local yaw headings (radians) along a trajectory in world frame."""
    pts = np.asarray(waypoints, dtype=np.float64)
    n = len(pts)
    headings = np.zeros(n, dtype=np.float64)
    if n <= 1:
        return headings

    # Forward difference at start
    dx0 = pts[1, 0] - pts[0, 0]
    dy0 = pts[1, 1] - pts[0, 1]
    headings[0] = math.atan2(dy0, dx0)

    # Central differences along interior
    for i in range(1, n - 1):
        dxi = pts[i + 1, 0] - pts[i - 1, 0]
        dyi = pts[i + 1, 1] - pts[i - 1, 1]
        if math.hypot(dxi, dyi) > 1e-4:
            headings[i] = math.atan2(dyi, dxi)
        else:
            headings[i] = headings[i - 1]

    # Backward difference at end
    dx_end = pts[-1, 0] - pts[-2, 0]
    dy_end = pts[-1, 1] - pts[-2, 1]
    if math.hypot(dx_end, dy_end) > 1e-4:
        headings[-1] = math.atan2(dy_end, dx_end)
    else:
        headings[-1] = headings[-2]

    return headings


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
    """Multi-camera projection engine with rigorous 3D near-plane & 2D canvas clipping."""

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

        if pts.shape[1] == 2:
            p_ego = np.column_stack([pts[:, 0], pts[:, 1], np.full(n, ground_z_m, dtype=np.float64)])
        else:
            p_ego = pts[:, :3]

        t_cam = np.asarray(cam.position_m, dtype=np.float64)
        p_rel = p_ego - t_cam[None, :]

        r_mat = cam.rotation_matrix_body_to_cam
        p_cam = p_rel @ r_mat.T

        xc = p_cam[:, 0]
        yc = p_cam[:, 1]
        zc = p_cam[:, 2]

        is_in_front = zc > self.near_clip_m
        safe_zc = np.where(is_in_front, zc, self.near_clip_m)

        u = cam.fx * (xc / safe_zc) + cam.cx
        v = cam.fy * (yc / safe_zc) + cam.cy
        points_2d = np.column_stack([u, v])

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
        ribbon_width_px: int = 14,
        decay_with_depth: bool = True,
        ground_z_m: float = 0.0,
    ) -> np.ndarray:
        """Render a single-channel spatial conditioning mask (H, W) using robust 3D/2D clipping.

        Values range from 0.0 (background) to 1.0 (on-trajectory ribbon).
        Guarantees zero cross-screen streaks when segments pass behind camera.
        """
        assert camera_name in self.rig.cameras, f"Unknown camera: {camera_name}"
        cam = self.rig.cameras[camera_name]
        mask = np.zeros((self.height, self.width), dtype=np.float32)

        pts = np.asarray(waypoints_ego, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        n = pts.shape[0]
        if n < 2:
            return mask

        if pts.shape[1] == 2:
            p_ego = np.column_stack([pts[:, 0], pts[:, 1], np.full(n, ground_z_m, dtype=np.float64)])
        else:
            p_ego = pts[:, :3]

        t_cam = np.asarray(cam.position_m, dtype=np.float64)
        p_cam = (p_ego - t_cam[None, :]) @ cam.rotation_matrix_body_to_cam.T

        # Process each segment with 3D near-plane and 2D canvas clipping
        for i in range(n - 1):
            res = project_and_clip_segment(
                p_cam[i], p_cam[i + 1], cam, self.width, self.height, self.near_clip_m
            )
            if res is None:
                continue

            (u0, v0), (u1, v1), z_avg = res
            w_px = max(1, int(ribbon_width_px * (10.0 / (z_avg + 5.0))))
            intensity = float(max(0.2, min(1.0, 15.0 / (z_avg + 5.0)))) if decay_with_depth else 1.0

            pt0 = (int(round(u0)), int(round(v0)))
            pt1 = (int(round(u1)), int(round(v1)))
            cv2.line(mask, pt0, pt1, intensity, w_px, lineType=cv2.LINE_AA)

        return mask

    def render_static_route_mask(
        self,
        camera_name: str,
        waypoints_ego: np.ndarray,
        ribbon_width_px: int = 14,
        decay_with_depth: bool = True,
        channel_dim: bool = True,
    ) -> np.ndarray:
        """Render Option A: Static route mask representing the entire 5-second future corridor.

        Returns:
            np.ndarray: float32 mask with values in [0.0, 1.0].
                        Shape is (1, H, W) if channel_dim=True, else (H, W).
        """
        raw_mask = self.render_spatial_control_mask(
            camera_name=camera_name,
            waypoints_ego=waypoints_ego,
            ribbon_width_px=ribbon_width_px,
            decay_with_depth=decay_with_depth,
        )
        if channel_dim:
            return raw_mask[np.newaxis, ...]
        return raw_mask

    def render_initial_frame_temporal_masks(
        self,
        camera_name: str,
        waypoints_ego: np.ndarray,
        num_frames: int = spec.FRAME_COUNT,
        beacon_radius_px: int = 12,
        trail_width_px: int = 8,
        decay_with_depth: bool = True,
        channel_dim: bool = True,
    ) -> np.ndarray:
        """Render Option B.1: Route-Initial-Frame Temporal Masks.

        Evaluates the planned future trajectory from the stationary vantage point of the
        camera at t=0.0s. At each future frame t in {0..num_frames-1} (tau_t = (t+1)*0.2s):
          1. Accentuated beacon at future waypoint P(t) (intensity 1.0)
          2. Traversed history trail from 0 to t (intensity 0.70)
          3. Subtle future guide ribbon from t to end (intensity 0.20)

        Returns:
            np.ndarray: float32 tensor of shape (num_frames, 1, H, W) or (num_frames, H, W).
        """
        assert camera_name in self.rig.cameras, f"Unknown camera: {camera_name}"
        cam = self.rig.cameras[camera_name]

        pts_in = np.asarray(waypoints_ego, dtype=np.float64)
        if pts_in.shape[0] < num_frames:
            alphas = np.linspace(0.0, 1.0, num_frames)
            orig_alphas = np.linspace(0.0, 1.0, pts_in.shape[0])
            pts_ego = np.column_stack([
                np.interp(alphas, orig_alphas, pts_in[:, 0]),
                np.interp(alphas, orig_alphas, pts_in[:, 1]),
            ])
        else:
            pts_ego = pts_in[:num_frames]

        n = len(pts_ego)
        t_cam = np.asarray(cam.position_m, dtype=np.float64)
        p_ego_3d = np.column_stack([pts_ego[:, 0], pts_ego[:, 1], np.zeros(n, dtype=np.float64)])
        p_cam = (p_ego_3d - t_cam[None, :]) @ cam.rotation_matrix_body_to_cam.T

        proj = self.project_waypoints_to_camera(camera_name, pts_ego)
        pts_2d = proj.points_2d
        vis = proj.is_visible
        depths = proj.depths

        masks = np.zeros((num_frames, self.height, self.width), dtype=np.float32)

        beacon_radii = np.zeros(num_frames, dtype=int)
        for i in range(num_frames):
            z_val = max(float(depths[i]), 1.0)
            beacon_radii[i] = max(2, int(beacon_radius_px * (10.0 / (z_val + 5.0))))

        for t in range(num_frames):
            frame_mask = masks[t]

            # 1. Subtle future guidance ribbon (from t to num_frames-1)
            for i in range(t, num_frames - 1):
                res = project_and_clip_segment(
                    p_cam[i], p_cam[i + 1], cam, self.width, self.height, self.near_clip_m
                )
                if res is None:
                    continue
                (u0, v0), (u1, v1), z_avg = res
                w_px = max(1, int((trail_width_px // 2) * (10.0 / (z_avg + 5.0))))
                intensity = 0.20 if decay_with_depth else 0.25
                cv2.line(frame_mask, (int(round(u0)), int(round(v0))), (int(round(u1)), int(round(v1))), intensity, w_px, lineType=cv2.LINE_AA)

            # 2. Active traversed history trail (from 0 to t)
            for i in range(0, t):
                res = project_and_clip_segment(
                    p_cam[i], p_cam[i + 1], cam, self.width, self.height, self.near_clip_m
                )
                if res is None:
                    continue
                (u0, v0), (u1, v1), z_avg = res
                w_px = max(1, int(trail_width_px * (10.0 / (z_avg + 5.0))))
                intensity = float(max(0.3, min(0.75, 12.0 / (z_avg + 5.0)))) if decay_with_depth else 0.70
                cv2.line(frame_mask, (int(round(u0)), int(round(v0))), (int(round(u1)), int(round(v1))), intensity, w_px, lineType=cv2.LINE_AA)

            # 3. Active waypoint beacon at P(t) (intensity 1.0)
            if vis[t]:
                pt = pts_2d[t]
                px = int(round(pt[0]))
                py = int(round(pt[1]))
                r_b = beacon_radii[t]
                cv2.circle(frame_mask, (px, py), r_b, 1.0, -1, lineType=cv2.LINE_AA)

        if channel_dim:
            return masks[:, np.newaxis, :, :]
        return masks

    def render_future_camera_spatial_masks(
        self,
        camera_name: str,
        waypoints_ego: np.ndarray,
        num_frames: int = spec.FRAME_COUNT,
        ribbon_width_px: int = 14,
        decay_with_depth: bool = True,
        channel_dim: bool = True,
        extrapolate_horizon: bool = False,
        extrapolate_steps: int = 5,
    ) -> np.ndarray:
        """Render Option B.2: Future-Camera-Frame Spatial Conditioning (Ego-Motion Aligned).

        Reconstructs the ego pose (position and yaw heading) at every future timestep t in {0..24}.
        At future frame t, the vehicle is at world position P(t) with yaw psi(t). The upcoming
        path points k in {t..24} are transformed into the local vehicle frame at time t:
            P_ego(t) = R_z(-psi_t) * (P_world - P_ego(t))
        and then projected into the future camera frame:
            P_cam(t) = R_body_to_cam * (P_ego(t) - t_cam)
        This models what the camera actually observes as the vehicle traverses the scene.

        Args:
            camera_name: Target camera ('front', 'front_left', etc.).
            waypoints_ego: (N, 2) array of planned trajectory points.
            num_frames: Number of output temporal frames (default: 25).
            ribbon_width_px: Line width for road corridor rendering.
            decay_with_depth: If True, attenuates line intensity and thickness with distance.
            channel_dim: If True, returns (num_frames, 1, H, W); else (num_frames, H, W).
            extrapolate_horizon: If True, kinematically extrapolates trajectory forward beyond
                                 the final frame to prevent the end-of-horizon corridor vanishing artifact.
            extrapolate_steps: Number of forward kinematic steps to extrapolate (default: 5 steps = 1.0s).

        Returns:
            np.ndarray: float32 tensor of shape (num_frames, 1, H, W) or (num_frames, H, W).
        """
        assert camera_name in self.rig.cameras, f"Unknown camera: {camera_name}"
        cam = self.rig.cameras[camera_name]

        pts_in = np.asarray(waypoints_ego, dtype=np.float64)
        if pts_in.shape[0] < num_frames:
            alphas = np.linspace(0.0, 1.0, num_frames)
            orig_alphas = np.linspace(0.0, 1.0, pts_in.shape[0])
            pts_world = np.column_stack([
                np.interp(alphas, orig_alphas, pts_in[:, 0]),
                np.interp(alphas, orig_alphas, pts_in[:, 1]),
            ])
        else:
            pts_world = pts_in.copy()

        if extrapolate_horizon and extrapolate_steps > 0 and len(pts_world) >= 2:
            d_last = pts_world[-1] - pts_world[-2]
            extra_pts = [pts_world[-1] + (m + 1) * d_last for m in range(extrapolate_steps)]
            pts_world = np.vstack([pts_world, extra_pts])

        headings = compute_trajectory_headings(pts_world)
        masks = np.zeros((num_frames, self.height, self.width), dtype=np.float32)

        t_cam = np.asarray(cam.position_m, dtype=np.float64)
        r_cam = cam.rotation_matrix_body_to_cam

        for t in range(num_frames):
            frame_mask = masks[t]
            ego_pos = pts_world[t]
            ego_yaw = headings[t]

            cos_y = math.cos(ego_yaw)
            sin_y = math.sin(ego_yaw)

            # Transform upcoming waypoints (from t to end) into local ego frame at time t
            upcoming_pts_world = pts_world[t:]
            n_up = len(upcoming_pts_world)
            if n_up < 2:
                continue

            dx = upcoming_pts_world[:, 0] - ego_pos[0]
            dy = upcoming_pts_world[:, 1] - ego_pos[1]

            x_ego = cos_y * dx + sin_y * dy
            y_ego = -sin_y * dx + cos_y * dy
            z_ego = np.zeros(n_up, dtype=np.float64)

            p_ego_t = np.column_stack([x_ego, y_ego, z_ego])
            p_cam_t = (p_ego_t - t_cam[None, :]) @ r_cam.T

            # Rasterize clipped segments in the future camera frame
            for i in range(n_up - 1):
                res = project_and_clip_segment(
                    p_cam_t[i], p_cam_t[i + 1], cam, self.width, self.height, self.near_clip_m
                )
                if res is None:
                    continue

                (u0, v0), (u1, v1), z_avg = res
                w_px = max(1, int(ribbon_width_px * (10.0 / (z_avg + 5.0))))
                intensity = float(max(0.25, min(1.0, 15.0 / (z_avg + 5.0)))) if decay_with_depth else 1.0

                pt0 = (int(round(u0)), int(round(v0)))
                pt1 = (int(round(u1)), int(round(v1)))
                cv2.line(frame_mask, pt0, pt1, intensity, w_px, lineType=cv2.LINE_AA)

        if channel_dim:
            return masks[:, np.newaxis, :, :]
        return masks

    def render_time_indexed_spatial_masks(
        self,
        camera_name: str,
        waypoints_ego: np.ndarray,
        num_frames: int = spec.FRAME_COUNT,
        mode: str = "future_camera",
        **kwargs,
    ) -> np.ndarray:
        """Unified interface for Option B time-indexed spatial masks.

        Args:
            camera_name: Target camera ('front', 'front_left', etc.).
            waypoints_ego: (N, 2) or (N, 3) trajectory coordinates.
            num_frames: Number of temporal frames (default: 25).
            mode: 'future_camera' (Option B.2, ego-motion aligned) or
                  'initial_frame' (Option B.1, stationary t=0 frame).

        Returns:
            np.ndarray: float32 tensor of shape (num_frames, 1, H, W).
        """
        if mode == "future_camera":
            return self.render_future_camera_spatial_masks(
                camera_name, waypoints_ego, num_frames=num_frames, **kwargs
            )
        elif mode == "initial_frame":
            return self.render_initial_frame_temporal_masks(
                camera_name, waypoints_ego, num_frames=num_frames, **kwargs
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'future_camera' or 'initial_frame'.")

    def render_static_route_masks_all_cameras(
        self,
        waypoints_ego: np.ndarray,
        ribbon_width_px: int = 14,
        decay_with_depth: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Render static route masks across all 6 surround cameras simultaneously."""
        return {
            cam: self.render_static_route_mask(
                cam,
                waypoints_ego,
                ribbon_width_px=ribbon_width_px,
                decay_with_depth=decay_with_depth,
                channel_dim=True,
            )
            for cam in spec.CAMERAS
        }

    def render_time_indexed_spatial_masks_all_cameras(
        self,
        waypoints_ego: np.ndarray,
        num_frames: int = spec.FRAME_COUNT,
        mode: str = "future_camera",
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Render time-indexed spatial masks across all 6 surround cameras simultaneously."""
        return {
            cam: self.render_time_indexed_spatial_masks(
                cam,
                waypoints_ego,
                num_frames=num_frames,
                mode=mode,
                **kwargs,
            )
            for cam in spec.CAMERAS
        }

    def batch_static_route_masks(
        self,
        trajectories: Sequence[np.ndarray],
        camera_name: str = "front",
        ribbon_width_px: int = 14,
    ) -> np.ndarray:
        """Render a batch of static route masks for a specified camera: [B, 1, H, W]."""
        b = len(trajectories)
        batch = np.zeros((b, 1, self.height, self.width), dtype=np.float32)
        for i, traj in enumerate(trajectories):
            batch[i] = self.render_static_route_mask(
                camera_name, traj, ribbon_width_px=ribbon_width_px, channel_dim=True
            )
        return batch

    def batch_time_indexed_spatial_masks(
        self,
        trajectories: Sequence[np.ndarray],
        camera_name: str = "front",
        num_frames: int = spec.FRAME_COUNT,
        mode: str = "future_camera",
    ) -> np.ndarray:
        """Render a batch of time-indexed spatial masks for a specified camera: [B, num_frames, 1, H, W]."""
        b = len(trajectories)
        batch = np.zeros((b, num_frames, 1, self.height, self.width), dtype=np.float32)
        for i, traj in enumerate(trajectories):
            batch[i] = self.render_time_indexed_spatial_masks(
                camera_name, traj, num_frames=num_frames, mode=mode, channel_dim=True
            )
        return batch
