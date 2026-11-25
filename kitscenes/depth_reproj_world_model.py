"""Full-FoV 3D Metric-Depth Cross-Camera & Multi-Frame Reprojection World Model.

Combines:
1. Verified official 6-camera calibration (K_source, R, t, nominal_ground_z=-1.60m) from
   ``configs/kitscenes_camera_calibration.json``.
2. Closed-form analytical backward homographies (road ground-plane + sky horizon blend)
   from native/supersampled sensor images across:
   - Primary camera at t=0
   - Adjacent 360-degree surround cameras (~27 deg FoV overlap, exposure-matched)
   - Past context frames (t = -0.2s, -0.4s, -1.0s, -2.0s) along past odometry
   Guaranteeing zero forward-splat grid dots and subpixel sharp road markings.
3. Optional 3D foreground object parallax layer (via Depth-Anything-V2 calibrated to
   road geometry) rendered densely over the analytical road base layer.
4. Ego-vehicle body anchoring (BMW 7 Series hood/roof/rack detected via metric 3D
   footprint + moving-frame temporal invariance).
5. Photographic hole completion (GPU LaMa when available; perspective-aware asphalt/
   OpenCV inpaint fallback cleanly avoiding edge streaking).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from kitscenes import spec
from kitscenes.video_engine import CameraRig

CAMERA_NEIGHBORS: Dict[str, Tuple[str, ...]] = {
    "front": ("front_left", "front_right"),
    "front_left": ("front", "rear_left"),
    "front_right": ("front", "rear_right"),
    "rear_left": ("front_left", "rear"),
    "rear_right": ("front_right", "rear"),
    "rear": ("rear_left", "rear_right"),
}


def compute_smooth_headings(trajectory: Sequence[Sequence[float]]) -> np.ndarray:
    """Estimate smooth per-step ego yaw headings (rad) from 25 [x, y] future waypoints."""
    pts = np.asarray(trajectory, dtype=np.float64)[:, :2]
    n = len(pts)
    pts_full = np.vstack([np.zeros((1, 2), dtype=np.float64), pts])
    headings = np.zeros(n, dtype=np.float64)
    cur_yaw = 0.0
    for i in range(n):
        p_prev = pts_full[max(0, i - 1)]
        p_next = pts_full[min(n, i + 2)]
        dp = p_next - p_prev
        if float(np.linalg.norm(dp)) > 0.08:
            cur_yaw = math.atan2(float(dp[1]), float(dp[0]))
        else:
            step_dp = pts_full[i + 1] - pts_full[i]
            if float(np.linalg.norm(step_dp)) > 1e-3:
                cur_yaw = math.atan2(float(step_dp[1]), float(step_dp[0]))
        headings[i] = cur_yaw
    return headings


def compute_past_poses(past_trajectory: Sequence[Sequence[float]]) -> Dict[str, Tuple[float, float, float]]:
    """Estimate ego poses (x, y, yaw) at t=-2.0s, -1.0s, -0.4s, -0.2s in the t=0 frame."""
    if past_trajectory is None or len(past_trajectory) < 2:
        return {}
    pts = np.asarray(past_trajectory, dtype=np.float64)[:, :2]
    origin = pts[-1].copy()
    rel = pts - origin
    heading0 = 0.0
    if len(rel) >= 2:
        dp0 = rel[-1] - rel[max(0, len(rel) - 4)]
        if float(np.linalg.norm(dp0)) > 0.05:
            heading0 = math.atan2(float(dp0[1]), float(dp0[0]))
    c0, s0 = math.cos(-heading0), math.sin(-heading0)
    r_align = np.array([[c0, -s0], [s0, c0]], dtype=np.float64)
    aligned = rel @ r_align.T

    def _pose_back(steps_back: int) -> Optional[Tuple[float, float, float]]:
        idx = len(aligned) - 1 - steps_back
        if idx < 0 or idx >= len(aligned):
            return None
        p_cur = aligned[idx]
        p_fwd = aligned[min(len(aligned) - 1, idx + 2)]
        dp = p_fwd - p_cur
        yaw = math.atan2(float(dp[1]), float(dp[0])) if float(np.linalg.norm(dp)) > 0.05 else 0.0
        return float(p_cur[0]), float(p_cur[1]), float(yaw)

    offsets = {"tm40": 20, "tm30": 15, "tm20": 10, "tm10": 5, "tm04": 2, "tm02": 1}
    return {tag: pose for tag, sb in offsets.items() if (pose := _pose_back(sb)) is not None}


def scale_intrinsics(
    k_source: np.ndarray,
    target_w: int,
    target_h: int,
    ref_w: int = 3488,
    ref_h: int = 2272,
) -> np.ndarray:
    """Scale 3488x2272 pinhole intrinsics K_source to (target_w, target_h)."""
    sx = float(target_w) / float(ref_w)
    sy = float(target_h) / float(ref_h)
    a = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return a @ np.asarray(k_source, dtype=np.float64)


def compute_analytical_ground_depth(
    k_mat: np.ndarray,
    r_mat: np.ndarray,
    t_vec: np.ndarray,
    width: int,
    height: int,
    ground_z_world: float = -1.60,
    min_depth: float = 1.5,
    max_depth: float = 220.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact ray directions in world frame and road-plane intersection depth."""
    k_inv = np.linalg.inv(k_mat)
    cam_origin = -r_mat.T @ t_vec
    us = np.arange(width, dtype=np.float64)
    vs = np.arange(height, dtype=np.float64)
    grid_u, grid_v = np.meshgrid(us, vs)
    pix = np.stack([grid_u, grid_v, np.ones_like(grid_u)], axis=-1)

    rays_cam = pix @ k_inv.T
    rays_world = rays_cam @ r_mat

    rz = rays_world[..., 2]
    road_hit = rz < -0.020
    denom = np.where(road_hit, rz, -1.0)
    lam = (ground_z_world - float(cam_origin[2])) / denom
    road_valid = road_hit & (lam >= min_depth) & (lam <= 85.0)
    depth_map = np.where(road_valid, np.clip(lam, min_depth, max_depth), max_depth)
    return depth_map.astype(np.float32), road_valid, rays_world.astype(np.float32)


def calibrate_depth_to_road(
    raw_depth: np.ndarray,
    road_depth: np.ndarray,
    road_valid: np.ndarray,
    is_disparity: bool = True,
    min_depth: float = 0.8,
    max_depth: float = 220.0,
) -> np.ndarray:
    """Align monocular depth/disparity map to KITScenes metric road geometry."""
    h, w = road_depth.shape
    raw = np.asarray(raw_depth, dtype=np.float64)
    if raw.shape != (h, w):
        raw = cv2.resize(raw.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float64)

    roi = np.zeros((h, w), dtype=bool)
    roi[int(0.62 * h):int(0.84 * h), int(0.28 * w):int(0.72 * w)] = True
    fit_mask = roi & road_valid & (road_depth >= 3.5) & (road_depth <= 28.0) & np.isfinite(raw) & (raw > 1e-4)
    if int(fit_mask.sum()) < 50:
        return road_depth.astype(np.float32)

    target_z = road_depth[fit_mask].astype(np.float64)
    pred_vals = raw[fit_mask]

    if is_disparity:
        target_inv = 1.0 / target_z
        a_mat = np.column_stack([pred_vals, np.ones_like(pred_vals)])
        coef, _, _, _ = np.linalg.lstsq(a_mat, target_inv, rcond=None)
        resid = np.abs(a_mat @ coef - target_inv)
        keep = resid <= np.percentile(resid, 75.0) + 1e-6
        if int(keep.sum()) >= 20:
            coef, _, _, _ = np.linalg.lstsq(a_mat[keep], target_inv[keep], rcond=None)
        slope, intercept = float(coef[0]), float(coef[1])
        if slope <= 1e-5:
            scale = float(np.median(target_inv / np.maximum(pred_vals, 1e-4)))
            inv_z = np.maximum(raw, 1e-4) * scale
        else:
            inv_z = np.maximum(slope * raw + intercept, 1.0 / max_depth)
        z_cal = 1.0 / inv_z
    else:
        ratios = target_z / np.maximum(pred_vals, 0.2)
        scale = float(np.clip(np.median(ratios), 0.35, 2.8))
        z_cal = raw * scale

    z_cal = np.clip(z_cal, min_depth, max_depth)
    rel_err = np.abs(z_cal - road_depth) / np.maximum(road_depth, 1.0)
    flat_road = road_valid & (road_depth <= 35.0) & (rel_err < 0.25)
    blend_w = np.where(flat_road, np.clip(1.0 - rel_err / 0.25, 0.0, 1.0) * 0.85, 0.0)
    z_out = blend_w * road_depth + (1.0 - blend_w) * z_cal
    return np.clip(z_out, min_depth, max_depth).astype(np.float32)


def compute_ego_body_mask(
    cam_name: str,
    depth_map: Optional[np.ndarray],
    k_mat: np.ndarray,
    r_mat: np.ndarray,
    t_vec: np.ndarray,
    height: int,
    width: int,
    t0_rgb: Optional[np.ndarray] = None,
    past_rgb_tm04: Optional[np.ndarray] = None,
    ego_speed_mps: float = 0.0,
) -> np.ndarray:
    """Detect pixels bolted to the ego BMW 7 Series body (hood, roof rack, trunk)."""
    mask = np.zeros((height, width), dtype=np.uint8)

    if depth_map is not None:
        k_inv = np.linalg.inv(k_mat)
        us = np.arange(width, dtype=np.float32)
        vs = np.arange(height, dtype=np.float32)
        gu, gv = np.meshgrid(us, vs)
        pix = np.stack([gu, gv, np.ones_like(gu)], axis=-1)
        pts_cam = (pix @ k_inv.T.astype(np.float32)) * depth_map[..., None]
        pts_ego = (pts_cam - t_vec.astype(np.float32)) @ r_mat.astype(np.float32)
        x, y, z = pts_ego[..., 0], pts_ego[..., 1], pts_ego[..., 2]
        in_box = (x >= -2.65) & (x <= 2.40) & (np.abs(y) <= 1.08) & (z >= -1.35) & (z <= 0.10)
        rows = np.arange(height, dtype=np.float32)[:, None]
        bottom_gate = rows >= (0.76 * height if cam_name in ("front", "rear") else 0.72 * height)
        mask = np.maximum(mask, (in_box & bottom_gate).astype(np.uint8))

    # Empirical temporal invariance check when moving: ego body doesn't move relative to sensor
    if t0_rgb is not None and past_rgb_tm04 is not None and ego_speed_mps > 2.5:
        t0_s = cv2.resize(t0_rgb, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
        p_s = cv2.resize(past_rgb_tm04, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
        diff = np.mean(np.abs(t0_s - p_s), axis=-1)
        rows = np.arange(height, dtype=np.float32)[:, None]
        bottom_strip = rows >= (0.80 * height)
        static_body = (diff < 6.0) & bottom_strip
        mask = np.maximum(mask, static_body.astype(np.uint8))

    # Conservative physical camera-rig corner guard for roof mount corners
    ys, xs = np.mgrid[0:height, 0:width]
    if cam_name == "front":
        # Central silver hood contour at bottom center
        hood = (ys >= 0.85 * height) & (xs >= 0.20 * width) & (xs <= 0.80 * width)
        mask = np.maximum(mask, hood.astype(np.uint8))
    elif cam_name == "front_left":
        corner = (ys >= 0.86 * height) & ((xs - 0.48 * width) / (0.52 * width) + (ys - 0.86 * height) / (0.14 * height) > 1.15)
        mask = np.maximum(mask, corner.astype(np.uint8))
    elif cam_name == "front_right":
        corner = (ys >= 0.86 * height) & ((0.48 * width - xs) / (0.48 * width) + (ys - 0.86 * height) / (0.14 * height) > 1.15)
        mask = np.maximum(mask, corner.astype(np.uint8))
    elif cam_name == "rear_left":
        corner = (ys >= 0.86 * height) & ((0.55 * width - xs) / (0.55 * width) + (ys - 0.86 * height) / (0.14 * height) > 1.15)
        mask = np.maximum(mask, corner.astype(np.uint8))
    elif cam_name == "rear_right":
        corner = (ys >= 0.86 * height) & ((xs - 0.52 * width) / (0.48 * width) + (ys - 0.86 * height) / (0.14 * height) > 1.15)
        mask = np.maximum(mask, corner.astype(np.uint8))
    elif cam_name == "rear":
        rack = (ys >= 0.87 * height) & (xs <= 0.58 * width)
        mask = np.maximum(mask, rack.astype(np.uint8))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(bool)


def _ego_world_rt(pose: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (R_ego, T_ego) mapping ego frame -> world (t=0) frame."""
    ex, ey, eyaw = pose
    c_y, s_y = math.cos(eyaw), math.sin(eyaw)
    r_e = np.array([[c_y, -s_y, 0.0], [s_y, c_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    t_e = np.array([ex, ey, 0.0], dtype=np.float64)
    return r_e, t_e


def compute_plane_homography_backward(
    k_src: np.ndarray,
    r_src: np.ndarray,
    t_src: np.ndarray,
    pose_src: Tuple[float, float, float],
    k_dst: np.ndarray,
    r_dst: np.ndarray,
    t_dst: np.ndarray,
    pose_dst: Tuple[float, float, float],
    ground_z_world: float = -1.60,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact backward homographies (H_ground, H_sky) mapping dst pixels -> src pixels."""
    re_s, te_s = _ego_world_rt(pose_src)
    re_t, te_t = _ego_world_rt(pose_dst)

    r_sw = r_src @ re_s.T
    t_sw = t_src - r_sw @ te_s

    r_tw = r_dst @ re_t.T
    t_tw = t_dst - r_tw @ te_t

    r_rel = r_sw @ r_tw.T
    t_rel = t_sw - r_rel @ t_tw

    n_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n_dst = r_tw @ n_world
    d_dst = float(ground_z_world + np.dot(n_dst, t_tw))
    if abs(d_dst) < 1e-4:
        d_dst = -1.414

    k_dst_inv = np.linalg.inv(k_dst)
    h_ground = k_src @ (r_rel + np.outer(t_rel, n_dst) / d_dst) @ k_dst_inv
    h_sky = k_src @ r_rel @ k_dst_inv
    return h_ground, h_sky


class Depth3DFullFoVWorldModel:
    """Calibrated Full-FoV Analytical + 3D Depth & Cross-Camera Reprojection World Model."""

    def __init__(
        self,
        rig: Optional[CameraRig] = None,
        device: str = "cpu",
        render_scale: int = 2,
        depth_fn: Optional[Callable[[Dict[str, np.ndarray]], Dict[str, Tuple[np.ndarray, bool]]]] = None,
        inpaint_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        crf_sharpen: float = 0.0,
    ) -> None:
        self.rig = rig or CameraRig(width=spec.WIDTH, height=spec.HEIGHT, is_provisional=False)
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.render_scale = max(1, int(render_scale))
        self.render_w = spec.WIDTH * self.render_scale
        self.render_h = spec.HEIGHT * self.render_scale
        self.depth_fn = depth_fn
        self.inpaint_fn = inpaint_fn
        self.crf_sharpen = float(crf_sharpen)

        self.k_render: Dict[str, np.ndarray] = {}
        self.r_mats: Dict[str, np.ndarray] = {}
        self.t_vecs: Dict[str, np.ndarray] = {}
        self.road_depths: Dict[str, np.ndarray] = {}
        self.road_masks: Dict[str, np.ndarray] = {}
        self.sky_weights: Dict[str, torch.Tensor] = {}
        self.ground_z = float(getattr(self.rig, "nominal_ground_z_ego", -1.60))

        # Shared destination pixel grid at render resolution
        us = torch.arange(self.render_w, dtype=torch.float32, device=self.device)
        vs = torch.arange(self.render_h, dtype=torch.float32, device=self.device)
        gv, gu = torch.meshgrid(vs, us, indexing="ij")
        self._dst_pix = torch.stack([gu, gv, torch.ones_like(gu)], dim=-1)  # (H, W, 3)

        for cam in spec.CAMERAS:
            pose = self.rig[cam]
            k_src = pose.k_source_matrix if pose.k_source_matrix is not None else scale_intrinsics(pose.k_matrix, 3488, 2272, spec.WIDTH, spec.HEIGHT)
            r_mat = pose.r_matrix if pose.r_matrix is not None else np.eye(3, dtype=np.float64)
            t_vec = pose.t_vec if pose.t_vec is not None else np.zeros(3, dtype=np.float64)
            k_ren = scale_intrinsics(k_src, self.render_w, self.render_h)

            self.k_render[cam] = k_ren
            self.r_mats[cam] = np.asarray(r_mat, dtype=np.float64)
            self.t_vecs[cam] = np.asarray(t_vec, dtype=np.float64)

            d_road, m_road, rays_w = compute_analytical_ground_depth(
                k_ren, self.r_mats[cam], self.t_vecs[cam], self.render_w, self.render_h, ground_z_world=self.ground_z
            )
            self.road_depths[cam] = d_road
            self.road_masks[cam] = m_road

            # Smooth vertical sky weight based on exact calibrated world ray pitch rz
            rz = torch.from_numpy(rays_w[..., 2]).to(self.device)
            # Transition smoothly around horizon rz in [-0.055, -0.005]
            sky_w = torch.clamp((rz - (-0.055)) / 0.050, 0.0, 1.0)
            self.sky_weights[cam] = sky_w

    def _to_tensor_img(self, img_rgb: np.ndarray) -> torch.Tensor:
        arr = np.asarray(img_rgb, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _warp_backward_analytical(
        self,
        src_cam: str,
        src_pose: Tuple[float, float, float],
        dst_cam: str,
        dst_pose: Tuple[float, float, float],
        src_img_hi: torch.Tensor,
        src_ego_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact closed-form backward homography sample (road + sky blend) at render resolution."""
        k_s = self.k_render[src_cam]
        k_d = self.k_render[dst_cam]
        h_g, h_sky = compute_plane_homography_backward(
            k_s, self.r_mats[src_cam], self.t_vecs[src_cam], src_pose,
            k_d, self.r_mats[dst_cam], self.t_vecs[dst_cam], dst_pose,
            ground_z_world=self.ground_z,
        )

        hg_t = torch.tensor(h_g, dtype=torch.float32, device=self.device)
        hs_t = torch.tensor(h_sky, dtype=torch.float32, device=self.device)

        pg = self._dst_pix @ hg_t.T
        ps = self._dst_pix @ hs_t.T

        # Points must have positive homogeneous coordinate w > 0 to lie in front of source camera
        wg = pg[..., 2]
        ws = ps[..., 2]
        front_g = wg > 1e-4
        front_s = ws > 1e-4

        ug = pg[..., 0] / torch.where(front_g, wg, torch.ones_like(wg))
        vg = pg[..., 1] / torch.where(front_g, wg, torch.ones_like(wg))
        us = ps[..., 0] / torch.where(front_s, ws, torch.ones_like(ws))
        vs = ps[..., 1] / torch.where(front_s, ws, torch.ones_like(ws))

        sw = self.sky_weights[dst_cam]
        u_src = sw * us + (1.0 - sw) * ug
        v_src = sw * vs + (1.0 - sw) * vg
        in_front = (sw > 0.5) & front_s | (sw <= 0.5) & front_g

        w_r, h_r = self.render_w, self.render_h
        in_bounds = (
            in_front
            & (u_src >= 1.0)
            & (u_src <= w_r - 2.0)
            & (v_src >= 1.0)
            & (v_src <= h_r - 2.0)
        )

        u_norm = (u_src / max(w_r - 1, 1)) * 2.0 - 1.0
        v_norm = (v_src / max(h_r - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)

        # Reject sampling from the source camera's bolted ego car body!
        hit_ego = (
            F.grid_sample(
                src_ego_mask.float().unsqueeze(0).unsqueeze(0),
                grid,
                mode="nearest",
                padding_mode="zeros",
                align_corners=True,
            )[0, 0]
            > 0.5
        )
        cov = in_bounds & (~hit_ego)

        sampled = F.grid_sample(
            src_img_hi,
            grid,
            mode="bicubic",
            padding_mode="zeros",
            align_corners=True,
        )[0].clamp(0.0, 1.0)
        return sampled, cov

    def _render_foreground_3d_overlay(
        self,
        pts_world: torch.Tensor,
        uv_norm: torch.Tensor,
        fg_mask_flat: torch.Tensor,
        dst_cam: str,
        dst_pose: Tuple[float, float, float],
        src_img_hi: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dense Z-buffered overlay ONLY for elevated 3D foreground objects (cars, poles, buildings)."""
        h_r, w_r = self.render_h, self.render_w
        if not bool(fg_mask_flat.any()):
            return (
                torch.zeros((3, h_r, w_r), dtype=torch.float32, device=self.device),
                torch.zeros((h_r, w_r), dtype=torch.bool, device=self.device),
                torch.full((h_r, w_r), 1e9, dtype=torch.float32, device=self.device),
            )

        re_t, te_t = _ego_world_rt(dst_pose)
        r_e_t = torch.tensor(re_t.T, dtype=torch.float32, device=self.device)
        t_e = torch.tensor(te_t, dtype=torch.float32, device=self.device)
        r_cam = torch.tensor(self.r_mats[dst_cam], dtype=torch.float32, device=self.device)
        t_cam = torch.tensor(self.t_vecs[dst_cam], dtype=torch.float32, device=self.device)
        k_out = torch.tensor(self.k_render[dst_cam], dtype=torch.float32, device=self.device)

        p_fg = pts_world[fg_mask_flat]
        uv_fg = uv_norm[fg_mask_flat]
        p_cam = ((p_fg - t_e) @ r_e_t.T) @ r_cam.T + t_cam
        z_dst = p_cam[:, 2]
        valid = (z_dst > 0.5) & (z_dst < 65.0)
        if not bool(valid.any()):
            return (
                torch.zeros((3, h_r, w_r), dtype=torch.float32, device=self.device),
                torch.zeros((h_r, w_r), dtype=torch.bool, device=self.device),
                torch.full((h_r, w_r), 1e9, dtype=torch.float32, device=self.device),
            )

        p_cam, z_dst, uv_fg = p_cam[valid], z_dst[valid], uv_fg[valid]
        proj = p_cam @ k_out.T
        u_dst = proj[:, 0] / z_dst
        v_dst = proj[:, 1] / z_dst
        in_b = (u_dst >= -0.5) & (u_dst <= w_r - 0.5) & (v_dst >= -0.5) & (v_dst <= h_r - 0.5)
        u_dst, v_dst, z_dst, uv_fg = u_dst[in_b], v_dst[in_b], z_dst[in_b], uv_fg[in_b]
        if u_dst.shape[0] == 0:
            return (
                torch.zeros((3, h_r, w_r), dtype=torch.float32, device=self.device),
                torch.zeros((h_r, w_r), dtype=torch.bool, device=self.device),
                torch.full((h_r, w_r), 1e9, dtype=torch.float32, device=self.device),
            )

        u0 = torch.floor(u_dst).long()
        v0 = torch.floor(v_dst).long()
        dx = torch.clamp(u_dst - u0.float(), 0.0, 1.0)
        dy = torch.clamp(v_dst - v0.float(), 0.0, 1.0)
        idx_l, z_l, w_l, uv_l = [], [], [], []
        for du, dv, wf in (
            (0, 0, (1.0 - dx) * (1.0 - dy)),
            (1, 0, dx * (1.0 - dy)),
            (0, 1, (1.0 - dx) * dy),
            (1, 1, dx * dy),
        ):
            uu, vv = u0 + du, v0 + dv
            ok = (uu >= 0) & (uu < w_r) & (vv >= 0) & (vv < h_r)
            if bool(ok.any()):
                idx_l.append(vv[ok] * w_r + uu[ok])
                z_l.append(z_dst[ok])
                w_l.append(wf[ok] + 1e-4)
                uv_l.append(uv_fg[ok])

        idx_cat = torch.cat(idx_l)
        z_cat = torch.cat(z_l)
        n_pix = h_r * w_r
        z_buf = torch.full((n_pix,), 1e9, dtype=torch.float32, device=self.device)
        z_buf.scatter_reduce_(0, idx_cat, z_cat, reduce="amin", include_self=True)
        min_z = z_buf[idx_cat]
        front_surf = (z_cat - min_z) <= (0.06 * min_z + 0.06)

        w_cat = torch.cat(w_l) * front_surf.float()
        uv_cat = torch.cat(uv_l, dim=0)
        bu = torch.zeros((n_pix,), dtype=torch.float32, device=self.device)
        bv = torch.zeros((n_pix,), dtype=torch.float32, device=self.device)
        bw = torch.zeros((n_pix,), dtype=torch.float32, device=self.device)
        bu.scatter_add_(0, idx_cat, uv_cat[:, 0] * w_cat)
        bv.scatter_add_(0, idx_cat, uv_cat[:, 1] * w_cat)
        bw.scatter_add_(0, idx_cat, w_cat)

        cov = (bw > 1e-5).view(h_r, w_r)
        gu_t = torch.where(cov, (bu / torch.clamp(bw, min=1e-5)).view(h_r, w_r), torch.zeros((h_r, w_r), device=self.device))
        gv_t = torch.where(cov, (bv / torch.clamp(bw, min=1e-5)).view(h_r, w_r), torch.zeros((h_r, w_r), device=self.device))
        grid = torch.stack([gu_t, gv_t], dim=-1).unsqueeze(0)

        # Fill small 1-2px gaps inside foreground objects with continuous depth
        cov_f = cov.float().unsqueeze(0).unsqueeze(0)
        nbr = F.avg_pool2d(cov_f, 3, stride=1, padding=1)[0, 0] * 9.0
        fill = (~cov) & (nbr >= 5.0)
        if bool(fill.any()):
            uv_fill = (F.avg_pool2d(grid.permute(0, 3, 1, 2) * cov_f, 3, stride=1, padding=1) * 9.0 / torch.clamp(nbr, min=1e-5)).permute(0, 2, 3, 1)
            grid = torch.where(fill.unsqueeze(0).unsqueeze(-1), uv_fill, grid)
            cov = cov | fill

        sampled = F.grid_sample(src_img_hi, grid, mode="bicubic", padding_mode="border", align_corners=True)[0].clamp(0.0, 1.0)
        return sampled, cov, z_buf.view(h_r, w_r)

    @staticmethod
    def _match_and_composite(
        base_rgb: torch.Tensor,
        base_mask: torch.Tensor,
        fill_rgb: torch.Tensor,
        fill_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Match exposure across camera overlap and feather-fill missing pixels."""
        overlap = base_mask & fill_mask
        fill_adj = fill_rgb
        if int(overlap.sum()) >= 200:
            b_ov = base_rgb[:, overlap]
            f_ov = fill_rgb[:, overlap]
            mean_b = b_ov.mean(dim=1, keepdim=True)
            mean_f = f_ov.mean(dim=1, keepdim=True)
            std_b = b_ov.std(dim=1, keepdim=True).clamp(min=1e-3)
            std_f = f_ov.std(dim=1, keepdim=True).clamp(min=1e-3)
            gain = (std_b / std_f).clamp(0.75, 1.35)
            fill_adj = ((fill_rgb.view(3, -1) - mean_f) * gain + mean_b).view_as(fill_rgb).clamp(0.0, 1.0)

        need = (~base_mask) & fill_mask
        # Soft feather right at the seam boundary (3px blur on mask edge)
        if bool(need.any()) and bool(base_mask.any()):
            need_f = need.float().unsqueeze(0).unsqueeze(0)
            smooth_w = F.avg_pool2d(need_f, 5, stride=1, padding=2)[0] * (base_mask | fill_mask).float().unsqueeze(0)
            w_fill = torch.where(need.unsqueeze(0), smooth_w.clamp(min=0.5), torch.zeros_like(smooth_w))
            w_fill = torch.where((~base_mask).unsqueeze(0) & fill_mask.unsqueeze(0), torch.ones_like(w_fill), w_fill)
            out_rgb = w_fill * fill_adj + (1.0 - w_fill) * base_rgb
        else:
            out_rgb = torch.where(need.unsqueeze(0), fill_adj, base_rgb)
        return out_rgb, (base_mask | fill_mask)

    def _fill_residual_holes(self, img_rgb_u8: np.ndarray, hole_mask: np.ndarray) -> np.ndarray:
        """Complete any remaining unseen border/disocclusion holes cleanly without streaks."""
        if not np.any(hole_mask):
            return img_rgb_u8
        if self.inpaint_fn is not None:
            return self.inpaint_fn(img_rgb_u8, hole_mask)

        out = img_rgb_u8.copy()
        h, w = hole_mask.shape
        valid = ~hole_mask
        ys_valid = np.where(valid.any(axis=1))[0]
        if len(ys_valid) == 0:
            return out

        # Clamp vertical mirror sources strictly below the horizon line for ground holes
        horizon_y = int(0.54 * h)
        road_valid_rows = ys_valid[ys_valid >= horizon_y]

        for y in range(h):
            if not hole_mask[y].any():
                continue
            if valid[y].any():
                xs_v = np.where(valid[y])[0]
                xs_h = np.where(hole_mask[y])[0]
                nearest_x = np.clip(xs_v[np.argmin(np.abs(xs_v[:, None] - xs_h[None, :]), axis=0)], 0, w - 1)
                out[y, xs_h] = out[y, nearest_x]
            elif y >= horizon_y and len(road_valid_rows) > 0:
                # Sample from local valid road band only (never reflect buildings/sky!)
                band_lo = int(road_valid_rows[max(0, len(road_valid_rows) - max(12, len(road_valid_rows) // 3))])
                band_hi = int(road_valid_rows[-1])
                if band_hi > band_lo:
                    y_src = band_lo + ((y - band_hi - 1) % max(1, band_hi - band_lo))
                else:
                    y_src = band_hi
                out[y] = out[y_src]
            else:
                y_ref = int(ys_valid[np.argmin(np.abs(ys_valid - y))])
                out[y] = out[y_ref]

        seam = cv2.dilate(hole_mask.astype(np.uint8), np.ones((9, 9), np.uint8)) - cv2.erode(
            hole_mask.astype(np.uint8), np.ones((3, 3), np.uint8)
        )
        seam = np.clip(seam, 0, 1) * hole_mask.astype(np.uint8)
        if np.any(seam):
            bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            bgr = cv2.inpaint(bgr, (seam * 255).astype(np.uint8), inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return out

    def generate_scenario(
        self,
        scenario_id: str,
        trajectory: Sequence[Sequence[float]],
        conditioning_images: Dict[str, Union[Path, np.ndarray]],
        past_images: Optional[Dict[Tuple[str, str], Union[Path, np.ndarray]]] = None,
        past_trajectory: Optional[Sequence[Sequence[float]]] = None,
        cameras: Optional[Sequence[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Generate 25-frame H.264-ready RGB sequences for requested cameras."""
        target_cameras = list(cameras) if cameras is not None else list(spec.CAMERAS)

        t0_hi_np: Dict[str, np.ndarray] = {}
        t0_hi_t: Dict[str, torch.Tensor] = {}
        t0_ren_np: Dict[str, np.ndarray] = {}
        for cam in spec.CAMERAS:
            if cam not in conditioning_images:
                raise FileNotFoundError(f"Missing t=0 image for {scenario_id}/{cam}")
            src = conditioning_images[cam]
            if isinstance(src, (str, Path)):
                bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise FileNotFoundError(f"Failed to read {src}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                rgb = np.asarray(src, dtype=np.uint8)
            t0_hi_np[cam] = rgb
            t0_hi_t[cam] = self._to_tensor_img(rgb)
            if rgb.shape[:2] != (self.render_h, self.render_w):
                t0_ren_np[cam] = cv2.resize(rgb, (self.render_w, self.render_h), interpolation=cv2.INTER_AREA)
            else:
                t0_ren_np[cam] = rgb

        # Estimate metric depth (optional foreground 3D overlay) & ego car body masks
        raw_depths = self.depth_fn(t0_ren_np) if self.depth_fn is not None else {}
        cal_depths: Dict[str, np.ndarray] = {}
        ego_masks: Dict[str, np.ndarray] = {}
        ego_masks_t: Dict[str, torch.Tensor] = {}
        fg_world: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        # Rough speed estimate from planned trajectory for empirical hood detection
        pts = np.asarray(trajectory, dtype=np.float64)[:, :2]
        ego_speed = float(np.linalg.norm(pts[-1] - pts[0]) / 4.8) if len(pts) >= 2 else 0.0

        for cam in spec.CAMERAS:
            d_cal: Optional[np.ndarray] = None
            if cam in raw_depths:
                raw_d, is_disp = raw_depths[cam]
                d_cal = calibrate_depth_to_road(
                    raw_d, self.road_depths[cam], self.road_masks[cam], is_disparity=is_disp
                )
                cal_depths[cam] = d_cal

            p_tm04 = None
            if past_images and (cam, "tm04") in past_images:
                ps = past_images[(cam, "tm04")]
                if isinstance(ps, (str, Path)):
                    pb = cv2.imread(str(ps), cv2.IMREAD_COLOR)
                    if pb is not None:
                        p_tm04 = cv2.cvtColor(pb, cv2.COLOR_BGR2RGB)
                else:
                    p_tm04 = np.asarray(ps, dtype=np.uint8)

            m_ego = compute_ego_body_mask(
                cam,
                d_cal,
                self.k_render[cam],
                self.r_mats[cam],
                self.t_vecs[cam],
                self.render_h,
                self.render_w,
                t0_rgb=t0_ren_np[cam],
                past_rgb_tm04=p_tm04,
                ego_speed_mps=ego_speed,
            )
            ego_masks[cam] = m_ego
            ego_masks_t[cam] = torch.from_numpy(m_ego).to(self.device)

            # Prepare 3D elevated foreground cloud ONLY when monocular depth was predicted
            if d_cal is not None:
                k_inv = torch.tensor(np.linalg.inv(self.k_render[cam]), dtype=torch.float32, device=self.device)
                r_ct = torch.tensor(self.r_mats[cam].T, dtype=torch.float32, device=self.device)
                t_c = torch.tensor(self.t_vecs[cam], dtype=torch.float32, device=self.device)
                z_t = torch.from_numpy(d_cal).to(self.device).unsqueeze(-1)
                pts_ego = (((self._dst_pix @ k_inv.T) * z_t) - t_c) @ r_ct.T
                # Elevated objects strictly above road surface (+0.35m) within 50m and outside ego hood
                elevated = (pts_ego[..., 2] > (self.ground_z + 0.35)) & (d_cal_t := z_t[..., 0] < 50.0)
                del d_cal_t
                fg_mask = elevated & (~ego_masks_t[cam])
                u_norm = (self._dst_pix[..., 0] / max(self.render_w - 1, 1)) * 2.0 - 1.0
                v_norm = (self._dst_pix[..., 1] / max(self.render_h - 1, 1)) * 2.0 - 1.0
                uv_n = torch.stack([u_norm, v_norm], dim=-1)
                fg_world[cam] = (pts_ego.reshape(-1, 3), uv_n.reshape(-1, 2), fg_mask.reshape(-1))

        # Load past frames and poses for border/rear gap completion
        past_sources: Dict[str, List[Tuple[Tuple[float, float, float], torch.Tensor]]] = {c: [] for c in spec.CAMERAS}
        if past_images and past_trajectory is not None:
            past_poses = compute_past_poses(past_trajectory)
            for tag in ("tm02", "tm04", "tm10", "tm20", "tm30", "tm40"):
                if tag not in past_poses:
                    continue
                p_pose = past_poses[tag]
                for cam in spec.CAMERAS:
                    if (cam, tag) not in past_images:
                        continue
                    psrc = past_images[(cam, tag)]
                    if isinstance(psrc, (str, Path)):
                        pbgr = cv2.imread(str(psrc), cv2.IMREAD_COLOR)
                        if pbgr is None:
                            continue
                        prgb = cv2.cvtColor(pbgr, cv2.COLOR_BGR2RGB)
                    else:
                        prgb = np.asarray(psrc, dtype=np.uint8)
                    past_sources[cam].append((p_pose, self._to_tensor_img(prgb)))

        headings = compute_smooth_headings(pts)
        outputs: Dict[str, np.ndarray] = {}

        for cam in target_cameras:
            frames_out = np.empty((spec.FRAME_COUNT, spec.HEIGHT, spec.WIDTH, 3), dtype=np.uint8)
            ego_mask_t = ego_masks_t[cam]
            t0_ren_t = F.interpolate(t0_hi_t[cam], size=(self.render_h, self.render_w), mode="area")[0]
            ego_blur = cv2.GaussianBlur(ego_masks[cam].astype(np.float32), (11, 11), 2.5)
            ego_alpha = torch.from_numpy(ego_blur).to(self.device).unsqueeze(0)

            for k in range(spec.FRAME_COUNT):
                pose_k = (float(pts[k, 0]), float(pts[k, 1]), float(headings[k]))
                if abs(pose_k[0]) < 1e-4 and abs(pose_k[1]) < 1e-4 and abs(pose_k[2]) < 1e-4:
                    frame_ren = t0_ren_t
                    cov_mask = torch.ones((self.render_h, self.render_w), dtype=torch.bool, device=self.device)
                else:
                    # 1. Exact closed-form backward homography from primary camera t=0
                    frame_ren, cov_mask = self._warp_backward_analytical(
                        cam, (0.0, 0.0, 0.0), cam, pose_k, t0_hi_t[cam], ego_mask_t
                    )

                    # 2. Adjacent 360-degree neighbor cameras fill horizontal/yaw margins
                    if not bool((cov_mask | ego_mask_t).all()):
                        for nbr in CAMERA_NEIGHBORS[cam]:
                            n_rgb, n_cov = self._warp_backward_analytical(
                                nbr, (0.0, 0.0, 0.0), cam, pose_k, t0_hi_t[nbr], ego_masks_t[nbr]
                            )
                            frame_ren, cov_mask = self._match_and_composite(frame_ren, cov_mask, n_rgb, n_cov)
                            if bool((cov_mask | ego_mask_t).all()):
                                break

                    # 3. Past frames (t = -0.2s .. -2.0s) fill expanding behind-car road regions!
                    if not bool((cov_mask | ego_mask_t).all()) and past_sources[cam]:
                        for p_pose, p_t in past_sources[cam]:
                            p_rgb, p_cov = self._warp_backward_analytical(
                                cam, p_pose, cam, pose_k, p_t, ego_mask_t
                            )
                            frame_ren, cov_mask = self._match_and_composite(frame_ren, cov_mask, p_rgb, p_cov)
                            if bool((cov_mask | ego_mask_t).all()):
                                break

                    # 4. Elevated 3D foreground object parallax layer (cars, poles, buildings)
                    if cam in fg_world:
                        pw_f, uv_f, m_f = fg_world[cam]
                        fg_rgb, fg_cov, _ = self._render_foreground_3d_overlay(
                            pw_f, uv_f, m_f, cam, pose_k, t0_hi_t[cam]
                        )
                        if bool(fg_cov.any()):
                            frame_ren = torch.where(fg_cov.unsqueeze(0), fg_rgb, frame_ren)
                            cov_mask = cov_mask | fg_cov

                    # 5. Re-composite bolted ego car hood/roof corner cleanly
                    frame_ren = ego_alpha * t0_ren_t + (1.0 - ego_alpha) * frame_ren
                    cov_mask = cov_mask | ego_mask_t

                if self.render_scale > 1:
                    frame_out_t = F.interpolate(
                        frame_ren.unsqueeze(0), size=(spec.HEIGHT, spec.WIDTH), mode="area"
                    )[0]
                    cov_out = (
                        F.interpolate(
                            cov_mask.float().unsqueeze(0).unsqueeze(0),
                            size=(spec.HEIGHT, spec.WIDTH),
                            mode="nearest",
                        )[0, 0]
                        > 0.5
                    )
                else:
                    frame_out_t = frame_ren
                    cov_out = cov_mask

                frame_np = (frame_out_t.permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                hole_np = ~cov_out.detach().cpu().numpy()
                if np.any(hole_np):
                    frame_np = self._fill_residual_holes(frame_np, hole_np)

                if self.crf_sharpen > 0.0:
                    blur = cv2.GaussianBlur(frame_np, (0, 0), 1.0)
                    frame_np = cv2.addWeighted(frame_np, 1.0 + self.crf_sharpen, blur, -self.crf_sharpen, 0)

                frames_out[k] = frame_np

            outputs[cam] = frames_out

        return outputs
