"""End-to-end multi-view video generation engine for the KITScenes-LongTail Challenge.

This module provides:
1. Camera extrinsic/intrinsic geometry handling for the 6 surround cameras:
   (front 0°, front_left +60°, front_right -60°, rear 180°, rear_left +120°, rear_right -120°)
   at the required 876x568 resolution.
2. Trajectory-conditioned motion parameter derivation converting 25 trajectory waypoints
   into frame-by-frame forward speed v, yaw rate w, heading, and camera-specific velocities.
3. Optical flow and ego-motion coherence (EMC) modeling guaranteeing that front + rear
   agree on yaw, and the 4 side cameras reflect vehicle speed.
4. Open-source world model backbones (Geometric/Warp, SVD, Action-Conditioned WM).
5. Robust H.264 mp4 video encoding (libx264, 5 fps, exactly 25 frames, 876x568).
6. High-throughput batch generation and submission archive packaging.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import archive, media, spec


# Camera angular offsets in degrees and radians relative to vehicle forward heading (+X).
CAMERA_YAW_DEG: Dict[str, float] = {
    "front": 0.0,
    "front_left": 60.0,
    "front_right": -60.0,
    "rear": 180.0,
    "rear_left": 120.0,
    "rear_right": -120.0,
}

CAMERA_YAW_RAD: Dict[str, float] = {
    cam: math.radians(deg) for cam, deg in CAMERA_YAW_DEG.items()
}

# Standard sensor rig positions relative to vehicle center [x, y, z] in meters.
# x: forward, y: left, z: up.
DEFAULT_CAMERA_POSITIONS: Dict[str, Tuple[float, float, float]] = {
    "front": (1.50, 0.00, 1.40),
    "front_left": (1.20, 0.60, 1.40),
    "front_right": (1.20, -0.60, 1.40),
    "rear": (-1.00, 0.00, 1.40),
    "rear_left": (-0.50, 0.60, 1.40),
    "rear_right": (-0.50, -0.60, 1.40),
}


def wrap_angle(angle: float) -> float:
    """Wrap angle to (-pi, pi] radians."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class CameraPose:
    """Pose and intrinsic parameters of a single camera in the rig."""
    name: str
    yaw_deg: float
    yaw_rad: float
    position_m: Tuple[float, float, float]
    width: int = spec.WIDTH
    height: int = spec.HEIGHT
    hfov_deg: float = 70.0

    @property
    def fx(self) -> float:
        return (self.width / 2.0) / math.tan(math.radians(self.hfov_deg / 2.0))

    @property
    def fy(self) -> float:
        return self.fx

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def rotation_matrix_body_to_cam(self) -> np.ndarray:
        """3x3 rotation matrix mapping vehicle body coordinates to camera coordinate system.

        Vehicle body: X=forward, Y=left, Z=up.
        Camera frame: x=right, y=down, z=forward along optical axis.
        """
        th = self.yaw_rad
        cos_th = math.cos(th)
        sin_th = math.sin(th)
        # R_yaw: vehicle body to vehicle yawed by camera angle
        # Optical axis = yawed X; camera right = - yawed Y; camera down = - Z
        return np.array(
            [
                [-sin_th, cos_th, 0.0],
                [0.0, 0.0, -1.0],
                [cos_th, sin_th, 0.0],
            ],
            dtype=np.float64,
        )


class CameraRig:
    """Manages the 6-camera surround rig conforming to KITScenes specifications."""

    def __init__(
        self,
        width: int = spec.WIDTH,
        height: int = spec.HEIGHT,
        hfov_deg: float = 70.0,
        positions: Optional[Dict[str, Tuple[float, float, float]]] = None,
    ):
        self.width = width
        self.height = height
        self.hfov_deg = hfov_deg
        pos_dict = positions or DEFAULT_CAMERA_POSITIONS

        self.cameras: Dict[str, CameraPose] = {
            cam: CameraPose(
                name=cam,
                yaw_deg=CAMERA_YAW_DEG[cam],
                yaw_rad=CAMERA_YAW_RAD[cam],
                position_m=pos_dict[cam],
                width=width,
                height=height,
                hfov_deg=hfov_deg,
            )
            for cam in spec.CAMERAS
        }

    def __getitem__(self, name: str) -> CameraPose:
        return self.cameras[name]

    def get_camera_heading(self, camera: str, ego_heading: float) -> float:
        """Absolute heading of camera in world frame given vehicle ego heading."""
        return wrap_angle(ego_heading + self.cameras[camera].yaw_rad)

    def get_camera_relative_velocity(
        self, camera: str, forward_speed: float, yaw_rate: float
    ) -> Tuple[float, float, float]:
        """Velocity components seen in camera coordinate frame.

        Returns:
            (v_forward, v_lateral, omega_yaw)
            - v_forward: along camera optical axis (positive = forward looming)
            - v_lateral: across image plane (positive = scene moving left, camera moving right)
            - omega_yaw: yaw rotation rate about vertical axis
        """
        th = self.cameras[camera].yaw_rad
        # Forward speed v projected along camera optical axis:
        v_forward = forward_speed * math.cos(th)
        # Forward speed v projected perpendicular to camera optical axis:
        v_lateral = -forward_speed * math.sin(th)
        # Yaw rate about vertical axis:
        omega_yaw = yaw_rate
        return v_forward, v_lateral, omega_yaw


@dataclass
class CameraMotionSlice:
    """Motion profile for one specific camera across the 25 frames."""
    camera: str
    v_forward: np.ndarray  # (25,) m/s along optical axis
    v_lateral: np.ndarray  # (25,) m/s across image sensor
    omega_yaw: np.ndarray  # (25,) rad/s yaw rate
    headings: np.ndarray   # (25,) camera absolute headings
    positions: np.ndarray  # (25, 3) 3D positions in world/ego frame


@dataclass
class TrajectoryMotionProfile:
    """Full trajectory motion analysis over the 25 frames (0.2 s to 5.0 s)."""
    waypoints: np.ndarray          # (25, 2) [x, y] coordinates
    dt: float                      # 0.2 s
    speeds: np.ndarray             # (25,) vehicle forward speed (m/s)
    yaw_rates: np.ndarray          # (25,) vehicle yaw rate (rad/s)
    headings: np.ndarray           # (25,) vehicle headings (rad)
    accelerations: np.ndarray      # (25,) longitudinal accelerations (m/s^2)
    camera_motions: Dict[str, CameraMotionSlice]
    emc_segment_speeds: Dict[str, float]  # speeds for 0-3s and 3-5s segments
    emc_steering_classes: Dict[str, str]  # steering classes for segments


class TrajectoryMotionAnalyzer:
    """Derives frame-by-frame motion parameters from planned trajectory waypoints."""

    def __init__(self, rig: Optional[CameraRig] = None, dt: float = spec.WAYPOINT_STEP_S):
        self.rig = rig or CameraRig()
        self.dt = dt

    def analyze(
        self,
        waypoints: Sequence[Sequence[float]],
        initial_pose: Optional[Tuple[float, float, float]] = None,
    ) -> TrajectoryMotionProfile:
        """Convert 25 (x, y) trajectory waypoints into kinematically coherent parameters.

        Args:
            waypoints: 25 [x, y] waypoints at 5 Hz (0.2s to 5.0s into the future).
            initial_pose: Optional (x0, y0, psi0) at t=0. Defaults to (0, 0, 0).
        """
        pts = np.asarray(waypoints, dtype=np.float64)
        if pts.shape != (spec.WAYPOINT_COUNT, 2):
            raise ValueError(
                f"expected ({spec.WAYPOINT_COUNT}, 2) waypoints, got {pts.shape}"
            )

        x0, y0, psi0 = initial_pose if initial_pose is not None else (0.0, 0.0, 0.0)

        # Full sequence including t=0
        all_pts = np.vstack([[x0, y0], pts])  # (26, 2)
        deltas = np.diff(all_pts, axis=0)      # (25, 2)
        displacements = np.linalg.norm(deltas, axis=1)  # (25,)
        speeds = displacements / self.dt

        # Heading determination
        headings = np.zeros(spec.WAYPOINT_COUNT, dtype=np.float64)
        current_heading = psi0

        for i in range(spec.WAYPOINT_COUNT):
            if displacements[i] > 1e-4:
                current_heading = math.atan2(deltas[i, 1], deltas[i, 0])
            headings[i] = current_heading

        # Yaw rates (finite differences with angle wrapping)
        all_headings = np.concatenate([[psi0], headings])
        heading_diffs = np.zeros(spec.WAYPOINT_COUNT, dtype=np.float64)
        for i in range(spec.WAYPOINT_COUNT):
            heading_diffs[i] = wrap_angle(all_headings[i + 1] - all_headings[i])
        yaw_rates = heading_diffs / self.dt

        # Accelerations
        all_speeds = np.concatenate([[speeds[0]], speeds])
        accelerations = np.diff(all_speeds) / self.dt

        # Per-camera motion profiles
        camera_motions: Dict[str, CameraMotionSlice] = {}
        for cam_name, cam in self.rig.cameras.items():
            th = cam.yaw_rad
            # v_forward along camera optical axis
            v_fwd = speeds * math.cos(th)
            # v_lateral across image plane (ground passing sideways)
            v_lat = -speeds * math.sin(th)
            # yaw rate
            w_yaw = yaw_rates.copy()
            # camera headings
            cam_headings = np.array([wrap_angle(h + th) for h in headings])

            # 3D positions in world/ego frame
            cam_pos_body = np.array(cam.position_m)
            cam_positions = np.zeros((spec.WAYPOINT_COUNT, 3), dtype=np.float64)
            for i in range(spec.WAYPOINT_COUNT):
                h = headings[i]
                cos_h = math.cos(h)
                sin_h = math.sin(h)
                # Rotate camera body offset by ego heading
                r_offset = np.array([
                    cos_h * cam_pos_body[0] - sin_h * cam_pos_body[1],
                    sin_h * cam_pos_body[0] + cos_h * cam_pos_body[1],
                    cam_pos_body[2],
                ])
                cam_positions[i, 0] = pts[i, 0] + r_offset[0]
                cam_positions[i, 1] = pts[i, 1] + r_offset[1]
                cam_positions[i, 2] = r_offset[2]

            camera_motions[cam_name] = CameraMotionSlice(
                camera=cam_name,
                v_forward=v_fwd,
                v_lateral=v_lat,
                omega_yaw=w_yaw,
                headings=cam_headings,
                positions=cam_positions,
            )

        # Segment metrics for EMC (0-3s is frames 0..14, 3-5s is frames 15..24)
        seg1_speed = float(np.mean(speeds[0:15]))
        seg2_speed = float(np.mean(speeds[15:25]))

        seg1_turn = float(np.sum(heading_diffs[0:15]))
        seg2_turn = float(np.sum(heading_diffs[15:25]))

        def classify_turn(rad: float) -> str:
            if rad > math.radians(10.0):
                return "left"
            if rad < -math.radians(10.0):
                return "right"
            return "straight"

        emc_segment_speeds = {"seg1_0_3s": seg1_speed, "seg2_3_5s": seg2_speed}
        emc_steering_classes = {
            "seg1_0_3s": classify_turn(seg1_turn),
            "seg2_3_5s": classify_turn(seg2_turn),
        }

        return TrajectoryMotionProfile(
            waypoints=pts,
            dt=self.dt,
            speeds=speeds,
            yaw_rates=yaw_rates,
            headings=headings,
            accelerations=accelerations,
            camera_motions=camera_motions,
            emc_segment_speeds=emc_segment_speeds,
            emc_steering_classes=emc_steering_classes,
        )


class VideoEncoder:
    """Encodes raw frame arrays into strictly compliant H.264 mp4 videos."""

    def __init__(
        self,
        fps: float = spec.FPS,
        width: int = spec.WIDTH,
        height: int = spec.HEIGHT,
        codec: str = "libx264",
        crf: int = 24,
        preset: str = "fast",
    ):
        self.fps = fps
        self.width = width
        self.height = height
        self.codec = codec
        self.crf = crf
        self.preset = preset
        self._ffmpeg_bin = shutil.which("ffmpeg")

    def encode(
        self,
        frames: np.ndarray,
        output_path: Union[str, Path],
        validate: bool = True,
    ) -> Path:
        """Encode (25, height, width, 3) uint8 numpy frames into an H.264 mp4 file.

        Args:
            frames: (25, H, W, 3) array with values 0..255 (RGB).
            output_path: Destination .mp4 file.
            validate: If True, probes the written file with ffprobe to verify 0 errors.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not isinstance(frames, np.ndarray):
            frames = np.asarray(frames, dtype=np.uint8)

        if frames.ndim != 4 or frames.shape[0] != spec.FRAME_COUNT:
            raise ValueError(
                f"frames must have shape ({spec.FRAME_COUNT}, {self.height}, "
                f"{self.width}, 3), got {frames.shape}"
            )

        if frames.shape[1] != self.height or frames.shape[2] != self.width:
            raise ValueError(
                f"frame resolution {frames.shape[2]}x{frames.shape[1]} does not "
                f"match {self.width}x{self.height}"
            )

        if frames.dtype != np.uint8:
            frames = np.clip(frames, 0, 255).astype(np.uint8)

        if self._ffmpeg_bin is None:
            raise RuntimeError("ffmpeg binary not found on PATH; required for video encoding")

        # Encode via ffmpeg rawvideo stdin pipe
        cmd = [
            self._ffmpeg_bin,
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self.width}x{self.height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            f"{self.fps:g}",
            "-i",
            "-",
            "-c:v",
            self.codec,
            "-pix_fmt",
            "yuv420p",
            "-r",
            f"{self.fps:g}",
            "-crf",
            str(self.crf),
            "-preset",
            self.preset,
            str(output_path),
        ]

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _, stderr = proc.communicate(input=frames.tobytes())

        if proc.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg encoding failed (code {proc.returncode}): {err_msg}")

        # Enforce size ceiling (< 50MB per video)
        file_size = output_path.stat().st_size
        if file_size > spec.MAX_VIDEO_BYTES:
            raise RuntimeError(
                f"encoded video size {file_size / 1024**2:.2f} MB exceeds 50 MB limit"
            )

        if validate:
            info = media.probe_video(output_path)
            errors, warnings = media.check_video(info)
            if errors:
                raise RuntimeError(
                    f"encoded video failed validation checks: {'; '.join(errors)}"
                )

        return output_path


class GeometricWorldModel:
    """Procedural geometric driving scene synthesizer and pipeline reference generator.

    Renders 360-degree trajectory-consistent surround views with exact geometric optical flow:
    - Front/rear views rotate in lockstep with vehicle yaw rate w.
    - Side views stream scenery at speed proportional to vehicle forward speed v * sin(60°).
    - Road perspective, lane markings, horizon, and roadside features move in exact
      kinematic synchronization with trajectory waypoints.
    - Designed as a lightweight, fast, deterministic local reference to verify video encoding,
      EMC optical flow agreement, and submission packaging without requiring GPU compute or heavy
      model checkpoints.
    - Note: Perceptual Quality (PQ) from procedural canvas rendering will score low on Q-Align
      benchmarks compared to real video diffusion models. Use a diffusion backbone (SVD/Vista/Drive-WM)
      for competitive leaderboard ranking.
    """

    def __init__(self, rig: Optional[CameraRig] = None):
        self.rig = rig or CameraRig()
        self.width = self.rig.width
        self.height = self.rig.height
        self.analyzer = TrajectoryMotionAnalyzer(self.rig)

    def _render_sky_and_ground(
        self,
        height: int,
        width: int,
        horizon_y: int,
        weather: str = "clear",
    ) -> np.ndarray:
        """Create baseline sky and ground gradients."""
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Sky colors
        if weather == "night":
            sky_top = np.array([5, 8, 18], dtype=np.float32)
            sky_horizon = np.array([15, 20, 35], dtype=np.float32)
            ground_base = np.array([20, 22, 25], dtype=np.float32)
        elif weather in ("rain", "snow"):
            sky_top = np.array([90, 100, 110], dtype=np.float32)
            sky_horizon = np.array([140, 150, 160], dtype=np.float32)
            ground_base = np.array([45, 48, 52], dtype=np.float32)
        else:
            sky_top = np.array([65, 120, 195], dtype=np.float32)
            sky_horizon = np.array([160, 195, 230], dtype=np.float32)
            ground_base = np.array([50, 52, 55], dtype=np.float32)

        # Sky gradient
        if horizon_y > 0:
            alpha = np.linspace(0.0, 1.0, horizon_y)[:, None, None]
            frame[:horizon_y] = np.clip(
                (1.0 - alpha) * sky_top + alpha * sky_horizon, 0, 255
            ).astype(np.uint8)

        # Ground plane
        ground_h = height - horizon_y
        if ground_h > 0:
            beta = np.linspace(0.0, 1.0, ground_h)[:, None, None]
            ground_far = ground_base * 0.7
            ground_near = ground_base * 1.15
            frame[horizon_y:] = np.clip(
                (1.0 - beta) * ground_far + beta * ground_near, 0, 255
            ).astype(np.uint8)

        return frame

    def render_scenario_camera(
        self,
        camera: str,
        motion: TrajectoryMotionProfile,
        weather: str = "clear",
    ) -> np.ndarray:
        """Render exactly 25 frames (25, 568, 876, 3) for a single camera view."""
        w, h = self.width, self.height
        cam_pose = self.rig[camera]
        cam_motion = motion.camera_motions[camera]

        horizon_y = int(cam_pose.cy)
        base_canvas = self._render_sky_and_ground(h, w, horizon_y, weather)

        frames = np.empty((spec.FRAME_COUNT, h, w, 3), dtype=np.uint8)
        accumulated_dist = 0.0

        for frame_idx in range(spec.FRAME_COUNT):
            frame = base_canvas.copy()
            v_lat = cam_motion.v_lateral[frame_idx]
            w_yaw = cam_motion.omega_yaw[frame_idx]
            ego_speed = motion.speeds[frame_idx]
            accumulated_dist += ego_speed * motion.dt

            # Background panorama shift based on camera yaw angle and vehicle heading
            total_heading = cam_motion.headings[frame_idx]
            # Shift distant horizon / features horizontally proportional to heading
            horiz_shift = int(cam_pose.fx * math.tan(w_yaw * motion.dt * 0.5))

            # Render road & perspective elements based on camera role
            if camera == "front":
                # Vanishing point shifts laterally with yaw rate
                vp_x = int(cam_pose.cx - cam_pose.fx * math.tan(w_yaw * 0.2))
                vp_x = max(100, min(w - 100, vp_x))
                vp_y = horizon_y

                # Road polygon
                road_y_indices = np.arange(vp_y, h)
                for y in road_y_indices:
                    t = (y - vp_y) / float(h - vp_y)
                    half_width = int(t * w * 0.42)
                    rx_left = max(0, vp_x - half_width)
                    rx_right = min(w, vp_x + half_width)
                    # Asphalt
                    frame[y, rx_left:rx_right] = [58, 60, 63]

                    # Road edge solid white lines
                    line_w = max(1, int(t * 5))
                    if rx_left + line_w < w:
                        frame[y, rx_left:rx_left + line_w] = [230, 230, 235]
                    if rx_right - line_w >= 0:
                        frame[y, rx_right - line_w:rx_right] = [230, 230, 235]

                    # Dashed centerline advancing with forward speed
                    dash_cycle = (y * 0.15 - accumulated_dist * 3.0) % 20.0
                    if dash_cycle < 10.0 and vp_x - line_w >= 0 and vp_x + line_w < w:
                        frame[y, vp_x - line_w:vp_x + line_w] = [240, 235, 120]

            elif camera == "rear":
                # Vanishing point
                vp_x = int(cam_pose.cx + cam_pose.fx * math.tan(w_yaw * 0.2))
                vp_x = max(100, min(w - 100, vp_x))
                vp_y = horizon_y

                road_y_indices = np.arange(vp_y, h)
                for y in road_y_indices:
                    t = (y - vp_y) / float(h - vp_y)
                    half_width = int(t * w * 0.42)
                    rx_left = max(0, vp_x - half_width)
                    rx_right = min(w, vp_x + half_width)
                    frame[y, rx_left:rx_right] = [58, 60, 63]

                    # Centerline receding with distance
                    dash_cycle = (y * 0.15 + accumulated_dist * 3.0) % 20.0
                    line_w = max(1, int(t * 5))
                    if dash_cycle < 10.0 and vp_x - line_w >= 0 and vp_x + line_w < w:
                        frame[y, vp_x - line_w:vp_x + line_w] = [240, 235, 120]

            else:
                # Side cameras: front_left, front_right, rear_left, rear_right
                # Optical flow is heavily lateral across the image plane (speed signal)
                # Render roadside textures, guardrails, and ground stripes streaming sideways
                side_sign = 1.0 if "left" in camera else -1.0
                stream_speed = abs(v_lat)  # ~ ego_speed * sin(60°) = 0.866 * v
                lateral_offset = (accumulated_dist * stream_speed * 0.5) % 80.0

                # Roadside grass / curb / asphalt
                curb_y = int(horizon_y + (h - horizon_y) * 0.25)
                frame[horizon_y:curb_y, :] = [65, 85, 55]  # grass/vegetation
                frame[curb_y:, :] = [56, 58, 62]            # road surface

                # Guardrail / roadside posts streaming with lateral velocity
                post_spacing = 60
                post_phase = int(side_sign * lateral_offset) % post_spacing
                for px in range(post_phase, w, post_spacing):
                    post_x = (px + horiz_shift) % w
                    post_top = max(0, horizon_y - 20)
                    post_bot = min(h, curb_y + 15)
                    frame[post_top:post_bot, max(0, post_x - 3):min(w, post_x + 3)] = [180, 185, 190]

                # Lateral ground surface motion stripes
                for y in range(curb_y, h, 12):
                    t = (y - curb_y) / float(h - curb_y)
                    stripe_offset = int(side_sign * (lateral_offset * (1.0 + t * 2.0))) % 40
                    for sx in range(stripe_offset, w, 40):
                        if 0 <= sx < w - 8:
                            frame[y:min(h, y + 2), sx:sx + 8] = [70, 72, 76]

            # Distant sun / light source matching weather
            if weather != "night":
                sun_x = int((w * 0.7 + math.sin(total_heading) * 100) % w)
                sun_y = int(horizon_y * 0.4)
                rr, cc = np.ogrid[:h, :w]
                sun_mask = (rr - sun_y) ** 2 + (cc - sun_x) ** 2 < 28 ** 2
                frame[sun_mask] = [255, 245, 200]

            frames[frame_idx] = frame

        return frames

    def generate_scenario(
        self,
        scenario_id: str,
        trajectory: Sequence[Sequence[float]],
        weather: str = "clear",
    ) -> Dict[str, np.ndarray]:
        """Generate all 6 camera videos for one scenario."""
        motion = self.analyzer.analyze(trajectory)
        result: Dict[str, np.ndarray] = {}

        for camera in spec.CAMERAS:
            result[camera] = self.render_scenario_camera(camera, motion, weather=weather)

        return result


def require_pillow():
    """Import PIL lazily, so a missing dependency reports what the backend needs."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "the svd backbone needs torch, diffusers, accelerate and Pillow, plus a "
            "downloaded checkpoint: pip install torch diffusers accelerate pillow"
        ) from exc
    return Image


class StableVideoDiffusionGenerator:
    """Wrapper for Stable Video Diffusion (SVD) with Kaggle 2x T4 optimizations."""

    def __init__(
        self,
        model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt",
        device: str = "cuda",
        torch_dtype: Optional[Any] = None,
        num_frames: int = spec.FRAME_COUNT,
        fps: int = int(spec.FPS),
    ):
        self.model_id = model_id
        self.device = device
        self.num_frames = num_frames
        self.fps = fps
        self.pipeline = None

        try:
            import torch
            from diffusers import StableVideoDiffusionPipeline

            dtype = torch_dtype or torch.float16
            self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                variant="fp16",
            )
            # Memory optimizations for 16GB T4 GPUs
            if device.startswith("cuda"):
                self.pipeline.enable_model_cpu_offload()
                if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
                    try:
                        self.pipeline.enable_xformers_memory_efficient_attention()
                    except Exception:
                        pass
        except (ImportError, Exception) as exc:
            # Fallback handled in generate if torch/diffusers not available locally
            self._init_error = str(exc)

    def generate_camera_video(
        self,
        conditioning_image: Any,
        speed: float,
        num_inference_steps: int = 15,
        decode_chunk_size: int = 4,
    ) -> np.ndarray:
        """Run SVD inference conditioned on image and trajectory motion bucket."""
        if self.pipeline is None:
            raise RuntimeError(
                f"SVD pipeline not loaded: {getattr(self, '_init_error', 'diffusers missing')}"
            )

        # Scale motion bucket from vehicle forward speed (1..255)
        motion_bucket = int(np.clip(speed * 12.0 + 10.0, 1.0, 255.0))

        output = self.pipeline(
            conditioning_image,
            num_frames=self.num_frames,
            fps=self.fps,
            motion_bucket_id=motion_bucket,
            num_inference_steps=num_inference_steps,
            decode_chunk_size=decode_chunk_size,
        )
        pil_frames = output.frames[0]
        # Resize to exact 876x568 resolution
        resized = [
            np.array(f.resize((spec.WIDTH, spec.HEIGHT))) for f in pil_frames
        ]
        return np.stack(resized, axis=0).astype(np.uint8)

    def generate_scenario(
        self,
        scenario_id: str,
        trajectory: Sequence[Sequence[float]],
        weather: str = "clear",
        conditioning_images: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        """Generate all 6 surround camera videos using Stable Video Diffusion.

        Args:
            scenario_id: 7-digit scenario id e.g. '0000000'.
            trajectory: 25 [x, y] coordinates.
            weather: Scene/weather preset.
            conditioning_images: mapping from camera name to a PIL Image or HxWx3 array.
                Required for all six cameras: these are the observed t=0 frames.
        """
        profile = TrajectoryMotionAnalyzer().analyze(trajectory)
        avg_speed = float(np.mean(profile.speeds))

        if not conditioning_images:
            raise ValueError(
                "the svd backbone needs the observed t=0 frame for every camera, and none "
                "were passed. Conditioning the model on a procedurally drawn canvas would "
                "produce video unrelated to the scenario, so this is refused rather than "
                "silently degraded."
            )
        missing = [name for name in spec.CAMERAS if name not in conditioning_images]
        if missing:
            raise ValueError(
                "the svd backbone needs the observed t=0 frame for every camera; "
                f"missing: {', '.join(missing)}"
            )

        image_module = require_pillow()
        results: Dict[str, np.ndarray] = {}

        for cam_name in spec.CAMERAS:
            supplied = conditioning_images[cam_name]
            if isinstance(supplied, (str, Path)):
                cond_img = image_module.open(str(supplied)).convert("RGB")
            elif isinstance(supplied, bytes):
                import io

                cond_img = image_module.open(io.BytesIO(supplied)).convert("RGB")
            elif isinstance(supplied, np.ndarray):
                cond_img = image_module.fromarray(
                    np.ascontiguousarray(supplied).astype(np.uint8)
                )
            else:
                cond_img = supplied

            frames = self.generate_camera_video(cond_img, speed=avg_speed)
            results[cam_name] = frames

        return results


def load_conditioning_images(
    observed_t0_dir: Path | str, scenario_id: str
) -> Dict[str, Path]:
    """Load mapping of camera_name -> image path for a scenario's observed t=0 frames."""
    sc_dir = Path(observed_t0_dir) / str(scenario_id).zfill(7)
    if not sc_dir.exists():
        raise FileNotFoundError(
            f"no observed t=0 directory for scenario {scenario_id!r} in {observed_t0_dir}"
        )
    images: Dict[str, Path] = {}
    for cam in spec.CAMERAS:
        img_path = sc_dir / f"{cam}.jpg"
        if not img_path.exists():
            img_path = sc_dir / f"{cam}.png"
        if not img_path.exists():
            raise FileNotFoundError(
                f"missing {cam} frame for scenario {scenario_id} at {img_path}"
            )
        images[cam] = img_path
    return images


class WorldModelPipeline:
    """Unified video generation engine supporting modular backbones."""

    def __init__(
        self,
        backbone: str = "geometric",
        rig: Optional[CameraRig] = None,
        encoder_crf: int = 24,
        device: str = "cuda",
    ):
        self.backbone_name = backbone
        self.rig = rig or CameraRig()
        self.encoder = VideoEncoder(crf=encoder_crf)
        self.analyzer = TrajectoryMotionAnalyzer(self.rig)
        self.device = device

        if backbone in ("geometric", "warp"):
            self.generator = GeometricWorldModel(self.rig)
        elif backbone == "svd":
            self.generator = StableVideoDiffusionGenerator(device=device)
        else:
            raise ValueError(f"unknown backbone '{backbone}'; expected 'geometric' or 'svd'")

    def generate_and_save_scenario(
        self,
        scenario_id: str,
        trajectory: Sequence[Sequence[float]],
        output_dir: Path,
        weather: str = "clear",
        conditioning_images: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """Generate all 6 camera videos for a scenario and encode them directly to disk.

        Args:
            scenario_id: 7-digit scenario id e.g. "0000000".
            trajectory: 25 [x, y] coordinates.
            output_dir: Directory where <scenario_id>/<camera>.mp4 will be placed.
            weather: Optional weather/scene modifier.
            conditioning_images: Optional conditioning images per camera.
        """
        output_dir = Path(output_dir)
        scenario_dir = output_dir / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # Generate frames
        if hasattr(self.generator, "generate_scenario"):
            import inspect
            sig = inspect.signature(self.generator.generate_scenario)
            if "conditioning_images" in sig.parameters:
                videos = self.generator.generate_scenario(
                    scenario_id=scenario_id,
                    trajectory=trajectory,
                    weather=weather,
                    conditioning_images=conditioning_images,
                )
            else:
                videos = self.generator.generate_scenario(
                    scenario_id=scenario_id,
                    trajectory=trajectory,
                    weather=weather,
                )
        else:
            raise NotImplementedError(f"Generator {type(self.generator)} does not implement generate_scenario")

        encoded_paths: Dict[str, Path] = {}
        for cam_name in spec.CAMERAS:
            frames = videos[cam_name]
            out_path = scenario_dir / f"{cam_name}.mp4"
            self.encoder.encode(frames, out_path, validate=True)
            encoded_paths[cam_name] = out_path

        return encoded_paths


def batch_generate_and_package(
    trajectories_file: Path,
    output_dir: Path,
    zip_path: Optional[Path] = None,
    scenario_ids: Optional[Sequence[str]] = None,
    allow_partial: bool = False,
    backbone: str = "geometric",
    device: str = "cuda",
    conditioning_images: Optional[Dict[str, Dict[str, Any]]] = None,
    observed_t0_dir: Optional[Union[Path, str]] = None,
    progress_callback: Optional[Any] = None,
) -> archive.ArchiveReport:
    """Generate videos for scenarios from trajectories.jsonl and pack into videos.zip.

    Args:
        trajectories_file: Path to validated trajectories.jsonl.
        output_dir: Intermediate renders directory (e.g. renders/).
        zip_path: Output archive path (default: <output_dir>/../videos.zip).
        scenario_ids: Subset of scenarios to generate (default: all 400).
        allow_partial: If True, allows packing a subset of scenarios for local testing.
        backbone: Generator backbone ('geometric' for fast mock, 'svd' for diffusion).
        device: Device to run generative backbones on ('cuda', 'cpu', 'mps').
        conditioning_images: Optional dict of scenario_id -> {camera_name: image}.
        observed_t0_dir: Optional directory with observed t=0 images (<id>/<camera>.jpg).
        progress_callback: Optional callable(index, total, scenario_id).
    """
    trajectories_file = Path(trajectories_file)
    output_dir = Path(output_dir)
    zip_path = Path(zip_path) if zip_path else output_dir.parent / "videos.zip"

    # Read trajectories
    trajs: Dict[str, List[List[float]]] = {}
    with open(trajectories_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            trajs[rec["scenario_id"]] = rec["future_trajectory"]

    target_ids = list(scenario_ids) if scenario_ids is not None else list(spec.SCENARIO_IDS)

    # Resolve conditioning up front. Resolving inside the loop would let a run of several
    # thousand GPU-seconds die at scenario 381 because one frame folder is absent, and
    # would hide which scenarios were missing.
    cond_by_scenario: Dict[str, Dict[str, Any]] = dict(conditioning_images or {})
    if observed_t0_dir is not None:
        if backbone != "svd":
            print(
                f"warning: observed_t0_dir has no effect with the '{backbone}' backbone, "
                f"which renders procedurally; only 'svd' consumes the frames",
                file=sys.stderr,
            )
        missing: List[str] = []
        for sc_id in target_ids:
            if sc_id in cond_by_scenario:
                continue
            try:
                cond_by_scenario[sc_id] = load_conditioning_images(observed_t0_dir, sc_id)
            except FileNotFoundError as exc:
                missing.append(str(exc))
        if missing:
            shown = "\n".join(f"  {entry}" for entry in missing[:10])
            more = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
            raise FileNotFoundError(
                f"{len(missing)} of {len(target_ids)} scenarios have no observed t=0 "
                f"frames in {observed_t0_dir}. Extract them before the run rather than "
                f"losing hours of GPU time partway through:\n"
                f"  python scripts/extract_t0_frames.py --limit {len(target_ids)}\n"
                f"{shown}{more}"
            )
    elif backbone == "svd":
        uncovered = [sc_id for sc_id in target_ids if sc_id not in cond_by_scenario]
        if uncovered:
            raise ValueError(
                f"the svd backbone needs the observed t=0 frame for every camera, and "
                f"{len(uncovered)} of {len(target_ids)} scenarios have none (e.g. "
                f"{uncovered[0]}); pass observed_t0_dir or conditioning_images"
            )

    pipeline = WorldModelPipeline(backbone=backbone, device=device)

    total = len(target_ids)
    for idx, sc_id in enumerate(target_ids):
        if sc_id not in trajs:
            raise KeyError(f"scenario '{sc_id}' not found in {trajectories_file}")
        cond_dict = cond_by_scenario.get(sc_id)
        pipeline.generate_and_save_scenario(
            scenario_id=sc_id,
            trajectory=trajs[sc_id],
            output_dir=output_dir,
            conditioning_images=cond_dict,
        )
        if progress_callback:
            progress_callback(idx + 1, total, sc_id)

    # Build and validate submission archive
    report = archive.build_archive(
        output_dir,
        output=zip_path,
        allow_partial=allow_partial,
        check_media=True,
    )
    return report
