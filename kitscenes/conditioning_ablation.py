"""4-Way Conditioning Ablation Framework for KITScenes-LongTail World Model.

Supports 4 explicit conditioning modes:
  - Mode 0: NO_SPATIAL_CONDITIONING
    (Frozen SVD-XT baseline, scalar motion bucket only, no spatial masks)
  - Mode 1: OPTION_B1_ORACLE
    (Option B.1 initial-frame temporal masks using ground-truth future trajectories)
  - Mode 2: OPTION_B1_PREDICTED
    (Option B.1 initial-frame temporal masks using predicted kNN/CV trajectories)
  - Mode 3: OPTION_B2_ORACLE
    (Option B.2 future-camera spatial masks using ground-truth future trajectories)

Exposes a clean interface to generate conditioned latents, feature maps, and ablation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from kitscenes import camera_projection, spec
from kitscenes.knn_trajectory import KNNTrajectoryPlanner
from kitscenes.spatial_adapter import SpatialAdapter, SpatialAdapterOutput


class ConditioningMode(IntEnum):
    """4-way conditioning ablation modes."""

    NO_SPATIAL_CONDITIONING = 0
    OPTION_B1_ORACLE = 1
    OPTION_B1_PREDICTED = 2
    OPTION_B2_ORACLE = 3


def parse_conditioning_mode(mode: Union[int, str, ConditioningMode]) -> ConditioningMode:
    """Resolve an integer, string, or enum into a ConditioningMode enum."""
    if isinstance(mode, ConditioningMode):
        return mode
    if isinstance(mode, int):
        return ConditioningMode(mode)

    norm = str(mode).strip().lower()
    if norm in ("0", "no_spatial_conditioning", "mode_0", "baseline", "none"):
        return ConditioningMode.NO_SPATIAL_CONDITIONING
    if norm in ("1", "option_b1_oracle", "mode_1", "b1_oracle", "oracle_b1"):
        return ConditioningMode.OPTION_B1_ORACLE
    if norm in ("2", "option_b1_predicted", "mode_2", "b1_predicted", "predicted_b1", "knn_b1"):
        return ConditioningMode.OPTION_B1_PREDICTED
    if norm in ("3", "option_b2_oracle", "mode_3", "b2_oracle", "oracle_b2"):
        return ConditioningMode.OPTION_B2_ORACLE

    raise ValueError(
        f"Unknown conditioning mode: '{mode}'. Valid modes: "
        f"{[m.name for m in ConditioningMode]} (or integers 0..3)"
    )


@dataclass
class ConditioningResult:
    """Container for conditioning output of a single ablation mode."""

    mode: ConditioningMode
    mode_name: str
    spatial_mask: Optional[torch.Tensor]
    motion_bucket: int
    adapter_output: Optional[SpatialAdapterOutput] = None
    conditioned_latents: Optional[torch.Tensor] = None
    trajectory_used: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConditioningAblationFramework:
    """Manages the 4-way spatial conditioning ablation across SVD-XT pipelines."""

    def __init__(
        self,
        spatial_adapter: Optional[SpatialAdapter] = None,
        projection_engine: Optional[camera_projection.CameraProjectionEngine] = None,
        knn_planner: Optional[KNNTrajectoryPlanner] = None,
        donor_path: Optional[Union[str, Path]] = None,
        default_height: int = spec.HEIGHT,
        default_width: int = spec.WIDTH,
        default_num_frames: int = spec.FRAME_COUNT,
    ) -> None:
        self.spatial_adapter = spatial_adapter
        self.height = default_height
        self.width = default_width
        self.num_frames = default_num_frames

        # Projection engine
        if projection_engine is not None:
            self.engine = projection_engine
        else:
            self.engine = camera_projection.CameraProjectionEngine(
                width=self.width, height=self.height
            )

        # kNN Planner initialization
        if knn_planner is not None:
            self.knn_planner = knn_planner
        else:
            d_path = Path(donor_path) if donor_path else Path("data/train_trajectories.jsonl")
            if d_path.exists():
                self.knn_planner = KNNTrajectoryPlanner(donor_path=d_path)
            else:
                self.knn_planner = KNNTrajectoryPlanner(donor_path=None)

    def compute_motion_bucket(self, trajectory: Sequence[Sequence[float]]) -> int:
        """Derive standard SVD motion bucket ID (1 to 255) from trajectory displacement."""
        pts = np.asarray(trajectory, dtype=float)
        if len(pts) < 2:
            return 127
        diffs = np.diff(pts, axis=0)
        total_dist = float(np.sum(np.linalg.norm(diffs, axis=1)))
        mean_speed = total_dist / (len(pts) * spec.WAYPOINT_STEP_S)
        return int(np.clip(10 + mean_speed * 12.0, 1, 255))

    def _extract_sample_fields(
        self, sample_data: Union[Dict[str, Any], Any]
    ) -> Tuple[str, np.ndarray, np.ndarray, str, str]:
        """Extract scenario_id, past_traj, gt_traj, instruction, scenario_type."""
        if isinstance(sample_data, dict):
            sid = str(sample_data.get("scenario_id", "0000400"))
            instruction = str(sample_data.get("driving_instruction", "drive straight on"))
            stype = str(sample_data.get("scenario_type", ""))

            gt = sample_data.get("gt_trajectory", sample_data.get("expert_like"))
            if gt is None:
                # Default straight line future
                gt = np.column_stack([np.linspace(1.0, 25.0, self.num_frames), np.zeros(self.num_frames)])
            elif isinstance(gt, torch.Tensor):
                gt = gt.detach().cpu().numpy()
            else:
                gt = np.asarray(gt, dtype=np.float64)

            past = sample_data.get("past_trajectory", sample_data.get("past"))
            if past is None:
                # Default past moving forward
                past = np.column_stack([np.linspace(-10.0, 0.0, 21), np.zeros(21)])
            elif isinstance(past, torch.Tensor):
                past = past.detach().cpu().numpy()
            else:
                past = np.asarray(past, dtype=np.float64)

            return sid, past, gt, instruction, stype

        raise TypeError(f"Expected dict-like sample, got {type(sample_data)}")

    def generate_mask(
        self,
        mode: Union[int, str, ConditioningMode],
        sample_data: Dict[str, Any],
        camera_name: str = "front",
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], int, Optional[np.ndarray], Dict[str, Any]]:
        """Generate spatial mask and motion bucket for a specific ablation mode.

        Returns:
            Tuple of:
              - mask: Tensor of shape [1, T, 1, H, W] (or zeros tensor for Mode 0)
              - motion_bucket: SVD motion bucket integer [1..255]
              - trajectory_used: (25, 2) trajectory used to render the mask
              - metadata: dict of ablation metadata
        """
        c_mode = parse_conditioning_mode(mode)
        sid, past_traj, gt_traj, instruction, stype = self._extract_sample_fields(sample_data)
        n_frames = num_frames or self.num_frames
        h = height or self.height
        w = width or self.width

        # Base motion bucket
        motion_bucket = self.compute_motion_bucket(gt_traj)

        if c_mode == ConditioningMode.NO_SPATIAL_CONDITIONING:
            # Mode 0: Zero mask (unconditioned baseline)
            zero_mask = torch.zeros((1, n_frames, 1, h, w), dtype=torch.float32)
            meta = {
                "mode": c_mode.name,
                "mode_id": int(c_mode),
                "is_spatial_conditioned": False,
                "description": "Baseline frozen SVD-XT, motion bucket only, zero spatial mask",
                "mask_mean": 0.0,
            }
            return zero_mask, motion_bucket, None, meta

        elif c_mode == ConditioningMode.OPTION_B1_ORACLE:
            # Mode 1: Option B.1 Initial-Frame masks using GT trajectory
            # Check if pre-rendered in sample
            if "initial_frame_temporal_masks" in sample_data and isinstance(
                sample_data["initial_frame_temporal_masks"], (torch.Tensor, np.ndarray)
            ):
                m = sample_data["initial_frame_temporal_masks"]
                if isinstance(m, np.ndarray):
                    m = torch.from_numpy(m)
                if m.ndim == 4:
                    mask_tensor = m.unsqueeze(0).float()
                else:
                    mask_tensor = m.float()
            else:
                raw_mask = self.engine.render_initial_frame_temporal_masks(
                    camera_name=camera_name,
                    waypoints_ego=gt_traj,
                    num_frames=n_frames,
                    channel_dim=True,
                )
                mask_tensor = torch.from_numpy(raw_mask).unsqueeze(0).float()

            meta = {
                "mode": c_mode.name,
                "mode_id": int(c_mode),
                "is_spatial_conditioned": True,
                "description": "Option B.1 Route-Initial-Frame Temporal Masks (Oracle GT)",
                "mask_mean": float(mask_tensor.mean().item()),
                "mask_max": float(mask_tensor.max().item()),
            }
            return mask_tensor, motion_bucket, gt_traj, meta

        elif c_mode == ConditioningMode.OPTION_B1_PREDICTED:
            # Mode 2: Option B.1 Initial-Frame masks using predicted kNN trajectory
            pred_traj = self.knn_planner.plan(
                scenario_id=sid,
                past_trajectory=past_traj,
                driving_instruction=instruction,
                scenario_type=stype,
            )
            pred_traj_sliced = pred_traj[:n_frames]
            raw_mask = self.engine.render_initial_frame_temporal_masks(
                camera_name=camera_name,
                waypoints_ego=pred_traj_sliced,
                num_frames=n_frames,
                channel_dim=True,
            )
            mask_tensor = torch.from_numpy(raw_mask).unsqueeze(0).float()
            n_eval = min(len(pred_traj_sliced), len(gt_traj))
            l2_err = float(np.mean(np.linalg.norm(pred_traj_sliced[:n_eval] - gt_traj[:n_eval], axis=1)))
            meta = {
                "mode": c_mode.name,
                "mode_id": int(c_mode),
                "is_spatial_conditioned": True,
                "description": "Option B.1 Route-Initial-Frame Temporal Masks (Predicted kNN)",
                "mask_mean": float(mask_tensor.mean().item()),
                "mask_max": float(mask_tensor.max().item()),
                "l2_to_gt": l2_err,
            }
            return mask_tensor, motion_bucket, pred_traj_sliced, meta

        elif c_mode == ConditioningMode.OPTION_B2_ORACLE:
            # Mode 3: Option B.2 Future-Camera spatial masks using GT trajectory
            if "future_camera_spatial_masks" in sample_data and isinstance(
                sample_data["future_camera_spatial_masks"], (torch.Tensor, np.ndarray)
            ):
                m = sample_data["future_camera_spatial_masks"]
                if isinstance(m, np.ndarray):
                    m = torch.from_numpy(m)
                if m.ndim == 4:
                    mask_tensor = m.unsqueeze(0).float()
                else:
                    mask_tensor = m.float()
            else:
                raw_mask = self.engine.render_future_camera_spatial_masks(
                    camera_name=camera_name,
                    waypoints_ego=gt_traj,
                    num_frames=n_frames,
                    channel_dim=True,
                )
                mask_tensor = torch.from_numpy(raw_mask).unsqueeze(0).float()

            meta = {
                "mode": c_mode.name,
                "mode_id": int(c_mode),
                "is_spatial_conditioned": True,
                "description": "Option B.2 Future-Camera Spatial Masks (Ego-Motion Aligned Oracle GT)",
                "mask_mean": float(mask_tensor.mean().item()),
                "mask_max": float(mask_tensor.max().item()),
            }
            return mask_tensor, motion_bucket, gt_traj, meta

        raise RuntimeError("Unreachable mode state.")

    def generate_conditioned_features(
        self,
        mode: Union[int, str, ConditioningMode],
        sample_data: Dict[str, Any],
        camera_name: str = "front",
        target_latent_shape: Optional[Tuple[int, int]] = None,
        adapter: Optional[SpatialAdapter] = None,
    ) -> ConditioningResult:
        """Generate conditioned latents and feature maps for the given mode.

        Args:
            mode: ConditioningMode (0..3).
            sample_data: Data record or dataset sample.
            camera_name: Target camera identifier ('front', etc.).
            target_latent_shape: Target latent (H_lat, W_lat) resolution.
            adapter: Optional SpatialAdapter instance (defaults to self.spatial_adapter).

        Returns:
            ConditioningResult with spatial_mask, adapter_output, and metadata.
        """
        c_mode = parse_conditioning_mode(mode)
        mask, bucket, traj_used, meta = self.generate_mask(
            mode=c_mode,
            sample_data=sample_data,
            camera_name=camera_name,
        )

        use_adapter = adapter or self.spatial_adapter
        adapter_out: Optional[SpatialAdapterOutput] = None

        if use_adapter is not None:
            if c_mode == ConditioningMode.NO_SPATIAL_CONDITIONING:
                # Mode 0: Guarantee strictly zero conditioning features
                b_size = mask.shape[0]
                t_size = mask.shape[1]
                c_lat = use_adapter.config.latent_channels
                h_lat = target_latent_shape[0] if target_latent_shape else self.height // 8
                w_lat = target_latent_shape[1] if target_latent_shape else self.width // 8
                zero_latents = torch.zeros((b_size, t_size, c_lat, h_lat, w_lat), dtype=mask.dtype, device=mask.device)
                adapter_out = SpatialAdapterOutput(
                    latent_conditioning=zero_latents,
                    down_block_residuals=None,
                    conditioning_type="zero_baseline",
                    metadata={"mode": c_mode.name, "active": False},
                )
            else:
                adapter_out = use_adapter(mask, target_latent_shape=target_latent_shape)

        return ConditioningResult(
            mode=c_mode,
            mode_name=c_mode.name,
            spatial_mask=mask,
            motion_bucket=bucket,
            adapter_output=adapter_out,
            trajectory_used=traj_used,
            metadata=meta,
        )

    def apply_conditioning_to_latents(
        self,
        mode: Union[int, str, ConditioningMode],
        noisy_latents: torch.Tensor,
        sample_data: Dict[str, Any],
        camera_name: str = "front",
        adapter: Optional[SpatialAdapter] = None,
    ) -> Tuple[torch.Tensor, ConditioningResult]:
        """Apply conditioning to an input noisy latent tensor [B, T, C_latent, H_lat, W_lat].

        Returns:
            Tuple of (conditioned_latents, conditioning_result).
        """
        c_mode = parse_conditioning_mode(mode)
        tgt_shape = (noisy_latents.shape[-2], noisy_latents.shape[-1])
        res = self.generate_conditioned_features(
            mode=c_mode,
            sample_data=sample_data,
            camera_name=camera_name,
            target_latent_shape=tgt_shape,
            adapter=adapter,
        )

        if c_mode == ConditioningMode.NO_SPATIAL_CONDITIONING:
            # Mode 0: baseline noisy latents strictly unchanged
            cond_latents = noisy_latents
        else:
            assert res.adapter_output is not None, "Adapter output required for spatial conditioning"
            cond_latents = noisy_latents + res.adapter_output.latent_conditioning

        res.conditioned_latents = cond_latents
        return cond_latents, res

    def run_4way_ablation(
        self,
        sample_data: Dict[str, Any],
        camera_name: str = "front",
        target_latent_shape: Optional[Tuple[int, int]] = None,
        adapter: Optional[SpatialAdapter] = None,
    ) -> Dict[str, ConditioningResult]:
        """Execute all 4 conditioning modes on a single scenario sample.

        Returns:
            Dictionary mapping mode name to ConditioningResult.
        """
        results: Dict[str, ConditioningResult] = {}
        for mode in ConditioningMode:
            res = self.generate_conditioned_features(
                mode=mode,
                sample_data=sample_data,
                camera_name=camera_name,
                target_latent_shape=target_latent_shape,
                adapter=adapter,
            )
            results[mode.name] = res

        return results
