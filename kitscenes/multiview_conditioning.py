"""Calibration-Aware Shared-State Conditioning (CSSC) Architecture for KITScenes.

Provides multi-view physical coherence across the 6 surround cameras:
  1. CameraGeometryEncoder: Ingests certified K, R, t calibration matrices & camera ID
  2. SurroundSceneFusion: 360-degree multi-view attention across all 6 cameras at t=0
  3. SharedStateConditioner: Cross-attentive conditioning linking shared 25-step ego-motion
     with camera geometry and local view observations
  4. MultiViewConditioner: Top-level module producing 1024-dim conditioning tokens
     compatible with SVD-XT cross-attention and spatial adapters.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from kitscenes import spec
from kitscenes.shared_motion import SharedMotionEncoder
from kitscenes.video_engine import CameraRig


class CameraGeometryEncoder(nn.Module):
    """Encodes 6-camera pinhole calibration (K_out, R, t) and camera ID into 1024-dim geometry tokens."""

    def __init__(
        self,
        num_cameras: int = 6,
        model_dim: int = 1024,
        rig: Optional[CameraRig] = None,
    ):
        super().__init__()
        self.num_cameras = num_cameras
        self.model_dim = model_dim
        self.rig = rig or CameraRig()

        # Learnable camera ID embedding
        self.cam_id_embed = nn.Embedding(num_cameras, model_dim // 2)

        # Numerical calibration feature projection:
        # R (9) + t (3) + K_norm (4: fx, fy, cx, cy) = 16 parameters
        self.calib_proj = nn.Sequential(
            nn.Linear(16, model_dim // 4),
            nn.GELU(),
            nn.Linear(model_dim // 4, model_dim // 2),
            nn.LayerNorm(model_dim // 2),
        )

        # Combined projection to model_dim
        self.out_proj = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

        # Precompute static calibration tensors for the 6 surround cameras
        self._build_static_calibration_tensors()

    def _build_static_calibration_tensors(self) -> None:
        """Extract and normalize static calibration parameters for all 6 cameras."""
        feats = []
        for _i, cam_name in enumerate(spec.CAMERAS):
            cam = self.rig[cam_name]
            r = cam.rotation_matrix_body_to_cam.flatten()  # 9
            t = cam.translation_vector.flatten()  # 3
            # Normalized intrinsics: fx/W, fy/H, cx/W, cy/H
            k_norm = np.array([
                cam.fx / cam.width,
                cam.fy / cam.height,
                cam.cx / cam.width,
                cam.cy / cam.height,
            ], dtype=np.float32)  # 4
            feat = np.concatenate([r, t, k_norm]).astype(np.float32)
            feats.append(feat)

        calib_tensor = torch.from_numpy(np.stack(feats, axis=0))  # [6, 16]
        self.register_buffer("static_calibration_features", calib_tensor)

    def forward(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Return geometry tokens for all 6 surround cameras: [B, 6, 1024]."""
        calib_feats = self.static_calibration_features.to(device=device)  # [6, 16]
        calib_tokens = self.calib_proj(calib_feats)  # [6, 512]

        cam_indices = torch.arange(self.num_cameras, device=device)
        id_tokens = self.cam_id_embed(cam_indices)  # [6, 512]

        combined = torch.cat([id_tokens, calib_tokens], dim=-1)  # [6, 1024]
        tokens = self.out_proj(combined)  # [6, 1024]

        # Expand across batch dimension: [B, 6, 1024]
        return tokens.unsqueeze(0).expand(batch_size, -1, -1)


class SurroundSceneFusion(nn.Module):
    """Fuses 360-degree observations across all 6 surround cameras into a coherent surround representation."""

    def __init__(
        self,
        model_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.model_dim = model_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        camera_tokens: torch.Tensor,  # [B, 6, 1024]
        geometry_tokens: torch.Tensor,  # [B, 6, 1024]
    ) -> torch.Tensor:
        """Cross-view self-attention fusion.

        Returns:
            fused_surround_tokens: [B, 6, 1024] surround context tokens.
        """
        x = camera_tokens + geometry_tokens  # [B, 6, 1024]
        fused = self.fusion_transformer(x)
        return self.out_norm(fused)


class MultiViewConditioner(nn.Module):
    """Calibration-Aware Shared-State Conditioning (CSSC) Module.

    Integrates:
      1. 6-camera visual observations at t=0 (CLIP embeddings [B, 6, 1024])
      2. 6-camera certified calibration geometry (K, R, t)
      3. Shared 25-step kinematic ego-motion states (x, y, psi, v, a, kappa)
    Produces target camera conditioning tokens [B, 25, 1024] and pooled token [B, 1, 1024].
    """

    def __init__(
        self,
        model_dim: int = 1024,
        num_heads: int = 8,
        motion_in_dim: int = 6,
        rig: Optional[CameraRig] = None,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.rig = rig or CameraRig()

        # Submodules
        self.motion_encoder = SharedMotionEncoder(
            in_dim=motion_in_dim, model_dim=model_dim, dropout=dropout
        )
        self.geometry_encoder = CameraGeometryEncoder(
            num_cameras=6, model_dim=model_dim, rig=self.rig
        )
        self.scene_fusion = SurroundSceneFusion(
            model_dim=model_dim, num_heads=num_heads, dropout=dropout
        )

        # Cross-attention: target camera geometry + local observation queries shared temporal motion
        self.motion_cross_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(model_dim)

        # Temporal refinement MLP projecting cross-attended motion into final conditioning sequence
        self.refinement_mlp = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

        # Global pooled summary projection
        self.pool_proj = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
        )

    def forward(
        self,
        camera_clip_embeds: Union[torch.Tensor, Dict[str, torch.Tensor]],
        motion_states: torch.Tensor,
        target_camera: str = "front",
    ) -> Dict[str, torch.Tensor]:
        """Compute calibration-aware conditioning tokens for target camera.

        Args:
            camera_clip_embeds: Tensor of shape [B, 6, 1024] in spec.CAMERAS order,
                                OR dict {cam_name: [B, 1, 1024]}.
            motion_states: Kinematic state tensor [B, 25, 6].
            target_camera: Target camera name (e.g. 'front', 'rear', etc.).

        Returns:
            Dict containing:
              - 'conditioning_tokens': [B, 25, 1024] frame-wise cross-attention sequence
              - 'pooled_token':        [B, 1, 1024] global conditioning token
              - 'fused_surround':      [B, 6, 1024] 360-degree scene tokens
              - 'temporal_motion':     [B, 25, 1024] shared ego-motion tokens
        """
        assert target_camera in spec.CAMERAS, f"Unknown target camera: {target_camera}"
        cam_idx = spec.CAMERAS.index(target_camera)

        # Format camera CLIP embeddings into [B, 6, 1024]
        if isinstance(camera_clip_embeds, dict):
            embed_list = [camera_clip_embeds[c] for c in spec.CAMERAS]
            # Ensure each is [B, 1, 1024] -> concatenate to [B, 6, 1024]
            clip_seq = torch.cat(embed_list, dim=1)
        else:
            clip_seq = camera_clip_embeds
            if clip_seq.ndim == 2:
                clip_seq = clip_seq.unsqueeze(0)

        b, num_cams, d = clip_seq.shape
        assert num_cams == 6, f"Expected 6 camera embeddings, got {num_cams}"
        assert d == self.model_dim, f"Expected embedding dim {self.model_dim}, got {d}"

        # 1. Encode shared 25-step ego-motion: [B, 25, 1024] and [B, 1, 1024]
        temporal_motion, pooled_motion = self.motion_encoder(motion_states)

        # 2. Encode 6-camera calibrated geometry: [B, 6, 1024]
        geom_tokens = self.geometry_encoder(batch_size=b, device=clip_seq.device)

        # 3. Fuse 360-degree surround scene: [B, 6, 1024]
        fused_surround = self.scene_fusion(clip_seq, geom_tokens)

        # 4. Extract target camera view token: [B, 1, 1024]
        target_view_token = fused_surround[:, cam_idx : cam_idx + 1, :]  # [B, 1, 1024]

        # 5. Expand target view token across 25 temporal steps: [B, 25, 1024]
        target_query = target_view_token.expand(-1, spec.FRAME_COUNT, -1)

        # 6. Multi-head cross-attention: Target view queries shared temporal motion
        attn_out, _ = self.motion_cross_attn(
            query=target_query,
            key=temporal_motion,
            value=temporal_motion,
        )
        cross_tokens = self.cross_norm(target_query + attn_out)  # [B, 25, 1024]

        # 7. Refine with shared temporal motion representation
        combined_temporal = torch.cat([cross_tokens, temporal_motion], dim=-1)  # [B, 25, 2048]
        conditioning_tokens = self.refinement_mlp(combined_temporal)  # [B, 25, 1024]

        # 8. Compute combined global pooled token
        pooled_combined = torch.cat([target_view_token, pooled_motion], dim=-1)  # [B, 1, 2048]
        pooled_token = self.pool_proj(pooled_combined)  # [B, 1, 1024]

        return {
            "conditioning_tokens": conditioning_tokens,
            "pooled_token": pooled_token,
            "fused_surround": fused_surround,
            "temporal_motion": temporal_motion,
        }

    def forward_all_cameras(
        self,
        camera_clip_embeds: Union[torch.Tensor, Dict[str, torch.Tensor]],
        motion_states: torch.Tensor,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Compute conditioning tokens for all 6 cameras simultaneously."""
        return {
            cam: self.forward(camera_clip_embeds, motion_states, target_camera=cam)
            for cam in spec.CAMERAS
        }
