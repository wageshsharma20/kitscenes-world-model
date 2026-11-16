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

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kitscenes import spec
from kitscenes.shared_motion import SharedMotionEncoder
from kitscenes.spatial_adapter import (
    MockSVDUNet,
    SpatialAdapter,
    SpatialAdapterConfig,
    c_noise_to_sigma,
    compute_svd_noise_tensors,
    sample_training_noise_levels,
    sigma_to_c_noise,
)
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

        # 9. Assemble 27-token shared context bank for SVD-XT cross-attention:
        # [target_view_token (1), pooled_token (1), conditioning_tokens (25)] -> [B, 27, 1024]
        encoder_hidden_states = torch.cat(
            [target_view_token, pooled_token, conditioning_tokens], dim=1
        )  # [B, 27, 1024]

        return {
            "conditioning_tokens": conditioning_tokens,
            "pooled_token": pooled_token,
            "target_view_token": target_view_token,
            "encoder_hidden_states": encoder_hidden_states,
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


class CSSCConditionedSVDModel(nn.Module):
    """End-to-end model wrapper integrating SVD-XT UNet with CSSC and SpatialAdapter.

    Full Conditioning Pipeline:
      1. MultiViewConditioner: Fuses 6-camera visual observations at t=0, certified pinhole
         calibration geometry, and 25-step shared kinematic ego-motion into a 27-token
         shared context bank: [target_view_token (1), pooled_summary (1), temporal_motion (25)].
      2. Diffusers UNet Cross-Attention: When encoder_hidden_states [B, 27, 1024] is passed,
         Diffusers internally performs repeat_interleave(num_frames, dim=0), allowing all 25
         generated video frames to cross-attend to any of the 27 context tokens.
      3. SpatialAdapter: Optionally injects frame-aligned 5D control masks [B, T, 1, H, W]
         directly into noisy video latents before UNet processing.
      4. SVD-XT UNet: Supports both MockSVDUNet and genuine diffusers.UNetSpatioTemporalConditionModel.

    Freezing Contract:
      - Backbones (UNet, VAE, CLIP Image Encoder) are strictly frozen (requires_grad = False).
      - MultiViewConditioner and SpatialAdapter parameters are trainable (requires_grad = True).
    """

    def __init__(
        self,
        unet: Optional[nn.Module] = None,
        conditioner: Optional[MultiViewConditioner] = None,
        spatial_adapter: Optional[SpatialAdapter] = None,
        adapter_config: Optional[SpatialAdapterConfig] = None,
        vae: Optional[nn.Module] = None,
        image_encoder: Optional[nn.Module] = None,
        scheduler: Optional[Any] = None,
        prediction_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.conditioner = conditioner or MultiViewConditioner()

        if spatial_adapter is not None:
            self.spatial_adapter = spatial_adapter
        elif adapter_config is not None:
            self.spatial_adapter = SpatialAdapter(adapter_config)
        else:
            self.spatial_adapter = SpatialAdapter(SpatialAdapterConfig())

        self.unet = unet or MockSVDUNet(
            in_channels=8,
            out_channels=4,
            block_out_channels=(320, 640, 1280, 1280),
        )
        self.vae = vae
        self.image_encoder = image_encoder
        self.scheduler = scheduler

        if prediction_type is not None:
            self.prediction_type = prediction_type
        elif hasattr(self.unet, "config") and getattr(self.unet.config, "prediction_type", None):
            self.prediction_type = self.unet.config.prediction_type
        else:
            self.prediction_type = "v_prediction"

        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze all parameters in UNet, VAE, and Image Encoder."""
        if self.unet is not None:
            self.unet.eval()
            for p in self.unet.parameters():
                p.requires_grad = False

        if self.vae is not None:
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad = False

        if self.image_encoder is not None:
            self.image_encoder.eval()
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        # Ensure conditioner and spatial adapter remain trainable
        self.conditioner.train()
        for p in self.conditioner.parameters():
            p.requires_grad = True

        if self.spatial_adapter is not None:
            self.spatial_adapter.train()
            for p in self.spatial_adapter.parameters():
                p.requires_grad = True

    def count_parameters(self) -> Dict[str, Any]:
        """Audit trainable CSSC + adapter parameters vs frozen backbone parameters."""
        conditioner_trainable = sum(p.numel() for p in self.conditioner.parameters() if p.requires_grad)
        adapter_trainable = (
            sum(p.numel() for p in self.spatial_adapter.parameters() if p.requires_grad)
            if self.spatial_adapter is not None
            else 0
        )
        total_trainable = conditioner_trainable + adapter_trainable
        backbone_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())

        return {
            "trainable_conditioner_params": conditioner_trainable,
            "trainable_adapter_params": adapter_trainable,
            "total_trainable_params": total_trainable,
            "frozen_backbone_params": backbone_frozen,
            "total_params": total_params,
        }

    def train(self, mode: bool = True) -> "CSSCConditionedSVDModel":
        """Set training mode while strictly keeping backbone in eval mode."""
        super().train(mode)
        if self.unet is not None:
            self.unet.eval()
        if self.vae is not None:
            self.vae.eval()
        if self.image_encoder is not None:
            self.image_encoder.eval()
        if hasattr(self, "conditioner"):
            self.conditioner.train(mode)
        if hasattr(self, "spatial_adapter") and self.spatial_adapter is not None:
            self.spatial_adapter.train(mode)
        return self

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: Union[int, float, torch.Tensor],
        camera_clip_embeds: Union[torch.Tensor, Dict[str, torch.Tensor]],
        motion_states: torch.Tensor,
        target_camera: str = "front",
        initial_image_latents: Optional[torch.Tensor] = None,
        spatial_masks: Optional[torch.Tensor] = None,
        added_time_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Any:
        """Forward pass through CSSC conditioning, spatial adapter, and SVD-XT UNet.

        Args:
            noisy_latents: Tensor of shape [B, T, 4, H_lat, W_lat] (or [B, T, 8, H_lat, W_lat] if pre-concatenated).
            timesteps: Continuous SVD scheduler timestep (c_noise) or tensor [B].
            camera_clip_embeds: 6-camera CLIP embeds [B, 6, 1024] or dict {cam: [B, 1, 1024]}.
            motion_states: 25-step kinematic state tensor [B, 25, 6].
            target_camera: Target camera name (e.g. 'front', 'rear', etc.).
            initial_image_latents: Conditioning frame latents [B, 4, H_lat, W_lat] or [B, T, 4, H_lat, W_lat].
            spatial_masks: Optional 5D spatial control masks [B, T, 1, H, W].
            added_time_ids: Optional micro-conditioning IDs [B, 3].
            return_dict: Whether to return a dict with extra metadata.

        Returns:
            UNet output sample [B, T, 4, H_lat, W_lat] (or dict if return_dict=True).
        """
        b, t = noisy_latents.shape[:2]
        device = noisy_latents.device
        dtype = noisy_latents.dtype

        # 1. Compute CSSC conditioning tokens and 27-token context bank
        cssc_out = self.conditioner(
            camera_clip_embeds=camera_clip_embeds,
            motion_states=motion_states,
            target_camera=target_camera,
        )
        encoder_hidden_states = cssc_out["encoder_hidden_states"]  # [B, 27, 1024]

        # 2. Apply SpatialAdapter if masks are provided
        if spatial_masks is not None and self.spatial_adapter is not None:
            conditioned_latents = self.spatial_adapter.apply_to_latent(noisy_latents, spatial_masks)
        else:
            conditioned_latents = noisy_latents

        # 3. Concatenate with initial_image_latents to form 8-channel UNet input
        if initial_image_latents is not None:
            if initial_image_latents.ndim == 4:
                cond_latents = initial_image_latents.unsqueeze(1).repeat(1, t, 1, 1, 1)
            elif initial_image_latents.shape[1] == 1 and t > 1:
                cond_latents = initial_image_latents.repeat(1, t, 1, 1, 1)
            else:
                cond_latents = initial_image_latents
            unet_input = torch.cat([conditioned_latents, cond_latents], dim=2)
        elif conditioned_latents.shape[2] == 8:
            unet_input = conditioned_latents
        else:
            raise ValueError(
                f"Expected 8-channel UNet input or initial_image_latents, got {conditioned_latents.shape[2]} channels"
            )

        # 4. Prepare UNet kwargs
        unet_kwargs: Dict[str, Any] = {
            "encoder_hidden_states": encoder_hidden_states,
        }
        if added_time_ids is not None:
            unet_kwargs["added_time_ids"] = added_time_ids
        elif hasattr(self.unet, "config") and getattr(self.unet.config, "addition_time_embed_dim", None):
            unet_kwargs["added_time_ids"] = torch.tensor(
                [[4.0, 127.0, 0.02]], device=device, dtype=dtype
            ).repeat(b, 1)

        # Ensure timestep is properly shaped tensor
        t_val = timesteps if isinstance(timesteps, torch.Tensor) else torch.tensor(timesteps, device=device, dtype=dtype)
        if t_val.ndim == 0:
            t_val = t_val.unsqueeze(0).repeat(b)
        elif t_val.shape[0] == 1 and b > 1:
            t_val = t_val.repeat(b)

        # 5. Forward through UNet
        try:
            unet_out = self.unet(unet_input, timestep=t_val, **unet_kwargs)
        except TypeError:
            unet_out = self.unet(unet_input, timestep=t_val)

        sample = unet_out.sample if hasattr(unet_out, "sample") else unet_out
        if isinstance(sample, tuple):
            sample = sample[0]

        if not return_dict:
            return sample

        return {
            "sample": sample,
            "encoder_hidden_states": encoder_hidden_states,
            "cssc_out": cssc_out,
        }

    def compute_latent_loss(
        self,
        target_latents: torch.Tensor,
        initial_image_latents: torch.Tensor,
        camera_clip_embeds: Union[torch.Tensor, Dict[str, torch.Tensor]],
        motion_states: torch.Tensor,
        target_camera: str = "front",
        spatial_masks: Optional[torch.Tensor] = None,
        timesteps: Optional[Union[int, float, torch.Tensor]] = None,
        c_noise: Optional[torch.Tensor] = None,
        sigmas: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        added_time_ids: Optional[torch.Tensor] = None,
        loss_type: str = "v_mse",
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute end-to-end training loss for CSSC + SVD-XT following SVD v-prediction math."""
        b, t, c, h_lat, w_lat = target_latents.shape
        device = target_latents.device
        dtype = target_latents.dtype

        if initial_image_latents.ndim == 4:
            cond_latents = initial_image_latents.unsqueeze(1).repeat(1, t, 1, 1, 1)
        elif initial_image_latents.shape[1] == 1 and t > 1:
            cond_latents = initial_image_latents.repeat(1, t, 1, 1, 1)
        else:
            cond_latents = initial_image_latents

        eps = noise if noise is not None else torch.randn_like(target_latents)

        if sigmas is not None:
            sig = sigmas.to(device=device, dtype=dtype)
            c_n = sigma_to_c_noise(sig) if c_noise is None else c_noise.to(device=device, dtype=dtype)
        elif c_noise is not None:
            c_n = c_noise.to(device=device, dtype=dtype)
            sig = c_noise_to_sigma(c_n)
        elif timesteps is not None:
            t_val = timesteps if isinstance(timesteps, torch.Tensor) else torch.tensor(timesteps, device=device, dtype=dtype)
            if t_val.ndim == 0:
                t_val = t_val.unsqueeze(0).repeat(b)
            elif t_val.shape[0] == 1 and b > 1:
                t_val = t_val.repeat(b)
            t_val = t_val.to(device=device, dtype=dtype)
            if t_val.max() > 10.0:
                t_norm = torch.clamp(t_val / 1000.0, 0.0, 1.0)
                c_n = t_norm * 3.19 - 1.55
                sig = c_noise_to_sigma(c_n)
            else:
                c_n = t_val
                sig = c_noise_to_sigma(c_n)
        else:
            if self.training:
                sig, c_n = sample_training_noise_levels(b, loc=1.0, scale=1.6, device=device, dtype=dtype)
            else:
                sig = torch.full((b,), 15.0, device=device, dtype=dtype)
                c_n = sigma_to_c_noise(sig)

        z_sigma, model_input, v_target, c_out, c_skip = compute_svd_noise_tensors(
            clean_latents=target_latents,
            noise=eps,
            sigmas=sig,
        )

        fwd_out = self.forward(
            noisy_latents=model_input,
            timesteps=c_n,
            camera_clip_embeds=camera_clip_embeds,
            motion_states=motion_states,
            target_camera=target_camera,
            initial_image_latents=cond_latents,
            spatial_masks=spatial_masks,
            added_time_ids=added_time_ids,
            return_dict=True,
        )
        pred_v = fwd_out["sample"]

        if self.prediction_type == "v_prediction":
            denoised_latents = pred_v * c_out + z_sigma * c_skip
            sig_reshaped = sig
            while len(sig_reshaped.shape) < len(target_latents.shape):
                sig_reshaped = sig_reshaped.unsqueeze(-1)
            pred_eps = pred_v / torch.sqrt(1.0 + sig_reshaped**2) + z_sigma * (sig_reshaped / (1.0 + sig_reshaped**2))
        else:
            pred_eps = pred_v
            sig_reshaped = sig
            while len(sig_reshaped.shape) < len(target_latents.shape):
                sig_reshaped = sig_reshaped.unsqueeze(-1)
            denoised_latents = z_sigma - sig_reshaped * pred_eps

        v_loss = F.mse_loss(pred_v, v_target)
        unweighted_latent_loss = F.mse_loss(denoised_latents, target_latents)

        sig_reshaped_sq = sig_reshaped**2
        sig_weight = (1.0 + sig_reshaped_sq) / (sig_reshaped_sq + 1e-7)
        weighted_latent_loss = torch.mean(sig_weight * (denoised_latents - target_latents) ** 2)

        if loss_type == "v_mse":
            loss = v_loss
        elif loss_type == "weighted_latent_mse":
            loss = weighted_latent_loss
        elif loss_type in ("noise_mse", "eps_mse"):
            loss = F.mse_loss(pred_eps, eps)
        elif loss_type == "latent_mse":
            loss = unweighted_latent_loss
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        metrics = {
            "loss": loss.item(),
            "v_loss": v_loss.item(),
            "unweighted_latent_loss": unweighted_latent_loss.item(),
            "weighted_latent_loss": weighted_latent_loss.item(),
            "sigmas_mean": sig.mean().item(),
            "c_noise_mean": c_n.mean().item(),
        }
        return loss, metrics
