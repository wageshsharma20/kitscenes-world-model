"""Shared Ego-Motion Representation & Temporal Motion Encoder for KITScenes CSSC.

Implements a unified 25-timestep kinematic ego-motion state representation:
    s_t = [x_t, y_t, psi_t, v_t, a_t, kappa_t]
along with numerically stable heading, acceleration, and curvature derivations,
sinusoidal temporal positional encodings, provenance metadata tracking, and a
PyTorch SharedMotionEncoder module producing 1024-dimensional cross-attention tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from kitscenes import spec


def compute_kinematics_from_waypoints(
    waypoints: np.ndarray,
    dt: float = 0.2,  # 5 Hz KITScenes challenge specification
    eps: float = 1e-5,
    max_curvature: float = 1.0,  # 1/m, minimum turn radius 1.0m
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute heading, velocity, acceleration, and curvature from 2D waypoints with strict numerical stability.

    Args:
        waypoints: (N, 2) array of (x, y) coordinates in ego vehicle frame.
        dt: Time delta between frames (seconds). Default 0.2s (5 Hz).
        eps: Small epsilon to prevent division by zero in stationary regimes.
        max_curvature: Maximum allowable absolute curvature in 1/meters.

    Returns:
        psi: (N,) Heading angles (radians).
        v: (N,) Linear speeds (m/s).
        a: (N,) Accelerations (m/s^2).
        kappa: (N,) Signed path curvature (1/m).
    """
    pts = np.asarray(waypoints, dtype=np.float64)
    n = pts.shape[0]
    if n == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
    if n == 1:
        return np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)

    # First derivatives (dx/dt, dy/dt) using forward, central, and backward differences
    vx = np.zeros(n, dtype=np.float64)
    vy = np.zeros(n, dtype=np.float64)

    # Start: forward difference
    vx[0] = (pts[1, 0] - pts[0, 0]) / dt
    vy[0] = (pts[1, 1] - pts[0, 1]) / dt

    # Interior: central differences
    if n > 2:
        vx[1:-1] = (pts[2:, 0] - pts[:-2, 0]) / (2.0 * dt)
        vy[1:-1] = (pts[2:, 1] - pts[:-2, 1]) / (2.0 * dt)

    # End: backward difference
    vx[-1] = (pts[-1, 0] - pts[-2, 0]) / dt
    vy[-1] = (pts[-1, 1] - pts[-2, 1]) / dt

    # Speed v = sqrt(vx^2 + vy^2)
    speed = np.sqrt(vx**2 + vy**2)

    # Heading psi = atan2(vy, vx)
    # When vehicle is stationary (speed < eps), propagate previous valid heading or default to 0.0
    psi = np.zeros(n, dtype=np.float64)
    last_valid_heading = 0.0
    for i in range(n):
        if speed[i] > eps:
            psi[i] = math.atan2(vy[i], vx[i])
            last_valid_heading = psi[i]
        else:
            psi[i] = last_valid_heading

    # Second derivatives (ax, ay)
    ax = np.zeros(n, dtype=np.float64)
    ay = np.zeros(n, dtype=np.float64)

    ax[0] = (vx[1] - vx[0]) / dt
    ay[0] = (vy[1] - vy[0]) / dt
    if n > 2:
        ax[1:-1] = (vx[2:] - vx[:-2]) / (2.0 * dt)
        ay[1:-1] = (vy[2:] - vy[:-2]) / (2.0 * dt)
    ax[-1] = (vx[-1] - vx[-2]) / dt
    ay[-1] = (vy[-1] - vy[-2]) / dt

    # Linear scalar acceleration along path: a = (vx*ax + vy*ay) / (v + eps)
    accel = np.where(speed > eps, (vx * ax + vy * ay) / np.maximum(speed, eps), 0.0)

    # Curvature: kappa = (vx * ay - vy * ax) / (speed^3 + eps)
    speed_cubed = speed**3
    kappa_raw = np.where(speed > eps, (vx * ay - vy * ax) / np.maximum(speed_cubed, eps**3), 0.0)
    kappa = np.clip(kappa_raw, -max_curvature, max_curvature)

    return psi, speed, accel, kappa


def build_sinusoidal_positional_encoding(
    timesteps: int,
    dim: int,
    max_period: float = 10000.0,
) -> torch.Tensor:
    """Generate standard sinusoidal positional embeddings [timesteps, dim]."""
    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
    )
    t = torch.arange(timesteps, dtype=torch.float32).unsqueeze(1)
    args = t * freqs.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros(timesteps, 1)], dim=-1)
    return embedding


@dataclass
class EgoMotionState:
    """Encapsulates a full 25-step kinematic trajectory with provenance metadata."""

    waypoints: np.ndarray  # [25, 2] in ego coordinates (x, y)
    dt: float = 0.2
    scenario_id: str = "unknown"
    source: str = "oracle"  # "oracle", "knn_retrieval_with_cv_blend", "cv_fallback", etc.
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Derived state components [25]
    psi: np.ndarray = field(init=False)
    v: np.ndarray = field(init=False)
    a: np.ndarray = field(init=False)
    kappa: np.ndarray = field(init=False)
    state_vector: np.ndarray = field(init=False)  # [25, 6]: [x, y, psi, v, a, kappa]

    def __post_init__(self) -> None:
        pts = np.asarray(self.waypoints, dtype=np.float64)
        if pts.shape[0] != spec.FRAME_COUNT or pts.shape[1] < 2:
            raise ValueError(
                f"Waypoints must have shape [{spec.FRAME_COUNT}, 2], got {pts.shape}"
            )
        self.waypoints = pts[:, :2]
        self.psi, self.v, self.a, self.kappa = compute_kinematics_from_waypoints(
            self.waypoints, dt=self.dt
        )
        self.state_vector = np.column_stack([
            self.waypoints[:, 0],
            self.waypoints[:, 1],
            self.psi,
            self.v,
            self.a,
            self.kappa,
        ]).astype(np.float32)

    @classmethod
    def from_waypoints(
        cls,
        waypoints: Sequence[Sequence[float]],
        scenario_id: str = "unknown",
        source: str = "oracle",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EgoMotionState:
        pts = np.asarray(waypoints, dtype=np.float64)
        if len(pts) < spec.FRAME_COUNT:
            # Resample or pad
            alphas = np.linspace(0.0, 1.0, spec.FRAME_COUNT)
            orig_alphas = np.linspace(0.0, 1.0, len(pts))
            pts = np.column_stack([
                np.interp(alphas, orig_alphas, pts[:, 0]),
                np.interp(alphas, orig_alphas, pts[:, 1]),
            ])
        elif len(pts) > spec.FRAME_COUNT:
            pts = pts[:spec.FRAME_COUNT]
        return cls(
            waypoints=pts,
            scenario_id=scenario_id,
            source=source,
            metadata=metadata or {},
        )

    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Return PyTorch tensor of shape [1, 25, 6]."""
        return torch.from_numpy(self.state_vector).unsqueeze(0).to(device=device)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "source": self.source,
            "dt": self.dt,
            "waypoints": self.waypoints.tolist(),
            "heading_rad": self.psi.tolist(),
            "speed_mps": self.v.tolist(),
            "accel_mps2": self.a.tolist(),
            "curvature_inv_m": self.kappa.tolist(),
            "metadata": self.metadata,
        }


class SharedMotionEncoder(nn.Module):
    """PyTorch module that maps 25-step kinematic ego-motion states into 1024-dim cross-attention tokens.

    Conforms to SVD-XT cross-attention projection contract:
        Input:  [B, 25, 6] (or [B, 25, D_in])
        Output: temporal_tokens: [B, 25, 1024]
                pooled_token:    [B, 1, 1024]
    """

    def __init__(
        self,
        in_dim: int = 6,
        model_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.05,
        num_timesteps: int = spec.FRAME_COUNT,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.model_dim = model_dim
        self.num_timesteps = num_timesteps

        # Input feature projection MLP with LayerNorm
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, model_dim),
            nn.LayerNorm(model_dim),
        )

        # Sinusoidal temporal positional encoding
        pe = build_sinusoidal_positional_encoding(num_timesteps, model_dim)
        self.register_buffer("positional_encoding", pe.unsqueeze(0))  # [1, 25, model_dim]

        # Temporal Transformer Encoder (pre-LN architecture)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection and final LayerNorm
        self.out_norm = nn.LayerNorm(model_dim)

        # Global sequence pooling attention head (bias=False because softmax is shift-invariant)
        self.pool_attn = nn.Linear(model_dim, 1, bias=False)

    def forward(
        self,
        motion_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            motion_states: Tensor of shape [B, 25, 6] (or [B, 25, in_dim]).

        Returns:
            temporal_tokens: [B, 25, 1024] token sequence for frame-wise cross-attention.
            pooled_token:    [B, 1, 1024] sequence summary token.
        """
        if motion_states.ndim == 2:
            # Add batch dimension if [25, 6]
            motion_states = motion_states.unsqueeze(0)

        b, t, d = motion_states.shape
        assert t == self.num_timesteps, f"Expected {self.num_timesteps} steps, got {t}"
        assert d == self.in_dim, f"Expected feature dim {self.in_dim}, got {d}"

        # 1. Project input kinematics to model_dim
        h = self.input_proj(motion_states)  # [B, 25, model_dim]

        # 2. Add sinusoidal temporal positional encodings
        h = h + self.positional_encoding[:, :t, :].to(dtype=h.dtype, device=h.device)

        # 3. Contextualize across temporal sequence via Transformer
        temporal_tokens = self.transformer(h)  # [B, 25, model_dim]
        temporal_tokens = self.out_norm(temporal_tokens)

        # 4. Attentive global pooling
        attn_weights = torch.softmax(self.pool_attn(temporal_tokens), dim=1)  # [B, 25, 1]
        pooled_token = torch.sum(temporal_tokens * attn_weights, dim=1, keepdim=True)  # [B, 1, model_dim]

        return temporal_tokens, pooled_token
