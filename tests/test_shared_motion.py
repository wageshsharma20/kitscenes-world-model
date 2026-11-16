"""Unit tests for shared ego-motion kinematic state representation and encoder."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kitscenes.shared_motion import (
    EgoMotionState,
    SharedMotionEncoder,
    compute_kinematics_from_waypoints,
)


def test_stationary_kinematics_numerical_stability():
    """Verify stationary vehicle (zero displacement) produces finite zeros with zero NaNs/Infs."""
    waypoints = np.zeros((25, 2), dtype=np.float64)
    psi, speed, accel, kappa = compute_kinematics_from_waypoints(waypoints)

    assert len(psi) == 25
    assert not np.any(np.isnan(psi))
    assert not np.any(np.isnan(speed))
    assert not np.any(np.isnan(accel))
    assert not np.any(np.isnan(kappa))

    assert np.allclose(speed, 0.0)
    assert np.allclose(accel, 0.0)
    assert np.allclose(kappa, 0.0)


def test_straight_acceleration_kinematics():
    """Verify constant acceleration straight path computes accurate speed and zero curvature."""
    # x(t) = 0.5 * a * t^2, a = 2.0 m/s^2, dt = 0.2s, 25 steps (t = 0.2 to 5.0s)
    dt = 0.2
    t = np.arange(1, 26) * dt
    a_const = 2.0
    x = 0.5 * a_const * (t**2)
    y = np.zeros_like(x)
    waypoints = np.column_stack([x, y])

    psi, speed, accel, kappa = compute_kinematics_from_waypoints(waypoints, dt=dt)

    # Heading should be 0.0 rad (straight forward along +X)
    assert np.allclose(psi, 0.0, atol=1e-5)

    # Speed should be strictly monotonically increasing
    assert np.all(np.diff(speed) > 0.0)

    # Curvature should be strictly 0.0 on straight line
    assert np.allclose(kappa, 0.0, atol=1e-5)


def test_constant_radius_circular_turn():
    """Verify circular arc produces constant curvature kappa = 1 / R."""
    radius = 20.0  # meters
    # Turning left: x = R * sin(theta), y = R * (1 - cos(theta))
    theta = np.linspace(0.05, 0.8, 25)
    x = radius * np.sin(theta)
    y = radius * (1.0 - np.cos(theta))
    waypoints = np.column_stack([x, y])

    psi, speed, accel, kappa = compute_kinematics_from_waypoints(waypoints)

    expected_kappa = 1.0 / radius  # 0.05 1/m
    # Central interior steps should match 1/R closely
    interior_kappa = kappa[3:-3]
    assert np.allclose(interior_kappa, expected_kappa, atol=0.01), f"Expected {expected_kappa}, got {interior_kappa}"


def test_ego_motion_state_dataclass_contract():
    """Verify EgoMotionState post-init validation, shape contract, and dictionary export."""
    pts = np.column_stack([np.linspace(1.0, 25.0, 25), np.zeros(25)])
    state = EgoMotionState(
        waypoints=pts,
        scenario_id="0000400",
        source="knn_retrieval_with_cv_blend",
        metadata={"donor_count": 5},
    )

    assert state.state_vector.shape == (25, 6)
    assert state.state_vector.dtype == np.float32

    # Test tensor export
    t_tensor = state.to_tensor()
    assert t_tensor.shape == (1, 25, 6)
    assert t_tensor.dtype == torch.float32

    # Test dictionary export
    d = state.to_dict()
    assert d["scenario_id"] == "0000400"
    assert d["source"] == "knn_retrieval_with_cv_blend"
    assert len(d["waypoints"]) == 25
    assert len(d["heading_rad"]) == 25
    assert len(d["speed_mps"]) == 25
    assert len(d["accel_mps2"]) == 25
    assert len(d["curvature_inv_m"]) == 25
    assert d["metadata"]["donor_count"] == 5

    # Shape validation error on wrong input length
    with pytest.raises(ValueError, match="Waypoints must have shape"):
        EgoMotionState(waypoints=np.zeros((10, 2)))


def test_shared_motion_encoder_forward_and_shapes():
    """Verify SharedMotionEncoder produces [B, 25, 1024] temporal tokens and [B, 1, 1024] pooled token."""
    encoder = SharedMotionEncoder(in_dim=6, model_dim=1024, num_layers=2)
    encoder.eval()

    # Batch of 4 trajectories
    x = torch.randn(4, 25, 6)
    with torch.no_grad():
        temporal_tokens, pooled_token = encoder(x)

    assert temporal_tokens.shape == (4, 25, 1024)
    assert pooled_token.shape == (4, 1, 1024)
    assert not torch.isnan(temporal_tokens).any()
    assert not torch.isnan(pooled_token).any()


def test_shared_motion_encoder_gradient_flow():
    """Verify backward pass propagates gradients through all layers of SharedMotionEncoder."""
    encoder = SharedMotionEncoder(in_dim=6, model_dim=256, num_layers=2)  # smaller dim for fast unit test
    encoder.train()

    x = torch.randn(2, 25, 6, requires_grad=True)
    temporal_tokens, pooled_token = encoder(x)
    loss = (temporal_tokens ** 2).mean() + (pooled_token ** 2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.norm(x.grad) > 0.0

    # Every parameter should have non-zero gradients
    for name, p in encoder.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient!"
        assert torch.norm(p.grad) > 0.0, f"Parameter {name} has zero gradient!"


def test_batch_invariance():
    """Verify each sequence in a batch produces the exact same output as when evaluated singly."""
    encoder = SharedMotionEncoder(in_dim=6, model_dim=256, num_layers=2, dropout=0.0)
    encoder.eval()

    x1 = torch.randn(1, 25, 6)
    x2 = torch.randn(1, 25, 6)
    x_batch = torch.cat([x1, x2], dim=0)

    with torch.no_grad():
        t_batch, p_batch = encoder(x_batch)
        t1, p1 = encoder(x1)
        t2, p2 = encoder(x2)

    assert torch.allclose(t_batch[0:1], t1, atol=1e-5)
    assert torch.allclose(t_batch[1:2], t2, atol=1e-5)
    assert torch.allclose(p_batch[0:1], p1, atol=1e-5)
    assert torch.allclose(p_batch[1:2], p2, atol=1e-5)
