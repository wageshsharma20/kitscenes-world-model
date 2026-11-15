"""Unit and regression tests for 3D-to-2D multi-camera projection and spatial conditioning."""

from __future__ import annotations

import numpy as np
import pytest

from kitscenes import camera_projection, spec


def test_front_camera_straight_motion():
    """Verify forward straight path projects to horizontal center and lower road half of front camera."""
    engine = camera_projection.CameraProjectionEngine()

    # Waypoints moving straight ahead: X = 2.0 to 50.0m, Y = 0.0m
    straight_waypoints = np.column_stack([
        np.linspace(2.0, 50.0, 25),
        np.zeros(25),
    ])

    # Project to front camera
    front_view = engine.project_waypoints_to_camera("front", straight_waypoints)

    assert front_view.camera_name == "front"
    assert front_view.visible_count > 0

    # 1. Horizontal centering: all points should lie near center column cx (438 px)
    cx = spec.WIDTH / 2.0
    u_pts = front_view.points_2d[front_view.is_visible, 0]
    assert np.allclose(u_pts, cx, atol=2.0), f"Straight path not centered horizontally: {u_pts}"

    # 2. Vertical position: ground is below camera mounting height (1.4m), so v > cy (284 px)
    cy = spec.HEIGHT / 2.0
    v_pts = front_view.points_2d[front_view.is_visible, 1]
    assert np.all(v_pts > cy), "Ground road waypoints must project below optical horizon"

    # 3. Perspective depth: points further away approach horizon cy as depth increases
    assert v_pts[-1] < v_pts[0], "Distant waypoints must appear closer to the horizon than near waypoints"

    # 4. Rear camera: points ahead must NOT be visible in rear camera
    rear_view = engine.project_waypoints_to_camera("rear", straight_waypoints)
    assert rear_view.visible_count == 0, "Forward points must not be visible in rear-facing camera"


def test_right_turn_camera_transition():
    """Verify a right turn curves to the right in front camera and enters front-right view."""
    engine = camera_projection.CameraProjectionEngine()

    # Arc turning 90 degrees right: X forward, Y negative (right)
    theta = np.linspace(0, np.pi / 2.0, 25)
    radius = 15.0
    turn_x = radius * np.sin(theta) + 2.0
    turn_y = -(radius * (1.0 - np.cos(theta)))
    right_turn_waypoints = np.column_stack([turn_x, turn_y])

    all_views = engine.project_all_cameras(right_turn_waypoints)

    front_proj = all_views["front"]
    front_right_proj = all_views["front_right"]
    front_left_proj = all_views["front_left"]

    # In front camera, u coordinate must drift to the right (u > cx) as the car turns right
    cx = spec.WIDTH / 2.0
    vis_front = front_proj.points_2d[front_proj.is_visible]
    assert vis_front[-1, 0] > cx + 50.0, "Right-turn waypoints must shift rightward on front canvas"

    # Front-right camera (yaw = -60 deg) must see the right-turning waypoints
    assert front_right_proj.visible_count > 0, "Right-turning waypoints must be visible in front_right camera"

    # Front-left camera (yaw = +60 deg) must have zero visibility of this sharp right turn
    assert front_left_proj.visible_count == 0, "Right turn must not be visible in front-left camera"


def test_near_plane_clipping_behind_camera():
    """Verify points behind camera or inside near plane are safely clipped without NaNs."""
    engine = camera_projection.CameraProjectionEngine(near_clip_m=1.0)

    # Point directly behind camera mount
    behind_pts = np.array([
        [-5.0, 0.0],
        [0.0, 0.0],
        [0.5, 0.0], # inside near clip
    ])

    proj = engine.project_waypoints_to_camera("front", behind_pts)
    assert not np.any(proj.is_visible)
    assert not np.any(np.isnan(proj.points_2d))
    assert not np.any(np.isinf(proj.points_2d))


def test_spatial_control_mask_generation():
    """Verify spatial control mask rasterization conforms to tensor contract."""
    engine = camera_projection.CameraProjectionEngine()

    straight_waypoints = np.column_stack([
        np.linspace(2.0, 30.0, 25),
        np.zeros(25),
    ])

    mask = engine.render_spatial_control_mask("front", straight_waypoints, ribbon_width_px=16)

    assert mask.shape == (spec.HEIGHT, spec.WIDTH)
    assert mask.dtype == np.float32
    assert np.all(mask >= 0.0) and np.all(mask <= 1.0)

    # Foreground pixels exist on road surface
    fg_count = np.sum(mask > 0.1)
    assert fg_count > 100, f"Expected rasterized ribbon pixels, got {fg_count}"

    # Upper sky should have zero mask
    assert np.all(mask[:int(spec.HEIGHT * 0.3), :] == 0.0), "Sky region must have zero mask intensity"
