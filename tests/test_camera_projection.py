"""Unit and regression tests for 3D-to-2D multi-camera projection and spatial conditioning."""

from __future__ import annotations

import numpy as np

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


def test_option_a_static_route_mask_shapes_and_values():
    """Verify Option A (Static Route Mask) shape contract [1, 568, 876] and multi-cam dictionary."""
    engine = camera_projection.CameraProjectionEngine()

    straight_waypoints = np.column_stack([
        np.linspace(2.0, 40.0, 25),
        np.zeros(25),
    ])

    # 1. Single camera with channel_dim=True -> (1, 568, 876)
    mask_c1 = engine.render_static_route_mask("front", straight_waypoints, channel_dim=True)
    assert mask_c1.shape == (1, spec.HEIGHT, spec.WIDTH)
    assert mask_c1.dtype == np.float32
    assert 0.0 <= mask_c1.min() and mask_c1.max() <= 1.0

    # 2. Single camera with channel_dim=False -> (568, 876)
    mask_c0 = engine.render_static_route_mask("front", straight_waypoints, channel_dim=False)
    assert mask_c0.shape == (spec.HEIGHT, spec.WIDTH)

    # 3. All 6 cameras simultaneously
    multi_cam = engine.render_static_route_masks_all_cameras(straight_waypoints)
    assert len(multi_cam) == 6
    for cam in spec.CAMERAS:
        assert cam in multi_cam
        assert multi_cam[cam].shape == (1, spec.HEIGHT, spec.WIDTH)
        assert multi_cam[cam].dtype == np.float32

    # 4. Batch rendering -> (B, 1, 568, 876)
    batch_trajs = [straight_waypoints, straight_waypoints * 0.8, straight_waypoints * 1.2]
    batch_tensor = engine.batch_static_route_masks(batch_trajs, camera_name="front")
    assert batch_tensor.shape == (3, 1, spec.HEIGHT, spec.WIDTH)
    assert batch_tensor.dtype == np.float32


def test_option_b_time_indexed_spatial_masks_shapes_and_progression():
    """Verify Option B.1 (Initial-Frame) and Option B.2 (Future-Camera-Frame) contracts."""
    engine = camera_projection.CameraProjectionEngine()

    straight_waypoints = np.column_stack([
        np.linspace(5.0, 50.0, 25),
        np.zeros(25),
    ])

    # 1. Option B.1: Initial-Frame Temporal Masks (viewed from stationary t=0 mount)
    init_masks = engine.render_initial_frame_temporal_masks("front", straight_waypoints, num_frames=25, channel_dim=True)
    assert init_masks.shape == (25, 1, spec.HEIGHT, spec.WIDTH)
    assert init_masks.dtype == np.float32
    assert 0.0 <= init_masks.min() and init_masks.max() <= 1.0

    # In initial frame, the beacon at waypoint P(t) moves forward toward horizon (v decreases)
    beacon_v_0 = np.argmax(np.max(init_masks[0, 0], axis=1))
    beacon_v_24 = np.argmax(np.max(init_masks[24, 0], axis=1))
    assert beacon_v_0 > beacon_v_24, f"Initial frame beacon must move toward horizon: v_0={beacon_v_0}, v_24={beacon_v_24}"

    # 2. Option B.2: Future-Camera-Frame Masks (ego-motion aligned)
    future_masks = engine.render_future_camera_spatial_masks("front", straight_waypoints, num_frames=25, channel_dim=True)
    assert future_masks.shape == (25, 1, spec.HEIGHT, spec.WIDTH)
    assert future_masks.dtype == np.float32
    assert 0.0 <= future_masks.min() and future_masks.max() <= 1.0

    # 3. All 6 cameras multi-cam dictionary
    multi_cam_future = engine.render_time_indexed_spatial_masks_all_cameras(straight_waypoints, num_frames=25, mode="future_camera")
    assert len(multi_cam_future) == 6
    for cam in spec.CAMERAS:
        assert cam in multi_cam_future
        assert multi_cam_future[cam].shape == (25, 1, spec.HEIGHT, spec.WIDTH)

    # 4. Batch time-indexed rendering -> (B, 25, 1, 568, 876)
    batch_trajs = [straight_waypoints, straight_waypoints * 0.9]
    batch_time = engine.batch_time_indexed_spatial_masks(batch_trajs, camera_name="front", num_frames=25, mode="future_camera")
    assert batch_time.shape == (2, 25, 1, spec.HEIGHT, spec.WIDTH)
    assert batch_time.dtype == np.float32


def test_3d_near_plane_clipping_cases():
    """Verify robust 3D line segment clipping against near plane z >= near_clip."""
    # Case 1: Both endpoints behind near plane (near_clip = 0.5m)
    p1 = np.array([0.0, 0.0, -5.0])
    p2 = np.array([0.0, 0.0, -0.5])
    assert camera_projection.clip_segment_3d_near_plane(p1, p2, near_clip_m=0.5) is None

    # Case 2: Both endpoints in front of near plane
    p1 = np.array([1.0, 2.0, 3.0])
    p2 = np.array([4.0, 5.0, 6.0])
    res = camera_projection.clip_segment_3d_near_plane(p1, p2, near_clip_m=0.5)
    assert res is not None
    assert np.allclose(res[0], p1) and np.allclose(res[1], p2)

    # Case 3: One endpoint behind, one in front
    p1 = np.array([0.0, 0.0, -2.0])
    p2 = np.array([0.0, 0.0, 10.0])
    res = camera_projection.clip_segment_3d_near_plane(p1, p2, near_clip_m=0.5)
    assert res is not None
    p1_clipped, p2_clipped = res
    assert np.isclose(p1_clipped[2], 0.5), f"Expected z=0.5, got {p1_clipped[2]}"
    assert np.allclose(p2_clipped, p2)

    # Case 4: Endpoints exactly on near plane
    p1 = np.array([-1.0, 0.0, 0.5])
    p2 = np.array([1.0, 0.0, 0.5])
    res = camera_projection.clip_segment_3d_near_plane(p1, p2, near_clip_m=0.5)
    assert res is not None
    assert np.isclose(res[0][2], 0.5) and np.isclose(res[1][2], 0.5)


def test_2d_canvas_clipping_boundaries():
    """Verify Cohen-Sutherland 2D clipping against canvas [0, W-1] x [0, H-1]."""
    w, h = spec.WIDTH, spec.HEIGHT  # 876, 568

    # 1. Segment crossing left boundary (u0 < 0)
    res = camera_projection.clip_segment_2d_canvas(-100.0, 300.0, 400.0, 300.0, width=w, height=h)
    assert res is not None
    (u0_c, v0_c), (u1_c, v1_c) = res
    assert np.isclose(u0_c, 0.0) and np.isclose(v0_c, 300.0)
    assert np.isclose(u1_c, 400.0) and np.isclose(v1_c, 300.0)

    # 2. Segment crossing right boundary (u1 > W-1)
    res = camera_projection.clip_segment_2d_canvas(400.0, 300.0, 1200.0, 300.0, width=w, height=h)
    assert res is not None
    (u0_c, v0_c), (u1_c, v1_c) = res
    assert np.isclose(u0_c, 400.0)
    assert np.isclose(u1_c, float(w - 1))

    # 3. Segment crossing top boundary (v0 < 0)
    res = camera_projection.clip_segment_2d_canvas(400.0, -50.0, 400.0, 200.0, width=w, height=h)
    assert res is not None
    (u0_c, v0_c), (u1_c, v1_c) = res
    assert np.isclose(v0_c, 0.0)

    # 4. Segment crossing bottom boundary (v1 > H-1)
    res = camera_projection.clip_segment_2d_canvas(400.0, 300.0, 400.0, 800.0, width=w, height=h)
    assert res is not None
    (u0_c, v0_c), (u1_c, v1_c) = res
    assert np.isclose(v1_c, float(h - 1))

    # 5. Segment entirely outside canvas
    res = camera_projection.clip_segment_2d_canvas(-200.0, -100.0, -50.0, -20.0, width=w, height=h)
    assert res is None


def test_rasterizer_zero_spurious_streaks():
    """Verify rasterizer does not produce streak artifacts when segment crosses behind vehicle."""
    engine = camera_projection.CameraProjectionEngine()

    # Trajectory with one point behind vehicle bumper (-10m) and one far ahead (30m)
    straddling_waypoints = np.array([
        [-10.0, 0.0],
        [30.0, 0.0],
    ])

    mask = engine.render_spatial_control_mask("front", straddling_waypoints)

    assert mask.shape == (spec.HEIGHT, spec.WIDTH)
    # Sky region must remain strictly zero
    sky_region = mask[:int(spec.HEIGHT * 0.35), :]
    assert np.all(sky_region == 0.0), "Sky region must have zero mask intensity even for straddling segments"
    # Road region should have valid rasterized ribbon
    road_region = mask[int(spec.HEIGHT * 0.5):, :]
    assert np.sum(road_region > 0.1) > 0, "Road region must contain projected ribbon"


def test_provisional_rig_attributes():
    """Verify camera rig exposes provisional metadata and supports 6-DoF pitch/roll."""
    rig = camera_projection.CameraRig(
        pitch_deg={"front": 2.0},
        roll_deg={"front": -1.0},
        is_provisional=True,
    )
    assert rig.is_provisional is True
    cam_front = rig["front"]
    assert cam_front.is_provisional is True
    assert cam_front.pitch_deg == 2.0
    assert cam_front.roll_deg == -1.0
    # Rotation matrix must be strictly orthogonal with det = +1
    R = cam_front.rotation_matrix_body_to_cam
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-5)


def test_future_camera_extrapolate_horizon():
    """Verify extrapolate_horizon prevents end-of-horizon corridor vanishing at frame 24."""
    engine = camera_projection.CameraProjectionEngine()

    straight_waypoints = np.column_stack([
        np.linspace(2.0, 50.0, 25),
        np.zeros(25),
    ])

    # Unpadded B.2: Frame 24 has n_up=1 so 0 segments -> empty mask
    m_unpadded = engine.render_future_camera_spatial_masks("front", straight_waypoints, extrapolate_horizon=False)
    assert np.sum(m_unpadded[24] > 0.05) == 0, "Unpadded B.2 frame 24 must be empty due to n_up < 2"

    # Extrapolated B.2: Extrapolates 5 forward steps -> frame 24 has valid corridor
    m_extrap = engine.render_future_camera_spatial_masks("front", straight_waypoints, extrapolate_horizon=True, extrapolate_steps=5)
    assert np.sum(m_extrap[24] > 0.05) > 500, "Extrapolated B.2 frame 24 must contain active road corridor"



