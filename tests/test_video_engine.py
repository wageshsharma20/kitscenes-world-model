import json
import math

import numpy as np
import pytest

from kitscenes import archive, media, spec, video_engine


def test_camera_rig_geometry():
    rig = video_engine.CameraRig()

    assert len(rig.cameras) == 6
    assert set(rig.cameras.keys()) == set(spec.CAMERAS)

    # Verify camera angles
    assert rig["front"].yaw_deg == pytest.approx(0.0)
    assert rig["front_left"].yaw_deg == pytest.approx(60.0)
    assert rig["front_right"].yaw_deg == pytest.approx(-60.0)
    assert rig["rear"].yaw_deg == pytest.approx(180.0)
    assert rig["rear_left"].yaw_deg == pytest.approx(120.0)
    assert rig["rear_right"].yaw_deg == pytest.approx(-120.0)

    # Verify intrinsics
    for cam in rig.cameras.values():
        assert cam.width == spec.WIDTH
        assert cam.height == spec.HEIGHT
        assert cam.cx == spec.WIDTH / 2.0
        assert cam.cy == spec.HEIGHT / 2.0
        assert cam.fx == pytest.approx(cam.fy)
        assert cam.fx > 500.0

        # Intrinsic matrix shape
        K = cam.intrinsic_matrix
        assert K.shape == (3, 3)
        assert K[0, 2] == cam.cx
        assert K[1, 2] == cam.cy
        assert K[2, 2] == 1.0


def test_straight_line_motion_analysis():
    rig = video_engine.CameraRig()
    analyzer = video_engine.TrajectoryMotionAnalyzer(rig)

    # 10 m/s straight line forward along +X: dt = 0.2s, displacement per step = 2.0m
    waypoints = [[2.0 * (k + 1), 0.0] for k in range(spec.WAYPOINT_COUNT)]
    profile = analyzer.analyze(waypoints)

    assert profile.speeds == pytest.approx(10.0)
    assert profile.yaw_rates == pytest.approx(0.0)
    assert profile.headings == pytest.approx(0.0)

    # Test camera-specific velocities
    cam_motions = profile.camera_motions
    assert set(cam_motions.keys()) == set(spec.CAMERAS)

    # Front: forward looming at 10 m/s, zero lateral
    assert cam_motions["front"].v_forward == pytest.approx(10.0)
    assert cam_motions["front"].v_lateral == pytest.approx(0.0)

    # Rear: receding at -10 m/s, zero lateral
    assert cam_motions["rear"].v_forward == pytest.approx(-10.0)
    assert cam_motions["rear"].v_lateral == pytest.approx(0.0)

    # Front-left (60 deg): v_forward = 10*cos(60°) = 5.0, v_lateral = -10*sin(60°) = -8.66
    assert cam_motions["front_left"].v_forward == pytest.approx(5.0)
    assert cam_motions["front_left"].v_lateral == pytest.approx(-10.0 * math.sin(math.pi / 3))

    # Front-right (-60 deg): v_forward = 5.0, v_lateral = +8.66
    assert cam_motions["front_right"].v_forward == pytest.approx(5.0)
    assert cam_motions["front_right"].v_lateral == pytest.approx(10.0 * math.sin(math.pi / 3))

    # Rear-left (120 deg): v_forward = -5.0, v_lateral = -8.66
    assert cam_motions["rear_left"].v_forward == pytest.approx(-5.0)
    assert cam_motions["rear_left"].v_lateral == pytest.approx(-10.0 * math.sin(math.pi / 3))

    # Rear-right (-120 deg): v_forward = -5.0, v_lateral = +8.66
    assert cam_motions["rear_right"].v_forward == pytest.approx(-5.0)
    assert cam_motions["rear_right"].v_lateral == pytest.approx(10.0 * math.sin(math.pi / 3))

    # Side cameras all reflect vehicle speed v via lateral motion magnitude ~ 8.66 m/s
    for side_cam in spec.SIDE_CAMERAS:
        assert np.abs(cam_motions[side_cam].v_lateral) == pytest.approx(
            10.0 * math.sin(math.radians(60.0))
        )


def test_turning_motion_yaw_agreement():
    analyzer = video_engine.TrajectoryMotionAnalyzer()

    # Left turn on a circle of radius R = 50m at speed v = 10m/s
    # angular speed w = v / R = 0.2 rad/s
    R = 50.0
    v = 10.0
    w = v / R
    dt = spec.WAYPOINT_STEP_S

    waypoints = []
    heading = 0.0
    x, y = 0.0, 0.0
    for _ in range(spec.WAYPOINT_COUNT):
        x += v * dt * math.cos(heading)
        y += v * dt * math.sin(heading)
        waypoints.append([x, y])
        heading += w * dt

    profile = analyzer.analyze(waypoints)

    # Front and rear must agree on yaw rate at every frame
    front_yaw = profile.camera_motions["front"].omega_yaw
    rear_yaw = profile.camera_motions["rear"].omega_yaw
    np.testing.assert_allclose(front_yaw, rear_yaw, atol=1e-5)
    np.testing.assert_allclose(front_yaw[1:], w, atol=0.05)

    # Steering class check
    assert profile.emc_steering_classes["seg1_0_3s"] == "left"
    assert profile.emc_steering_classes["seg2_3_5s"] == "left"


def test_analyzer_rejects_invalid_waypoints():
    analyzer = video_engine.TrajectoryMotionAnalyzer()

    with pytest.raises(ValueError, match="expected"):
        analyzer.analyze([[0.0, 0.0]] * 10)  # only 10 points instead of 25


def test_encode_video_compliance(tmp_path):
    encoder = video_engine.VideoEncoder()
    # Generate 25 test frames
    frames = np.zeros((spec.FRAME_COUNT, spec.HEIGHT, spec.WIDTH, 3), dtype=np.uint8)
    frames[:, :, :, 0] = 50
    frames[:, :, :, 1] = 100
    frames[:, :, :, 2] = 150

    out_file = tmp_path / "test_encoded.mp4"
    result_path = encoder.encode(frames, out_file, validate=True)
    assert result_path == out_file
    assert out_file.exists()

    # Probe and check with kitscenes.media
    info = media.probe_video(out_file)
    errors, warnings = media.check_video(info)

    assert info.codec == "h264"
    assert info.width == spec.WIDTH
    assert info.height == spec.HEIGHT
    assert info.frame_count == spec.FRAME_COUNT
    assert info.fps == pytest.approx(spec.FPS)
    assert errors == []
    assert warnings == []
    assert out_file.stat().st_size < spec.MAX_VIDEO_BYTES


def test_geometric_world_model_scenario_generation():
    model = video_engine.GeometricWorldModel()
    waypoints = [[2.0 * (k + 1), 0.0] for k in range(spec.WAYPOINT_COUNT)]

    videos = model.generate_scenario("0000000", waypoints)

    assert set(videos.keys()) == set(spec.CAMERAS)
    for cam_frames in videos.values():
        assert isinstance(cam_frames, np.ndarray)
        assert cam_frames.shape == (spec.FRAME_COUNT, spec.HEIGHT, spec.WIDTH, 3)
        assert cam_frames.dtype == np.uint8
        # Ensure not completely black
        assert cam_frames.mean() > 10.0


def test_batch_generate_and_package_partial(tmp_path):
    # Create sample trajectories for 2 scenarios
    scenarios = ["0000000", "0000001"]
    traj_path = tmp_path / "trajectories.jsonl"
    with open(traj_path, "w", encoding="utf-8") as f:
        for sc in scenarios:
            pts = [[round(1.5 * (k + 1), 3), round(0.1 * k, 3)] for k in range(spec.WAYPOINT_COUNT)]
            f.write(json.dumps({"scenario_id": sc, "future_trajectory": pts}) + "\n")

    renders_dir = tmp_path / "renders"
    zip_path = tmp_path / "test_videos.zip"

    report = video_engine.batch_generate_and_package(
        trajectories_file=traj_path,
        output_dir=renders_dir,
        zip_path=zip_path,
        scenario_ids=scenarios,
        allow_partial=True,
    )

    assert report.ok
    assert report.videos == 12  # 2 scenarios * 6 cameras
    assert report.scenarios_covered == 2
    assert report.scenarios_complete == 2
    assert zip_path.exists()

    # Inspect the generated archive
    inspection = archive.inspect_archive(zip_path, check_media=True)
    assert inspection.videos == 12
    assert inspection.scenarios_complete == 2
    # Errors should be empty because allow_partial=True (inspect_archive checks errors)
    # Note: inspect_archive might report missing front for the remaining 398 scenarios as error,
    # let's verify media errors are zero.
    media_errors = [e for e in inspection.errors if not e.startswith("missing front.mp4")]
    assert media_errors == []


def test_encoder_rejects_invalid_shapes(tmp_path):
    encoder = video_engine.VideoEncoder()
    bad_frames = np.zeros((10, spec.HEIGHT, spec.WIDTH, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="shape"):
        encoder.encode(bad_frames, tmp_path / "bad.mp4")


def test_svd_backbone_refuses_to_run_without_conditioning_frames():
    """A diffusion backbone conditioned on a procedural canvas would be meaningless."""
    generator = video_engine.StableVideoDiffusionGenerator()
    waypoints = [[2.0 * (k + 1), 0.0] for k in range(spec.WAYPOINT_COUNT)]

    with pytest.raises(ValueError, match="observed t=0 frame"):
        generator.generate_scenario("0000000", waypoints)


def test_svd_backbone_names_the_missing_cameras():
    generator = video_engine.StableVideoDiffusionGenerator()
    waypoints = [[2.0 * (k + 1), 0.0] for k in range(spec.WAYPOINT_COUNT)]

    with pytest.raises(ValueError, match="rear"):
        generator.generate_scenario(
            "0000000", waypoints, conditioning_images={"front": object()}
        )


def test_load_conditioning_images(tmp_path):
    sc_dir = tmp_path / "0000042"
    sc_dir.mkdir(parents=True)
    for cam in spec.CAMERAS:
        (sc_dir / f"{cam}.jpg").write_bytes(b"mock_jpeg")

    images = video_engine.load_conditioning_images(tmp_path, "0000042")
    assert len(images) == 6
    for cam in spec.CAMERAS:
        assert cam in images
        assert images[cam].exists()

    # Missing camera view raises FileNotFoundError
    (sc_dir / "rear.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="missing rear frame"):
        video_engine.load_conditioning_images(tmp_path, "0000042")


def test_batch_generate_and_package_with_observed_t0_dir(tmp_path):
    scenarios = ["0000000"]
    traj_path = tmp_path / "trajectories.jsonl"
    with open(traj_path, "w", encoding="utf-8") as f:
        pts = [[round(1.5 * (k + 1), 3), 0.0] for k in range(spec.WAYPOINT_COUNT)]
        f.write(json.dumps({"scenario_id": "0000000", "future_trajectory": pts}) + "\n")

    t0_dir = tmp_path / "observed_t0"
    sc_t0 = t0_dir / "0000000"
    sc_t0.mkdir(parents=True)
    for cam in spec.CAMERAS:
        (sc_t0 / f"{cam}.jpg").write_bytes(b"dummy")

    renders_dir = tmp_path / "renders"
    zip_path = tmp_path / "out.zip"

    report = video_engine.batch_generate_and_package(
        trajectories_file=traj_path,
        output_dir=renders_dir,
        zip_path=zip_path,
        scenario_ids=scenarios,
        allow_partial=True,
        observed_t0_dir=t0_dir,
    )
    assert report.ok
    assert report.videos == 6




def test_missing_conditioning_frames_fail_before_any_rendering(tmp_path):
    """A 400-video GPU run must not discover a missing frame folder at scenario 381."""
    traj_path = tmp_path / "trajectories.jsonl"
    pts = [[round(1.5 * (k + 1), 3), 0.0] for k in range(spec.WAYPOINT_COUNT)]
    with open(traj_path, "w", encoding="utf-8") as f:
        for sc_id in ("0000000", "0000001"):
            f.write(json.dumps({"scenario_id": sc_id, "future_trajectory": pts}) + "\n")

    t0_dir = tmp_path / "observed_t0"
    complete = t0_dir / "0000000"
    complete.mkdir(parents=True)
    for cam in spec.CAMERAS:
        (complete / f"{cam}.jpg").write_bytes(b"dummy")
    (t0_dir / "0000001").mkdir(parents=True)  # present but empty

    with pytest.raises(FileNotFoundError, match="0000001"):
        video_engine.batch_generate_and_package(
            trajectories_file=traj_path,
            output_dir=tmp_path / "renders",
            zip_path=tmp_path / "out.zip",
            scenario_ids=["0000000", "0000001"],
            allow_partial=True,
            observed_t0_dir=t0_dir,
        )

    assert not (tmp_path / "renders").exists()


def test_svd_batch_refuses_to_start_without_conditioning(tmp_path):
    traj_path = tmp_path / "trajectories.jsonl"
    pts = [[round(1.5 * (k + 1), 3), 0.0] for k in range(spec.WAYPOINT_COUNT)]
    traj_path.write_text(
        json.dumps({"scenario_id": "0000000", "future_trajectory": pts}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="observed t=0 frame"):
        video_engine.batch_generate_and_package(
            trajectories_file=traj_path,
            output_dir=tmp_path / "renders",
            zip_path=tmp_path / "out.zip",
            scenario_ids=["0000000"],
            allow_partial=True,
            backbone="svd",
        )
