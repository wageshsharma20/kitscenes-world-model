"""Unit tests for the 6-camera retrieval-assisted world model engine."""

import json
import math

import numpy as np
import pytest

from kitscenes import media, spec, video_engine
from kitscenes.retrieval_engine import (
    AlignedVideoSynthesizer,
    DonorIndex,
    DonorScenario,
    InstructionTextExtractor,
    KinematicsFeatureExtractor,
    MultimodalRanker,
    RetrievalWorldModel,
    ScenarioTypeMatcher,
    SingleDonorViolationError,
    TargetScenario,
    VisualFeatureExtractor,
    estimate_optical_flow,
    read_image_to_numpy,
    smooth_seam_transition,
)


def _make_straight_trajectory(speed_mps: float = 10.0) -> list[list[float]]:
    dt = spec.WAYPOINT_STEP_S
    return [
        [round(speed_mps * (k + 1) * dt, 3), 0.0]
        for k in range(spec.WAYPOINT_COUNT)
    ]


def _make_turning_trajectory(
    speed_mps: float = 8.0, yaw_rate_rps: float = 0.20
) -> list[list[float]]:
    dt = spec.WAYPOINT_STEP_S
    waypoints = []
    x, y, heading = 0.0, 0.0, 0.0
    for _ in range(spec.WAYPOINT_COUNT):
        heading += yaw_rate_rps * dt
        x += speed_mps * dt * math.cos(heading)
        y += speed_mps * dt * math.sin(heading)
        waypoints.append([round(x, 3), round(y, 3)])
    return waypoints


# ---------------------------------------------------------------------------
# Donor Index Tests
# ---------------------------------------------------------------------------

def test_donor_index_storage_and_lookup():
    index = DonorIndex()
    traj = _make_straight_trajectory(10.0)
    donor = DonorScenario(
        scenario_id="donor_001",
        future_trajectory=traj,
        driving_instruction="drive straight on",
        scenario_type="highway",
    )
    index.add_donor(donor)

    assert len(index) == 1
    assert index.get_donor("donor_001").scenario_id == "donor_001"
    assert donor.kinematic_features is not None
    assert donor.kinematic_features["mean_speed"] == pytest.approx(10.0, rel=1e-2)
    assert donor.text_features is not None


def test_donor_index_default_bank():
    bank = DonorIndex.create_default_bank()
    assert len(bank) >= 8

    # Verify canonical maneuvers are present
    donor_ids = [d.scenario_id for d in bank]
    assert "donor_highway_straight" in donor_ids
    assert "donor_left_turn_intersection" in donor_ids
    assert "donor_right_turn_intersection" in donor_ids
    assert "donor_overtake_truck" in donor_ids
    assert "donor_nighttime_straight" in donor_ids


# ---------------------------------------------------------------------------
# Feature Extraction Tests
# ---------------------------------------------------------------------------

def test_visual_feature_extractor_similarity():
    extractor = VisualFeatureExtractor()

    # Identical images
    img1 = np.full((spec.HEIGHT, spec.WIDTH, 3), 120, dtype=np.uint8)
    feat1 = extractor.extract(img1)
    assert feat1.shape == (156,)
    assert extractor.similarity(feat1, feat1) == pytest.approx(1.0, abs=1e-4)

    # Similar daytime images
    img2 = np.full((spec.HEIGHT, spec.WIDTH, 3), 130, dtype=np.uint8)
    feat2 = extractor.extract(img2)
    assert extractor.similarity(feat1, feat2) > 0.95

    # Drastically different dark night image
    img_night = np.full((spec.HEIGHT, spec.WIDTH, 3), 5, dtype=np.uint8)
    feat_night = extractor.extract(img_night)
    assert extractor.similarity(feat1, feat_night) < 0.60


def test_kinematics_feature_extractor():
    extractor = KinematicsFeatureExtractor()

    straight_10 = extractor.extract(_make_straight_trajectory(10.0))
    straight_10_dup = extractor.extract(_make_straight_trajectory(10.0))
    straight_20 = extractor.extract(_make_straight_trajectory(20.0))
    left_turn = extractor.extract(_make_turning_trajectory(8.0, 0.25))
    right_turn = extractor.extract(_make_turning_trajectory(8.0, -0.25))

    # Identical trajectory has similarity 1.0
    assert extractor.similarity(straight_10, straight_10_dup) == pytest.approx(1.0, abs=1e-4)

    # Yaw changes
    assert straight_10["yaw_change"] == pytest.approx(0.0, abs=1e-3)
    assert left_turn["yaw_change"] > 0.8
    assert right_turn["yaw_change"] < -0.8

    # Left turn vs right turn has lower kinematic similarity than identical
    sim_turns = extractor.similarity(left_turn, right_turn)
    sim_left_straight = extractor.similarity(left_turn, straight_10)
    assert sim_turns < 0.65
    assert sim_left_straight < 0.65

    # Speed difference penalizes similarity
    sim_speeds = extractor.similarity(straight_10, straight_20)
    assert sim_speeds < 0.70


def test_instruction_text_extractor():
    extractor = InstructionTextExtractor()

    # Identical
    assert extractor.similarity("drive straight on", "drive straight on") == pytest.approx(1.0)

    # Substring / related intent
    sim_rel = extractor.similarity(
        "turn left at the intersection", "turn left"
    )
    assert sim_rel > 0.70

    # Directional contrast penalty: left vs right
    sim_contrast = extractor.similarity(
        "turn left at the intersection", "turn right at the intersection"
    )
    assert sim_contrast < 0.35

    # Completely different actions
    sim_diff = extractor.similarity("drive straight on", "slow down and stop at traffic light")
    assert sim_diff < 0.40


def test_scenario_type_matcher():
    matcher = ScenarioTypeMatcher()

    assert matcher.match("highway", "highway") == 1.0
    assert matcher.match("0 specifically selected", "highway") == 0.70
    assert matcher.match("5 nighttime", "nighttime") == 1.0

    # Night vs day lighting conflict
    assert matcher.match("5 nighttime", "clear daytime") == 0.10


# ---------------------------------------------------------------------------
# Multimodal Ranker & Weights Tests
# ---------------------------------------------------------------------------

def test_multimodal_ranker_weights_and_selection():
    # Setup 3 donors
    donor_straight = DonorScenario(
        scenario_id="donor_straight",
        future_trajectory=_make_straight_trajectory(12.0),
        driving_instruction="drive straight on",
        scenario_type="highway",
    )
    donor_left = DonorScenario(
        scenario_id="donor_left",
        future_trajectory=_make_turning_trajectory(8.0, 0.22),
        driving_instruction="turn left at the intersection",
        scenario_type="intersection",
    )
    donor_night = DonorScenario(
        scenario_id="donor_night",
        future_trajectory=_make_straight_trajectory(12.0),
        driving_instruction="drive straight on at night",
        scenario_type="5 nighttime",
    )

    candidates = [donor_straight, donor_left, donor_night]

    # Target 1: turning left
    target_left = TargetScenario(
        scenario_id="target_001",
        future_trajectory=_make_turning_trajectory(8.0, 0.20),
        driving_instruction="turn left",
        scenario_type="intersection",
    )

    ranker = MultimodalRanker()
    ranked = ranker.rank(target_left, candidates, top_k=3)

    assert len(ranked) == 3
    assert ranked[0].donor.scenario_id == "donor_left"
    assert ranked[0].composite_score > ranked[1].composite_score
    assert "sim_kinematics" in ranked[0].similarity_breakdown
    assert "sim_instruction" in ranked[0].similarity_breakdown

    # Target 2: night cruise
    target_night = TargetScenario(
        scenario_id="target_002",
        future_trajectory=_make_straight_trajectory(12.0),
        driving_instruction="drive straight on at night",
        scenario_type="5 nighttime",
    )
    best_night = ranker.select_best(target_night, candidates)
    assert best_night.donor.scenario_id == "donor_night"


def test_multimodal_ranker_weight_sensitivity():
    donor_a = DonorScenario(
        scenario_id="donor_a",
        future_trajectory=_make_straight_trajectory(10.0),
        driving_instruction="turn left",  # mismatched instruction
        scenario_type="urban",
    )
    donor_b = DonorScenario(
        scenario_id="donor_b",
        future_trajectory=_make_straight_trajectory(25.0),  # mismatched speed
        driving_instruction="drive straight on",             # matching instruction
        scenario_type="urban",
    )

    target = TargetScenario(
        scenario_id="target",
        future_trajectory=_make_straight_trajectory(10.0),
        driving_instruction="drive straight on",
        scenario_type="urban",
    )

    # Heavily weight kinematics -> donor_a should win
    ranker_kin = MultimodalRanker(w_visual=0.0, w_kinematics=1.0, w_instruction=0.0, w_type=0.0)
    assert ranker_kin.select_best(target, [donor_a, donor_b]).donor.scenario_id == "donor_a"

    # Heavily weight instruction text -> donor_b should win
    ranker_text = MultimodalRanker(w_visual=0.0, w_kinematics=0.0, w_instruction=1.0, w_type=0.0)
    assert ranker_text.select_best(target, [donor_a, donor_b]).donor.scenario_id == "donor_b"


# ---------------------------------------------------------------------------
# Strict Single-Donor Policy Tests
# ---------------------------------------------------------------------------

def test_strict_single_donor_policy_enforcement():
    synthesizer = AlignedVideoSynthesizer()
    traj = _make_straight_trajectory(10.0)

    target = TargetScenario(scenario_id="0000000", future_trajectory=traj)
    donor1 = DonorScenario(scenario_id="donor_111", future_trajectory=traj)
    donor2 = DonorScenario(scenario_id="donor_222", future_trajectory=traj)

    # Consistent donor across all 6 cameras succeeds
    consistent_cams = {cam: donor1 for cam in spec.CAMERAS}
    vids = synthesizer.synthesize(target, donor1, camera_donors=consistent_cams)
    assert len(vids) == 6
    assert set(vids.keys()) == set(spec.CAMERAS)

    # Inconsistent donors across cameras MUST raise SingleDonorViolationError
    inconsistent_cams = dict(consistent_cams)
    inconsistent_cams["rear"] = donor2  # Violation!

    with pytest.raises(SingleDonorViolationError, match="Single-Donor Policy violation"):
        synthesizer.synthesize(target, donor1, camera_donors=inconsistent_cams)


# ---------------------------------------------------------------------------
# Seam Smoothing & Motion Blending Tests
# ---------------------------------------------------------------------------

def test_motion_alignment_and_seam_smoothing():
    h, w = spec.HEIGHT, spec.WIDTH
    # Create distinct target t=0 frame and donor future footage
    target_t0 = np.full((h, w, 3), 70, dtype=np.uint8)
    donor_future = np.full((spec.FRAME_COUNT, h, w, 3), 160, dtype=np.uint8)

    smoothed = smooth_seam_transition(
        target_t0=target_t0,
        donor_frames=donor_future,
        blend_frames=4,
    )

    assert smoothed.shape == (spec.FRAME_COUNT, h, w, 3)
    assert smoothed.dtype == np.uint8

    # Frame 0 has substantial influence from target t=0 (mean closer to 70 than 160)
    mean_f0 = float(smoothed[0].mean())
    assert 70.0 < mean_f0 < 130.0

    # Frame 2 is an intermediate cross-dissolve blend
    mean_f2 = float(smoothed[2].mean())
    assert mean_f0 < mean_f2 < 160.0

    # Frame 4 (k >= blend_frames) is 100% pure donor future
    assert smoothed[4].mean() == pytest.approx(160.0)
    assert smoothed[24].mean() == pytest.approx(160.0)


def test_optical_flow_estimator():
    h, w = spec.HEIGHT, spec.WIDTH
    target = np.zeros((h, w, 3), dtype=np.uint8)
    target[100:200, 100:200] = 200

    donor = np.zeros((h, w, 3), dtype=np.uint8)
    donor[100:200, 110:210] = 200

    u, v = estimate_optical_flow(target, donor)
    assert u.shape == (h, w)
    assert v.shape == (h, w)
    assert u.dtype == np.float32
    assert v.dtype == np.float32


# ---------------------------------------------------------------------------
# End-to-End RetrievalWorldModel & Pipeline Integration Tests
# ---------------------------------------------------------------------------

def test_retrieval_world_model_end_to_end():
    bank = DonorIndex.create_default_bank()
    model = RetrievalWorldModel(donor_index=bank)

    # Request left turn
    target_traj = _make_turning_trajectory(8.0, 0.20)
    t0_frames = {
        cam: np.full((spec.HEIGHT, spec.WIDTH, 3), 100, dtype=np.uint8)
        for cam in spec.CAMERAS
    }

    videos = model.generate_scenario(
        scenario_id="0000042",
        trajectory=target_traj,
        driving_instruction="turn left at the intersection",
        scenario_type="intersection",
        conditioning_images=t0_frames,
    )

    assert set(videos.keys()) == set(spec.CAMERAS)
    assert model.last_retrieval_result is not None
    assert model.last_retrieval_result.donor.scenario_id == "donor_left_turn_intersection"

    for _cam, frames in videos.items():
        assert isinstance(frames, np.ndarray)
        assert frames.shape == (spec.FRAME_COUNT, spec.HEIGHT, spec.WIDTH, 3)
        assert frames.dtype == np.uint8
        assert frames.mean() > 10.0


def test_world_model_pipeline_accepts_retrieval_backbone():
    pipe = video_engine.WorldModelPipeline(backbone="retrieval")
    assert pipe.backbone_name == "retrieval"
    assert isinstance(pipe.generator, RetrievalWorldModel)

    # Generate scenario through pipeline
    waypoints = _make_straight_trajectory(12.0)
    vids = pipe.generator.generate_scenario("0000000", waypoints)
    assert len(vids) == 6
    for cam_frames in vids.values():
        assert cam_frames.shape == (spec.FRAME_COUNT, spec.HEIGHT, spec.WIDTH, 3)
        assert cam_frames.dtype == np.uint8


def test_batch_generate_and_package_with_retrieval(tmp_path):
    scenarios = ["0000000"]
    traj_path = tmp_path / "trajectories.jsonl"
    with open(traj_path, "w", encoding="utf-8") as f:
        pts = _make_straight_trajectory(12.0)
        f.write(json.dumps({"scenario_id": "0000000", "future_trajectory": pts}) + "\n")

    # Write mock observed t0 frames
    t0_dir = tmp_path / "observed_t0"
    sc_dir = t0_dir / "0000000"
    sc_dir.mkdir(parents=True)
    for cam in spec.CAMERAS:
        img_arr = np.full((spec.HEIGHT, spec.WIDTH, 3), 90, dtype=np.uint8)
        # Save as raw ppm or write bytes
        (sc_dir / f"{cam}.png").write_bytes(read_image_to_numpy(img_arr).tobytes()[:100])
        # Use numpy array directly for conditioning_images
    cond_images = {
        "0000000": {
            cam: np.full((spec.HEIGHT, spec.WIDTH, 3), 95, dtype=np.uint8)
            for cam in spec.CAMERAS
        }
    }

    renders_dir = tmp_path / "renders"
    zip_path = tmp_path / "test_retrieval.zip"

    report = video_engine.batch_generate_and_package(
        trajectories_file=traj_path,
        output_dir=renders_dir,
        zip_path=zip_path,
        scenario_ids=scenarios,
        allow_partial=True,
        backbone="retrieval",
        conditioning_images=cond_images,
    )

    assert report.ok
    assert report.videos == 6
    assert report.scenarios_complete == 1
    assert zip_path.exists()

    # Validate generated mp4 with kitscenes.media
    sample_mp4 = renders_dir / "0000000" / "front.mp4"
    assert sample_mp4.exists()
    info = media.probe_video(sample_mp4)
    errors, warnings = media.check_video(info)
    assert info.codec == "h264"
    assert info.frame_count == spec.FRAME_COUNT
    assert info.fps == pytest.approx(spec.FPS)
    assert info.width == spec.WIDTH
    assert info.height == spec.HEIGHT
    assert errors == []
