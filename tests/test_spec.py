from kitscenes import spec


def test_video_contract_matches_the_published_rules():
    assert spec.FRAME_COUNT == 25
    assert spec.FPS == 5.0
    assert (spec.WIDTH, spec.HEIGHT) == (876, 568)


def test_trajectory_timing_covers_0_2_to_5_seconds():
    assert spec.WAYPOINT_COUNT == 25
    assert spec.WAYPOINT_STEP_S == 0.2
    assert spec.FIRST_WAYPOINT_S == 0.2
    assert spec.LAST_WAYPOINT_S == 5.0


def test_scenario_identifiers_are_seven_digit_and_zero_padded():
    assert len(spec.SCENARIO_IDS) == 400
    assert spec.SCENARIO_IDS[0] == "0000000"
    assert spec.SCENARIO_IDS[-1] == "0000399"


def test_upload_limits():
    assert spec.MAX_ARCHIVE_BYTES == 3 * 1024**3
    assert spec.MAX_VIDEO_BYTES == 50 * 1024**2


def test_pq_weights_leave_a_front_only_submission_at_half():
    assert spec.PQ_FRONT_WEIGHT + spec.PQ_SIDE_WEIGHT * 5 == 1.0
