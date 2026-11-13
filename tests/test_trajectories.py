import json

import pytest

from kitscenes import spec, trajectories


def straight(start=0.0, speed=8.0, count=spec.WAYPOINT_COUNT):
    return [
        [round(start + speed * spec.WAYPOINT_STEP_S * index, 3), 0.0]
        for index in range(1, count + 1)
    ]


def write(tmp_path, records, name="trajectories.jsonl"):
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                record if isinstance(record, str) else json.dumps(record)
            )
            handle.write("\n")
    return path


def records_for_all_scenarios(points=None):
    return [
        {
            "scenario_id": scenario_id,
            "future_trajectory": points if points is not None else straight(),
        }
        for scenario_id in spec.SCENARIO_IDS
    ]


def test_accepts_a_complete_file(tmp_path):
    report = trajectories.inspect_trajectories(write(tmp_path, records_for_all_scenarios()))

    assert report.ok, report.errors
    assert report.count == 400
    assert report.warnings == []


def test_accepts_any_record_order(tmp_path):
    records = records_for_all_scenarios()[::-1]

    report = trajectories.inspect_trajectories(write(tmp_path, records))

    assert report.ok, report.errors


def test_reports_short_files_with_the_missing_ids(tmp_path):
    report = trajectories.inspect_trajectories(
        write(tmp_path, records_for_all_scenarios()[:-1])
    )

    assert not report.ok
    assert any("missing 1 scenario" in message for message in report.errors)
    assert any("0000399" in message for message in report.errors)


def test_rejects_malformed_ids(tmp_path):
    records = records_for_all_scenarios()
    records[0]["scenario_id"] = "0"

    report = trajectories.inspect_trajectories(write(tmp_path, records))

    assert any("is not one of" in message for message in report.errors)


def test_rejects_wrong_waypoint_count(tmp_path):
    records = records_for_all_scenarios(points=straight(count=24))

    report = trajectories.inspect_trajectories(write(tmp_path, records))

    assert any("found 24" in message for message in report.errors)


def test_rejects_non_finite_coordinates(tmp_path):
    points = straight()
    points[3] = [float("nan"), 0.0]
    records = [
        {"scenario_id": spec.SCENARIO_IDS[0], "future_trajectory": points}
    ] + records_for_all_scenarios()[1:]

    report = trajectories.inspect_trajectories(write(tmp_path, records))

    assert any("finite numbers" in message for message in report.errors)


def test_rejects_booleans(tmp_path):
    points = straight()
    points[0] = [True, 0.0]
    records = [
        {"scenario_id": spec.SCENARIO_IDS[0], "future_trajectory": points}
    ] + records_for_all_scenarios()[1:]

    report = trajectories.inspect_trajectories(write(tmp_path, records))

    assert any("finite numbers" in message for message in report.errors)


def test_rejects_duplicate_scenarios(tmp_path):
    records = records_for_all_scenarios()
    records[1]["scenario_id"] = records[0]["scenario_id"]

    report = trajectories.inspect_trajectories(write(tmp_path, records))

    assert any("duplicate scenario_id" in message for message in report.errors)


def test_rejects_blank_lines(tmp_path):
    records = records_for_all_scenarios()
    records.insert(5, "")

    report = trajectories.inspect_trajectories(write(tmp_path, records))

    assert any("blank line" in message for message in report.errors)


def test_rejects_broken_json(tmp_path):
    records = records_for_all_scenarios()
    records[2] = '{"scenario_id": "0000002", '

    report = trajectories.inspect_trajectories(write(tmp_path, records))

    assert any("invalid JSON" in message for message in report.errors)


def test_flags_stationary_trajectories(tmp_path):
    points = [[0.0, 0.0] for _ in range(spec.WAYPOINT_COUNT)]

    report = trajectories.inspect_trajectories(
        write(tmp_path, [
            {"scenario_id": scenario_id, "future_trajectory": points}
            for scenario_id in spec.SCENARIO_IDS
        ])
    )

    assert report.ok
    assert any("look stationary" in message for message in report.warnings)


def test_flags_implausible_speed(tmp_path):
    points = straight(speed=120.0)

    report = trajectories.inspect_trajectories(
        write(tmp_path, [
            {"scenario_id": scenario_id, "future_trajectory": points}
            for scenario_id in spec.SCENARIO_IDS
        ])
    )

    assert report.ok
    assert any("implausible speed" in message for message in report.warnings)


@pytest.mark.parametrize("suffix", [".json", ".txt"])
def test_rejects_wrong_extension(tmp_path, suffix):
    path = tmp_path / f"trajectories{suffix}"
    path.write_text("")
    report = trajectories.inspect_trajectories(path)

    assert any("expected a .jsonl" in message for message in report.errors)


def test_check_reference_trajectory_guards_redacted_sentinels():
    # Sentinels as present in the gated test split
    redacted = [[-100.0, -100.0]]
    with pytest.raises(trajectories.RedactedReferenceError) as exc_info:
        trajectories.check_reference_trajectory(redacted, scenario_id="0000000")
    assert "redacted to [[-100.0, -100.0]]" in str(exc_info.value)
    assert "MMS (Multi-Maneuver Score) is provably uncomputable locally" in str(exc_info.value)

    # When allow_redacted=True, returns False instead of raising
    assert trajectories.check_reference_trajectory(redacted, allow_redacted=True) is False


def test_check_reference_trajectory_accepts_valid():
    valid_points = [[float(i), 0.0] for i in range(spec.WAYPOINT_COUNT)]
    assert trajectories.check_reference_trajectory(valid_points) is True

