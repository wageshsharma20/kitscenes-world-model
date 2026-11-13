import zipfile

from kitscenes import archive, spec


def make_archive(path, entries):
    """Write a zip from ``{arcname: bytes}``."""
    with zipfile.ZipFile(path, "w") as zf:
        for name, payload in entries.items():
            zf.writestr(name, payload)
    return path


def test_reports_missing_front_videos(tmp_path):
    path = make_archive(
        tmp_path / "videos.zip", {"0000000/front_left.mp4": b"", "0000000/rear.mp4": b""}
    )
    report = archive.inspect_archive(path, check_media=False)

    assert not report.ok
    assert any("missing front.mp4 for 400" in message for message in report.errors)


def test_rejects_unknown_camera_names(tmp_path):
    path = make_archive(tmp_path / "videos.zip", {"0000000/top.mp4": b""})
    report = archive.inspect_archive(path, check_media=False)

    assert any("unknown camera 'top'" in message for message in report.errors)


def test_rejects_duplicate_entries(tmp_path):
    path = tmp_path / "videos.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("0000000/front.mp4", b"")
        zf.writestr("0000000/front.mp4", b"")
    report = archive.inspect_archive(path, check_media=False)

    assert any("duplicate video" in message for message in report.errors)


def test_rejects_non_video_files_in_scenario_folders(tmp_path):
    entries = {"0000000/front.mp4": b"", "0000000/notes.txt": b""}
    path = make_archive(tmp_path / "videos.zip", entries)
    report = archive.inspect_archive(path, check_media=False)

    assert any("only contain <camera>.mp4" in message for message in report.errors)


def test_counts_coverage_and_flags_partial_scenarios(tmp_path):
    entries = {
        "0000000/front.mp4": b"",
        "0000000/front_left.mp4": b"",
        "0000001/front.mp4": b"",
    }
    report = archive.inspect_archive(
        make_archive(tmp_path / "videos.zip", entries), check_media=False
    )

    assert report.videos == 3
    assert report.scenarios_covered == 2
    assert report.scenarios_complete == 0
    assert any("miss at least one camera" in message for message in report.warnings)


def test_ignores_entries_outside_scenario_folders(tmp_path):
    entries = {"0000000/front.mp4": b"", "readme.txt": b"", "__MACOSX/x": b""}
    report = archive.inspect_archive(
        make_archive(tmp_path / "videos.zip", entries), check_media=False
    )

    assert any("ignored by the server" in message for message in report.warnings)
    assert any("missing front.mp4 for 399" in message for message in report.errors)


def test_skipped_media_check_is_reported(tmp_path):
    path = make_archive(tmp_path / "videos.zip", {"0000000/front.mp4": b""})
    report = archive.inspect_archive(path, check_media=False)

    assert any("encoding was not checked" in message for message in report.warnings)


def test_deep_check_catches_a_good_video(tmp_path, make_clip):
    clip = make_clip("clip.mp4")
    path = tmp_path / "videos.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.write(clip, arcname="0000000/front.mp4")

    report = archive.inspect_archive(path, check_media=True)

    assert report.media_checked == 1
    assert not any("0000000/front.mp4" in message for message in report.errors)


def test_deep_check_catches_wrong_encoding(tmp_path, make_clip):
    clip = make_clip("short.mp4", frames=20)
    path = tmp_path / "videos.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.write(clip, arcname="0000000/front.mp4")

    report = archive.inspect_archive(path, check_media=True)

    assert any("20 frames" in message for message in report.errors)


def test_package_refuses_incomplete_render_folders(tmp_path, make_clip):
    clip = make_clip("clip.mp4")
    renders = tmp_path / "renders" / "0000000"
    renders.mkdir(parents=True)
    (renders / "front.mp4").write_bytes(clip.read_bytes())

    report = archive.build_archive(tmp_path / "renders", tmp_path / "videos.zip")

    assert not report.ok
    assert any("missing front.mp4 for 399" in message for message in report.errors)
    assert not (tmp_path / "videos.zip").exists()


def test_package_writes_a_partial_archive_on_request(tmp_path, make_clip):
    clip = make_clip("clip.mp4")
    renders = tmp_path / "renders" / "0000000"
    renders.mkdir(parents=True)
    (renders / "front.mp4").write_bytes(clip.read_bytes())
    for camera in spec.CAMERAS[1:]:
        (renders / f"{camera}.mp4").write_bytes(clip.read_bytes())

    report = archive.build_archive(
        tmp_path / "renders", tmp_path / "videos.zip", allow_partial=True
    )

    assert report.ok, report.errors
    assert report.videos == 6
    assert report.scenarios_complete == 1
    with zipfile.ZipFile(tmp_path / "videos.zip") as zf:
        assert sorted(zf.namelist()) == sorted(
            f"0000000/{camera}.mp4" for camera in spec.CAMERAS
        )
