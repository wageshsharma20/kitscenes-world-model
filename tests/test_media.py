import pytest

from kitscenes import media


def test_accepts_the_specified_encoding(make_clip):
    info = media.probe_video(make_clip())
    errors, warnings = media.check_video(info)

    assert (info.codec, info.width, info.height) == ("h264", 876, 568)
    assert info.frame_count == 25
    assert info.fps == pytest.approx(5.0)
    assert errors == []
    assert warnings == []


def test_rejects_wrong_frame_count(make_clip):
    info = media.probe_video(make_clip("thirty.mp4", frames=30))
    errors, _ = media.check_video(info)

    assert any("30 frames" in message for message in errors)


def test_rejects_wrong_frame_rate(make_clip):
    info = media.probe_video(make_clip("ten_fps.mp4", fps=10))
    errors, _ = media.check_video(info)

    assert any("fps" in message for message in errors)


def test_rejects_non_h264(make_clip):
    info = media.probe_video(make_clip("mpeg4.mp4", codec="mpeg4"))
    errors, _ = media.check_video(info)

    assert any("H.264" in message for message in errors)


def test_warns_on_aspect_ratio_drift(make_clip):
    info = media.probe_video(make_clip("wide.mp4", size="640x360"))
    errors, warnings = media.check_video(info)

    assert errors == []
    assert any("aspect ratio" in message for message in warnings)


def test_same_aspect_different_size_is_only_a_warning(make_clip):
    info = media.probe_video(make_clip("scaled.mp4", size="438x284"))
    errors, warnings = media.check_video(info)

    assert errors == []
    assert any("438x284" in message for message in warnings)


def test_probe_error_names_the_file(tmp_path):
    broken = tmp_path / "not_a_video.mp4"
    broken.write_bytes(b"not an mp4")

    with pytest.raises(media.ProbeError):
        media.probe_video(broken)
