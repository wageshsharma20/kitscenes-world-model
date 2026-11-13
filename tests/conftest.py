import shutil
import subprocess

import pytest

FFMPEG = shutil.which("ffmpeg")


@pytest.fixture(scope="session")
def make_clip(tmp_path_factory):
    """Write a real H.264 clip and return its path.

    Encoding is the only way to exercise the probe path honestly; a faked mp4 would
    only test the fake.
    """
    if FFMPEG is None:
        pytest.skip("ffmpeg is required for media tests")

    root = tmp_path_factory.mktemp("clips")

    def factory(
        name="clip.mp4",
        *,
        frames=25,
        fps=5,
        size="876x568",
        codec="libx264",
    ):
        path = root / name
        subprocess.run(
            [
                FFMPEG,
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"testsrc=size={size}:rate={fps}",
                "-frames:v",
                str(frames),
                "-c:v",
                codec,
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
        )
        return path

    return factory
