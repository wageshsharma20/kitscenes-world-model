"""Tests for the t=0 frame extractor.

The real file lives in a gated 34 GB parquet, so these tests build a synthetic one with
the same shape: one list<struct<bytes>> column per camera, 21 frames, last one at t=0.
"""

import importlib.util
from pathlib import Path

import pytest

pyarrow = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_t0_frames.py"


@pytest.fixture(scope="module")
def extractor():
    spec = importlib.util.spec_from_file_location("extract_t0_frames", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frames(prefix: str, count: int = 21):
    """21 frames ending in the t=0 payload, the way the dataset stores them."""
    return [{"bytes": b"earlier"}] * (count - 1) + [{"bytes": prefix}]


def write_parquet(path, scenario_ids, cameras):
    rows = []
    for scenario_id in scenario_ids:
        row = {"scenario_id": scenario_id}
        for camera in cameras:
            row[camera] = frames(f"{scenario_id}-{camera}-t0".encode())
        rows.append(row)
    pq.write_table(pyarrow.Table.from_pylist(rows), path)
    return path


@pytest.fixture
def parquet_file(tmp_path, extractor):
    cameras = list(extractor.CAMERA_FRAME_COLUMNS.values())
    return write_parquet(
        tmp_path / "split.parquet", ["0000000", "0000001"], cameras
    )


def test_saves_only_the_last_frame_of_each_camera(extractor, parquet_file, tmp_path):
    out = tmp_path / "t0"

    result = extractor.process_parquet_file(
        parquet_file, out, resize=False, force=True
    )

    assert len(result) == 2
    assert len(result["0000000"]) == 6
    for camera, column in extractor.CAMERA_FRAME_COLUMNS.items():
        saved = out / "0000000" / f"{camera}.jpg"
        assert saved.read_bytes() == f"0000000-{column}-t0".encode()


def test_second_run_skips_completed_scenarios(extractor, parquet_file, tmp_path):
    out = tmp_path / "t0"
    extractor.process_parquet_file(parquet_file, out, resize=False, force=True)

    again = extractor.process_parquet_file(parquet_file, out, resize=False)

    assert again == {}


def test_force_reextracts(extractor, parquet_file, tmp_path):
    out = tmp_path / "t0"
    extractor.process_parquet_file(parquet_file, out, resize=False, force=True)

    forced = extractor.process_parquet_file(parquet_file, out, resize=False, force=True)

    assert len(forced) == 2


def test_partial_scenario_is_treated_as_incomplete(extractor, parquet_file, tmp_path):
    out = tmp_path / "t0"
    extractor.process_parquet_file(parquet_file, out, resize=False, force=True)
    (out / "0000000" / "rear.jpg").unlink()

    rerun = extractor.process_parquet_file(parquet_file, out, resize=False)

    assert list(rerun) == ["0000000"]
    assert (out / "0000000" / "rear.jpg").exists()


def test_target_and_limit_filters(extractor, parquet_file, tmp_path):
    out = tmp_path / "t0"

    only_one = extractor.process_parquet_file(
        parquet_file, out, target_scenarios={"0000001"}, resize=False, force=True
    )
    assert list(only_one) == ["0000001"]

    limited = extractor.process_parquet_file(
        parquet_file, out / "second", limit=1, resize=False, force=True
    )
    assert len(limited) == 1


def test_rejects_parquet_without_camera_columns(extractor, tmp_path):
    path = tmp_path / "empty.parquet"
    pq.write_table(pyarrow.Table.from_pylist([{"scenario_id": "0000000"}]), path)

    with pytest.raises(RuntimeError, match="camera frame columns"):
        extractor.process_parquet_file(path, tmp_path / "out", force=True)
