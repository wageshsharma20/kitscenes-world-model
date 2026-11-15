"""Unit tests for extract_t0_frames script with synthetic multi-row-group Parquet."""

import io
from pathlib import Path
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import extract_t0_frames


def _create_synthetic_jpeg_bytes(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    img = Image.new("RGB", (64, 64), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def synthetic_parquet_path(tmp_path: Path) -> Path:
    """Create a 3-row-group Parquet file with multiple camera columns."""
    img1 = _create_synthetic_jpeg_bytes((255, 0, 0))
    img2 = _create_synthetic_jpeg_bytes((0, 255, 0))
    img3 = _create_synthetic_jpeg_bytes((0, 0, 255))

    schema = pa.schema([
        ("scenario_id", pa.string()),
        ("frames_camera_front", pa.list_(pa.struct([("bytes", pa.binary())]))),
        ("frames_camera_front_left", pa.list_(pa.struct([("bytes", pa.binary())]))),
        ("frames_camera_rear", pa.list_(pa.struct([("bytes", pa.binary())]))),
    ])

    file_path = tmp_path / "synthetic_test.parquet"

    with pq.ParquetWriter(file_path, schema) as writer:
        # Row Group 0: Scenario 0000000
        t0 = pa.Table.from_arrays([
            pa.array(["0000000"]),
            pa.array([[{"bytes": img1}]]),
            pa.array([[{"bytes": img1}]]),
            pa.array([[{"bytes": img1}]]),
        ], schema=schema)
        writer.write_table(t0)

        # Row Group 1: Scenario 0000001
        t1 = pa.Table.from_arrays([
            pa.array(["0000001"]),
            pa.array([[{"bytes": img2}]]),
            pa.array([[{"bytes": img2}]]),
            pa.array([[{"bytes": img2}]]),
        ], schema=schema)
        writer.write_table(t1)

        # Row Group 2: Scenario 0000002
        t2 = pa.Table.from_arrays([
            pa.array(["0000002"]),
            pa.array([[{"bytes": img3}]]),
            pa.array([[{"bytes": img3}]]),
            pa.array([[{"bytes": img3}]]),
        ], schema=schema)
        writer.write_table(t2)

    return file_path


def test_front_camera_only_extraction_and_early_stopping(synthetic_parquet_path: Path, tmp_path: Path):
    out_dir = tmp_path / "extracted_t0"

    results, stats = extract_t0_frames.process_parquet_file(
        parquet_source=synthetic_parquet_path,
        output_dir=out_dir,
        target_scenarios={"0000001"},
        target_cameras={"front"},
        resize=False,
        return_stats=True,
    )

    # 1. Verify only scenario 0000001 was extracted
    assert set(results.keys()) == {"0000001"}
    assert (out_dir / "0000001" / "front.jpg").is_file()
    assert not (out_dir / "0000001" / "front_left.jpg").exists()
    assert not (out_dir / "0000001" / "rear.jpg").exists()
    assert not (out_dir / "0000000").exists()
    assert not (out_dir / "0000002").exists()

    # 2. Verify only requested camera column was read
    assert "frames_camera_front" in stats["compressed_bytes_by_column"]
    assert "frames_camera_front_left" not in stats["compressed_bytes_by_column"]
    assert "frames_camera_rear" not in stats["compressed_bytes_by_column"]

    # 3. Verify early stopping: row groups 0 and 1 scanned, row group 2 stopped early
    assert stats["row_groups_scanned"] == 2
    assert stats["matching_row_groups"] == [1]
    assert stats["scenarios_in_matching_groups"] == [1]
    assert stats["total_compressed_bytes_read"] > 0


def test_multiple_cameras_extraction(synthetic_parquet_path: Path, tmp_path: Path):
    out_dir = tmp_path / "extracted_multi_cam"

    results, stats = extract_t0_frames.process_parquet_file(
        parquet_source=synthetic_parquet_path,
        output_dir=out_dir,
        target_scenarios={"0000000"},
        target_cameras={"front", "rear"},
        resize=False,
        return_stats=True,
    )

    assert set(results.keys()) == {"0000000"}
    assert (out_dir / "0000000" / "front.jpg").is_file()
    assert (out_dir / "0000000" / "rear.jpg").is_file()
    assert not (out_dir / "0000000" / "front_left.jpg").exists()

    # Scenario 0000000 is in row group 0 -> stopped immediately after row group 0
    assert stats["row_groups_scanned"] == 1
    assert stats["matching_row_groups"] == [0]


def test_missing_scenario_causes_clear_failure(synthetic_parquet_path: Path, tmp_path: Path):
    out_dir = tmp_path / "extracted_missing"

    exit_code = extract_t0_frames.main([
        "--parquet", str(synthetic_parquet_path),
        "--output-dir", str(out_dir),
        "--scenarios", "9999999",
        "--cameras", "front",
    ])

    assert exit_code == 3
