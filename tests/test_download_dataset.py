"""Tests for dataset download utilities and metadata streaming."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


# Ensure scripts and project root are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import download_dataset  # noqa: E402


def test_build_parser_defaults():
    parser = download_dataset.build_parser()
    args = parser.parse_args([])
    assert args.split == "test"
    assert args.output_dir == Path("data")
    assert not args.metadata_only
    assert not args.dry_run
    assert args.token == ""


def test_build_parser_custom_args():
    parser = download_dataset.build_parser()
    args = parser.parse_args([
        "--token", "hf_test_123",
        "--metadata-only",
        "--dry-run",
        "--split", "val",
        "--output-dir", "custom_data",
    ])
    assert args.token == "hf_test_123"
    assert args.metadata_only is True
    assert args.dry_run is True
    assert args.split == "val"
    assert args.output_dir == Path("custom_data")


def test_space_check_fit():
    # 5 GB required, 100 GB free -> fits
    result = download_dataset.space_check(5 * 1024**3, 100 * 1024**3)
    assert result is None


def test_space_check_exceed():
    # 34 GB required, 11 GB free -> refuses
    result = download_dataset.space_check(34 * 1024**3, 11 * 1024**3)
    assert result is not None
    assert "needs 34.0 GiB but only 11.0 GiB is free" in result


def test_main_missing_token(monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    exit_code = download_dataset.main(["--split", "test"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Hugging Face token is required" in captured.err


def test_main_dry_run_metadata_only(monkeypatch, capsys):
    mock_sibling = MagicMock()
    mock_sibling.rfilename = "data/test-00000-of-00001.parquet"
    mock_sibling.size = 34 * 1024**3

    mock_info = MagicMock()
    mock_info.siblings = [mock_sibling]

    with patch("huggingface_hub.HfApi.repo_info", return_value=mock_info):
        exit_code = download_dataset.main([
            "--token", "hf_mock",
            "--metadata-only",
            "--dry-run",
            "--split", "test",
        ])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Mode: --metadata-only" in captured.out
        assert "Dry run" in captured.out
