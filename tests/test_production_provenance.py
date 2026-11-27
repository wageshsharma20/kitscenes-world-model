"""Tests for submission-provenance manifest generation in scripts/run_production_reproj.py."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kitscenes import spec
from scripts import run_production_reproj as prod


def _write_fake_scenario_inputs(root: Path, sid: str) -> None:
    scen_dir = root / sid
    scen_dir.mkdir(parents=True, exist_ok=True)
    for cam in spec.CAMERAS:
        # Valid non-empty placeholder JPEG not needed when _gpu_worker is monkeypatched
        (scen_dir / f"{cam}.jpg").write_bytes(b"fake")


def test_default_render_scale_remains_two():
    args = prod.parse_args([])
    assert args.render_scale == 2


def test_camera_calibration_provenance_is_deterministic_and_hashed():
    c1 = prod._collect_camera_calibration_provenance()
    c2 = prod._collect_camera_calibration_provenance()
    assert c1["rig_params_sha256"] == c2["rig_params_sha256"]
    assert len(c1["rig_params_sha256"]) == 64
    assert c1["config_file_sha256"] is not None
    assert len(c1["config_file_sha256"]) == 64


def test_provenance_manifest_records_all_fields_and_crf_clamping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    traj_path = tmp_path / "trajectories.jsonl"
    traj_lines = [
        json.dumps({"scenario_id": "0000000", "future_trajectory": [[1.0 * i, 0.0] for i in range(spec.WAYPOINT_COUNT)]}),
        json.dumps({"scenario_id": "0000001", "future_trajectory": [[0.8 * i, 0.1] for i in range(spec.WAYPOINT_COUNT)]}),
    ]
    traj_path.write_text("\n".join(traj_lines) + "\n", encoding="utf-8")

    hist_path = tmp_path / "history.jsonl"
    hist_lines = [
        json.dumps({"scenario_id": "0000000", "scenario_type": "1 heavy rain"}),
        json.dumps({"scenario_id": "0000001", "scenario_type": "4 intersection"}),
    ]
    hist_path.write_text("\n".join(hist_lines) + "\n", encoding="utf-8")

    t0_dir = tmp_path / "test_t0_native"
    _write_fake_scenario_inputs(t0_dir, "0000000")
    _write_fake_scenario_inputs(t0_dir, "0000001")

    out_videos = tmp_path / "out_videos"
    # Pre-seed scenario 0000001 so it takes the resume branch cleanly
    seeded_dir = out_videos / "0000001"
    seeded_dir.mkdir(parents=True, exist_ok=True)
    for cam in spec.CAMERAS:
        (seeded_dir / f"{cam}.mp4").write_bytes(b"\x00" * 2048)

    out_zip = tmp_path / "videos.zip"
    manifest_path = tmp_path / "generation_manifest.json"

    class FakeModel:
        def __init__(self, **kwargs):
            self.last_hole_fraction = {}

        def generate_scenario(self, sid, fut_traj, cond, past_images=None, past_trajectory=None):
            self.last_hole_fraction = {cam: [0.01, 0.02] for cam in spec.CAMERAS}
            return {cam: np.zeros((spec.FRAME_COUNT, spec.HEIGHT, spec.WIDTH, 3), dtype=np.uint8) for cam in spec.CAMERAS}

    def fake_encode(frames_rgb, out_mp4: Path, crf: int = 14):
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        # 2500 bytes per file * 12 files > tiny max-zip-gib threshold to trigger CRF clamping
        out_mp4.write_bytes(b"A" * 2500)

    def fake_reencode(mp4_path: Path, crf: int):
        mp4_path.write_bytes(b"B" * 1500)

    def fake_build_archive(videos_dir, output=None, allow_partial=False, check_media=True):
        output = Path(output)
        output.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
        return SimpleNamespace(
            ok=True,
            errors=[],
            warnings=[],
            size_bytes=output.stat().st_size,
            videos=12,
            scenarios_covered=2,
            scenarios_complete=2,
            media_checked=12,
            summary_lines=lambda: ["fake archive ok"],
            as_dict=lambda: {
                "path": str(output),
                "ok": True,
                "errors": [],
                "warnings": [],
                "size_bytes": output.stat().st_size,
                "videos": 12,
                "scenarios_covered": 2,
                "scenarios_complete": 2,
                "media_checked": 12,
            },
        )

    monkeypatch.setattr(prod, "Depth3DFullFoVWorldModel", FakeModel)
    monkeypatch.setattr(prod, "encode_rgb_h264", fake_encode)
    monkeypatch.setattr(prod, "_reencode_mp4_crf", fake_reencode)
    monkeypatch.setattr(prod.archive, "build_archive", fake_build_archive)

    argv = [
        "--trajectories", str(traj_path),
        "--history", str(hist_path),
        "--t0-dir", str(t0_dir),
        "--out-videos-dir", str(out_videos),
        "--out-zip", str(out_zip),
        "--manifest-out", str(manifest_path),
        "--no-depth",
        "--no-lama",
        "--min-lama-hole-pixels", "600",
        "--crf-sharpen", "0.24",
        "--crf", "17",
        "--fallback-crf", "18",
        "--max-zip-gib", "0.00001",
        "--gpus", "",
        "--pack-zip",
        "--allow-partial",
    ]
    rc = prod.main(argv)
    assert rc == 0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["cli"]["argv"] == argv
    assert manifest["git"]["commit"] is not None
    assert isinstance(manifest["git"]["dirty"], bool)

    rcfg = manifest["render_config"]
    assert rcfg["render_scale"] == 2
    assert rcfg["depth_enabled"] is False
    assert rcfg["depth_model"] is None
    assert rcfg["lama_enabled"] is False
    assert rcfg["min_lama_hole_pixels"] == 600
    assert rcfg["crf_sharpen"] == pytest.approx(0.24)
    assert rcfg["crf"] == 17
    assert rcfg["fallback_crf"] == 18
    assert rcfg["clamped_to_fallback_crf"] is True
    assert rcfg["effective_crf"] == 18

    assert manifest["inputs"]["trajectories_sha256"] == prod._sha256_file(traj_path)
    assert len(manifest["camera_calibration"]["rig_params_sha256"]) == 64

    s0 = manifest["scenarios"]["0000000"]
    s1 = manifest["scenarios"]["0000001"]
    assert s0["status"] == "rendered"
    assert s0["scenario_type"] == "1 heavy rain"
    assert s0["depth_active"] is False
    assert s0["cameras"]["front"]["status"] == "rendered"
    assert s0["cameras"]["front"]["size_bytes"] == 1500
    assert s0["cameras"]["front"]["effective_crf"] == 18
    assert s0["cameras"]["front"]["hole_fraction_mean"] == pytest.approx(0.015)

    assert s1["status"] == "resumed"
    assert s1["cameras"]["rear"]["status"] == "resumed"
    assert s1["cameras"]["rear"]["effective_crf"] == 18

    assert manifest["summary"]["scenarios_total"] == 2
    assert manifest["summary"]["scenarios_rendered"] == 1
    assert manifest["summary"]["scenarios_resumed"] == 1
    assert manifest["summary"]["cameras_total"] == 12
    assert manifest["archive"]["ok"] is True
    assert len(manifest["archive"]["sha256"]) == 64
