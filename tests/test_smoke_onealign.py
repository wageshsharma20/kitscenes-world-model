"""Tests for the upstream OneAlign smoke-test harness (5-frame input, 5-level probs, T4 vs L4 gate)."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch

from kitscenes import spec
from scripts import eval_world_model_qalign as eq
from scripts import smoke_test_onealign as sm


def test_five_level_distribution_and_worldmark_identity():
    logits = torch.tensor([[3.0, 1.5, 0.0, -2.0, -4.0]], dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)[0]
    expected_raw = float((probs * torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])).sum().item())

    dist = sm.extract_five_level_distribution(logits, reported_score=expected_raw)
    assert set(dist["level_probs"]) == {"excellent", "good", "fair", "poor", "bad"}
    assert dist["prob_sum"] == pytest.approx(1.0, abs=1e-4)
    assert dist["q_raw"] == pytest.approx(expected_raw, abs=1e-4)
    assert dist["pq_worldmark_100"] == pytest.approx(100.0 * (expected_raw - 1.0) / 4.0, abs=1e-2)


def test_extract_five_level_distribution_detects_score_mismatch():
    logits = torch.tensor([[3.0, 1.0, 0.0, -1.0, -2.0]])
    with pytest.raises(AssertionError, match="diverges"):
        sm.extract_five_level_distribution(logits, reported_score=1.2)


def test_select_smoke_clip_keys_deterministic_spread():
    keys = [f"{i:07d}/{cam}.mp4" for i in range(20) for cam in spec.CAMERAS]
    s1 = sm.select_smoke_clip_keys(keys, limit=12)
    s2 = sm.select_smoke_clip_keys(keys, limit=12)
    assert s1 == s2
    assert len(s1) == 12
    assert any(k.endswith("/front.mp4") for k in s1)
    assert any(not k.endswith("/front.mp4") for k in s1)


def test_upstream_1fps_video_input_and_scoring_roundtrip(tmp_path: Path):
    mp4_path = tmp_path / "0000000" / "front.mp4"
    frames = np.zeros((spec.FRAME_COUNT, spec.HEIGHT, spec.WIDTH, 3), dtype=np.uint8)
    for idx in range(spec.FRAME_COUNT):
        frames[idx, :, :, 0] = min(255, 20 + idx * 8)
        frames[idx, :, :, 1] = 80
    eq.encode_rgb_h264(frames, mp4_path, crf=18)

    sampled = sm.sample_upstream_1fps_frames(mp4_path)
    assert sampled["input_contract_ok"] is True
    assert sampled["source_frame_count"] == 25
    assert sampled["source_fps"] == pytest.approx(5.0, abs=0.05)
    assert sampled["sampled_indices"] == [0, 5, 10, 15, 20]
    assert sampled["sampled_count"] == 5

    class MockOneAlign:
        preferential_ids_ = [10, 11, 12, 13, 14]

        def score(self, images, task_="quality", input_="image", return_dict=False):
            assert task_ == "quality"
            assert input_ == "video"
            assert return_dict is True
            assert len(images) == 1 and len(images[0]) == 5
            logits = torch.tensor([[2.2, 1.4, -0.5, -2.0, -3.5]])
            w = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
            sc = torch.softmax(logits, -1) @ w
            return {"logits": logits, "scores": sc}

    res = sm.score_video_with_distribution(MockOneAlign(), mp4_path)
    assert res["sampled_indices"] == [0, 5, 10, 15, 20]
    assert res["q_raw"] > 4.0
    assert res["level_probs"]["excellent"] > res["level_probs"]["good"]


def test_verify_input_only_on_zip_cli(tmp_path: Path):
    mp4_path = tmp_path / "src" / "0000000" / "front.mp4"
    frames = np.full((spec.FRAME_COUNT, spec.HEIGHT, spec.WIDTH, 3), 120, dtype=np.uint8)
    eq.encode_rgb_h264(frames, mp4_path, crf=20)

    zip_path = tmp_path / "mini.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(mp4_path, arcname="0000000/front.mp4")

    out_json = tmp_path / "verify.json"
    rc = sm.main([
        "score",
        "--videos-zip", str(zip_path),
        "--limit", "1",
        "--verify-input-only",
        "--out", str(out_json),
    ])
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["summary"]["all_inputs_valid_5fps_25f"] is True
    assert payload["clips"][0]["sampled_indices"] == [0, 5, 10, 15, 20]


def test_compare_8bit_vs_fp16_gate_pass_and_fail(tmp_path: Path):
    keys = [f"{i:07d}/front.mp4" for i in range(6)]
    fp16_vals = [4.1, 4.3, 4.5, 4.7, 4.85, 4.95]

    def _make_report(vals, shift=0.0):
        clips = []
        for k, q in zip(keys, vals):
            qv = q + shift
            clips.append({
                "clip_key": k,
                "q_raw": qv,
                "pq_worldmark_100": 100.0 * (qv - 1.0) / 4.0,
                "level_probs": {"excellent": 0.8, "good": 0.2, "fair": 0.0, "poor": 0.0, "bad": 0.0},
            })
        return {"model": {"requested_precision": "mock"}, "clips": clips}

    r16 = _make_report(fp16_vals)
    r8_good = _make_report(fp16_vals, shift=-0.04)
    stypes = {f"{i:07d}": ("1 heavy rain" if i < 3 else "4 intersection") for i in range(6)}
    comp_good = sm.compare_smoke_reports(r8_good, r16, scenario_types=stypes)
    assert comp_good["gate_passed"] is True
    assert comp_good["spearman_rho"] == pytest.approx(1.0)
    assert comp_good["mean_offset_q_raw"] == pytest.approx(-0.04)
    assert comp_good["mae_pq_100"] == pytest.approx(1.0, abs=1e-3)
    assert comp_good["top_quartile_agreement"] == pytest.approx(1.0)
    assert comp_good["bottom_quartile_agreement"] == pytest.approx(1.0)
    assert set(comp_good["by_scenario_type"]) == {"1 heavy rain", "4 intersection"}

    r8_bad = _make_report(list(reversed(fp16_vals)))
    comp_bad = sm.compare_smoke_reports(r8_bad, r16)
    assert comp_bad["gate_passed"] is False
    assert comp_bad["pairwise_order_agreement"] == pytest.approx(0.0)
    assert comp_bad["top_quartile_agreement"] == pytest.approx(0.0)
    assert comp_bad["bottom_quartile_agreement"] == pytest.approx(0.0)
