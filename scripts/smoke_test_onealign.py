#!/usr/bin/env python3
"""Smoke test and 8-bit vs FP16 parity gate for upstream OneAlign video scoring.

Verifies the full `q-future/one-align` video-scoring contract:
  1. Upstream 1 fps frame sampling (`[0, 5, 10, 15, 20]` for 25-frame, 5 fps clips).
  2. Upstream `score([frames], task_="quality", input_="video", return_dict=True)` output:
     5-level logits/probabilities over `["excellent", "good", "fair", "poor", "bad"]`,
     raw score (`probs @ [5, 4, 3, 2, 1]`), and WorldMark `[0, 100]` PQ.
  3. 8-bit LLM hybrid loading on 16 GiB T4 (`--precision 8bit`) vs pure FP16 on 24 GiB+ L4
     (`--precision fp16`), recording load latency, peak VRAM, and per-clip throughput.
  4. Rank preservation (`compare` mode): Spearman rho, pairwise concordance, score offset,
     and 5-level probability L1 drift between 8-bit (T4) and FP16 (L4) reports.

Examples:
  # CPU-only input contract check on the current archive (extracts only 12 clips):
  python scripts/smoke_test_onealign.py score --videos-zip submissions/videos.zip \\
      --limit 12 --verify-input-only --out /tmp/onealign_input_check.json

  # T4 8-bit run inside /tmp/qalign_venv:
  python scripts/smoke_test_onealign.py score --videos-zip submissions/videos.zip \\
      --limit 12 --precision 8bit --out /tmp/onealign_t4_8bit.json

  # L4 FP16 reference run on the same clips:
  python scripts/smoke_test_onealign.py score --videos-zip submissions/videos.zip \\
      --limit 12 --precision fp16 --out /tmp/onealign_l4_fp16.json

  # Compare T4 8-bit vs L4 FP16 parity gate:
  python scripts/smoke_test_onealign.py compare \\
      --eightbit /tmp/onealign_t4_8bit.json --fp16 /tmp/onealign_l4_fp16.json \\
      --out reports/onealign_t4_vs_l4_smoke.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import zipfile

import cv2
import numpy as np
from scipy import stats
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitscenes import spec
from scripts.eval_world_model_qalign import _attach_onealign_weight_tensor, load_one_align_model

LEVEL_NAMES: Tuple[str, ...] = ("excellent", "good", "fair", "poor", "bad")
LEVEL_WEIGHTS_RAW: Tuple[float, ...] = (5.0, 4.0, 3.0, 2.0, 1.0)
LEVEL_WEIGHTS_UNIT_100: Tuple[float, ...] = (100.0, 75.0, 50.0, 25.0, 0.0)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sample_upstream_1fps_frames(mp4_path: Path) -> Dict[str, Any]:
    """Decode and sample frames at 1 fps matching `q_align/evaluate/scorer.py:load_video`."""
    from PIL import Image

    sampled_indices: List[int]
    frames_pil: List[Any]
    backend: str

    try:
        from decord import VideoReader, cpu

        vr = VideoReader(str(mp4_path), ctx=cpu(0))
        total_frames = len(vr)
        fps = float(vr.get_avg_fps() or spec.FPS)
        step = max(1, int(round(fps)))
        n_sec = max(1, int(round(total_frames / max(fps, 1e-6))))
        sampled_indices = [i * step for i in range(n_sec) if i * step < total_frames]
        batch = vr.get_batch(sampled_indices).asnumpy()
        frames_pil = [Image.fromarray(batch[i]) for i in range(len(sampled_indices))]
        backend = "decord"
    except Exception:
        cap = cv2.VideoCapture(str(mp4_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or spec.FPS)
        step = max(1, int(round(fps)))
        sampled_indices = []
        frames_pil = []
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % step == 0:
                sampled_indices.append(idx)
                frames_pil.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            idx += 1
        cap.release()
        if total_frames <= 0:
            total_frames = idx
        backend = "opencv"

    if not frames_pil:
        raise RuntimeError(f"Decoded 0 frames from {mp4_path}")

    frame_stats = []
    for pos, (fidx, img) in enumerate(zip(sampled_indices, frames_pil)):
        arr = np.asarray(img, dtype=np.uint8)
        frame_stats.append(
            {
                "sample_slot": pos,
                "frame_index": int(fidx),
                "width": int(arr.shape[1]),
                "height": int(arr.shape[0]),
                "mean_luma": round(float(arr.mean()), 3),
                "std_luma": round(float(arr.std()), 3),
            }
        )

    expected_indices = list(range(0, spec.FRAME_COUNT, int(spec.FPS)))
    contract_ok = (
        total_frames == spec.FRAME_COUNT
        and abs(fps - float(spec.FPS)) <= 0.05
        and sampled_indices == expected_indices
    )

    return {
        "frames": frames_pil,
        "backend": backend,
        "source_frame_count": int(total_frames),
        "source_fps": round(fps, 4),
        "sampled_indices": sampled_indices,
        "sampled_count": len(frames_pil),
        "expected_indices": expected_indices,
        "input_contract_ok": contract_ok,
        "frame_stats": frame_stats,
        "video_sha256": _sha256_file(mp4_path),
        "size_bytes": int(mp4_path.stat().st_size),
    }


def extract_five_level_distribution(logits: Any, reported_score: Optional[float] = None) -> Dict[str, Any]:
    """Compute softmax probabilities over the five OneAlign quality levels and verify identity."""
    logits_t = torch.as_tensor(logits, dtype=torch.float32).reshape(-1)
    if logits_t.numel() != len(LEVEL_NAMES):
        raise ValueError(f"Expected 5 quality level logits, got shape {tuple(logits_t.shape)}")

    probs_t = torch.softmax(logits_t, dim=-1)
    probs = [float(x) for x in probs_t.tolist()]
    logits_list = [float(x) for x in logits_t.tolist()]

    q_raw = float(sum(p * w for p, w in zip(probs, LEVEL_WEIGHTS_RAW)))
    pq_worldmark_100 = float(sum(p * w for p, w in zip(probs, LEVEL_WEIGHTS_UNIT_100)))
    pq_from_raw = 100.0 * (q_raw - 1.0) / 4.0

    if abs(pq_worldmark_100 - pq_from_raw) > 1e-2:
        raise AssertionError(
            f"Probability weight mismatch: worldmark={pq_worldmark_100:.4f} vs raw={pq_from_raw:.4f}"
        )
    if reported_score is not None and abs(float(reported_score) - q_raw) > 0.02:
        raise AssertionError(
            f"Reported score {reported_score:.4f} diverges from softmax expected value {q_raw:.4f}"
        )

    return {
        "level_logits": {name: round(val, 6) for name, val in zip(LEVEL_NAMES, logits_list)},
        "level_probs": {name: round(val, 6) for name, val in zip(LEVEL_NAMES, probs)},
        "prob_sum": round(float(sum(probs)), 6),
        "q_raw": round(q_raw, 6),
        "pq_worldmark_100": round(pq_worldmark_100, 4),
    }


def detect_model_precision(model: Any) -> str:
    """Inspect model modules to report whether 8-bit LLM quantization is active."""
    for mod in model.modules():
        if mod.__class__.__name__ == "Linear8bitLt":
            return "8bit_hybrid"
    for param in model.parameters():
        return str(param.dtype).replace("torch.", "")
    return "unknown"


def verify_preferential_tokens(model: Any) -> Dict[str, Any]:
    """Check `model.preferential_ids_` maps cleanly onto the 5 quality rating words."""
    pref_ids = list(getattr(model, "preferential_ids_", []))
    tokenizer = getattr(model, "tokenizer", None)
    decoded: List[str] = []
    if tokenizer is not None and pref_ids:
        for tid in pref_ids:
            try:
                decoded.append(str(tokenizer.decode([int(tid)])).strip())
            except Exception:
                decoded.append(str(tid))
    return {
        "preferential_ids": [int(x) for x in pref_ids],
        "decoded_tokens": decoded,
        "tokens_match_expected": decoded == list(LEVEL_NAMES) if decoded else None,
    }


def load_one_align_for_smoke_test(precision: str = "auto") -> Tuple[Any, Dict[str, Any]]:
    """Load `q-future/one-align` under explicit `8bit`, `fp16`, or `auto` policy."""
    if precision not in ("auto", "8bit", "fp16"):
        raise ValueError(f"Unsupported precision {precision!r}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t_load = time.time()
    if precision == "auto":
        md = load_one_align_model()
    else:
        from transformers import AutoModelForCausalLM

        kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        if precision == "8bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["vision_model", "visual_abstractor", "lm_head"],
            )
        try:
            md = AutoModelForCausalLM.from_pretrained("q-future/one-align", attn_implementation="eager", **kwargs)
        except TypeError:
            md = AutoModelForCausalLM.from_pretrained("q-future/one-align", **kwargs)
        _attach_onealign_weight_tensor(md)

    load_time_s = time.time() - t_load
    detected_precision = detect_model_precision(md)
    if precision == "fp16" and detected_precision == "8bit_hybrid":
        raise RuntimeError("Requested `fp16` mode but 8-bit quantized modules were loaded.")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    vram_total = (
        torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )
    peak_vram = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )

    meta = {
        "requested_precision": precision,
        "detected_precision": detected_precision,
        "load_time_s": round(load_time_s, 3),
        "gpu_name": gpu_name,
        "gpu_vram_total_gib": round(vram_total, 2),
        "post_load_peak_vram_gib": round(peak_vram, 3),
        "preferential_tokens": verify_preferential_tokens(md),
    }
    return md, meta


def score_video_with_distribution(model: Any, mp4_path: Path) -> Dict[str, Any]:
    """Score one MP4 clip via upstream video mode and unpack its 5-level distribution."""
    sampled = sample_upstream_1fps_frames(mp4_path)
    frames = sampled["frames"]
    if not sampled["input_contract_ok"]:
        raise ValueError(
            f"Clip {mp4_path} failed upstream 5-frame/5fps contract: "
            f"frames={sampled['source_frame_count']} fps={sampled['source_fps']} indices={sampled['sampled_indices']}"
        )

    t0 = time.time()
    with torch.inference_mode():
        res = model.score([frames], task_="quality", input_="video", return_dict=True)
    latency_s = time.time() - t0

    if not isinstance(res, dict) or "logits" not in res or "scores" not in res:
        raise RuntimeError(f"Upstream score(..., return_dict=True) returned unexpected object: {type(res)}")

    raw_score = float(torch.as_tensor(res["scores"]).reshape(-1)[0].item())
    dist = extract_five_level_distribution(res["logits"], reported_score=raw_score)
    return {
        "source_frame_count": sampled["source_frame_count"],
        "source_fps": sampled["source_fps"],
        "sampled_indices": sampled["sampled_indices"],
        "sampled_count": sampled["sampled_count"],
        "backend": sampled["backend"],
        "video_sha256": sampled["video_sha256"],
        "size_bytes": sampled["size_bytes"],
        "latency_s": round(latency_s, 4),
        **dist,
    }


def select_smoke_clip_keys(
    available_keys: Sequence[str],
    *,
    explicit_clips: Sequence[str] = (),
    scenarios: Sequence[str] = (),
    limit: int = 12,
) -> List[str]:
    """Pick deterministic, reproducible clips (front + surround) across scenarios."""
    key_set = set(available_keys)
    if explicit_clips:
        selected: List[str] = []
        for raw in explicit_clips:
            norm = raw if raw.endswith(".mp4") else f"{raw}.mp4"
            if norm not in key_set:
                raise KeyError(f"Requested clip {norm!r} not found in source")
            selected.append(norm)
        return selected

    by_sid: Dict[str, List[str]] = {}
    for key in sorted(available_keys):
        parts = Path(key).parts
        if len(parts) < 2:
            continue
        sid = parts[-2]
        by_sid.setdefault(sid, []).append(key)

    sids = sorted(by_sid.keys())
    if scenarios:
        want = {s.strip().zfill(7) for s in scenarios if s.strip()}
        sids = [sid for sid in sids if sid in want]
    if not sids:
        return []

    if limit <= 0 or limit >= len(available_keys):
        return [key for sid in sids for key in sorted(by_sid[sid])]

    # Deterministic spread across scenarios, rotating front + surround views so all camera types are tested.
    chosen: List[str] = []
    cam_cycle = ("front.mp4", "front_left.mp4", "front_right.mp4", "rear.mp4", "rear_left.mp4", "rear_right.mp4")
    # Pick evenly spaced scenario anchors first, front camera prioritized, then surround views if slots remain.
    n_anchors = min(len(sids), max(1, limit))
    if n_anchors == 1:
        anchor_indices = [0]
    else:
        anchor_indices = sorted({round(i * (len(sids) - 1) / (n_anchors - 1)) for i in range(n_anchors)})

    for step_idx, s_idx in enumerate(anchor_indices):
        sid = sids[s_idx]
        sid_keys = set(by_sid[sid])
        # First 2/3 of slots test front cameras (50% PQ weight); remaining rotate across surround cameras.
        pref_cam = "front.mp4" if step_idx % 3 != 2 else cam_cycle[step_idx % len(cam_cycle)]
        candidate = f"{sid}/{pref_cam}"
        if candidate not in sid_keys:
            candidate = sorted(sid_keys)[0]
        if candidate not in chosen:
            chosen.append(candidate)
        if len(chosen) >= limit:
            break

    return chosen[:limit]


def _spearman_rank_corr(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, Optional[float]]:
    if len(xs) < 2:
        return float("nan"), None
    if len(set(xs)) <= 1 or len(set(ys)) <= 1:
        return float("nan"), None
    res = stats.spearmanr(xs, ys)
    rho = float(res.statistic)
    pval = float(res.pvalue) if not math.isnan(float(res.pvalue)) else None
    return rho, pval


def _pairwise_order_agreement(a: Sequence[float], b: Sequence[float]) -> float:
    """Fraction of pairs with non-tied `b` ordering preserved in `a`."""
    concordant = 0
    total = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            db = b[i] - b[j]
            if abs(db) < 1e-9:
                continue
            total += 1
            da = a[i] - a[j]
            if da * db > 0:
                concordant += 1
    return (concordant / total) if total > 0 else 1.0


def compare_smoke_reports(
    eightbit_report: Dict[str, Any],
    fp16_report: Dict[str, Any],
    *,
    min_spearman: float = 0.95,
    min_pairwise_agreement: float = 0.90,
) -> Dict[str, Any]:
    """Verify 8-bit T4 scores preserve FP16 L4 ranking and measure bias/level drift."""
    clips_8 = {c["clip_key"]: c for c in eightbit_report.get("clips", [])}
    clips_16 = {c["clip_key"]: c for c in fp16_report.get("clips", [])}
    shared = sorted(set(clips_8) & set(clips_16))
    if not shared:
        raise ValueError("No overlapping clip_key entries between 8-bit and FP16 smoke reports.")

    q8 = [float(clips_8[k]["q_raw"]) for k in shared]
    q16 = [float(clips_16[k]["q_raw"]) for k in shared]
    pq8 = [float(clips_8[k]["pq_worldmark_100"]) for k in shared]
    pq16 = [float(clips_16[k]["pq_worldmark_100"]) for k in shared]

    q_diffs = np.asarray(q8, dtype=np.float64) - np.asarray(q16, dtype=np.float64)
    pq_diffs = np.asarray(pq8, dtype=np.float64) - np.asarray(pq16, dtype=np.float64)
    prob_l1s = [
        sum(
            abs(float(clips_8[k]["level_probs"][lvl]) - float(clips_16[k]["level_probs"][lvl]))
            for lvl in LEVEL_NAMES
        )
        for k in shared
    ]

    rho, pval = _spearman_rank_corr(q8, q16)
    pearson_r = (
        float(np.corrcoef(q8, q16)[0, 1])
        if len(shared) >= 2 and len(set(q8)) > 1 and len(set(q16)) > 1
        else float("nan")
    )
    agreement = _pairwise_order_agreement(q8, q16)

    per_clip = [
        {
            "clip_key": k,
            "q_8bit": clips_8[k]["q_raw"],
            "q_fp16": clips_16[k]["q_raw"],
            "delta_q": round(float(clips_8[k]["q_raw"] - clips_16[k]["q_raw"]), 6),
            "pq_8bit": clips_8[k]["pq_worldmark_100"],
            "pq_fp16": clips_16[k]["pq_worldmark_100"],
            "delta_pq": round(float(clips_8[k]["pq_worldmark_100"] - clips_16[k]["pq_worldmark_100"]), 4),
            "level_prob_l1": round(prob_l1s[idx], 6),
        }
        for idx, k in enumerate(shared)
    ]

    gate_passed = (
        not math.isnan(rho)
        and rho >= min_spearman
        and agreement >= min_pairwise_agreement
    )

    return {
        "paired_count": len(shared),
        "eightbit_model": eightbit_report.get("model", {}),
        "fp16_model": fp16_report.get("model", {}),
        "spearman_rho": None if math.isnan(rho) else round(rho, 6),
        "spearman_pvalue": None if pval is None else round(pval, 8),
        "pearson_r": None if math.isnan(pearson_r) else round(pearson_r, 6),
        "pairwise_order_agreement": round(agreement, 6),
        "mean_offset_q_raw": round(float(np.mean(q_diffs)), 6),
        "std_offset_q_raw": round(float(np.std(q_diffs, ddof=1) if len(q_diffs) > 1 else 0.0), 6),
        "max_abs_diff_q_raw": round(float(np.max(np.abs(q_diffs))), 6),
        "mean_offset_pq_100": round(float(np.mean(pq_diffs)), 4),
        "std_offset_pq_100": round(float(np.std(pq_diffs, ddof=1) if len(pq_diffs) > 1 else 0.0), 4),
        "max_abs_diff_pq_100": round(float(np.max(np.abs(pq_diffs))), 4),
        "mean_level_prob_l1": round(float(np.mean(prob_l1s)), 6),
        "thresholds": {
            "min_spearman": min_spearman,
            "min_pairwise_agreement": min_pairwise_agreement,
        },
        "gate_passed": gate_passed,
        "per_clip": per_clip,
    }


def _run_score_cmd(args: argparse.Namespace) -> int:
    if (args.videos_zip is None) == (args.clips_dir is None):
        raise ValueError("Provide exactly one of --videos-zip or --clips-dir.")

    explicit = [c.strip() for c in args.clips.split(",") if c.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    tmp_dir: Optional[tempfile.TemporaryDirectory] = None
    try:
        clip_paths: List[Tuple[str, Path]] = []
        if args.videos_zip is not None:
            with zipfile.ZipFile(args.videos_zip, "r") as zf:
                names = [n for n in zf.namelist() if n.endswith(".mp4") and not n.startswith("__MACOSX")]
                selected = select_smoke_clip_keys(
                    names,
                    explicit_clips=explicit,
                    scenarios=scenarios,
                    limit=args.limit,
                )
                tmp_dir = tempfile.TemporaryDirectory(prefix="onealign_smoke_")
                tmp_root = Path(tmp_dir.name)
                for key in selected:
                    target = tmp_root / key
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(key) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    clip_paths.append((key, target))
        else:
            root = args.clips_dir
            all_mp4s = [str(p.relative_to(root).as_posix()) for p in sorted(root.rglob("*.mp4"))]
            selected = select_smoke_clip_keys(
                all_mp4s,
                explicit_clips=explicit,
                scenarios=scenarios,
                limit=args.limit,
            )
            clip_paths = [(key, root / key) for key in selected]

        if not clip_paths:
            raise RuntimeError("No clips selected for smoke test.")

        if args.verify_input_only:
            model = None
            model_meta: Dict[str, Any] = {"requested_precision": "input_only", "detected_precision": "cpu_verify"}
        else:
            model, model_meta = load_one_align_for_smoke_test(args.precision)

        clip_records: List[Dict[str, Any]] = []
        for idx, (clip_key, mp4) in enumerate(clip_paths, start=1):
            if args.verify_input_only:
                sampled = sample_upstream_1fps_frames(mp4)
                if not sampled["input_contract_ok"]:
                    raise ValueError(f"{clip_key} failed upstream input contract: {sampled['sampled_indices']}")
                rec = {
                    "clip_key": clip_key,
                    "source_frame_count": sampled["source_frame_count"],
                    "source_fps": sampled["source_fps"],
                    "sampled_indices": sampled["sampled_indices"],
                    "sampled_count": sampled["sampled_count"],
                    "backend": sampled["backend"],
                    "video_sha256": sampled["video_sha256"],
                    "size_bytes": sampled["size_bytes"],
                    "frame_stats": sampled["frame_stats"],
                }
            else:
                res = score_video_with_distribution(model, mp4)
                rec = {"clip_key": clip_key, **res}
                print(
                    f"  [{idx:02d}/{len(clip_paths)}] {clip_key}: q={rec['q_raw']:.4f} "
                    f"PQ={rec['pq_worldmark_100']:.2f} probs={rec['level_probs']} "
                    f"({rec['latency_s']:.2f}s)",
                    flush=True,
                )
            clip_records.append(rec)

        if not args.verify_input_only and torch.cuda.is_available():
            model_meta["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)

        summary: Dict[str, Any] = {"clip_count": len(clip_records)}
        if not args.verify_input_only:
            latencies = [float(r["latency_s"]) for r in clip_records]
            summary.update(
                {
                    "mean_q_raw": round(float(np.mean([r["q_raw"] for r in clip_records])), 4),
                    "mean_pq_worldmark_100": round(
                        float(np.mean([r["pq_worldmark_100"] for r in clip_records])), 3
                    ),
                    "mean_clip_latency_s": round(float(np.mean(latencies)), 4),
                    "clips_per_sec": round(len(clip_records) / max(float(sum(latencies)), 1e-6), 3),
                }
            )
        else:
            summary["all_inputs_valid_5fps_25f"] = True

        payload = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "verify_input_only" if args.verify_input_only else "score",
            "source": str(args.videos_zip or args.clips_dir),
            "model": model_meta,
            "summary": summary,
            "clips": clip_records,
        }
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[+] Wrote smoke test report -> {args.out}", flush=True)
        else:
            print(json.dumps(payload, indent=2), flush=True)
        return 0
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()


def _run_compare_cmd(args: argparse.Namespace) -> int:
    rep8 = json.loads(args.eightbit.read_text(encoding="utf-8"))
    rep16 = json.loads(args.fp16.read_text(encoding="utf-8"))
    comp = compare_smoke_reports(
        rep8,
        rep16,
        min_spearman=args.min_spearman,
        min_pairwise_agreement=args.min_pairwise_agreement,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(comp, indent=2), encoding="utf-8")
        print(f"[+] Wrote 8-bit vs FP16 comparison -> {args.out}", flush=True)
    else:
        print(json.dumps(comp, indent=2), flush=True)
    return 0 if comp["gate_passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OneAlign upstream video-scoring smoke test & precision comparator.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score", help="Verify 5-frame upstream video input and run OneAlign scoring.")
    p_score.add_argument("--videos-zip", type=Path, default=None)
    p_score.add_argument("--clips-dir", type=Path, default=None)
    p_score.add_argument("--clips", type=str, default="", help="Comma-separated <sid>/<camera>.mp4 keys")
    p_score.add_argument("--scenarios", type=str, default="", help="Comma-separated scenario IDs")
    p_score.add_argument("--limit", type=int, default=12)
    p_score.add_argument("--precision", choices=["auto", "8bit", "fp16"], default="auto")
    p_score.add_argument("--verify-input-only", action="store_true", help="CPU-only input contract check without loading OneAlign")
    p_score.add_argument("--out", type=Path, default=None)

    p_cmp = sub.add_parser("compare", help="Compare T4 8-bit report against L4 FP16 reference report.")
    p_cmp.add_argument("--eightbit", type=Path, required=True)
    p_cmp.add_argument("--fp16", type=Path, required=True)
    p_cmp.add_argument("--min-spearman", type=float, default=0.95)
    p_cmp.add_argument("--min-pairwise-agreement", type=float, default=0.90)
    p_cmp.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "score":
        return _run_score_cmd(args)
    return _run_compare_cmd(args)


if __name__ == "__main__":
    raise SystemExit(main())
