#!/usr/bin/env python3
"""Calibrate Depth3DFullFoVWorldModel against official val GT using Q-Align (OneAlign) + EMC.

Two clean execution stages so modern transformers (Depth-Anything-V2) and Q-Align
(transformers==4.36.1) run in isolated environments without dependency conflicts:

  Stage 1 (`--stage render`):
    Reads extracted validation scenarios (or unpacks val_calib.tar), renders H.264 876x568
    5fps videos across requested model variants, computes local PSNR & optical-flow EMC,
    and writes render_manifest.json.

  Stage 2 (`--stage score`):
    Loads `q-future/one-align` (`trust_remote_code=True`, fp16, device_map="auto") in 1fps
    video mode matching WorldMark's official evaluation, scores all GT and candidate MP4s,
    computes normalized PQ + camera/slice breakdown, and prints the summary leaderboard table.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitscenes import spec
from kitscenes.depth_reproj_world_model import Depth3DFullFoVWorldModel
from kitscenes.emc_eval import compute_video_optical_flow_emc

SLICE_ORDER: Tuple[str, ...] = (
    "0 specifically selected",
    "1 heavy rain",
    "2 construction zone",
    "3 overtake or lane change",
    "4 intersection",
    "5 nighttime",
    "6 snow & wintry mix",
)

CAMERA_WEIGHTS: Dict[str, float] = {
    "front": 0.50,
    "front_left": 0.10,
    "front_right": 0.10,
    "rear": 0.10,
    "rear_left": 0.10,
    "rear_right": 0.10,
}

VARIANT_SPECS: Dict[str, Dict[str, object]] = {
    "analytical_r2": {"use_depth": False, "use_lama": False, "render_scale": 2, "crf_sharpen": 0.0, "min_lama_hole_pixels": 0},
    "analytical_sharp": {"use_depth": False, "use_lama": False, "render_scale": 2, "crf_sharpen": 0.16, "min_lama_hole_pixels": 0},
    "analytical_sharp24": {"use_depth": False, "use_lama": False, "render_scale": 2, "crf_sharpen": 0.24, "min_lama_hole_pixels": 0},
    "analytical_sel_lama_sharp": {"use_depth": False, "use_lama": True, "render_scale": 2, "crf_sharpen": 0.16, "min_lama_hole_pixels": 600},
    "analytical_lama_r2": {"use_depth": False, "use_lama": True, "render_scale": 2, "crf_sharpen": 0.0, "min_lama_hole_pixels": 0},
    "analytical_lama_sharp": {"use_depth": False, "use_lama": True, "render_scale": 2, "crf_sharpen": 0.16, "min_lama_hole_pixels": 0},
    "depth_r2": {"use_depth": True, "use_lama": False, "render_scale": 2, "crf_sharpen": 0.0, "min_lama_hole_pixels": 0},
    "depth_lama_r2": {"use_depth": True, "use_lama": True, "render_scale": 2, "crf_sharpen": 0.0, "min_lama_hole_pixels": 0},
    "depth_lama_sharp": {"use_depth": True, "use_lama": True, "render_scale": 2, "crf_sharpen": 0.14, "min_lama_hole_pixels": 0},
    # Stage B incremental candidates (A: Baseline .. E: Selective refinement)
    "cand_a_baseline": {
        "use_depth": False,
        "use_lama": False,
        "render_scale": 2,
        "crf_sharpen": 0.24,
        "min_lama_hole_pixels": 0,
        "adaptive_filter": False,
        "photometric_refine": False,
    },
    "cand_b_scale1": {
        "use_depth": False,
        "use_lama": False,
        "render_scale": 1,
        "crf_sharpen": 0.24,
        "min_lama_hole_pixels": 0,
        "adaptive_filter": False,
        "photometric_refine": False,
    },
    "cand_c_adaptive": {
        "use_depth": False,
        "use_lama": False,
        "render_scale": 1,
        "crf_sharpen": 0.24,
        "min_lama_hole_pixels": 0,
        "adaptive_filter": True,
        "photometric_refine": False,
    },
    "cand_d_photometric": {
        "use_depth": False,
        "use_lama": False,
        "render_scale": 1,
        "crf_sharpen": 0.24,
        "min_lama_hole_pixels": 0,
        "adaptive_filter": True,
        "photometric_refine": True,
    },
    "cand_e_selective": {
        "use_depth": False,
        "use_lama": True,
        "render_scale": 1,
        "crf_sharpen": 0.24,
        "min_lama_hole_pixels": 1200,
        "adaptive_filter": True,
        "photometric_refine": True,
    },
}

LEVEL_NAMES: Tuple[str, ...] = ("excellent", "good", "fair", "poor", "bad")
LEVEL_WEIGHTS_RAW: Tuple[float, ...] = (5.0, 4.0, 3.0, 2.0, 1.0)
LEVEL_WEIGHTS_UNIT_100: Tuple[float, ...] = (100.0, 75.0, 50.0, 25.0, 0.0)


def extract_five_level_distribution(logits: object, reported_score: Optional[float] = None) -> Dict[str, object]:
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


def encode_rgb_h264(frames_rgb: np.ndarray, out_mp4: Path, fps: float = float(spec.FPS), crf: int = 14) -> None:
    """Encode (N, H, W, 3) uint8 RGB array cleanly into H.264 MP4."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg binary not found in PATH.")

    n_frames, height, width, _ = frames_rgb.shape
    if n_frames != spec.FRAME_COUNT or width != spec.WIDTH or height != spec.HEIGHT:
        raise ValueError(f"Expected (25, 568, 876, 3), got {frames_rgb.shape}")

    tmp_mp4 = out_mp4.with_name(f".{out_mp4.stem}.tmp{out_mp4.suffix}")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(tmp_mp4),
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        proc.stdin.write(np.ascontiguousarray(frames_rgb, dtype=np.uint8).tobytes())
        proc.stdin.close()
        err = proc.stderr.read() if proc.stderr else b""
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg exited with {ret} for {out_mp4}: {err.decode(errors='replace')}")
        os.replace(tmp_mp4, out_mp4)
    except Exception:
        tmp_mp4.unlink(missing_ok=True)
        raise


def read_video_rgb(path: Path) -> np.ndarray:
    """Decode MP4 frames as (N, H, W, 3) uint8 RGB numpy array."""
    cap = cv2.VideoCapture(str(path))
    frames: List[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames, axis=0)


def compute_psnr(gt_rgb: np.ndarray, pred_rgb: np.ndarray) -> float:
    mse = float(np.mean((gt_rgb.astype(np.float32) - pred_rgb.astype(np.float32)) ** 2))
    return float(20.0 * np.log10(255.0 / max(np.sqrt(mse), 1e-6)))


def build_depth_fn(
    model_id: str,
    device: str = "cuda:0",
) -> Callable[[Dict[str, np.ndarray]], Dict[str, Tuple[np.ndarray, bool]]]:
    """Load Depth-Anything-V2 metric outdoor model and wrap batch camera inference."""
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    target_dev = torch.device(device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if target_dev.type == "cuda" else torch.float32
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id, torch_dtype=dtype).to(target_dev).eval()
    is_disp = "metric" not in model_id.lower()

    def _predict(imgs_rgb: Dict[str, np.ndarray]) -> Dict[str, Tuple[np.ndarray, bool]]:
        cams = list(imgs_rgb.keys())
        pil_list = [Image.fromarray(imgs_rgb[c]) for c in cams]
        inputs = processor(images=pil_list, return_tensors="pt")
        inputs = {k: v.to(device=target_dev, dtype=dtype if v.is_floating_point() else None) for k, v in inputs.items()}
        with torch.inference_mode():
            pred = model(**inputs).predicted_depth
        out: Dict[str, Tuple[np.ndarray, bool]] = {}
        for idx, cam in enumerate(cams):
            h, w = imgs_rgb[cam].shape[:2]
            d_t = torch.nn.functional.interpolate(
                pred[idx : idx + 1].unsqueeze(0).float(),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )[0, 0]
            out[cam] = (d_t.detach().cpu().numpy().astype(np.float32), is_disp)
        return out

    return _predict


def build_lama_fn(device: str = "cuda:0") -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Wrap SimpleLama while strictly cropping mod-8 padding and blending only holes."""
    from PIL import Image
    from simple_lama_inpainting import SimpleLama

    target_dev = torch.device(device if torch.cuda.is_available() else "cpu")
    lama = SimpleLama(device=target_dev)

    def _inpaint(img_rgb_u8: np.ndarray, hole_mask: np.ndarray) -> np.ndarray:
        h, w = img_rgb_u8.shape[:2]
        mask_u8 = (hole_mask.astype(np.uint8)) * 255
        # Small dilation heals jagged homography boundary edges cleanly inside LaMa
        dil = cv2.dilate(mask_u8, np.ones((5, 5), np.uint8))
        res_pil = lama(Image.fromarray(img_rgb_u8), Image.fromarray(dil))
        res_np = np.asarray(res_pil, dtype=np.uint8)[:h, :w]
        if res_np.shape[:2] != (h, w):
            res_np = cv2.resize(res_np, (w, h), interpolation=cv2.INTER_LINEAR)
        alpha = (cv2.GaussianBlur(dil.astype(np.float32) / 255.0, (5, 5), 1.2))[..., None]
        alpha = np.where(mask_u8[..., None] > 0, 1.0, alpha)
        out = alpha * res_np.astype(np.float32) + (1.0 - alpha) * img_rgb_u8.astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

    return _inpaint


def load_val_records(
    val_traj_path: Path,
    val_calib_root: Path,
    limit: int = 0,
    scenarios: str = "",
    stratified: bool = True,
) -> List[Dict[str, object]]:
    """Match validation trajectory records with extracted t=0/past/GT calibration folders."""
    t0_dir = val_calib_root / "val_t0_native"
    fut_dir = val_calib_root / "val_future_videos"
    want = {s.strip().zfill(7) for s in scenarios.split(",") if s.strip()} if scenarios else set()
    records: List[Dict[str, object]] = []
    with open(val_traj_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            sid = str(r["scenario_id"]).zfill(7)
            if want and sid not in want:
                continue
            if not (t0_dir / sid / "front.jpg").exists():
                continue
            if not (fut_dir / sid / "front.mp4").exists():
                continue
            records.append(r)
    records.sort(key=lambda x: str(x["scenario_id"]))
    if limit > 0 and limit < len(records):
        if stratified:
            buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
            for rec in records:
                buckets[str(rec.get("scenario_type", "unknown"))].append(rec)
            ordered_slices = list(SLICE_ORDER) + sorted(s for s in buckets if s not in SLICE_ORDER)
            picked: List[Dict[str, object]] = []
            idx_by_slice = {s: 0 for s in ordered_slices}
            while len(picked) < limit:
                progressed = False
                for st in ordered_slices:
                    bucket = buckets.get(st, [])
                    pos = idx_by_slice[st]
                    if pos < len(bucket):
                        picked.append(bucket[pos])
                        idx_by_slice[st] = pos + 1
                        progressed = True
                        if len(picked) >= limit:
                            break
                if not progressed:
                    break
            records = sorted(picked, key=lambda x: str(x["scenario_id"]))
        else:
            records = records[:limit]
    return records


def ensure_val_calib_extracted(val_dir: Optional[Path], val_tar: Optional[Path], work_dir: Path) -> Path:
    if val_dir is not None and (val_dir / "val_t0_native").exists():
        return val_dir
    if val_tar is None or not val_tar.exists():
        raise FileNotFoundError(f"Neither --val-dir ({val_dir}) nor --val-tar ({val_tar}) is available.")
    extract_parent = work_dir / "_unpacked"
    target = extract_parent / "val_calib"
    if (target / "val_t0_native").exists():
        return target
    extract_parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] Extracting {val_tar} -> {extract_parent} ...", flush=True)
    with tarfile.open(val_tar, "r") as tf:
        tf.extractall(extract_parent)
    return target


def run_render_stage(args: argparse.Namespace) -> int:
    val_root = ensure_val_calib_extracted(args.val_dir, args.val_tar, args.out_dir)
    records = load_val_records(
        args.val_trajectories,
        val_root,
        limit=args.limit,
        scenarios=args.scenarios,
        stratified=not args.no_stratify,
    )
    if not records:
        raise RuntimeError(f"No matching validation calibration scenarios found under {val_root}")

    requested = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in requested:
        if v not in VARIANT_SPECS:
            raise ValueError(f"Unknown variant {v!r}; expected one of {list(VARIANT_SPECS)}")

    need_depth = any(bool(VARIANT_SPECS[v]["use_depth"]) for v in requested)
    need_lama = any(bool(VARIANT_SPECS[v]["use_lama"]) for v in requested)

    depth_fn = build_depth_fn(args.depth_model, device=args.device) if need_depth else None
    lama_fn = build_lama_fn(device=args.device) if need_lama else None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    git_sha = None
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True).strip()
    except Exception:
        pass
    manifest: Dict[str, object] = {
        "val_root": str(val_root),
        "git_commit": git_sha,
        "crf": int(args.crf),
        "depth_model": args.depth_model if need_depth else None,
        "scenarios": [],
        "variants": requested,
        "variant_configs": {
            v: {**VARIANT_SPECS[v], "crf": int(args.crf)} for v in requested
        },
        "entries": [],
    }

    t0_base = val_root / "val_t0_native"
    past_base = val_root / "val_past"
    fut_base = val_root / "val_future_videos"

    for rec in records:
        sid = str(rec["scenario_id"]).zfill(7)
        manifest["scenarios"].append(
            {
                "scenario_id": sid,
                "scenario_type": rec.get("scenario_type", "unknown"),
                "driving_instruction": rec.get("driving_instruction", ""),
            }
        )
        for cam in spec.CAMERAS:
            gt_mp4 = fut_base / sid / f"{cam}.mp4"
            manifest["entries"].append(
                {
                    "key": f"gt__{sid}__{cam}",
                    "variant": "gt",
                    "scenario_id": sid,
                    "scenario_type": rec.get("scenario_type", "unknown"),
                    "camera": cam,
                    "mp4_path": str(gt_mp4),
                    "psnr": 100.0,
                    "emc_local": None,
                }
            )

    for vname in requested:
        vcfg = VARIANT_SPECS[vname]
        model = Depth3DFullFoVWorldModel(
            device=args.device,
            render_scale=int(vcfg["render_scale"]),
            depth_fn=depth_fn if vcfg["use_depth"] else None,
            inpaint_fn=lama_fn if vcfg["use_lama"] else None,
            crf_sharpen=float(vcfg["crf_sharpen"]),
            min_lama_hole_pixels=int(vcfg.get("min_lama_hole_pixels", 0)),
            adaptive_filter=bool(vcfg.get("adaptive_filter", False)),
            photometric_refine=bool(vcfg.get("photometric_refine", False)),
        )
        t_v = time.time()
        variant_bytes = 0
        variant_videos = 0
        print(f"\n=== Rendering variant `{vname}` across {len(records)} validation scenarios ===", flush=True)
        for idx, rec in enumerate(records, start=1):
            sid = str(rec["scenario_id"]).zfill(7)
            cond = {c: t0_base / sid / f"{c}.jpg" for c in spec.CAMERAS}
            past_imgs: Dict[Tuple[str, str], Path] = {}
            for cam in spec.CAMERAS:
                for tag in ("tm02", "tm04", "tm10", "tm20", "tm30", "tm40"):
                    pp = past_base / sid / f"{cam}_{tag}.jpg"
                    if pp.exists():
                        past_imgs[(cam, tag)] = pp

            t_s = time.time()
            gen = model.generate_scenario(
                sid,
                rec["expert_like"],
                cond,
                past_images=past_imgs,
                past_trajectory=rec.get("past"),
            )
            dt = time.time() - t_s

            psnrs: List[float] = []
            scenario_paths: Dict[str, Path] = {}
            scen_psnr_by_cam: Dict[str, float] = {}
            for cam in spec.CAMERAS:
                out_mp4 = args.out_dir / "videos" / vname / sid / f"{cam}.mp4"
                encode_rgb_h264(gen[cam], out_mp4, crf=args.crf)
                variant_bytes += out_mp4.stat().st_size
                variant_videos += 1
                scenario_paths[cam] = out_mp4
                gt_rgb = read_video_rgb(fut_base / sid / f"{cam}.mp4")
                psnr_val = compute_psnr(gt_rgb, gen[cam])
                psnrs.append(psnr_val)
                scen_psnr_by_cam[cam] = psnr_val

            emc_val: Optional[float] = None
            try:
                emc_res = compute_video_optical_flow_emc(scenario_paths, rec["expert_like"])
                emc_val = float(emc_res.get("emc", 0.0))
            except Exception:
                emc_val = None

            for cam in spec.CAMERAS:
                manifest["entries"].append(
                    {
                        "key": f"{vname}__{sid}__{cam}",
                        "variant": vname,
                        "scenario_id": sid,
                        "scenario_type": rec.get("scenario_type", "unknown"),
                        "camera": cam,
                        "mp4_path": str(scenario_paths[cam]),
                        "psnr": round(scen_psnr_by_cam[cam], 3),
                        "emc_local": (round(emc_val, 4) if (cam == "front" and emc_val is not None) else None),
                    }
                )
            print(
                f"  [{idx:02d}/{len(records)}] {vname} {sid} ({rec.get('scenario_type','')[:18]:18s}) "
                f"| mean PSNR={np.mean(psnrs):.2f} dB (front={psnrs[0]:.2f}, emc={emc_val}) | {dt:.2f}s",
                flush=True,
            )
        avg_kb = (variant_bytes / max(variant_videos, 1)) / 1024.0
        proj_gib = ((variant_bytes / max(variant_videos, 1)) * 2400.0) / (1024.0 ** 3)
        print(
            f"[+] Variant `{vname}` finished in {time.time() - t_v:.1f}s "
            f"| avg={avg_kb:.1f} KiB/vid (2400-video proj={proj_gib:.2f} GiB)",
            flush=True,
        )

    manifest_path = args.out_dir / "render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[+] Wrote {manifest_path} ({len(manifest['entries'])} video entries)", flush=True)
    return 0


def frames_1fps_pil(mp4_path: Path) -> List[object]:
    """Match WorldMark's exact `frames_1fps`: 1 frame/sec (`[0, 5, 10, 15, 20]` at 5fps)."""
    from PIL import Image

    try:
        from decord import VideoReader, cpu

        vr = VideoReader(str(mp4_path), ctx=cpu(0))
        fps = vr.get_avg_fps() or float(spec.FPS)
        step = max(1, int(round(fps)))
        return [Image.fromarray(vr[i].asnumpy()) for i in range(0, len(vr), step)]
    except Exception:
        cap = cv2.VideoCapture(str(mp4_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or float(spec.FPS)
        step = max(1, int(round(fps)))
        sampled: List[object] = []
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % step == 0:
                sampled.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            idx += 1
        cap.release()
        return sampled


def _attach_onealign_weight_tensor(md) -> None:
    """Attach OneAlign MOS weights onto a real device (never `meta`) and sync to logits."""
    lm_w = getattr(getattr(md, "lm_head", None), "weight", None)
    lm_dev = getattr(lm_w, "device", None)
    if lm_dev is None or lm_dev.type == "meta":
        lm_dev = getattr(md, "device", torch.device("cpu"))
    if getattr(lm_dev, "type", "cpu") == "meta":
        lm_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    md.weight_tensor = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], dtype=torch.float16, device=lm_dev)

    def _sync_weight_dev(_mod, _inp, out):
        logits = out["logits"] if isinstance(out, dict) else getattr(out, "logits", None)
        if logits is not None and md.weight_tensor.device != logits.device:
            md.weight_tensor = md.weight_tensor.to(logits.device)

    md.register_forward_hook(_sync_weight_dev)


def load_one_align_model():
    """Load `q-future/one-align` (pure FP16 on >=20GiB VRAM / multi-GPU; 8-bit hybrid fallback on single 15GiB T4)."""
    from transformers import AutoModelForCausalLM

    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available() and torch.cuda.device_count() == 1:
        vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram_gib < 18.0:
            try:
                from transformers import BitsAndBytesConfig

                bnb_cfg = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_skip_modules=["vision_model", "visual_abstractor", "lm_head"],
                )
                print("[*] Single 15-16GiB GPU detected -> loading OneAlign with 8-bit LLM + FP16 vision/head...", flush=True)
                md = AutoModelForCausalLM.from_pretrained(
                    "q-future/one-align",
                    quantization_config=bnb_cfg,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
                _attach_onealign_weight_tensor(md)
                return md
            except Exception as exc:  # noqa: BLE001
                print(f"[!] 8-bit load failed ({exc}), falling back to FP16 offload...", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
                Path("/tmp/onealign_offload").mkdir(parents=True, exist_ok=True)
                kwargs["max_memory"] = {0: "13500MiB", "cpu": "11GiB"}
                kwargs["offload_folder"] = "/tmp/onealign_offload"
    try:
        md = AutoModelForCausalLM.from_pretrained("q-future/one-align", attn_implementation="eager", **kwargs)
    except TypeError:
        md = AutoModelForCausalLM.from_pretrained("q-future/one-align", **kwargs)
    _attach_onealign_weight_tensor(md)
    return md


def _norm_pq_pair(q_pred: float, q_gt: float) -> Tuple[float, float, float]:
    raw_100 = 100.0 * (q_pred - 1.0) / 4.0
    rel_unit = min(100.0, 100.0 * max(q_pred - 1.0, 0.0) / max(q_gt - 1.0, 1e-4))
    rel_raw = min(100.0, 100.0 * q_pred / max(q_gt, 1e-4))
    return raw_100, rel_unit, rel_raw


def _score_onealign_video(md, frames: List[object]) -> Dict[str, object]:
    """Run upstream OneAlign video scorer and capture 5-level distribution when exposed."""
    with torch.inference_mode():
        try:
            res = md.score([frames], task_="quality", input_="video", return_dict=True)
        except TypeError:
            q_val = float(md.score([frames], task_="quality", input_="video"))
            return {"q_raw": q_val, "pq_worldmark_100": round(100.0 * (q_val - 1.0) / 4.0, 4)}

    if isinstance(res, dict) and "logits" in res and "scores" in res:
        q_val = float(torch.as_tensor(res["scores"]).reshape(-1)[0].item())
        return extract_five_level_distribution(res["logits"], reported_score=q_val)
    q_val = float(torch.as_tensor(res).reshape(-1)[0].item())
    return {"q_raw": q_val, "pq_worldmark_100": round(100.0 * (q_val - 1.0) / 4.0, 4)}


def _compute_paired_deltas(
    cand_scores: Dict[str, float],
    base_scores: Dict[str, float],
    scenarios: List[Dict[str, object]],
) -> Dict[str, object]:
    """Compare candidate vs baseline scenario-level PQ preserving paired slice breakdown."""
    diffs: List[float] = []
    by_slice: Dict[str, List[float]] = defaultdict(list)
    improved = regressed = tied = 0
    for sc in scenarios:
        sid = str(sc["scenario_id"])
        stype = str(sc["scenario_type"])
        if sid not in cand_scores or sid not in base_scores:
            continue
        delta = cand_scores[sid] - base_scores[sid]
        diffs.append(delta)
        by_slice[stype].append(delta)
        if delta > 1e-4:
            improved += 1
        elif delta < -1e-4:
            regressed += 1
        else:
            tied += 1
    if not diffs:
        return {}
    return {
        "n_scenarios": len(diffs),
        "mean_delta": round(float(np.mean(diffs)), 4),
        "std_delta": round(float(np.std(diffs, ddof=1) if len(diffs) > 1 else 0.0), 4),
        "min_delta": round(float(np.min(diffs)), 4),
        "max_delta": round(float(np.max(diffs)), 4),
        "scenarios_improved": improved,
        "scenarios_regressed": regressed,
        "scenarios_tied": tied,
        "slice_mean_deltas": {
            s: round(float(np.mean(by_slice[s])), 4) for s in SLICE_ORDER if by_slice.get(s)
        },
    }


def run_score_stage(args: argparse.Namespace) -> int:
    manifest_path = args.out_dir / "render_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing render_manifest.json in {args.out_dir}")
    manifest = json.loads(manifest_path.read_text())
    entries: List[Dict[str, object]] = manifest["entries"]

    scores_path = args.out_dir / "qalign_scores.json"
    cached: Dict[str, Dict[str, object]] = json.loads(scores_path.read_text()) if scores_path.exists() else {}

    pending = [e for e in entries if e["key"] not in cached]
    if pending:
        print(f"[*] Loading q-future/one-align to score {len(pending)} videos ({len(cached)} cached)...", flush=True)
        md = load_one_align_model()
        t_start = time.time()
        for idx, entry in enumerate(pending, start=1):
            fr = frames_1fps_pil(Path(str(entry["mp4_path"])))
            scored = _score_onealign_video(md, fr)
            cached[str(entry["key"])] = {**entry, **scored, "n_frames_1fps": len(fr)}
            if idx % 20 == 0 or idx == len(pending):
                scores_path.write_text(json.dumps(cached, indent=2))
                rate = idx / max(time.time() - t_start, 1e-3)
                print(
                    f"  [Q-Align {idx:03d}/{len(pending)}] {entry['key']} -> q={scored['q_raw']:.4f} "
                    f"({rate:.2f} vid/s)",
                    flush=True,
                )
    else:
        print(f"[+] All {len(cached)} Q-Align scores loaded from cache.", flush=True)

    # Compute normalized PQ per variant, scenario, slice, and camera
    variants = ["gt"] + list(manifest["variants"])
    scenarios = manifest["scenarios"]
    variant_cfgs = manifest.get("variant_configs", {})

    print("\n================ Q-ALIGN OFFICIAL CALIBRATION SUMMARY ================", flush=True)
    header = (
        f"{'Variant':<20s} | {'PQ_norm1(1-5)':<13s} | {'PQ_ratio':<8s} | {'Q_raw':<6s} | "
        f"{'Front':<6s} | {'Surr5':<6s} | {'PSNR':<6s} | {'EMC':<6s}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)

    summary_out: Dict[str, object] = {}
    scen_pq1_by_variant: Dict[str, Dict[str, float]] = {}
    for v in variants:
        scen_pqs_1: List[float] = []
        scen_pqs_r: List[float] = []
        raw_qs: List[float] = []
        raw_100s: List[float] = []
        front_pqs: List[float] = []
        side_pqs: List[float] = []
        psnrs: List[float] = []
        emcs: List[float] = []
        by_slice: Dict[str, List[float]] = defaultdict(list)
        video_rows: List[Dict[str, object]] = []
        scen_pq1_map: Dict[str, float] = {}

        for sc in scenarios:
            sid = str(sc["scenario_id"])
            stype = str(sc["scenario_type"])
            w_pq1 = 0.0
            w_pqr = 0.0
            for cam, w in CAMERA_WEIGHTS.items():
                gt_rec = cached[f"gt__{sid}__{cam}"]
                pr_rec = cached[f"{v}__{sid}__{cam}"]
                q_gt = float(gt_rec["q_raw"])
                q_pr = float(pr_rec["q_raw"])
                raw100, p1, pr = _norm_pq_pair(q_pr, q_gt)
                w_pq1 += w * p1
                w_pqr += w * pr
                raw_qs.append(q_pr)
                raw_100s.append(raw100)
                video_rows.append(
                    {
                        "scenario_id": sid,
                        "scenario_type": stype,
                        "camera": cam,
                        "pq_worldmark_100": round(raw100, 3),
                        "pq_norm_unit": round(p1, 3),
                        "pq_norm_ratio": round(pr, 3),
                        "q_raw": round(q_pr, 4),
                        "q_gt": round(q_gt, 4),
                        "level_probs": pr_rec.get("level_probs"),
                        "psnr": pr_rec.get("psnr"),
                    }
                )
                if cam == "front":
                    front_pqs.append(p1)
                else:
                    side_pqs.append(p1)
                if pr_rec.get("psnr") is not None and v != "gt":
                    psnrs.append(float(pr_rec["psnr"]))
                if pr_rec.get("emc_local") is not None:
                    emcs.append(float(pr_rec["emc_local"]))
            scen_pqs_1.append(w_pq1)
            scen_pqs_r.append(w_pqr)
            scen_pq1_map[sid] = w_pq1
            by_slice[stype].append(w_pq1)

        scen_pq1_by_variant[v] = scen_pq1_map
        m_pq1 = float(np.mean(scen_pqs_1))
        m_pqr = float(np.mean(scen_pqs_r))
        m_q = float(np.mean(raw_qs))
        m_wm = float(np.mean(raw_100s))
        m_fr = float(np.mean(front_pqs))
        m_sd = float(np.mean(side_pqs))
        m_psnr = float(np.mean(psnrs)) if psnrs else float("nan")
        m_emc = float(np.mean(emcs)) if emcs else float("nan")
        print(
            f"{v:<20s} | {m_pq1:13.2f} | {m_pqr:8.2f} | {m_q:6.3f} | "
            f"{m_fr:6.2f} | {m_sd:6.2f} | {m_psnr:6.2f} | {m_emc:6.3f}",
            flush=True,
        )
        slice_means = {s: float(np.mean(by_slice[s])) for s in SLICE_ORDER if by_slice.get(s)}
        worst_25 = sorted(video_rows, key=lambda r: (r["pq_norm_unit"], r["pq_worldmark_100"]))[:25]
        summary_out[v] = {
            "config": variant_cfgs.get(v) or (
                {**VARIANT_SPECS[v], "crf": manifest.get("crf")} if v in VARIANT_SPECS else None
            ),
            "pq_norm_unit": round(m_pq1, 3),
            "pq_norm_ratio": round(m_pqr, 3),
            "pq_worldmark_mean": round(m_wm, 3),
            "q_raw_mean": round(m_q, 4),
            "front_pq": round(m_fr, 3),
            "surround5_pq": round(m_sd, 3),
            "mean_psnr": None if np.isnan(m_psnr) else round(m_psnr, 3),
            "mean_emc_local": None if np.isnan(m_emc) else round(m_emc, 4),
            "slices": {k: round(val, 2) for k, val in slice_means.items()},
            "worst_25_videos": worst_25,
        }

    baseline_var = args.baseline_variant if args.baseline_variant in scen_pq1_by_variant else (
        manifest["variants"][0] if manifest["variants"] else None
    )
    if baseline_var is not None:
        base_map = scen_pq1_by_variant[baseline_var]
        for v in manifest["variants"]:
            if v == baseline_var:
                continue
            summary_out[v]["paired_vs_baseline"] = {
                "baseline_variant": baseline_var,
                **_compute_paired_deltas(scen_pq1_by_variant[v], base_map, scenarios),
            }

    print("\n--- Per-Slice Breakdown (PQ_norm1) ---", flush=True)
    for v in manifest["variants"]:
        sl = summary_out[v]["slices"]
        line = " | ".join(f"{k[:14]}={val:.1f}" for k, val in sl.items())
        delta_txt = ""
        paired = summary_out[v].get("paired_vs_baseline")
        if paired:
            delta_txt = f" | dBase={paired['mean_delta']:+.2f} (+{paired['scenarios_improved']}/-{paired['scenarios_regressed']})"
        print(f"  {v:<20s}: {line}{delta_txt}", flush=True)

    summary_path = args.out_dir / "calibration_summary.json"
    summary_path.write_text(json.dumps(summary_out, indent=2))
    print(f"\n[+] Saved calibration summary (incl. worst 25 videos) to {summary_path}", flush=True)
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render & evaluate Depth3DFullFoVWorldModel with official Q-Align.")
    parser.add_argument("--stage", choices=["render", "score"], required=True)
    parser.add_argument("--val-dir", type=Path, default=None, help="Directory holding val_t0_native/, val_past/, val_future_videos/")
    parser.add_argument("--val-tar", type=Path, default=None, help="Path to val_calib.tar if not extracted")
    parser.add_argument("--val-trajectories", type=Path, default=PROJECT_ROOT / "data" / "val_trajectories.jsonl")
    parser.add_argument("--out-dir", type=Path, default=Path("/kaggle/working/qalign_calib"))
    parser.add_argument("--variants", type=str, default="cand_a_baseline,cand_b_scale1,cand_c_adaptive,cand_d_photometric,cand_e_selective")
    parser.add_argument("--baseline-variant", type=str, default="cand_a_baseline")
    parser.add_argument("--depth-model", type=str, default="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--crf", type=int, default=14)
    parser.add_argument("--scenarios", type=str, default="", help="Optional comma-separated validation scenario IDs")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of val scenarios (0 = all matched)")
    parser.add_argument("--no-stratify", action="store_true", help="Disable stratified slice sampling when --limit > 0")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.stage == "render":
        return run_render_stage(args)
    return run_score_stage(args)


if __name__ == "__main__":
    raise SystemExit(main())
