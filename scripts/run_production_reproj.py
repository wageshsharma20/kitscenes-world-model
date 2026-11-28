#!/usr/bin/env python3
"""Dual-GPU production renderer for all 400 KITScenes test scenarios (2,400 MP4 videos).

Renders calibrated full-FoV 3D multi-view + multi-frame reprojection videos at native
3488x2272 -> 876x568 @ 5fps H.264, supporting parallel multi-GPU workers (`--gpus 0,1`),
monocular Depth-Anything-V2 foreground elevation, LaMa hole completion, and verified
ZIP archive assembly (`<= 3 GiB`) via `kitscenes.archive.build_archive`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitscenes import archive, spec
from kitscenes.depth_reproj_world_model import Depth3DFullFoVWorldModel
from kitscenes.esrgan import build_realesrgan_blend_fn, encode_rgb_h264_preset
from kitscenes.video_engine import CameraRig
from scripts.eval_world_model_qalign import build_depth_fn, build_lama_fn, encode_rgb_h264


def _sha256_file(path: Optional[Path]) -> Optional[str]:
    """Compute streaming SHA-256 hex digest of a file, or None if missing."""
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_git_provenance(repo_root: Path) -> Dict[str, Any]:
    """Record current git commit SHA and working-tree dirty state cleanly."""
    info: Dict[str, Any] = {
        "commit": None,
        "short_commit": None,
        "dirty": None,
        "dirty_files": [],
    }
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if head.returncode == 0:
            commit = head.stdout.strip()
            info["commit"] = commit
            info["short_commit"] = commit[:7]
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if status.returncode == 0:
            entries = [line.rstrip() for line in status.stdout.splitlines() if line.strip()]
            info["dirty"] = bool(entries)
            info["dirty_files"] = entries
    except OSError:
        pass
    return info


def _collect_camera_calibration_provenance(
    width: int = spec.WIDTH,
    height: int = spec.HEIGHT,
    calibration_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build deterministic hashes covering both calibration JSON and active CameraRig matrices."""
    cfg_file = calibration_path or (PROJECT_ROOT / "configs" / "kitscenes_camera_calibration.json")
    rig = CameraRig(
        width=width,
        height=height,
        is_provisional=False,
        calibration_path=cfg_file if cfg_file.exists() else None,
    )
    canonical = {
        "width": int(rig.width),
        "height": int(rig.height),
        "is_provisional": bool(rig.is_provisional),
        "is_calibrated": bool(getattr(rig, "is_calibrated", not rig.is_provisional)),
        "calibration_source": rig.calibration_source,
        "nominal_ground_z_ego": float(rig.nominal_ground_z_ego),
        "cameras": {
            cam: {
                "yaw_deg": float(rig.cameras[cam].yaw_deg),
                "position_m": [float(v) for v in rig.cameras[cam].position_m],
                "K_source": (
                    rig.cameras[cam].k_source_matrix.tolist()
                    if rig.cameras[cam].k_source_matrix is not None
                    else None
                ),
                "K": rig.cameras[cam].intrinsic_matrix.tolist(),
                "R": rig.cameras[cam].rotation_matrix_body_to_cam.tolist(),
                "t": rig.cameras[cam].translation_vector.tolist(),
            }
            for cam in spec.CAMERAS
        },
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "config_file": str(cfg_file) if cfg_file.exists() else None,
        "config_file_sha256": _sha256_file(cfg_file),
        "rig_params_sha256": hashlib.sha256(encoded).hexdigest(),
        "calibration_source": rig.calibration_source,
        "nominal_ground_z_ego": float(rig.nominal_ground_z_ego),
    }


def _scenario_uses_depth(cfg: Dict[str, Any], scenario_type: str) -> bool:
    return bool(cfg["use_depth"]) and (scenario_type not in set(cfg.get("no_depth_slices", ())))


def _camera_hole_stats(last_hole_fraction: Optional[Dict[str, List[float]]], cam: str) -> Dict[str, float]:
    vals = (last_hole_fraction or {}).get(cam) or []
    if not vals:
        return {}
    return {
        "hole_fraction_mean": round(float(sum(vals) / len(vals)), 6),
        "hole_fraction_max": round(float(max(vals)), 6),
    }


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Persist manifest atomically so readers never observe half-written JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def load_test_inputs(
    traj_path: Path,
    history_path: Path,
) -> List[Tuple[str, List[List[float]], Optional[List[List[float]]], str]]:
    """Merge future trajectory plan + past odometry + scenario_type for all 400 test scenarios."""
    past_by_sid: Dict[str, List[List[float]]] = {}
    type_by_sid: Dict[str, str] = {}
    if history_path.exists():
        with open(history_path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                sid = str(r["scenario_id"]).zfill(7)
                past = r.get("past_trajectory") or r.get("trajectory", {}).get("past")
                if past:
                    past_by_sid[sid] = past
                type_by_sid[sid] = str(r.get("scenario_type", ""))

    items: List[Tuple[str, List[List[float]], Optional[List[List[float]]], str]] = []
    with open(traj_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            sid = str(r["scenario_id"]).zfill(7)
            fut = r.get("future_trajectory") or r.get("trajectory")
            if not fut or len(fut) != spec.WAYPOINT_COUNT:
                raise ValueError(f"Expected 25 waypoints for {sid}, got {len(fut or [])}")
            items.append((sid, fut, past_by_sid.get(sid), type_by_sid.get(sid, "")))
    items.sort(key=lambda x: x[0])
    return items


def unpack_if_needed(src_dir: Optional[Path], tar_path: Optional[Path], dst_parent: Path, expected_sub: str) -> Path:
    if src_dir is not None and src_dir.exists():
        return src_dir
    if tar_path is None or not tar_path.exists():
        raise FileNotFoundError(f"Missing directory {src_dir} and archive {tar_path}")
    target = dst_parent / expected_sub
    if target.exists() and any(target.iterdir()):
        return target
    dst_parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] Extracting {tar_path} -> {dst_parent} ...", flush=True)
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(dst_parent)
    if not target.exists():
        raise FileNotFoundError(f"Expected unpacked subdirectory {target} inside {tar_path}")
    return target


def _scan_camera_files(
    scen_dir: Path,
    *,
    default_status: str,
    effective_crf: Optional[int],
) -> Dict[str, Dict[str, Any]]:
    cameras: Dict[str, Dict[str, Any]] = {}
    for cam in spec.CAMERAS:
        mp4 = scen_dir / f"{cam}.mp4"
        if mp4.exists() and mp4.stat().st_size > 1024:
            cameras[cam] = {
                "status": default_status,
                "path": str(mp4),
                "size_bytes": mp4.stat().st_size,
                "effective_crf": effective_crf,
            }
        else:
            cameras[cam] = {
                "status": "missing",
                "path": str(mp4),
                "size_bytes": mp4.stat().st_size if mp4.exists() else 0,
                "effective_crf": None,
            }
    return cameras


def _gpu_worker(
    worker_id: int,
    gpu_idx: Optional[int],
    tasks: List[Tuple[str, List[List[float]], Optional[List[List[float]]], str]],
    t0_dir: Path,
    past_dir: Optional[Path],
    out_videos_dir: Path,
    cfg: Dict[str, Any],
) -> Tuple[int, float, Dict[str, Dict[str, Any]]]:
    if gpu_idx is not None and torch.cuda.is_available():
        device = f"cuda:{gpu_idx}"
        torch.cuda.set_device(gpu_idx)
    else:
        device = "cpu"

    depth_fn = build_depth_fn(str(cfg["depth_model"]), device=device) if cfg["use_depth"] else None
    lama_fn = build_lama_fn(device=device) if cfg["use_lama"] else None
    esr_day = float(cfg.get("esrgan_alpha", 0.0))
    esr_night = float(cfg.get("esrgan_night_wdn_alpha", 0.0))
    esrgan_fn = (
        build_realesrgan_blend_fn(
            Path(str(cfg.get("esrgan_weights_dir") or (PROJECT_ROOT / "weights" / "esrgan"))),
            device=device,
            blend_alpha=esr_day,
            min_luma=float(cfg.get("esrgan_min_luma", 35.0)),
            night_blend_alpha=esr_night,
        )
        if (esr_day > 0.0 or esr_night > 0.0)
        else None
    )
    model_kwargs = {
        "device": device,
        "render_scale": int(cfg["render_scale"]),
        "inpaint_fn": lama_fn,
        "crf_sharpen": float(cfg["crf_sharpen"]),
        "min_lama_hole_pixels": int(cfg.get("min_lama_hole_pixels", 0)),
        "min_lama_luma": float(cfg.get("min_lama_luma", 0.0)),
        "ego_static_min_luma": float(cfg.get("ego_static_min_luma", 0.0)),
        "adaptive_filter": bool(cfg.get("adaptive_filter", False)),
        "photometric_refine": bool(cfg.get("photometric_refine", False)),
        "photometric_strength": float(cfg.get("photometric_strength", 1.0)),
        "saturation_strength": float(cfg.get("saturation_strength", 0.90)),
        "equalize_strength": float(cfg.get("equalize_strength", 0.20)),
    }
    model_depth = Depth3DFullFoVWorldModel(depth_fn=depth_fn, **model_kwargs)
    no_depth_slices = set(cfg.get("no_depth_slices", ()))
    model_analytical = (
        Depth3DFullFoVWorldModel(depth_fn=None, **model_kwargs)
        if (no_depth_slices and depth_fn is not None)
        else model_depth
    )

    t_start = time.time()
    rendered = 0
    scenario_statuses: Dict[str, Dict[str, Any]] = {}
    for idx, (sid, fut_traj, past_traj, stype) in enumerate(tasks, start=1):
        scen_dir = out_videos_dir / sid
        depth_active = _scenario_uses_depth(cfg, stype)
        if all((scen_dir / f"{c}.mp4").exists() and (scen_dir / f"{c}.mp4").stat().st_size > 1024 for c in spec.CAMERAS):
            scenario_statuses[sid] = {
                "scenario_type": stype,
                "depth_active": depth_active,
                "status": "resumed",
                "cameras": _scan_camera_files(scen_dir, default_status="resumed", effective_crf=None),
            }
            rendered += 1
            continue

        cond = {c: t0_dir / sid / f"{c}.jpg" for c in spec.CAMERAS}
        past_imgs: Dict[Tuple[str, str], Path] = {}
        if past_dir is not None:
            for cam in spec.CAMERAS:
                for tag in ("tm02", "tm04", "tm10", "tm20", "tm30", "tm40"):
                    pp = past_dir / sid / f"{cam}_{tag}.jpg"
                    if pp.exists():
                        past_imgs[(cam, tag)] = pp

        active_model = model_analytical if stype in no_depth_slices else model_depth
        gen = active_model.generate_scenario(
            sid,
            fut_traj,
            cond,
            past_images=past_imgs if past_imgs else None,
            past_trajectory=past_traj,
        )
        scen_dir.mkdir(parents=True, exist_ok=True)
        x264_preset = str(cfg.get("x264_preset", "medium"))
        front_crf = cfg.get("front_crf")
        for cam in spec.CAMERAS:
            frames_cam = esrgan_fn(gen[cam]) if esrgan_fn is not None else gen[cam]
            cam_crf = int(front_crf) if (cam == "front" and front_crf is not None) else int(cfg["crf"])
            if x264_preset == "medium":
                encode_rgb_h264(frames_cam, scen_dir / f"{cam}.mp4", crf=cam_crf)
            else:
                encode_rgb_h264_preset(frames_cam, scen_dir / f"{cam}.mp4", crf=cam_crf, preset=x264_preset)

        cam_entries = _scan_camera_files(scen_dir, default_status="rendered", effective_crf=int(cfg["crf"]))
        if front_crf is not None and "front" in cam_entries and cam_entries["front"].get("effective_crf") is not None:
            cam_entries["front"]["effective_crf"] = int(front_crf)
        hole_map = getattr(active_model, "last_hole_fraction", {})
        for cam in spec.CAMERAS:
            cam_entries[cam].update(_camera_hole_stats(hole_map, cam))

        scenario_statuses[sid] = {
            "scenario_type": stype,
            "depth_active": depth_active,
            "status": "rendered",
            "cameras": cam_entries,
        }
        rendered += 1

        if idx % 10 == 0 or idx == len(tasks):
            elapsed = time.time() - t_start
            rate = idx / max(elapsed, 1e-3)
            print(
                f"  [Worker {worker_id} ({device})] {idx:03d}/{len(tasks)} ({sid}) "
                f"| {elapsed:.1f}s | {rate:.2f} scen/s",
                flush=True,
            )

    return rendered, time.time() - t_start, scenario_statuses


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate all 2,400 production KITScenes videos cleanly.")
    parser.add_argument("--trajectories", type=Path, default=PROJECT_ROOT / "submissions" / "trajectories.jsonl")
    parser.add_argument("--history", type=Path, default=PROJECT_ROOT / "data" / "history.jsonl")
    parser.add_argument("--t0-dir", type=Path, default=None)
    parser.add_argument("--t0-tar", type=Path, default=None)
    parser.add_argument("--past-dir", type=Path, default=None)
    parser.add_argument("--past-tar", type=Path, default=None)
    parser.add_argument("--unpack-dir", type=Path, default=Path("/kaggle/working/unpacked_test"))
    parser.add_argument("--out-videos-dir", type=Path, default=Path("/kaggle/working/production_videos"))
    parser.add_argument("--out-zip", type=Path, default=Path("/kaggle/working/videos.zip"))
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Optional output JSON path for provenance manifest (defaults to <out-videos-dir>/generation_manifest.json)",
    )
    parser.add_argument("--depth-model", type=str, default="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf")
    parser.add_argument("--no-depth", action="store_true", default=False)
    parser.add_argument("--no-depth-slices", type=str, default="", help="Comma-separated scenario_types to run without depth")
    parser.add_argument("--no-lama", action="store_true", default=False)
    parser.add_argument("--min-lama-hole-pixels", type=int, default=0, help="Skip LaMa when hole pixel count is below threshold")
    parser.add_argument("--min-lama-luma", type=float, default=0.0, help="Skip LaMa when mean frame luma is below threshold")
    parser.add_argument("--ego-static-min-luma", type=float, default=0.0, help="Require min luma for empirical static ego check")
    parser.add_argument("--render-scale", type=int, default=2)
    parser.add_argument("--adaptive-filter", action="store_true", default=False, help="Enable Jacobian-aware minification mip filtering")
    parser.add_argument("--photometric-refine", action="store_true", default=False, help="Anchor exposure and saturation to t=0 frame")
    parser.add_argument("--photometric-strength", type=float, default=1.0)
    parser.add_argument("--saturation-strength", type=float, default=0.90)
    parser.add_argument("--equalize-strength", type=float, default=0.20)
    parser.add_argument("--crf-sharpen", type=float, default=0.0)
    parser.add_argument("--crf", type=int, default=14)
    parser.add_argument("--front-crf", type=int, default=None, help="Optional separate CRF for weighted front camera")
    parser.add_argument("--x264-preset", type=str, default="medium", help="libx264 preset (e.g. medium, slow)")
    parser.add_argument("--esrgan-alpha", type=float, default=0.0, help="Blend weight for daylight realesr-general-x4v3 restoration")
    parser.add_argument("--esrgan-night-wdn-alpha", type=float, default=0.0, help="Blend weight for nighttime realesr-general-wdn-x4v3 restoration")
    parser.add_argument("--esrgan-min-luma", type=float, default=35.0, help="Luma cutoff between night WDN and daylight RealESRGAN")
    parser.add_argument("--esrgan-weights-dir", type=Path, default=PROJECT_ROOT / "weights" / "esrgan")
    parser.add_argument("--max-zip-gib", type=float, default=2.95, help="Auto-transcode to --fallback-crf if packed archive exceeds this GiB limit")
    parser.add_argument("--fallback-crf", type=int, default=18, help="Fallback CRF if initial archive exceeds --max-zip-gib (0 disables)")
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated CUDA device indices (or empty for CPU)")
    parser.add_argument("--scenarios", type=str, default="", help="Optional comma-separated scenario IDs subset")
    parser.add_argument("--scenario-range", type=str, default="", help="Optional 0-based Python slice START:END across sorted scenarios")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--pack-zip", action="store_true", default=False)
    parser.add_argument("--allow-partial", action="store_true", default=False)
    return parser.parse_args(argv)


def _reencode_mp4_crf(mp4_path: Path, crf: int, preset: str = "medium") -> None:
    """Re-encode one MP4 atomically to a higher CRF when clamping final archive size."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg binary not found in PATH.")
    tmp_path = mp4_path.with_name(f".{mp4_path.stem}.crf{crf}.tmp{mp4_path.suffix}")
    try:
        subprocess.run(
            [
                ffmpeg_bin,
                "-y",
                "-v",
                "error",
                "-i",
                str(mp4_path),
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                str(tmp_path),
            ],
            check=True,
        )
        os.replace(tmp_path, mp4_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _reencode_with_optional_preset(mp4_path: Path, crf: int, preset: str) -> None:
    """Call legacy 2-arg _reencode_mp4_crf when preset is default medium."""
    if preset == "medium":
        _reencode_mp4_crf(mp4_path, crf)
    else:
        _reencode_mp4_crf(mp4_path, crf, preset=preset)


def _clamp_videos_crf_if_oversized(
    videos_dir: Path,
    max_bytes: int,
    fallback_crf: int,
    *,
    preset: str = "medium",
    preserve_front_first: bool = False,
    surround_base_crf: Optional[int] = None,
) -> bool:
    mp4s = sorted(videos_dir.rglob("*.mp4"))
    total_bytes = sum(p.stat().st_size for p in mp4s)
    if total_bytes <= max_bytes or fallback_crf <= 0:
        return False

    if preserve_front_first and surround_base_crf is not None:
        surr = [p for p in mp4s if p.stem != "front"]
        for s_crf in range(surround_base_crf + 1, max(fallback_crf, surround_base_crf + 1) + 1):
            print(
                f"[!] Total MP4 bytes ({total_bytes / (1024**3):.2f} GiB) exceeds {max_bytes / (1024**3):.2f} GiB; "
                f"stepping surround5 ({len(surr)} videos) to CRF {s_crf} (preserving front) ...",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda p, c=s_crf: _reencode_with_optional_preset(p, c, preset), surr))
            total_bytes = sum(p.stat().st_size for p in mp4s)
            print(f"[+] Surround clamp CRF {s_crf} -> {total_bytes / (1024**3):.2f} GiB", flush=True)
            if total_bytes <= max_bytes:
                return True

    print(
        f"[!] Total MP4 bytes ({total_bytes / (1024**3):.2f} GiB) exceeds {max_bytes / (1024**3):.2f} GiB; "
        f"clamping {len(mp4s)} videos to CRF {fallback_crf} ...",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda p: _reencode_with_optional_preset(p, fallback_crf, preset), mp4s))
    new_bytes = sum(p.stat().st_size for p in mp4s)
    print(f"[+] Clamped total MP4 bytes -> {new_bytes / (1024**3):.2f} GiB", flush=True)
    return True


def _build_initial_manifest(
    *,
    raw_argv: List[str],
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    items: List[Tuple[str, List[List[float]], Optional[List[List[float]]], str]],
    gpu_ids: List[int],
) -> Dict[str, Any]:
    """Assemble initial generation provenance manifest before workers execute."""
    return {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "finished_at_utc": None,
        "git": _collect_git_provenance(PROJECT_ROOT),
        "cli": {
            "argv": raw_argv,
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        },
        "render_config": {
            "render_scale": int(args.render_scale),
            "adaptive_filter": bool(cfg["adaptive_filter"]),
            "photometric_refine": bool(cfg["photometric_refine"]),
            "photometric_strength": float(cfg["photometric_strength"]),
            "saturation_strength": float(cfg["saturation_strength"]),
            "equalize_strength": float(cfg["equalize_strength"]),
            "depth_enabled": bool(cfg["use_depth"]),
            "depth_model": str(args.depth_model) if cfg["use_depth"] else None,
            "no_depth_slices": list(cfg["no_depth_slices"]),
            "lama_enabled": bool(cfg["use_lama"]),
            "min_lama_hole_pixels": int(args.min_lama_hole_pixels),
            "min_lama_luma": float(cfg.get("min_lama_luma", 0.0)),
            "ego_static_min_luma": float(cfg.get("ego_static_min_luma", 0.0)),
            "esrgan_alpha": float(cfg.get("esrgan_alpha", 0.0)),
            "esrgan_night_wdn_alpha": float(cfg.get("esrgan_night_wdn_alpha", 0.0)),
            "esrgan_min_luma": float(cfg.get("esrgan_min_luma", 35.0)),
            "crf_sharpen": float(args.crf_sharpen),
            "crf": int(args.crf),
            "front_crf": None if args.front_crf is None else int(args.front_crf),
            "x264_preset": str(args.x264_preset),
            "fallback_crf": int(args.fallback_crf),
            "max_zip_gib": float(args.max_zip_gib),
            "effective_crf": int(args.crf),
            "clamped_to_fallback_crf": False,
        },
        "inputs": {
            "trajectories_path": str(args.trajectories),
            "trajectories_sha256": _sha256_file(args.trajectories),
            "history_path": str(args.history),
            "history_sha256": _sha256_file(args.history),
        },
        "camera_calibration": _collect_camera_calibration_provenance(),
        "runtime": {
            "gpu_ids": gpu_ids,
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "archive": None,
        "summary": {},
        "scenarios": {
            sid: {
                "scenario_type": stype,
                "depth_active": _scenario_uses_depth(cfg, stype),
                "status": "pending",
                "cameras": {
                    cam: {
                        "status": "pending",
                        "path": str(args.out_videos_dir / sid / f"{cam}.mp4"),
                        "size_bytes": 0,
                        "effective_crf": None,
                    }
                    for cam in spec.CAMERAS
                },
            }
            for sid, _fut, _past, stype in items
        },
    }


def _summarize_manifest_statuses(manifest: Dict[str, Any]) -> Dict[str, int]:
    """Compute scenario and camera counts by generation status."""
    scen_counts = {"total": 0, "rendered": 0, "resumed": 0, "failed": 0, "pending": 0}
    cam_counts = {"total": 0, "rendered": 0, "resumed": 0, "failed": 0, "pending": 0, "missing": 0}
    for entry in manifest["scenarios"].values():
        scen_counts["total"] += 1
        st = str(entry.get("status", "pending"))
        scen_counts[st] = scen_counts.get(st, 0) + 1
        for cam_info in entry["cameras"].values():
            cam_counts["total"] += 1
            cst = str(cam_info.get("status", "pending"))
            cam_counts[cst] = cam_counts.get(cst, 0) + 1
    return {
        "scenarios_total": scen_counts["total"],
        "scenarios_rendered": scen_counts["rendered"],
        "scenarios_resumed": scen_counts["resumed"],
        "scenarios_failed": scen_counts["failed"],
        "scenarios_pending": scen_counts["pending"],
        "cameras_total": cam_counts["total"],
        "cameras_rendered": cam_counts["rendered"],
        "cameras_resumed": cam_counts["resumed"],
        "cameras_failed": cam_counts["failed"],
        "cameras_pending": cam_counts["pending"],
        "cameras_missing": cam_counts["missing"],
    }


def _reconcile_manifest_with_disk(
    manifest: Dict[str, Any],
    out_videos_dir: Path,
    *,
    clamped_crf: Optional[int] = None,
    initial_crf: Optional[int] = None,
) -> None:
    """Refresh file sizes/statuses on disk so even partial/clamped runs stay accurate."""
    for sid, entry in manifest["scenarios"].items():
        scen_dir = out_videos_dir / sid
        cams = entry["cameras"]
        for cam in spec.CAMERAS:
            mp4 = scen_dir / f"{cam}.mp4"
            info = cams[cam]
            if mp4.exists() and mp4.stat().st_size > 1024:
                info["size_bytes"] = mp4.stat().st_size
                if info["status"] in ("pending", "missing"):
                    info["status"] = "rendered"
                    if info.get("effective_crf") is None and initial_crf is not None:
                        info["effective_crf"] = initial_crf
                if clamped_crf is not None:
                    info["effective_crf"] = clamped_crf
            elif info["status"] == "pending":
                info["status"] = "failed"
                info["size_bytes"] = mp4.stat().st_size if mp4.exists() else 0
        if entry["status"] == "pending":
            statuses = {cams[c]["status"] for c in spec.CAMERAS}
            if statuses <= {"rendered", "resumed"}:
                entry["status"] = "resumed" if statuses == {"resumed"} else "rendered"
            else:
                entry["status"] = "failed"
    manifest["summary"] = _summarize_manifest_statuses(manifest)


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    args = parse_args(argv)
    t0_dir = unpack_if_needed(args.t0_dir, args.t0_tar, args.unpack_dir, "test_t0_native")
    past_dir: Optional[Path] = None
    if args.past_dir is not None or (args.past_tar is not None and args.past_tar.exists()):
        past_dir = unpack_if_needed(args.past_dir, args.past_tar, args.unpack_dir, "test_past")

    items = load_test_inputs(args.trajectories, args.history)
    if args.scenarios.strip():
        want = {s.strip().zfill(7) for s in args.scenarios.split(",") if s.strip()}
        items = [it for it in items if it[0] in want]
    if args.scenario_range.strip():
        parts = args.scenario_range.strip().split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"--scenario-range must be START:END, got {args.scenario_range!r}")
        start_idx = int(parts[0]) if parts[0] else 0
        end_idx = int(parts[1]) if parts[1] else len(items)
        items = items[start_idx:end_idx]
    if args.limit > 0:
        items = items[: args.limit]
    if not items:
        raise RuntimeError("No scenarios selected to render.")

    n_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",") if g.strip() and int(g.strip()) < n_cuda]

    cfg: Dict[str, Any] = {
        "depth_model": args.depth_model,
        "use_depth": not args.no_depth,
        "no_depth_slices": tuple(s.strip() for s in args.no_depth_slices.split(",") if s.strip()),
        "use_lama": not args.no_lama,
        "min_lama_hole_pixels": args.min_lama_hole_pixels,
        "min_lama_luma": float(args.min_lama_luma),
        "ego_static_min_luma": float(args.ego_static_min_luma),
        "render_scale": args.render_scale,
        "adaptive_filter": bool(args.adaptive_filter),
        "photometric_refine": bool(args.photometric_refine),
        "photometric_strength": float(args.photometric_strength),
        "saturation_strength": float(args.saturation_strength),
        "equalize_strength": float(args.equalize_strength),
        "crf_sharpen": args.crf_sharpen,
        "crf": args.crf,
        "front_crf": args.front_crf,
        "x264_preset": args.x264_preset,
        "esrgan_alpha": float(args.esrgan_alpha),
        "esrgan_night_wdn_alpha": float(args.esrgan_night_wdn_alpha),
        "esrgan_min_luma": float(args.esrgan_min_luma),
        "esrgan_weights_dir": str(args.esrgan_weights_dir),
    }

    manifest_path = args.manifest_out or (args.out_videos_dir / "generation_manifest.json")
    manifest = _build_initial_manifest(raw_argv=raw_argv, args=args, cfg=cfg, items=items, gpu_ids=gpu_ids)
    manifest["summary"] = _summarize_manifest_statuses(manifest)
    _write_json_atomic(manifest_path, manifest)

    t0 = time.time()
    print(f"[*] Rendering {len(items)} scenarios | GPUs={gpu_ids or ['cpu']} | cfg={cfg}", flush=True)

    clamped_crf: Optional[int] = None
    try:
        if len(gpu_ids) <= 1:
            gpu_sel = gpu_ids[0] if gpu_ids else None
            count, elapsed, worker_statuses = _gpu_worker(
                0, gpu_sel, items, t0_dir, past_dir, args.out_videos_dir, cfg
            )
            manifest["scenarios"].update(worker_statuses)
            print(f"[+] Single-worker complete: {count} scenarios in {elapsed:.1f}s", flush=True)
        else:
            # Pre-warm HuggingFace & LaMa weights in main process before spawning multi-GPU workers
            if cfg["use_depth"]:
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation

                AutoImageProcessor.from_pretrained(str(cfg["depth_model"]))
                AutoModelForDepthEstimation.from_pretrained(str(cfg["depth_model"]))
            if cfg["use_lama"]:
                from simple_lama_inpainting.models.model import LAMA_MODEL_URL
                from simple_lama_inpainting.utils.util import download_model

                download_model(LAMA_MODEL_URL)

            shards = [items[i :: len(gpu_ids)] for i in range(len(gpu_ids))]
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=len(gpu_ids)) as pool:
                jobs = [
                    pool.apply_async(
                        _gpu_worker,
                        (wid, gpu_ids[wid], shards[wid], t0_dir, past_dir, args.out_videos_dir, cfg),
                    )
                    for wid in range(len(gpu_ids))
                ]
                for job in jobs:
                    _count, _elapsed, worker_statuses = job.get()
                    manifest["scenarios"].update(worker_statuses)
            print(f"[+] Parallel Dual-GPU render complete in {time.time() - t0:.1f}s", flush=True)

        if args.pack_zip:
            max_bytes = int(args.max_zip_gib * (1024 ** 3))
            if args.fallback_crf > 0 and _clamp_videos_crf_if_oversized(
                args.out_videos_dir,
                max_bytes,
                args.fallback_crf,
                preset=str(args.x264_preset),
                preserve_front_first=(args.front_crf is not None),
                surround_base_crf=int(args.crf),
            ):
                clamped_crf = int(args.fallback_crf)
            print(f"[*] Validating and packing {args.out_zip} ...", flush=True)
            report = archive.build_archive(
                args.out_videos_dir,
                output=args.out_zip,
                allow_partial=args.allow_partial,
                check_media=True,
            )
            if args.fallback_crf > 0 and not report.ok and args.out_zip.exists() and args.out_zip.stat().st_size > max_bytes:
                if _clamp_videos_crf_if_oversized(
                    args.out_videos_dir,
                    0,
                    args.fallback_crf,
                    preset=str(args.x264_preset),
                    preserve_front_first=(args.front_crf is not None),
                    surround_base_crf=int(args.crf),
                ):
                    clamped_crf = int(args.fallback_crf)
                report = archive.build_archive(
                    args.out_videos_dir,
                    output=args.out_zip,
                    allow_partial=args.allow_partial,
                    check_media=True,
                )
            manifest["archive"] = {
                **report.as_dict(),
                "sha256": _sha256_file(args.out_zip) if args.out_zip.exists() else None,
            }
            for line in report.summary_lines():
                print(f"    {line}", flush=True)
            if not report.ok:
                for err in report.errors[:20]:
                    print(f"[!] Archive error: {err}", file=sys.stderr, flush=True)
                return 1
    finally:
        if clamped_crf is not None:
            manifest["render_config"]["clamped_to_fallback_crf"] = True
            manifest["render_config"]["effective_crf"] = clamped_crf
        _reconcile_manifest_with_disk(
            manifest,
            args.out_videos_dir,
            clamped_crf=clamped_crf,
            initial_crf=int(args.crf),
        )
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json_atomic(manifest_path, manifest)
        print(f"[+] Wrote provenance manifest -> {manifest_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
