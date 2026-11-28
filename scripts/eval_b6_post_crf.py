#!/usr/bin/env python3
"""Stage B.6: Strict <=2.92 GiB CRF-preset, night-safe sharpen peak, and RealESRGAN test on Stage B.5 clips."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import urllib.request

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitscenes import spec
from kitscenes.emc_eval import compute_video_optical_flow_emc

REALESRGAN_X4V3_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
)
REALESRGAN_WDN_X4V3_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth"
)


def decode_mp4_rgb(mp4_path: Path) -> np.ndarray:
    """Decode MP4 into (T, H, W, 3) uint8 RGB frames."""
    cap = cv2.VideoCapture(str(mp4_path))
    frames: List[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"Failed to decode frames from {mp4_path}")
    return np.stack(frames, axis=0)


def encode_rgb_h264_preset(
    frames_rgb: np.ndarray,
    out_mp4: Path,
    crf: int,
    preset: str = "slow",
    fps: float = float(spec.FPS),
) -> None:
    """Encode (N, H, W, 3) uint8 RGB array into yuv420p H.264 MP4 with explicit preset."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg binary not found in PATH.")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    n_f, h, w, _c = frames_rgb.shape
    assert n_f > 0
    tmp = out_mp4.with_name(f".{out_mp4.stem}.tmp{out_mp4.suffix}")
    try:
        proc = subprocess.Popen(
            [
                ffmpeg_bin,
                "-y",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{w}x{h}",
                "-r",
                f"{fps:.6f}",
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(int(crf)),
                "-pix_fmt",
                "yuv420p",
                str(tmp),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _out, err = proc.communicate(input=np.ascontiguousarray(frames_rgb, dtype=np.uint8).tobytes())
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg encode failed ({out_mp4}): {err.decode(errors='ignore')}")
        os.replace(tmp, out_mp4)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def reencode_mp4_preset(src_mp4: Path, dst_mp4: Path, crf: int, preset: str = "slow") -> None:
    """Re-encode an existing MP4 cleanly at a target CRF and x264 preset."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg binary not found in PATH.")
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_mp4.with_name(f".{dst_mp4.stem}.tmp{dst_mp4.suffix}")
    try:
        subprocess.run(
            [
                ffmpeg_bin,
                "-y",
                "-v",
                "error",
                "-i",
                str(src_mp4),
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(int(crf)),
                "-pix_fmt",
                "yuv420p",
                str(tmp),
            ],
            check=True,
        )
        os.replace(tmp, dst_mp4)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def project_2400_gib(videos_dir: Path, n_scenarios: int) -> Tuple[float, Dict[str, float]]:
    """Project 400-scenario (2400-video) archive GiB and per-camera GiB breakdown."""
    by_cam: Dict[str, int] = {c: 0 for c in spec.CAMERAS}
    for mp4 in videos_dir.rglob("*.mp4"):
        if mp4.stem in by_cam:
            by_cam[mp4.stem] += mp4.stat().st_size
    scale = 400.0 / float(max(n_scenarios, 1)) / float(1024**3)
    per_cam = {c: round(by_cam[c] * scale, 3) for c in spec.CAMERAS}
    return float(sum(by_cam.values()) * scale), per_cam


def apply_luma_safe_unsharp(frames_rgb: np.ndarray, extra_amount: float = 0.30, min_luma: float = 45.0) -> np.ndarray:
    """Apply extra unsharp mask only on bright/daylight frames so night ISO noise is untouched."""
    out = frames_rgb.copy()
    for i in range(out.shape[0]):
        gray_mean = float(cv2.cvtColor(out[i], cv2.COLOR_RGB2GRAY).mean())
        if gray_mean >= min_luma and extra_amount > 0.0:
            blur = cv2.GaussianBlur(out[i], (0, 0), 1.0)
            out[i] = cv2.addWeighted(out[i], 1.0 + extra_amount, blur, -extra_amount, 0)
    return out


def _load_spandrel_model(weights_dir: Path, url: str, dev: torch.device) -> Tuple[object, bool]:
    """Download if needed and load one spandrel SR model onto `dev` (fp16 on CUDA)."""
    try:
        import spandrel
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "spandrel"], check=True)
        import spandrel

    weights_dir.mkdir(parents=True, exist_ok=True)
    w_path = weights_dir / Path(url).name
    if not w_path.exists():
        print(f"[*] Downloading {url} -> {w_path} ...", flush=True)
        urllib.request.urlretrieve(url, w_path)

    model = spandrel.ModelLoader(device=dev).load_from_file(str(w_path)).model.eval()
    use_fp16 = dev.type == "cuda"
    if use_fp16:
        model = model.half()
    return model, use_fp16


def build_realesrgan_blend_fn(
    weights_dir: Path,
    device: str = "cuda:0",
    blend_alpha: float = 0.45,
    min_luma: float = 35.0,
    night_blend_alpha: float = 0.0,
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Load realesr-general-x4v3 (and optional WDN night denoiser) via spandrel in fp16."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model_day, use_fp16 = _load_spandrel_model(weights_dir, REALESRGAN_X4V3_URL, dev)
    model_night = (
        _load_spandrel_model(weights_dir, REALESRGAN_WDN_X4V3_URL, dev)[0]
        if night_blend_alpha > 0.0
        else None
    )

    @torch.inference_mode()
    def _enhance(frames_rgb: np.ndarray) -> np.ndarray:
        out = frames_rgb.copy()
        n_f, h, w, _c = out.shape
        mean_luma = float(cv2.cvtColor(out[0], cv2.COLOR_RGB2GRAY).mean())
        is_night = mean_luma < min_luma
        if is_night and model_night is None:
            return out
        active_model = model_night if is_night else model_day
        alpha = night_blend_alpha if is_night else blend_alpha
        for start in range(0, n_f, 2):
            chunk = out[start : start + 2]
            ten = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(dev).float().div_(255.0)
            if use_fp16:
                ten = ten.half()
            sr = active_model(ten).float().clamp_(0.0, 1.0)
            sr_ds = torch.nn.functional.interpolate(sr, size=(h, w), mode="area")
            blended = (1.0 - alpha) * ten.float() + alpha * sr_ds
            out[start : start + 2] = (
                blended.clamp_(0.0, 1.0).mul_(255.0).round_().byte().permute(0, 2, 3, 1).cpu().numpy()
            )
        return out

    return _enhance


def _repack_variant_under_budget(
    src_entries: List[Dict[str, object]],
    dst_root: Path,
    variant_name: str,
    n_scenarios: int,
    crf_schedule: Sequence[Tuple[int, int]],
    preset: str = "slow",
    frame_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    pred_trajs: Optional[Dict[str, List[List[float]]]] = None,
) -> Tuple[List[Dict[str, object]], float, Tuple[int, int]]:
    """Build one strict <=2.92 GiB submission-compliant variant from existing candidate clips."""
    v_dir = dst_root / "videos" / variant_name
    v_dir.mkdir(parents=True, exist_ok=True)
    cached_frames: Dict[str, np.ndarray] = {}

    existing_ok = all(
        (v_dir / str(e["scenario_id"]) / f"{e['camera']}.mp4").exists()
        and (v_dir / str(e["scenario_id"]) / f"{e['camera']}.mp4").stat().st_size > 1024
        for e in src_entries
    )
    chosen_pair = crf_schedule[-1]
    if existing_ok:
        proj_gib, cam_gib = project_2400_gib(v_dir, n_scenarios)
        if proj_gib <= 2.92:
            print(f"  [{variant_name}] Reusing cached <=2.92 GiB clips -> proj={proj_gib:.3f} GiB | {cam_gib}", flush=True)
            chosen_pair = crf_schedule[0]
        else:
            existing_ok = False

    if not existing_ok:
        if frame_transform is not None:
            for e in src_entries:
                sid, cam = str(e["scenario_id"]), str(e["camera"])
                cached_frames[f"{sid}__{cam}"] = frame_transform(decode_mp4_rgb(Path(str(e["mp4_path"]))))

        for f_crf, s_crf in crf_schedule:
            def _job(e: Dict[str, object], f_c: int = f_crf, s_c: int = s_crf) -> None:
                sid, cam = str(e["scenario_id"]), str(e["camera"])
                crf = f_c if cam == "front" else s_c
                dst_mp4 = v_dir / sid / f"{cam}.mp4"
                if frame_transform is not None:
                    encode_rgb_h264_preset(cached_frames[f"{sid}__{cam}"], dst_mp4, crf=crf, preset=preset)
                else:
                    reencode_mp4_preset(Path(str(e["mp4_path"])), dst_mp4, crf=crf, preset=preset)

            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(_job, src_entries))

            proj_gib, cam_gib = project_2400_gib(v_dir, n_scenarios)
            chosen_pair = (f_crf, s_crf)
            print(
                f"  [{variant_name}] crf=({f_crf},{s_crf}) preset={preset} -> proj={proj_gib:.3f} GiB | {cam_gib}",
                flush=True,
            )
            if proj_gib <= 2.92:
                break

    cached_frames.clear()
    emc_by_sid: Dict[str, Optional[float]] = {}
    if pred_trajs:
        by_sid_paths: Dict[str, Dict[str, Path]] = {}
        for e in src_entries:
            sid, cam = str(e["scenario_id"]), str(e["camera"])
            by_sid_paths.setdefault(sid, {})[cam] = v_dir / sid / f"{cam}.mp4"
        for sid, cam_paths in by_sid_paths.items():
            traj = pred_trajs.get(sid)
            if traj:
                try:
                    res = compute_video_optical_flow_emc(cam_paths, traj)
                    emc_by_sid[sid] = round(float(res.get("emc", 0.0)), 4)
                except Exception:
                    emc_by_sid[sid] = None

    out_entries: List[Dict[str, object]] = []
    for e in src_entries:
        sid, cam = str(e["scenario_id"]), str(e["camera"])
        mp4 = v_dir / sid / f"{cam}.mp4"
        emc_val = emc_by_sid[sid] if sid in emc_by_sid else e.get("emc_local")
        out_entries.append(
            {
                "key": f"{variant_name}__{sid}__{cam}",
                "variant": variant_name,
                "scenario_id": sid,
                "scenario_type": e["scenario_type"],
                "camera": cam,
                "mp4_path": str(mp4),
                "size_bytes": mp4.stat().st_size,
                "psnr": e.get("psnr"),
                "emc_local": emc_val,
            }
        )
    final_gib, _ = project_2400_gib(v_dir, n_scenarios)
    return out_entries, final_gib, chosen_pair


def main() -> int:
    b5_dir = Path("/content/stage_b5_calib")
    b6_dir = Path("/content/stage_b6_calib")
    b6_dir.mkdir(parents=True, exist_ok=True)

    m5 = json.loads((b5_dir / "render_manifest.json").read_text())
    s5 = json.loads((b5_dir / "qalign_scores.json").read_text())
    scenarios = m5["scenarios"]
    n_scen = len(scenarios)

    entries_by_v: Dict[str, List[Dict[str, object]]] = {}
    for e in m5["entries"]:
        entries_by_v.setdefault(str(e["variant"]), []).append(e)

    print("=== Stage B.5 Per-Camera Projected GiB ===", flush=True)
    for vn in ("cand_f_lumagate", "cand_k_native", "cand_m_sharp38"):
        gib, per_cam = project_2400_gib(b5_dir / "videos" / vn, n_scen)
        print(f"  {vn:18s}: {gib:.3f} GiB | {per_cam}", flush=True)

    pred_trajs: Dict[str, List[List[float]]] = {}
    val_root = Path(str(m5.get("val_root", "")))
    val_traj_p = next(
        (p for p in (val_root / "val_trajectories.jsonl", PROJECT_ROOT / "data" / "val_trajectories.jsonl") if p.exists()),
        None,
    )
    train_traj_p = next(
        (p for p in (val_root / "train_trajectories.jsonl", PROJECT_ROOT / "data" / "train_trajectories.jsonl") if p.exists()),
        PROJECT_ROOT / "data" / "train_trajectories.jsonl",
    )
    if val_traj_p is not None:
        from scripts.eval_world_model_qalign import _load_predicted_trajectories

        pred_trajs = _load_predicted_trajectories(
            "ridge",
            train_path=train_traj_p,
            val_path=val_traj_p,
            pred_path=None,
        )

    t0 = time.time()
    # 1. cand_m_sharp38 compressed cleanly with -preset slow to <= 2.92 GiB (high front priority)
    m_15_entries, m15_gib, m15_crf = _repack_variant_under_budget(
        entries_by_v["cand_m_sharp38"],
        b6_dir,
        "b6_m_slow_asym1",
        n_scen,
        crf_schedule=[(15, 19), (16, 19), (16, 20), (17, 20)],
        preset="slow",
    )
    # 2. cand_m_sharp38 with uniform CRF under <= 2.92 GiB (to verify front priority vs uniform CRF)
    m_unif_entries, munif_gib, munif_crf = _repack_variant_under_budget(
        entries_by_v["cand_m_sharp38"],
        b6_dir,
        "b6_m_slow_unif",
        n_scen,
        crf_schedule=[(18, 18), (19, 19)],
        preset="slow",
    )
    # 3. Night-safe extra sharpen (~0.54 daylight, 0.24 night) + -preset slow <= 2.92 GiB
    s58_entries, s58_gib, s58_crf = _repack_variant_under_budget(
        entries_by_v["cand_k_native"],
        b6_dir,
        "b6_sharp54_night",
        n_scen,
        crf_schedule=[(15, 19), (16, 19), (16, 20), (17, 20)],
        preset="slow",
        frame_transform=lambda arr: apply_luma_safe_unsharp(arr, extra_amount=0.28, min_luma=45.0),
        pred_trajs=pred_trajs,
    )
    # 4. RealESRGAN x4v3 fp16 blend on daylight cand_k_native frames + -preset slow <= 2.92 GiB
    esr_fn = build_realesrgan_blend_fn(b6_dir / "weights", device="cuda:0", blend_alpha=0.45, min_luma=35.0)
    esr_entries, esr_gib, esr_crf = _repack_variant_under_budget(
        entries_by_v["cand_k_native"],
        b6_dir,
        "b6_esrgan45_asym",
        n_scen,
        crf_schedule=[(15, 19), (16, 19), (16, 20)],
        preset="slow",
        frame_transform=esr_fn,
        pred_trajs=pred_trajs,
    )
    print(f"[+] Built 4 Stage B.6 <=2.92 GiB variants in {time.time() - t0:.1f}s", flush=True)

    keep_base = ("gt", "cand_f_lumagate", "cand_m_sharp38")
    all_entries = [e for vn in keep_base for e in entries_by_v[vn]]
    all_entries.extend(m_15_entries + m_unif_entries + s58_entries + esr_entries)

    v_cfgs = dict(m5.get("variant_configs", {}))
    v_cfgs = {k: v for k, v in v_cfgs.items() if k in ("cand_f_lumagate", "cand_m_sharp38")}
    v_cfgs["b6_m_slow_asym1"] = {"src": "cand_m_sharp38", "preset": "slow", "crf": list(m15_crf), "proj_gib": round(m15_gib, 3)}
    v_cfgs["b6_m_slow_unif"] = {"src": "cand_m_sharp38", "preset": "slow", "crf": list(munif_crf), "proj_gib": round(munif_gib, 3)}
    v_cfgs["b6_sharp54_night"] = {"src": "cand_k_native+unsharp28_day", "preset": "slow", "crf": list(s58_crf), "proj_gib": round(s58_gib, 3)}
    v_cfgs["b6_esrgan45_asym"] = {"src": "cand_k_native+esrgan45_day", "preset": "slow", "crf": list(esr_crf), "proj_gib": round(esr_gib, 3)}

    b6_manifest = {
        "variants": ["cand_f_lumagate", "cand_m_sharp38", "b6_m_slow_asym1", "b6_m_slow_unif", "b6_sharp54_night", "b6_esrgan45_asym"],
        "variant_configs": v_cfgs,
        "scenarios": scenarios,
        "entries": all_entries,
    }
    (b6_dir / "render_manifest.json").write_text(json.dumps(b6_manifest, indent=2))
    seed_scores = {k: v for k, v in s5.items() if k.startswith(("gt__", "cand_f_lumagate__", "cand_m_sharp38__"))}
    (b6_dir / "qalign_scores.json").write_text(json.dumps(seed_scores, indent=2))
    print(f"[+] Wrote manifest ({len(all_entries)} entries) & seeded {len(seed_scores)} Q-Align scores", flush=True)
    print("STAGE_B6_PREP_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
