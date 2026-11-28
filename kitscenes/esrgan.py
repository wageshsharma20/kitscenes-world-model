"""Shared RealESRGAN (Compact x4v3 + WDN night denoiser) & H.264 preset encoding utilities."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable, Optional, Tuple
import urllib.request

import cv2
import numpy as np
import torch

from kitscenes import spec

REALESRGAN_X4V3_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
)
REALESRGAN_WDN_X4V3_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth"
)


def encode_rgb_h264_preset(
    frames_rgb: np.ndarray,
    out_mp4: Path,
    crf: int,
    preset: str = "slow",
    fps: float = float(spec.FPS),
) -> None:
    """Encode (N, H, W, 3) uint8 RGB array into yuv420p H.264 MP4 with explicit libx264 preset."""
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
    blend_alpha: float = 0.75,
    min_luma: float = 35.0,
    night_blend_alpha: float = 0.65,
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Load realesr-general-x4v3 (and optional WDN night denoiser) via spandrel in fp16."""
    if blend_alpha <= 0.0 and night_blend_alpha <= 0.0:
        return None
    dev = torch.device(device if (device != "cpu" and torch.cuda.is_available()) else "cpu")
    model_day, use_fp16 = (
        _load_spandrel_model(weights_dir, REALESRGAN_X4V3_URL, dev)
        if blend_alpha > 0.0
        else (None, dev.type == "cuda")
    )
    model_night = (
        _load_spandrel_model(weights_dir, REALESRGAN_WDN_X4V3_URL, dev)[0]
        if night_blend_alpha > 0.0
        else None
    )

    @torch.inference_mode()
    def _enhance(frames_rgb: np.ndarray) -> np.ndarray:
        out = np.ascontiguousarray(frames_rgb, dtype=np.uint8).copy()
        n_f, h, w, _c = out.shape
        mean_luma = float(cv2.cvtColor(out[0], cv2.COLOR_RGB2GRAY).mean())
        is_night = mean_luma < min_luma
        active_model = model_night if is_night else model_day
        alpha = float(night_blend_alpha if is_night else blend_alpha)
        if active_model is None or alpha <= 0.0:
            return out
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
