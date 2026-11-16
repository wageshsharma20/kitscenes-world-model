#!/usr/bin/env python3
"""Encode genuine KITScenes training videos and t=0 frames into SVD VAE latents.

Applies:
1. Aspect ratio adaptation: 876x568 -> 888x576 -> 1024x576 with edge replication padding.
2. Normalization: [0, 255] uint8 -> [-1.0, 1.0] float32.
3. Genuine SVD VAE encoding via diffusers AutoencoderKLTemporalDecoder.
4. Latent scaling: latents = latent_dist.sample(generator=seed_gen) * vae.config.scaling_factor.
5. Saves clean latents to data/train_latents/<scenario_id>.pt:
   - target_latents: (1, 25, 4, 72, 128)
   - initial_latents: (1, 1, 4, 72, 128)
   - metadata: scenario_id, frame_count, fps, sha256 hashes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffusers import AutoencoderKLTemporalDecoder
from kitscenes import spec
from scripts.run_kaggle_svd_single_camera import adapt_aspect_ratio_to_svd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def encode_frame_to_latent(
    vae: AutoencoderKLTemporalDecoder,
    pil_img: Image.Image,
    device: str,
    generator: Optional[torch.Generator] = None,
    use_mode: bool = False,
    noise_aug_strength: float = 0.0,
) -> torch.Tensor:
    """Pad, normalize, and encode a single frame to latent slice (1, 4, 72, 128).

    If noise_aug_strength > 0, Gaussian noise is added to the normalized image tensor.
    If use_mode is True, dist.mode() is used instead of dist.sample().
    """
    padded_img, _ = adapt_aspect_ratio_to_svd(pil_img, target_width=1024, target_height=576)
    arr = np.array(padded_img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, 576, 1024)

    if noise_aug_strength > 0.0:
        noise = torch.randn_like(tensor, generator=generator) if generator is not None else torch.randn_like(tensor)
        tensor = tensor + noise_aug_strength * noise

    with torch.no_grad():
        dist = vae.encode(tensor).latent_dist
        if use_mode:
            latent = dist.mode() * vae.config.scaling_factor
        else:
            latent = dist.sample(generator=generator) * vae.config.scaling_factor
    return latent  # (1, 4, 72, 128)


def encode_scenario(
    scenario_id: str,
    observed_t0_dir: Path,
    future_videos_dir: Path,
    output_dir: Path,
    vae: AutoencoderKLTemporalDecoder,
    device: str,
    seed: int = 42,
) -> Dict[str, Any]:
    """Encode initial frame and 25 future frames for a single scenario."""
    gen = torch.Generator(device=device if device != "mps" else "cpu").manual_seed(seed)

    t0_path = observed_t0_dir / scenario_id / "front.jpg"
    video_dir = future_videos_dir / scenario_id / "front"
    video_mp4 = future_videos_dir / scenario_id / "front.mp4"

    if not t0_path.exists():
        raise FileNotFoundError(f"Initial frame missing: {t0_path}")
    if not video_dir.exists():
        raise FileNotFoundError(f"Future frames directory missing: {video_dir}")

    # 1. Encode Initial Frame (t=0) with 0.02 noise augmentation and posterior mode()
    img_t0 = Image.open(t0_path).convert("RGB")
    lat_init = encode_frame_to_latent(
        vae, img_t0, device, gen, use_mode=True, noise_aug_strength=0.02
    )  # (1, 4, 72, 128)

    # 2. Encode 25 Future Video Frames (clean sample, no noise aug)
    frame_files = sorted(video_dir.glob("frame_*.jpg"))
    if len(frame_files) != spec.FRAME_COUNT:
        raise ValueError(f"Expected {spec.FRAME_COUNT} frames in {video_dir}, found {len(frame_files)}")

    latent_slices: List[torch.Tensor] = []
    for f_idx, f_path in enumerate(frame_files):
        img_f = Image.open(f_path).convert("RGB")
        lat_f = encode_frame_to_latent(vae, img_f, device, gen, use_mode=False, noise_aug_strength=0.0)
        latent_slices.append(lat_f.cpu())

    # Stack along temporal dimension: (1, 25, 4, 72, 128)
    target_latents = torch.stack(latent_slices, dim=1).squeeze(2)

    assert target_latents.shape == (1, spec.FRAME_COUNT, 4, 72, 128), f"Bad shape: {target_latents.shape}"
    assert lat_init.shape == (1, 4, 72, 128), f"Bad shape: {lat_init.shape}"
    assert not torch.isnan(target_latents).any(), "NaN found in target latents"
    assert not torch.isnan(lat_init).any(), "NaN found in initial latents"

    # Save to disk
    out_file = output_dir / f"{scenario_id}.pt"
    metadata = {
        "scenario_id": scenario_id,
        "camera": "front",
        "num_frames": spec.FRAME_COUNT,
        "fps": spec.FPS,
        "latent_shape": list(target_latents.shape),
        "initial_latent_shape": list(lat_init.shape),
        "scaling_factor": float(vae.config.scaling_factor),
        "initial_latent_mode": "mode",
        "initial_noise_aug_strength": 0.02,
        "t0_sha256": sha256_file(t0_path),
        "mp4_sha256": sha256_file(video_mp4) if video_mp4.exists() else None,
        "status": "ENCODED_GENUINE_VAE",
    }

    torch.save(
        {
            "target_latents": target_latents,
            "initial_latents": lat_init.cpu().unsqueeze(1),  # (1, 1, 4, 72, 128)
            "metadata": metadata,
        },
        out_file,
    )

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode genuine videos to SVD VAE latents.")
    parser.add_argument("--observed-t0-dir", type=Path, default=Path("data/train_observed_t0"))
    parser.add_argument("--future-videos-dir", type=Path, default=Path("data/train_future_videos"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/train_latents"))
    parser.add_argument("--model-id", type=str, default="stabilityai/stable-video-diffusion-img2vid-xt")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ENCODING GENUINE TRAINING VIDEOS TO SVD-XT VAE LATENTS")
    print(f"Device: {args.device} | Output Directory: {args.output_dir}")
    print("=" * 80)

    t0 = time.time()
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.model_id,
        subfolder="vae",
        local_files_only=True,
    ).to(args.device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    print(f"Pretrained VAE loaded in {time.time() - t0:.2f}s")

    # Discover scenarios
    scenarios = sorted([d.name for d in args.future_videos_dir.iterdir() if d.is_dir() and (d / "front").exists()])
    print(f"Discovered {len(scenarios)} verified scenarios: {scenarios}")

    manifest_records = []
    for idx, sid in enumerate(scenarios):
        t_seq = time.time()
        print(f"[{idx+1}/{len(scenarios)}] Encoding scenario {sid}...", end=" ", flush=True)
        meta = encode_scenario(
            scenario_id=sid,
            observed_t0_dir=args.observed_t0_dir,
            future_videos_dir=args.future_videos_dir,
            output_dir=args.output_dir,
            vae=vae,
            device=args.device,
        )
        dur = time.time() - t_seq
        print(f"DONE in {dur:.2f}s | Target: {meta['latent_shape']}")
        manifest_records.append(meta)

    # Save manifest
    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"total_encoded": len(manifest_records), "records": manifest_records}, f, indent=2)

    print("=" * 80)
    print(f"Successfully encoded {len(manifest_records)} genuine sequences in {time.time() - t0:.2f}s!")
    print(f"Latents saved to: {args.output_dir}")
    print(f"Manifest saved to: {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
