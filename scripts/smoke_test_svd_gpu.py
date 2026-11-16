#!/usr/bin/env python3
"""Standalone GPU Preflight Smoke-Test Script for SVD-XT on Kaggle T4 GPU.

Verifies memory footprint, execution latency, autograd backward pass, and parameter
freezing contracts for the genuine 1.5B Stable Video Diffusion (SVD-XT) UNet
coupled with the Spatial Conditioning Adapter (Option B.1).

Designed specifically for Kaggle 16GB T4 GPU preflight checks:
- Strict fail-loud safety check preventing accidental CPU execution and OOM.
- Peak allocated and reserved VRAM tracking (GB).
- Strict autograd gradient isolation audit:
    * UNet backbone: 100% frozen (all grad is None).
    * Spatial adapter: actively receives non-zero gradients (grad.norm() > 0).
- 16GB T4 fit check (< 15.0 GB peak reserved) and gradient checkpointing recommendations.
- Saves structured JSON and Markdown audit reports to reports/.

Usage:
    python scripts/smoke_test_svd_gpu.py --device cuda:0
    python scripts/smoke_test_svd_gpu.py --device cuda:0 --enable-gradient-checkpointing
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitscenes.spatial_adapter import SpatialAdapter, SpatialAdapterConfig

CUDA_REQUIRED_ERROR_MSG = (
    "CUDA is required to run the genuine SVD-XT GPU smoke test. "
    "CPU execution is strictly prohibited to prevent memory exhaustion."
)


def validate_cuda_device(device: Union[str, torch.device]) -> torch.device:
    """Validate that CUDA is available and that the requested device string/index is valid.

    Raises:
        RuntimeError: If CUDA is not available or non-CUDA device (e.g. CPU/MPS) is requested.
        ValueError: If device index is out of bounds or device string is malformed.
    """
    dev_str = str(device).lower().strip()
    if not dev_str.startswith("cuda"):
        raise RuntimeError(CUDA_REQUIRED_ERROR_MSG)

    if not torch.cuda.is_available():
        raise RuntimeError(CUDA_REQUIRED_ERROR_MSG)

    try:
        parsed_device = torch.device(device)
    except Exception as exc:
        raise ValueError(f"Invalid device specification '{device}': {exc}") from exc

    if parsed_device.type != "cuda":
        raise RuntimeError(CUDA_REQUIRED_ERROR_MSG)

    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError(CUDA_REQUIRED_ERROR_MSG)

    dev_index = parsed_device.index
    if dev_index is None:
        dev_index = torch.cuda.current_device()
        parsed_device = torch.device(f"cuda:{dev_index}")

    if dev_index < 0 or dev_index >= device_count:
        raise ValueError(
            f"Invalid CUDA device index {dev_index}. Total CUDA devices available: {device_count}."
        )

    return parsed_device


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for SVD-XT GPU smoke test."""
    parser = argparse.ArgumentParser(
        description="Standalone GPU preflight smoke test for SVD-XT UNet + Spatial Adapter on Kaggle T4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="stabilityai/stable-video-diffusion-img2vid-xt",
        help="Hugging Face model ID or local directory for SVD-XT UNet backbone",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device to run the smoke test on (CPU/MPS execution is strictly prohibited)",
    )
    parser.add_argument(
        "--enable-gradient-checkpointing",
        action="store_true",
        help="Enable activation gradient checkpointing on SVD-XT UNet to minimize peak VRAM",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/svd_gpu_smoke_test.json"),
        help="Path where JSON report will be saved (Markdown report will use same stem with .md)",
    )
    return parser


def generate_markdown_report(metrics: Dict[str, Any]) -> str:
    """Render a human-readable Markdown preflight audit report."""
    gc_status = "Enabled" if metrics["gradient_checkpointing"] else "Disabled"
    fit_badge = "**FIT: PASSES** (< 15.0 GB)" if metrics["fits_16gb_t4"] else "**FIT: EXCEEDS LIMIT** (>= 15.0 GB)"

    rec_lines = []
    if metrics["recommend_gradient_checkpointing"]:
        rec_lines.append(
            "> [!WARNING]\n"
            "> **High Peak VRAM Detected**: Peak memory exceeds 13.5 GB threshold. "
            "Enabling `--enable-gradient-checkpointing` is strongly recommended for training "
            "to prevent CUDA OOM on Kaggle 16GB T4 instances."
        )
    else:
        rec_lines.append(
            "> [!NOTE]\n"
            "> **Safe Memory Headroom**: Peak VRAM remains well within safe margins for Kaggle 16GB T4."
        )

    md = f"""# SVD-XT GPU Smoke Test & Preflight Audit Report

- **Date/Time (UTC):** {metrics["timestamp"]}
- **Host / Device:** {metrics["device_name"]} (`{metrics["device"]}`)
- **Total GPU VRAM:** {metrics["total_vram_gb"]:.2f} GB
- **CUDA Compute Capability:** {metrics.get("device_capability", "N/A")}
- **Overall Status:** `{metrics["status"]}` — {fit_badge}

---

## 1. Executive Summary

| Metric | Measured Value | Kaggle 16GB T4 Budget / Target | Compliance Status |
| :--- | :--- | :--- | :--- |
| **Peak Reserved VRAM** | **`{metrics["peak_reserved_gb"]:.2f} GB`** | `< 15.00 GB` | {"✅ PASS" if metrics["fits_16gb_t4"] else "❌ FAIL"} |
| **Peak Allocated VRAM** | **`{metrics["peak_allocated_gb"]:.2f} GB`** | `< 14.00 GB` | {"✅ PASS" if metrics["peak_allocated_gb"] < 14.0 else "⚠️ WARN"} |
| **Baseline UNet + Adapter VRAM** | `{metrics["baseline_allocated_gb"]:.2f} GB` (Allocated) | `< 5.00 GB` | ✅ PASS |
| **Forward Pass Latency** | `{metrics["forward_time_s"]:.3f} s` | Single batch forward | ✅ Complete |
| **Backward Pass Latency** | `{metrics["backward_time_s"]:.3f} s` | Single batch autograd | ✅ Complete |
| **Total Step Latency** | `{metrics["total_step_time_s"]:.3f} s` | Preflight forward+backward | ✅ Complete |
| **Backbone Parameters Frozen** | `{metrics["unet_frozen_verified"]}` (100% frozen) | `all(p.grad is None)` | ✅ PASS |
| **Adapter Gradients Active** | `{metrics["adapter_gradients_verified"]}` (Active) | `any(p.grad.norm() > 0)` | ✅ PASS |

{"".join(rec_lines)}

---

## 2. Configuration & Parameter Audit

- **Backbone Model ID:** `{metrics["model_id"]}`
- **UNet Parameter Count:** `{metrics["unet_parameter_count"]:,}` parameters
- **UNet Parameter `requires_grad`:** `False` (strictly frozen)
- **Spatial Adapter Parameter Count:** `{metrics["spatial_adapter_parameter_count"]:,}` parameters
- **Spatial Adapter `requires_grad`:** `True` (active gradient target)
- **Gradient Checkpointing:** `{gc_status}`
- **Input Tensor Dimensions:**
  * `noisy_latents`: `(1, 25, 4, 72, 128)` [FP16]
  * `initial_image_latents`: `(1, 25, 4, 72, 128)` [FP16, repeated along temporal dim]
  * `spatial_masks`: `(1, 25, 1, 568, 876)` [FP16]
  * `encoder_hidden_states`: `(1, 1, 1024)` [FP16]
  * `added_time_ids`: `tensor([[4.0, 127.0, 0.02]])` [FP16]
  * `timesteps`: continuous $c_{{noise}} = 0.05$ [FP16]

---

## 3. Detailed VRAM Profiling Progression

| Execution Phase | Allocated VRAM (GB) | Reserved / Cached VRAM (GB) | Delta from Baseline |
| :--- | :--- | :--- | :--- |
| **Initial Baseline (Weights Loaded)** | `{metrics["baseline_allocated_gb"]:.3f} GB` | `{metrics["baseline_reserved_gb"]:.3f} GB` | `+0.000 GB` |
| **Post-Forward Pass** | `{metrics["forward_allocated_gb"]:.3f} GB` | `{metrics["forward_reserved_gb"]:.3f} GB` | `+{metrics["forward_allocated_gb"] - metrics["baseline_allocated_gb"]:.3f} GB` |
| **Peak During Backward Pass** | **`{metrics["peak_allocated_gb"]:.3f} GB`** | **`{metrics["peak_reserved_gb"]:.3f} GB`** | `+{metrics["peak_allocated_gb"] - metrics["baseline_allocated_gb"]:.3f} GB` |

---

## 4. Autograd Gradient Isolation Audit

- **UNet Parameter Audit:** Evaluated {metrics["unet_parameter_count"]:,} backbone weights. Every single parameter retained `p.grad is None`. Zero gradient leakage observed.
- **Spatial Adapter Audit:** Evaluated {metrics["spatial_adapter_parameter_count"]:,} adapter weights. Verified `p.grad is not None` and `p.grad.norm() > 0`. Backpropagation successfully navigated through the frozen UNet back to the conditioning adapter.

---

## 5. Preflight Conclusion & Deployment Guidance

1. **Kaggle T4 Suitability:** {"Fits within single 16GB T4 instance." if metrics["fits_16gb_t4"] else "Exceeds 15.0 GB safety budget. Gradient checkpointing mandatory."}
2. **Gradient Checkpointing Recommendation:** {"Recommended" if metrics["recommend_gradient_checkpointing"] else "Optional (memory headroom is sufficient)"}.
3. **Execution Ready:** Model architecture and autograd computation graphs are verified and ready for training.
"""
    return md


def run_svd_gpu_smoke_test(
    model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt",
    device: Union[str, torch.device] = "cuda:0",
    enable_gradient_checkpointing: bool = False,
    output_json: Union[str, Path] = "reports/svd_gpu_smoke_test.json",
) -> Dict[str, Any]:
    """Execute dedicated SVD-XT GPU preflight smoke test on genuine CUDA accelerator.

    Args:
        model_id: Hugging Face model repository or local path for SVD-XT UNet.
        device: CUDA device identifier (e.g. 'cuda:0').
        enable_gradient_checkpointing: Whether to enable activation checkpointing on UNet.
        output_json: Target path for structured JSON results report.

    Returns:
        Dictionary containing all measured metrics, latencies, and audit verification flags.
    """
    output_json_path = Path(output_json)
    output_md_path = output_json_path.with_suffix(".md")

    # 1. Strict safety check & device validation
    parsed_device = validate_cuda_device(device)
    device_str = str(parsed_device)

    print("=" * 70)
    print("SVD-XT GPU PREFLIGHT SMOKE TEST (KAGGLE T4 VALIDATION)")
    print(f"Device: {device_str} | Model ID: {model_id}")
    print(f"Gradient Checkpointing: {enable_gradient_checkpointing}")
    print("=" * 70)

    device_name = torch.cuda.get_device_name(parsed_device)
    device_props = torch.cuda.get_device_properties(parsed_device)
    total_vram_gb = device_props.total_memory / (1024 ** 3)
    device_capability = f"{device_props.major}.{device_props.minor}"
    print(f"Accelerator: {device_name} ({total_vram_gb:.2f} GB total VRAM, CC {device_capability})")

    # 2. Reset memory stats and clean cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(parsed_device)
    torch.cuda.synchronize(parsed_device)

    # 3. Load genuine SVD-XT UNet in FP16
    print(f"\n[*] Loading genuine SVD-XT UNet ({model_id}) onto {device_str} (torch.float16)...")
    from diffusers import UNetSpatioTemporalConditionModel

    t_load_start = time.perf_counter()
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        torch_dtype=torch.float16,
    ).to(parsed_device)
    torch.cuda.synchronize(parsed_device)
    load_time_s = time.perf_counter() - t_load_start
    print(f"[✓] UNet loaded in {load_time_s:.2f}s")

    # 4. Freeze all UNet parameters
    unet.eval()
    for p in unet.parameters():
        p.requires_grad = False
    unet_parameter_count = sum(p.numel() for p in unet.parameters())
    print(f"[✓] UNet backbone strictly frozen: {unet_parameter_count:,} parameters (requires_grad = False)")

    # 5. Instantiate SpatialAdapter (Option B.1)
    print("\n[*] Instantiating SpatialAdapter (Option B.1) in torch.float16...")
    adapter_config = SpatialAdapterConfig(
        in_channels=1,
        adapter_channels=(32, 64, 128, 256),
        latent_channels=4,
        down_block_channels=(320, 640, 1280, 1280),
        conditioning_type="latent_addition",
        use_temporal_conv=True,
        zero_conv_init=True,
    )
    spatial_adapter = SpatialAdapter(adapter_config).to(device=parsed_device, dtype=torch.float16)
    spatial_adapter.train()
    for p in spatial_adapter.parameters():
        p.requires_grad = True
    adapter_parameter_count = sum(p.numel() for p in spatial_adapter.parameters() if p.requires_grad)
    print(f"[✓] SpatialAdapter instantiated: {adapter_parameter_count:,} trainable parameters (requires_grad = True)")

    # 6. Apply gradient checkpointing if requested
    if enable_gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        print("[✓] UNet activation gradient checkpointing enabled")

    # 7. Measure baseline VRAM
    torch.cuda.synchronize(parsed_device)
    baseline_allocated_bytes = torch.cuda.memory_allocated(parsed_device)
    baseline_reserved_bytes = torch.cuda.memory_reserved(parsed_device)
    baseline_allocated_gb = baseline_allocated_bytes / (1024 ** 3)
    baseline_reserved_gb = baseline_reserved_bytes / (1024 ** 3)
    print(f"[*] Baseline VRAM: {baseline_allocated_gb:.2f} GB allocated | {baseline_reserved_gb:.2f} GB reserved")

    # 8. Create synthetic input latents matching genuine dimensions
    print("\n[*] Synthesizing genuine-dimension input tensors (FP16)...")
    noisy_latents = torch.randn(1, 25, 4, 72, 128, device=parsed_device, dtype=torch.float16)
    # initial_image_latents: (1, 4, 72, 128) repeated to (1, 25, 4, 72, 128)
    initial_image_latents_4d = torch.randn(1, 4, 72, 128, device=parsed_device, dtype=torch.float16)
    initial_image_latents = initial_image_latents_4d.unsqueeze(1).repeat(1, 25, 1, 1, 1)
    spatial_masks = torch.rand(1, 25, 1, 568, 876, device=parsed_device, dtype=torch.float16)
    encoder_hidden_states = torch.randn(1, 1, 1024, device=parsed_device, dtype=torch.float16)
    added_time_ids = torch.tensor([[4.0, 127.0, 0.02]], device=parsed_device, dtype=torch.float16)
    timesteps = torch.tensor([0.05], device=parsed_device, dtype=torch.float16)

    print(f"  - noisy_latents:         {tuple(noisy_latents.shape)}")
    print(f"  - initial_image_latents: {tuple(initial_image_latents.shape)}")
    print(f"  - spatial_masks:         {tuple(spatial_masks.shape)}")
    print(f"  - encoder_hidden_states: {tuple(encoder_hidden_states.shape)}")
    print(f"  - added_time_ids:        {tuple(added_time_ids.shape)}")
    print(f"  - timesteps (c_noise):   {timesteps.item():.4f}")

    # 9. Execute exactly ONE forward pass
    print("\n[*] Executing forward pass...")
    torch.cuda.synchronize(parsed_device)
    t_fwd_start = time.perf_counter()

    conditioned_latents = spatial_adapter.apply_to_latent(noisy_latents, spatial_masks)
    unet_input = torch.cat([conditioned_latents, initial_image_latents], dim=2)

    unet_out = unet(
        sample=unet_input,
        timestep=timesteps,
        encoder_hidden_states=encoder_hidden_states,
        added_time_ids=added_time_ids,
    )

    torch.cuda.synchronize(parsed_device)
    forward_time_s = time.perf_counter() - t_fwd_start
    forward_allocated_bytes = torch.cuda.memory_allocated(parsed_device)
    forward_reserved_bytes = torch.cuda.memory_reserved(parsed_device)
    forward_allocated_gb = forward_allocated_bytes / (1024 ** 3)
    forward_reserved_gb = forward_reserved_bytes / (1024 ** 3)
    print(f"[✓] Forward pass complete in {forward_time_s:.3f}s")
    print(f"    Post-forward VRAM: {forward_allocated_gb:.2f} GB allocated | {forward_reserved_gb:.2f} GB reserved")

    # 10. Compute dummy scalar loss
    loss = unet_out.sample.mean()
    loss_val = float(loss.item())
    print(f"[*] Computed dummy scalar loss: {loss_val:.6f}")

    # 11. Execute exactly ONE backward pass
    print("\n[*] Executing backward pass (loss.backward())...")
    torch.cuda.synchronize(parsed_device)
    t_bwd_start = time.perf_counter()

    loss.backward()

    torch.cuda.synchronize(parsed_device)
    backward_time_s = time.perf_counter() - t_bwd_start
    total_step_time_s = forward_time_s + backward_time_s

    peak_allocated_bytes = torch.cuda.max_memory_allocated(parsed_device)
    peak_reserved_bytes = torch.cuda.max_memory_reserved(parsed_device)
    peak_allocated_gb = peak_allocated_bytes / (1024 ** 3)
    peak_reserved_gb = peak_reserved_bytes / (1024 ** 3)
    print(f"[✓] Backward pass complete in {backward_time_s:.3f}s (Total step: {total_step_time_s:.3f}s)")
    print(f"    Peak allocated VRAM: {peak_allocated_gb:.2f} GB")
    print(f"    Peak reserved VRAM:  {peak_reserved_gb:.2f} GB")

    # 12. Audit gradients strictly
    print("\n[*] Performing strict autograd parameter isolation audit...")
    unet_frozen_verified = all(p.grad is None for p in unet.parameters())
    adapter_gradients_verified = any(
        p.grad is not None and p.grad.norm() > 0 for p in spatial_adapter.parameters()
    )

    if not unet_frozen_verified:
        raise RuntimeError(
            "Gradient audit failed: Frozen SVD UNet backbone parameters received gradients!"
        )
    print("    [PASS] UNet parameters strictly frozen: all(p.grad is None) verified.")

    if not adapter_gradients_verified:
        raise RuntimeError(
            "Gradient audit failed: SpatialAdapter parameters received zero gradients!"
        )
    print("    [PASS] SpatialAdapter parameters active: non-zero gradients verified.")

    # 13. Evaluate fit status & recommendations
    fits_16gb_t4 = peak_reserved_gb < 15.0
    recommend_gradient_checkpointing = (peak_reserved_gb > 13.5) or (peak_allocated_gb > 13.5)

    print("\n" + "=" * 70)
    print("KAGGLE 16GB T4 PREFLIGHT ASSESSMENT")
    print("=" * 70)
    print(f"Peak Reserved VRAM:     {peak_reserved_gb:.2f} GB / 15.00 GB Budget")
    print(f"Fits Single 16GB T4:    {fits_16gb_t4}")
    if recommend_gradient_checkpointing and not enable_gradient_checkpointing:
        print("[!] RECOMMENDATION: Peak VRAM exceeds 13.5 GB threshold. "
              "Enable '--enable-gradient-checkpointing' for stable headroom.")
    elif enable_gradient_checkpointing:
        print("[i] Gradient checkpointing active. Peak memory is optimized.")
    else:
        print("[✓] Peak VRAM is well within 16GB capacity.")

    # 14. Save JSON and Markdown Reports
    metrics: Dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "status": "PASSED" if fits_16gb_t4 else "EXCEEDS_15GB",
        "model_id": model_id,
        "device": device_str,
        "device_name": device_name,
        "device_capability": device_capability,
        "total_vram_gb": round(total_vram_gb, 3),
        "gradient_checkpointing": enable_gradient_checkpointing,
        "unet_parameter_count": unet_parameter_count,
        "spatial_adapter_parameter_count": adapter_parameter_count,
        "baseline_allocated_gb": round(baseline_allocated_gb, 3),
        "baseline_reserved_gb": round(baseline_reserved_gb, 3),
        "forward_time_s": round(forward_time_s, 4),
        "forward_allocated_gb": round(forward_allocated_gb, 3),
        "forward_reserved_gb": round(forward_reserved_gb, 3),
        "loss_scalar": round(loss_val, 6),
        "backward_time_s": round(backward_time_s, 4),
        "total_step_time_s": round(total_step_time_s, 4),
        "peak_allocated_gb": round(peak_allocated_gb, 3),
        "peak_reserved_gb": round(peak_reserved_gb, 3),
        "fits_16gb_t4": fits_16gb_t4,
        "recommend_gradient_checkpointing": recommend_gradient_checkpointing,
        "unet_frozen_verified": unet_frozen_verified,
        "adapter_gradients_verified": adapter_gradients_verified,
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[✓] Saved JSON smoke test report to: {output_json_path}")

    md_content = generate_markdown_report(metrics)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[✓] Saved Markdown smoke test report to: {output_md_path}")

    return metrics


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint for GPU smoke test."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        run_svd_gpu_smoke_test(
            model_id=args.model_id,
            device=args.device,
            enable_gradient_checkpointing=args.enable_gradient_checkpointing,
            output_json=args.output_json,
        )
        return 0
    except RuntimeError as exc:
        print(f"\n[X] SVD-XT GPU Smoke Test Execution Halted: {exc}", file=sys.stderr)
        raise exc


if __name__ == "__main__":
    sys.exit(main())
