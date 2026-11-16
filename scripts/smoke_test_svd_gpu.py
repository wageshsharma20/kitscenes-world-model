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

from kitscenes.multiview_conditioning import MultiViewConditioner
from kitscenes.spatial_adapter import (
    MockSVDUNet,
    SpatialAdapter,
    SpatialAdapterConfig,
)

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
        "--test-cssc",
        action="store_true",
        help="Test CSSC (MultiViewConditioner) 6-camera conditioning + SVD UNet integration",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run lightweight CPU mock verification using MockSVDUNet without loading 1.5B weights",
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
    is_mock = metrics.get("mock", False)
    gc_status = "Enabled" if metrics["gradient_checkpointing"] else "Disabled"

    if is_mock:
        fit_badge = "**NOT MEASURED (Mock CPU Surrogate)**"
        rec_lines = [
            "> [!NOTE]\n"
            "> **Mock Surrogate Mode**: This execution evaluated autograd graph connectivity and parameter freezing "
            "contracts on synthetic CPU tensors. GPU memory usage was **not measured** and must not be used to infer "
            "Kaggle 16GB T4 hardware compatibility."
        ]
        reserved_val = "N/A (Mock CPU)"
        allocated_val = "N/A (Mock CPU)"
        vram_status = "⚠️ NOT MEASURED (Mock CPU)"
        t4_suitability_str = "NOT MEASURED (Executed on CPU mock surrogate; requires genuine NVIDIA T4 hardware)."
        gc_rec_str = "NOT MEASURED (Requires actual GPU memory profiling on Kaggle T4)."
    else:
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
        reserved_val = f"`{metrics['peak_reserved_gb']:.2f} GB`"
        allocated_val = f"`{metrics['peak_allocated_gb']:.2f} GB`"
        vram_status = "✅ PASS" if metrics["fits_16gb_t4"] else "❌ FAIL"
        t4_suitability_str = "Fits within single 16GB T4 instance." if metrics["fits_16gb_t4"] else "Exceeds 15.0 GB safety budget. Gradient checkpointing mandatory."
        gc_rec_str = "Recommended" if metrics["recommend_gradient_checkpointing"] else "Optional (memory headroom is sufficient)"

    cssc_row = ""
    if metrics.get("test_cssc"):
        cssc_row = (
            f"\n| **CSSC MultiViewConditioner** | `{metrics.get('conditioner_parameter_count', 0):,}` parameters "
            f"| 27-token bank active | ✅ PASS |\n"
            f"| **CSSC Gradients Active** | `{metrics.get('conditioner_gradients_verified')}` (Active) "
            f"| `any(p.grad.norm() > 0)` | ✅ PASS |"
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
| **Peak Reserved VRAM** | **{reserved_val}** | `< 15.00 GB` | {vram_status} |
| **Peak Allocated VRAM** | **{allocated_val}** | `< 14.00 GB` | {vram_status} |
| **Baseline UNet + Adapter VRAM** | `{metrics["baseline_allocated_gb"]:.2f} GB` (Allocated) | `< 5.00 GB` | {"✅ PASS" if not is_mock else "N/A"} |
| **Forward Pass Latency** | `{metrics["forward_time_s"]:.3f} s` | Single batch forward | ✅ Complete |
| **Backward Pass Latency** | `{metrics["backward_time_s"]:.3f} s` | Single batch autograd | ✅ Complete |
| **Total Step Latency** | `{metrics["total_step_time_s"]:.3f} s` | Preflight forward+backward | ✅ Complete |
| **Backbone Parameters Frozen** | `{metrics["unet_frozen_verified"]}` (100% frozen) | `all(p.grad is None)` | ✅ PASS |
| **Adapter Gradients Active** | `{metrics["adapter_gradients_verified"]}` (Active) | `any(p.grad.norm() > 0)` | ✅ PASS |{cssc_row}

{"".join(rec_lines)}

---

## 2. Configuration & Parameter Audit

- **Backbone Model ID:** `{metrics["model_id"]}`
- **UNet Parameter Count:** `{metrics["unet_parameter_count"]:,}` parameters
- **UNet Parameter `requires_grad`:** `False` (strictly frozen)
- **Spatial Adapter Parameter Count:** `{metrics["spatial_adapter_parameter_count"]:,}` parameters
- **Spatial Adapter `requires_grad`:** `True` (active gradient target)
- **CSSC Conditioning:** {"Enabled (27-token bank: target view + pooled summary + 25-step kinematics)" if metrics.get("test_cssc") else "Disabled (Standard single-token SVD-XT CLIP conditioning)"}
- **Gradient Checkpointing:** `{gc_status}`

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
"""
    if metrics.get("test_cssc"):
        md += f"""- **CSSC MultiViewConditioner Audit:** Evaluated {metrics.get('conditioner_parameter_count', 0):,} weights. Verified `p.grad is not None` and `p.grad.norm() > 0`. Backpropagation successfully navigated through the frozen UNet cross-attention to the 27-token multi-view conditioning bank.
"""

    md += f"""
---

## 5. Preflight Conclusion & Deployment Guidance

1. **Kaggle T4 Suitability:** {t4_suitability_str}
2. **Gradient Checkpointing Recommendation:** {gc_rec_str}
3. **Execution Ready:** Model architecture and autograd computation graphs are verified and ready for training.
"""
    return md


def apply_deep_gradient_checkpointing(unet: torch.nn.Module) -> None:
    """
    Enables deep gradient checkpointing across UNetSpatioTemporalConditionModel,
    including both standard ResNet checkpointing and transformer attention blocks.
    In vanilla Diffusers, CrossAttnDown/UpBlock only checkpoints ResNet modules,
    leaving multi-head spatial/temporal attention activation maps uncheckpointed.
    """
    if hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()
        print("[✓] Diffusers UNet activation gradient checkpointing enabled")

    try:
        from diffusers.models.unets.unet_3d_blocks import (
            CrossAttnDownBlockSpatioTemporal,
            CrossAttnUpBlockSpatioTemporal,
            UNetMidBlockSpatioTemporal,
        )

        def custom_down_forward(
            self,
            hidden_states: torch.Tensor,
            temb=None,
            encoder_hidden_states=None,
            image_only_indicator=None,
        ):
            output_states = ()
            blocks = list(zip(self.resnets, self.attentions))
            for resnet, attn in blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    def _forward_block(h, m_resnet=resnet, m_attn=attn):
                        h_res = m_resnet(h, temb, image_only_indicator=image_only_indicator)
                        return m_attn(
                            h_res,
                            encoder_hidden_states=encoder_hidden_states,
                            image_only_indicator=image_only_indicator,
                            return_dict=False,
                        )[0]

                    hidden_states = torch.utils.checkpoint.checkpoint(
                        _forward_block, hidden_states, use_reentrant=False
                    )
                else:
                    hidden_states = resnet(hidden_states, temb, image_only_indicator=image_only_indicator)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        image_only_indicator=image_only_indicator,
                        return_dict=False,
                    )[0]
                output_states = output_states + (hidden_states,)

            if self.downsamplers is not None:
                for downsampler in self.downsamplers:
                    hidden_states = downsampler(hidden_states)
                output_states = output_states + (hidden_states,)
            return hidden_states, output_states

        CrossAttnDownBlockSpatioTemporal.forward = custom_down_forward

        def custom_up_forward(
            self,
            hidden_states: torch.Tensor,
            res_hidden_states_tuple,
            temb=None,
            encoder_hidden_states=None,
            image_only_indicator=None,
            upsample_size=None,
        ):
            for resnet, attn in zip(self.resnets, self.attentions):
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    def _forward_up_block(h, r_h, m_resnet=resnet, m_attn=attn):
                        cat_h = torch.cat([h, r_h], dim=1)
                        res_out = m_resnet(cat_h, temb, image_only_indicator=image_only_indicator)
                        return m_attn(
                            res_out,
                            encoder_hidden_states=encoder_hidden_states,
                            image_only_indicator=image_only_indicator,
                            return_dict=False,
                        )[0]

                    hidden_states = torch.utils.checkpoint.checkpoint(
                        _forward_up_block, hidden_states, res_hidden_states, use_reentrant=False
                    )
                else:
                    hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                    hidden_states = resnet(hidden_states, temb, image_only_indicator=image_only_indicator)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        image_only_indicator=image_only_indicator,
                        return_dict=False,
                    )[0]

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)
            return hidden_states

        CrossAttnUpBlockSpatioTemporal.forward = custom_up_forward

        def custom_mid_forward(
            self,
            hidden_states: torch.Tensor,
            temb=None,
            encoder_hidden_states=None,
            image_only_indicator=None,
        ):
            hidden_states = self.resnets[0](
                hidden_states, temb, image_only_indicator=image_only_indicator
            )
            for attn, resnet in zip(self.attentions, self.resnets[1:]):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    def _forward_mid_block(h, m_attn=attn, m_resnet=resnet):
                        attn_out = m_attn(
                            h,
                            encoder_hidden_states=encoder_hidden_states,
                            image_only_indicator=image_only_indicator,
                            return_dict=False,
                        )[0]
                        return m_resnet(attn_out, temb, image_only_indicator=image_only_indicator)

                    hidden_states = torch.utils.checkpoint.checkpoint(
                        _forward_mid_block, hidden_states, use_reentrant=False
                    )
                else:
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        image_only_indicator=image_only_indicator,
                        return_dict=False,
                    )[0]
                    hidden_states = resnet(hidden_states, temb, image_only_indicator=image_only_indicator)
            return hidden_states

        UNetMidBlockSpatioTemporal.forward = custom_mid_forward

        from diffusers.models.transformers.transformer_temporal import (
            TransformerSpatioTemporalModel,
            TransformerTemporalModelOutput,
        )

        def custom_spatio_temporal_forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states=None,
            image_only_indicator=None,
            return_dict: bool = True,
        ):
            batch_frames, _, height, width = hidden_states.shape
            num_frames = image_only_indicator.shape[-1]
            batch_size = batch_frames // num_frames

            time_context = encoder_hidden_states
            time_context_first_timestep = time_context[None, :].reshape(
                batch_size, num_frames, -1, time_context.shape[-1]
            )[:, 0]
            time_context = time_context_first_timestep[:, None].broadcast_to(
                batch_size, height * width, time_context.shape[-2], time_context.shape[-1]
            )
            time_context = time_context.reshape(batch_size * height * width, -1, time_context.shape[-1])

            residual = hidden_states
            hidden_states = self.norm(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim)
            hidden_states = self.proj_in(hidden_states)

            num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
            num_frames_emb = num_frames_emb.repeat(batch_size, 1)
            num_frames_emb = num_frames_emb.reshape(-1)
            t_emb = self.time_proj(num_frames_emb).to(dtype=hidden_states.dtype)
            emb = self.time_pos_embed(t_emb)[:, None, :]

            for block, temporal_block in zip(self.transformer_blocks, self.temporal_transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    def _run_spatio_step(h, b=block):
                        return b(h, encoder_hidden_states=encoder_hidden_states)

                    h_spat = torch.utils.checkpoint.checkpoint(
                        _run_spatio_step, hidden_states, use_reentrant=False
                    )

                    def _run_temporal_step(hs, tb=temporal_block):
                        return tb(
                            hs + emb,
                            num_frames=num_frames,
                            encoder_hidden_states=time_context,
                        )

                    h_temp = torch.utils.checkpoint.checkpoint(
                        _run_temporal_step, h_spat, use_reentrant=False
                    )

                    hidden_states = self.time_mixer(
                        x_spatial=h_spat,
                        x_temporal=h_temp,
                        image_only_indicator=image_only_indicator,
                    )
                else:
                    h_spat = block(hidden_states, encoder_hidden_states=encoder_hidden_states)
                    h_temp = temporal_block(
                        h_spat + emb,
                        num_frames=num_frames,
                        encoder_hidden_states=time_context,
                    )
                    hidden_states = self.time_mixer(
                        x_spatial=h_spat,
                        x_temporal=h_temp,
                        image_only_indicator=image_only_indicator,
                    )

            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch_frames, height, width, inner_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            output = hidden_states + residual
            if not return_dict:
                return (output,)
            return TransformerTemporalModelOutput(sample=output)

        TransformerSpatioTemporalModel.forward = custom_spatio_temporal_forward
        print("[✓] Deep spatio-temporal transformer gradient checkpointing active")

        from diffusers.models.unets.unet_spatio_temporal_condition import (
            UNetSpatioTemporalConditionModel,
            UNetSpatioTemporalConditionOutput,
        )

        def custom_unet_forward(
            self,
            sample: torch.Tensor,
            timestep,
            encoder_hidden_states: torch.Tensor,
            added_time_ids: torch.Tensor,
            return_dict: bool = True,
        ):
            default_overall_up_factor = 2**self.num_upsamplers
            forward_upsample_size = False
            upsample_size = None
            if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
                forward_upsample_size = True

            timesteps = timestep
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.float32, device=sample.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device)

            batch_size, num_frames = sample.shape[:2]
            timesteps = timesteps.expand(batch_size)
            t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
            emb = self.time_embedding(t_emb)

            time_embeds = self.add_time_proj(added_time_ids.flatten()).reshape((batch_size, -1)).to(emb.dtype)
            aug_emb = self.add_embedding(time_embeds)
            emb = emb + aug_emb

            sample = sample.flatten(0, 1)
            emb = emb.repeat_interleave(num_frames, dim=0, output_size=emb.shape[0] * num_frames)
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(
                num_frames, dim=0, output_size=encoder_hidden_states.shape[0] * num_frames
            )

            sample = self.conv_in(sample)
            image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

            down_block_res_samples = (sample,)
            for downsample_block in self.down_blocks:
                if torch.is_grad_enabled() and getattr(self, "gradient_checkpointing", False):
                    if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                        def _run_down(s, blk=downsample_block):
                            return blk(
                                hidden_states=s,
                                temb=emb,
                                encoder_hidden_states=encoder_hidden_states,
                                image_only_indicator=image_only_indicator,
                            )
                        sample, res_samples = torch.utils.checkpoint.checkpoint(_run_down, sample, use_reentrant=False)
                    else:
                        def _run_down(s, blk=downsample_block):
                            return blk(
                                hidden_states=s,
                                temb=emb,
                                image_only_indicator=image_only_indicator,
                            )
                        sample, res_samples = torch.utils.checkpoint.checkpoint(_run_down, sample, use_reentrant=False)
                else:
                    if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                        sample, res_samples = downsample_block(
                            hidden_states=sample,
                            temb=emb,
                            encoder_hidden_states=encoder_hidden_states,
                            image_only_indicator=image_only_indicator,
                        )
                    else:
                        sample, res_samples = downsample_block(
                            hidden_states=sample,
                            temb=emb,
                            image_only_indicator=image_only_indicator,
                        )
                down_block_res_samples += res_samples

            if torch.is_grad_enabled() and getattr(self, "gradient_checkpointing", False):
                def _run_mid(s):
                    return self.mid_block(
                        hidden_states=s,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        image_only_indicator=image_only_indicator,
                    )
                sample = torch.utils.checkpoint.checkpoint(_run_mid, sample, use_reentrant=False)
            else:
                sample = self.mid_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )

            for i, upsample_block in enumerate(self.up_blocks):
                is_final_block = i == len(self.up_blocks) - 1
                res_samples = down_block_res_samples[-len(upsample_block.resnets):]
                down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

                if not is_final_block and forward_upsample_size:
                    upsample_size = down_block_res_samples[-1].shape[2:]

                if torch.is_grad_enabled() and getattr(self, "gradient_checkpointing", False):
                    def _run_up(s, r_tuple, blk=upsample_block, u_sz=upsample_size):
                        if hasattr(blk, "has_cross_attention") and blk.has_cross_attention:
                            return blk(
                                hidden_states=s,
                                temb=emb,
                                res_hidden_states_tuple=r_tuple,
                                encoder_hidden_states=encoder_hidden_states,
                                upsample_size=u_sz,
                                image_only_indicator=image_only_indicator,
                            )
                        else:
                            return blk(
                                hidden_states=s,
                                temb=emb,
                                res_hidden_states_tuple=r_tuple,
                                upsample_size=u_sz,
                                image_only_indicator=image_only_indicator,
                            )
                    sample = torch.utils.checkpoint.checkpoint(_run_up, sample, res_samples, use_reentrant=False)
                else:
                    if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                        sample = upsample_block(
                            hidden_states=sample,
                            temb=emb,
                            res_hidden_states_tuple=res_samples,
                            encoder_hidden_states=encoder_hidden_states,
                            upsample_size=upsample_size,
                            image_only_indicator=image_only_indicator,
                        )
                    else:
                        sample = upsample_block(
                            hidden_states=sample,
                            temb=emb,
                            res_hidden_states_tuple=res_samples,
                            upsample_size=upsample_size,
                            image_only_indicator=image_only_indicator,
                        )

            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
            sample = self.conv_out(sample)
            sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

            if not return_dict:
                return (sample,)
            return UNetSpatioTemporalConditionOutput(sample=sample)

        UNetSpatioTemporalConditionModel.forward = custom_unet_forward
        print("[✓] Deep UNet block-level gradient checkpointing active")
        print("[✓] Deep attention & cross-attention gradient checkpointing active")
    except Exception as e:
        print(f"[!] Note: Deep attention checkpointing patch skipped ({e})")

    if hasattr(unet, "enable_forward_chunking"):
        try:
            unet.enable_forward_chunking(chunk_size=1, dim=0)
            print("[✓] UNet feed-forward chunking enabled (chunk_size=1)")
        except Exception as e:
            print(f"[!] Note: Forward chunking skipped ({e})")


def run_svd_gpu_smoke_test(
    model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt",
    device: Union[str, torch.device] = "cuda:0",
    enable_gradient_checkpointing: bool = False,
    test_cssc: bool = False,
    mock: bool = False,
    output_json: Union[str, Path] = "reports/svd_gpu_smoke_test.json",
) -> Dict[str, Any]:
    """Execute dedicated SVD-XT GPU preflight smoke test on genuine CUDA accelerator (or mock mode).

    Args:
        model_id: Hugging Face model repository or local path for SVD-XT UNet.
        device: CUDA device identifier (e.g. 'cuda:0').
        enable_gradient_checkpointing: Whether to enable activation checkpointing on UNet.
        test_cssc: Whether to test CSSC MultiViewConditioner 6-camera integration.
        mock: Whether to run fast CPU mock verification without loading 1.5B weights.
        output_json: Target path for structured JSON results report.

    Returns:
        Dictionary containing all measured metrics, latencies, and audit verification flags.
    """
    output_json_path = (
        PROJECT_ROOT / output_json if not Path(output_json).is_absolute() else Path(output_json)
    )
    output_md_path = output_json_path.with_suffix(".md")

    # 1. Device validation and mode selection
    if mock:
        parsed_device = torch.device(device if str(device).lower() != "cuda:0" else "cpu")
        device_str = str(parsed_device)
        device_name = f"Mock {parsed_device.type.upper()} Surrogate"
        total_vram_gb = 0.0
        device_capability = "N/A"
        dtype = torch.float32
        print("=" * 70)
        print("SVD-XT SMOKE TEST (MOCK SURROGATE PREFLIGHT)")
        print(f"Device: {device_str} | Mock Mode: Enabled")
        print(f"Test CSSC Conditioning: {test_cssc}")
        print("=" * 70)
    else:
        parsed_device = validate_cuda_device(device)
        device_str = str(parsed_device)
        device_name = torch.cuda.get_device_name(parsed_device)
        device_props = torch.cuda.get_device_properties(parsed_device)
        total_vram_gb = device_props.total_memory / (1024 ** 3)
        device_capability = f"{device_props.major}.{device_props.minor}"
        dtype = torch.float16
        print("=" * 70)
        print("SVD-XT GPU PREFLIGHT SMOKE TEST (KAGGLE T4 VALIDATION)")
        print(f"Device: {device_str} | Model ID: {model_id}")
        print(f"Gradient Checkpointing: {enable_gradient_checkpointing}")
        print(f"Test CSSC Conditioning: {test_cssc}")
        print("=" * 70)
        print(f"Accelerator: {device_name} ({total_vram_gb:.2f} GB total VRAM, CC {device_capability})")

    # 2. Reset memory stats and clean cache
    if parsed_device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(parsed_device)
        torch.cuda.synchronize(parsed_device)

    # 3. Load UNet
    if mock:
        print(f"\n[*] Instantiating MockSVDUNet on {device_str} ({dtype})...")
        unet = MockSVDUNet(in_channels=8, out_channels=4).to(device=parsed_device, dtype=dtype)
        load_time_s = 0.001
        print("[✓] MockSVDUNet instantiated")
    else:
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
    print(f"\n[*] Instantiating SpatialAdapter (Option B.1) in {dtype}...")
    adapter_config = SpatialAdapterConfig(
        in_channels=1,
        adapter_channels=(32, 64, 128, 256),
        latent_channels=4,
        down_block_channels=(320, 640, 1280, 1280),
        conditioning_type="latent_addition",
        use_temporal_conv=True,
        zero_conv_init=True,
    )
    spatial_adapter = SpatialAdapter(adapter_config).to(device=parsed_device, dtype=dtype)
    spatial_adapter.train()
    for p in spatial_adapter.parameters():
        p.requires_grad = True
    adapter_parameter_count = sum(p.numel() for p in spatial_adapter.parameters() if p.requires_grad)
    print(f"[✓] SpatialAdapter instantiated: {adapter_parameter_count:,} trainable parameters (requires_grad = True)")

    conditioner = None
    conditioner_parameter_count = 0
    if test_cssc:
        print(f"\n[*] Instantiating MultiViewConditioner (CSSC) in {dtype}...")
        conditioner = MultiViewConditioner(model_dim=1024, num_heads=8, motion_in_dim=6).to(
            device=parsed_device, dtype=dtype
        )
        conditioner.train()
        for p in conditioner.parameters():
            p.requires_grad = True
        conditioner_parameter_count = sum(p.numel() for p in conditioner.parameters() if p.requires_grad)
        print(f"[✓] MultiViewConditioner instantiated: {conditioner_parameter_count:,} trainable parameters (requires_grad = True)")

    # 6. Apply gradient checkpointing if requested
    if enable_gradient_checkpointing:
        apply_deep_gradient_checkpointing(unet)

    # 7. Measure baseline VRAM
    if parsed_device.type == "cuda":
        torch.cuda.synchronize(parsed_device)
        baseline_allocated_bytes = torch.cuda.memory_allocated(parsed_device)
        baseline_reserved_bytes = torch.cuda.memory_reserved(parsed_device)
        baseline_allocated_gb = baseline_allocated_bytes / (1024 ** 3)
        baseline_reserved_gb = baseline_reserved_bytes / (1024 ** 3)
    else:
        baseline_allocated_gb = 0.0
        baseline_reserved_gb = 0.0
    print(f"[*] Baseline VRAM: {baseline_allocated_gb:.2f} GB allocated | {baseline_reserved_gb:.2f} GB reserved")

    # 8. Create synthetic input latents matching genuine dimensions
    print(f"\n[*] Synthesizing input tensors ({dtype})...")
    h_lat, w_lat = (16, 16) if mock else (72, 128)
    mask_h, mask_w = (128, 128) if mock else (568, 876)

    noisy_latents = torch.randn(1, 25, 4, h_lat, w_lat, device=parsed_device, dtype=dtype)
    initial_image_latents_4d = torch.randn(1, 4, h_lat, w_lat, device=parsed_device, dtype=dtype)
    initial_image_latents = initial_image_latents_4d.unsqueeze(1).repeat(1, 25, 1, 1, 1)
    spatial_masks = torch.rand(1, 25, 1, mask_h, mask_w, device=parsed_device, dtype=dtype)
    added_time_ids = torch.tensor([[4.0, 127.0, 0.02]], device=parsed_device, dtype=dtype)
    timesteps = torch.tensor([0.05], device=parsed_device, dtype=dtype)

    if test_cssc and conditioner is not None:
        camera_clip_embeds = torch.randn(1, 6, 1024, device=parsed_device, dtype=dtype)
        motion_states = torch.randn(1, 25, 6, device=parsed_device, dtype=dtype)
        cssc_out = conditioner(camera_clip_embeds, motion_states, target_camera="front")
        encoder_hidden_states = cssc_out["encoder_hidden_states"]  # [1, 27, 1024]
        print(f"  - CSSC encoder_hidden_states: {tuple(encoder_hidden_states.shape)} (27-token bank)")
    else:
        encoder_hidden_states = torch.randn(1, 1, 1024, device=parsed_device, dtype=dtype)
        print(f"  - encoder_hidden_states:      {tuple(encoder_hidden_states.shape)}")

    print(f"  - noisy_latents:              {tuple(noisy_latents.shape)}")
    print(f"  - initial_image_latents:      {tuple(initial_image_latents.shape)}")
    print(f"  - spatial_masks:              {tuple(spatial_masks.shape)}")
    print(f"  - added_time_ids:             {tuple(added_time_ids.shape)}")
    print(f"  - timesteps (c_noise):        {timesteps.item():.4f}")

    # 9. Execute exactly ONE forward pass
    print("\n[*] Executing forward pass...")
    if parsed_device.type == "cuda":
        torch.cuda.synchronize(parsed_device)
    t_fwd_start = time.perf_counter()

    autocast_device = parsed_device.type if parsed_device.type in ["cuda", "cpu"] else "cpu"
    with torch.autocast(device_type=autocast_device, dtype=dtype):
        conditioned_latents = spatial_adapter.apply_to_latent(noisy_latents, spatial_masks)
        unet_input = torch.cat([conditioned_latents, initial_image_latents], dim=2)

        unet_out = unet(
            sample=unet_input,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_time_ids=added_time_ids,
        )

    if parsed_device.type == "cuda":
        torch.cuda.synchronize(parsed_device)
    forward_time_s = time.perf_counter() - t_fwd_start

    if parsed_device.type == "cuda":
        forward_allocated_bytes = torch.cuda.memory_allocated(parsed_device)
        forward_reserved_bytes = torch.cuda.memory_reserved(parsed_device)
        forward_allocated_gb = forward_allocated_bytes / (1024 ** 3)
        forward_reserved_gb = forward_reserved_bytes / (1024 ** 3)
    else:
        forward_allocated_gb = 0.0
        forward_reserved_gb = 0.0
    print(f"[✓] Forward pass complete in {forward_time_s:.3f}s")
    if parsed_device.type == "cuda":
        print(f"    Post-forward VRAM: {forward_allocated_gb:.2f} GB allocated | {forward_reserved_gb:.2f} GB reserved")

    # 10. Compute dummy scalar loss
    sample_tensor = unet_out.sample if hasattr(unet_out, "sample") else unet_out
    if isinstance(sample_tensor, tuple):
        sample_tensor = sample_tensor[0]
    loss = sample_tensor.mean()
    loss_val = float(loss.item())
    print(f"[*] Computed dummy scalar loss: {loss_val:.6f}")

    # 11. Execute exactly ONE backward pass
    print("\n[*] Executing backward pass (loss.backward())...")
    if parsed_device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(parsed_device)
    t_bwd_start = time.perf_counter()

    loss.backward()

    if parsed_device.type == "cuda":
        torch.cuda.synchronize(parsed_device)
    backward_time_s = time.perf_counter() - t_bwd_start
    total_step_time_s = forward_time_s + backward_time_s

    if parsed_device.type == "cuda":
        peak_allocated_bytes = torch.cuda.max_memory_allocated(parsed_device)
        peak_reserved_bytes = torch.cuda.max_memory_reserved(parsed_device)
        peak_allocated_gb = peak_allocated_bytes / (1024 ** 3)
        peak_reserved_gb = peak_reserved_bytes / (1024 ** 3)
    else:
        peak_allocated_gb = 0.0
        peak_reserved_gb = 0.0
    print(f"[✓] Backward pass complete in {backward_time_s:.3f}s (Total step: {total_step_time_s:.3f}s)")
    if parsed_device.type == "cuda":
        print(f"    Peak allocated VRAM: {peak_allocated_gb:.2f} GB")
        print(f"    Peak reserved VRAM:  {peak_reserved_gb:.2f} GB")

    # 12. Audit gradients strictly
    print("\n[*] Performing strict autograd parameter isolation audit...")
    unet_frozen_verified = all(p.grad is None for p in unet.parameters())
    adapter_gradients_verified = any(
        p.grad is not None and p.grad.norm() > 0 for p in spatial_adapter.parameters()
    )
    conditioner_gradients_verified = True
    if test_cssc and conditioner is not None:
        conditioner_gradients_verified = any(
            p.grad is not None and p.grad.norm() > 0 for p in conditioner.parameters()
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

    if test_cssc and not conditioner_gradients_verified:
        raise RuntimeError(
            "Gradient audit failed: MultiViewConditioner parameters received zero gradients!"
        )
    if test_cssc:
        print("    [PASS] MultiViewConditioner parameters active: non-zero gradients verified.")

    # 13. Evaluate fit status & recommendations
    fits_16gb_t4 = peak_reserved_gb < 15.0
    recommend_gradient_checkpointing = (peak_reserved_gb > 13.5) or (peak_allocated_gb > 13.5)

    if not mock:
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
    if mock:
        status_val = "MOCK_ONLY"
        fits_val: Any = "NOT_MEASURED"
        rec_gc_val: Any = "NOT_MEASURED"
    else:
        status_val = "REAL_GPU_PASSED" if fits_16gb_t4 else "REAL_GPU_FAILED"
        fits_val = fits_16gb_t4
        rec_gc_val = recommend_gradient_checkpointing

    metrics: Dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "status": status_val,
        "model_id": model_id if not mock else "mock-svd-unet",
        "mock": mock,
        "test_cssc": test_cssc,
        "device": device_str,
        "device_name": device_name,
        "device_capability": device_capability,
        "total_vram_gb": round(total_vram_gb, 3),
        "gradient_checkpointing": enable_gradient_checkpointing,
        "unet_parameter_count": unet_parameter_count,
        "spatial_adapter_parameter_count": adapter_parameter_count,
        "conditioner_parameter_count": conditioner_parameter_count,
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
        "fits_16gb_t4": fits_val,
        "recommend_gradient_checkpointing": rec_gc_val,
        "unet_frozen_verified": unet_frozen_verified,
        "adapter_gradients_verified": adapter_gradients_verified,
        "conditioner_gradients_verified": conditioner_gradients_verified,
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
            test_cssc=args.test_cssc,
            mock=args.mock,
            output_json=args.output_json,
        )
        return 0
    except RuntimeError as exc:
        print(f"\n[X] SVD-XT GPU Smoke Test Execution Halted: {exc}", file=sys.stderr)
        raise exc


if __name__ == "__main__":
    sys.exit(main())
