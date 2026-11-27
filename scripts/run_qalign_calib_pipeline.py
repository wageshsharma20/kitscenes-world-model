#!/usr/bin/env python3
"""End-to-end GPU calibration runner (Colab & Kaggle compatible).

Runs:
1. Environment & GPU verification + `--no-deps simple-lama-inpainting` setup
2. Validation multi-slice extraction (`--extract-val`) unless `--val-tar`/`--val-dir` provided
3. Rendering candidate variants via `scripts/eval_world_model_qalign.py --stage render`
4. Isolated `transformers==4.36.1` venv setup (via `virtualenv --system-site-packages`)
5. Official 1fps `q-future/one-align` scoring via `--stage score`
6. Diagnostic summary of lowest-scoring videos + sample JPEGs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Optional, Sequence

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: Sequence[str], check: bool = True) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(list(cmd), check=check)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run full Q-Align validation calibration pipeline.")
    parser.add_argument("--work-dir", type=Path, default=Path("/content/kitscenes_work"))
    parser.add_argument("--val-tar", type=Path, default=None)
    parser.add_argument("--val-dir", type=Path, default=None)
    parser.add_argument("--variants", type=str, default="cand_a_baseline,cand_b_scale1,cand_c_adaptive,cand_d_photometric,cand_e_selective")
    parser.add_argument("--crf", type=int, default=14)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--scenarios", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--score-only", action="store_true", help="Skip val extraction & video rendering; score existing render_manifest.json")
    args = parser.parse_args(argv)

    _run(["nvidia-smi"], check=False)
    import torch

    print(
        f"PyTorch={torch.__version__} Python={sys.version.split()[0]} "
        f"CUDA={torch.cuda.is_available()} GPUs={torch.cuda.device_count()}",
        flush=True,
    )
    for idx in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(idx)
        print(f"  GPU {idx}: {p.name} ({p.total_memory / (1024 ** 3):.2f} GiB)", flush=True)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    eval_script = PROJECT_ROOT / "scripts" / "eval_world_model_qalign.py"
    extract_script = PROJECT_ROOT / "scripts" / "extract_multiframe_observations.py"
    out_dir = args.work_dir / "qalign_calib"
    out_dir.mkdir(parents=True, exist_ok=True)
    variants_list = [v.strip() for v in args.variants.split(",") if v.strip()]

    if not args.score_only:
        _run(["bash", "-lc", "which ffmpeg >/dev/null || (apt-get -qq update && apt-get -qq install -y ffmpeg)"])
        _run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "simple-lama-inpainting"])
        _run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "pyarrow", "virtualenv"])

        val_dir = args.val_dir
        if val_dir is None and args.val_tar is None:
            tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
            if not tok:
                raise RuntimeError("HF_TOKEN env var required when neither --val-dir nor --val-tar is provided.")
            val_dir = args.work_dir / "val_cache"
            _run(
                [
                    sys.executable,
                    str(extract_script),
                    "--out-root",
                    str(val_dir),
                    "--workers",
                    "4",
                    "--extract-val",
                ]
            )

        render_cmd = [
            sys.executable,
            str(eval_script),
            "--stage",
            "render",
            "--out-dir",
            str(out_dir),
            "--variants",
            args.variants,
            "--crf",
            str(args.crf),
            "--device",
            args.device,
        ]
        if val_dir is not None:
            render_cmd.extend(["--val-dir", str(val_dir)])
        if args.val_tar is not None:
            render_cmd.extend(["--val-tar", str(args.val_tar)])
        if args.scenarios.strip():
            render_cmd.extend(["--scenarios", args.scenarios.strip()])
        if args.limit > 0:
            render_cmd.extend(["--limit", str(args.limit)])
        _run(render_cmd)

        # Save lightweight f24 sample JPEGs across diverse slices for visual inspection
        samples_dir = out_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        for sid, tag in [("0000900", "snow"), ("0000902", "turn"), ("0000903", "rain"), ("0000904", "constr"), ("0000906", "night")]:
            for var in variants_list:
                for cam in ("front", "rear", "front_left"):
                    mp4 = out_dir / "videos" / var / sid / f"{cam}.mp4"
                    if not mp4.exists():
                        continue
                    cap = cv2.VideoCapture(str(mp4))
                    last_fr = None
                    while True:
                        ok, fr = cap.read()
                        if not ok:
                            break
                        last_fr = fr
                    cap.release()
                    if last_fr is not None:
                        cv2.imwrite(str(samples_dir / f"{sid}_{tag}_{var}_{cam}_f24.jpg"), last_fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    else:
        _run([sys.executable, "-m", "pip", "install", "-q", "virtualenv"])

    # Create isolated venv with transformers==4.36.1 + CUDA 13 compatible bitsandbytes using virtualenv
    venv_dir = Path("/tmp/qalign_venv")
    if not (venv_dir / "bin" / "python").exists():
        _run([sys.executable, "-m", "virtualenv", "--system-site-packages", str(venv_dir)])
    q_pip = str(venv_dir / "bin" / "pip")
    q_py = str(venv_dir / "bin" / "python")
    _run([
        q_pip,
        "install",
        "-q",
        "--no-deps",
        "--force-reinstall",
        "transformers==4.36.1",
        "tokenizers==0.15.2",
        "huggingface-hub==0.25.2",
        "accelerate==0.29.3",
        "bitsandbytes>=0.50.2",
    ])
    _run([q_pip, "install", "-q", "icecream", "sentencepiece", "protobuf"])

    _run([q_py, str(eval_script), "--stage", "score", "--out-dir", str(out_dir)])

    scores = json.loads((out_dir / "qalign_scores.json").read_text())
    best_var = max(
        variants_list,
        key=lambda v: json.loads((out_dir / "calibration_summary.json").read_text())[v]["pq_norm_unit"],
    )
    print(f"\n--- Lowest 10 Per-Camera PQ Videos for `{best_var}` ---", flush=True)
    rows = []
    for _k, rec in scores.items():
        if rec.get("variant") == best_var:
            gt_q = float(scores[f"gt__{rec['scenario_id']}__{rec['camera']}"]["q_raw"])
            pr_q = float(rec["q_raw"])
            rel = min(100.0, 100.0 * max(pr_q - 1.0, 0.0) / max(gt_q - 1.0, 1e-4))
            rows.append((rel, rec["scenario_id"], rec["scenario_type"], rec["camera"], pr_q, gt_q, rec.get("psnr")))
    rows.sort()
    for rel, sid, st, cam, pq, gq, psnr in rows[:10]:
        print(f"  {sid} ({st[:14]:14s}) {cam:11s}: PQ={rel:5.2f} (q={pq:.3f} vs gt={gq:.3f}, psnr={psnr})", flush=True)

    # Free uncompressed temp MP4s only if requested? Keep videos on Colab 112GB disk in case we inspect, just clean _unpacked
    shutil.rmtree(out_dir / "_unpacked", ignore_errors=True)
    print("\nCALIBRATION COMPLETE! Summary JSON:", flush=True)
    print((out_dir / "calibration_summary.json").read_text(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
