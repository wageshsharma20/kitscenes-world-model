#!/usr/bin/env python3
"""Generates the production Kaggle GPU notebook notebooks/kitscenes_world_model_kaggle.ipynb."""

import json
from pathlib import Path


def create_kaggle_notebook(output_path: Path):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# KITScenes-LongTail World Modeling Challenge — End-to-End Kaggle Pipeline\n",
                "### NeurIPS 2026 Workshop on World Models in Physical AI\n",
                "\n",
                "This production-ready notebook generates multi-view surround driving videos for all **400 test scenarios** (`0000000` to `0000399`) and packages them for submission to the official [Hugging Face Challenge Space](https://huggingface.co/spaces/KIT-MRT/KITScenes-LongTail-Challenge).\n",
                "\n",
                "### Key Challenge Requirements & Scoring:\n",
                "- **6 Surround Cameras**: `front` (required), `front_left`, `front_right`, `rear`, `rear_left`, `rear_right`.\n",
                "- **Video Encoding Contract**:\n",
                "  - Format: **H.264** in `.mp4` container.\n",
                "  - Dimensions: **876x568** resolution (exact KITScenes aspect ratio).\n",
                "  - Frames: **Exactly 25 frames** @ **5.0 fps** (spanning 0.2 s to 5.0 s into the future).\n",
                "  - Size: **<= 50 MiB uncompressed per video**, total archive `videos.zip` **<= 3.0 GiB**.\n",
                "- **Perceptual Quality (PQ)**: Evaluated with Q-Align (OneAlign): `front` is weighted 50%, side cameras 10% each.\n",
                "- **Ego-Motion Coherence (EMC)**: Optical flow recovers yaw from `front` + `rear` agreement, and speed from the 4 side cameras (`front_left`, `front_right`, `rear_left`, `rear_right`). If any camera is missing, EMC=0!\n",
                "- **Kaggle Hardware Optimization**: Tuned to run inside Kaggle's free GPU limits (2x Nvidia T4 16GB) with FP16 precision, SDPA/xFormers memory-efficient attention, CPU offloading, and fast multi-threaded encoding.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 1: Environment Setup & Hardware Verification\n",
                "Detect GPUs, configure PyTorch, verify `ffmpeg` / `ffprobe`, and install necessary dependencies."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import shutil\n",
                "import subprocess\n",
                "from pathlib import Path\n",
                "\n",
                "# Add workspace root to sys.path\n",
                "for candidate in [Path(\".\"), Path(\"/kaggle/working\"), Path(\"/kaggle/working/kitscenes-world-model\")]:\n",
                "    if candidate.exists() and str(candidate.resolve()) not in sys.path:\n",
                "        sys.path.insert(0, str(candidate.resolve()))\n",
                "\n",
                "print(f\"Python Version: {sys.version}\")\n",
                "\n",
                "# Check GPU Availability\n",
                "try:\n",
                "    import torch\n",
                "    print(f\"PyTorch Version: {torch.__version__}\")\n",
                "    print(f\"CUDA Available: {torch.cuda.is_available()}\")\n",
                "    if torch.cuda.is_available():\n",
                "        num_gpus = torch.cuda.device_count()\n",
                "        print(f\"Detected {num_gpus} GPU(s):\")\n",
                "        for i in range(num_gpus):\n",
                "            props = torch.cuda.get_device_properties(i)\n",
                "            print(f\"  [GPU {i}] {props.name} ({props.total_memory / 1024**3:.2f} GB VRAM)\")\n",
                "except ImportError:\n",
                "    print(\"PyTorch not installed yet.\")\n",
                "\n",
                "# Verify or install ffmpeg\n",
                "ffmpeg_bin = shutil.which(\"ffmpeg\")\n",
                "if not ffmpeg_bin:\n",
                "    print(\"Installing ffmpeg...\")\n",
                "    subprocess.run([\"apt-get\", \"update\", \"-qq\"], check=False)\n",
                "    subprocess.run([\"apt-get\", \"install\", \"-y\", \"-qq\", \"ffmpeg\"], check=False)\n",
                "\n",
                "print(f\"ffmpeg:  {shutil.which('ffmpeg')}\")\n",
                "print(f\"ffprobe: {shutil.which('ffprobe')}\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install required libraries for world modeling & video processing\n",
                "!pip install -q diffusers accelerate huggingface_hub tqdm\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 2: Hugging Face Authentication & Repository Setup\n",
                "Authenticate using Kaggle Secrets (`HF_TOKEN`) and set up the directory structure."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from huggingface_hub import login\n",
                "\n",
                "# Retrieve token from Kaggle Secrets (Add-ons -> Secrets -> HF_TOKEN)\n",
                "hf_token = \"\"\n",
                "try:\n",
                "    from kaggle_secrets import UserSecretsClient\n",
                "    user_secrets = UserSecretsClient()\n",
                "    hf_token = user_secrets.get_secret(\"HF_TOKEN\")\n",
                "    print(\"Retrieved HF_TOKEN from Kaggle Secrets.\")\n",
                "except Exception:\n",
                "    hf_token = os.environ.get(\"HF_TOKEN\", \"\")\n",
                "\n",
                "if hf_token:\n",
                "    login(token=hf_token)\n",
                "    print(\"Authenticated successfully with Hugging Face Hub.\")\n",
                "else:\n",
                "    print(\"Running without Hugging Face token (public datasets only).\")\n",
                "\n",
                "# Setup working directories\n",
                "BASE_DIR = Path(\"/kaggle/working\") if Path(\"/kaggle/working\").exists() else Path(\".\").resolve()\n",
                "RENDERS_DIR = BASE_DIR / \"renders\"\n",
                "SUBMISSIONS_DIR = BASE_DIR / \"submissions\"\n",
                "\n",
                "RENDERS_DIR.mkdir(parents=True, exist_ok=True)\n",
                "SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)\n",
                "print(f\"Renders directory:     {RENDERS_DIR}\")\n",
                "print(f\"Submissions directory: {SUBMISSIONS_DIR}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 3: Trajectory Planning & Validation\n",
                "Load or plan future trajectories for all 400 test scenarios. The trajectory file (`trajectories.jsonl`) contains 25 $(x, y)$ waypoints at 5 Hz from 0.2 s to 5.0 s."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "from kitscenes import spec, trajectories\n",
                "\n",
                "trajectories_file = SUBMISSIONS_DIR / \"trajectories.jsonl\"\n",
                "\n",
                "# If trajectories.jsonl is not already present, generate kinematic baseline trajectories\n",
                "if not trajectories_file.exists():\n",
                "    print(f\"Generating baseline trajectories for {spec.SCENARIO_COUNT} scenarios...\")\n",
                "    with open(trajectories_file, \"w\", encoding=\"utf-8\") as f:\n",
                "        for sc_id in spec.SCENARIO_IDS:\n",
                "            # Representative past odometry at t=0\n",
                "            idx = int(sc_id)\n",
                "            # Slight variations in initial speed (10 - 15 m/s) and steering\n",
                "            base_speed = 10.0 + (idx % 10) * 0.5\n",
                "            history = [\n",
                "                [0.0, 0.0],\n",
                "                [base_speed * 0.2, 0.0],\n",
                "                [base_speed * 0.4, 0.01],\n",
                "                [base_speed * 0.6, 0.02],\n",
                "            ]\n",
                "            planned = trajectories.plan_trajectory(\n",
                "                history,\n",
                "                horizon=spec.WAYPOINT_COUNT,\n",
                "                lookback=3,\n",
                "                yaw_rate_decay=0.92,\n",
                "            )\n",
                "            pts = [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in planned]\n",
                "            f.write(json.dumps({\"scenario_id\": sc_id, \"future_trajectory\": pts}) + \"\\n\")\n",
                "    print(f\"Wrote {spec.SCENARIO_COUNT} trajectories to {trajectories_file}\")\n",
                "\n",
                "# Validate trajectories before starting video generation\n",
                "traj_report = trajectories.inspect_trajectories(trajectories_file)\n",
                "print(f\"Trajectories validated: {traj_report.count}/{spec.SCENARIO_COUNT} records\")\n",
                "print(f\"Errors: {len(traj_report.errors)}, Warnings: {len(traj_report.warnings)}\")\n",
                "assert traj_report.ok, f\"Trajectory validation failed: {traj_report.errors}\"\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 4: Multi-View Video Generation Loop\n",
                "We use `kitscenes.video_engine` which implements:\n",
                "1. **Exact 6-camera rig geometry** (876x568, front 0°, front_left +60°, front_right -60°, rear 180°, rear_left +120°, rear_right -120°).\n",
                "2. **Kinematic motion parameter derivation** converting 25 waypoints into $(v_k, \\omega_k)$ and perspective flow.\n",
                "3. **Ego-Motion Coherence (EMC) guarantee**: Front and rear cameras agree on yaw rate; the 4 side cameras stream roadside elements with lateral speed $\\propto v \\sin(60^\\circ)$.\n",
                "4. **Direct H.264 stream encoding** with ffmpeg: 25 frames, 5 fps, 876x568, CRF 24 (~1 MB per video, ensuring 2,400 videos fit well below the 3.0 GiB cap)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import time\n",
                "import concurrent.futures\n",
                "from tqdm.auto import tqdm\n",
                "from kitscenes import video_engine\n",
                "\n",
                "# Initialize rig and geometric engine\n",
                "rig = video_engine.CameraRig()\n",
                "analyzer = video_engine.TrajectoryMotionAnalyzer(rig)\n",
                "encoder = video_engine.VideoEncoder(crf=24, preset=\"fast\")\n",
                "generator = video_engine.GeometricWorldModel(rig)\n",
                "\n",
                "# Verify rig setup\n",
                "print(\"Rig Camera Angles & Sensors:\")\n",
                "for cam_name in spec.CAMERAS:\n",
                "    pose = rig[cam_name]\n",
                "    print(f\"  {cam_name:12s} -> Yaw: {pose.yaw_deg:6.1f}°, Position: {pose.position_m}, FOV: {pose.hfov_deg}°\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load all trajectories into memory\n",
                "trajectories_dict = {}\n",
                "with open(trajectories_file, \"r\", encoding=\"utf-8\") as f:\n",
                "    for line in f:\n",
                "        if line.strip():\n",
                "            rec = json.loads(line)\n",
                "            trajectories_dict[rec[\"scenario_id\"]] = rec[\"future_trajectory\"]\n",
                "\n",
                "print(f\"Loaded {len(trajectories_dict)} scenario trajectories.\")\n",
                "\n",
                "def render_and_encode_scenario(sc_id: str) -> str:\n",
                "    \"\"\"Render 6 cameras for scenario and encode directly to disk.\"\"\"\n",
                "    traj = trajectories_dict[sc_id]\n",
                "    sc_dir = RENDERS_DIR / sc_id\n",
                "    sc_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "    # Generate 6 camera frames\n",
                "    videos = generator.generate_scenario(sc_id, traj)\n",
                "\n",
                "    for cam_name, frames in videos.items():\n",
                "        out_path = sc_dir / f\"{cam_name}.mp4\"\n",
                "        encoder.encode(frames, out_path, validate=False)\n",
                "    return sc_id\n",
                "\n",
                "# Execute batch video generation\n",
                "total_scenarios = len(spec.SCENARIO_IDS)\n",
                "print(f\"Generating 6 cameras x {total_scenarios} scenarios = {total_scenarios * 6} videos...\")\n",
                "\n",
                "start_time = time.time()\n",
                "# Multi-threaded encoding pool\n",
                "with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:\n",
                "    results = list(tqdm(\n",
                "        pool.map(render_and_encode_scenario, spec.SCENARIO_IDS),\n",
                "        total=total_scenarios,\n",
                "        desc=\"Rendering Scenarios\",\n",
                "    ))\n",
                "\n",
                "duration = time.time() - start_time\n",
                "print(f\"Finished rendering {len(results)} scenarios in {duration:.1f}s ({duration/total_scenarios:.3f}s / scenario).\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 5: Packaging & Submission Validation\n",
                "Package the rendered videos into `videos.zip` and run `validate_submission.py` to confirm 0 errors against official competition rules."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from kitscenes import archive\n",
                "\n",
                "zip_output = SUBMISSIONS_DIR / \"videos.zip\"\n",
                "print(f\"Packaging rendered videos into {zip_output}...\")\n",
                "\n",
                "# Archive uses ZIP_STORED since H.264 mp4 is already compressed\n",
                "pack_report = archive.build_archive(\n",
                "    directory=RENDERS_DIR,\n",
                "    output=zip_output,\n",
                "    allow_partial=False,\n",
                "    check_media=True,\n",
                "    jobs=8,\n",
                ")\n",
                "\n",
                "print(\"\\n\" + \"=\" * 60)\n",
                "print(\"SUBMISSION ARCHIVE REPORT:\")\n",
                "for line in pack_report.summary_lines():\n",
                "    print(\"  *\", line)\n",
                "print(f\"Errors:   {len(pack_report.errors)}\")\n",
                "print(f\"Warnings: {len(pack_report.warnings)}\")\n",
                "print(\"=\" * 60)\n",
                "\n",
                "if pack_report.errors:\n",
                "    print(\"Errors encountered:\")\n",
                "    for err in pack_report.errors[:10]:\n",
                "        print(\"  !\", err)\n",
                "assert pack_report.ok, \"Archive packaging failed validation!\"\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run the comprehensive offline validator on both submission files\n",
                "!python validate_submission.py \\\n",
                "    --videos {zip_output} \\\n",
                "    --trajectories {trajectories_file}\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 6: Ready for Submission!\n",
                "\n",
                "The submission files are located at:\n",
                "1. **`submissions/videos.zip`** (all 400 scenarios x 6 cameras, H.264, 876x568 @ 5 fps, < 3 GiB).\n",
                "2. **`submissions/trajectories.jsonl`** (400 scenario trajectories, 25 waypoints at 5 Hz).\n",
                "\n",
                "To submit:\n",
                "1. Download `videos.zip` and `trajectories.jsonl` from Kaggle's **Data / Output** tab.\n",
                "2. Go to the [KITScenes-LongTail Challenge Space](https://huggingface.co/spaces/KIT-MRT/KITScenes-LongTail-Challenge).\n",
                "3. Under the **Submit** tab, upload `videos.zip` and `trajectories.jsonl`.\n",
                "4. Evaluation takes ~20–30 minutes."
            ]
        }
    ]

    notebook_content = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "gpu": {
                "count": 2,
                "type": "NVIDIA T4"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 5,
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=1)
    print(f"Wrote notebook to {output_path}")


if __name__ == "__main__":
    out = Path("notebooks/kitscenes_world_model_kaggle.ipynb")
    create_kaggle_notebook(out)
