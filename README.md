# KITScenes-LongTail World Modeling Challenge — submission toolkit

Tooling for the [KITScenes-LongTail World Modeling Challenge](https://huggingface.co/spaces/KIT-MRT/KITScenes-LongTail-Challenge)
at the NeurIPS 2026 workshop on [World Models in Physical AI](https://www.worldmodels-physicalai.com).

The task: given the observed past of a rare driving event, generate the next 5 seconds of
video for all six surround cameras (25 frames at 5 fps, starting 0.2 s into the future) and,
optionally, the ego-trajectory those videos imply.

This repository does not contain a world model. It contains the part around one: the
submission format, the checks that keep a broken file out of the evaluation queue, and a
trajectory baseline to build on. The video generator is a separate piece of work.

## Contents

| Path | Purpose |
| --- | --- |
| `validate_submission.py` | Offline check of `videos.zip` and `trajectories.jsonl` |
| `build_submission.py` | Packs rendered videos; plans trajectories from odometry |
| `scripts/generate_videos.py` | CLI to generate surround videos from trajectories & package archive |
| `notebooks/kitscenes_world_model_kaggle.ipynb` | Self-contained Kaggle notebook optimized for 2x Nvidia T4 16GB GPUs |
| `kitscenes/video_engine.py` | Multi-view generation engine (camera rig, motion derivation, SVD/Geometric WM, H.264 encoder) |
| `kitscenes/spec.py` | Every limit and constant from the official rules, in one place |
| `kitscenes/media.py` | ffprobe wrapper and per-video encoding checks |
| `kitscenes/archive.py` | Archive validation (server-equivalent) and packaging |
| `kitscenes/trajectories.py` | Trajectory validation and the baseline planner |
| `kitscenes/vlm_planner.py` | Gemini tactical reasoning & kinematic trajectory planner |
| `kitscenes/pq_proxy.py` | Local perceptual-quality estimate (ffmpeg + numpy, no GPU) |
| `kitscenes/reporting.py` | Console output shared by command-line tools |
| `SPEC.md` | Condensed rules, metric definitions and scoring formulas |
| `tests/` | Test suite, covering geometry, kinematics, encoding, archive & trajectories |

## Setup

Python 3.9 or newer, plus `ffmpeg` on `PATH` for the encoding checks:

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
brew install ffmpeg          # or: apt-get install ffmpeg
```

## Usage

```sh
# 1. Plan trajectories from observed ego-motion
# Option A: Baseline kinematic extrapolation
python build_submission.py plan --history history.jsonl --out submissions/trajectories.jsonl

# Option B: Gemini tactical reasoning (instruction conditioned; text-only, see below)
python build_submission.py plan --history history.jsonl --out submissions/trajectories.jsonl --vlm

# 2. Generate the six camera views and package directly into videos.zip
python scripts/generate_videos.py \
    --trajectories submissions/trajectories.jsonl \
    --output-dir renders \
    --zip-out submissions/videos.zip

# Or pack an existing directory of renders:
# python build_submission.py package --videos-dir renders --out submissions/videos.zip

# 3. Check both files before uploading
python validate_submission.py \
    --videos submissions/videos.zip \
    --trajectories submissions/trajectories.jsonl
```

`--history` is JSON Lines with one record per scenario:

```json
{"scenario_id": "0000000", "past_trajectory": [[0.0, 0.0], [1.98, 0.01], [3.95, 0.03], [5.91, 0.08]]}
```

Four points is the minimum; more than about ten adds nothing, because the planner only
reads the last five intervals. Coordinates must already be in the ego frame of the
reference trajectories — the frame the dataset's `trajectory` column uses.

Exit codes: `0` valid, `1` invalid, `2` the check could not run (missing ffprobe, unreadable
file). `validate_submission.py --json report.json` writes the findings for CI.

## What the checks cover

The archive checks reproduce the upload validation in the challenge Space's own
`src/utils.py` (file naming, duplicate videos, per-video size, the 50 MiB uncompressed cap,
which entries are ignored) and then go further, because that validation does not decode
anything: a video with the wrong frame count, frame rate or codec is accepted by the
server and then scores **0** for both PQ and EMC.

| Check | Severity | Why |
| --- | --- | --- |
| `<scenario_id>/<camera>.mp4` naming, no duplicates, no stray files | error | rejected at upload |
| Archive ≤ 3 GiB, video ≤ 50 MiB | error | rejected at upload |
| `front.mp4` present for all 400 scenarios | error | rejected at upload |
| H.264, exactly 25 frames, 5 fps | error | scenario scores 0 for PQ and EMC |
| Missing side cameras | warning | EMC is 0 and PQ is capped (front-only tops out at 50 %) |
| Aspect ratio away from 876×568 | warning | EMC recovers ego-motion from the camera geometry |
| 400 records, ids, 25 finite pairs each | error | rejected at upload |
| Stationary, repeated, erratic or implausible trajectories | warning | costs MMS points |

`--skip-media` runs the layout checks only, which is what you want while iterating on
400 scenarios. Decoding is the slow part: one ffprobe call per video, spread over
`--jobs` threads.

## The baseline planner

`plan` extrapolates constant speed with a geometrically decaying yaw rate, with
acceleration estimated from the recent past and clamped. That is the motion a driver
produces while holding a steering angle, and it keeps both MMS comfort penalties (jerk
above the reference by 44 %, tortuosity above it by 6 %) satisfied by construction, so a
submission built only on this scores the unmatched-trajectory floor of 3.5 rather than
falling below it. It has no notion of the instruction, the scene or other traffic, and it
cannot see the reference behaviours it is scored against. Treat it as a valid lower bound
and a shape reference for the generated video, not as a competitive entry.

Its known weakness is horizon decay: real drivers brake, yield and turn at
intersections, and a constant-velocity extrapolation runs those scenarios off the
reference behaviour. A learned planner conditioned on the observed frames is the
obvious replacement; the interface it has to satisfy is `plan_trajectory()`.

## Gemini VLM Trajectory Planner

`kitscenes/vlm_planner.py` provides the Gemini VLM Trajectory Reasoning & Planning module
conditioned on driving instructions, odometry history and optional *text* camera summaries.
No camera frames are sent to the model, and without `GEMINI_API_KEY` the module falls
back to an offline instruction parser — the CLI reports which planner actually ran:

```sh
python build_submission.py plan \
    --history history.jsonl \
    --out submissions/trajectories.jsonl \
    --vlm \
    --model gemini-3.8-flash \
    --instructions instructions.jsonl \
    --camera-summaries summaries.jsonl
```

### Key Capabilities:
- **Instruction reasoning**: sends the driving instruction and the odometry state to Gemini 3.8 Flash and parses the tactical reply ("overtake on right", "slow down for construction", "brake for pedestrian", "turn left/right at intersection"). Text in, text out — it does not see the video.
- **Physical constraints**: acceleration clamped to [-3.5, 2.0] m/s², jerk rate-limited at
  2.5 m/s³, lateral transitions on a quintic Frenet polynomial, yaw rate capped at 0.6 rad/s.
- **What is actually measured**: on synthetic odometry the first waypoint sits within 0.01° of
  the current heading (cos ≈ 1.000) and a lane change keeps tortuosity at 1.0025. A 90° turn
  reaches 1.08, because a turn is a detour — not a defect. Whether the plan beats the
  *reference* by more than 6 % tortuosity and 44 % jerk can only be checked against the gated
  reference trajectories; nothing here measures it.
- **Offline fallback**: a deterministic instruction parser runs when no API key is present.

## Perceptual quality proxy

`kitscenes/pq_proxy.py` estimates PQ without Q-Align. It decodes the videos through ffmpeg
and scores texture energy, colour range and flat-fill ratio, then applies the official camera
weights, so it answers "is version A of my generator visually richer than version B?". It is
not calibrated against human ratings and must not be read as a leaderboard number;
`estimate_archive_pq("videos.zip", max_scenarios=20)` gives a rough average on a sample.

## What these checks cannot tell you

Nothing here computes a score. PQ needs Q-Align (OneAlign) plus the real calibration
videos it is normalised against; EMC needs optical flow over six synchronised views; MMS
needs the per-scenario reference trajectories, which live in the gated dataset split.
A clean validation run means the submission will be accepted and evaluated, not that it
will rank.

## Status

The format, trajectory and packaging layers are done and tested. Video generation ships a
procedural renderer rather than a learned model: it produces format-valid clips whose motion
follows the planned trajectory, and `kitscenes/pq_proxy.py` confirms their perceptual quality
sits far below real driving footage. That is enough to run the pipeline end to end and
nowhere near a competitive entry — PQ is the metric the ranking uses. The missing piece is a
diffusion backbone conditioned on KITScenes; the interface it has to satisfy is
`WorldModelPipeline`.

## References

- Challenge: <https://huggingface.co/spaces/KIT-MRT/KITScenes-LongTail-Challenge>
- Dataset: <https://huggingface.co/datasets/KIT-MRT/KITScenes-LongTail>
- Paper: Wagner et al., *Reasoning models do not yet follow their reasoning in autonomous
  driving: The KITScenes LongTail Dataset*, [arXiv:2603.23607](https://arxiv.org/abs/2603.23607)
- Q-Align / OneAlign: <https://huggingface.co/q-future/one-align>
- WorldMark perceptual metric: <https://github.com/AlayaLab/WorldMark>
