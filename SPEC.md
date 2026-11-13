# Submission specification and scoring

Condensed from the "Submit" and "About" tabs of the official challenge Space and its
`src/utils.py`, read on 2026-11-13. When this file and the Space disagree, the Space wins;
`kitscenes/spec.py` is the machine-readable copy that the tools use.

- Challenge window: **2026-11-07 to 2026-12-03, AoE**
- Prizes: **€1000 / €600 / €400** for first, second and third place
- Evaluation round trip: typically **20–30 minutes** per submission

## Videos file (`videos.zip`)

```
videos.zip
├── 0000000/
│   ├── front.mp4          required
│   ├── front_left.mp4     optional
│   ├── front_right.mp4    optional
│   ├── rear.mp4           optional
│   ├── rear_left.mp4      optional
│   └── rear_right.mp4     optional
├── ...
└── 0000399/front.mp4
```

| Property | Requirement |
| --- | --- |
| Scenario folders | `0000000` … `0000399`, zero-padded to seven digits |
| Camera names | `front`, `front_left`, `front_right`, `rear`, `rear_left`, `rear_right` |
| Required view | `front.mp4` for every scenario; the other five are optional but scored |
| Container / codec | `.mp4`, H.264 |
| Frames | exactly 25 |
| Frame rate | 5 fps |
| Time span | 0.2 s to 5.0 s of the future |
| Resolution | keep the KITScenes aspect ratio, e.g. 876×568 |
| Archive size | ≤ 3 GiB |
| Per-video size | ≤ 50 MiB uncompressed |

A scenario whose video breaks the encoding contract **scores 0 for both PQ and EMC**.
Folders outside the 400 scenario ids and hidden files are ignored by the server; any other
file inside a scenario folder is rejected.

## Trajectories file (`trajectories.jsonl`)

Optional. Without it, EMC, MMS and the L2 columns are N/A.

- Exactly 400 lines, one JSON object per line.
- Keys: `scenario_id` (string, `"0000000"`…`"0000399"`) and `future_trajectory`.
- `future_trajectory`: 25 `[x, y]` pairs at 5 Hz from 0.2 s to 5.0 s; the first pair is
  0.2 s into the future.
- Order does not matter: records are matched by `scenario_id`.

```json
{"scenario_id": "0000000", "future_trajectory": [[x0, y0], [x1, y1], "…", [x24, y24]]}
```

## Ranking

Submissions are ranked by **PQ**. Ties are broken by **EMC × MMS**. The leaderboard also
reports EMC, MMS, L2 (mean and median) and PQ per scenario type (selected, rain,
construction, overtaking, intersection, night, snow); those per-type columns are diagnostic
and do not enter the ranking.

### Perceptual quality (PQ)

Each camera video is rated by Q-Align (OneAlign) on the levels excellent (5), good (4),
fair (3), poor (2) and bad (1), and the expected rating is mapped to `100 * (rating - 1) / 4`.
Scores are normalised against real long-tail calibration videos.

```
PQ_scenario = 0.50 * PQ_front + 0.10 * (PQ_front_left + PQ_front_right
                                       + PQ_rear + PQ_rear_left + PQ_rear_right)
```

Missing or unreadable videos score 0, so a front-only submission caps at 50 % of the
per-scenario maximum. PQ is the mean over the 400 scenarios.

### Ego-motion coherence (EMC)

Optical flow recovers ego-motion from the generated videos. Yaw comes from `front` and
`rear`, which must agree; speed comes from the four side cameras. Planned and recovered
trajectories are each cut into two segments (0–3 s and 3–5 s) and each segment gets a
steering class and an acceleration class. A class matches only if the segment speeds are
also close:

```
(min(v_a, v_b) + 0.5) / (max(v_a, v_b) + 0.5) >= 0.5
```

If any of the six camera videos is missing, every class of that scenario counts as a
mismatch. EMC is the fraction of matching classes over all scenarios, segments and class
types (4 per scenario), normalised by the real reference videos and clamped at 1.0.

Consequence for generation: the videos have to agree with the submitted trajectory, not
just look plausible. Generate each view with the ego-motion conditioned on the planned
waypoints, otherwise EMC drops towards 0 while PQ looks fine.

### Multi-maneuver score (MMS)

Trajectories are compared against three per-scenario reference behaviours:

| Reference | Base MMS |
| --- | --- |
| Expert-like | 10 |
| Wrong speed | 7 |
| Neglect instruction | 4 |
| Off road without crashing | 1 |
| Crash | 0 |
| Unmatched | 3.5 |

Before matching, a trajectory whose first waypoint does not point in the same direction as
the expert's first waypoint scores 0:

```
dot(ref[0], pred[0]) < 0.5 * |ref[0]|  ->  0
```

The best-matching reference is chosen with velocity-scaled lateral/longitudinal thresholds
(1.8 m / 3.6 m at low speed, doubled at ≥ 11 m/s), similarity is the minimum of the two
axis scores, and the score is `similarity * (base - comfort_penalties)`, floored at
`3.5 - comfort_penalties`. Comfort penalties subtract 1 each when the trajectory's jerk is
more than 44 % above the reference, and 1 when its tortuosity is more than 6 % above it.

Tortuosity is path length over start-to-end distance, so a stop that ends where it started
is penalised. Jerk is the mean magnitude of the third finite difference.

## Rules and eligibility

- Teams are allowed; a participant may belong to one team only.
- At least one member needs a Hugging Face account to submit.
- **5 submissions per rolling 30-day window**, enforced per team and per account.
- Only the best-PQ submission of a team decides the ranking.
- Participants must be 18 or older, and must not be resident in Quebec (Canada), Brazil or
  Italy; employees of government bodies are excluded with narrow exceptions for state
  universities.
- Winners are invited to record a 4-minute method video for the workshop.

## Citation

```bibtex
@misc{wagner2026reasoningmodelsfollowreasoning,
  title={Reasoning models do not yet follow their reasoning in autonomous driving: The KITScenes LongTail Dataset},
  author={Royden Wagner and Omer Sahin Tas and Jaime Villa and Felix Hauser and Yinzhe Shen and Marlon Steiner and Dominik Strutz and Carlos Fernandez and Quentin Delfosse and Christoph Weinhuber and Christian Kinzig and Guillermo S. Gutierrez-Cabello and Hendrik Königshof and Fabian Immel and Richard Schwarzkopf and Nils Alexander Rack and Kevin Rösch and Kaiwen Wang and Jan-Hendrik Pauls and Martin Lauer and Igor Gilitschenski and Holger Caesar and Christoph Stiller},
  year={2026},
  eprint={2603.23607},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.23607}
}
```
