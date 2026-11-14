"""Challenge constants, transcribed from the official rules.

Source: "Submit" and "About" tabs of
https://huggingface.co/spaces/KIT-MRT/KITScenes-LongTail-Challenge
and the submission checks in that Space's ``src/utils.py`` (read 2026-11-13).

Two kinds of numbers live here:

* limits the server rejects outright (archive size, per-video size, file naming,
  trajectory record count);
* properties the server does not reject but that zero or degrade the score
  (frame count, frame rate, codec, camera coverage, aspect ratio).

Keeping both in one module means the validator, the packager and the tests cannot
drift apart. Change a value here, not in a script.
"""

TEST_SCENARIO_COUNT = 400
SCENARIO_COUNT = TEST_SCENARIO_COUNT
SCENARIO_IDS = tuple(f"{index:07d}" for index in range(SCENARIO_COUNT))
SCENARIO_ID_SET = frozenset(SCENARIO_IDS)

# Dataset split definitions from official Parquet splits:
# Train: 500 scenarios (0000400 - 0000899)
# Val: 100 scenarios (0000900 - 0000999)
# Test: 400 scenarios (0000000 - 0000399)
TRAIN_SCENARIO_COUNT = 500
TRAIN_SCENARIO_IDS = tuple(f"{index:07d}" for index in range(400, 900))
TRAIN_SCENARIO_ID_SET = frozenset(TRAIN_SCENARIO_IDS)

VAL_SCENARIO_COUNT = 100
VAL_SCENARIO_IDS = tuple(f"{index:07d}" for index in range(900, 1000))
VAL_SCENARIO_ID_SET = frozenset(VAL_SCENARIO_IDS)

# Canonical 10-scenario validation subset covering all 7 scenario types & core maneuvers:
# 0000900: snow & wintry mix, drive straight on
# 0000901: overtake or lane change, use right lane
# 0000902: intersection, turn right
# 0000903: heavy rain, drive straight on
# 0000904: construction zone, drive straight on
# 0000905: specifically selected, turn right
# 0000906: nighttime, use left lane
# 0000911: specifically selected, turn left
# 0000919: overtake or lane change, overtake truck driving on the right
# 0000984: intersection, u-turn
VALIDATION_10_IDS = (
    "0000900",
    "0000901",
    "0000902",
    "0000903",
    "0000904",
    "0000905",
    "0000906",
    "0000911",
    "0000919",
    "0000984",
)

CAMERAS = (
    "front",
    "front_left",
    "front_right",
    "rear",
    "rear_left",
    "rear_right",
)
FRONT_CAMERA = "front"
# EMC estimates speed from the four side views and yaw from front + rear.
SIDE_CAMERAS = ("front_left", "front_right", "rear_left", "rear_right")
YAW_CAMERAS = ("front", "rear")

# Video: exactly 25 frames at 5 fps, covering 0.2 s to 5.0 s of the future.
FRAME_COUNT = 25
FPS = 5.0
FPS_TOLERANCE = 0.01
WIDTH = 876
HEIGHT = 568
ASPECT_RATIO = WIDTH / HEIGHT
ASPECT_RATIO_TOLERANCE = 0.01
ACCEPTED_CODECS = frozenset({"h264"})

# Trajectory: 25 (x, y) pairs at 5 Hz, the first one 0.2 s into the future.
WAYPOINT_COUNT = 25
WAYPOINT_STEP_S = 1.0 / FPS
FIRST_WAYPOINT_S = WAYPOINT_STEP_S
LAST_WAYPOINT_S = WAYPOINT_COUNT * WAYPOINT_STEP_S
TRAJECTORY_PRECISION = 3

# Upload limits. The Space caps the upload at "3gb" (Gradio) and rejects any
# single video whose uncompressed size exceeds 50 MiB.
MAX_ARCHIVE_BYTES = 3 * 1024**3
MAX_VIDEO_BYTES = 50 * 1024**2

# Scoring constants, needed to reason about a submission even though the
# validator cannot reproduce the metrics (see README).
PQ_FRONT_WEIGHT = 0.50
PQ_SIDE_WEIGHT = 0.10
EMC_SEGMENTS_S = ((0.0, 3.0), (3.0, 5.0))
EMC_CLASSES_PER_SCENARIO = 4
EMC_SPEED_RATIO_MIN = 0.5
EMC_SPEED_OFFSET = 0.5
MMS_BASE_EXPERT = 10
MMS_BASE_WRONG_SPEED = 7
MMS_BASE_UNMATCHED = 3.5
MMS_JERK_PENALTY_RATIO = 0.44
MMS_TORTUOSITY_PENALTY_RATIO = 0.06

# Team/account submission budget: 5 per rolling 30-day window.
SUBMISSIONS_PER_WINDOW = 5
SUBMISSION_WINDOW_DAYS = 30

DEADLINE = "2026-12-03"
DEADLINE_TIMEZONE = "AoE"

# Evaluation Metric Labels
# Official Challenge Leaderboard evaluates Perceptual Quality using Q-Align (OneAlign).
# Local evaluations use PQ_proxy (Laplacian texture variance + color dynamic range - flat fill penalties).
METRIC_LABEL_OFFICIAL_PQ = "PQ"
METRIC_LABEL_PROXY_PQ = "PQ_proxy"

