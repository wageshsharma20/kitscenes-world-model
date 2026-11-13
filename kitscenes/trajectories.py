"""Trajectory file validation, and a baseline planner.

The trajectory file is optional, but without it the leaderboard reports N/A for EMC,
MMS and L2. It is also the only part of the submission that can be produced without a
generative model, which makes it the right place to start: a physically plausible
plan is what the generated videos then have to agree with.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import spec

HISTORY_KEYS = ("past_trajectory", "history", "past_xy")

# Planning defaults. Speeds are metres per second, angles radians.
LOOKBACK = 5
YAW_RATE_DECAY = 0.9
MAX_YAW_RATE = 0.6
MAX_ACCELERATION = 2.0
MAX_SPEED = 55.0
PRECISION = 3


@dataclass
class TrajectoryReport:
    path: Optional[Path] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    count: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary_lines(self) -> List[str]:
        return [f"records: {self.count}/{spec.SCENARIO_COUNT}"]

    def as_dict(self) -> dict:
        return {
            "path": str(self.path) if self.path else None,
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "count": self.count,
        }


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _speed_profile(points: np.ndarray) -> np.ndarray:
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return steps / spec.WAYPOINT_STEP_S


def _tortuosity(points: np.ndarray) -> float:
    """Path length over straight-line distance, as the MMS comfort penalty defines it."""
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    chord = float(np.linalg.norm(points[-1] - points[0]))
    return length / chord if chord > 0 else math.inf


def inspect_trajectories(path: Path) -> TrajectoryReport:
    """Validate a ``trajectories.jsonl`` against the official schema.

    Ordering is not checked: the server matches records by ``scenario_id``, so a file
    is valid as long as it contains each of the 400 ids exactly once.
    """
    path = Path(path)
    report = TrajectoryReport(path=path)

    if not path.exists():
        report.errors.append(f"no such file: {path}")
        return report
    if path.suffix.lower() != ".jsonl":
        report.errors.append(
            f"expected a .jsonl file, got '{path.suffix or 'no extension'}'"
        )
        return report

    seen: Dict[str, int] = {}
    quality: Dict[str, List[str]] = {
        "stationary": [],
        "implausible speed": [],
        "repeated waypoints": [],
        "erratic": [],
    }
    extra_keys: List[str] = []

    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                report.errors.append(f"line {number}: blank line")
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                report.errors.append(f"line {number}: invalid JSON ({exc.msg})")
                continue
            if not isinstance(record, dict):
                report.errors.append(f"line {number}: expected a JSON object")
                continue

            missing = [
                key
                for key in ("scenario_id", "future_trajectory")
                if key not in record
            ]
            if missing:
                report.errors.append(
                    f"line {number}: missing key(s) {', '.join(missing)}"
                )
                continue
            if extra_keys == [] and len(record) > 2:
                extra_keys = sorted(set(record) - {"scenario_id", "future_trajectory"})

            scenario_id = record["scenario_id"]
            if not isinstance(scenario_id, str) or scenario_id not in spec.SCENARIO_ID_SET:
                report.errors.append(
                    f"line {number}: scenario_id {scenario_id!r} is not one of "
                    f"'0000000'...'0000399'"
                )
                continue
            if scenario_id in seen:
                report.errors.append(
                    f"line {number}: duplicate scenario_id '{scenario_id}' "
                    f"(first seen on line {seen[scenario_id]})"
                )
                continue
            seen[scenario_id] = number

            points = record["future_trajectory"]
            if not isinstance(points, list) or len(points) != spec.WAYPOINT_COUNT:
                found = len(points) if isinstance(points, list) else type(points).__name__
                report.errors.append(
                    f"line {number}: future_trajectory must hold "
                    f"{spec.WAYPOINT_COUNT} points, found {found}"
                )
                continue
            if not all(
                isinstance(point, (list, tuple))
                and len(point) == 2
                and all(_is_number(value) and math.isfinite(value) for value in point)
                for point in points
            ):
                report.errors.append(
                    f"line {number}: every waypoint must be a pair of finite numbers"
                )
                continue

            array = np.asarray(points, dtype=float)
            displacement = float(np.abs(array).max())
            if displacement < 0.5:
                quality["stationary"].append(scenario_id)
                continue
            if _speed_profile(array).max() > MAX_SPEED:
                quality["implausible speed"].append(scenario_id)
            if np.any(np.all(np.diff(array, axis=0) == 0, axis=1)):
                quality["repeated waypoints"].append(scenario_id)
            if _tortuosity(array) > 3.0:
                quality["erratic"].append(scenario_id)

    report.count = len(seen)

    missing_ids = [identifier for identifier in spec.SCENARIO_IDS if identifier not in seen]
    if missing_ids:
        report.errors.append(
            f"missing {len(missing_ids)} scenario(s): "
            f"{', '.join(missing_ids[:10])}"
            f"{', ...' if len(missing_ids) > 10 else ''}"
        )

    if extra_keys:
        report.warnings.append(
            f"records carry extra keys ({', '.join(extra_keys)}); they are ignored"
        )

    for label, scenarios in quality.items():
        if scenarios:
            report.warnings.append(
                f"{len(scenarios)} trajectory(ies) look {label} "
                f"(e.g. {', '.join(scenarios[:3])}); this costs MMS points"
            )

    return report


def _to_xy(points: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[0] < 4:
        raise ValueError("need at least four history points")
    if array.shape[1] == 3:
        array = array[:, :2]
    elif array.shape[1] != 2:
        raise ValueError(
            f"history must be N x 2 (or N x 3), got {array.shape[1]} columns"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("history contains non-finite coordinates")
    return array


def _circular_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    sine = float(np.sum(weights * np.sin(angles)))
    cosine = float(np.sum(weights * np.cos(angles)))
    return math.atan2(sine, cosine)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def plan_trajectory(
    history: Sequence[Sequence[float]],
    *,
    horizon: int = spec.WAYPOINT_COUNT,
    lookback: int = LOOKBACK,
    yaw_rate_decay: float = YAW_RATE_DECAY,
) -> np.ndarray:
    """Extrapolate a constant-speed, decaying-yaw-rate future from odometry history.

    The motion model is the one a vehicle follows while the driver holds the wheel:
    speed carries an acceleration estimated from the recent past and clamped to
    ``MAX_ACCELERATION``, yaw rate decays geometrically, so the path relaxes onto a
    straight line instead of circling. Both comfort penalties in MMS (jerk and
    tortuosity) are essentially satisfied by construction, which is what makes this a
    usable lower bound rather than a guess.

    Returns an ``(horizon, 2)`` array in the coordinate frame of ``history``.
    """
    points = _to_xy(history)
    step = spec.WAYPOINT_STEP_S

    deltas = np.diff(points, axis=0)[-lookback:]
    displacements = np.linalg.norm(deltas, axis=1)
    speeds = displacements / step
    moving = displacements > 1e-3

    if moving.any():
        headings = np.arctan2(deltas[moving, 1], deltas[moving, 0])
        heading = _circular_mean(headings, speeds[moving])
        speed = float(np.median(speeds[moving][-3:]))
        if len(headings) >= 2:
            # Wrap each turn into (-pi, pi] so a heading crossing +-pi does not read
            # as a full circle.
            turns = np.angle(np.exp(1j * np.diff(headings)))
            yaw_rate = float(np.median(turns)) / step
        else:
            yaw_rate = 0.0
    else:
        # A stopped vehicle has no heading; assume the ego frame's forward axis, which
        # is the convention of the reference trajectories.
        heading, speed, yaw_rate = 0.0, 0.0, 0.0

    if len(speeds) >= 2:
        acceleration = (speeds[-1] - speeds[0]) / (len(speeds) * step)
    else:
        acceleration = 0.0
    acceleration = float(np.clip(acceleration, -MAX_ACCELERATION, MAX_ACCELERATION))
    yaw_rate = float(np.clip(yaw_rate, -MAX_YAW_RATE, MAX_YAW_RATE))

    x, y = float(points[-1, 0]), float(points[-1, 1])
    future = np.empty((horizon, 2), dtype=float)

    for index in range(horizon):
        x += speed * step * math.cos(heading)
        y += speed * step * math.sin(heading)
        future[index] = (x, y)
        heading = _wrap(heading + yaw_rate * step)
        yaw_rate *= yaw_rate_decay
        speed = float(np.clip(speed + acceleration * step, 0.0, MAX_SPEED))

    return future


def plan_file(source: Path, destination: Path) -> int:
    """Plan every scenario in ``source`` and write a submission trajectory file.

    ``source`` is JSON Lines with one ``{"scenario_id": ..., "past_trajectory": [[x, y], ...]}``
    record per scenario. Returns the number of trajectories written.
    """
    source = Path(source)
    destination = Path(destination)

    if not source.exists():
        raise ValueError(f"no such file: {source}")

    planned: Dict[str, np.ndarray] = {}
    three_columns = False
    with open(source, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            scenario_id = record.get("scenario_id")
            if scenario_id not in spec.SCENARIO_ID_SET:
                raise ValueError(f"line {number}: unknown scenario_id {scenario_id!r}")
            history = next(
                (record[key] for key in HISTORY_KEYS if key in record), None
            )
            if history is None:
                raise ValueError(
                    f"line {number}: expected one of {', '.join(HISTORY_KEYS)}"
                )
            if history and len(history[0]) == 3:
                three_columns = True
            planned[scenario_id] = plan_trajectory(history)

    if three_columns:
        print(
            "warning: history has three columns, using the first two as (x, y) in the "
            "bird's-eye-view plane; check the axis order of your export"
        )

    if not planned:
        raise ValueError(f"{source} contains no records")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        for scenario_id in sorted(planned):
            points = np.round(planned[scenario_id], PRECISION).tolist()
            handle.write(
                json.dumps({"scenario_id": scenario_id, "future_trajectory": points})
                + "\n"
            )

    return len(planned)
