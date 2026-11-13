"""Validation and assembly of the videos archive.

``inspect_archive`` mirrors the upload checks the challenge Space runs
(``validate_videos_file`` in its ``src/utils.py``) so that a local pass means the
server will accept the file, then adds the encoding checks that validation skips.
``build_archive`` writes the archive in the layout the server expects.
"""

import concurrent.futures
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

from . import media, spec


@dataclass
class ArchiveReport:
    path: Optional[Path] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    size_bytes: int = 0
    videos: int = 0
    scenarios_covered: int = 0
    scenarios_complete: int = 0
    media_checked: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary_lines(self) -> List[str]:
        lines = [
            f"videos: {self.videos}, scenarios: {self.scenarios_covered}"
            f"/{spec.SCENARIO_COUNT}, complete six-camera scenarios: "
            f"{self.scenarios_complete}",
        ]
        if self.size_bytes:
            size = (
                f"{self.size_bytes / 1024**2:.1f} MiB"
                if self.size_bytes < 1024**3
                else f"{self.size_bytes / 1024**3:.2f} GiB"
            )
            lines.append(
                f"archive: {size} "
                f"({self.size_bytes / spec.MAX_ARCHIVE_BYTES:.1%} of the 3 GiB cap)"
            )
        if self.media_checked:
            lines.append(f"videos decoded and checked: {self.media_checked}")
        return lines

    def as_dict(self) -> dict:
        return {
            "path": str(self.path) if self.path else None,
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "size_bytes": self.size_bytes,
            "videos": self.videos,
            "scenarios_covered": self.scenarios_covered,
            "scenarios_complete": self.scenarios_complete,
            "media_checked": self.media_checked,
        }


def _inspect_media(
    archive_path: Path, filename: str
) -> Tuple[List[str], List[str]]:
    """Probe one archived video by streaming it to a temporary file.

    Opening a dedicated ZipFile instance per thread prevents shared file descriptor
    concurrency issues and Bad CRC-32 errors during parallel extraction.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
        temporary = Path(handle.name)
        with zipfile.ZipFile(archive_path, "r") as zf:
            with zf.open(filename) as source:
                shutil.copyfileobj(source, handle)
    try:
        return media.check_video(media.probe_video(temporary))
    except media.ProbeError as exc:
        return [f"{filename}: {exc}"], []
    finally:
        temporary.unlink(missing_ok=True)


def _complete_scenarios(videos: Dict[Tuple[str, str], object]) -> int:
    return len(
        [
            scenario
            for scenario in spec.SCENARIO_IDS
            if all((scenario, camera) in videos for camera in spec.CAMERAS)
        ]
    )


def inspect_archive(
    path: Path,
    *,
    check_media: bool = True,
    jobs: int = 8,
) -> ArchiveReport:
    """Validate ``path`` and return every problem found.

    ``check_media`` decodes each archived video with ffprobe. It is the only way to
    catch frame-count, frame-rate and codec drift, and it costs one probe per video:
    about a minute for a complete submission on a laptop.
    """
    path = Path(path)
    report = ArchiveReport(path=path)

    if not path.exists():
        report.errors.append(f"no such file: {path}")
        return report
    if path.suffix.lower() != ".zip":
        report.errors.append(
            f"expected a .zip archive, got '{path.suffix or 'no extension'}'"
        )
        return report

    report.size_bytes = path.stat().st_size
    if report.size_bytes > spec.MAX_ARCHIVE_BYTES:
        report.errors.append(
            f"archive is {report.size_bytes / 1024**3:.2f} GiB, the upload cap is 3 GiB"
        )

    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        report.errors.append(f"not a readable zip archive ({exc})")
        return report

    with zf:
        videos: Dict[Tuple[str, str], zipfile.ZipInfo] = {}
        ignored: List[str] = []

        for entry in zf.infolist():
            if entry.is_dir():
                continue
            parts = PurePosixPath(entry.filename).parts
            if "__MACOSX" in parts:
                continue
            if len(parts) < 2:
                ignored.append(entry.filename)
                continue

            scenario_id, file_name = parts[-2], parts[-1]
            if scenario_id not in spec.SCENARIO_ID_SET:
                # The server ignores scenario folders outside 0000000-0000399.
                ignored.append(entry.filename)
                continue
            if file_name.startswith("."):
                continue
            if not file_name.endswith(".mp4"):
                report.errors.append(
                    f"{entry.filename}: scenario folders may only contain <camera>.mp4"
                )
                continue

            camera = file_name[: -len(".mp4")]
            if camera not in spec.CAMERAS:
                report.errors.append(
                    f"{entry.filename}: unknown camera '{camera}', expected one of "
                    f"{', '.join(spec.CAMERAS)}"
                )
                continue
            if (scenario_id, camera) in videos:
                report.errors.append(f"duplicate video {scenario_id}/{camera}.mp4")
                continue
            if entry.file_size > spec.MAX_VIDEO_BYTES:
                report.errors.append(
                    f"{entry.filename}: {entry.file_size / 1024**2:.1f} MB exceeds the "
                    f"50 MB per-video limit"
                )

            videos[(scenario_id, camera)] = entry

        report.videos = len(videos)
        report.scenarios_covered = len({scenario for scenario, _ in videos})
        report.scenarios_complete = _complete_scenarios(videos)

        if ignored:
            preview = ", ".join(ignored[:3])
            report.warnings.append(
                f"{len(ignored)} entr{'y' if len(ignored) == 1 else 'ies'} outside "
                f"the {spec.SCENARIO_COUNT} scenario folders are ignored by the server "
                f"({preview}{', ...' if len(ignored) > 3 else ''})"
            )

        missing_front = [
            scenario
            for scenario in spec.SCENARIO_IDS
            if (scenario, spec.FRONT_CAMERA) not in videos
        ]
        if missing_front:
            preview = ", ".join(missing_front[:10])
            report.errors.append(
                f"missing front.mp4 for {len(missing_front)} scenario(s): {preview}"
                f"{', ...' if len(missing_front) > 10 else ''}"
            )

        partial = [
            scenario
            for scenario in spec.SCENARIO_IDS
            if (scenario, spec.FRONT_CAMERA) in videos
            and any((scenario, camera) not in videos for camera in spec.CAMERAS)
        ]
        if partial:
            ceiling = 100 * (
                spec.PQ_FRONT_WEIGHT + spec.PQ_SIDE_WEIGHT * len(spec.SIDE_CAMERAS)
            )
            report.warnings.append(
                f"{len(partial)} scenario(s) miss at least one camera: EMC is 0 for "
                f"them and their PQ is capped at {ceiling:.0f}% of the per-scenario "
                f"maximum"
            )

        if not check_media:
            if videos:
                report.warnings.append(
                    "encoding was not checked (--skip-media): a wrong frame count, "
                    "frame rate or codec zeroes the scenario's PQ and EMC"
                )
            return report

        if not videos:
            return report

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            results = list(
                pool.map(
                    lambda name: _inspect_media(path, name),
                    [entry.filename for entry in videos.values()],
                )
            )
        report.media_checked = len(results)

        for errors, warnings in results:
            report.errors.extend(errors)
            report.warnings.extend(warnings)

    return report


def _collect_videos(
    directory: Path,
) -> Tuple[Dict[Tuple[str, str], Path], List[str]]:
    found: Dict[Tuple[str, str], Path] = {}
    problems: List[str] = []

    for path in sorted(directory.rglob("*.mp4")):
        if "__MACOSX" in path.parts:
            continue
        scenario_id, camera = path.parent.name, path.stem
        if scenario_id not in spec.SCENARIO_ID_SET:
            problems.append(f"{path.relative_to(directory)}: not a scenario folder")
            continue
        if camera not in spec.CAMERAS:
            problems.append(f"{path.relative_to(directory)}: unknown camera '{camera}'")
            continue
        found[(scenario_id, camera)] = path

    return found, problems


def build_archive(
    directory: Path,
    output: Optional[Path] = None,
    *,
    allow_partial: bool = False,
    check_media: bool = True,
) -> ArchiveReport:
    """Pack ``directory`` into a submission archive and report what went in.

    MP4 payload is already compressed, so entries are stored rather than deflated:
    that keeps the write fast and the archive size predictable against the 3 GiB cap.
    Nothing is written when the inputs fail validation.
    """
    directory = Path(directory)
    output = Path(output) if output else directory.parent / "videos.zip"
    report = ArchiveReport(path=output)

    if not directory.is_dir():
        report.errors.append(f"no such directory: {directory}")
        return report

    videos, problems = _collect_videos(directory)
    report.errors.extend(problems)

    missing_front = [
        scenario
        for scenario in spec.SCENARIO_IDS
        if (scenario, spec.FRONT_CAMERA) not in videos
    ]
    if missing_front and not allow_partial:
        report.errors.append(
            f"missing front.mp4 for {len(missing_front)} scenario(s), e.g. "
            f"{', '.join(missing_front[:5])}. Pass --allow-partial to pack a subset for "
            f"local checks; the server rejects an incomplete archive"
        )
    elif missing_front:
        report.warnings.append(
            f"partial archive: {len(missing_front)} scenario(s) have no front.mp4, "
            f"so the server will reject this file"
        )

    if check_media:
        for (scenario_id, camera), path in sorted(videos.items()):
            try:
                errors, warnings = media.check_video(media.probe_video(path))
            except media.ProbeError as exc:
                report.errors.append(f"{scenario_id}/{camera}.mp4: {exc}")
                continue
            report.errors.extend(f"{scenario_id}/{camera}.mp4: {e}" for e in errors)
            report.warnings.extend(f"{scenario_id}/{camera}.mp4: {w}" for w in warnings)
            report.media_checked += 1

    if report.errors:
        return report

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as zf:
        for (scenario_id, camera), path in sorted(videos.items()):
            zf.write(path, arcname=f"{scenario_id}/{camera}.mp4")

    report.size_bytes = output.stat().st_size
    report.videos = len(videos)
    report.scenarios_covered = len({scenario for scenario, _ in videos})
    report.scenarios_complete = _complete_scenarios(videos)

    if report.size_bytes > spec.MAX_ARCHIVE_BYTES:
        report.errors.append(
            f"written archive is {report.size_bytes / 1024**3:.2f} GiB, above the "
            f"3 GiB cap"
        )
    elif report.size_bytes > 0.9 * spec.MAX_ARCHIVE_BYTES:
        report.warnings.append(
            f"{report.size_bytes / 1024**3:.2f} GiB is close to the 3 GiB cap; lower the "
            f"bitrate before adding the remaining cameras"
        )

    return report
