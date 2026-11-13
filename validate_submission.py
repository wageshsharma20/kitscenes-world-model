#!/usr/bin/env python3
"""Check a submission against the KITScenes-LongTail World Modeling Challenge rules.

Validates the videos archive and the trajectories file, locally and offline, before
the upload and the 20-30 minute evaluation round trip. A clean run means the server
will accept the files; it does not predict the scores, which depend on the model.

Exit codes: 0 both files valid, 1 at least one file invalid, 2 the check could not run
(missing path, ffprobe unavailable).
"""

import argparse
import json
import sys
from pathlib import Path

from kitscenes import archive, media, spec, trajectories
from kitscenes.reporting import print_result, print_section


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a videos archive and a trajectories file for the KITScenes-LongTail "
            "World Modeling Challenge."
        ),
        epilog=(
            "examples:\n"
            "  python validate_submission.py --videos videos.zip "
            "--trajectories trajectories.jsonl\n"
            "  python validate_submission.py --videos videos.zip --skip-media\n"
            "      (layout and size checks only)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--videos", type=Path, help="submission archive (.zip)")
    parser.add_argument("--trajectories", type=Path, help="trajectory file (.jsonl)")
    parser.add_argument(
        "--skip-media",
        action="store_true",
        help="skip decoding videos (frame count, frame rate and codec go unchecked)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="videos decoded in parallel (default: 8)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="also write the findings to this file as JSON",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.videos and not args.trajectories:
        parser.error("pass --videos and/or --trajectories")

    report = {}
    total_errors = 0
    total_warnings = 0

    if args.trajectories:
        inspection = trajectories.inspect_trajectories(args.trajectories)
        print_section(
            str(args.trajectories),
            inspection.errors,
            inspection.warnings,
            inspection.summary_lines(),
        )
        report["trajectories"] = inspection.as_dict()
        total_errors += len(inspection.errors)
        total_warnings += len(inspection.warnings)

    if args.videos:
        try:
            inspection = archive.inspect_archive(
                args.videos, check_media=not args.skip_media, jobs=args.jobs
            )
        except (media.ProbeError, OSError) as exc:
            # A missing ffprobe or an unreadable file is an environment problem, not
            # a verdict on the submission.
            print(f"error: could not inspect {args.videos}: {exc}", file=sys.stderr)
            return 2
        print_section(
            str(args.videos),
            inspection.errors,
            inspection.warnings,
            inspection.summary_lines(),
        )
        report["videos"] = inspection.as_dict()
        total_errors += len(inspection.errors)
        total_warnings += len(inspection.warnings)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"report written to {args.json}")

    print_result(total_errors == 0, total_errors, total_warnings)
    print(
        f"deadline {spec.DEADLINE} ({spec.DEADLINE_TIMEZONE}); "
        f"{spec.SUBMISSIONS_PER_WINDOW} submissions per rolling "
        f"{spec.SUBMISSION_WINDOW_DAYS}-day window"
    )

    return 1 if total_errors else 0


if __name__ == "__main__":
    sys.exit(main())
