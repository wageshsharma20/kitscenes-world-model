#!/usr/bin/env python3
"""Assemble submission files from generated predictions.

    package  render outputs on disk        -> videos.zip
    plan     observed odometry as JSONL    -> trajectories.jsonl

Both subcommands refuse to write a file that the server would reject.
"""

import argparse
import sys
from pathlib import Path

from kitscenes import archive, media, trajectories, vlm_planner
from kitscenes.reporting import print_result, print_section


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the two files the KITScenes-LongTail challenge asks for.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python build_submission.py package --videos-dir renders/ "
            "--out submissions/videos.zip\n"
            "  python build_submission.py plan --history history.jsonl "
            "--out submissions/trajectories.jsonl\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack = subparsers.add_parser(
        "package", help="pack <scenario_id>/<camera>.mp4 renders into videos.zip"
    )
    pack.add_argument(
        "--videos-dir",
        type=Path,
        required=True,
        help="directory holding <scenario_id>/<camera>.mp4",
    )
    pack.add_argument(
        "--out", type=Path, help="archive to write (default: <videos-dir>/../videos.zip)"
    )
    pack.add_argument(
        "--allow-partial",
        action="store_true",
        help="pack an incomplete set of scenarios or cameras (local checks only)",
    )
    pack.add_argument(
        "--skip-media",
        action="store_true",
        help="do not decode the videos before packing",
    )

    planner = subparsers.add_parser(
        "plan", help="write trajectories.jsonl from observed ego-motion"
    )
    planner.add_argument(
        "--history",
        type=Path,
        required=True,
        help='JSON Lines with {"scenario_id": ..., "past_trajectory": [[x, y], ...]}',
    )
    planner.add_argument(
        "--out", type=Path, required=True, help="trajectory file to write"
    )
    planner.add_argument(
        "--vlm",
        action="store_true",
        help="use Gemini VLM tactical reasoning for trajectory planning",
    )
    planner.add_argument(
        "--model",
        type=str,
        default="gemini-3.8-flash",
        help="Gemini model name for VLM reasoning (default: gemini-3.8-flash)",
    )
    planner.add_argument(
        "--instructions",
        type=Path,
        help="optional JSON or JSONL file with driving instructions per scenario",
    )
    planner.add_argument(
        "--camera-summaries",
        type=Path,
        help="optional JSON or JSONL file with camera scene summaries per scenario",
    )
    planner.add_argument(
        "--api-key",
        type=str,
        help="Gemini API key (defaults to GEMINI_API_KEY or GOOGLE_API_KEY env var)",
    )
    return parser


def _package(args) -> int:
    report = archive.build_archive(
        args.videos_dir,
        args.out,
        allow_partial=args.allow_partial,
        check_media=not args.skip_media,
    )
    print_section(
        f"packaging {args.videos_dir}",
        report.errors,
        report.warnings,
        report.summary_lines(),
    )
    print_result(report.ok, len(report.errors), len(report.warnings))
    if report.ok:
        print(f"written: {report.path}")
        print(f"next: python validate_submission.py --videos {report.path}")
        return 0
    return 1


def _planner(args):
    """Return (label, note) describing the planner that will actually run.

    Without an API key the VLM module falls back to its offline instruction parser,
    and reporting that run as "Gemini VLM" would misattribute the result.
    """
    if not getattr(args, "vlm", False):
        return "baseline kinematic extrapolation", None

    client = vlm_planner.GeminiPlannerClient(model=args.model, api_key=args.api_key)
    if client.api_key:
        return f"Gemini {args.model}", None
    return (
        "heuristic fallback",
        "no GEMINI_API_KEY or GOOGLE_API_KEY set: the tactical plan came from the "
        "offline instruction parser, not from the model",
    )


def _plan(args) -> int:
    try:
        if getattr(args, "vlm", False):
            written = vlm_planner.plan_vlm_file(
                source=args.history,
                destination=args.out,
                model=args.model,
                instructions_path=args.instructions,
                camera_summaries_path=args.camera_summaries,
                api_key=args.api_key,
            )
        else:
            written = trajectories.plan_file(args.history, args.out)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    label, note = _planner(args)
    if note:
        print(f"warning: {note}", file=sys.stderr)
    print(f"planned {written} trajectories using {label} from {args.history}")
    print(f"written: {args.out}")
    report = trajectories.inspect_trajectories(args.out)
    print_section("checking the result", report.errors, report.warnings, report.summary_lines())
    print_result(report.ok, len(report.errors), len(report.warnings))
    return 0 if report.ok else 1


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "package":
        try:
            return _package(args)
        except (media.ProbeError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    return _plan(args)


if __name__ == "__main__":
    sys.exit(main())
