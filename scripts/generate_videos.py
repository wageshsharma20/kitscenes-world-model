#!/usr/bin/env python3
"""CLI for generating multi-view surround driving videos from trajectory predictions.

Usage examples:
  # Generate and pack videos for all scenarios in trajectories.jsonl
  python scripts/generate_videos.py --trajectories submissions/trajectories.jsonl \\
      --output-dir renders/ --zip-out submissions/videos.zip

  # Test on 2 scenarios with partial packing allowed
  python scripts/generate_videos.py --trajectories submissions/trajectories.jsonl \\
      --limit 2 --allow-partial
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitscenes import spec, video_engine  # noqa: E402  (imports follow the sys.path setup)
from kitscenes.reporting import print_result, print_section  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate 6-camera surround driving videos for KITScenes-LongTail Challenge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trajectories",
        type=Path,
        required=True,
        help="Path to trajectories.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("renders"),
        help="Directory to store rendered MP4 files",
    )
    parser.add_argument(
        "--zip-out",
        type=Path,
        default=Path("submissions/videos.zip"),
        help="Output submission zip file",
    )
    parser.add_argument(
        "--backbone",
        choices=["geometric", "svd"],
        default="geometric",
        help="Video generation backbone",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for generative backbones (cuda, cpu, mps)",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        help="Comma-separated scenario IDs to generate (e.g. 0000000,0000001)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of scenarios to generate",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=24,
        help="H.264 CRF quality (18-28)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow packing fewer than 400 scenarios for local testing",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.trajectories.exists():
        print(f"error: trajectory file not found: {args.trajectories}", file=sys.stderr)
        return 1

    scenario_ids = None
    if args.scenarios:
        scenario_ids = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    elif args.limit:
        scenario_ids = list(spec.SCENARIO_IDS[:args.limit])

    target_count = len(scenario_ids) if scenario_ids else spec.SCENARIO_COUNT
    print(f"Starting video generation for {target_count} scenario(s) using backbone '{args.backbone}'...")

    def progress_callback(idx: int, total: int, sc_id: str):
        if idx % 10 == 0 or idx == total or total <= 10:
            print(f"[{idx}/{total}] Generated 6 cameras for scenario {sc_id}")

    try:
        report = video_engine.batch_generate_and_package(
            trajectories_file=args.trajectories,
            output_dir=args.output_dir,
            zip_path=args.zip_out,
            scenario_ids=scenario_ids,
            allow_partial=args.allow_partial,
            backbone=args.backbone,
            device=args.device,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        print(f"error: generation failed: {exc}", file=sys.stderr)
        return 2

    print_section(
        f"Archive packaging: {args.zip_out}",
        report.errors,
        report.warnings,
        report.summary_lines(),
    )
    print_result(report.ok, len(report.errors), len(report.warnings))

    if report.ok:
        print(f"Successfully generated and packaged: {args.zip_out}")
        print(f"Validate with: python validate_submission.py --videos {args.zip_out} --trajectories {args.trajectories}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
