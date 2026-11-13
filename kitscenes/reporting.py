"""Shared console output for the command-line tools."""

import sys
from typing import Iterable, List

PREVIEW = 20


def _print(lines: Iterable[str], stream) -> None:
    for line in lines:
        print(line, file=stream)


def print_section(
    title: str,
    errors: List[str],
    warnings: List[str],
    summary: Iterable[str] = (),
) -> None:
    print(title)
    _print((f"  error: {message}" for message in errors[:PREVIEW]), sys.stderr)
    if len(errors) > PREVIEW:
        print(f"  error: ... and {len(errors) - PREVIEW} more", file=sys.stderr)
    _print((f"  warning: {message}" for message in warnings[:PREVIEW]), sys.stderr)
    if len(warnings) > PREVIEW:
        print(f"  warning: ... and {len(warnings) - PREVIEW} more", file=sys.stderr)
    _print((f"  {line}" for line in summary), sys.stdout)
    print()


def _plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def print_result(ok: bool, errors: int, warnings: int) -> None:
    state = "valid" if ok else "invalid"
    print(
        f"result: {state} ({_plural(errors, 'error')}, "
        f"{_plural(warnings, 'warning')})"
    )
