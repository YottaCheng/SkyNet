#!/usr/bin/env python3
"""Sealed final Month-7 experiment runner.

Default mode is fail-closed. Hardening and CI must use ``--dry-run``, which
never opens Month 7 and never makes live API calls.

``--execute-final`` is the later authorised entrypoint. It is not invoked by
this hardening task.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1]
SRC = IMPL / "src"
sys.path.insert(0, str(SRC))

from attack_lab.final_experiment import (  # noqa: E402
    FORBIDDEN_EXECUTE_WITHOUT_FLAG,
    FinalRunnerError,
    load_protocol,
    run_dry_run,
    run_execute_final_preflight_only,
)
from attack_lab.final_protocol import DEFAULT_PROTOCOL_PATH  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL_PATH,
        help="Path to the frozen final protocol JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and execute the frozen pipeline on fixtures. No Month 7, no API.",
    )
    parser.add_argument(
        "--execute-final",
        action="store_true",
        help="Authorised Month-7 execution. Do not use during pre-Month-7 hardening.",
    )
    parser.add_argument(
        "--output-parent",
        type=Path,
        default=None,
        help="Optional parent under 05_outputs/experiments/.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        protocol = load_protocol(args.protocol)
        if args.dry_run and args.execute_final:
            raise FinalRunnerError("Pass only one of --dry-run or --execute-final.")
        if args.dry_run:
            result = run_dry_run(protocol=protocol, output_parent=args.output_parent)
            print(json.dumps({"status": result["status"], "run_dir": result["run_dir"]}))
            return 0
        if args.execute_final:
            run_execute_final_preflight_only(protocol)
            return 2
        raise FinalRunnerError(FORBIDDEN_EXECUTE_WITHOUT_FLAG)
    except (FinalRunnerError, OSError, ValueError) as exc:
        print(f"FAIL CLOSED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
