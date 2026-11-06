"""Argument parsing contract for the future unified perf-test entry point."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from rtp_llm.test.perf_test.errors import ConfigurationError


@dataclass(frozen=True)
class CliRequest:
    mode: str
    action: str
    config: Optional[str]
    run_dir: Optional[str]
    cases: Tuple[str, ...]
    dry_run: bool


class _ConfigArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ConfigurationError(f"command line: {message}")


def create_parser() -> argparse.ArgumentParser:
    parser = _ConfigArgumentParser(prog="perf_test", add_help=True)
    parser.add_argument("--mode", required=True, choices=("prefill", "decode"))
    parser.add_argument("--action", choices=("run", "resume", "profile", "report"))
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="regenerate a prefill report from --run-dir without starting an engine",
    )
    parser.add_argument("--config")
    parser.add_argument("--run-dir")
    parser.add_argument("--cases", nargs="+", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _non_empty(option: str, value: Optional[str]) -> Optional[str]:
    if value is not None and not value:
        raise ConfigurationError(f"{option} must be a non-empty path")
    return value


def parse_cli(argv: Sequence[str]) -> CliRequest:
    """Parse and validate inputs without loading config or persisted run state."""
    namespace = create_parser().parse_args(list(argv))
    if namespace.skip_test:
        if namespace.mode != "prefill":
            raise ConfigurationError("--skip-test requires --mode=prefill")
        if namespace.action not in (None, "report"):
            raise ConfigurationError(
                "--skip-test is only compatible with --action=report"
            )
    if namespace.action is None and not namespace.skip_test:
        raise ConfigurationError("--action is required unless --skip-test is set")
    action = "report" if namespace.skip_test else namespace.action
    config = _non_empty("--config", namespace.config)
    run_dir = _non_empty("--run-dir", namespace.run_dir)
    cases = tuple(namespace.cases)

    if action == "run":
        if config is None:
            raise ConfigurationError("--action=run requires --config")
        if run_dir is not None:
            raise ConfigurationError("--action=run does not accept --run-dir")
        if cases:
            raise ConfigurationError("--cases is only valid with --action=profile")
    else:
        if run_dir is None:
            raise ConfigurationError(f"--action={action} requires --run-dir")
        if config is not None:
            raise ConfigurationError(
                f"--action={action} does not accept --config overrides"
            )
        if cases and action != "profile":
            raise ConfigurationError("--cases is only valid with --action=profile")

    if any(not case for case in cases):
        raise ConfigurationError("--cases values must be non-empty case identifiers")
    if len(set(cases)) != len(cases):
        raise ConfigurationError("--cases contains duplicate case identifiers")

    return CliRequest(
        mode=namespace.mode,
        action=action,
        config=config,
        run_dir=run_dir,
        cases=cases,
        dry_run=namespace.dry_run,
    )
