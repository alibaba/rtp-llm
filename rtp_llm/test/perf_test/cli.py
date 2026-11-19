"""CLI shape validation; mode inference and capability checks happen in planning."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from rtp_llm.test.perf_test.errors import ConfigurationError


@dataclass(frozen=True)
class CliRequest:
    mode: Optional[str]
    action: str
    config: Optional[str]
    run_dir: Optional[str]
    cases: Tuple[str, ...]
    dry_run: bool
    run_context: Optional[str] = None
    bazel_startup_args: Tuple[str, ...] = ()
    bazel_args: Tuple[str, ...] = ()
    broker_host: Optional[str] = None
    maintenance: Optional[str] = None
    skip_test: bool = False
    overrides: Tuple[str, ...] = ()
    input_config: Optional[Dict[str, Any]] = None


class _ConfigArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ConfigurationError(f"command line: {message}")


def create_parser() -> argparse.ArgumentParser:
    parser = _ConfigArgumentParser(prog="perf_test", add_help=True)
    parser.add_argument("--mode", choices=("prefill", "decode"))
    parser.add_argument("--action", choices=("run", "resume", "profile", "report"))
    parser.add_argument(
        "--skip-test", action="store_true", help="offline prefill report from --run-dir"
    )
    parser.add_argument(
        "--maintenance", choices=("status", "cancel", "collect", "clean")
    )
    parser.add_argument("--config")
    parser.add_argument("--run-dir")
    parser.add_argument("--cases", nargs="+", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-context")
    parser.add_argument("--bazel-startup-arg", action="append", default=[])
    parser.add_argument("--bazel-arg", action="append", default=[])
    parser.add_argument("--broker-host")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE"
    )
    return parser


def _non_empty(option: str, value: Optional[str]) -> Optional[str]:
    if value is not None and not value:
        raise ConfigurationError(f"{option} must be a non-empty path")
    return value


def parse_cli(argv: Sequence[str]) -> CliRequest:
    namespace = create_parser().parse_args(list(argv))
    config = _non_empty("--config", namespace.config)
    run_dir = _non_empty("--run-dir", namespace.run_dir)
    cases = tuple(namespace.cases)
    if namespace.overrides and (
        not config
        or namespace.action not in (None, "run")
        or namespace.maintenance
        or namespace.skip_test
    ):
        raise ConfigurationError("--set requires a new --config run")
    if namespace.maintenance:
        if (
            not run_dir
            or config
            or namespace.action
            or namespace.skip_test
            or cases
            or namespace.mode
            or namespace.run_context
            or namespace.bazel_startup_arg
            or namespace.bazel_arg
            or namespace.broker_host
        ):
            raise ConfigurationError(
                "--maintenance requires only --run-dir (and optional --dry-run)"
            )
        action = "report"  # maintenance is dispatched separately, never measured.
    else:
        if namespace.skip_test:
            if namespace.mode not in (None, "prefill"):
                raise ConfigurationError("--skip-test requires prefill mode")
            if namespace.action not in (None, "report"):
                raise ConfigurationError(
                    "--skip-test is only compatible with --action=report"
                )
        action = (
            "report"
            if namespace.skip_test
            else namespace.action or ("run" if config is not None else "report")
        )
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
    if any(not case for case in cases) or len(set(cases)) != len(cases):
        raise ConfigurationError(
            "--cases must contain unique non-empty case identifiers"
        )
    return CliRequest(
        mode=namespace.mode,
        action=action,
        config=config,
        run_dir=run_dir,
        cases=cases,
        dry_run=namespace.dry_run,
        run_context=_non_empty("--run-context", namespace.run_context),
        bazel_startup_args=tuple(namespace.bazel_startup_arg),
        bazel_args=tuple(namespace.bazel_arg),
        broker_host=namespace.broker_host,
        maintenance=namespace.maintenance,
        skip_test=namespace.skip_test,
        overrides=tuple(namespace.overrides),
    )
