"""pytest plugin: apply CI marker/path presets from pyproject.toml.

Reads [tool.rtp_llm.pytest_ci].profiles.<name> from the nearest pyproject.toml
(github-opensource root or open_source/pyproject.toml).

Use --rtp-ci-profile=NAME.
Can be combined with -m; final markexpr is:
    (<profile markexpr>) and (<user -m markexpr>)

When a profile is active, [tool.rtp_llm.pytest_ci].default_pytest_cli is applied:
sets PYTEST_ARGS for --remote-session workers and updates local pytest options
unless PYTEST_ARGS is already set in the environment.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

from rtp_llm.test.ci_profile_support import (
    find_pyproject_with_ci_profiles,
    load_pytest_ci_config,
    resolve_profile_paths,
)


_active_profile_name: str | None = None
_forbid_skips = False
_skipped_reports: list[str] = []


def _get_profile(root: Path, name: str) -> Dict[str, Any]:
    pyproject, pytest_ci = load_pytest_ci_config(root)
    if pyproject is None:
        raise pytest.UsageError(
            f"--rtp-ci-profile: no pyproject.toml found (searched from {root})"
        )
    try:
        profiles = pytest_ci["profiles"]
    except KeyError as exc:
        raise pytest.UsageError(
            f"--rtp-ci-profile: missing [tool.rtp_llm.pytest_ci].profiles in {pyproject}"
        ) from exc
    if name not in profiles:
        known = ", ".join(sorted(profiles.keys()))
        raise pytest.UsageError(
            f"--rtp-ci-profile: unknown profile {name!r}. Known: {known}"
        )
    prof = profiles[name]
    if not isinstance(prof, dict):
        raise pytest.UsageError(f"--rtp-ci-profile: profile {name!r} must be a table")
    if "markexpr" not in prof:
        raise pytest.UsageError(
            f"--rtp-ci-profile: profile {name!r} missing 'markexpr'"
        )
    return prof


def _get_pytest_ci_section(root: Path) -> Dict[str, Any]:
    _, pytest_ci = load_pytest_ci_config(root)
    return pytest_ci


def _apply_default_cli(config: pytest.Config, cli: str) -> None:
    """Apply -v / --tb= / --timeout= to config (best-effort for pytest-timeout).

    Only sets defaults — never overwrites values the user explicitly passed on
    the command line (detected via ``config.getoption`` returning non-default).
    """
    # Snapshot user-explicit values *before* we touch anything.
    user_timeout = config.getoption("--timeout", default=None)
    for tok in shlex.split(cli):
        if tok == "-v":
            config.option.verbose = max(getattr(config.option, "verbose", 0), 1)
        elif tok == "-vv":
            config.option.verbose = max(getattr(config.option, "verbose", 0), 2)
        elif tok.startswith("--tb="):
            config.option.tbstyle = tok.split("=", 1)[1]
        elif tok.startswith("--timeout="):
            # Only apply default timeout when the user did not pass --timeout.
            if user_timeout is None:
                val = int(tok.split("=", 1)[1])
                if hasattr(config.option, "timeout"):
                    setattr(config.option, "timeout", val)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("rtp-ci-profile", "RTP-LLM CI profiles (pyproject.toml)")
    group.addoption(
        "--rtp-ci-profile",
        action="store",
        default=None,
        metavar="NAME",
        help=(
            "Load markexpr/paths from [tool.rtp_llm.pytest_ci].profiles.NAME "
            "in pyproject.toml"
        ),
    )
    group.addoption(
        "--rtp-ci-allow-subset",
        action="store_true",
        default=False,
        help="Allow -k/path filtering to collect fewer tests than profile expected_count.",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    global _active_profile_name, _forbid_skips, _skipped_reports

    name = config.getoption("--rtp-ci-profile")
    _active_profile_name = None
    _forbid_skips = False
    _skipped_reports = []
    config._rtp_ci_forbid_skips = False  # type: ignore[attr-defined]
    if not name:
        return

    _active_profile_name = name

    root = Path(config.rootpath)
    pytest_ci = _get_pytest_ci_section(root)
    default_cli = (
        pytest_ci.get("default_pytest_cli") or "-v --tb=short --timeout=300"
    ).strip()
    if default_cli:
        _apply_default_cli(config, default_cli)

    prof = _get_profile(root, name)
    if name == "py_ut_amd" and config.getoption("--remote-session", default=False):
        workers = config.getoption("--remote-workers", default=4)
        if workers < 8:
            raise pytest.UsageError(
                "py_ut_amd requires --remote-workers=8 for moriep coverage"
            )
    expected_count = prof.get("expected_count")
    if expected_count is not None:
        if not isinstance(expected_count, int) or expected_count < 1:
            raise pytest.UsageError(
                f"--rtp-ci-profile: profile {name!r} expected_count must be a positive integer"
            )
        config._rtp_ci_expected_count = expected_count  # type: ignore[attr-defined]
    minimum_count = prof.get("minimum_count", 1)
    if not isinstance(minimum_count, int) or minimum_count < 1:
        raise pytest.UsageError(
            f"--rtp-ci-profile: profile {name!r} minimum_count must be a positive integer"
        )
    config._rtp_ci_minimum_count = minimum_count  # type: ignore[attr-defined]
    _forbid_skips = bool(prof.get("forbid_skips", False))
    config._rtp_ci_forbid_skips = _forbid_skips  # type: ignore[attr-defined]
    markexpr = prof["markexpr"]
    if not isinstance(markexpr, str) or not markexpr.strip():
        raise pytest.UsageError(
            f"--rtp-ci-profile: profile {name!r} has empty markexpr"
        )
    profile_markexpr = markexpr.strip()
    user_markexpr = (getattr(config.option, "markexpr", None) or "").strip()
    if user_markexpr:
        config.option.markexpr = f"({profile_markexpr}) and ({user_markexpr})"
    else:
        config.option.markexpr = profile_markexpr

    # Profile-level gpu_type is only a session-level fallback. Per-test remote
    # dispatch must use each case's @pytest.mark.gpu(type=..., count=...) marker.
    gpu_type = prof.get("gpu_type")
    if (
        gpu_type
        and config.getoption("--remote-session", default=False)
        and not getattr(config.option, "remote_gpu_type", None)
    ):
        config.option.remote_gpu_type = gpu_type

    paths = prof.get("paths")
    if paths is not None:
        if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            raise pytest.UsageError(
                f"--rtp-ci-profile: profile {name!r} 'paths' must be a list of strings"
            )
        # Restrict collection roots (e.g. smoke file or frontend dirs only)
        config.args[:] = resolve_profile_paths(root, paths)

    ignore_paths = prof.get("ignore_paths")
    if ignore_paths is not None:
        if not isinstance(ignore_paths, list) or not all(
            isinstance(p, str) for p in ignore_paths
        ):
            raise pytest.UsageError(
                f"--rtp-ci-profile: profile {name!r} 'ignore_paths' must be a list of strings"
            )
        ignored = list(getattr(config.option, "ignore", None) or [])
        for path in resolve_profile_paths(root, ignore_paths):
            if path not in ignored:
                ignored.append(path)
        config.option.ignore = ignored

    isolated_paths = prof.get("isolated_paths")
    if isolated_paths is not None and (
        not isinstance(isolated_paths, list)
        or not all(isinstance(p, str) for p in isolated_paths)
    ):
        raise pytest.UsageError(
            f"--rtp-ci-profile: profile {name!r} 'isolated_paths' must be a list of strings"
        )


def validate_ci_profile_count(
    config: pytest.Config, actual: int, *, context: str = "collected"
) -> None:
    """Reject empty or drifted CI profile results.

    Broad Python UT profiles track main and therefore use ``minimum_count`` to
    reject 0/0 runs. Frozen smoke/perf/platform manifests additionally declare
    ``expected_count`` for exact parity.
    """
    minimum = getattr(config, "_rtp_ci_minimum_count", None)
    if minimum is None:
        return
    if actual < minimum:
        raise pytest.UsageError(
            f"CI profile {_active_profile_name!r} {context} {actual} tests; "
            f"expected at least {minimum}. A CI profile must never pass as 0/0."
        )

    expected = getattr(config, "_rtp_ci_expected_count", None)
    if (
        expected is not None
        and not config.getoption("--rtp-ci-allow-subset")
        and actual != expected
    ):
        raise pytest.UsageError(
            f"CI profile {_active_profile_name!r} {context} {actual} tests; "
            f"expected {expected}. Update the manifest/profile together, or use "
            "--rtp-ci-allow-subset for an intentional manual subset."
        )


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session: pytest.Session) -> None:
    """Fail before execution when a local/per-test CI profile loses cases."""
    if session.config.getoption("--remote-session", default=False):
        # Session controllers intentionally skip local collection. The remote
        # plugin validates the merged worker JUnit after execution instead.
        return
    validate_ci_profile_count(session.config, len(session.items))
    if _active_profile_name == "py_ut_amd" and not session.config.getoption(
        "--rtp-ci-allow-subset"
    ):
        from rtp_llm.test.amd_coverage import validate_amd_coverage

        try:
            validate_amd_coverage(item.nodeid for item in session.items)
        except ValueError as exc:
            raise pytest.UsageError(str(exc)) from exc


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if _forbid_skips and report.skipped:
        _skipped_reports.append(report.nodeid)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """A selected Smoke case must execute; a skip is not coverage."""
    if not _forbid_skips or not _skipped_reports:
        return
    unique = sorted(set(_skipped_reports))
    print(
        f"CI profile {_active_profile_name!r} skipped {len(unique)} selected test(s):\n"
        + "\n".join(f"  {nodeid}" for nodeid in unique),
        file=sys.stderr,
    )
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
