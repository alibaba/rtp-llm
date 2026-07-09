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
from pathlib import Path
from typing import Any, Dict

import pytest

from rtp_llm.test.ci_profile_support import (
    find_pyproject_with_ci_profiles,
    load_pytest_ci_config,
    resolve_profile_paths,
)


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


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    name = config.getoption("--rtp-ci-profile")
    if not name:
        return

    root = Path(config.rootpath)
    pytest_ci = _get_pytest_ci_section(root)
    default_cli = (
        pytest_ci.get("default_pytest_cli") or "-v --tb=short --timeout=300"
    ).strip()
    if default_cli:
        _apply_default_cli(config, default_cli)

    prof = _get_profile(root, name)
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
